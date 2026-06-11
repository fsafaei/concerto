"""Gate the built distribution metadata against the PyPI release contract.

Reads the sdist + wheel under ``dist/`` and exits non-zero unless every
check passes. Runs as a pre-upload gate in ``.github/workflows/release.yml``
and locally via ``make release-preflight``. Complements (does not replace)
``twine check --strict``, which validates that the long description renders.

Checks:

- exactly one wheel and one sdist are present;
- distribution name is ``concerto-multirobot``;
- version matches the release tag (``--expect-tag vX.Y.Z`` => ``X.Y.Z``),
  or ``[project].version`` in pyproject.toml when no tag is given;
- a single ``Development Status`` classifier, pinned to ``4 - Beta``;
- the Apache licence classifier is present;
- the required ``Project-URL`` labels are all present;
- the long description is non-trivial Markdown (the README ships as the
  PyPI project page);
- with ``--check-resolvable``: every non-extra runtime dependency exists on
  production PyPI. Guards against TestPyPI-only deps (harl-aht shipped to
  TestPyPI for the 0.x line — see ``[tool.uv.index]`` in pyproject.toml);
  publishing while such a dep is absent from pypi.org would break
  ``pip install concerto-multirobot`` for end users and leave the dep name
  open to squatting.

Usage:

    uv build
    uv run python scripts/check_dist_metadata.py [--expect-tag v0.7.0] \
        [--check-resolvable] [--dist-dir dist]
"""

from __future__ import annotations

import argparse
import email
import email.message
import re
import sys
import tarfile
import tomllib
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

DIST_NAME = "concerto-multirobot"

REQUIRED_DEV_STATUS = "Development Status :: 4 - Beta"
REQUIRED_LICENCE_CLASSIFIER = "License :: OSI Approved :: Apache Software License"
REQUIRED_URL_LABELS = frozenset({"Homepage", "Documentation", "Repository", "Issues", "Changelog"})

# The README is ~20 kB; anything materially shorter means the long
# description was dropped or truncated at build time.
MIN_DESCRIPTION_CHARS = 2_000

PYPI_JSON_URL = "https://pypi.org/pypi/{name}/json"
HTTP_NOT_FOUND = 404

# PEP 508 leading distribution name, e.g. "harl-aht" in
# "harl-aht>=0.1.0,<0.2.0" or "qpsolvers" in "qpsolvers[clarabel]>=4.0".
_REQUIRES_DIST_NAME_RE = re.compile(r"^\s*([A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)")


def _read_wheel_metadata(wheel: Path) -> email.message.Message:
    """Return the parsed ``*.dist-info/METADATA`` from a wheel."""
    with zipfile.ZipFile(wheel) as zf:
        names = [n for n in zf.namelist() if n.endswith(".dist-info/METADATA")]
        if len(names) != 1:
            raise ValueError(f"{wheel.name}: expected one METADATA, found {names}")
        return email.message_from_bytes(zf.read(names[0]))


def _read_sdist_metadata(sdist: Path) -> email.message.Message:
    """Return the parsed top-level ``PKG-INFO`` from an sdist."""
    with tarfile.open(sdist) as tf:
        names = [n for n in tf.getnames() if n.count("/") == 1 and n.endswith("/PKG-INFO")]
        if len(names) != 1:
            raise ValueError(f"{sdist.name}: expected one top-level PKG-INFO, found {names}")
        member = tf.extractfile(names[0])
        if member is None:
            raise ValueError(f"{sdist.name}: {names[0]} is not a regular file")
        with member:
            return email.message_from_bytes(member.read())


def _expected_version(expect_tag: str | None) -> str:
    """Resolve the expected version from the release tag or pyproject.toml.

    Tags follow release-please's ``v<version>`` convention; a tag that does
    not start with ``v`` is rejected rather than guessed at.
    """
    if expect_tag is not None:
        if not expect_tag.startswith("v"):
            raise ValueError(f"tag {expect_tag!r} does not follow the vX.Y.Z convention")
        return expect_tag[1:]
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    with pyproject.open("rb") as fh:
        return tomllib.load(fh)["project"]["version"]


def _runtime_dep_names(meta: email.message.Message) -> list[str]:
    """Distribution names of non-extra runtime deps from ``Requires-Dist``."""
    names: list[str] = []
    for req in meta.get_all("Requires-Dist") or []:
        # Split off the environment marker; deps guarded by `extra ==`
        # belong to optional extras and do not gate a default install.
        spec, _, marker = req.partition(";")
        if "extra ==" in marker:
            continue
        match = _REQUIRES_DIST_NAME_RE.match(spec)
        if match:
            names.append(match.group(1))
    return names


def _missing_on_pypi(names: list[str]) -> list[str]:
    """Return the subset of distribution names absent from production PyPI."""
    missing: list[str] = []
    for name in names:
        url = PYPI_JSON_URL.format(name=name)
        try:
            # S310 exemption: the scheme is pinned by the constant
            # https template above, not caller-controlled.
            with urllib.request.urlopen(url, timeout=30):  # noqa: S310
                pass
        except urllib.error.HTTPError as exc:
            if exc.code == HTTP_NOT_FOUND:
                missing.append(name)
            else:
                raise
    return missing


def _check_artifact(label: str, meta: email.message.Message, expected_version: str) -> list[str]:
    """Return human-readable failures for one artifact's metadata."""
    failures: list[str] = []

    name = meta.get("Name", "")
    if name != DIST_NAME:
        failures.append(f"{label}: Name is {name!r}, expected {DIST_NAME!r}")

    version = meta.get("Version", "")
    if version != expected_version:
        failures.append(f"{label}: Version is {version!r}, expected {expected_version!r}")

    classifiers = meta.get_all("Classifier") or []
    dev_status = [c for c in classifiers if c.startswith("Development Status ::")]
    if dev_status != [REQUIRED_DEV_STATUS]:
        failures.append(
            f"{label}: Development Status classifiers are {dev_status}, "
            f"expected exactly [{REQUIRED_DEV_STATUS!r}]"
        )
    if REQUIRED_LICENCE_CLASSIFIER not in classifiers:
        failures.append(f"{label}: missing classifier {REQUIRED_LICENCE_CLASSIFIER!r}")

    url_labels = {u.partition(",")[0].strip() for u in (meta.get_all("Project-URL") or [])}
    missing_urls = REQUIRED_URL_LABELS - url_labels
    if missing_urls:
        failures.append(f"{label}: missing Project-URL labels {sorted(missing_urls)}")

    content_type = meta.get("Description-Content-Type", "")
    if not content_type.startswith("text/markdown"):
        failures.append(
            f"{label}: Description-Content-Type is {content_type!r}, "
            "expected text/markdown (README is the PyPI long description)"
        )
    description = meta.get_payload()
    if not isinstance(description, str) or len(description) < MIN_DESCRIPTION_CHARS:
        size = len(description) if isinstance(description, str) else 0
        failures.append(
            f"{label}: long description is {size} chars "
            f"(< {MIN_DESCRIPTION_CHARS}); README embedding looks broken"
        )

    return failures


def main(argv: list[str] | None = None) -> int:
    """Validate dist/ artifacts; exit non-zero with all failures listed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dist-dir", type=Path, default=Path("dist"), help="directory holding the built artifacts"
    )
    parser.add_argument(
        "--expect-tag",
        default=None,
        help="release tag (vX.Y.Z); the artifact version must equal X.Y.Z. "
        "Defaults to [project].version from pyproject.toml.",
    )
    parser.add_argument(
        "--check-resolvable",
        action="store_true",
        help="verify every non-extra runtime dep exists on production PyPI",
    )
    args = parser.parse_args(argv)

    try:
        expected_version = _expected_version(args.expect_tag)
    except ValueError as exc:
        print(f"[dist-metadata] {exc}", file=sys.stderr)
        return 2

    wheels = sorted(args.dist_dir.glob("*.whl"))
    sdists = sorted(args.dist_dir.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        print(
            f"[dist-metadata] expected exactly one wheel and one sdist in "
            f"{args.dist_dir}/, found wheels={[w.name for w in wheels]} "
            f"sdists={[s.name for s in sdists]}; run a clean `uv build`.",
            file=sys.stderr,
        )
        return 2

    wheel_meta = _read_wheel_metadata(wheels[0])
    sdist_meta = _read_sdist_metadata(sdists[0])

    failures = _check_artifact(wheels[0].name, wheel_meta, expected_version)
    failures += _check_artifact(sdists[0].name, sdist_meta, expected_version)

    if args.check_resolvable:
        missing = _missing_on_pypi(_runtime_dep_names(wheel_meta))
        if missing:
            failures.append(
                f"runtime deps absent from production PyPI: {missing} — "
                "publishing now would break `pip install concerto-multirobot` "
                "(and leaves those names open to squatting). Publish the dep "
                "to pypi.org first."
            )

    if failures:
        print("[dist-metadata] release contract violations:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    print(
        f"[dist-metadata] OK ({wheels[0].name}, {sdists[0].name}: "
        f"version {expected_version}, classifiers, URLs, long description"
        f"{', deps resolvable on PyPI' if args.check_resolvable else ''})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
