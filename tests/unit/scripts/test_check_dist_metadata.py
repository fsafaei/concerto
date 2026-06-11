# SPDX-License-Identifier: Apache-2.0
"""Unit tests for :mod:`scripts.check_dist_metadata` (release pre-flight gate).

Exercises the offline metadata contract against synthetic wheel + sdist
artefacts. The ``--check-resolvable`` network path (production-PyPI
resolvability) is covered down to the Requires-Dist parsing layer only;
the live HTTP probe runs in the release workflow, not in CI tests.
"""

from __future__ import annotations

import importlib.util
import tarfile
import zipfile
from pathlib import Path

import pytest

_SCRIPT = Path("scripts/check_dist_metadata.py")

_GOOD_METADATA = (
    "Metadata-Version: 2.4\n"
    "Name: concerto-multirobot\n"
    "Version: 0.7.0\n"
    "Classifier: Development Status :: 4 - Beta\n"
    "Classifier: License :: OSI Approved :: Apache Software License\n"
    "Project-URL: Homepage, https://github.com/fsafaei/concerto\n"
    "Project-URL: Documentation, https://fsafaei.github.io/concerto/\n"
    "Project-URL: Repository, https://github.com/fsafaei/concerto\n"
    "Project-URL: Issues, https://github.com/fsafaei/concerto/issues\n"
    "Project-URL: Changelog, https://github.com/fsafaei/concerto/blob/main/CHANGELOG.md\n"
    "Requires-Dist: numpy>=1.26,<2.2\n"
    "Requires-Dist: harl-aht>=0.1.0,<0.2.0\n"
    'Requires-Dist: rliable>=1.0,<2.0; extra == "eval"\n'
    "Description-Content-Type: text/markdown\n"
    "\n" + ("# CONCERTO\n\nLong description body.\n" * 100)
)


@pytest.fixture(scope="module")
def gate_module():  # type: ignore[no-untyped-def]
    """Load check_dist_metadata.py as a module from its path on disk."""
    import sys

    spec = importlib.util.spec_from_file_location("check_dist_metadata", _SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_dist_metadata"] = module
    spec.loader.exec_module(module)
    return module


def _write_dist(dist_dir: Path, metadata: str, version: str = "0.7.0") -> None:
    """Materialise a synthetic wheel + sdist carrying ``metadata``."""
    dist_dir.mkdir(parents=True, exist_ok=True)
    stem = f"concerto_multirobot-{version}"
    wheel = dist_dir / f"{stem}-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as zf:
        zf.writestr(f"{stem}.dist-info/METADATA", metadata)
    sdist = dist_dir / f"{stem}.tar.gz"
    pkg_info = dist_dir / "PKG-INFO"
    pkg_info.write_text(metadata, encoding="utf-8")
    with tarfile.open(sdist, "w:gz") as tf:
        tf.add(pkg_info, arcname=f"{stem}/PKG-INFO")
    pkg_info.unlink()


class TestContract:
    """The offline release-contract checks."""

    def test_good_artifacts_pass(self, gate_module, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
        _write_dist(tmp_path / "dist", _GOOD_METADATA)
        rc = gate_module.main(["--dist-dir", str(tmp_path / "dist"), "--expect-tag", "v0.7.0"])
        assert rc == 0

    def test_version_must_match_tag(self, gate_module, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
        _write_dist(tmp_path / "dist", _GOOD_METADATA)
        rc = gate_module.main(["--dist-dir", str(tmp_path / "dist"), "--expect-tag", "v0.8.0"])
        assert rc == 1

    def test_tag_without_v_prefix_rejected(self, gate_module, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
        _write_dist(tmp_path / "dist", _GOOD_METADATA)
        rc = gate_module.main(["--dist-dir", str(tmp_path / "dist"), "--expect-tag", "0.7.0"])
        assert rc == 2

    def test_alpha_classifier_fails(self, gate_module, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
        bad = _GOOD_METADATA.replace(
            "Development Status :: 4 - Beta", "Development Status :: 3 - Alpha"
        )
        _write_dist(tmp_path / "dist", bad)
        rc = gate_module.main(["--dist-dir", str(tmp_path / "dist"), "--expect-tag", "v0.7.0"])
        assert rc == 1

    def test_missing_project_url_fails(self, gate_module, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
        bad = "\n".join(
            line
            for line in _GOOD_METADATA.splitlines()
            if not line.startswith("Project-URL: Changelog")
        )
        _write_dist(tmp_path / "dist", bad)
        rc = gate_module.main(["--dist-dir", str(tmp_path / "dist"), "--expect-tag", "v0.7.0"])
        assert rc == 1

    def test_short_description_fails(self, gate_module, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
        bad = _GOOD_METADATA.split("\n\n", 1)[0] + "\n\nstub\n"
        _write_dist(tmp_path / "dist", bad)
        rc = gate_module.main(["--dist-dir", str(tmp_path / "dist"), "--expect-tag", "v0.7.0"])
        assert rc == 1

    def test_empty_dist_dir_is_usage_error(self, gate_module, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
        (tmp_path / "dist").mkdir()
        rc = gate_module.main(["--dist-dir", str(tmp_path / "dist"), "--expect-tag", "v0.7.0"])
        assert rc == 2


class TestRequiresDistParsing:
    """Extras-guarded deps must not gate a default install."""

    def test_extra_marked_deps_excluded(self, gate_module, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
        _write_dist(tmp_path / "dist", _GOOD_METADATA)
        wheel = next((tmp_path / "dist").glob("*.whl"))
        meta = gate_module._read_wheel_metadata(wheel)
        names = gate_module._runtime_dep_names(meta)
        assert "harl-aht" in names
        assert "numpy" in names
        assert "rliable" not in names
