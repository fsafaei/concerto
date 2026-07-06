#!/usr/bin/env python3
r"""Upload the prepared hosting packages to the maintainer's Hugging Face namespace.

Founder-side companion to ``prepare_hosting.py`` (ADR-028 §Decision;
ADR-012 §Decision licence). Safety properties:

* the token comes from the ``HF_TOKEN`` environment variable **only**
  — it is never read from a file, never echoed, never committed — and
  the script refuses to run without it;
* the script prints the full upload plan (dataset repo ids, every
  file with its size, per-package and total byte counts) and then
  **stops** unless ``--yes`` was passed — there is no interactive
  fallthrough;
* re-runs are idempotent: files whose SHA-256 already matches the
  hosted copy (per the remote ``manifest.json``) are skipped, so an
  interrupted or repeated upload only transfers what changed;
* ``huggingface_hub`` lives in the optional ``hosting`` dependency
  group, not in the wheel's runtime dependencies.

Usage::

    export HF_TOKEN=<maintainer token with write scope>   # never commit this
    uv run --group hosting python scripts/release/upload_hosting.py \\
        [--namespace fsafaei] [--source dist/hosting] [--yes]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from chamber.evaluation.hosting import (  # noqa: E402
    HOSTING_ARTIFACTS,
    PACKAGE_MANIFEST_NAME,
    PACKAGE_SUMS_NAME,
)

DEFAULT_NAMESPACE = "fsafaei"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_files(package: Path) -> list[Path]:
    """Every file in the package, sorted by repo-relative path."""
    return sorted(
        (p for p in package.rglob("*") if p.is_file()),
        key=lambda p: p.relative_to(package).as_posix(),
    )


def _print_plan(packages: list[Path], namespace: str) -> None:
    grand_total = 0
    print("upload-hosting: plan")
    for package in packages:
        manifest = json.loads((package / PACKAGE_MANIFEST_NAME).read_text(encoding="utf-8"))
        files = _package_files(package)
        total = sum(p.stat().st_size for p in files)
        grand_total += total
        print(
            f"\n  datasets/{namespace}/{package.name}  <-  {package}\n"
            f"  ({len(files)} files, {total / 1_048_576:.1f} MiB, "
            f"built from git {manifest['git_sha']})"
        )
        for path in files:
            rel = path.relative_to(package).as_posix()
            print(f"    {path.stat().st_size:>12,}  {rel}")
    print(
        f"\nupload-hosting: total {grand_total / 1_048_576:.1f} MiB across {len(packages)} datasets"
    )


def _remote_manifest_bytes(repo_id: str, token: str) -> bytes | None:
    """The hosted ``manifest.json`` bytes, or ``None`` if repo/file is absent."""
    from huggingface_hub import hf_hub_download  # noqa: PLC0415  (hosting-only dep)
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError  # noqa: PLC0415

    try:
        remote_manifest = hf_hub_download(
            repo_id=repo_id,
            filename=PACKAGE_MANIFEST_NAME,
            repo_type="dataset",
            token=token,
        )
    except (RepositoryNotFoundError, EntryNotFoundError):
        return None
    return Path(remote_manifest).read_bytes()


def _upload_package(package: Path, repo_id: str, token: str) -> None:
    """Create the dataset repo if needed and push only the changed files.

    Skip rule (idempotent re-runs): a payload file is skipped when its
    SHA-256 matches the hosted ``manifest.json`` entry. ``manifest.json``
    and ``SHA256SUMS.txt`` themselves are not listed in the manifest;
    they are skipped only when the local and hosted manifests are
    byte-identical (the build is deterministic, so an identical manifest
    implies an identical digest list).
    """
    from huggingface_hub import CommitOperationAdd, HfApi  # noqa: PLC0415  (hosting-only dep)

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)

    remote_bytes = _remote_manifest_bytes(repo_id, token)
    remote: dict[str, str] = {}
    if remote_bytes is not None:
        remote = {entry["path"]: entry["sha256"] for entry in json.loads(remote_bytes)["files"]}
    manifest_unchanged = (
        remote_bytes is not None and (package / PACKAGE_MANIFEST_NAME).read_bytes() == remote_bytes
    )

    operations: list[Any] = []
    skipped = 0
    for path in _package_files(package):
        rel = path.relative_to(package).as_posix()
        if rel in (PACKAGE_MANIFEST_NAME, PACKAGE_SUMS_NAME):
            unchanged = manifest_unchanged
        else:
            unchanged = remote.get(rel) == _sha256_file(path)
        if unchanged:
            skipped += 1
            continue
        operations.append(CommitOperationAdd(path_in_repo=rel, path_or_fileobj=str(path)))
    if not operations:
        print(f"upload-hosting: {package.name} unchanged ({skipped} files) — skipping")
        return
    print(
        f"upload-hosting: {package.name} — uploading {len(operations)} file(s), {skipped} unchanged"
    )
    api.create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        operations=operations,
        commit_message=f"CHAMBER-Bench v1.0 {package.name} ({len(operations)} changed files)",
    )
    print(f"upload-hosting: done  https://huggingface.co/datasets/{repo_id}")


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    parser = argparse.ArgumentParser(
        prog="upload-hosting",
        description="Push the prepared hosting packages to Hugging Face.",
    )
    parser.add_argument(
        "--namespace",
        default=DEFAULT_NAMESPACE,
        help=f"Hugging Face user or org owning the datasets (default: {DEFAULT_NAMESPACE})",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=REPO_ROOT / "dist" / "hosting",
        help="prepared package root (default: dist/hosting)",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="actually upload; without this the script only prints the plan",
    )
    args = parser.parse_args(argv)

    if not os.environ.get("HF_TOKEN"):
        print("upload-hosting: FAIL — set HF_TOKEN in the environment (never commit it).")
        return 2
    token = os.environ["HF_TOKEN"]

    packages = [args.source / name for name in HOSTING_ARTIFACTS]
    missing = [str(p) for p in packages if not (p / PACKAGE_MANIFEST_NAME).is_file()]
    if missing:
        print("upload-hosting: FAIL — missing prepared package(s):")
        for path in missing:
            print(f"  {path}")
        print("Run scripts/release/prepare_hosting.py first.")
        return 2

    _print_plan(packages, args.namespace)

    if not args.yes:
        print("upload-hosting: dry run only — re-run with --yes to upload.")
        return 0

    try:
        import huggingface_hub  # noqa: F401, PLC0415  (hosting-only dep)
    except ImportError:
        print(
            "upload-hosting: FAIL — huggingface_hub is not installed. Run via\n"
            "  uv run --group hosting python scripts/release/upload_hosting.py ..."
        )
        return 2

    for package in packages:
        _upload_package(package, f"{args.namespace}/{package.name}", token)
    return 0


if __name__ == "__main__":
    sys.exit(main())
