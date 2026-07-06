#!/usr/bin/env python3
"""Build the three CHAMBER-Bench hosting packages under dist/hosting/.

Deterministic packaging of the released data artifacts (ADR-028
§Decision; ADR-027 §Versioning; ADR-012 §Decision licence): the
partner sets, the leaderboard bundles, and the reference/probe
trajectories. All logic lives in :mod:`chamber.evaluation.hosting`;
this wrapper adds the release-discipline guards:

* refuses a dirty working tree unless ``--allow-dirty`` (a hosted
  artifact must correspond to a commit);
* records the producing git commit in each package's
  ``manifest.json``;
* prints per-package file counts, byte totals, and the SHA-256 of
  each ``manifest.json`` so the founder can quote them in the release
  notes.

The founder-side upload is ``scripts/release/upload_hosting.py``.

Usage::

    uv run python scripts/release/prepare_hosting.py [--dest dist/hosting] [--allow-dirty]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from chamber.evaluation.hosting import (  # noqa: E402
    HOSTING_DEST_DIRNAME,
    PACKAGE_MANIFEST_NAME,
    build_all,
)


def _git(*args: str) -> str:
    return subprocess.run(  # noqa: S603
        ("git", *args),  # noqa: S607
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    parser = argparse.ArgumentParser(
        prog="prepare-hosting",
        description="Build the three hosting packages deterministically.",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=REPO_ROOT / HOSTING_DEST_DIRNAME,
        help=f"destination root (default: {HOSTING_DEST_DIRNAME})",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="build from a locally-modified tree (local dev only)",
    )
    args = parser.parse_args(argv)

    dirty = bool(_git("status", "--porcelain"))
    if dirty and not args.allow_dirty:
        print(
            "prepare-hosting: FAIL — working tree is dirty; commit first or "
            "pass --allow-dirty (a hosted artifact must correspond to a commit)."
        )
        return 7
    git_sha = _git("rev-parse", "HEAD") + ("-dirty" if dirty else "")

    dest: Path = args.dest
    if dest.exists():
        shutil.rmtree(dest)
    packages = build_all(REPO_ROOT, dest, git_sha=git_sha)

    print(f"prepare-hosting: built {len(packages)} packages under {dest} (git {git_sha})")
    for package in packages:
        manifest_path = package / PACKAGE_MANIFEST_NAME
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        total_bytes = sum(entry["bytes"] for entry in manifest["files"])
        manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        print(
            f"  {package.name}: {len(manifest['files'])} files, "
            f"{total_bytes / 1_048_576:.1f} MiB, "
            f"manifest.json sha256 {manifest_sha}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
