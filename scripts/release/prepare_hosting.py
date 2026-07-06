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
    HOSTING_ARTIFACTS,
    HOSTING_DEST_DIRNAME,
    HOSTING_SOURCE_DIRNAME,
    PACKAGE_CARD_NAME,
    PACKAGE_MANIFEST_NAME,
    build_all,
)


def _ensure_viewer_disabled(repo_root: Path) -> list[Path]:
    """Write ``viewer: false`` into each committed dataset card's frontmatter.

    The packages are download-and-verify evidence archives with
    deliberately heterogeneous per-episode schemas, not row-browseable
    tables; without ``viewer: false`` the Hugging Face auto-viewer
    raises a CastError banner on the dataset page. Idempotent: cards
    that already carry the key are left byte-identical. Returns the
    cards that were modified (the caller fails the dirty-tree gate so
    the fix gets committed rather than shipped silently).
    """
    modified: list[Path] = []
    for name in HOSTING_ARTIFACTS:
        card = repo_root / HOSTING_SOURCE_DIRNAME / name / PACKAGE_CARD_NAME
        text = card.read_text(encoding="utf-8")
        if not text.startswith("---\n"):
            card.write_text(f"---\nviewer: false\n---\n\n{text}", encoding="utf-8")
            modified.append(card)
            continue
        frontmatter, _, _ = text[4:].partition("\n---\n")
        if "viewer:" in frontmatter:
            continue
        card.write_text("---\nviewer: false\n" + text[4:], encoding="utf-8")
        modified.append(card)
    return modified


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

    fixed_cards = _ensure_viewer_disabled(REPO_ROOT)
    for card in fixed_cards:
        print(f"prepare-hosting: wrote 'viewer: false' into {card.relative_to(REPO_ROOT)}")

    dirty = bool(_git("status", "--porcelain"))
    if dirty and not args.allow_dirty:
        print(
            "prepare-hosting: FAIL — working tree is dirty; commit first or "
            "pass --allow-dirty (a hosted artifact must correspond to a commit)."
        )
        if fixed_cards:
            print(
                "prepare-hosting: (the dataset-card fixes above are part of the "
                "dirt — commit them and re-run)"
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
