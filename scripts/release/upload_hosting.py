#!/usr/bin/env python3
r"""Upload the prepared hosting packages to the maintainer's Hugging Face namespace.

Founder-side companion to ``prepare_hosting.py`` (ADR-028 §Decision;
ADR-012 §Decision licence). Safety properties:

* the token comes from the ``HF_TOKEN`` environment variable and is
  never read from a file, never echoed, and never committed;
* the script prints exactly what it will upload (repo ids, file
  counts, byte totals, manifest digests) and then **stops** unless
  ``--yes`` was passed — there is no interactive fallthrough;
* ``huggingface_hub`` is imported lazily so the project's dependency
  tree stays free of hosting tooling; run via
  ``uv run --with huggingface_hub python scripts/release/upload_hosting.py``.

Usage::

    export HF_TOKEN=<maintainer token>          # never commit this
    uv run --with huggingface_hub python scripts/release/upload_hosting.py \\
        --namespace <hf-user-or-org> [--source dist/hosting] [--yes]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

HOSTING_ARTIFACTS = (
    "chamber-partner-sets",
    "chamber-leaderboard-bundles",
    "chamber-reference-trajectories",
)


def _describe(package: Path) -> str:
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    total_bytes = sum(entry["bytes"] for entry in manifest["files"])
    return (
        f"{len(manifest['files'])} files, {total_bytes / 1_048_576:.1f} MiB, "
        f"built from git {manifest['git_sha']}"
    )


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    parser = argparse.ArgumentParser(
        prog="upload-hosting",
        description="Push the prepared hosting packages to Hugging Face.",
    )
    parser.add_argument(
        "--namespace",
        required=True,
        help="Hugging Face user or organisation that will own the datasets",
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

    packages = [args.source / name for name in HOSTING_ARTIFACTS]
    missing = [str(p) for p in packages if not (p / "manifest.json").is_file()]
    if missing:
        print("upload-hosting: FAIL — missing prepared package(s):")
        for path in missing:
            print(f"  {path}")
        print("Run scripts/release/prepare_hosting.py first.")
        return 2

    print("upload-hosting: plan")
    for package in packages:
        repo_id = f"{args.namespace}/{package.name}"
        print(f"  datasets/{repo_id}  <-  {package}  ({_describe(package)})")

    if not args.yes:
        print("upload-hosting: dry run only — re-run with --yes to upload.")
        return 0

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("upload-hosting: FAIL — set HF_TOKEN in the environment (never commit it).")
        return 2

    try:
        from huggingface_hub import HfApi  # noqa: PLC0415  (lazy: hosting-only dep)
    except ImportError:
        print(
            "upload-hosting: FAIL — huggingface_hub is not installed. Run via\n"
            "  uv run --with huggingface_hub python scripts/release/upload_hosting.py ..."
        )
        return 2

    api = HfApi(token=token)
    for package in packages:
        repo_id = f"{args.namespace}/{package.name}"
        print(f"upload-hosting: uploading {package.name} -> datasets/{repo_id}")
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
        api.upload_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=str(package),
            commit_message=f"CHAMBER-Bench v1.0 {package.name} ({_describe(package)})",
        )
        print(f"upload-hosting: done  https://huggingface.co/datasets/{repo_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
