# SPDX-License-Identifier: Apache-2.0
"""``chamber-eval verify`` — the leaderboard admission gate (ADR-028 §Decision 3).

Recomputes everything derivable in a v3 bundle — summary statistics
from the raw episode records (byte-reproducible bootstrap, ADR-002
P6), every file hash against both integrity layers, partner identity
hashes against the registry, and the prereg tag linkage when
referenced — prints the PASS/FAIL table, and exits non-zero on any
failure. CPU-only, on a bundle produced anywhere: this is the command
an external reproducer runs first (ADR-028 §Decision 3).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from chamber.evaluation.bundles import render_check_table, verify_bundle_dir

_USAGE_EXIT_CODE = 2
_VERIFY_FAIL_EXIT_CODE = 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chamber-eval verify",
        description=(
            "Verify a v3 result bundle: schema, file-hash manifest + "
            "SHA256SUMS, recomputed summary statistics, partner identity "
            "hashes, prereg tag linkage. Exits non-zero on any failure."
        ),
    )
    parser.add_argument("bundle_dir", type=Path, help="Bundle directory to verify.")
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help=(
            "Tolerate a dirty-flagged bundle (local development loops only; "
            "the leaderboard admission gate never passes this)."
        ),
    )
    return parser


def run(argv: list[str]) -> int:
    """Entry point for ``chamber-eval verify`` (ADR-028 §Decision 3).

    Returns ``0`` when every check passes; ``1`` on any check failure;
    ``2`` when the bundle directory does not exist.
    """
    args = _build_parser().parse_args(argv)
    if not args.bundle_dir.is_dir():
        print(f"error: {args.bundle_dir} is not a directory", file=sys.stderr)
        return _USAGE_EXIT_CODE
    rows = verify_bundle_dir(args.bundle_dir, repo_path=Path.cwd(), allow_dirty=args.allow_dirty)
    print(render_check_table(rows))
    return 0 if all(row.ok for row in rows) else _VERIFY_FAIL_EXIT_CODE


__all__ = ["run"]
