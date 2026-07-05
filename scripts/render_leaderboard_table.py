#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Regenerate (or drift-check) the README CHAMBER-Bench leaderboard.

The block between the ``CHAMBER-BENCH-LEADERBOARD`` markers in
``README.md`` is generated from the committed, verified result bundles
listed in each task's ``LEADERBOARD_BUNDLES.txt`` under
``spikes/results/benchmark/`` (ADR-027 §Reporting rules; every bundle
is re-verified via the ``chamber-eval verify`` check table before a
number is read — unverifiable entries refuse the render). Hand edits
to the block are overwritten; ``make verify-readme-tables`` and CI run
the ``--check`` mode so a committed README that drifts from the
verified bundles fails the build. Mirrors
``scripts/render_task_table.py`` (the CB-02 convention).

Requires the project environment, so invoke via uv::

    uv run python scripts/render_leaderboard_table.py          # rewrite README.md
    uv run python scripts/render_leaderboard_table.py --check  # report drift, write nothing

Exit codes:
    0  README block matches the render (or was rewritten).
    1  ``--check`` found drift.
    2  Usage error / markers missing / a listed bundle failed verification.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from chamber.evaluation.leaderboard import (
    README_LEADERBOARD_BEGIN,
    README_LEADERBOARD_END,
    LeaderboardInputError,
    render_readme_leaderboard,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_README = _REPO_ROOT / "README.md"

#: Exit code for ``--check`` drift, distinct from argparse's 2.
DRIFT_EXIT_CODE: int = 1


def splice_block(readme_text: str, block: str) -> str:
    """Return ``readme_text`` with the marker-delimited block replaced by ``block``.

    ``block`` is the full render including both markers. Raises
    ``ValueError`` when the markers are missing or out of order.
    """
    begin = readme_text.find(README_LEADERBOARD_BEGIN)
    end = readme_text.find(README_LEADERBOARD_END)
    if begin == -1 or end == -1 or end < begin:
        msg = (
            "README.md is missing the CHAMBER-BENCH-LEADERBOARD markers "
            "(or they are out of order); re-add both marker comments."
        )
        raise ValueError(msg)
    tail = readme_text[end + len(README_LEADERBOARD_END) :]
    return readme_text[:begin] + block.rstrip("\n") + tail


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    parser = argparse.ArgumentParser(
        description="Regenerate (or drift-check) the README CHAMBER-Bench leaderboard."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit 1 if the committed README block differs from the verified render.",
    )
    args = parser.parse_args(argv)

    readme_text = _README.read_text(encoding="utf-8")
    try:
        expected = splice_block(readme_text, render_readme_leaderboard(repo_path=_REPO_ROOT))
    except (LeaderboardInputError, ValueError) as exc:
        print(f"render_leaderboard_table: {exc}", file=sys.stderr)
        return 2

    if args.check:
        if readme_text != expected:
            print(
                "render_leaderboard_table: README.md leaderboard drifts from the "
                "verified bundles; run "
                "`uv run python scripts/render_leaderboard_table.py` and commit.",
                file=sys.stderr,
            )
            return DRIFT_EXIT_CODE
        print("render_leaderboard_table: README.md leaderboard matches the verified bundles.")
        return 0

    if readme_text == expected:
        print("render_leaderboard_table: README.md already up to date.")
        return 0
    _README.write_text(expected, encoding="utf-8")
    print("render_leaderboard_table: README.md leaderboard regenerated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
