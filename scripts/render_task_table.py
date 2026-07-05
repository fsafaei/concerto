#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Regenerate (or drift-check) the README CHAMBER-Bench task table.

The table between the ``CHAMBER-BENCH-TASK-TABLE`` markers in
``README.md`` is generated from the ``chamber.tasks`` registry
(ADR-027 §Versioning: the registry is the single source of truth for
the suite composition). Hand edits to the block are overwritten;
``make verify-readme-tables`` and CI run the ``--check`` mode so a
committed README that drifts from the registry fails the build.

Requires the project environment (imports ``chamber.tasks``), so
invoke via uv::

    uv run python scripts/render_task_table.py          # rewrite README.md
    uv run python scripts/render_task_table.py --check  # report drift, write nothing

Exit codes:
    0  README block matches the registry render (or was rewritten).
    1  ``--check`` found drift.
    2  Usage error / markers missing from README.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from chamber.tasks.render import (
    README_TABLE_BEGIN,
    README_TABLE_END,
    render_readme_table,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_README = _REPO_ROOT / "README.md"

#: Exit code for ``--check`` drift, distinct from argparse's 2.
DRIFT_EXIT_CODE: int = 1


def splice_table(readme_text: str, table: str) -> str:
    """Return ``readme_text`` with the marker-delimited block replaced by ``table``.

    ``table`` is the full render including both markers. Raises
    ``ValueError`` when the markers are missing or out of order.
    """
    begin = readme_text.find(README_TABLE_BEGIN)
    end = readme_text.find(README_TABLE_END)
    if begin == -1 or end == -1 or end < begin:
        msg = (
            "README.md is missing the CHAMBER-BENCH-TASK-TABLE markers "
            "(or they are out of order); re-add both marker comments."
        )
        raise ValueError(msg)
    tail = readme_text[end + len(README_TABLE_END) :]
    return readme_text[:begin] + table.rstrip("\n") + tail


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    parser = argparse.ArgumentParser(
        description="Regenerate (or drift-check) the README CHAMBER-Bench task table."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit 1 if the committed README block differs from the registry render.",
    )
    args = parser.parse_args(argv)

    readme_text = _README.read_text(encoding="utf-8")
    try:
        expected = splice_table(readme_text, render_readme_table())
    except ValueError as exc:
        print(f"render_task_table: {exc}", file=sys.stderr)
        return 2

    if args.check:
        if readme_text != expected:
            print(
                "render_task_table: README.md task table drifts from the "
                "chamber.tasks registry; run "
                "`uv run python scripts/render_task_table.py` and commit.",
                file=sys.stderr,
            )
            return DRIFT_EXIT_CODE
        print("render_task_table: README.md task table matches the registry.")
        return 0

    if readme_text == expected:
        print("render_task_table: README.md already up to date.")
        return 0
    _README.write_text(expected, encoding="utf-8")
    print("render_task_table: README.md task table regenerated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
