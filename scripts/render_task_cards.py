#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Regenerate (or drift-check) the ``docs/reference/tasks/`` task cards.

One card per task version plus an ``index.md``, all rendered from the
``chamber.tasks`` registry (ADR-027 §Consequences: cards can never
drift from code). Hand edits are overwritten; ``make
verify-readme-tables`` and CI run the ``--check`` mode, which fails on
content drift **and** on membership drift (a committed card whose task
left the registry, or a registered task with no committed card).

Requires the project environment (imports ``chamber.tasks``), so
invoke via uv::

    uv run python scripts/render_task_cards.py          # rewrite docs/reference/tasks/
    uv run python scripts/render_task_cards.py --check  # report drift, write nothing

Exit codes:
    0  Cards match the registry render (or were rewritten).
    1  ``--check`` found drift (content or membership).
    2  Usage error.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from chamber.tasks.render import render_all_cards

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CARDS_DIR = _REPO_ROOT / "docs" / "reference" / "tasks"

#: Exit code for ``--check`` drift, distinct from argparse's 2.
DRIFT_EXIT_CODE: int = 1


def diff_cards(cards_dir: Path) -> list[str]:
    """Return human-readable drift findings for ``cards_dir`` (empty = clean)."""
    expected = render_all_cards()
    committed = {p.name: p.read_text(encoding="utf-8") for p in sorted(cards_dir.glob("*.md"))}
    findings = [f"missing card: {name}" for name in sorted(set(expected) - set(committed))]
    findings.extend(
        f"stale card not in the registry: {name}" for name in sorted(set(committed) - set(expected))
    )
    findings.extend(
        f"content drift: {name}"
        for name in sorted(set(expected) & set(committed))
        if expected[name] != committed[name]
    )
    return findings


def write_cards(cards_dir: Path) -> list[str]:
    """(Re)write every card under ``cards_dir``; return the written file names."""
    cards_dir.mkdir(parents=True, exist_ok=True)
    expected = render_all_cards()
    for stale in sorted({p.name for p in cards_dir.glob("*.md")} - set(expected)):
        (cards_dir / stale).unlink()
    for name, content in sorted(expected.items()):
        (cards_dir / name).write_text(content, encoding="utf-8")
    return sorted(expected)


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    parser = argparse.ArgumentParser(
        description="Regenerate (or drift-check) the docs/reference/tasks/ task cards."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit 1 if docs/reference/tasks/ differs from the registry render.",
    )
    args = parser.parse_args(argv)

    if args.check:
        findings = diff_cards(_CARDS_DIR)
        if findings:
            for finding in findings:
                print(f"render_task_cards: {finding}", file=sys.stderr)
            print(
                "render_task_cards: docs/reference/tasks/ drifts from the "
                "chamber.tasks registry; run "
                "`uv run python scripts/render_task_cards.py` and commit.",
                file=sys.stderr,
            )
            return DRIFT_EXIT_CODE
        print("render_task_cards: docs/reference/tasks/ matches the registry.")
        return 0

    written = write_cards(_CARDS_DIR)
    print(f"render_task_cards: wrote {len(written)} files under docs/reference/tasks/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
