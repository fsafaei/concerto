#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Regenerate (or drift-check) the ``docs/reference/partners/`` partner cards.

One card per set member plus per-set and top-level indexes, rendered
from the :mod:`chamber.partners.sets` registry and the committed
fingerprint archives (ADR-009 §Decision as amended 2026-07-05; ADR-027
§Consequences: cards can never drift from code or evidence). Hand edits
are overwritten; ``make verify-readme-tables`` and CI run the
``--check`` mode, which fails on content drift **and** membership drift.

Requires the project environment, so invoke via uv::

    uv run python scripts/render_partner_cards.py          # rewrite docs/reference/partners/
    uv run python scripts/render_partner_cards.py --check  # report drift, write nothing

Exit codes:
    0  Cards match the registry + archive render (or were rewritten).
    1  ``--check`` found drift (content or membership).
    2  Usage error / missing fingerprint archive.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import chamber.partners  # noqa: F401  - registers the v1 sets
from chamber.partners.cards import archive_rel_dir, render_all_partner_cards
from chamber.partners.sets import get_partner_set, list_partner_sets, parse_set_slug

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CARDS_DIR = _REPO_ROOT / "docs" / "reference" / "partners"

#: Exit code for ``--check`` drift, distinct from argparse's 2.
DRIFT_EXIT_CODE: int = 1


def _load_fingerprints() -> dict[str, dict[str, Any]]:
    """Load every registered set's committed ``fingerprints.json`` (loud on absence)."""
    payloads: dict[str, dict[str, Any]] = {}
    for slug in list_partner_sets():
        set_id, version = parse_set_slug(slug)
        spec = get_partner_set(set_id, version=version)
        path = _REPO_ROOT / archive_rel_dir(spec) / "fingerprints.json"
        if not path.is_file():
            msg = (
                f"missing committed fingerprint archive for {slug}: {path} "
                "(run scripts/generate_partner_fingerprints.py and commit the archive)"
            )
            raise FileNotFoundError(msg)
        payloads[slug] = json.loads(path.read_text(encoding="utf-8"))
    return payloads


def diff_cards(cards_dir: Path) -> list[str]:
    """Return human-readable drift findings for ``cards_dir`` (empty = clean)."""
    expected = render_all_partner_cards(_load_fingerprints())
    committed = {
        str(p.relative_to(cards_dir)): p.read_text(encoding="utf-8")
        for p in sorted(cards_dir.rglob("*.md"))
    }
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
    expected = render_all_partner_cards(_load_fingerprints())
    if cards_dir.is_dir():
        for stale in sorted(
            {str(p.relative_to(cards_dir)) for p in cards_dir.rglob("*.md")} - set(expected)
        ):
            (cards_dir / stale).unlink()
    for name, content in sorted(expected.items()):
        target = cards_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return sorted(expected)


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    parser = argparse.ArgumentParser(
        description="Regenerate (or drift-check) the docs/reference/partners/ cards."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit 1 if docs/reference/partners/ differs from the registry render.",
    )
    args = parser.parse_args(argv)

    try:
        if args.check:
            findings = diff_cards(_CARDS_DIR)
            if findings:
                for finding in findings:
                    print(f"partner-cards drift: {finding}", file=sys.stderr)
                return DRIFT_EXIT_CODE
            print(f"partner cards match the registry render ({_CARDS_DIR}).")
            return 0
        written = write_cards(_CARDS_DIR)
        print(f"wrote {len(written)} partner cards under {_CARDS_DIR}.")
        return 0
    except (FileNotFoundError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
