# SPDX-License-Identifier: Apache-2.0
"""``chamber-render-tables`` console entry point (ADR-014 §Decision, ADR-008 §Decision).

Renders the three-table safety report (ADR-014) and the leaderboard
(ADR-008) from JSON archives produced by the spike runners and the
:mod:`chamber.evaluation` pipeline. The CLI is intentionally thin:
the heavy lifting lives in :mod:`chamber.evaluation.render`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import chamber
from chamber.evaluation import (
    LeaderboardEntry,
    render_leaderboard,
    render_three_table_safety_report,
)


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="chamber-render-tables",
        description=(
            "Render the ADR-014 three-table safety report and the ADR-008 "
            "leaderboard from JSON archives."
        ),
    )
    parser.add_argument(
        "--safety-report",
        type=Path,
        default=None,
        help="Path to a three_tables.json from concerto.safety.reporting.",
    )
    parser.add_argument(
        "--leaderboard",
        type=Path,
        default=None,
        help="Path to a leaderboard.json (list of LeaderboardEntry dicts).",
    )
    parser.add_argument(
        "--fmt",
        choices=("markdown", "latex"),
        default="markdown",
        help="Output format for the safety report (default: markdown).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write the rendered output here; print to stdout when omitted.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``chamber-render-tables`` console script (ADR-014 §Decision).

    Without arguments, prints the banner so the install can be smoke-
    tested. With ``--safety-report`` or ``--leaderboard``, renders the
    corresponding artefact and either prints it or writes it to
    ``--output``.

    Args:
        argv: Optional list of CLI arguments (testing hook).

    Returns:
        ``0`` on success; ``2`` when required JSON files are missing.
    """
    args = _parse_args(argv)
    if args.safety_report is None and args.leaderboard is None:
        print(f"chamber-render-tables  (CHAMBER {chamber.__version__})")
        print("Usage: chamber-render-tables --safety-report PATH | --leaderboard PATH")
        return 0

    chunks: list[str] = []
    if args.safety_report is not None:
        if not args.safety_report.exists():
            print(f"error: {args.safety_report} not found", file=sys.stderr)
            return 2
        report = _read_json(args.safety_report)
        if not isinstance(report, dict):
            print(
                f"error: {args.safety_report} is not a JSON object",
                file=sys.stderr,
            )
            return 2
        chunks.append(render_three_table_safety_report(report, fmt=args.fmt))

    if args.leaderboard is not None:
        if not args.leaderboard.exists():
            print(f"error: {args.leaderboard} not found", file=sys.stderr)
            return 2
        raw = _read_json(args.leaderboard)
        if not isinstance(raw, list):
            print(
                f"error: {args.leaderboard} must contain a JSON list of entries",
                file=sys.stderr,
            )
            return 2
        entries = [LeaderboardEntry.model_validate(item) for item in raw]
        chunks.append(render_leaderboard(entries))

    rendered = "\n".join(chunks)
    if args.output is not None:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered)
    return 0
