# SPDX-License-Identifier: Apache-2.0
"""``chamber-eval admission`` — run the Tier-2 admission protocol (ADR-027 §Admission protocol).

One command, one committed artefact: gate on the pre-registration tag
(exit ``PREREG_MISMATCH_EXIT_CODE`` before any cell runs — ADR-007
§Discipline), refuse a dirty working tree unless ``--allow-dirty``
(exit ``DIRTY_TREE_EXIT_CODE``), execute the A1/A2/A3 checks — plus A4
ego-robustness when the prereg commits ``c_min_ego`` (ADR-027
§Admission A4) — via
:func:`chamber.evaluation.admission.run_admission` (every measured cell
an ADR-028 v3 bundle), and write the immutable admission archive the
registry flips cite (invariant I8).
"""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

from pydantic import ValidationError

from chamber.evaluation.admission import (
    AdmissionError,
    admission_spec_from_prereg,
    run_admission,
)
from chamber.evaluation.bundles import DIRTY_TREE_EXIT_CODE
from chamber.evaluation.prereg import (
    PREREG_MISMATCH_EXIT_CODE,
    PreregistrationError,
    load_prereg_document,
    verify_git_tag,
)

_USAGE_EXIT_CODE = 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chamber-eval admission",
        description=(
            "Run the ADR-027 Tier-2 admission protocol (A1 solvability / A2 "
            "two-robot infeasibility / A3 partner-relevance / A4 instrument "
            "ego-robustness for instrument-based contrasts) under a tag-locked "
            "pre-registration and write the immutable admission archive."
        ),
    )
    parser.add_argument(
        "--prereg",
        type=Path,
        required=True,
        help="Document-form pre-registration YAML carrying the 'admission' block.",
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="Admission archive directory to create."
    )
    parser.add_argument(
        "--date",
        required=True,
        help="Archive date label (e.g. 2026-07-05); recorded, never sampled (P6).",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help=(
            "Run from a dirty working tree anyway; the report is marked "
            "dirty: true and is never registry-flip evidence."
        ),
    )
    parser.add_argument(
        "--render-backend",
        default=None,
        help="Render backend forwarded to SAPIEN cell runners ('none' on headless hosts).",
    )
    return parser


def run(argv: list[str]) -> int:
    """Entry point for ``chamber-eval admission`` (ADR-027 §Admission protocol).

    Returns ``0`` on success (whatever the verdict — a CONTROL demotion
    is a successful protocol run); ``PREREG_MISMATCH_EXIT_CODE`` (4)
    when the pre-registration gate fails before any cell runs;
    ``DIRTY_TREE_EXIT_CODE`` (7) on a dirty tree without
    ``--allow-dirty``; ``2`` on usage errors.
    """
    args = _build_parser().parse_args(argv)
    repo_path = Path.cwd()

    # ADR-007 §Discipline: the gate runs before anything is measured.
    try:
        doc = load_prereg_document(args.prereg)
        verify_git_tag(doc, args.prereg, repo_path=repo_path)
        spec = admission_spec_from_prereg(doc)
    except (PreregistrationError, ValidationError, OSError, ValueError, AdmissionError) as exc:
        print(f"error: pre-registration gate failed: {exc}", file=sys.stderr)
        return PREREG_MISMATCH_EXIT_CODE

    repro_command = "chamber-eval admission " + shlex.join(argv)
    try:
        report = run_admission(
            spec,
            out_dir=args.out,
            repo_path=repo_path,
            prereg_path=args.prereg,
            date_stamp=str(args.date),
            repro_command=repro_command,
            allow_dirty=args.allow_dirty,
            render_backend=args.render_backend,
        )
    except AdmissionError as exc:
        message = str(exc)
        if "dirty" in message:
            print(f"error: {message}", file=sys.stderr)
            return DIRTY_TREE_EXIT_CODE
        print(f"error: {message}", file=sys.stderr)
        return _USAGE_EXIT_CODE
    except PreregistrationError as exc:  # re-verification inside the runner
        print(f"error: pre-registration gate failed: {exc}", file=sys.stderr)
        return PREREG_MISMATCH_EXIT_CODE

    dirty_note = "  [DIRTY — not flip evidence]" if report.dirty else ""
    print(
        f"chamber-eval admission: {report.task_id}@v{report.task_version} -> "
        f"{report.verdict} (checks: "
        + ", ".join(f"{c.check}={c.outcome}" for c in report.checks)
        + f"); archive at {args.out}{dirty_note}"
    )
    return 0


__all__ = ["run"]
