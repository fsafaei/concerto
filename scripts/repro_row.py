#!/usr/bin/env python3
r"""Re-run one committed leaderboard bundle and require an exact summary match.

The reproduction contract (ADR-028 §Decision; principle P6) is that a
CPU re-run of a leaderboard row under the committed ``uv.lock`` and
the recorded seed schedule reproduces the bundle's summary statistics
exactly. This script executes that contract for one bundle:

1. read the committed bundle's ``REPRO.txt`` (the single source of
   truth for the row's command — nothing is duplicated here or in the
   workflow that calls this),
2. re-run it with a fresh ``--out`` directory,
3. ``chamber-eval verify`` the fresh bundle, and
4. compare the fresh ``bundle.json`` summary fields
   (``success_iqm``, ``success_mean``, ``success_ci_low``,
   ``success_ci_high``, ``n_episodes``) against the committed ones,
   failing on any difference.

Used by ``.github/workflows/repro-rows.yml`` (weekly + on demand) for
the CPU-feasible scripted rows; runnable locally for any row whose
checkpoints are present.

Usage::

    uv run python scripts/repro_row.py spikes/results/benchmark/cocarry-v1/b-rnd-2026-07-05 \\
        --out out/repro/b-rnd [--allow-dirty]
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

_OUT_FLAG = re.compile(r"--out\s+\S+")

SUMMARY_FIELDS = (
    "n_episodes",
    "success_mean",
    "success_iqm",
    "success_ci_low",
    "success_ci_high",
)


def _summary(bundle_dir: Path) -> dict[str, float]:
    bundle = json.loads((bundle_dir / "bundle.json").read_text(encoding="utf-8"))
    summary = bundle["summary"]
    return {field: summary[field] for field in SUMMARY_FIELDS}


def _run(command: list[str]) -> None:
    print(f"repro-row: $ {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)  # noqa: S603


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    parser = argparse.ArgumentParser(
        prog="repro-row",
        description="Re-run one committed leaderboard bundle from its REPRO.txt.",
    )
    parser.add_argument("bundle", type=Path, help="committed bundle directory")
    parser.add_argument("--out", type=Path, required=True, help="fresh output directory")
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="pass --allow-dirty to chamber-eval (local dev only)",
    )
    args = parser.parse_args(argv)

    bundle_dir = (REPO_ROOT / args.bundle).resolve()
    repro_file = bundle_dir / "REPRO.txt"
    if not repro_file.is_file():
        raise SystemExit(f"repro-row: FAIL — {repro_file} not found")

    command_text = repro_file.read_text(encoding="utf-8").replace("\\\n", " ").strip()
    if not _OUT_FLAG.search(command_text):
        raise SystemExit(f"repro-row: FAIL — no --out flag in {repro_file}")
    out_dir = REPO_ROOT / args.out
    shutil.rmtree(out_dir, ignore_errors=True)
    command_text = _OUT_FLAG.sub(f"--out {args.out.as_posix()}", command_text)
    command = command_text.split()
    if args.allow_dirty:
        command.append("--allow-dirty")

    _run(["uv", "run", *command])
    verify = ["uv", "run", "chamber-eval", "verify", args.out.as_posix()]
    if args.allow_dirty:
        verify.append("--allow-dirty")
    _run(verify)

    committed = _summary(bundle_dir)
    fresh = _summary(out_dir)
    mismatches = [
        f"  {field}: committed {committed[field]!r} vs fresh {fresh[field]!r}"
        for field in SUMMARY_FIELDS
        if committed[field] != fresh[field]
    ]
    if mismatches:
        print(f"repro-row: FAIL — summary mismatch for {args.bundle}:")
        print("\n".join(mismatches))
        return 1
    print(
        f"repro-row: PASS — {args.bundle} reproduced exactly "
        f"({committed['n_episodes']:.0f} episodes, "
        f"IQM {committed['success_iqm']:.3f})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
