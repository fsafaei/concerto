#!/usr/bin/env python3
"""Prove the README quickstart executes verbatim (ADR-028 §Validation criteria).

The README's five-minute quickstart promises that its command blocks
"are executed verbatim by CI on every push". This script makes that
promise checkable: it extracts every ``bash`` fenced block between the
``## Five-minute quickstart`` heading and the next ``## `` heading,
and

* in check mode (default) asserts the extracted commands are the
  smoke-eval commands the CI job runs (same task, policy, partner,
  seeds, and episodes as ``scripts/smoke_eval.sh``), so the README
  cannot silently drift from what CI proves;
* with ``--execute`` runs every extracted command line in order via
  ``bash -euo pipefail`` on the current checkout, so CI executes the
  README text itself, not a parallel copy.

Stdlib-only on purpose: the CI job runs it before a full project sync.

Usage::

    uv run python scripts/check_readme_quickstart.py            # drift check
    uv run python scripts/check_readme_quickstart.py --execute  # run the blocks
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
README = REPO_ROOT / "README.md"
SMOKE_EVAL = REPO_ROOT / "scripts" / "smoke_eval.sh"

QUICKSTART_HEADING = "## Five-minute quickstart"

#: Flags whose values legitimately differ between the README (a fresh
#: user-facing output directory) and smoke_eval.sh (a mktemp dir).
_OUT_FLAG = re.compile(r"--out\s+\S+")
_WS = re.compile(r"\s+")


def extract_quickstart_commands(readme_text: str) -> list[str]:
    """Return the command lines of every bash block in the quickstart section."""
    try:
        start = readme_text.index(QUICKSTART_HEADING)
    except ValueError:
        raise SystemExit(
            f"check-readme-quickstart: FAIL — heading {QUICKSTART_HEADING!r} not found in {README}"
        ) from None
    section = readme_text[start + len(QUICKSTART_HEADING) :]
    next_heading = re.search(r"^## ", section, flags=re.MULTILINE)
    if next_heading:
        section = section[: next_heading.start()]
    blocks = re.findall(r"```bash\n(.*?)```", section, flags=re.DOTALL)
    if not blocks:
        raise SystemExit(
            "check-readme-quickstart: FAIL — no ```bash blocks in the quickstart section"
        )
    commands: list[str] = []
    # Join backslash-continued lines so each command is one logical line.
    for block in blocks:
        logical = block.replace("\\\n", " ")
        for raw_line in logical.splitlines():
            line = raw_line.strip()
            if line and not line.startswith("#"):
                commands.append(line)
    return commands


def _normalise(command: str) -> str:
    """Collapse whitespace and neutralise the ``--out`` value."""
    return _WS.sub(" ", _OUT_FLAG.sub("--out <DIR>", command)).strip()


def check_matches_smoke_eval(commands: list[str]) -> None:
    """Assert the quickstart's eval pair matches scripts/smoke_eval.sh."""
    smoke_text = SMOKE_EVAL.read_text(encoding="utf-8").replace("\\\n", " ")
    smoke_run = next(
        (
            _WS.sub(" ", line).strip()
            for line in smoke_text.splitlines()
            if "chamber-eval run" in line and not line.lstrip().startswith("#")
        ),
        None,
    )
    if smoke_run is None:
        raise SystemExit(
            f"check-readme-quickstart: FAIL — no chamber-eval run line in {SMOKE_EVAL}"
        )
    readme_run = next((c for c in commands if "chamber-eval run" in c), None)
    readme_verify = next((c for c in commands if "chamber-eval verify" in c), None)
    if readme_run is None or readme_verify is None:
        raise SystemExit(
            "check-readme-quickstart: FAIL — quickstart must contain a "
            "chamber-eval run and a chamber-eval verify command"
        )
    # smoke_eval.sh appends the optional "${allow_dirty[@]}" expansion and
    # quotes its mktemp output dir; strip both before comparing.
    smoke_norm = _normalise(
        smoke_run.replace('"${allow_dirty[@]}"', "").replace('"$tmp/bundle"', "OUT")
    ).replace("--out OUT", "--out <DIR>")
    readme_norm = _normalise(readme_run)
    if smoke_norm != readme_norm:
        raise SystemExit(
            "check-readme-quickstart: FAIL — README quickstart run command "
            "drifted from scripts/smoke_eval.sh:\n"
            f"  README:        {readme_norm}\n"
            f"  smoke_eval.sh: {smoke_norm}"
        )
    print("check-readme-quickstart: PASS (README quickstart matches smoke_eval.sh)")


def execute(commands: list[str]) -> None:
    """Run every quickstart command line in order; any failure fails the check."""
    # The quickstart's first command writes a fresh bundle; make re-runs
    # (e.g. re-triggered CI on a persistent runner) start clean.
    out_dirs = {
        match.group(0).split(None, 1)[1]
        for command in commands
        for match in [_OUT_FLAG.search(command)]
        if match
    }
    for out_dir in out_dirs:
        shutil.rmtree(REPO_ROOT / out_dir, ignore_errors=True)
    for command in commands:
        print(f"check-readme-quickstart: $ {command}", flush=True)
        result = subprocess.run(  # noqa: S603
            ["bash", "-euo", "pipefail", "-c", command],  # noqa: S607
            cwd=REPO_ROOT,
            check=False,
        )
        if result.returncode != 0:
            raise SystemExit(
                f"check-readme-quickstart: FAIL — command exited {result.returncode}: {command}"
            )
    print(f"check-readme-quickstart: PASS ({len(commands)} commands executed)")


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    parser = argparse.ArgumentParser(
        prog="check-readme-quickstart",
        description="Prove the README quickstart executes verbatim.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="run the extracted commands instead of only checking drift",
    )
    args = parser.parse_args(argv)
    commands = extract_quickstart_commands(README.read_text(encoding="utf-8"))
    check_matches_smoke_eval(commands)
    if args.execute:
        execute(commands)
    return 0


if __name__ == "__main__":
    sys.exit(main())
