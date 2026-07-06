#!/usr/bin/env python3
"""Trace every leaderboard row to a documented reproduction command.

CHAMBER-Bench's reproducibility promise (ADR-027 §Reporting rules;
ADR-028 §Decision) is only honest if the documentation cannot drift
from the evidence. This script pins three invariants between the
committed bundles and ``docs/how-to/reproduce-results.md``:

1. **Command fidelity.** For every bundle listed in a per-task
   ``LEADERBOARD_BUNDLES.txt`` manifest, the doc contains its
   ``REPRO.txt`` command verbatim, modulo the ``uv run`` prefix,
   line-continuation whitespace, and the ``--out`` directory (a
   reproducer must write to a fresh directory, never into the
   committed archive).
2. **Bundle traceability.** The doc names every committed bundle
   path ("expected bundle") and carries a ``chamber-eval verify``
   invocation for it.
3. **README coverage.** Every bundle path referenced by the README
   leaderboard block appears in the doc, so no rendered row lacks a
   reproduction recipe.

Stdlib-only; run by ``make verify-readme-tables`` and the
``verify-readme-tables`` CI job.

Usage::

    uv run python scripts/check_repro_docs.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DOC = REPO_ROOT / "docs" / "how-to" / "reproduce-results.md"
README = REPO_ROOT / "README.md"
BENCHMARK_ROOT = REPO_ROOT / "spikes" / "results" / "benchmark"
LEADERBOARD_MANIFEST = "LEADERBOARD_BUNDLES.txt"

README_LEADERBOARD = re.compile(
    r"<!-- CHAMBER-BENCH-LEADERBOARD:BEGIN.*?-->(.*?)"
    r"<!-- CHAMBER-BENCH-LEADERBOARD:END -->",
    flags=re.DOTALL,
)
_OUT_FLAG = re.compile(r"--out\s+\S+")
_WS = re.compile(r"\s+")


def _normalise(command: str) -> str:
    command = command.replace("\\\n", " ").strip()
    command = command.removeprefix("uv run ")
    return _WS.sub(" ", _OUT_FLAG.sub("--out <DIR>", command))


def read_manifest_bundles() -> list[Path]:
    """Return every bundle dir listed in a per-task LEADERBOARD_BUNDLES.txt."""
    bundles: list[Path] = []
    for manifest in sorted(BENCHMARK_ROOT.glob(f"*/{LEADERBOARD_MANIFEST}")):
        for raw_line in manifest.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if line and not line.startswith("#"):
                bundles.append(REPO_ROOT / line)
    return bundles


def doc_commands(doc_text: str) -> set[str]:
    """Return the normalised command lines of every bash block in the doc."""
    commands: set[str] = set()
    for block in re.findall(r"```bash\n(.*?)```", doc_text, flags=re.DOTALL):
        logical = block.replace("\\\n", " ")
        for raw_line in logical.splitlines():
            line = raw_line.strip()
            if line and not line.startswith("#"):
                commands.add(_normalise(line))
    return commands


def main() -> int:
    """Entry point; returns the process exit code."""
    doc_text = DOC.read_text(encoding="utf-8")
    commands = doc_commands(doc_text)
    failures: list[str] = []

    bundles = read_manifest_bundles()
    if not bundles:
        failures.append(f"no bundles found under {BENCHMARK_ROOT}/*/{LEADERBOARD_MANIFEST}")

    for bundle in bundles:
        rel = bundle.relative_to(REPO_ROOT).as_posix()
        repro = bundle / "REPRO.txt"
        if not repro.is_file():
            failures.append(f"{rel}: missing REPRO.txt")
            continue
        want = _normalise(repro.read_text(encoding="utf-8"))
        if want not in commands:
            failures.append(f"{rel}: REPRO.txt command not documented:\n    {want}")
        if rel not in doc_text:
            failures.append(f"{rel}: bundle path not named in {DOC.name}")
        if _normalise(f"chamber-eval verify {rel}") not in commands:
            failures.append(f"{rel}: no `chamber-eval verify {rel}` in {DOC.name}")

    readme_block = README_LEADERBOARD.search(README.read_text(encoding="utf-8"))
    if readme_block is None:
        failures.append("README leaderboard markers not found")
    else:
        readme_bundles = set(
            re.findall(r"`(spikes/results/benchmark/[^`]+)`", readme_block.group(1))
        )
        failures.extend(
            f"README leaderboard bundle {rel} not covered in {DOC.name}"
            for rel in sorted(readme_bundles)
            if rel not in doc_text
        )

    if failures:
        print(f"check-repro-docs: FAIL — {len(failures)} problem(s)")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print(
        f"check-repro-docs: PASS ({len(bundles)} leaderboard bundles traced to documented commands)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
