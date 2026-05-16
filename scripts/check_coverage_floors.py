#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Gate per-package line-coverage against the project's plan §6 floors.

The project's pytest setup (``[tool.coverage.report] fail_under = 80``
in ``pyproject.toml``) only enforces an aggregate floor. That is not
sufficient: a regression in :mod:`concerto.safety` (where plan/03 §6 #8
demands ≥90% coverage on the safety stack) could be masked by
:mod:`chamber.benchmarks` improvements, because the aggregate would
still clear 80%.

This script reads the ``coverage.xml`` written by pytest's ``--cov-report=xml``
(see :data:`COVERAGE_FLOORS` below for the table of per-package floors
and their plan §6 citations) and exits non-zero if any named package
is missing from the report or below its floor. Wired into ``make
verify`` as the ``verify-coverage-floors`` target, run after ``make
test`` so the ``coverage.xml`` is fresh.

A *missing* package is treated as a failure rather than as "no
measurement, vacuous pass". A package contributing zero lines to the
coverage report is the worst kind of silent drift — it means an
``__init__.py``-only directory was created, or a refactor moved the
module out of the source tree, or the package isn't on
``[tool.coverage.run] source``. The verbose stderr report names the
missing package alongside its plan §6 anchor so the maintainer can
decide whether to update the table or restore the package.

Usage::

    uv run python scripts/check_coverage_floors.py
    uv run python scripts/check_coverage_floors.py --coverage-xml path/to/coverage.xml

References (plan §6):

- plan/02 §6 #2 — chamber.comm ≥90%
- plan/03 §6 #8 — concerto.safety ≥90%
- plan/04 §6 #5 — chamber.partners ≥90%
- plan/05 §6 #8 — concerto.training ≥85%
- plan/06 §6 #6 — chamber.evaluation ≥90%
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class _Floor:
    """One row of :data:`COVERAGE_FLOORS`.

    Carries the floor and the plan §6 anchor so the failure report can
    cite both and the maintainer never has to grep across the codebase
    to remember *why* a given package has the floor it does.
    """

    floor: float  # in [0, 1]; compared against coverage.xml's ``line-rate``
    plan_ref: str  # human-readable plan §6 citation


#: Per-package line-coverage floors. The keys match the dotted ``name``
#: attribute on ``<package>`` elements in pytest's ``coverage.xml``
#: (which is the directory path under ``src/`` rendered with ``.`` as
#: the separator — e.g. ``src/concerto/safety/`` → ``src.concerto.safety``).
#:
#: Floor values are *line-rate* (decimal in ``[0, 1]``), matching the
#: XML attribute exactly. The numeric values are pinned by the plan §6
#: criteria cited in the ``plan_ref`` field — change only via a plan
#: amendment, not by silently relaxing the numbers here.
#:
#: A future package added to one of the plans gets a new row here +
#: the corresponding plan §6 amendment. A package retired from one of
#: the plans gets the row deleted here in the same PR that retires
#: the directory.
COVERAGE_FLOORS: dict[str, _Floor] = {
    "src.chamber.comm": _Floor(floor=0.90, plan_ref="plan/02 §6 #2"),
    "src.concerto.safety": _Floor(floor=0.90, plan_ref="plan/03 §6 #8"),
    "src.chamber.partners": _Floor(floor=0.90, plan_ref="plan/04 §6 #5"),
    "src.concerto.training": _Floor(floor=0.85, plan_ref="plan/05 §6 #8"),
    "src.chamber.evaluation": _Floor(floor=0.90, plan_ref="plan/06 §6 #6"),
}

#: Exit code on any floor violation. Distinct from argparse's 2 ("bad
#: usage") so a Make target or CI pipeline can tell the failure modes
#: apart.
COVERAGE_FLOOR_VIOLATION_EXIT_CODE: int = 1


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns the process exit code (plan §6 floors).

    Args:
        argv: Optional argv override for tests; ``None`` (default)
            reads from :data:`sys.argv` per argparse's usual contract.

    Returns:
        ``0`` when every package in :data:`COVERAGE_FLOORS` is present
        in the report at-or-above its floor;
        :data:`COVERAGE_FLOOR_VIOLATION_EXIT_CODE` on any miss.
    """
    args = _build_parser().parse_args(argv)
    xml_path: Path = args.coverage_xml
    if not xml_path.exists():
        print(
            f"check-coverage-floors: coverage report not found at {xml_path}. "
            "Run `make test` (or `uv run pytest`) first — the test target "
            "writes coverage.xml as a side-effect.",
            file=sys.stderr,
        )
        return COVERAGE_FLOOR_VIOLATION_EXIT_CODE

    observed = _parse_line_rates(xml_path)
    violations = _check_floors(observed)
    if not violations:
        _print_pass_summary(observed)
        return 0
    _print_violation_report(violations)
    return COVERAGE_FLOOR_VIOLATION_EXIT_CODE


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="check-coverage-floors",
        description=(
            "Enforce per-package line-coverage floors against pytest's "
            "coverage.xml. See COVERAGE_FLOORS in the script for the "
            "canonical table + plan §6 citations."
        ),
    )
    parser.add_argument(
        "--coverage-xml",
        type=Path,
        default=Path("coverage.xml"),
        help=(
            "Path to the coverage.xml emitted by pytest's --cov-report=xml "
            "(default: coverage.xml at the current working directory)."
        ),
    )
    return parser


def _parse_line_rates(xml_path: Path) -> dict[str, float]:
    """Read ``coverage.xml`` → ``{package_name: line_rate}``.

    Skips the aggregate root ``<coverage>`` element and reads only the
    leaf ``<package>`` elements, which carry the per-directory
    line-rate. The Cobertura schema's ``line-rate`` attribute is a
    decimal in ``[0, 1]``.
    """
    tree = ET.parse(xml_path)  # noqa: S314 — coverage.xml is a project artefact.
    return {
        pkg.get("name", ""): float(pkg.get("line-rate", "0"))
        for pkg in tree.getroot().findall(".//package")
        if pkg.get("name")
    }


@dataclass(frozen=True)
class _Violation:
    """A single per-package floor violation (missing OR below-floor)."""

    package: str
    floor: float
    plan_ref: str
    observed: float | None  # ``None`` when the package is missing
    kind: str  # ``"missing"`` or ``"below_floor"``


def _check_floors(observed: dict[str, float]) -> list[_Violation]:
    """Compare ``observed`` against :data:`COVERAGE_FLOORS`."""
    violations: list[_Violation] = []
    for package, spec in COVERAGE_FLOORS.items():
        if package not in observed:
            violations.append(
                _Violation(
                    package=package,
                    floor=spec.floor,
                    plan_ref=spec.plan_ref,
                    observed=None,
                    kind="missing",
                )
            )
            continue
        if observed[package] < spec.floor:
            violations.append(
                _Violation(
                    package=package,
                    floor=spec.floor,
                    plan_ref=spec.plan_ref,
                    observed=observed[package],
                    kind="below_floor",
                )
            )
    return violations


def _print_pass_summary(observed: dict[str, float]) -> None:
    """Emit a one-line-per-floor PASS summary on stdout.

    Called only when :func:`_check_floors` returned an empty list, so
    every key in :data:`COVERAGE_FLOORS` is guaranteed to be present in
    ``observed`` — the indexing below is safe by construction.
    """
    print("check-coverage-floors: PASS")
    for package, spec in COVERAGE_FLOORS.items():
        rate = observed[package]
        print(
            f"  PASS {package:30s} {rate * 100:6.2f}% "
            f"(floor {spec.floor * 100:.0f}%; {spec.plan_ref})"
        )


def _print_violation_report(violations: list[_Violation]) -> None:
    """Emit a multi-line violation report on stderr."""
    print(
        f"check-coverage-floors: FAIL — {len(violations)} "
        f"package{'s' if len(violations) != 1 else ''} below floor or missing",
        file=sys.stderr,
    )
    for v in violations:
        if v.kind == "missing":
            print(
                f"  FAIL {v.package:30s} MISSING (expected ≥{v.floor * 100:.0f}%; {v.plan_ref})",
                file=sys.stderr,
            )
        else:
            assert v.observed is not None  # noqa: S101 — narrow types for type-checker.
            gap = (v.floor - v.observed) * 100
            print(
                f"  FAIL {v.package:30s} {v.observed * 100:6.2f}% "
                f"< floor {v.floor * 100:.0f}% (gap {gap:.2f} pp; {v.plan_ref})",
                file=sys.stderr,
            )


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
