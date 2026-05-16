# SPDX-License-Identifier: Apache-2.0
"""``scripts/check_coverage_floors.py`` unit tests (plan/02 §6 #2; plan/03 §6 #8;
plan/04 §6 #5; plan/05 §6 #8; plan/06 §6 #6).

Pins the script's contract against three synthetic ``coverage.xml``
fixtures:

- All five named packages meet their floor → exit 0.
- One named package drops 1 pp below its floor → exit non-zero with
  the offending package and the gap named on stderr.
- One named package is missing from the report → exit non-zero with
  the missing package named loudly (missing = regression in test
  coverage *scope*, not "no measurement").

The script is loaded by file path (``scripts/`` is not a Python
package; loading via :func:`importlib.util.spec_from_file_location`
keeps it self-contained without contaminating the import layout).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "check_coverage_floors.py"


@pytest.fixture(scope="module")
def script_module() -> ModuleType:
    """Load ``scripts/check_coverage_floors.py`` as a private module."""
    spec = importlib.util.spec_from_file_location("_check_coverage_floors_under_test", _SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_coverage_xml(path: Path, line_rates: dict[str, float]) -> None:
    """Emit a minimal-but-valid coverage.xml fixture (Cobertura schema)."""
    body = ["<?xml version='1.0' ?>"]
    body.append(
        '<coverage version="7.13.5" lines-valid="100" lines-covered="80" '
        'line-rate="0.80" branches-valid="0" branches-covered="0" '
        'branch-rate="0" complexity="0">'
    )
    body.append("  <packages>")
    for name, rate in line_rates.items():
        body.append(
            f'    <package name="{name}" line-rate="{rate:.4f}" branch-rate="0" complexity="0">'
        )
        body.append("      <classes/>")
        body.append("    </package>")
    body.append("  </packages>")
    body.append("</coverage>")
    path.write_text("\n".join(body), encoding="utf-8")


_PASSING_FIXTURE: dict[str, float] = {
    "src.concerto.safety": 0.95,
    "src.concerto.training": 0.95,
    "src.chamber.comm": 0.95,
    "src.chamber.partners": 0.95,
    "src.chamber.evaluation": 0.95,
}


class TestCoverageFloors:
    """``scripts/check_coverage_floors.py`` contract (plan/0[2-6] §6)."""

    def test_all_packages_meet_floor_returns_zero(
        self,
        script_module: ModuleType,
        tmp_path: Path,
    ) -> None:
        """All five floors met → exit 0."""
        xml_path = tmp_path / "coverage.xml"
        _write_coverage_xml(xml_path, _PASSING_FIXTURE)
        rc = script_module.main(["--coverage-xml", str(xml_path)])
        assert rc == 0, f"expected exit 0, got {rc}"

    def test_one_package_drops_below_floor_is_named_loudly(
        self,
        script_module: ModuleType,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A 1 pp drop on one package → exit non-zero; package + gap on stderr."""
        rates = dict(_PASSING_FIXTURE)
        # concerto.safety floor is 90% (plan/03 §6 #8); 89% is 1 pp below.
        rates["src.concerto.safety"] = 0.89
        xml_path = tmp_path / "coverage.xml"
        _write_coverage_xml(xml_path, rates)
        rc = script_module.main(["--coverage-xml", str(xml_path)])
        assert rc != 0, f"expected non-zero exit, got {rc}"
        captured = capsys.readouterr()
        # Offending package + the floor + the observed value all named on stderr.
        assert "src.concerto.safety" in captured.err
        assert "89" in captured.err
        assert "90" in captured.err

    def test_missing_package_is_named_loudly(
        self,
        script_module: ModuleType,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A floor-targeted package missing from the report → exit non-zero.

        Missing means the package contributed no measurement to the run,
        which is a regression in test coverage *scope* — the worst kind
        of silent drift. The script must surface it loudly, not treat it
        as "no measurement, vacuous pass".
        """
        rates = dict(_PASSING_FIXTURE)
        del rates["src.chamber.evaluation"]
        xml_path = tmp_path / "coverage.xml"
        _write_coverage_xml(xml_path, rates)
        rc = script_module.main(["--coverage-xml", str(xml_path)])
        assert rc != 0, f"expected non-zero exit, got {rc}"
        captured = capsys.readouterr()
        assert "src.chamber.evaluation" in captured.err
        # "missing" / "absent" — exact wording is in the script; the test
        # only asserts the keyword is present (case-insensitive).
        assert "missing" in captured.err.lower() or "absent" in captured.err.lower()
