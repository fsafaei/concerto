# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``chamber-spike summarize-month3`` (plan/07 §6 #7; §T5b.10).

Exercises the Month-3 lock-priority report against four synthetic
fixtures, one per top-line recommendation state:

- All 6 axes pass their ≥20 pp gap → **Accept-Validated**.
- AS + OM pass, CR + CM + PF + SA fail (or absent) → **Accept-Partial-Defer**;
  Stage-2 gate trigger surfaced.
- All Stage 1b axes measured + failing → **Stop**; Stage-1 gate trigger surfaced.
- Stage 1 measured Stage 1a only (no Stage 1b yet) → **Defer**;
  Phase-1 4-week guardrail surfaced.

Test surface is at the ``_AxisResult`` decision-logic level. The
underlying pacluster bootstrap is exercised elsewhere
(``tests/unit/test_bootstrap.py``); this file pins the report's
contract independently so a renderer regression trips CI without
re-running the bootstrap.
"""

from __future__ import annotations

import pytest

from chamber.cli._spike_summarize_month3 import (
    RECOMMENDATION_ACCEPT_PARTIAL_DEFER,
    RECOMMENDATION_ACCEPT_VALIDATED,
    RECOMMENDATION_DEFER,
    RECOMMENDATION_STOP,
    _AxisResult,
    decide_recommendation,
    render_report,
)


def _axis_result(
    axis: str,
    *,
    stage: str,
    ci_low_pp: float,
    ci_high_pp: float | None = None,
    gap_iqm_pp: float | None = None,
    n_seeds: int = 5,
    n_episodes: int = 100,
) -> _AxisResult:
    """Build a minimal ``_AxisResult`` for a decision-rule fixture."""
    return _AxisResult(
        axis=axis,
        stage=stage,
        n_seeds=n_seeds,
        n_episodes=n_episodes,
        gap_iqm_pp=gap_iqm_pp if gap_iqm_pp is not None else ci_low_pp + 5.0,
        ci_low_pp=ci_low_pp,
        ci_high_pp=ci_high_pp if ci_high_pp is not None else ci_low_pp + 10.0,
    )


def _all_pass_fixture() -> list[_AxisResult]:
    """All 6 axes measured at Stage 1b/2/3, every ci_low_pp ≥ 20."""
    return [
        _axis_result("AS", stage="1b", ci_low_pp=35.0),
        _axis_result("OM", stage="1b", ci_low_pp=32.0),
        _axis_result("CR", stage="2", ci_low_pp=28.0),
        _axis_result("CM", stage="2", ci_low_pp=40.0),
        _axis_result("PF", stage="3", ci_low_pp=25.0),
        _axis_result("SA", stage="3", ci_low_pp=22.0),
    ]


def _stage1_cleared_others_fail_fixture() -> list[_AxisResult]:
    """AS + OM pass at Stage 1b; CR + CM + PF + SA all measured + failing."""
    return [
        _axis_result("AS", stage="1b", ci_low_pp=30.0),
        _axis_result("OM", stage="1b", ci_low_pp=25.0),
        _axis_result("CR", stage="2", ci_low_pp=8.0),
        _axis_result("CM", stage="2", ci_low_pp=12.0),
        _axis_result("PF", stage="3", ci_low_pp=15.0),
        _axis_result("SA", stage="3", ci_low_pp=5.0),
    ]


def _all_fail_fixture() -> list[_AxisResult]:
    """All 6 axes measured at Stage 1b/2/3, every ci_low_pp < 20."""
    return [
        _axis_result("AS", stage="1b", ci_low_pp=10.0),
        _axis_result("OM", stage="1b", ci_low_pp=8.0),
        _axis_result("CR", stage="2", ci_low_pp=5.0),
        _axis_result("CM", stage="2", ci_low_pp=12.0),
        _axis_result("PF", stage="3", ci_low_pp=15.0),
        _axis_result("SA", stage="3", ci_low_pp=3.0),
    ]


def _stage1a_only_fixture() -> list[_AxisResult]:
    """AS + OM run at Stage 1a only (rig validation, no ≥20 pp measurement)."""
    return [
        _axis_result("AS", stage="1a", ci_low_pp=0.0, gap_iqm_pp=0.0, ci_high_pp=0.0),
        _axis_result("OM", stage="1a", ci_low_pp=0.0, gap_iqm_pp=0.0, ci_high_pp=0.0),
    ]


class TestDecideRecommendation:
    """Decision rule (ADR-007 §Implementation staging + ADR-008 §Decision)."""

    def test_all_axes_pass_returns_accept_validated(self) -> None:
        assert decide_recommendation(_all_pass_fixture()) == RECOMMENDATION_ACCEPT_VALIDATED

    def test_stage1_cleared_others_fail_returns_accept_partial_defer(self) -> None:
        assert (
            decide_recommendation(_stage1_cleared_others_fail_fixture())
            == RECOMMENDATION_ACCEPT_PARTIAL_DEFER
        )

    def test_all_axes_fail_returns_stop(self) -> None:
        assert decide_recommendation(_all_fail_fixture()) == RECOMMENDATION_STOP

    def test_stage1a_only_returns_defer(self) -> None:
        assert decide_recommendation(_stage1a_only_fixture()) == RECOMMENDATION_DEFER


class TestRenderReportAcceptValidated:
    """All-pass fixture → Markdown report with Accept-Validated banner + locked ADRs."""

    @pytest.fixture
    def report(self) -> str:
        return render_report(_all_pass_fixture())

    def test_top_line_recommendation_is_accept_validated(self, report: str) -> None:
        assert "**Recommendation: Accept-Validated**" in report

    def test_per_axis_table_lists_all_six_axes_in_staging_order(self, report: str) -> None:
        idx = {axis: report.find(f"| {axis} |") for axis in ("AS", "OM", "CR", "CM", "PF", "SA")}
        # All six axes are present in the per-axis evidence table.
        for axis, position in idx.items():
            assert position > 0, f"axis {axis!r} missing from per-axis evidence table"
        # ADR-007 staging order: AS, OM, CR, CM, PF, SA.
        positions = [idx["AS"], idx["OM"], idx["CR"], idx["CM"], idx["PF"], idx["SA"]]
        assert positions == sorted(positions), f"axes not in ADR-007 staging order: {positions}"

    def test_adr_action_table_locks_adr_008_default_bundle(self, report: str) -> None:
        assert "CM x PF x CR" in report

    def test_adr_action_table_locks_adr_011_baseline_set(self, report: str) -> None:
        # ADR-011 action row names the locked baseline set.
        assert "ADR-011" in report
        assert "baseline" in report.lower()

    def test_stage_gate_section_says_proceed_to_next_stage(self, report: str) -> None:
        # Stage 1b cleared → Stage 2 PERMITTED; Stage 2 cleared → Stage 3 PERMITTED.
        assert "Stage 2 launch: PERMITTED" in report
        assert "Stage 3 launch: PERMITTED" in report


class TestRenderReportAcceptPartialDefer:
    """Stage 1 cleared, others fail → Accept-Partial-Defer + Stage-2 trigger."""

    @pytest.fixture
    def report(self) -> str:
        return render_report(_stage1_cleared_others_fail_fixture())

    def test_top_line_recommendation_is_accept_partial_defer(self, report: str) -> None:
        assert "**Recommendation: Accept-Partial-Defer**" in report

    def test_report_surfaces_stage_2_gate_trigger(self, report: str) -> None:
        """CR + CM failed at Stage 2 ⇒ Stage 3 launch BLOCKED + Stage-2 trigger named."""
        # Stage 2 was measured and both axes failed; Stage 3 launch is therefore
        # blocked, and the §Stage 2 gate trigger fires.
        assert "Stage 3 launch: BLOCKED" in report
        assert "ADR-007 §Stage 2 gate" in report

    def test_adr_008_bundle_blocked_or_falls_back(self, report: str) -> None:
        """Headline CM x PF x CR not composable → bundle held or falls back."""
        adr_008_lines = [line for line in report.splitlines() if "ADR-008" in line and "|" in line]
        assert adr_008_lines, "ADR-008 row missing from ADR-by-ADR action table"
        adr_008 = adr_008_lines[0]
        # Action cell says either "Block bundle lock" or names a fallback.
        assert "Block" in adr_008 or "Option" in adr_008 or "Hold" in adr_008


class TestRenderReportStop:
    """All Stage 1b axes measured + failing → Stop, name Stage-1 gate trigger."""

    @pytest.fixture
    def report(self) -> str:
        return render_report(_all_fail_fixture())

    def test_top_line_recommendation_is_stop(self, report: str) -> None:
        assert "**Recommendation: Stop — ADR re-review**" in report

    def test_report_surfaces_stage_1_gate_trigger(self, report: str) -> None:
        assert "ADR-007 §Stage 1 gate" in report

    def test_adr_007_action_says_re_review(self, report: str) -> None:
        adr_007_lines = [line for line in report.splitlines() if "ADR-007" in line and "|" in line]
        assert adr_007_lines, "ADR-007 row missing from ADR-by-ADR action table"
        # Action cell names re-review of the §Decision.
        assert any("Re-review" in line or "re-review" in line for line in adr_007_lines), (
            f"ADR-007 action did not name re-review; got: {adr_007_lines!r}"
        )


class TestRenderReportDefer:
    """Stage 1a only (no Stage 1b yet) → Defer, name 4-week guardrail."""

    @pytest.fixture
    def report(self) -> str:
        return render_report(_stage1a_only_fixture())

    def test_top_line_recommendation_is_defer(self, report: str) -> None:
        assert "**Recommendation: Defer" in report
        assert "Stage 1b" in report

    def test_report_surfaces_4_week_guardrail(self, report: str) -> None:
        # ADR-007 §Stage 1b guardrail: ≤4 weeks after Month-3 lock review.
        assert "4 week" in report or "four week" in report.lower()

    def test_stage_1a_rows_render_as_n_a_gate(self, report: str) -> None:
        # Stage 1a rows have ``gate_pass = n/a`` (rig validation only).
        as_row_lines = [line for line in report.splitlines() if line.lstrip().startswith("| AS |")]
        assert as_row_lines, "AS row missing from per-axis evidence table"
        assert "n/a" in as_row_lines[0]


class TestReportProvenance:
    """Section 8: provenance footer carries chamber version + axis count."""

    def test_provenance_section_mentions_chamber_version(self) -> None:
        import chamber

        report = render_report(_all_pass_fixture())
        assert "Provenance" in report
        assert chamber.__version__ in report


class TestSummarizeMonth3CLI:
    """``chamber-spike summarize-month3`` end-to-end (CLI dispatch + empty-dir path)."""

    def test_help_resolves(self, capsys: pytest.CaptureFixture[str]) -> None:
        """``chamber-spike summarize-month3 --help`` prints argparse usage + flags."""
        from chamber.cli.spike import main

        with pytest.raises(SystemExit) as excinfo:
            main(["summarize-month3", "--help"])
        assert excinfo.value.code == 0
        captured = capsys.readouterr()
        assert "summarize-month3" in captured.out
        # The four custom flags appear in --help output.
        for flag in ("--results-dir", "--output", "--gate-pp", "--n-resamples"):
            assert flag in captured.out, f"--help missing flag {flag!r}"

    def test_empty_results_dir_renders_defer_report(
        self,
        tmp_path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """No SpikeRun JSONs found → report renders ``Defer`` (Stage 1b absent)."""
        from chamber.cli.spike import main

        results_dir = tmp_path / "results"
        results_dir.mkdir()
        rc = main(["summarize-month3", "--results-dir", str(results_dir)])
        assert rc == 0
        captured = capsys.readouterr()
        assert "**Recommendation: Defer" in captured.out
        assert "Stage 1b" in captured.out

    def test_missing_results_dir_exits_nonzero(
        self,
        tmp_path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A ``--results-dir`` that does not exist exits non-zero with stderr message."""
        from chamber.cli.spike import main

        missing = tmp_path / "nope"
        rc = main(["summarize-month3", "--results-dir", str(missing)])
        assert rc != 0
        captured = capsys.readouterr()
        assert "not found" in captured.err
