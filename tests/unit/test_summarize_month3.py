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

from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    from chamber.evaluation.results import SubStage


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

    def test_stage_1a_with_positive_gap_still_returns_defer(self) -> None:
        """Belt-and-braces: a Stage-1a row with a spurious positive gap still routes to Defer.

        Stage-1a is rig validation; the ≥20 pp gate is **not** measured
        under it (ADR-007 §Stage 1a). If a future regression were to
        wire Stage-1a episodes through a path that happens to produce
        a non-zero gap (e.g. an env factory that diverges across
        conditions), the recommendation must still be Defer — the gate
        is not load-bearing for Stage-1a. This pin is independent of
        the gap value and complements
        :meth:`test_stage1a_only_returns_defer` (the zero-gap case).
        """
        results = [
            _axis_result("AS", stage="1a", ci_low_pp=30.0, ci_high_pp=40.0, gap_iqm_pp=35.0),
            _axis_result("OM", stage="1a", ci_low_pp=25.0, ci_high_pp=35.0, gap_iqm_pp=28.0),
        ]
        assert decide_recommendation(results) == RECOMMENDATION_DEFER


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
        # Pin the full trigger phrase so a future re-labelling that introduces
        # "ADR-007 §Stage 1b gate trigger fired" doesn't silently match.
        assert "ADR-007 §Stage 1 gate trigger" in report

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
    """Section 8: provenance footer carries chamber version + per-axis archives."""

    def test_provenance_section_mentions_chamber_version(self) -> None:
        import chamber

        report = render_report(_all_pass_fixture())
        assert "Provenance" in report
        assert chamber.__version__ in report

    def test_provenance_lists_per_axis_archives_when_threaded(self) -> None:
        """When the caller threads a ``_ProvenanceFooter`` with archives, they appear."""
        import chamber
        from chamber.cli._spike_summarize_month3 import _ProvenanceFooter

        provenance = _ProvenanceFooter(
            chamber_version=chamber.__version__,
            git_sha="deadbeef",
            timestamp_utc="2026-05-15T00:00:00+00:00",
            gate_pp=20.0,
            n_resamples=2000,
            seed=0,
            axis_archives=[
                ("AS", "spikes/results/stage1-AS-test/spike_as.json", "abc", "tag-AS"),
                ("OM", "spikes/results/stage1-OM-test/spike_om.json", "def", "tag-OM"),
            ],
        )
        report = render_report(_all_pass_fixture(), provenance=provenance)
        assert "deadbeef" in report
        assert "spikes/results/stage1-AS-test/spike_as.json" in report
        assert "tag-OM" in report


class TestBundleCompositionPhrase:
    """ADR-008 §Decision bundle composition (Option A / Option B / hold paths)."""

    def test_default_when_all_three_headline_axes_pass(self) -> None:
        from chamber.cli._spike_summarize_month3 import _bundle_composition_phrase

        phrase = _bundle_composition_phrase({"AS", "OM", "CR", "CM", "PF", "SA"})
        assert "CM x PF x CR" in phrase
        assert "default" in phrase

    def test_option_a_when_cr_failed(self) -> None:
        """CR ∉ passed but CM, PF ∈ passed → Option A."""
        from chamber.cli._spike_summarize_month3 import _bundle_composition_phrase

        phrase = _bundle_composition_phrase({"AS", "OM", "CM", "PF"})
        assert "Option A" in phrase
        assert "partner-familiarity" in phrase

    def test_option_b_when_pf_failed(self) -> None:
        """PF ∉ passed but CM, CR ∈ passed → Option B."""
        from chamber.cli._spike_summarize_month3 import _bundle_composition_phrase

        phrase = _bundle_composition_phrase({"AS", "OM", "CM", "CR"})
        assert "Option B" in phrase
        assert "degraded-partner" in phrase

    def test_hold_when_more_than_one_headline_axis_failed(self) -> None:
        """Neither Option A nor B reachable → hold the bundle lock."""
        from chamber.cli._spike_summarize_month3 import _bundle_composition_phrase

        phrase = _bundle_composition_phrase({"AS", "OM"})
        assert "hold" in phrase.lower()


class TestSummarizeMonth3FileDiscovery:
    """File-discovery contract: read SpikeRun archives only; ignore sibling artefacts.

    Per the module docstring of ``chamber.cli._spike_summarize_month3``:
    the summarizer reads SpikeRun JSONs and not ``leaderboard.json``
    because :class:`LeaderboardEntry` does not carry CI bounds. The
    2026-05-17 maintainer triage caught a regression where the
    discovery glob was the lenient ``*.json``, which made every
    ``leaderboard.json`` produced by ``chamber-eval`` trip a noisy
    13-field :class:`pydantic.ValidationError` against the
    :class:`SpikeRun` schema. The fix constrains the glob to
    ``spike_*.json`` so sibling artefacts are silently skipped.
    """

    def test_summarizer_ignores_leaderboard_json_in_results_dir(
        self,
        tmp_path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A ``leaderboard.json`` in the results dir is silently skipped.

        Per Gap E design: ``summarize-month3`` reads SpikeRun archives
        only. ``leaderboard.json`` belongs to ``chamber-eval``'s
        PARTIAL-row consumer; feeding it to the summarizer triggered
        SpikeRun ValidationError noise (2026-05-17 maintainer triage).
        """
        from chamber.cli.spike import main
        from chamber.evaluation.results import (
            ConditionPair,
            EpisodeResult,
            HRSVector,
            HRSVectorEntry,
            LeaderboardEntry,
            SpikeRun,
        )

        results_dir = tmp_path / "results" / "stage1-AS-test"
        results_dir.mkdir(parents=True)

        # A minimal-but-valid SpikeRun archive. 1 seed, 2 paired
        # episodes (one homo, one hetero) — enough for the pacluster
        # bootstrap to construct pairs and emit a row.
        spike_run = SpikeRun(
            spike_id="stage1_as_test",
            prereg_sha="0" * 40,
            git_tag="prereg-stage1-AS-test",
            axis="AS",
            # ADR-016 §Decision: required field. Test exercises the
            # file-discovery glob, not Stage-1a routing — defaulting
            # to "1b" so the four-state logic runs.
            sub_stage="1b",
            condition_pair=ConditionPair(
                homogeneous_id="homo_id",
                heterogeneous_id="hetero_id",
            ),
            seeds=[0],
            episode_results=[
                EpisodeResult(
                    seed=0,
                    episode_idx=0,
                    initial_state_seed=0,
                    success=True,
                    metadata={"condition": "homo_id"},
                ),
                EpisodeResult(
                    seed=0,
                    episode_idx=0,
                    initial_state_seed=0,
                    success=False,
                    metadata={"condition": "hetero_id"},
                ),
            ],
        )
        (results_dir / "spike_as.json").write_text(spike_run.model_dump_json(), encoding="utf-8")

        # A leaderboard.json shaped per LeaderboardEntry — extra
        # fields (method_id, spike_runs, hrs_vector, hrs_scalar,
        # violation_rate, fallback_rate) and missing every SpikeRun
        # required key. This is the exact shape ``chamber-eval``
        # produces as a sibling artefact alongside SpikeRun archives.
        leaderboard = LeaderboardEntry(
            method_id="test-method",
            spike_runs=["stage1_as_test"],
            hrs_vector=HRSVector(
                entries=[HRSVectorEntry(axis="AS", score=0.5, gap_pp=0.0, weight=1.0)]
            ),
            hrs_scalar=0.5,
            violation_rate=0.0,
            fallback_rate=0.0,
        )
        (results_dir / "leaderboard.json").write_text(
            leaderboard.model_dump_json(), encoding="utf-8"
        )

        rc = main(
            [
                "summarize-month3",
                "--results-dir",
                str(tmp_path / "results"),
            ]
        )
        assert rc == 0
        captured = capsys.readouterr()

        # The summarizer rendered a report and the planted spike_as.json
        # was loaded (not the Missing-row fallback the renderer emits
        # for an axis with no archive). The "Stage 1b (measured)"
        # status cell is the load-bearing discriminator: it only
        # renders for a loaded SpikeRun whose stage resolves to "1b"
        # (the default for AS via ``_DEFAULT_STAGE_BY_AXIS``). A weaker
        # ``"| AS |"`` assertion would silently pass even if the glob
        # picked zero files, because the renderer emits a Missing row
        # per axis in ``_AXIS_ORDER``.
        assert "Stage 1b (measured)" in captured.out, (
            "summarize-month3 did not render the AS row from the planted "
            f"spike_as.json archive (Stage 1b status cell absent). stdout was: {captured.out!r}"
        )

        # The summarizer did NOT emit a parser-error stderr note for
        # the leaderboard.json. The file is a sibling artefact, not a
        # malformed SpikeRun, so the noisy "skipping … not a valid
        # SpikeRun" stderr line is itself the regression we want to
        # avoid (it would re-appear if the file-discovery glob fell
        # back to ``*.json``).
        assert "leaderboard.json" not in captured.err, (
            "summarize-month3 surfaced leaderboard.json as a parser "
            "error; the file-discovery glob should silently skip "
            "non-SpikeRun sibling artefacts. stderr was: "
            f"{captured.err!r}"
        )

        # Pin the underlying invariant directly: confirm a
        # leaderboard.json is parseable as a LeaderboardEntry but
        # NOT as a SpikeRun (i.e. the file-discovery glob is the
        # only thing standing between us and a 13-field
        # ValidationError). If this assertion fails, the
        # LeaderboardEntry / SpikeRun schemas have converged and the
        # glob constraint may be redundant — re-check.
        with pytest.raises(ValueError, match="SpikeRun"):
            SpikeRun.model_validate_json(
                (results_dir / "leaderboard.json").read_text(encoding="utf-8")
            )
        LeaderboardEntry.model_validate_json(
            (results_dir / "leaderboard.json").read_text(encoding="utf-8")
        )

        # Sanity: the planted spike_as.json round-trips as a SpikeRun.
        SpikeRun.model_validate_json((results_dir / "spike_as.json").read_text(encoding="utf-8"))


class TestSummarizeMonth3SubStageRouting:
    """End-to-end Stage-1a routing via the SpikeRun.sub_stage wire-format field.

    PR 2 of the 2026-05-17 triage replaces the previous (broken)
    ``EpisodeResult.metadata["stage"]`` affordance with a typed
    ``SpikeRun.sub_stage`` field. The summarizer reads this field
    directly via :func:`_stage_from_spike_run`. The end-to-end pins
    below confirm that the summarizer routes correctly off the wire-
    format field across the CLI surface.

    Distinct from :class:`TestDecideRecommendation` which exercises
    the decision rule at the :class:`_AxisResult` level (one struct-
    construction step inside the summarizer pipeline): these tests
    drive the full ``chamber-spike summarize-month3`` CLI path from
    a planted SpikeRun JSON on disk.
    """

    def _plant_spike_run(
        self,
        *,
        results_dir,
        axis: str,
        sub_stage: SubStage,
        n_seeds: int = 1,
        n_episodes_per_seed: int = 2,
        homo_success: bool = True,
        hetero_success: bool = False,
    ):
        """Plant a minimal SpikeRun JSON archive on disk for the summarizer to load.

        ``homo_success`` and ``hetero_success`` parametrise the paired
        gap so the test can synthesise both a zero-gap and a wide-gap
        archive without re-running a real spike.
        """
        from chamber.evaluation.results import (
            ConditionPair,
            EpisodeResult,
            SpikeRun,
        )

        homo_id = f"{axis}_homo"
        hetero_id = f"{axis}_hetero"
        episode_results = []
        for seed in range(n_seeds):
            for episode_idx in range(n_episodes_per_seed):
                episode_results.append(
                    EpisodeResult(
                        seed=seed,
                        episode_idx=episode_idx,
                        initial_state_seed=seed * 1000 + episode_idx,
                        success=homo_success,
                        metadata={"condition": homo_id},
                    )
                )
                episode_results.append(
                    EpisodeResult(
                        seed=seed,
                        episode_idx=episode_idx,
                        initial_state_seed=seed * 1000 + episode_idx,
                        success=hetero_success,
                        metadata={"condition": hetero_id},
                    )
                )
        spike_run = SpikeRun(
            spike_id=f"stage_{sub_stage}_{axis.lower()}_test",
            prereg_sha="0" * 40,
            git_tag=f"prereg-stage{sub_stage}-{axis}-test",
            axis=axis,
            sub_stage=sub_stage,
            condition_pair=ConditionPair(homogeneous_id=homo_id, heterogeneous_id=hetero_id),
            seeds=list(range(n_seeds)),
            episode_results=episode_results,
        )
        archive_dir = results_dir / f"stage{sub_stage}-{axis}-test"
        archive_dir.mkdir(parents=True)
        (archive_dir / f"spike_{axis.lower()}.json").write_text(
            spike_run.model_dump_json(), encoding="utf-8"
        )

    def test_stage_1a_archive_routes_to_defer_regardless_of_gap(
        self,
        tmp_path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Stage-1a archive on disk → ``Defer`` even when episode success rates diverge.

        Plants an AS + OM SpikeRun pair with ``sub_stage="1a"`` and a
        50 pp gap (homo passes; hetero fails). Under ADR-007 §Stage 1a
        the ≥20 pp gate is not measured, so the summarizer MUST route
        to ``Defer — Stage 1b not yet measured`` regardless of the
        synthesised gap (ADR-016 §Decision; PR 2 of the 2026-05-17
        triage). This is the load-bearing end-to-end pin that the
        rendering text matches PR-3's regenerated archives.
        """
        from chamber.cli.spike import main

        results_dir = tmp_path / "results"
        results_dir.mkdir()
        self._plant_spike_run(
            results_dir=results_dir,
            axis="AS",
            sub_stage="1a",
            homo_success=True,
            hetero_success=False,
        )
        self._plant_spike_run(
            results_dir=results_dir,
            axis="OM",
            sub_stage="1a",
            homo_success=True,
            hetero_success=False,
        )
        rc = main(["summarize-month3", "--results-dir", str(results_dir)])
        assert rc == 0
        captured = capsys.readouterr()
        assert "**Recommendation: Defer" in captured.out, (
            f"summarize-month3 did not route to Defer for Stage-1a archives. "
            f"stdout:\n{captured.out}"
        )
        # The renderer marks Stage-1a rows as ``n/a`` for the gate column.
        assert "n/a" in captured.out

    def test_stage_1b_archive_with_zero_gap_routes_to_stop(
        self,
        tmp_path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Stage-1b archive with no separation → ``Stop`` (the four-state logic).

        Plants an AS + OM SpikeRun pair with ``sub_stage="1b"`` and a
        zero gap (both conditions identical-success). Under the four-
        state logic the summarizer routes to ``Stop`` — confirms the
        sub_stage="1b" path still flows through the existing decision
        rule unchanged.
        """
        from chamber.cli.spike import main

        results_dir = tmp_path / "results"
        results_dir.mkdir()
        # Zero gap: both conditions succeed every episode.
        self._plant_spike_run(
            results_dir=results_dir,
            axis="AS",
            sub_stage="1b",
            homo_success=True,
            hetero_success=True,
        )
        self._plant_spike_run(
            results_dir=results_dir,
            axis="OM",
            sub_stage="1b",
            homo_success=True,
            hetero_success=True,
        )
        rc = main(["summarize-month3", "--results-dir", str(results_dir)])
        assert rc == 0
        captured = capsys.readouterr()
        assert "**Recommendation: Stop" in captured.out, (
            f"summarize-month3 did not route to Stop for zero-gap Stage-1b "
            f"archives. stdout:\n{captured.out}"
        )


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

    def test_output_flag_writes_report_to_disk(
        self,
        tmp_path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """``--output <path>`` writes the report; stderr carries the "wrote" status."""
        from chamber.cli.spike import main

        results_dir = tmp_path / "results"
        results_dir.mkdir()
        output_path = tmp_path / "report.md"
        rc = main(
            [
                "summarize-month3",
                "--results-dir",
                str(results_dir),
                "--output",
                str(output_path),
            ]
        )
        assert rc == 0
        assert output_path.exists()
        body = output_path.read_text(encoding="utf-8")
        assert "**Recommendation: Defer" in body
        captured = capsys.readouterr()
        assert "wrote" in captured.err
        assert str(output_path) in captured.err
