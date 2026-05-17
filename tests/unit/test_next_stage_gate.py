# SPDX-License-Identifier: Apache-2.0
"""``chamber-spike next-stage`` gate tests (T5b.1; plan/07 §5; ADR-007 §Implementation staging).

Each test stages a synthetic SpikeRun (via the B7 dry-run path) and
then invokes ``chamber-spike next-stage`` against it. The dry-run's
deterministic episode layout makes the per-axis gap CI lower bound
predictable: with the default 10/20 hetero success rate, IQM gap is
~50pp and the gate passes; with 18/20 hetero success the gap is 0
and the gate fails.

Stage groupings (ADR-007 §Implementation staging):

- Stage 1: AS + OM
- Stage 2: CR + CM
- Stage 3: PF + SA
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from chamber.cli._spike_next_stage import NEXT_STAGE_GATE_EXIT_CODE
from chamber.cli.spike import main
from chamber.evaluation.results import (
    ConditionPair,
    EpisodeResult,
    SpikeRun,
    SubStage,
)

if TYPE_CHECKING:
    from pathlib import Path


def _emit_dry_run(
    *,
    tmp_path: Path,
    axis: str,
    hetero_success_count: int = 10,
    filename: str | None = None,
) -> Path:
    """Stage a dry-run SpikeRun JSON via ``chamber-spike run`` and return its path."""
    out = tmp_path / (filename or f"spike_{axis.lower()}.json")
    rc = main(
        [
            "run",
            "--axis",
            axis,
            "--dry-run",
            "--dry-run-hetero-success-count",
            str(hetero_success_count),
            "--output",
            str(out),
        ]
    )
    assert rc == 0, f"failed to stage dry-run for axis {axis}"
    return out


# ---------------------------------------------------------------------------
# Happy path: at least one prior-stage axis passes
# ---------------------------------------------------------------------------


class TestGatePasses:
    """plan/07 §5: gate passes when >=1 prior-stage axis has ci_low_pp >= 20."""

    def test_stage1_single_axis_passes(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        spike = _emit_dry_run(tmp_path=tmp_path, axis="AS", hetero_success_count=10)
        rc = main(
            [
                "next-stage",
                "--prior-stage",
                "1",
                "--spike-runs",
                str(spike),
            ]
        )
        assert rc == 0
        captured = capsys.readouterr()
        assert "PASS" in captured.out
        assert "axis=AS" in captured.out

    def test_stage1_both_axes_passing(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        as_spike = _emit_dry_run(tmp_path=tmp_path, axis="AS")
        om_spike = _emit_dry_run(tmp_path=tmp_path, axis="OM")
        rc = main(
            [
                "next-stage",
                "--prior-stage",
                "1",
                "--spike-runs",
                str(as_spike),
                str(om_spike),
            ]
        )
        assert rc == 0
        captured = capsys.readouterr()
        assert "2/2 axis(es) pass" in captured.out

    def test_one_pass_and_one_fail_still_clears_gate(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """ADR-007 §Implementation staging: at least one axis clearing >=20pp clears the stage."""
        as_pass = _emit_dry_run(tmp_path=tmp_path, axis="AS", hetero_success_count=10)
        om_fail = _emit_dry_run(tmp_path=tmp_path, axis="OM", hetero_success_count=18)
        rc = main(
            [
                "next-stage",
                "--prior-stage",
                "1",
                "--spike-runs",
                str(as_pass),
                str(om_fail),
            ]
        )
        assert rc == 0
        captured = capsys.readouterr()
        assert "PASS" in captured.out
        assert "1/2 axis(es) pass" in captured.out


# ---------------------------------------------------------------------------
# Failure path: no prior-stage axis passes
# ---------------------------------------------------------------------------


class TestGateFails:
    """plan/07 §5: gate fails when no prior-stage axis meets the threshold."""

    def test_stage1_both_axes_failing(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        as_fail = _emit_dry_run(tmp_path=tmp_path, axis="AS", hetero_success_count=18)
        om_fail = _emit_dry_run(tmp_path=tmp_path, axis="OM", hetero_success_count=18)
        rc = main(
            [
                "next-stage",
                "--prior-stage",
                "1",
                "--spike-runs",
                str(as_fail),
                str(om_fail),
            ]
        )
        assert rc == NEXT_STAGE_GATE_EXIT_CODE
        captured = capsys.readouterr()
        assert "FAIL" in captured.err
        assert "0/2 axis(es) pass" in captured.err

    def test_single_failing_axis(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        spike = _emit_dry_run(tmp_path=tmp_path, axis="AS", hetero_success_count=20)
        rc = main(
            [
                "next-stage",
                "--prior-stage",
                "1",
                "--spike-runs",
                str(spike),
            ]
        )
        assert rc == NEXT_STAGE_GATE_EXIT_CODE


# ---------------------------------------------------------------------------
# Wrong-stage / missing inputs
# ---------------------------------------------------------------------------


class TestInputValidation:
    """Failure modes for malformed or wrong-stage inputs."""

    def test_no_prior_stage_axis_in_runs(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A Stage-2 SpikeRun handed to --prior-stage 1 trips the gate."""
        cr_spike = _emit_dry_run(tmp_path=tmp_path, axis="CR", hetero_success_count=10)
        rc = main(
            [
                "next-stage",
                "--prior-stage",
                "1",
                "--spike-runs",
                str(cr_spike),
            ]
        )
        assert rc == NEXT_STAGE_GATE_EXIT_CODE
        captured = capsys.readouterr()
        assert "no SpikeRun on the prior stage" in captured.err

    def test_missing_spike_run_file(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        missing = tmp_path / "does_not_exist.json"
        rc = main(
            [
                "next-stage",
                "--prior-stage",
                "1",
                "--spike-runs",
                str(missing),
            ]
        )
        assert rc == NEXT_STAGE_GATE_EXIT_CODE
        captured = capsys.readouterr()
        assert "not found" in captured.err

    def test_bad_prior_stage_choice_errors(self) -> None:
        with pytest.raises(SystemExit) as excinfo:
            main(["next-stage", "--prior-stage", "4", "--spike-runs", "x.json"])
        assert excinfo.value.code == 2  # argparse "bad usage"


# ---------------------------------------------------------------------------
# Determinism (ADR-002 P6) + tunable threshold
# ---------------------------------------------------------------------------


class TestDeterminismAndThreshold:
    """Bootstrap RNG is seeded; --gate-pp is tunable."""

    def test_two_invocations_same_seed_same_verdict(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        spike = _emit_dry_run(tmp_path=tmp_path, axis="AS")
        # Drop the dry-run staging output so the captures below contain
        # only the next-stage CLI lines.
        capsys.readouterr()
        first = main(
            [
                "next-stage",
                "--prior-stage",
                "1",
                "--spike-runs",
                str(spike),
                "--seed",
                "7",
            ]
        )
        out_first = capsys.readouterr().out
        second = main(
            [
                "next-stage",
                "--prior-stage",
                "1",
                "--spike-runs",
                str(spike),
                "--seed",
                "7",
            ]
        )
        out_second = capsys.readouterr().out
        assert first == second == 0
        # ci_low_pp column should appear and be byte-identical across runs.
        assert "ci_low_pp" in out_first
        assert out_first == out_second

    def test_tighter_gate_can_fail_a_pass(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A high --gate-pp threshold can override a default-pass dry-run."""
        spike = _emit_dry_run(tmp_path=tmp_path, axis="AS", hetero_success_count=10)
        # Default (gate-pp=20) → passes; gate-pp=80 should fail (gap is ~50pp).
        rc = main(
            [
                "next-stage",
                "--prior-stage",
                "1",
                "--spike-runs",
                str(spike),
                "--gate-pp",
                "80.0",
            ]
        )
        assert rc == NEXT_STAGE_GATE_EXIT_CODE


# ---------------------------------------------------------------------------
# Sub-stage filter (P1.01; ADR-016 §Open questions; ADR-007 §Stage 1a)
# ---------------------------------------------------------------------------


def _plant_spike_run(
    *,
    tmp_path: Path,
    axis: str,
    sub_stage: SubStage,
    spike_id: str | None = None,
    hetero_success_count: int = 10,
    filename: str | None = None,
) -> Path:
    """Plant a synthetic SpikeRun JSON with a chosen ``sub_stage``.

    Mirrors the dry-run path in ``chamber.cli._spike_run`` (5 seeds x
    20 episodes per (seed, condition); homo always succeeds, hetero
    succeeds for the first ``hetero_success_count`` of each seed) so
    the paired-cluster bootstrap sees the same gap pattern as
    ``_emit_dry_run`` — but lets the caller pick ``sub_stage``
    directly. The dry-run helper hard-codes ``"1b"`` for AS / OM (see
    ``_DRY_RUN_SUB_STAGE_BY_AXIS``), so the four ``TestSubStageFilter``
    tests need a separate constructor to span ``"1a"`` / ``"1b"`` /
    ``"2"``.
    """
    condition_pair = ConditionPair(
        homogeneous_id=f"{axis}_homo",
        heterogeneous_id=f"{axis}_hetero",
    )
    episode_results: list[EpisodeResult] = []
    for seed in (0, 1, 2, 3, 4):
        for episode_idx in range(20):
            initial_state_seed = seed * 10_000 + episode_idx
            episode_results.append(
                EpisodeResult(
                    seed=seed,
                    episode_idx=episode_idx,
                    initial_state_seed=initial_state_seed,
                    success=True,
                    metadata={"condition": condition_pair.homogeneous_id},
                )
            )
            episode_results.append(
                EpisodeResult(
                    seed=seed,
                    episode_idx=episode_idx,
                    initial_state_seed=initial_state_seed,
                    success=episode_idx < hetero_success_count,
                    metadata={"condition": condition_pair.heterogeneous_id},
                )
            )
    spike_run = SpikeRun(
        spike_id=spike_id or f"test_stage{sub_stage}_{axis.lower()}",
        prereg_sha="0" * 40,
        git_tag=f"prereg-stage{sub_stage}-{axis}-test",
        axis=axis,
        sub_stage=sub_stage,
        condition_pair=condition_pair,
        seeds=[0, 1, 2, 3, 4],
        episode_results=episode_results,
    )
    path = tmp_path / (filename or f"spike_{axis.lower()}.json")
    path.write_text(spike_run.model_dump_json(), encoding="utf-8")
    return path


class TestSubStageFilter:
    """P1.01: ``next-stage`` skips Stage-1a archives (ADR-016 §Open questions).

    ADR-007 §Stage 1a — Stage 1a is rig validation; the ≥20 pp gate is
    NOT measured under 1a (the adapter ships with the always-zero ego
    by design). Including a 1a archive in the gate would bootstrap a
    synthetic gap against rig-validation data and could route
    Phase-1-launch decisions off it. PR #152 closed the same class of
    mis-routing for ``summarize-month3``; these tests pin the parallel
    fix for ``next-stage``.
    """

    def test_all_stage_1a_input_fails_with_specific_message(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Every Stage-1 archive is 1a → exit code 5 with all-1a-specific FAIL line."""
        as_1a = _plant_spike_run(
            tmp_path=tmp_path,
            axis="AS",
            sub_stage="1a",
            spike_id="stage1_as_test_1a",
            filename="spike_as.json",
        )
        om_1a = _plant_spike_run(
            tmp_path=tmp_path,
            axis="OM",
            sub_stage="1a",
            spike_id="stage1_om_test_1a",
            filename="spike_om.json",
        )
        rc = main(
            [
                "next-stage",
                "--prior-stage",
                "1",
                "--spike-runs",
                str(as_1a),
                str(om_1a),
            ]
        )
        assert rc == NEXT_STAGE_GATE_EXIT_CODE
        err = capsys.readouterr().err
        assert "every Stage-1 SpikeRun in --spike-runs is Stage-1a" in err, err
        assert "ADR-007 §Stage 1a" in err, err
        assert "sub_stage='1b'" in err, err

    def test_mixed_1a_and_1b_input_evaluates_on_1b_only(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Mixed 1a + 1b → gate runs on 1b only; 1a is skipped non-fatally."""
        as_1a = _plant_spike_run(
            tmp_path=tmp_path,
            axis="AS",
            sub_stage="1a",
            spike_id="stage1_as_test_1a",
            hetero_success_count=20,  # zero gap — would fail if evaluated
            filename="spike_as.json",
        )
        om_1b = _plant_spike_run(
            tmp_path=tmp_path,
            axis="OM",
            sub_stage="1b",
            spike_id="stage1_om_test_1b",
            hetero_success_count=10,  # ~50 pp gap — clears the 20 pp gate
            filename="spike_om.json",
        )
        rc = main(
            [
                "next-stage",
                "--prior-stage",
                "1",
                "--spike-runs",
                str(as_1a),
                str(om_1b),
            ]
        )
        captured = capsys.readouterr()
        assert rc == 0, f"gate should pass on OM 1b; stderr={captured.err}"
        assert "skipping 1 Stage-1a SpikeRun" in captured.err, captured.err
        assert "stage1_as_test_1a" in captured.err, captured.err
        assert "ADR-007 §Stage 1a" in captured.err, captured.err
        assert "ADR-016 §Open questions" in captured.err, captured.err
        assert "PASS" in captured.out
        assert "axis=OM" in captured.out
        # AS was filtered out by sub_stage; its axis line MUST NOT appear.
        assert "axis=AS" not in captured.out

    def test_pure_1b_input_unchanged_behavior(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Pure 1b → no skipping note; verdict identical to pre-fix dry-run path."""
        as_1b = _plant_spike_run(
            tmp_path=tmp_path,
            axis="AS",
            sub_stage="1b",
            hetero_success_count=10,
            filename="spike_as.json",
        )
        om_1b = _plant_spike_run(
            tmp_path=tmp_path,
            axis="OM",
            sub_stage="1b",
            hetero_success_count=10,
            filename="spike_om.json",
        )
        rc = main(
            [
                "next-stage",
                "--prior-stage",
                "1",
                "--spike-runs",
                str(as_1b),
                str(om_1b),
            ]
        )
        captured = capsys.readouterr()
        assert rc == 0
        assert "skipping" not in captured.err
        assert "Stage-1a" not in captured.err
        assert "2/2 axis(es) pass" in captured.out

    def test_stage_2_filter_is_noop(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Stage 2 has no ``2a`` sub_stage → filter is structurally a no-op."""
        cr_2 = _plant_spike_run(
            tmp_path=tmp_path,
            axis="CR",
            sub_stage="2",
            hetero_success_count=10,
            filename="spike_cr.json",
        )
        cm_2 = _plant_spike_run(
            tmp_path=tmp_path,
            axis="CM",
            sub_stage="2",
            hetero_success_count=10,
            filename="spike_cm.json",
        )
        rc = main(
            [
                "next-stage",
                "--prior-stage",
                "2",
                "--spike-runs",
                str(cr_2),
                str(cm_2),
            ]
        )
        captured = capsys.readouterr()
        assert rc == 0
        assert "skipping" not in captured.err
        assert "Stage-1a" not in captured.err
        assert "2/2 axis(es) pass" in captured.out
