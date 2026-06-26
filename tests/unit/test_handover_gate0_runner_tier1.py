# SPDX-License-Identifier: Apache-2.0
"""Tier-1 smoke for the Gate-0 runner (ADR-026; Rev 2 §6).

Exercises the per-cell episode loop, determinism, and the analysis pipeline on a TINY
synthetic config (no committed measurement; this tests the harness, not the Gate-0
result). The full grid run is PR C against the tagged prereg.
"""

from __future__ import annotations

from typing import Any

from chamber.spikes.handover_place_gate0.decision import VERDICT_LIMB1_FAIL
from chamber.spikes.handover_place_gate0.runner import (
    analyze,
    clearance_threshold_overlay,
    run_cell_episodes,
    run_gate0,
)

_ENV_PARAMS: dict[str, Any] = {
    "lateral_window_m": 1.0e-3,
    "angular_window_deg": 5.0,
    "seating_force_limit_n": 75.0,
    "translation_range_m": 0.10,
    "reacquire_range_deg": 170.0,
    "contact_stiffness_n_per_m": 3.75e4,
    "angular_stiffness_n_per_deg": 7.5,
}
_MISMATCH = {"grasp_pose_bias_deg": 30.0, "grasp_pose_sigma_deg": 10.0}


class TestCellEpisodes:
    def test_runs_and_is_deterministic(self) -> None:
        def run():
            return run_cell_episodes(
                variant="mismatched",
                clearance_factor=0.2,
                takt_s=1.0,
                arm_basis="fast",
                place_cycle_s=0.37,
                regrasp_duration_s=1.06,
                free_regrasp=False,
                seeds=[0, 1],
                episodes_per_seed=3,
                env_params=_ENV_PARAMS,
                presenter_params=_MISMATCH,
            )

        a, b = run(), run()
        assert len(a) == 6
        assert [(e.seed, e.episode_idx, e.success) for e in a] == [
            (e.seed, e.episode_idx, e.success) for e in b
        ]

    def test_paired_initial_state_seed_shared_across_variants(self) -> None:
        def _run(variant: str, pp: dict[str, float]):
            return run_cell_episodes(
                variant=variant,
                clearance_factor=0.2,
                takt_s=1.0,
                arm_basis="fast",
                place_cycle_s=0.37,
                regrasp_duration_s=1.06,
                free_regrasp=False,
                seeds=[0],
                episodes_per_seed=2,
                env_params=_ENV_PARAMS,
                presenter_params=pp,
            )

        m = _run("matched", {})
        mm = _run("mismatched", _MISMATCH)
        assert [e.initial_state_seed for e in m] == [e.initial_state_seed for e in mm]


class TestOverlay:
    def test_threshold_rises_with_clearance(self) -> None:
        rows = clearance_threshold_overlay(
            clearance_factors=[0.2, 0.7],
            angular_window_deg=3.0,
            matched_grasp_pose_sigma_deg=2.0,
            mismatched_grasp_pose_bias_deg=30.0,
            mismatched_grasp_pose_sigma_deg=10.0,
        )
        assert rows[0]["binding_threshold_deg"] < rows[1]["binding_threshold_deg"]
        # More mismatch mass clears the lower-clearance (tighter wrist) threshold.
        assert rows[0]["mismatch_mass_clearing"] > rows[1]["mismatch_mass_clearing"]


class TestRunGate0Pipeline:
    def test_tiny_grid_produces_a_verdict(self) -> None:
        params: dict[str, Any] = {
            "seeds": [0, 1],
            "episodes_per_seed": 4,
            "clearance_factor_sweep": [0.2, 0.7],
            "takt_grid_s": [0.5, 2.0],
            "arm_bases": {
                "fast": {"place_cycle_s": 0.37, "regrasp_duration_s": 1.06},
                "slow": {"place_cycle_s": 0.58, "regrasp_duration_s": 1.90},
            },
            "env_params": _ENV_PARAMS,
            "n_boot": 200,
            "matched_presenter_params": {},
            "mismatched_presenter_params": {"grasp_pose_sigma_deg": 10.0},
            "mismatched_grasp_pose_bias_sweep_deg": [15.0, 45.0],
            "matched_grasp_pose_sigma_deg": 2.0,
            "mismatched_grasp_pose_sigma_deg": 10.0,
            "prereg_sha": "tiny-smoke",
            "git_tag": "tiny-smoke",
        }
        spike_run, analysis = run_gate0(params)
        assert spike_run.schema_version == 2
        assert spike_run.episode_results
        assert analysis["verdict"] in {
            "COUPLING_VALID",
            "COUPLING_VALID_INTRINSIC",
            "WASHOUT",
            "WASHOUT_FOR_REAL_CELLS",
            "LIMB1_FAIL",
            "INDETERMINATE",
        }
        # The (clearance x mismatch) overlay: 2 clearances x 2 biases.
        assert len(analysis["clearance_threshold_overlay"]) == 4
        # The measured (clearance x mismatch) coupling region: 2 clearances x 2 biases.
        assert len(analysis["mismatch_coupling_region"]) == 4

    def test_analyze_empty_is_limb1_fail(self) -> None:
        out = analyze(
            [],
            clearance_factors=[0.2],
            takt_grid_s=[1.0],
            arm_bases=["fast"],
            n_boot=100,
            overlay=[],
        )
        assert out["verdict"] == VERDICT_LIMB1_FAIL
