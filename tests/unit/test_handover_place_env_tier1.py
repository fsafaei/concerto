# SPDX-License-Identifier: Apache-2.0
"""Tier-1 contract tests for the kinematic handover-and-place env (ADR-026; non-gating).

Pure-Python, no SAPIEN: determinism, the downstream place/seat gate, the re-grasp
budget accounting, and the three binding regimes in the GRASP-POSE channel (matched
success, budget-mediated mismatch failure, kinematically intrinsic failure). Also
verifies the Rev-2 channel split: lateral is success-side (translated away), grasp-pose
is coupling-side. The explicit numeric parameters are mechanism-demonstration values,
NOT the externally-sourced Gate-0 figures.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from chamber.envs.handover_place import (
    HANDOVER_EGO_DIM,
    HANDOVER_MATCHED_REFERENCE,
    HANDOVER_PHASE_PLACE,
    HANDOVER_PHASE_PRESENT,
    HANDOVER_PRESENTATION_DIM,
    HANDOVER_PRESENTATION_MISMATCH,
    HandoverPlaceEnv,
    evaluate_handover_place_success,
    handover_conditions,
    make_handover_place_env,
    resolve_placement,
)

# Mechanism-demonstration parameters (NOT the committed Gate-0 spec).
_ENV_KW: dict[str, Any] = {
    "lateral_window_m": 1.0e-3,
    "angular_window_deg": 3.0,
    "seating_force_limit_n": 75.0,
    "regrasp_budget_s": 1.0,
    "regrasp_duration_s": 2.0,
    "translation_range_m": 0.10,
    "wrist_correction_deg": 15.0,
    "reacquire_range_deg": 90.0,
    "contact_stiffness_n_per_m": 3.75e4,
    "angular_stiffness_n_per_deg": 12.5,
}


def _present(lateral=(0.0, 0.0), grasp_pose_error_deg=0.0, skew=0.0):
    a = np.zeros(HANDOVER_PRESENTATION_DIM)
    a[0:2] = lateral
    a[2] = grasp_pose_error_deg
    a[3] = skew
    return a


def _ego(translation=(0.0, 0.0), reorient=0.0, regrasp=False):
    a = np.zeros(HANDOVER_EGO_DIM)
    a[0:2] = translation
    a[2] = reorient
    a[3] = 1.0 if regrasp else 0.0
    return a


class TestConditionTable:
    def test_locked_condition_names(self) -> None:
        conds = handover_conditions()
        assert set(conds) == {HANDOVER_MATCHED_REFERENCE, HANDOVER_PRESENTATION_MISMATCH}
        assert conds[HANDOVER_MATCHED_REFERENCE].presenter == "matched"
        assert conds[HANDOVER_PRESENTATION_MISMATCH].presenter == "mismatched"

    def test_unknown_condition_rejected(self) -> None:
        with pytest.raises(KeyError):
            HandoverPlaceEnv(condition_id="nope")


class TestDeterminism:
    def test_reset_same_seed_byte_identical(self) -> None:
        env_a = make_handover_place_env(root_seed=0, **_ENV_KW)
        env_b = make_handover_place_env(root_seed=0, **_ENV_KW)
        obs_a, info_a = env_a.reset(seed=7)
        obs_b, info_b = env_b.reset(seed=7)
        np.testing.assert_array_equal(obs_a["target_pose"], obs_b["target_pose"])
        assert info_a["initial_state_seed"] == info_b["initial_state_seed"] == 7

    def test_reset_different_seed_differs(self) -> None:
        env = make_handover_place_env(**_ENV_KW)
        obs_a, _ = env.reset(seed=1)
        target_a = obs_a["target_pose"].copy()
        obs_b, _ = env.reset(seed=2)
        assert not np.array_equal(target_a, obs_b["target_pose"])

    def test_full_episode_reproducible(self) -> None:
        def run(seed: int):
            env = make_handover_place_env(condition_id=HANDOVER_PRESENTATION_MISMATCH, **_ENV_KW)
            env.reset(seed=seed)
            env.step(_present(grasp_pose_error_deg=25.0))
            _, reward, term, trunc, info = env.step(_ego(regrasp=True))
            return reward, term, trunc, info

        assert run(3) == run(3)


class TestPhaseMachine:
    def test_phase_advances(self) -> None:
        env = make_handover_place_env(**_ENV_KW)
        obs, _ = env.reset(seed=0)
        assert obs["phase"] == HANDOVER_PHASE_PRESENT
        assert obs["lateral_offset"] is None
        assert obs["grasp_pose_error_deg"] is None
        obs, _, term, _, _ = env.step(_present(grasp_pose_error_deg=1.0))
        assert obs["phase"] == HANDOVER_PHASE_PLACE
        assert obs["lateral_offset"] is not None
        assert obs["grasp_pose_error_deg"] is not None
        assert term is False
        _, _, term, trunc, _ = env.step(_ego())
        assert term is True
        assert trunc is False

    def test_presentation_dim_validated(self) -> None:
        env = make_handover_place_env(**_ENV_KW)
        env.reset(seed=0)
        with pytest.raises(ValueError, match="presentation action must have dim"):
            env.step(np.zeros(3))

    def test_ego_dim_validated(self) -> None:
        env = make_handover_place_env(**_ENV_KW)
        env.reset(seed=0)
        env.step(_present(grasp_pose_error_deg=1.0))
        with pytest.raises(ValueError, match="ego action must have dim"):
            env.step(np.zeros(2))


class TestToleranceGate:
    def test_success_inside_all_windows(self) -> None:
        assert evaluate_handover_place_success(
            0.5e-3,
            1.0,
            20.0,
            lateral_window_m=1.0e-3,
            angular_window_deg=3.0,
            seating_force_limit_n=75.0,
        )

    def test_angular_over_window_fails(self) -> None:
        assert not evaluate_handover_place_success(
            0.5e-3,
            5.0,
            20.0,
            lateral_window_m=1.0e-3,
            angular_window_deg=3.0,
            seating_force_limit_n=75.0,
        )

    def test_force_over_limit_fails_even_if_pose_ok(self) -> None:
        assert not evaluate_handover_place_success(
            0.5e-3,
            1.0,
            120.0,
            lateral_window_m=1.0e-3,
            angular_window_deg=3.0,
            seating_force_limit_n=75.0,
        )


class TestRegimes:
    def test_matched_small_grasp_pose_succeeds(self) -> None:
        out = resolve_placement(
            lateral_offset_m=(1.0e-4, 0.0),
            grasp_pose_error_deg=1.0,
            timing_skew_s=0.0,
            ego_translation_m=(-1.0e-4, 0.0),
            ego_reorient_deg=-1.0,
            ego_regrasp_requested=False,
            **_ENV_KW,
        )
        assert out.success
        assert out.failure_mode == "none"

    def test_budget_mediated_failure(self) -> None:
        # Grasp-pose error past the wrist range but within the re-acquire range: the
        # re-grasp does not fit the budget -> fails; a free re-grasp would rescue it.
        out = resolve_placement(
            lateral_offset_m=(0.0, 0.0),
            grasp_pose_error_deg=25.0,
            timing_skew_s=0.0,
            ego_translation_m=(0.0, 0.0),
            ego_reorient_deg=-15.0,
            ego_regrasp_requested=True,
            **_ENV_KW,
        )
        assert not out.success
        assert out.regrasp_budget_blocked
        assert not out.regrasp_executed
        assert out.failure_mode == "budget_mediated"
        assert "angular" in out.binding_conjunct

    def test_free_regrasp_rescues_budget_mediated(self) -> None:
        kw = {**_ENV_KW, "regrasp_budget_s": float("inf")}
        out = resolve_placement(
            lateral_offset_m=(0.0, 0.0),
            grasp_pose_error_deg=25.0,
            timing_skew_s=0.0,
            ego_translation_m=(0.0, 0.0),
            ego_reorient_deg=-15.0,
            ego_regrasp_requested=True,
            **kw,
        )
        assert out.success
        assert out.regrasp_executed

    def test_intrinsic_failure_beyond_reacquire(self) -> None:
        # Grasp-pose error beyond the re-acquire range: even a free re-grasp leaves a
        # residual.
        kw = {**_ENV_KW, "regrasp_budget_s": float("inf")}
        out = resolve_placement(
            lateral_offset_m=(0.0, 0.0),
            grasp_pose_error_deg=120.0,
            timing_skew_s=0.0,
            ego_translation_m=(0.0, 0.0),
            ego_reorient_deg=-15.0,
            ego_regrasp_requested=True,
            **kw,
        )
        assert not out.success
        assert out.beyond_reacquire
        assert out.failure_mode == "intrinsic"


class TestChannelSplit:
    """Rev-2 T3: lateral is success-side (only breaks at huge offsets beyond reach);
    grasp-pose is coupling-side."""

    def test_lateral_is_success_side_only_beyond_reach(self) -> None:
        # A lateral offset within the translation reach is fully corrected.
        out = resolve_placement(
            lateral_offset_m=(0.05, 0.0),
            grasp_pose_error_deg=0.0,
            timing_skew_s=0.0,
            ego_translation_m=(-0.05, 0.0),
            ego_reorient_deg=0.0,
            ego_regrasp_requested=False,
            **_ENV_KW,
        )
        assert out.success
        # Only an offset beyond the translation reach breaks the lateral window.
        out_far = resolve_placement(
            lateral_offset_m=(0.2, 0.0),
            grasp_pose_error_deg=0.0,
            timing_skew_s=0.0,
            ego_translation_m=(-0.2, 0.0),
            ego_reorient_deg=0.0,
            ego_regrasp_requested=False,
            **_ENV_KW,
        )
        assert not out_far.success
        assert "lateral" in out_far.binding_conjunct

    def test_grasp_pose_failure_is_angular(self) -> None:
        out = resolve_placement(
            lateral_offset_m=(0.0, 0.0),
            grasp_pose_error_deg=25.0,
            timing_skew_s=0.0,
            ego_translation_m=(0.0, 0.0),
            ego_reorient_deg=-15.0,
            ego_regrasp_requested=True,
            **_ENV_KW,
        )
        assert "angular" in out.binding_conjunct
        assert "lateral" not in out.binding_conjunct


class TestBudgetAccounting:
    def test_regrasp_executes_when_affordable(self) -> None:
        out = resolve_placement(
            lateral_offset_m=(0.0, 0.0),
            grasp_pose_error_deg=25.0,
            timing_skew_s=0.0,
            ego_translation_m=(0.0, 0.0),
            ego_reorient_deg=-15.0,
            ego_regrasp_requested=True,
            **{**_ENV_KW, "regrasp_budget_s": 3.0, "regrasp_duration_s": 2.0},
        )
        assert out.regrasp_executed
        assert not out.regrasp_budget_blocked

    def test_timing_skew_eats_budget(self) -> None:
        out = resolve_placement(
            lateral_offset_m=(0.0, 0.0),
            grasp_pose_error_deg=25.0,
            timing_skew_s=1.5,
            ego_translation_m=(0.0, 0.0),
            ego_reorient_deg=-15.0,
            ego_regrasp_requested=True,
            **{**_ENV_KW, "regrasp_budget_s": 3.0, "regrasp_duration_s": 2.0},
        )
        assert out.regrasp_budget_blocked
        assert not out.regrasp_executed

    def test_factory_free_regrasp_sets_infinite_budget(self) -> None:
        env = make_handover_place_env(free_regrasp=True, **_ENV_KW)
        assert env.regrasp_budget_s == float("inf")


class TestNoPolicyLeakInObs:
    def test_obs_exposes_only_task_and_physical_state(self) -> None:
        env = make_handover_place_env(**_ENV_KW)
        obs, _ = env.reset(seed=0)
        obs, *_ = env.step(_present(lateral=(1.0e-4, 0.0), grasp_pose_error_deg=5.0))
        assert set(obs) == {
            "phase",
            "target_pose",
            "spec",
            "lateral_offset",
            "grasp_pose_error_deg",
        }
        for forbidden in ("policy", "weights", "partner", "reward", "logits"):
            assert forbidden not in obs
            assert forbidden not in obs["spec"]
