# SPDX-License-Identifier: Apache-2.0
"""Tier-1 tests for the scripted competent ego corrector (ADR-026; ADR-009).

The ego translates the lateral offset away and reorients the grasp-pose error in-grasp,
requesting a re-grasp only when the wrist alone cannot resolve it within the angular
window. It must read only the presented physical state + task spec — never any
presenter policy — and an end-to-end loop must reproduce the matched-success /
mismatch-failure split in the grasp-pose channel.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from chamber.agents.handover_ego_scripted import ScriptedHandoverEgo
from chamber.envs.handover_place import (
    HANDOVER_EGO_DIM,
    HANDOVER_MATCHED_REFERENCE,
    HANDOVER_PRESENTATION_MISMATCH,
    make_handover_place_env,
)
from chamber.partners.handover_presenter import HandoverPresenterPartner, presenter_spec

_ENV_KW: dict[str, Any] = {
    "lateral_window_m": 1.0e-3,
    "angular_window_deg": 5.0,
    "seating_force_limit_n": 75.0,
    "regrasp_budget_s": 1.0,
    "regrasp_duration_s": 2.0,
    "translation_range_m": 0.10,
    "wrist_correction_deg": 15.0,
    "reacquire_range_deg": 90.0,
    "contact_stiffness_n_per_m": 3.75e4,
    "angular_stiffness_n_per_deg": 7.5,
}


def _obs(*, lateral, grasp_pose_error_deg):
    return {
        "phase": 1,
        "target_pose": np.zeros(3, dtype=np.float64),
        "lateral_offset": np.asarray(lateral, dtype=np.float64),
        "grasp_pose_error_deg": float(grasp_pose_error_deg),
        "spec": {
            "lateral_window_m": 1.0e-3,
            "angular_window_deg": 5.0,
            "seating_force_limit_n": 75.0,
            "regrasp_budget_s": 1.0,
        },
    }


class TestCorrection:
    def test_action_shape(self) -> None:
        ego = ScriptedHandoverEgo(translation_range_m=0.10, wrist_correction_deg=15.0)
        action = ego.act(_obs(lateral=(0.0, 0.0), grasp_pose_error_deg=0.0))
        assert np.asarray(action).shape == (HANDOVER_EGO_DIM,)

    def test_translates_lateral_offset(self) -> None:
        ego = ScriptedHandoverEgo(translation_range_m=0.10, wrist_correction_deg=15.0)
        action = ego.act(_obs(lateral=(1.0e-3, 0.0), grasp_pose_error_deg=0.0))
        # Translation opposes the offset.
        assert action[0] < 0.0

    def test_translation_clipped_to_reach(self) -> None:
        ego = ScriptedHandoverEgo(translation_range_m=0.10, wrist_correction_deg=15.0)
        action = ego.act(_obs(lateral=(1.0, 0.0), grasp_pose_error_deg=0.0))
        assert float(np.linalg.norm(action[0:2])) <= 0.10 + 1e-12

    def test_reorient_clipped_to_wrist_range(self) -> None:
        ego = ScriptedHandoverEgo(translation_range_m=0.10, wrist_correction_deg=15.0)
        action = ego.act(_obs(lateral=(0.0, 0.0), grasp_pose_error_deg=120.0))
        assert abs(action[2]) <= 15.0 + 1e-12

    def test_requests_regrasp_for_large_grasp_pose_error(self) -> None:
        ego = ScriptedHandoverEgo(translation_range_m=0.10, wrist_correction_deg=15.0)
        action = ego.act(_obs(lateral=(0.0, 0.0), grasp_pose_error_deg=25.0))
        assert action[3] == 1.0

    def test_no_regrasp_for_small_grasp_pose_error(self) -> None:
        ego = ScriptedHandoverEgo(translation_range_m=0.10, wrist_correction_deg=15.0)
        action = ego.act(_obs(lateral=(0.0, 0.0), grasp_pose_error_deg=1.0))
        assert action[3] == 0.0


class TestReadsOnlyAllowedInfo:
    def test_act_needs_only_pose_spec_and_grasp_pose(self) -> None:
        ego = ScriptedHandoverEgo()
        obs = _obs(lateral=(0.0, 0.0), grasp_pose_error_deg=0.0)
        assert set(obs) == {
            "phase",
            "target_pose",
            "lateral_offset",
            "grasp_pose_error_deg",
            "spec",
        }
        ego.act(obs)  # succeeds with no presenter-policy information available


class TestEndToEndRegimeSplit:
    """matched succeeds; the grasp-pose mismatch fails under budget but is rescued by a
    free re-grasp (budget-mediated binding in the grasp-pose channel)."""

    def _run(self, *, condition_id, variant, free_regrasp, seeds):
        env = make_handover_place_env(
            condition_id=condition_id, free_regrasp=free_regrasp, **_ENV_KW
        )
        presenter = HandoverPresenterPartner(presenter_spec(variant))
        ego = ScriptedHandoverEgo(
            translation_range_m=_ENV_KW["translation_range_m"],
            wrist_correction_deg=_ENV_KW["wrist_correction_deg"],
        )
        successes = 0
        for seed in seeds:
            obs, _ = env.reset(seed=seed)
            presenter.reset(seed=seed)
            obs, _, _, _, _ = env.step(presenter.act(obs))
            _, _, _, _, info = env.step(ego.act(obs))
            successes += int(info["success"])
        return successes / len(seeds)

    def test_matched_solvable(self) -> None:
        rate = self._run(
            condition_id=HANDOVER_MATCHED_REFERENCE,
            variant="matched",
            free_regrasp=False,
            seeds=range(40),
        )
        assert rate >= 0.9

    def test_mismatch_degrades_under_budget(self) -> None:
        rate = self._run(
            condition_id=HANDOVER_PRESENTATION_MISMATCH,
            variant="mismatched",
            free_regrasp=False,
            seeds=range(40),
        )
        assert rate <= 0.5

    def test_free_regrasp_rescues_mismatch(self) -> None:
        budgeted = self._run(
            condition_id=HANDOVER_PRESENTATION_MISMATCH,
            variant="mismatched",
            free_regrasp=False,
            seeds=range(40),
        )
        free = self._run(
            condition_id=HANDOVER_PRESENTATION_MISMATCH,
            variant="mismatched",
            free_regrasp=True,
            seeds=range(40),
        )
        assert free > budgeted
