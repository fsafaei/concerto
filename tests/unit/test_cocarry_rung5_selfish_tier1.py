# SPDX-License-Identifier: Apache-2.0
# pyright: reportPrivateImportUsage=false, reportAttributeAccessIssue=false
"""Tier-1 (no-SAPIEN-scene) tests for the Rung-5 Arm-B selfish teammates (ADR-026 §D4; ADR-009).

Covers the pure-Python surface that needs no SAPIEN scene:

- :mod:`chamber.partners.cocarry_selfish` — the three non-co-designed variants
  register, parse their spec, and their ``act`` returns an 8-D action that is
  finite, bounded, and deterministic across reloads, with the defensive hold on
  incomplete obs. The matched-EXCEPT-objective contract is pinned directly on
  the objective hooks: ``cocarry_selfish_goal`` targets the goal centroid (not
  the cooperative bar-end), ``cocarry_selfish_effort`` dead-bands + down-scales
  its own command, and ``cocarry_selfish_station`` caches and holds its own
  start pose. The shared machinery (gains, joint step, geometry) is the matched
  ``_CoCarryShiftedBase`` unchanged, so the isolation (differ only in objective)
  is structural.

Tier-2 SAPIEN/CUDA-gated coverage (the real shifted-eval + competence gate) is
the Stage-2 calibration runner.

ADR-026 §Decision 4; ADR-009 §Decision; Rung-5 co-design (CD) measurement.
"""

from __future__ import annotations

import numpy as np
import pytest

from chamber.partners.cocarry_policy_shift import _CoCarryShiftedBase
from chamber.partners.cocarry_selfish import COCARRY_SELFISH_CANDIDATES

_READY_Q7 = np.array([0.0, np.pi / 8, 0.0, -np.pi * 5 / 8, 0.0, np.pi * 3 / 4, -np.pi / 4])


def _partner(class_name: str):
    """Build a selfish teammate on the partner seat (the env geometry)."""
    from chamber.partners.api import PartnerSpec
    from chamber.partners.registry import load_partner

    extra = {
        "uid": "panda_partner",
        "base_xyz": "0.5,0,0",
        "base_yaw_deg": "180",
        "end_sign": "-1",
        "bar_half_len": "0.115",
    }
    return load_partner(PartnerSpec(class_name, 0, None, None, extra))


def _obs(q7: np.ndarray, goal: np.ndarray) -> dict:
    """Synthetic partner-seat obs: panda_partner qpos + goal leaf (task leaves only)."""
    return {
        "agent": {"panda_partner": {"qpos": np.concatenate([q7, [0.04, 0.04]]).astype(np.float32)}},
        "extra": {"goal_pos": goal.astype(np.float32)},
    }


class TestSelfishRegistration:
    """The three Arm-B variants register under their canonical class names (ADR-026 §D4)."""

    def test_candidate_tuple_has_three_distinct(self) -> None:
        assert len(COCARRY_SELFISH_CANDIDATES) == 3
        assert len(set(COCARRY_SELFISH_CANDIDATES)) == 3

    def test_all_registered(self) -> None:
        from chamber.partners.registry import list_registered

        registered = list_registered()
        for cls in COCARRY_SELFISH_CANDIDATES:
            assert cls in registered


class TestSelfishActContract:
    """Every variant returns a finite, bounded, deterministic 8-D action (ADR-009)."""

    @pytest.mark.parametrize("cls", COCARRY_SELFISH_CANDIDATES)
    def test_act_eight_dim_open_gripper_bounded(self, cls: str) -> None:
        ctrl = _partner(cls)
        ctrl.reset(seed=0)
        a = ctrl.act(_obs(_READY_Q7, np.array([0.0, 0.12, 0.28])))
        assert a.shape == (8,)
        assert a.dtype == np.float32
        assert a[7] == pytest.approx(1.0)  # gripper open
        assert np.all(np.abs(a) <= 1.0 + 1e-6)
        assert np.all(np.isfinite(a))

    @pytest.mark.parametrize("cls", COCARRY_SELFISH_CANDIDATES)
    def test_act_deterministic_across_reloads(self, cls: str) -> None:
        obs = _obs(_READY_Q7, np.array([0.0, 0.12, 0.28]))
        a = _partner(cls)
        a.reset(seed=0)
        b = _partner(cls)
        b.reset(seed=0)
        np.testing.assert_array_equal(a.act(obs), b.act(obs))

    @pytest.mark.parametrize("cls", COCARRY_SELFISH_CANDIDATES)
    def test_incomplete_obs_holds(self, cls: str) -> None:
        ctrl = _partner(cls)
        ctrl.reset(seed=0)
        a = ctrl.act({"agent": {}, "extra": {}})
        assert a.shape == (8,)
        assert a[7] == pytest.approx(1.0)
        np.testing.assert_array_equal(a[:7], np.zeros(7, dtype=np.float32))

    @pytest.mark.parametrize("cls", COCARRY_SELFISH_CANDIDATES)
    def test_matched_class_and_gains(self, cls: str) -> None:
        # The isolation guarantee: every variant is the matched-impedance class
        # with the matched gains (differs only in the objective hook).
        ctrl = _partner(cls)
        assert isinstance(ctrl, _CoCarryShiftedBase)
        assert ctrl._KP == _CoCarryShiftedBase._KP
        assert ctrl._STEP_MAX_M == _CoCarryShiftedBase._STEP_MAX_M
        assert ctrl._DAMPING == _CoCarryShiftedBase._DAMPING


class TestSelfishGoalObjective:
    """cocarry_selfish_goal targets the goal centroid, not the cooperative bar-end (ADR-026 §D4)."""

    def test_target_is_the_goal_centroid_not_the_bar_end(self) -> None:
        ctrl = _partner("cocarry_selfish_goal")
        ctrl.reset(seed=0)
        goal = np.array([0.0, 0.12, 0.28])
        target = ctrl._target_world(goal, {})
        np.testing.assert_array_equal(target, goal)
        # The cooperative bar-end (what the matched/Arm-A controllers target)
        # is offset by end_sign*bar_half_len in x — the selfish-goal variant
        # ignores that split.
        coop = ctrl._cooperative_target_world(goal)
        assert not np.allclose(target, coop)


class TestSelfishEffortObjective:
    """cocarry_selfish_effort dead-bands and down-scales its own command (ADR-026 §D4)."""

    def test_deadband_zeroes_small_error(self) -> None:
        ctrl = _partner("cocarry_selfish_effort")
        ctrl.reset(seed=0)
        small = np.array([0.01, 0.0, 0.0])  # norm 0.01 < deadband 0.04
        np.testing.assert_array_equal(ctrl._cartesian_command(small, {}), np.zeros(3))

    def test_downscaled_authority_on_large_error(self) -> None:
        ctrl = _partner("cocarry_selfish_effort")
        ctrl.reset(seed=0)
        large = np.array([0.5, 0.0, 0.0])
        matched = np.clip(ctrl._KP * large, -ctrl._STEP_MAX_M, ctrl._STEP_MAX_M)
        cmd = ctrl._cartesian_command(large, {})
        # contributes, but at reduced authority vs the matched law.
        assert np.linalg.norm(cmd) > 0.0
        assert np.linalg.norm(cmd) < np.linalg.norm(matched)


class TestSelfishStationObjective:
    """cocarry_selfish_station caches and holds its own start pose (ADR-026 §D4)."""

    def test_target_is_cached_and_goal_independent(self) -> None:
        ctrl = _partner("cocarry_selfish_station")
        ctrl.reset(seed=0)
        obs = _obs(_READY_Q7, np.array([0.0, 0.12, 0.28]))
        first = ctrl._target_world(np.array([0.0, 0.12, 0.28]), obs)
        # A later step with a DIFFERENT goal returns the SAME cached station target.
        second = ctrl._target_world(np.array([0.5, 0.5, 0.5]), obs)
        np.testing.assert_array_equal(first, second)

    def test_reset_clears_the_station_cache(self) -> None:
        ctrl = _partner("cocarry_selfish_station")
        ctrl.reset(seed=0)
        obs_a = _obs(_READY_Q7, np.array([0.0, 0.12, 0.28]))
        first = ctrl._target_world(np.array([0.0, 0.12, 0.28]), obs_a).copy()
        ctrl.reset(seed=0)
        obs_b = _obs(np.zeros(7), np.array([0.0, 0.12, 0.28]))  # different start qpos
        refreshed = ctrl._target_world(np.array([0.0, 0.12, 0.28]), obs_b)
        assert not np.allclose(first, refreshed)
