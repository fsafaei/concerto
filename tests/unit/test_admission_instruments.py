# SPDX-License-Identifier: Apache-2.0
"""Tier-1 tests for the admission-protocol instruments (ADR-027 §Admission protocol).

Covers the three net-new policy instruments — the zero-action ablated
partner (A2), the partner-blind co-carry ego (A3 / B-BLIND), the
scripted-competent pick-place ego (REF-SCRIPT) — and the wrap
extractors' pure filtering logic. Everything here is CPU-only and
SAPIEN-free: the controllers are exercised on synthetic observation
dicts (the Tier-1 contract-shape pattern).
"""

from __future__ import annotations

import numpy as np
import pytest

from chamber.benchmarks.admission_cells import resolve_cell_runner, resolve_wrap_extractor
from chamber.partners.ablation import PARTNER_ABLATED_ZERO_CLASS, PartnerAblatedZero
from chamber.partners.api import PartnerSpec
from chamber.partners.cocarry_blind import (
    COCARRY_BLIND_IMPEDANCE_CLASS,
    CoCarryBlindImpedancePartner,
)
from chamber.partners.registry import list_registered


class TestPartnerAblatedZero:
    """The A2 zero-action intervention (ADR-027 §Admission protocol)."""

    def test_registered(self) -> None:
        assert PARTNER_ABLATED_ZERO_CLASS in list_registered()

    def test_emits_zeros_of_bound_dim(self) -> None:
        partner = PartnerAblatedZero(
            PartnerSpec(PARTNER_ABLATED_ZERO_CLASS, 0, None, None, {"action_dim": "13"})
        )
        partner.reset(seed=3)
        action = partner.act({"anything": "ignored"})
        assert action.shape == (13,)
        assert not action.any()

    @pytest.mark.parametrize("raw", ["0", "-2", "x"])
    def test_bad_action_dim_is_loud(self, raw: str) -> None:
        with pytest.raises(ValueError, match="action_dim"):
            PartnerAblatedZero(
                PartnerSpec(PARTNER_ABLATED_ZERO_CLASS, 0, None, None, {"action_dim": raw})
            )

    def test_missing_action_dim_is_loud(self) -> None:
        with pytest.raises(ValueError, match="action_dim"):
            PartnerAblatedZero(PartnerSpec(PARTNER_ABLATED_ZERO_CLASS, 0, None, None, {}))


def _blind_obs(qpos: list[float], goal: tuple[float, float, float]) -> dict[str, object]:
    return {
        "agent": {"panda_wristcam": {"qpos": np.asarray([*qpos, 0.04, 0.04])}},
        "extra": {"goal_pos": np.asarray(goal)},
    }


_READY_ARM = [0.0, np.pi / 8, 0.0, -np.pi * 5 / 8, 0.0, np.pi * 3 / 4, -np.pi / 4]


class TestCoCarryBlindEgo:
    """B-BLIND: dead-reckoned joints, integral disabled (ADR-027 A3)."""

    def _make(self) -> CoCarryBlindImpedancePartner:
        return CoCarryBlindImpedancePartner(
            PartnerSpec(
                COCARRY_BLIND_IMPEDANCE_CLASS,
                0,
                None,
                None,
                {"uid": "panda_wristcam", "base_xyz": "0,0,0", "end_sign": "1"},
            )
        )

    def test_registered(self) -> None:
        assert COCARRY_BLIND_IMPEDANCE_CLASS in list_registered()

    def test_integral_gain_forced_to_zero(self) -> None:
        blind = CoCarryBlindImpedancePartner(
            PartnerSpec(COCARRY_BLIND_IMPEDANCE_CLASS, 0, None, None, {"ki": "0.6"})
        )
        assert blind._ki == 0.0  # the ki strip is the contract under test

    def test_ignores_measured_qpos_after_boot(self) -> None:
        """After the first step, a perturbed measured qpos must not change the action.

        The coupling channel is proprioceptive (the load enters through
        the measured joints); the blind ego dead-reckons instead, so a
        partner-induced deviation is invisible to it by construction.
        """
        blind_a = self._make()
        blind_b = self._make()
        blind_a.reset(seed=0)
        blind_b.reset(seed=0)
        obs = _blind_obs(_READY_ARM, (0.4, 0.0, 0.3))
        first_a = blind_a.act(obs)
        first_b = blind_b.act(obs)
        np.testing.assert_array_equal(first_a, first_b)
        # Perturb the measured joints for ego B only (a simulated
        # bar-transmitted load); both must keep emitting identically.
        perturbed = _blind_obs([q + 0.05 for q in _READY_ARM], (0.4, 0.0, 0.3))
        np.testing.assert_array_equal(blind_a.act(obs), blind_b.act(perturbed))

    def test_reset_clears_dead_reckoning(self) -> None:
        blind = self._make()
        obs = _blind_obs(_READY_ARM, (0.4, 0.0, 0.3))
        first = blind.act(obs).copy()
        blind.act(obs)
        blind.reset(seed=0)
        np.testing.assert_array_equal(blind.act(obs), first)

    def test_defensive_on_missing_keys(self) -> None:
        blind = self._make()
        action = blind.act({"agent": {}})
        assert action.shape == (8,)
        assert action[7] == 1.0
        assert not action[:7].any()


class TestScriptedPickPlaceEgo:
    """REF-SCRIPT phase machine on synthetic obs (ADR-011 as amended)."""

    @staticmethod
    def _obs(
        *,
        tcp: tuple[float, float, float],
        cube: tuple[float, float, float],
        goal: tuple[float, float, float],
        with_partner: bool = True,
    ) -> dict[str, object]:
        agent: dict[str, object] = {
            "panda_wristcam": {"qpos": np.asarray([*_READY_ARM, 0.04, 0.04])}
        }
        if with_partner:
            agent["fetch"] = {"qpos": np.zeros(15)}
        return {
            "agent": agent,
            "extra": {
                "tcp_pose": np.asarray([*tcp, 1.0, 0.0, 0.0, 0.0]),
                "cube_pose": np.asarray([*cube, 1.0, 0.0, 0.0, 0.0]),
                "goal_pos": np.asarray(goal),
            },
        }

    def test_action_shape_and_bounds(self) -> None:
        from chamber.agents.pickplace_ego_scripted import ScriptedPickPlaceEgo

        ego = ScriptedPickPlaceEgo()
        ego.reset(seed=0)
        action = ego.act(
            self._obs(tcp=(0.0, 0.0, 0.3), cube=(0.0, 0.0, 0.02), goal=(0.0, 0.1, 0.2))
        )
        assert action.shape == (8,)
        assert np.all(action[:7] >= -1.0)
        assert np.all(action[:7] <= 1.0)
        assert action[7] == 1.0  # approach with the gripper open

    def test_phase_progression_to_grasp(self) -> None:
        from chamber.agents.pickplace_ego_scripted import ScriptedPickPlaceEgo

        ego = ScriptedPickPlaceEgo()
        ego.reset(seed=0)
        cube = (0.0, 0.0, 0.02)
        # TCP already hovering right above the cube → transitions to descend.
        ego.act(self._obs(tcp=(0.0, 0.0, 0.10), cube=cube, goal=(0.0, 0.1, 0.2)))
        assert ego.phase == "descend"
        # TCP at grasp depth → transitions to grasp and closes the gripper.
        action = ego.act(self._obs(tcp=(0.0, 0.0, 0.018), cube=cube, goal=(0.0, 0.1, 0.2)))
        assert ego.phase == "grasp"
        assert action[7] == -1.0

    def test_partner_mask_is_enforced_and_blind_matches_reference(self) -> None:
        """The masked ego works without any partner subtree — and (the
        ADR-026 §Decision 3 construct fact) emits exactly the reference
        action, because the controller never reads a partner leaf."""
        from chamber.agents.pickplace_ego_scripted import ScriptedPickPlaceEgo

        reference = ScriptedPickPlaceEgo(mask_partner_obs=False)
        blind = ScriptedPickPlaceEgo(mask_partner_obs=True)
        reference.reset(seed=0)
        blind.reset(seed=0)
        obs_full = self._obs(tcp=(0.0, 0.0, 0.3), cube=(0.0, 0.0, 0.02), goal=(0.0, 0.1, 0.2))
        obs_bare = self._obs(
            tcp=(0.0, 0.0, 0.3), cube=(0.0, 0.0, 0.02), goal=(0.0, 0.1, 0.2), with_partner=False
        )
        np.testing.assert_array_equal(reference.act(obs_full), blind.act(obs_full))
        np.testing.assert_array_equal(blind.act(obs_full), blind.act(obs_bare))


class TestRegistries:
    """Loud-fail registry style (ADR-009 §Decision)."""

    def test_unknown_cell_runner_lists_known(self) -> None:
        with pytest.raises(KeyError, match="cocarry_scripted"):
            resolve_cell_runner("nope")

    def test_unknown_wrap_extractor_lists_known(self) -> None:
        with pytest.raises(KeyError, match="handover_gate0_limb1"):
            resolve_wrap_extractor("nope")

    def test_known_keys_resolve(self) -> None:
        assert callable(resolve_cell_runner("cocarry_scripted"))
        assert callable(resolve_cell_runner("pickplace_scripted"))
        assert callable(resolve_cell_runner("handover_presenter_ablated"))
        assert callable(resolve_wrap_extractor("handover_gate0_limb1"))
        assert callable(resolve_wrap_extractor("handover_gate0_limb2"))
