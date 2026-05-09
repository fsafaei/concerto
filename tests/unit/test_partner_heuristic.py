# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``chamber.partners.heuristic`` (T4.4).

Covers ADR-009 §Consequences (Stage-3 draft-zoo entry #1) and ADR-003 §Decision
(comm channel pose source).
"""

from __future__ import annotations

import numpy as np
import pytest

from chamber.partners.api import PartnerSpec
from chamber.partners.heuristic import ScriptedHeuristicPartner


def _spec(**extra: str) -> PartnerSpec:
    extra_full = {"uid": "fetch", "target_xy": "1.0,2.0", "action_dim": "2"}
    extra_full.update(extra)
    return PartnerSpec(
        class_name="scripted_heuristic",
        seed=0,
        checkpoint_step=None,
        weights_uri=None,
        extra=extra_full,
    )


class TestConstruction:
    def test_invalid_target_xy_format_raises(self) -> None:
        """plan/04 §3.4: target_xy must be 'x,y'."""
        with pytest.raises(ValueError, match="target_xy"):
            ScriptedHeuristicPartner(_spec(target_xy="1.0"))

    def test_invalid_target_xy_three_components_raises(self) -> None:
        """plan/04 §3.4: target_xy must be exactly two components, not three."""
        with pytest.raises(ValueError, match="target_xy"):
            ScriptedHeuristicPartner(_spec(target_xy="1.0,2.0,3.0"))

    def test_invalid_target_xy_components_raises(self) -> None:
        """plan/04 §3.4: components must be floats."""
        with pytest.raises(ValueError, match="target_xy"):
            ScriptedHeuristicPartner(_spec(target_xy="a,b"))

    def test_invalid_action_dim_raises(self) -> None:
        """plan/04 §3.4: action_dim is a positive int ≥ 2."""
        with pytest.raises(ValueError, match="action_dim"):
            ScriptedHeuristicPartner(_spec(action_dim="x"))

    def test_action_dim_too_small_raises(self) -> None:
        """Action dim must be at least 2 to carry an xy step."""
        with pytest.raises(ValueError, match="action_dim"):
            ScriptedHeuristicPartner(_spec(action_dim="1"))


class TestDeterminism:
    def test_two_calls_produce_identical_actions(self) -> None:
        """ADR-009 §Decision; P6: same obs → same action."""
        partner = ScriptedHeuristicPartner(_spec())
        obs = {"agent": {"fetch": {"state": np.array([0.0, 0.0, 0.0], dtype=np.float32)}}}
        a = partner.act(obs)
        b = partner.act(obs)
        np.testing.assert_array_equal(a, b)

    def test_seed_does_not_change_action(self) -> None:
        """ADR-009 §Decision; P6: heuristic is deterministic regardless of seed."""
        partner = ScriptedHeuristicPartner(_spec())
        obs = {"agent": {"fetch": {"state": np.array([0.5, 0.5, 0.0], dtype=np.float32)}}}
        partner.reset(seed=0)
        a = partner.act(obs)
        partner.reset(seed=999)
        b = partner.act(obs)
        np.testing.assert_array_equal(a, b)

    def test_deterministic_kwarg_does_not_change_action(self) -> None:
        """The heuristic ignores the deterministic kwarg (it has no stochastic mode)."""
        partner = ScriptedHeuristicPartner(_spec())
        obs = {"agent": {"fetch": {"state": np.array([0.5, 0.5, 0.0], dtype=np.float32)}}}
        a = partner.act(obs, deterministic=True)
        b = partner.act(obs, deterministic=False)
        np.testing.assert_array_equal(a, b)


class TestActionShapeAndBounds:
    def test_action_shape_matches_action_dim(self) -> None:
        """plan/04 §3.4: action vector length = spec.extra['action_dim']."""
        partner = ScriptedHeuristicPartner(_spec(action_dim="6"))
        obs = {"agent": {"fetch": {"state": np.zeros(3, dtype=np.float32)}}}
        action = partner.act(obs)
        assert action.shape == (6,)

    def test_action_dtype_is_float32(self) -> None:
        """plan/04 §3.4: float32 to match Box action spaces in tests/fakes."""
        partner = ScriptedHeuristicPartner(_spec())
        obs = {"agent": {"fetch": {"state": np.zeros(3, dtype=np.float32)}}}
        action = partner.act(obs)
        assert action.dtype == np.float32

    def test_action_components_clipped_to_unit(self) -> None:
        """plan/04 §3.4: each component is in [-1, 1] regardless of distance to target."""
        partner = ScriptedHeuristicPartner(_spec(target_xy="100.0,-100.0"))
        obs = {"agent": {"fetch": {"state": np.zeros(3, dtype=np.float32)}}}
        action = partner.act(obs)
        assert action[0] == pytest.approx(1.0)
        assert action[1] == pytest.approx(-1.0)

    def test_extra_components_are_zero(self) -> None:
        """plan/04 §3.4: only components 0/1 carry the planar step."""
        partner = ScriptedHeuristicPartner(_spec(action_dim="4"))
        obs = {"agent": {"fetch": {"state": np.zeros(3, dtype=np.float32)}}}
        action = partner.act(obs)
        assert action[2] == 0.0
        assert action[3] == 0.0


class TestPoseSources:
    def test_reads_from_comm_channel_when_present(self) -> None:
        """ADR-003 §Decision: comm pose takes precedence over agent state."""
        partner = ScriptedHeuristicPartner(_spec(target_xy="0.0,0.0"))
        obs = {
            "agent": {"fetch": {"state": np.array([5.0, 5.0, 0.0], dtype=np.float32)}},
            "comm": {"pose": {"fetch": {"xyz": (-0.5, 0.5, 0.0)}}},
        }
        action = partner.act(obs)
        # Step is target - pose = (0 - -0.5, 0 - 0.5) = (0.5, -0.5)
        assert action[0] == pytest.approx(0.5)
        assert action[1] == pytest.approx(-0.5)

    def test_falls_back_to_agent_state_when_comm_missing(self) -> None:
        """ADR-003 §Decision: agent.state[:2] is the fallback for comm-less envs."""
        partner = ScriptedHeuristicPartner(_spec(target_xy="0.0,0.0"))
        obs = {"agent": {"fetch": {"state": np.array([0.3, -0.4, 0.0], dtype=np.float32)}}}
        action = partner.act(obs)
        assert action[0] == pytest.approx(-0.3)
        assert action[1] == pytest.approx(0.4)

    def test_falls_back_to_origin_when_uid_absent(self) -> None:
        """Defensive: missing uid → assume origin so we never crash on partial obs."""
        partner = ScriptedHeuristicPartner(_spec(target_xy="0.5,-0.5"))
        action = partner.act({"agent": {}})
        assert action[0] == pytest.approx(0.5)
        assert action[1] == pytest.approx(-0.5)

    @pytest.mark.parametrize(
        "obs",
        [
            pytest.param({}, id="entirely-empty"),
            pytest.param({"comm": "not-a-mapping"}, id="comm-not-mapping"),
            pytest.param({"comm": {}}, id="comm-empty"),
            pytest.param({"comm": {"pose": "not-a-mapping"}}, id="pose-not-mapping"),
            pytest.param({"comm": {"pose": {}}}, id="pose-empty"),
            pytest.param(
                {"comm": {"pose": {"fetch": "not-a-mapping"}}},
                id="pose-uid-not-mapping",
            ),
            pytest.param(
                {"comm": {"pose": {"fetch": {}}}},
                id="pose-uid-no-xyz",
            ),
            pytest.param(
                {"comm": {"pose": {"fetch": {"xyz": (0.1,)}}}},
                id="xyz-too-short",
            ),
            pytest.param({"agent": "not-a-mapping"}, id="agent-not-mapping"),
            pytest.param({"agent": {"fetch": "not-a-mapping"}}, id="agent-uid-not-mapping"),
            pytest.param({"agent": {"fetch": {}}}, id="agent-uid-no-state"),
            pytest.param(
                {
                    "agent": {
                        "fetch": {"state": np.zeros(1, dtype=np.float32)},
                    },
                },
                id="state-too-short",
            ),
        ],
    )
    def test_malformed_obs_falls_through_to_origin(self, obs: dict) -> None:
        """ADR-003 §Decision: malformed pose surfaces fall back to origin, not raise.

        A partner that crashed on a missing/typo'd key would corrupt the
        Stage-3 PF gap metric (plan/04 §1). The three-tier fallback
        (``comm.pose[uid].xyz`` → ``agent[uid].state[:2]`` → origin) is
        intentional; this test pins the contract so future refactors of
        :meth:`_read_agent_xy` don't silently start raising.
        """
        partner = ScriptedHeuristicPartner(_spec(target_xy="0.5,-0.5"))
        action = partner.act(obs)
        assert action[0] == pytest.approx(0.5)
        assert action[1] == pytest.approx(-0.5)
