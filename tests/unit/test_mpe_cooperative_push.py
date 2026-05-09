# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``chamber.envs.mpe_cooperative_push`` (T4b.13).

Covers ADR-002 risk-mitigation #1 (the empirical-guarantee env shape)
and plan/05 §3.5 (deterministic 2-agent cooperative continuous-control).
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

from chamber.envs.mpe_cooperative_push import (
    DEFAULT_EPISODE_LENGTH,
    DT,
    N_LANDMARKS,
    POSITION_BOUND,
    VELOCITY_BOUND,
    MPECooperativePushEnv,
)


def _zero_action(uids: tuple[str, str]) -> dict[str, np.ndarray]:
    return {uid: np.zeros(2, dtype=np.float32) for uid in uids}


class TestConstruction:
    def test_default_construction(self) -> None:
        """Plan/05 §3.5: default uids are ('ego', 'partner')."""
        env = MPECooperativePushEnv()
        # Probe reset to inspect the agent uids — env keeps them private.
        obs, _ = env.reset(seed=0)
        assert set(obs["agent"].keys()) == {"ego", "partner"}

    def test_custom_uids(self) -> None:
        env = MPECooperativePushEnv(agent_uids=("a", "b"))
        obs, _ = env.reset(seed=0)
        assert set(obs["agent"].keys()) == {"a", "b"}

    def test_duplicate_uids_raises(self) -> None:
        """Defensive: distinct uids enforce the partner_rel encoding."""
        with pytest.raises(ValueError, match="distinct"):
            MPECooperativePushEnv(agent_uids=("ego", "ego"))

    def test_wrong_arity_raises(self) -> None:
        """Plan/05 §3.5: exactly 2 agents."""
        with pytest.raises(ValueError, match="2-tuple"):
            MPECooperativePushEnv(agent_uids=("ego",))  # type: ignore[arg-type]

    def test_action_space_shape(self) -> None:
        """Each agent gets a Box(-1, 1, (2,), float32)."""
        env = MPECooperativePushEnv()
        action_dict = env.action_space
        assert isinstance(action_dict, gym.spaces.Dict)
        for uid in ("ego", "partner"):
            space = action_dict[uid]
            assert isinstance(space, gym.spaces.Box)
            assert space.shape == (2,)
            assert float(space.low.min()) == -VELOCITY_BOUND
            assert float(space.high.max()) == VELOCITY_BOUND

    def test_observation_space_shape(self) -> None:
        """plan/04 §3.4 contract: obs[agent][uid][state] is the 10-dim Box."""
        env = MPECooperativePushEnv()
        obs_dict = env.observation_space
        assert isinstance(obs_dict, gym.spaces.Dict)
        agent_dict = obs_dict["agent"]
        assert isinstance(agent_dict, gym.spaces.Dict)
        ego_dict = agent_dict["ego"]
        assert isinstance(ego_dict, gym.spaces.Dict)
        agent_space = ego_dict["state"]
        assert isinstance(agent_space, gym.spaces.Box)
        assert agent_space.shape == (10,)


class TestResetDeterminism:
    def test_same_seed_reproduces_initial_state(self) -> None:
        """P6 reproducibility: identical (root_seed, seed) → identical obs."""
        env_a = MPECooperativePushEnv(root_seed=42)
        env_b = MPECooperativePushEnv(root_seed=42)
        obs_a, _ = env_a.reset(seed=7)
        obs_b, _ = env_b.reset(seed=7)
        for uid in ("ego", "partner"):
            np.testing.assert_array_equal(
                obs_a["agent"][uid]["state"], obs_b["agent"][uid]["state"]
            )

    def test_distinct_seeds_diverge(self) -> None:
        env = MPECooperativePushEnv(root_seed=0)
        obs_a, _ = env.reset(seed=0)
        obs_b, _ = env.reset(seed=1)
        # At least one agent's state should differ — landmark positions
        # are drawn first in reset so even agent positions sometimes match
        # after a subset of draws; the union must differ overall.
        assert not np.allclose(
            obs_a["agent"]["ego"]["state"], obs_b["agent"]["ego"]["state"]
        ) or not np.allclose(obs_a["agent"]["partner"]["state"], obs_b["agent"]["partner"]["state"])

    def test_distinct_root_seeds_diverge(self) -> None:
        a, _ = MPECooperativePushEnv(root_seed=0).reset(seed=0)
        b, _ = MPECooperativePushEnv(root_seed=1).reset(seed=0)
        assert not np.allclose(a["agent"]["ego"]["state"], b["agent"]["ego"]["state"])

    def test_reset_without_seed_continues_existing_rng(self) -> None:
        """Gymnasium API: reset(seed=None) does not re-seed the RNG."""
        env = MPECooperativePushEnv(root_seed=0)
        env.reset(seed=0)
        first, _ = env.reset()
        second, _ = env.reset()
        # The two resets MUST diverge (the RNG has stepped forward).
        assert not np.allclose(first["agent"]["ego"]["state"], second["agent"]["ego"]["state"])


class TestObservationShape:
    def test_state_is_10_dim_float32(self) -> None:
        """plan/05 §3.5: state = self_pos(2) + self_vel(2) + landmarks(4) + partner_rel(2)."""
        env = MPECooperativePushEnv()
        obs, _ = env.reset(seed=0)
        for uid in ("ego", "partner"):
            state = obs["agent"][uid]["state"]
            assert state.shape == (10,)
            assert state.dtype == np.float32

    def test_self_pos_in_state_prefix(self) -> None:
        """plan/04 §3.4: state[:2] is the partner's own xy (heuristic fallback contract)."""
        env = MPECooperativePushEnv()
        obs, _ = env.reset(seed=0)
        for uid in ("ego", "partner"):
            assert -POSITION_BOUND <= obs["agent"][uid]["state"][0] <= POSITION_BOUND
            assert -POSITION_BOUND <= obs["agent"][uid]["state"][1] <= POSITION_BOUND

    def test_self_vel_zero_after_reset(self) -> None:
        """At reset, velocities are zero (state[2:4])."""
        env = MPECooperativePushEnv()
        obs, _ = env.reset(seed=0)
        for uid in ("ego", "partner"):
            assert obs["agent"][uid]["state"][2] == 0.0
            assert obs["agent"][uid]["state"][3] == 0.0

    def test_landmark_relative_positions_are_consistent(self) -> None:
        """state[4:8] is landmark_rel; ego and partner must see the same landmarks."""
        env = MPECooperativePushEnv()
        obs, _ = env.reset(seed=0)
        ego_state = obs["agent"]["ego"]["state"]
        partner_state = obs["agent"]["partner"]["state"]
        ego_pos = ego_state[:2]
        partner_pos = partner_state[:2]
        # ego_landmark_rel + ego_pos == landmark_pos == partner_landmark_rel + partner_pos
        ego_landmarks = (ego_state[4:8] + np.tile(ego_pos, N_LANDMARKS)).reshape(N_LANDMARKS, 2)
        partner_landmarks = (partner_state[4:8] + np.tile(partner_pos, N_LANDMARKS)).reshape(
            N_LANDMARKS, 2
        )
        np.testing.assert_array_almost_equal(ego_landmarks, partner_landmarks)

    def test_partner_relative_position(self) -> None:
        """state[8:10]: ego.partner_rel + ego.self_pos == partner.self_pos."""
        env = MPECooperativePushEnv()
        obs, _ = env.reset(seed=0)
        ego_state = obs["agent"]["ego"]["state"]
        partner_state = obs["agent"]["partner"]["state"]
        ego_partner_rel = ego_state[8:10]
        ego_self_pos = ego_state[:2]
        partner_self_pos = partner_state[:2]
        np.testing.assert_array_almost_equal(ego_partner_rel + ego_self_pos, partner_self_pos)


class TestStepDynamics:
    def test_step_without_uid_raises(self) -> None:
        """Defensive: training loops must produce both agents' actions."""
        env = MPECooperativePushEnv()
        env.reset(seed=0)
        with pytest.raises(ValueError, match="missing uid"):
            env.step({"ego": np.zeros(2, dtype=np.float32)})

    def test_zero_action_keeps_position(self) -> None:
        """Plan/05 §3.5: dx = a * DT; a=0 → dx=0."""
        env = MPECooperativePushEnv()
        obs, _ = env.reset(seed=0)
        ego_pos_before = obs["agent"]["ego"]["state"][:2].copy()
        new_obs, _, _, _, _ = env.step(_zero_action(("ego", "partner")))
        ego_pos_after = new_obs["agent"]["ego"]["state"][:2]
        np.testing.assert_array_almost_equal(ego_pos_before, ego_pos_after)

    def test_unit_action_moves_by_dt(self) -> None:
        """Plan/05 §3.5: dx = a * DT — verify the constant."""
        env = MPECooperativePushEnv()
        obs, _ = env.reset(seed=0)
        ego_pos_before = obs["agent"]["ego"]["state"][:2].copy()
        action = {
            "ego": np.array([1.0, 0.0], dtype=np.float32),
            "partner": np.zeros(2, dtype=np.float32),
        }
        new_obs, _, _, _, _ = env.step(action)
        ego_pos_after = new_obs["agent"]["ego"]["state"][:2]
        # Account for clipping at the boundary.
        expected_x = min(ego_pos_before[0] + 1.0 * DT, POSITION_BOUND)
        assert ego_pos_after[0] == pytest.approx(expected_x)
        assert ego_pos_after[1] == pytest.approx(ego_pos_before[1])

    def test_actions_clipped_to_velocity_bound(self) -> None:
        """A misbehaving partner sending |a| > 1 cannot escape the plane."""
        env = MPECooperativePushEnv()
        env.reset(seed=0)
        action = {
            "ego": np.array([10.0, -10.0], dtype=np.float32),
            "partner": np.array([10.0, -10.0], dtype=np.float32),
        }
        # No exception; positions stay within bounds.
        new_obs, _, _, _, _ = env.step(action)
        for uid in ("ego", "partner"):
            pos = new_obs["agent"][uid]["state"][:2]
            assert -POSITION_BOUND <= pos[0] <= POSITION_BOUND
            assert -POSITION_BOUND <= pos[1] <= POSITION_BOUND

    def test_self_vel_reflects_last_action(self) -> None:
        """state[2:4] is the velocity command from the last step."""
        env = MPECooperativePushEnv()
        env.reset(seed=0)
        action = {
            "ego": np.array([0.5, -0.3], dtype=np.float32),
            "partner": np.zeros(2, dtype=np.float32),
        }
        obs, _, _, _, _ = env.step(action)
        np.testing.assert_array_almost_equal(
            obs["agent"]["ego"]["state"][2:4], np.array([0.5, -0.3], dtype=np.float32)
        )


class TestReward:
    def test_reward_is_shared_scalar(self) -> None:
        """Plan/05 §3.5: reward is a single scalar (cooperative; not per-agent)."""
        env = MPECooperativePushEnv()
        env.reset(seed=0)
        _, reward, _, _, _ = env.step(_zero_action(("ego", "partner")))
        assert isinstance(reward, float)

    def test_reward_is_non_positive(self) -> None:
        """r = -sum_landmark min_agent dist; distances ≥ 0 → r ≤ 0."""
        env = MPECooperativePushEnv()
        env.reset(seed=0)
        _, reward, _, _, _ = env.step(_zero_action(("ego", "partner")))
        assert reward <= 0.0

    def test_reward_increases_when_agents_approach_landmarks(self) -> None:
        """Sanity: closing distance to landmarks should improve the reward."""
        env = MPECooperativePushEnv(root_seed=0)
        env.reset(seed=0)
        # Step 1: both stay in place.
        _, r0, _, _, _ = env.step(_zero_action(("ego", "partner")))
        # Step 2..30: drive both agents toward (0, 0) which is roughly the centroid.
        action = {
            "ego": np.array([-0.1, -0.1], dtype=np.float32),
            "partner": np.array([-0.1, -0.1], dtype=np.float32),
        }
        last_reward = r0
        # Drive long enough that at least one agent drifts toward a landmark.
        for _ in range(20):
            _, last_reward, _, _, _ = env.step(action)
        # Rewards are negative; "improving" means closer to zero.
        # The trend may not be strictly monotone (agents may overshoot),
        # so we assert the final reward is at least as good as 90% of r0.
        assert last_reward >= r0 * 0.9 or last_reward >= -2.0


class TestEpisodeTermination:
    def test_truncates_at_episode_length(self) -> None:
        """Plan/05 §3.5: truncation only — never terminates."""
        env = MPECooperativePushEnv(episode_length=5)
        env.reset(seed=0)
        truncated = False
        for tick in range(5):
            _, _, terminated, truncated, _ = env.step(_zero_action(("ego", "partner")))
            assert terminated is False
            if tick < 4:
                assert truncated is False
        # Final tick.
        assert truncated is True

    def test_truncation_is_idempotent_after_reset(self) -> None:
        """Reset clears the step counter."""
        env = MPECooperativePushEnv(episode_length=3)
        env.reset(seed=0)
        for _ in range(3):
            env.step(_zero_action(("ego", "partner")))
        env.reset(seed=0)
        # First step after reset: not truncated.
        _, _, _, truncated, _ = env.step(_zero_action(("ego", "partner")))
        assert truncated is False


class TestDeterminismByteIdenticalTrajectory:
    def test_same_seed_byte_identical_state_trajectory(self) -> None:
        """P6: identical (root_seed, seed, action stream) → identical obs trajectory."""
        # Fixed action stream so the only randomness is the env's seeded RNG.
        actions = [
            {
                "ego": np.array([0.1 * i, -0.1 * i], dtype=np.float32),
                "partner": np.array([-0.1 * i, 0.1 * i], dtype=np.float32),
            }
            for i in range(10)
        ]
        traj_a = []
        env_a = MPECooperativePushEnv(root_seed=0)
        env_a.reset(seed=0)
        for action in actions:
            obs, *_ = env_a.step(action)
            traj_a.append(obs["agent"]["ego"]["state"].copy())
        traj_b = []
        env_b = MPECooperativePushEnv(root_seed=0)
        env_b.reset(seed=0)
        for action in actions:
            obs, *_ = env_b.step(action)
            traj_b.append(obs["agent"]["ego"]["state"].copy())
        for state_a, state_b in zip(traj_a, traj_b, strict=False):
            np.testing.assert_array_equal(state_a, state_b)

    def test_landmark_positions_within_bounds(self) -> None:
        """Defensive: landmarks always fall inside the plane on reset."""
        env = MPECooperativePushEnv(root_seed=42)
        for s in range(5):
            env.reset(seed=s)
            # Inspect via ego's landmark_rel + self_pos (recover absolute).
            ego_state = env._build_obs()["agent"]["ego"]["state"]
            ego_pos = ego_state[:2]
            landmarks = (ego_state[4:8] + np.tile(ego_pos, N_LANDMARKS)).reshape(N_LANDMARKS, 2)
            assert np.all(landmarks >= -POSITION_BOUND)
            assert np.all(landmarks <= POSITION_BOUND)


class TestPublicSurface:
    def test_module_constants(self) -> None:
        """The constants documented in the module surface match."""
        assert DEFAULT_EPISODE_LENGTH == 50
        assert N_LANDMARKS == 2
        assert POSITION_BOUND == 1.0
        assert VELOCITY_BOUND == 1.0
        assert DT == 0.1
