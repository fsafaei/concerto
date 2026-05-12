# SPDX-License-Identifier: Apache-2.0
"""Tier-1 wrapper-structure tests for the Stage-0 training adapter (T4b.3).

Mirrors the two-tier pattern of
``tests/integration/test_stage0_smoke.py``. This file exercises the
adapter wired to :class:`tests.fakes.FakeMultiAgentEnv` so it always
runs on CPU. The Tier-2 real-SAPIEN smoke lives in
``test_stage0_adapter_real.py`` and skips on CPU-only hosts.

Covers:

- ADR-001 §Validation criteria — the 3-robot uid tuple flows through
  the adapter unchanged.
- plan/05 §3.4 — the adapter exposes the
  :class:`~concerto.training.ego_aht.EnvLike` shape
  :func:`chamber.benchmarks.ego_ppo_trainer.EgoPPOTrainer.from_config`
  reads (Box action_space[ego_uid], Box observation_space["agent"][ego_uid]["state"]).
- Action augmentation: inner uids not in the trainer's ``agent_uids``
  receive zero-action injection.
- State synthesis: when the inner per-uid obs has no ``state`` key but
  has 1-D Box channels, the adapter builds a flat state vector.
- ``ChamberEnvCompatibilityError`` propagation from
  :func:`chamber.benchmarks.stage0_smoke_adapter.make_stage0_training_env`
  on Vulkan-unavailable hosts (the same error message format that
  :func:`chamber.benchmarks.stage0_smoke.make_stage0_env` already
  emits).
"""

from __future__ import annotations

from typing import ClassVar

import gymnasium as gym
import numpy as np
import pytest

from chamber.benchmarks.stage0_smoke import _SMOKE_ROBOT_UIDS
from chamber.benchmarks.stage0_smoke_adapter import (
    DEFAULT_EPISODE_LENGTH,
    _Stage0TrainingAdapter,
    make_stage0_training_env,
)
from chamber.envs.errors import ChamberEnvCompatibilityError
from chamber.utils.device import sapien_gpu_available
from concerto.training.ego_aht import EnvLike
from tests.fakes import FakeMultiAgentEnv

_TRAINING_AGENT_UIDS: tuple[str, str] = ("panda_wristcam", "fetch")


class TestPassThroughOnFakeEnv:
    """FakeMultiAgentEnv already exposes a ``state`` channel for every uid."""

    def test_action_space_only_exposes_training_uids(self) -> None:
        """plan/05 §3.4: action_space drops inner uids the trainer doesn't drive."""
        inner = FakeMultiAgentEnv(agent_uids=_SMOKE_ROBOT_UIDS)
        adapter = _Stage0TrainingAdapter(
            inner, agent_uids=_TRAINING_AGENT_UIDS, episode_length=10, root_seed=0
        )
        assert isinstance(adapter.action_space, gym.spaces.Dict)
        # gym.spaces.Dict re-orders keys alphabetically; the contract is
        # set equality, not order equality.
        assert set(adapter.action_space.spaces.keys()) == set(_TRAINING_AGENT_UIDS)

    def test_ego_action_space_is_box(self) -> None:
        """ADR-002 §Decisions: EgoPPOTrainer.from_config requires action_space[ego_uid] = Box."""
        inner = FakeMultiAgentEnv(agent_uids=_SMOKE_ROBOT_UIDS)
        adapter = _Stage0TrainingAdapter(
            inner, agent_uids=_TRAINING_AGENT_UIDS, episode_length=10, root_seed=0
        )
        assert isinstance(adapter.action_space, gym.spaces.Dict)
        ego_box = adapter.action_space.spaces["panda_wristcam"]
        assert isinstance(ego_box, gym.spaces.Box)
        assert ego_box.shape == (2,)

    def test_obs_space_ego_state_is_box(self) -> None:
        """ADR-002 §Decisions: observation_space['agent'][ego_uid]['state'] = Box."""
        inner = FakeMultiAgentEnv(agent_uids=_SMOKE_ROBOT_UIDS)
        adapter = _Stage0TrainingAdapter(
            inner, agent_uids=_TRAINING_AGENT_UIDS, episode_length=10, root_seed=0
        )
        assert isinstance(adapter.observation_space, gym.spaces.Dict)
        agent_space = adapter.observation_space.spaces["agent"]
        assert isinstance(agent_space, gym.spaces.Dict)
        ego_obs_space = agent_space.spaces["panda_wristcam"]
        assert isinstance(ego_obs_space, gym.spaces.Dict)
        ego_state_box = ego_obs_space.spaces["state"]
        assert isinstance(ego_state_box, gym.spaces.Box)
        # FakeMultiAgentEnv's "state" channel has shape (3,).
        assert ego_state_box.shape == (3,)

    def test_satisfies_envlike_protocol(self) -> None:
        """T4b.11: the adapter satisfies the runtime-checkable EnvLike Protocol."""
        inner = FakeMultiAgentEnv(agent_uids=_SMOKE_ROBOT_UIDS)
        adapter = _Stage0TrainingAdapter(
            inner, agent_uids=_TRAINING_AGENT_UIDS, episode_length=10, root_seed=0
        )
        assert isinstance(adapter, EnvLike)


class TestActionAugmentation:
    def test_step_injects_zero_action_for_missing_inner_uids(self) -> None:
        """plan/05 §3.4: inner uids not driven by the trainer receive zero actions."""
        inner = FakeMultiAgentEnv(agent_uids=_SMOKE_ROBOT_UIDS)
        adapter = _Stage0TrainingAdapter(
            inner, agent_uids=_TRAINING_AGENT_UIDS, episode_length=10, root_seed=0
        )
        adapter.reset(seed=0)
        action = {
            "panda_wristcam": np.array([0.5, -0.5], dtype=np.float32),
            "fetch": np.array([0.1, 0.2], dtype=np.float32),
        }
        adapter.step(action)
        # Inner saw all three uids; the un-driven one got zero.
        last_action = inner._actions_received[-1]
        assert set(last_action.keys()) == set(_SMOKE_ROBOT_UIDS)
        np.testing.assert_array_equal(
            last_action["allegro_hand_right"], np.zeros(2, dtype=np.float32)
        )
        # Driven uids passed through unchanged.
        np.testing.assert_array_equal(last_action["panda_wristcam"], action["panda_wristcam"])
        np.testing.assert_array_equal(last_action["fetch"], action["fetch"])

    def test_missing_training_uid_action_raises(self) -> None:
        """Defensive: a step() missing one of the training uids must fail loudly."""
        inner = FakeMultiAgentEnv(agent_uids=_SMOKE_ROBOT_UIDS)
        adapter = _Stage0TrainingAdapter(
            inner, agent_uids=_TRAINING_AGENT_UIDS, episode_length=10, root_seed=0
        )
        adapter.reset(seed=0)
        with pytest.raises(ValueError, match="missing uid"):
            adapter.step({"panda_wristcam": np.zeros(2, dtype=np.float32)})


class TestTruncationHorizon:
    def test_step_count_truncation_fires_at_episode_length(self) -> None:
        """ADR-001 §Validation criteria: adapter enforces an episode_length boundary."""
        inner = FakeMultiAgentEnv(agent_uids=_SMOKE_ROBOT_UIDS)
        adapter = _Stage0TrainingAdapter(
            inner, agent_uids=_TRAINING_AGENT_UIDS, episode_length=3, root_seed=0
        )
        adapter.reset(seed=0)
        zero_action = {uid: np.zeros(2, dtype=np.float32) for uid in _TRAINING_AGENT_UIDS}
        _, _, _, trunc_1, _ = adapter.step(zero_action)
        assert trunc_1 is False
        _, _, _, trunc_2, _ = adapter.step(zero_action)
        assert trunc_2 is False
        _, _, _, trunc_3, _ = adapter.step(zero_action)
        assert trunc_3 is True

    def test_reset_resets_step_counter(self) -> None:
        """Defensive: reset() must clear the truncation counter."""
        inner = FakeMultiAgentEnv(agent_uids=_SMOKE_ROBOT_UIDS)
        adapter = _Stage0TrainingAdapter(
            inner, agent_uids=_TRAINING_AGENT_UIDS, episode_length=2, root_seed=0
        )
        adapter.reset(seed=0)
        zero_action = {uid: np.zeros(2, dtype=np.float32) for uid in _TRAINING_AGENT_UIDS}
        adapter.step(zero_action)
        _, _, _, truncated_at_2, _ = adapter.step(zero_action)
        assert truncated_at_2 is True
        adapter.reset(seed=1)
        _, _, _, truncated_at_1_after_reset, _ = adapter.step(zero_action)
        assert truncated_at_1_after_reset is False


class TestStateSynthesis:
    """Inner env without a ``state`` channel — adapter synthesises one."""

    @staticmethod
    def _build_inner_without_state() -> gym.Env[dict, dict]:  # type: ignore[type-arg]
        """Fake env exposing 1-D Box channels but no ``state`` key."""

        class _NoStateFake(gym.Env):  # type: ignore[type-arg]
            metadata: ClassVar[dict[str, list[str]]] = {"render_modes": []}  # type: ignore[misc]

            def __init__(self) -> None:
                self.action_space = gym.spaces.Dict(
                    {
                        uid: gym.spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)
                        for uid in _SMOKE_ROBOT_UIDS
                    }
                )
                self.observation_space = gym.spaces.Dict(
                    {
                        "agent": gym.spaces.Dict(
                            {
                                uid: gym.spaces.Dict(
                                    {
                                        "joint_pos": gym.spaces.Box(
                                            -1.0, 1.0, (4,), dtype=np.float32
                                        ),
                                        "joint_vel": gym.spaces.Box(
                                            -1.0, 1.0, (4,), dtype=np.float32
                                        ),
                                    }
                                )
                                for uid in _SMOKE_ROBOT_UIDS
                            }
                        )
                    }
                )

            def reset(  # type: ignore[override]
                self, *, seed: int | None = None, options: dict | None = None
            ) -> tuple[dict, dict]:
                del seed, options
                obs = {
                    "agent": {
                        uid: {
                            "joint_pos": np.arange(4, dtype=np.float32),
                            "joint_vel": np.arange(4, 8, dtype=np.float32),
                        }
                        for uid in _SMOKE_ROBOT_UIDS
                    }
                }
                return obs, {}

            def step(  # type: ignore[override]
                self, action: dict
            ) -> tuple[dict, float, bool, bool, dict]:
                del action
                obs = {
                    "agent": {
                        uid: {
                            "joint_pos": np.ones(4, dtype=np.float32),
                            "joint_vel": -np.ones(4, dtype=np.float32),
                        }
                        for uid in _SMOKE_ROBOT_UIDS
                    }
                }
                return obs, 0.0, False, False, {}

        return _NoStateFake()

    def test_synthesised_state_box_is_concat_of_1d_channels(self) -> None:
        """ADR-001 §Validation criteria: synthesised ``state`` = concat(sorted 1-D Box channels)."""
        inner = self._build_inner_without_state()
        adapter = _Stage0TrainingAdapter(
            inner, agent_uids=_TRAINING_AGENT_UIDS, episode_length=10, root_seed=0
        )
        assert isinstance(adapter.observation_space, gym.spaces.Dict)
        agent_space = adapter.observation_space.spaces["agent"]
        assert isinstance(agent_space, gym.spaces.Dict)
        ego_obs_space = agent_space.spaces["panda_wristcam"]
        assert isinstance(ego_obs_space, gym.spaces.Dict)
        ego_state_box = ego_obs_space.spaces["state"]
        assert isinstance(ego_state_box, gym.spaces.Box)
        # joint_pos(4,) + joint_vel(4,) sorted alphabetically → state(8,).
        assert ego_state_box.shape == (8,)

    def test_synthesised_state_vector_at_step(self) -> None:
        """ADR-001 §Validation criteria: per-step ``state`` is concat(joint_pos, joint_vel)."""
        inner = self._build_inner_without_state()
        adapter = _Stage0TrainingAdapter(
            inner, agent_uids=_TRAINING_AGENT_UIDS, episode_length=10, root_seed=0
        )
        obs, _ = adapter.reset(seed=0)
        state = obs["agent"]["panda_wristcam"]["state"]
        # reset() obs: joint_pos=[0,1,2,3], joint_vel=[4,5,6,7]; alphabetised → [0..7].
        np.testing.assert_array_equal(state, np.arange(8, dtype=np.float32))


class TestValidationErrors:
    def test_rejects_uid_not_in_inner_action_space(self) -> None:
        """Defensive: agent_uids must be a subset of the inner env's uids."""
        inner = FakeMultiAgentEnv(agent_uids=_SMOKE_ROBOT_UIDS)
        with pytest.raises(ValueError, match="not in inner env uids"):
            _Stage0TrainingAdapter(
                inner, agent_uids=("not_a_real_robot",), episode_length=10, root_seed=0
            )

    def test_rejects_empty_agent_uids(self) -> None:
        """Defensive: at least one uid (the ego) is required."""
        inner = FakeMultiAgentEnv(agent_uids=_SMOKE_ROBOT_UIDS)
        with pytest.raises(ValueError, match="agent_uids must have at least one"):
            _Stage0TrainingAdapter(inner, agent_uids=(), episode_length=10, root_seed=0)

    def test_make_stage0_training_env_rejects_uid_not_in_smoke_set(self) -> None:
        """Defensive: training agent_uids must be a subset of _SMOKE_ROBOT_UIDS."""
        with pytest.raises(ValueError, match="_SMOKE_ROBOT_UIDS"):
            make_stage0_training_env(
                agent_uids=("not_a_real_robot",), episode_length=10, root_seed=0
            )


@pytest.mark.skipif(
    sapien_gpu_available(),
    reason="Vulkan is available; CPU-only error path not reachable here",
)
def test_make_stage0_training_env_raises_compat_error_without_gpu() -> None:
    """ADR-001 §Risks: make_stage0_training_env surfaces the Vulkan-missing error."""
    with pytest.raises(ChamberEnvCompatibilityError, match="SAPIEN/Vulkan"):
        make_stage0_training_env(
            agent_uids=_TRAINING_AGENT_UIDS,
            episode_length=DEFAULT_EPISODE_LENGTH,
            root_seed=0,
        )
