# SPDX-License-Identifier: Apache-2.0
"""Tier-1 tests for :class:`chamber.envs.Stage0StateSynthesizer` (closes #184).

Pins the synthesis contract the Stage-0 factory now relies on:

- Per-uid ``state`` Box is injected when missing, computed as the
  alphabetical concatenation of 1-D Box channels (matches
  :func:`chamber.benchmarks.stage0_smoke_adapter._synthesised_state_space`
  so the training adapter remains a pass-through).
- Uids that already expose ``state`` are passed through untouched.
- ``num_envs=1`` vectorised shapes ``(1, n)`` are accepted and ravelled.
- Top-level non-``agent`` keys (``comm``, ``meta``, ``extra``) flow
  through unchanged.
- The wrapper rejects malformed observation / action spaces at
  construction.
"""

from __future__ import annotations

from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
import pytest

from chamber.envs import Stage0StateSynthesizer
from chamber.envs.errors import ChamberEnvCompatibilityError
from tests.fakes import FakeMultiAgentEnv


class _PerUidChannelsEnv(gym.Env):  # type: ignore[type-arg]
    """Stage-0-like env with per-uid Dict channels and NO top-level ``state``.

    Mirrors the shape :func:`chamber.benchmarks.stage0_smoke.make_stage0_env`
    produces under ``obs_mode="state_dict"`` after
    :class:`chamber.envs.TextureFilterObsWrapper` strips visual channels:
    one uid with ``joint_pos`` + ``joint_vel`` 1-D Boxes (needs synth),
    one uid that already has ``state`` (pass-through), one uid carrying a
    ``(1, n)`` vectorised ``joint_pos`` shape (the ManiSkill ``num_envs=1``
    rank-2 case the helper relaxes).
    """

    metadata: ClassVar[dict[str, object]] = {"render_modes": []}  # type: ignore[misc]

    def __init__(self) -> None:
        super().__init__()
        self.action_space = gym.spaces.Dict(
            {
                "needs_synth": gym.spaces.Box(-1.0, 1.0, (2,), np.float32),
                "has_state": gym.spaces.Box(-1.0, 1.0, (3,), np.float32),
                "vectorised": gym.spaces.Box(-1.0, 1.0, (4,), np.float32),
            }
        )
        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Dict(
                    {
                        "needs_synth": gym.spaces.Dict(
                            {
                                "joint_pos": gym.spaces.Box(-1.0, 1.0, (5,), np.float32),
                                "joint_vel": gym.spaces.Box(-1.0, 1.0, (5,), np.float32),
                            }
                        ),
                        "has_state": gym.spaces.Dict(
                            {
                                "state": gym.spaces.Box(-1.0, 1.0, (7,), np.float32),
                                "joint_pos": gym.spaces.Box(-1.0, 1.0, (3,), np.float32),
                            }
                        ),
                        "vectorised": gym.spaces.Dict(
                            {
                                "joint_pos": gym.spaces.Box(-1.0, 1.0, (1, 4), np.float32),
                            }
                        ),
                    }
                ),
                "comm": gym.spaces.Box(-1.0, 1.0, (2,), np.float32),
            }
        )

    def _obs(self) -> dict[str, Any]:
        return {
            "agent": {
                "needs_synth": {
                    "joint_pos": np.arange(5, dtype=np.float32),
                    "joint_vel": np.full(5, 10.0, dtype=np.float32),
                },
                "has_state": {
                    "state": np.full(7, 0.5, dtype=np.float32),
                    "joint_pos": np.full(3, 0.25, dtype=np.float32),
                },
                "vectorised": {
                    "joint_pos": np.arange(4, dtype=np.float32).reshape(1, 4),
                },
            },
            "comm": np.zeros(2, dtype=np.float32),
        }

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]:
        super().reset(seed=seed)
        return self._obs(), {}

    def step(self, action: dict) -> tuple[dict, float, bool, bool, dict]:
        del action
        return self._obs(), 0.0, False, False, {}


class TestStage0StateSynthesizer:
    """Pin the per-uid state-injection contract (ADR-001 §Validation criteria; #184)."""

    def test_pass_through_when_state_already_present(self) -> None:
        """A uid that already has ``state`` is left untouched (matches adapter policy)."""
        inner = FakeMultiAgentEnv(agent_uids=("a", "b"))
        wrapped = Stage0StateSynthesizer(inner)
        # FakeMultiAgentEnv already exposes per-uid ``state`` Boxes, so the
        # wrapper should not register any uid for synthesis.
        assert wrapped._synthesise_state_for == set()
        assert wrapped.observation_space == inner.observation_space
        obs, _ = wrapped.reset(seed=0)
        for uid in ("a", "b"):
            assert obs["agent"][uid]["state"].shape == (3,)
            # Identity pass-through on the state vector itself.
            np.testing.assert_array_equal(obs["agent"][uid]["state"], np.zeros(3, dtype=np.float32))

    def test_synthesises_state_for_uid_without_state_key(self) -> None:
        """Missing ``state`` is concatenated from 1-D Box channels in alphabetical order."""
        inner = _PerUidChannelsEnv()
        wrapped = Stage0StateSynthesizer(inner)
        assert "needs_synth" in wrapped._synthesise_state_for
        # Alphabetical key order: joint_pos, then joint_vel.
        assert wrapped._state_synthesis_keys["needs_synth"] == ("joint_pos", "joint_vel")
        obs_space = wrapped.observation_space
        assert isinstance(obs_space, gym.spaces.Dict)
        agent_space = obs_space.spaces["agent"]
        assert isinstance(agent_space, gym.spaces.Dict)
        uid_space = agent_space.spaces["needs_synth"]
        assert isinstance(uid_space, gym.spaces.Dict)
        synth_space = uid_space.spaces["state"]
        assert isinstance(synth_space, gym.spaces.Box)
        assert synth_space.shape == (10,)  # 5 joint_pos + 5 joint_vel
        obs, _ = wrapped.reset(seed=0)
        # joint_pos = arange(5) = [0,1,2,3,4]; joint_vel = full(5, 10.0).
        expected = np.concatenate(
            [np.arange(5, dtype=np.float32), np.full(5, 10.0, dtype=np.float32)]
        )
        np.testing.assert_array_equal(obs["agent"]["needs_synth"]["state"], expected)

    def test_pass_through_per_uid_when_state_present_on_one_uid_only(self) -> None:
        """Mixed envs: only uids without ``state`` get synthesis."""
        inner = _PerUidChannelsEnv()
        wrapped = Stage0StateSynthesizer(inner)
        # has_state already had a 7-D state Box; the wrapper should not have
        # registered it for synthesis.
        assert "has_state" not in wrapped._synthesise_state_for
        obs_space = wrapped.observation_space
        assert isinstance(obs_space, gym.spaces.Dict)
        agent_space = obs_space.spaces["agent"]
        assert isinstance(agent_space, gym.spaces.Dict)
        uid_space = agent_space.spaces["has_state"]
        assert isinstance(uid_space, gym.spaces.Dict)
        has_state_space = uid_space.spaces["state"]
        assert isinstance(has_state_space, gym.spaces.Box)
        assert has_state_space.shape == (7,)
        obs, _ = wrapped.reset(seed=0)
        np.testing.assert_array_equal(
            obs["agent"]["has_state"]["state"], np.full(7, 0.5, dtype=np.float32)
        )

    def test_vectorised_num_envs_one_shape_is_ravelled(self) -> None:
        """``(1, n)`` per-channel shapes are squeezed and ravelled into the state Box."""
        inner = _PerUidChannelsEnv()
        wrapped = Stage0StateSynthesizer(inner)
        assert "vectorised" in wrapped._synthesise_state_for
        obs_space = wrapped.observation_space
        assert isinstance(obs_space, gym.spaces.Dict)
        agent_space = obs_space.spaces["agent"]
        assert isinstance(agent_space, gym.spaces.Dict)
        uid_space = agent_space.spaces["vectorised"]
        assert isinstance(uid_space, gym.spaces.Dict)
        synth_space = uid_space.spaces["state"]
        assert isinstance(synth_space, gym.spaces.Box)
        # 4-D joint_pos (vectorised (1,4) → squeezed (4,)).
        assert synth_space.shape == (4,)
        obs, _ = wrapped.reset(seed=0)
        np.testing.assert_array_equal(
            obs["agent"]["vectorised"]["state"], np.arange(4, dtype=np.float32)
        )

    def test_non_agent_top_level_keys_pass_through(self) -> None:
        """``comm`` (and any other top-level key) flow through untouched."""
        inner = _PerUidChannelsEnv()
        wrapped = Stage0StateSynthesizer(inner)
        obs, _ = wrapped.reset(seed=0)
        assert "comm" in obs
        np.testing.assert_array_equal(obs["comm"], np.zeros(2, dtype=np.float32))

    def test_observation_space_dict_invariant(self) -> None:
        """The wrapper's ``observation_space`` is a Dict containing ``agent``."""
        inner = _PerUidChannelsEnv()
        wrapped = Stage0StateSynthesizer(inner)
        assert isinstance(wrapped.observation_space, gym.spaces.Dict)
        assert "agent" in wrapped.observation_space.spaces


class TestStage0StateSynthesizerRejectsMalformedSpaces:
    """Construction-time validation (ADR-001 §Validation criteria)."""

    def test_rejects_non_dict_observation_space(self) -> None:
        class _BadObsEnv(gym.Env):  # type: ignore[type-arg]
            def __init__(self) -> None:
                super().__init__()
                self.observation_space = gym.spaces.Box(0.0, 1.0, (4,), np.float32)
                self.action_space = gym.spaces.Dict(
                    {"a": gym.spaces.Box(-1.0, 1.0, (2,), np.float32)}
                )

        with pytest.raises(TypeError, match=r"gym\.spaces\.Dict observation"):
            Stage0StateSynthesizer(_BadObsEnv())

    def test_rejects_non_dict_action_space(self) -> None:
        class _BadActionEnv(gym.Env):  # type: ignore[type-arg]
            def __init__(self) -> None:
                super().__init__()
                self.observation_space = gym.spaces.Dict(
                    {"agent": gym.spaces.Dict({"a": gym.spaces.Dict({})})}
                )
                self.action_space = gym.spaces.Box(0.0, 1.0, (4,), np.float32)

        with pytest.raises(TypeError, match=r"gym\.spaces\.Dict action space"):
            Stage0StateSynthesizer(_BadActionEnv())

    def test_rejects_uid_with_no_1d_box_channels(self) -> None:
        """A uid whose per-uid Dict has only image-shaped channels is unsynthesisable."""

        class _ImageOnlyEnv(gym.Env):  # type: ignore[type-arg]
            def __init__(self) -> None:
                super().__init__()
                self.action_space = gym.spaces.Dict(
                    {"a": gym.spaces.Box(-1.0, 1.0, (2,), np.float32)}
                )
                self.observation_space = gym.spaces.Dict(
                    {
                        "agent": gym.spaces.Dict(
                            {
                                "a": gym.spaces.Dict(
                                    {
                                        "rgb": gym.spaces.Box(0.0, 1.0, (4, 4, 3), np.float32),
                                    }
                                ),
                            }
                        ),
                    }
                )

        with pytest.raises(ChamberEnvCompatibilityError, match="no 1-D Box channels"):
            Stage0StateSynthesizer(_ImageOnlyEnv())
