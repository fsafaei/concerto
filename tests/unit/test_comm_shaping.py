# SPDX-License-Identifier: Apache-2.0
"""Unit tests for CommShapingWrapper.

Covers: obs["comm"] present post-reset and post-step; stub channel returns {};
observation space contains 'comm' key; incompatible obs space raises error.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

from chamber.comm import FixedFormatCommChannel
from chamber.envs.comm_shaping import CommShapingWrapper
from chamber.envs.errors import ChamberEnvCompatibilityError
from tests.fakes import FakeMultiAgentEnv


def _make_wrapper(uids: tuple[str, ...] = ("a", "b")) -> CommShapingWrapper:
    inner = FakeMultiAgentEnv(agent_uids=uids)
    channel = FixedFormatCommChannel()
    return CommShapingWrapper(inner, channel=channel)


class TestCommKeyPresent:
    def test_comm_key_in_reset_obs(self) -> None:
        """obs['comm'] must exist after reset() (ADR-001 cond. b)."""
        wrapper = _make_wrapper()
        obs, _ = wrapper.reset(seed=0)
        assert "comm" in obs

    def test_comm_key_in_step_obs(self) -> None:
        """obs['comm'] must exist after step() (ADR-001 cond. b)."""
        wrapper = _make_wrapper()
        wrapper.reset(seed=0)
        obs, *_ = wrapper.step({"a": np.array([0.0, 0.0]), "b": np.array([0.0, 0.0])})
        assert "comm" in obs

    def test_stub_channel_emits_schema_versioned_packet(self) -> None:
        """M2 T2.1 stub channel emits a CommPacket carrying SCHEMA_VERSION (ADR-003 §Decision)."""
        from chamber.comm import SCHEMA_VERSION

        wrapper = _make_wrapper()
        obs, _ = wrapper.reset(seed=0)
        comm = obs["comm"]
        assert comm["schema_version"] == SCHEMA_VERSION
        expected_keys = {"schema_version", "pose", "task_state", "aoi", "learned_overlay"}
        assert set(comm.keys()) == expected_keys


class TestObservationSpace:
    def test_observation_space_has_comm_key(self) -> None:
        """Wrapped observation space includes 'comm' (ADR-001 §Decision item 3)."""
        wrapper = _make_wrapper()
        assert "comm" in wrapper.observation_space.spaces  # type: ignore[attr-defined]

    def test_original_keys_preserved(self) -> None:
        """Original obs space keys are preserved after wrapping."""
        wrapper = _make_wrapper()
        assert "agent" in wrapper.observation_space.spaces  # type: ignore[attr-defined]


class TestChannelReset:
    def test_channel_reset_called_on_env_reset(self) -> None:
        """channel.reset() is invoked when the env resets (ADR-003 §Decision)."""
        from chamber.comm import SCHEMA_VERSION, CommPacket

        inner = FakeMultiAgentEnv()

        class _TrackingChannel:
            reset_count = 0

            def reset(self, *, seed: int | None = None) -> None:
                del seed
                self.__class__.reset_count += 1

            def encode(self, state: object) -> CommPacket:
                del state
                return CommPacket(
                    schema_version=SCHEMA_VERSION,
                    pose={},
                    task_state={},
                    aoi={},
                    learned_overlay=None,
                )

            def decode(self, packet: CommPacket) -> dict[str, object]:
                del packet
                return {}

        wrapper = CommShapingWrapper(inner, channel=_TrackingChannel())
        wrapper.reset(seed=0)
        wrapper.reset(seed=1)
        assert _TrackingChannel.reset_count == 2


class TestValidation:
    def test_rejects_non_dict_observation_space(self) -> None:
        """Wrapping env with non-Dict obs space raises ChamberEnvCompatibilityError."""

        class _BoxObsEnv(gym.Env):  # type: ignore[type-arg]
            action_space = gym.spaces.Box(-1, 1, (2,))
            observation_space = gym.spaces.Box(0, 1, (4,))

            def reset(self, **kw):
                return np.zeros(4), {}  # type: ignore[override]

            def step(self, action):
                return np.zeros(4), 0.0, False, False, {}  # type: ignore[override]

        with pytest.raises(ChamberEnvCompatibilityError):
            CommShapingWrapper(_BoxObsEnv(), channel=FixedFormatCommChannel())
