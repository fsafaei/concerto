# SPDX-License-Identifier: Apache-2.0
"""Unit tests for PerAgentActionRepeatWrapper.

Covers: repeat=1 pass-through, repeat=5 action count, mid-episode reset,
missing uid defaults to 1, and incompatible action space rejection.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

from chamber.envs.action_repeat import PerAgentActionRepeatWrapper
from chamber.envs.errors import ChamberEnvCompatibilityError
from tests.fakes import FakeMultiAgentEnv


def _make_wrapper(
    repeat: dict[str, int], uids: tuple[str, ...] = ("a", "b")
) -> tuple[PerAgentActionRepeatWrapper, FakeMultiAgentEnv]:
    inner = FakeMultiAgentEnv(agent_uids=uids)
    wrapper = PerAgentActionRepeatWrapper(inner, action_repeat=repeat)
    wrapper.reset(seed=0)
    return wrapper, inner


class TestPassThrough:
    def test_repeat_one_every_action_dispatched(self) -> None:
        """repeat=1 → inner env receives every submitted action (ADR-001 §Decision item 2)."""
        wrapper, inner = _make_wrapper({"a": 1, "b": 1})
        n = 20
        for i in range(n):
            action = {
                "a": np.array([float(i), 0.0], dtype=np.float32),
                "b": np.array([0.0, 0.0], dtype=np.float32),
            }
            wrapper.step(action)
        received_a = [r["a"][0] for r in inner._actions_received]
        # Each step should propagate the submitted action unchanged.
        assert received_a == [float(i) for i in range(n)]


class TestRepeatCount:
    def test_repeat_5_exact_action_count(self) -> None:
        """repeat=5 → 20 distinct actions over 100 steps (ADR-001 cond. c)."""
        wrapper, inner = _make_wrapper({"a": 5, "b": 1})
        for i in range(100):
            action = {
                "a": np.array([float(i), 0.0], dtype=np.float32),
                "b": np.array([0.0, 0.0], dtype=np.float32),
            }
            wrapper.step(action)
        received_a = [r["a"][0] for r in inner._actions_received]
        # Count transitions (new action dispatched to inner env).
        transitions = 1 + sum(received_a[i] != received_a[i - 1] for i in range(1, 100))
        assert transitions == 20  # ceil(100 / 5)

    def test_repeat_10_exact_action_count(self) -> None:
        """repeat=10 → 10 distinct actions over 100 steps."""
        wrapper, inner = _make_wrapper({"a": 10, "b": 1})
        for i in range(100):
            action = {
                "a": np.array([float(i), 0.0], dtype=np.float32),
                "b": np.array([0.0, 0.0], dtype=np.float32),
            }
            wrapper.step(action)
        received_a = [r["a"][0] for r in inner._actions_received]
        transitions = 1 + sum(received_a[i] != received_a[i - 1] for i in range(1, 100))
        assert transitions == 10  # ceil(100 / 10)


class TestReset:
    def test_mid_episode_reset_clears_state(self) -> None:
        """reset() clears per-agent counters and last-action cache (ADR-001 §Decision item 2)."""
        wrapper, inner = _make_wrapper({"a": 5, "b": 1})
        # Step partway through a repeat interval.
        for i in range(3):
            wrapper.step(
                {
                    "a": np.array([99.0, 0.0], dtype=np.float32),
                    "b": np.array([0.0, 0.0], dtype=np.float32),
                }
            )
        # Reset should clear state.
        wrapper.reset(seed=1)
        # First step after reset must accept the new action (not the cached 99.0).
        new_action = {
            "a": np.array([7.0, 0.0], dtype=np.float32),
            "b": np.array([0.0, 0.0], dtype=np.float32),
        }
        wrapper.step(new_action)
        last_received_a = inner._actions_received[-1]["a"][0]
        assert last_received_a == 7.0

    def test_reset_return_value_passthrough(self) -> None:
        """reset() returns obs and info from the inner env."""
        wrapper, _ = _make_wrapper({"a": 2})
        obs, info = wrapper.reset(seed=42)
        assert "agent" in obs
        assert isinstance(info, dict)


class TestMissingUid:
    def test_uid_not_in_action_repeat_defaults_to_one(self) -> None:
        """Agent uid absent from action_repeat dict defaults to repeat=1."""
        # Only agent 'a' is registered; 'b' should pass through every step.
        wrapper, inner = _make_wrapper({"a": 5})
        for i in range(10):
            action = {
                "a": np.array([float(i), 0.0], dtype=np.float32),
                "b": np.array([float(i * 10), 0.0], dtype=np.float32),
            }
            wrapper.step(action)
        # 'b' should receive every submitted action (repeat=1 default).
        received_b = [r["b"][0] for r in inner._actions_received]
        assert received_b == [float(i * 10) for i in range(10)]


class TestValidation:
    def test_rejects_non_dict_action_space(self) -> None:
        """Wrapping an env with a non-Dict action space raises ChamberEnvCompatibilityError."""

        class _BoxEnv(gym.Env):  # type: ignore[type-arg]
            action_space = gym.spaces.Box(-1, 1, (4,))
            observation_space = gym.spaces.Box(0, 1, (2,))

            def reset(self, **kw):
                return np.zeros(2), {}  # type: ignore[override]

            def step(self, action):
                return np.zeros(2), 0.0, False, False, {}  # type: ignore[override]

        with pytest.raises(ChamberEnvCompatibilityError):
            PerAgentActionRepeatWrapper(_BoxEnv(), {"a": 2})

    def test_rejects_zero_repeat_count(self) -> None:
        inner = FakeMultiAgentEnv()
        with pytest.raises(ValueError, match="≥ 1"):
            PerAgentActionRepeatWrapper(inner, {"a": 0})
