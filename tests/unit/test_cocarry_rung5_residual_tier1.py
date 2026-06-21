# SPDX-License-Identifier: Apache-2.0
"""Tier-1 (no-SAPIEN) tests for the Rung-5 residual-on-impedance ego wrapper.

Pins the escalation contract on a fake inner env: the ego seat's APPLIED action
is clip(base(obs) + residual, -1, 1), the partner seat passes through untouched,
and the frozen base is reset with the episode seed. ADR-026 §Decision 4.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from chamber.envs.cocarry_obs import CoCarryResidualBaseEgoWrapper


class _FakeInner(gym.Env):  # type: ignore[type-arg]
    """Minimal 2-agent env recording the action it is stepped with."""

    def __init__(self) -> None:
        self.action_space = gym.spaces.Dict(
            {"ego": gym.spaces.Box(-1.0, 1.0, (8,)), "p": gym.spaces.Box(-1.0, 1.0, (8,))}
        )
        self.observation_space = gym.spaces.Dict({})
        self.received: Any = None

    def reset(self, *, seed: int | None = None, options: Any = None) -> tuple[Any, dict[str, Any]]:
        del seed, options
        return {"obs": 0}, {}

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        self.received = action
        return {"obs": 1}, 0.0, False, False, {}


class _FakeBase:
    """A frozen base controller emitting a fixed action."""

    def __init__(self, a: list[float]) -> None:
        self._a = np.asarray(a, dtype=np.float32)
        self.reset_seed: int | None = -1

    def reset(self, *, seed: int | None = None) -> None:
        self.reset_seed = seed

    def act(self, obs: Any, *, deterministic: bool = True) -> np.ndarray:
        del obs, deterministic
        return self._a


def _wrap(base_a: list[float]) -> tuple[CoCarryResidualBaseEgoWrapper, _FakeInner, _FakeBase]:
    inner = _FakeInner()
    base = _FakeBase(base_a)
    return CoCarryResidualBaseEgoWrapper(inner, base_ego=base, ego_uid="ego"), inner, base


class TestResidualBaseEgoWrapper:
    def test_reset_seeds_the_base(self) -> None:
        w, _inner, base = _wrap([0.5] * 8)
        w.reset(seed=7)
        assert base.reset_seed == 7

    def test_ego_action_is_base_plus_residual(self) -> None:
        w, inner, _base = _wrap([0.5] * 8)
        w.reset(seed=0)
        w.step({"ego": np.full(8, 0.3, dtype=np.float32), "p": np.full(8, 0.1, dtype=np.float32)})
        np.testing.assert_allclose(inner.received["ego"], np.full(8, 0.8), atol=1e-6)

    def test_ego_action_is_clipped_to_unit_box(self) -> None:
        w, inner, _base = _wrap([0.5] * 8)
        w.reset(seed=0)
        # 0.5 + 0.8 = 1.3 -> clipped to 1.0; -0.5 side: 0.5 + (-1.9) = -1.4 -> -1.0
        w.step({"ego": np.full(8, 0.8, dtype=np.float32), "p": np.zeros(8, dtype=np.float32)})
        np.testing.assert_allclose(inner.received["ego"], np.full(8, 1.0), atol=1e-6)

    def test_partner_action_passes_through(self) -> None:
        w, inner, _base = _wrap([0.0] * 8)
        w.reset(seed=0)
        partner = np.linspace(-1.0, 1.0, 8, dtype=np.float32)
        w.step({"ego": np.zeros(8, dtype=np.float32), "p": partner})
        np.testing.assert_array_equal(inner.received["p"], partner)

    def test_zero_residual_reproduces_the_base(self) -> None:
        # The "starts near base" property: a zero residual applies the base exactly.
        w, inner, _base = _wrap([0.42] * 8)
        w.reset(seed=0)
        w.step({"ego": np.zeros(8, dtype=np.float32), "p": np.zeros(8, dtype=np.float32)})
        np.testing.assert_allclose(inner.received["ego"], np.full(8, 0.42), atol=1e-6)
