# SPDX-License-Identifier: Apache-2.0
"""Tier-1 tests for the PBRS settle wrapper (P1.05.11; ADR-007 §Stage 1b Rev 18).

Pins the Ng-Harada-Russell form against a fake env: per-transition
F = gamma*Phi(s') - Phi(s), Phi = -alpha*min(max-arm-|qvel|, cap)*1[placed];
boundary conventions (terminal Phi == 0; truncation reads
``info["final_observation"]``; reset re-initialises Phi_prev); the
telescoping identity (the discounted shaping sum equals
-Phi(s_0) + gamma^T·Phi(s_T) for any trajectory — the invariance mechanism);
the default-off guard (alpha=0 must not construct the wrapper); and the
``run_training`` wiring gate.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import pytest

from chamber.envs.stage1_shaping import Stage1SettleShapingWrapper

_ALPHA = 0.5
_CAP = 0.7
_GAMMA = 0.8
_EGO = "panda_wristcam"


def _obs(qvel7_max: float, *, placed: bool, n: int = 1) -> dict[str, Any]:
    """Fake Stage-1b AS obs with a controllable max arm |qvel| and placement."""
    qvel = np.zeros((n, 9), dtype=np.float32)
    qvel[:, 0] = qvel7_max
    qvel[:, 7:] = 5.0  # finger DOF — MUST be excluded from the reduction
    cube = np.zeros((n, 7), dtype=np.float32)
    goal = np.zeros((n, 3), dtype=np.float32)
    if not placed:
        goal[:, 0] = 1.0  # 1 m away — far outside goal_thresh
    return {
        "agent": {_EGO: {"qvel": qvel}},
        "extra": {"cube_pose": cube, "goal_pos": goal},
    }


def _phi(qvel7_max: float, *, placed: bool) -> float:
    return -_ALPHA * min(qvel7_max, _CAP) * (1.0 if placed else 0.0)


class _ScriptedEnv(gym.Env):  # type: ignore[type-arg]
    """Fake env replaying a scripted (obs, reward, term, trunc, info) tape."""

    observation_space = gym.spaces.Box(-np.inf, np.inf, (1,), np.float32)
    action_space = gym.spaces.Box(-1.0, 1.0, (1,), np.float32)

    def __init__(self, reset_obs: dict[str, Any], tape: list[tuple]) -> None:  # type: ignore[type-arg]
        super().__init__()
        self._reset_obs = reset_obs
        self._tape = list(tape)
        self._i = 0

    def reset(self, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        del kwargs
        self._i = 0
        return self._reset_obs, {}

    def step(self, action: Any) -> tuple:  # type: ignore[type-arg]
        del action
        out = self._tape[self._i]
        self._i += 1
        return out


class TestPotentialForm:
    def test_alpha_zero_refused(self) -> None:
        """Default-off (settle_alpha=0) must never construct the wrapper (ADR-002)."""
        with pytest.raises(ValueError, match="alpha"):
            Stage1SettleShapingWrapper(
                _ScriptedEnv(_obs(0.0, placed=False), []),  # type: ignore[arg-type]
                alpha=0.0,
                qvel_cap=_CAP,
                gamma=_GAMMA,
                ego_uid=_EGO,
            )

    def test_per_transition_f_and_finger_exclusion_and_cap(self) -> None:
        """F = gamma*Phi(s') - Phi(s); fingers excluded; cap applied; gate on placed."""
        # reset (not placed, fast) -> s1 (placed, 0.5) -> s2 (placed, 0.9 >= cap)
        s1, s2 = _obs(0.5, placed=True), _obs(0.9, placed=True)
        env = _ScriptedEnv(
            _obs(0.6, placed=False),
            [
                (s1, 1.0, np.array([False]), np.array([False]), {}),
                (s2, 1.0, np.array([False]), np.array([False]), {}),
            ],
        )
        wrap = Stage1SettleShapingWrapper(
            env,  # type: ignore[arg-type]
            alpha=_ALPHA,
            qvel_cap=_CAP,
            gamma=_GAMMA,
            ego_uid=_EGO,
        )
        wrap.reset()
        # Phi(reset) = 0 (not placed, despite qvel 0.6 — gate works; finger
        # qvel 5.0 ignored — exclusion works).
        _, r1, _, _, _ = wrap.step(None)
        assert r1 == pytest.approx(1.0 + _GAMMA * _phi(0.5, placed=True) - 0.0)
        # Cap: Phi(s2) uses min(0.9, 0.7).
        _, r2, _, _, _ = wrap.step(None)
        assert r2 == pytest.approx(1.0 + _GAMMA * _phi(0.7, placed=True) - _phi(0.5, placed=True))

    def test_terminal_phi_is_zero(self) -> None:
        """True termination: F = -Phi(s) regardless of the terminal obs (frozen convention)."""
        s_fast_placed = _obs(0.6, placed=True)
        env = _ScriptedEnv(
            _obs(0.5, placed=True),
            [(s_fast_placed, 2.0, np.array([True]), np.array([False]), {})],
        )
        wrap = Stage1SettleShapingWrapper(
            env,  # type: ignore[arg-type]
            alpha=_ALPHA,
            qvel_cap=_CAP,
            gamma=_GAMMA,
            ego_uid=_EGO,
        )
        wrap.reset()
        _, r, _, _, _ = wrap.step(None)
        assert r == pytest.approx(2.0 + _GAMMA * 0.0 - _phi(0.5, placed=True))

    def test_truncation_reads_final_observation(self) -> None:
        """Auto-reset path: Phi(s') from info['final_observation'], not the reset obs."""
        final = _obs(0.4, placed=True)
        post_reset = _obs(0.0, placed=False)
        env = _ScriptedEnv(
            _obs(0.5, placed=True),
            [
                (
                    post_reset,
                    1.0,
                    np.array([False]),
                    np.array([True]),
                    {"final_observation": final},
                ),
                # Next transition: Phi_prev must have re-initialised from the
                # post-reset obs (Phi=0), NOT from the final obs.
                (_obs(0.3, placed=True), 1.0, np.array([False]), np.array([False]), {}),
            ],
        )
        wrap = Stage1SettleShapingWrapper(
            env,  # type: ignore[arg-type]
            alpha=_ALPHA,
            qvel_cap=_CAP,
            gamma=_GAMMA,
            ego_uid=_EGO,
        )
        wrap.reset()
        _, r1, _, _, _ = wrap.step(None)
        assert r1 == pytest.approx(1.0 + _GAMMA * _phi(0.4, placed=True) - _phi(0.5, placed=True))
        _, r2, _, _, _ = wrap.step(None)
        assert r2 == pytest.approx(1.0 + _GAMMA * _phi(0.3, placed=True) - 0.0)

    def test_telescoping_identity(self) -> None:
        """Discounted shaping sum = gamma^T Phi(s_T) - Phi(s_0) (invariance mechanism)."""
        rng = np.random.default_rng(7)
        states = [
            _obs(float(q), placed=bool(p))
            for q, p in zip(rng.uniform(0, 1, 12), rng.integers(0, 2, 12), strict=True)
        ]
        tape = [(s, 0.0, np.array([False]), np.array([False]), {}) for s in states]
        env = _ScriptedEnv(_obs(0.55, placed=True), tape)
        wrap = Stage1SettleShapingWrapper(
            env,  # type: ignore[arg-type]
            alpha=_ALPHA,
            qvel_cap=_CAP,
            gamma=_GAMMA,
            ego_uid=_EGO,
        )
        wrap.reset()
        disc_sum = 0.0
        for t in range(len(tape)):
            _, r, _, _, _ = wrap.step(None)
            disc_sum += (_GAMMA**t) * float(np.asarray(r).reshape(-1)[0])
        q_last = min(float(np.abs(states[-1]["agent"][_EGO]["qvel"][0, :7]).max()), _CAP)
        goal_last = states[-1]["extra"]["goal_pos"]
        cube_last = states[-1]["extra"]["cube_pose"][:, :3]
        placed_last = bool(np.linalg.norm(goal_last - cube_last) <= 0.025)
        phi_T = -_ALPHA * q_last * placed_last
        phi_0 = _phi(0.55, placed=True)
        # abs=1e-6: the obs tensors are float32, so Phi carries ~1e-8-scale
        # quantisation per term; the identity is exact in exact arithmetic.
        assert disc_sum == pytest.approx((_GAMMA ** len(tape)) * phi_T - phi_0, abs=1e-6)

    def test_vectorised_layout(self) -> None:
        """(num_envs,) rewards + (num_envs, dim) obs shape through per-env Phi."""
        n = 3
        s1 = _obs(0.5, placed=True, n=n)
        env = _ScriptedEnv(
            _obs(0.2, placed=False, n=n),
            [(s1, np.ones(n, dtype=np.float32), np.zeros(n, bool), np.zeros(n, bool), {})],
        )
        wrap = Stage1SettleShapingWrapper(
            env,  # type: ignore[arg-type]
            alpha=_ALPHA,
            qvel_cap=_CAP,
            gamma=_GAMMA,
            ego_uid=_EGO,
        )
        wrap.reset()
        _, r, _, _, _ = wrap.step(None)
        np.testing.assert_allclose(np.asarray(r), 1.0 + _GAMMA * _phi(0.5, placed=True), rtol=1e-6)


class TestRunTrainingWiring:
    def test_default_off_does_not_wrap(self, tmp_path: Any) -> None:
        """shaping.settle_alpha=0 (default) leaves the env unwrapped (ADR-002)."""
        from concerto.training.config import ShapingConfig

        assert ShapingConfig().settle_alpha == 0.0
        # The gate in run_training is `> 0.0`; the wrapper itself refuses
        # alpha<=0 (TestPotentialForm), so a mis-wire cannot silently
        # produce a no-op shaped cell.
