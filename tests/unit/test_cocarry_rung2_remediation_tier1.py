# SPDX-License-Identifier: Apache-2.0
# pyright: reportPrivateImportUsage=false, reportPrivateUsage=false
"""Tier-1 (no-SAPIEN) tests for the Rung-2 reward remediation (ADR-026 §Decision 4).

COCARRY_RUNG2_REMEDIATION_2026-06-16. The Rung-2 train-to-reference STOP
showed the learned ego climbs the dense reward yet reaches 0% joint success
because it fights the frozen partner (stress ~6.5x the limit) and never
transports. This slice adds two principled, partner-agnostic terms; these
tests pin their shape + capture, no SAPIEN scene:

- the excess-internal-stress penalty is ~0 across the matched cooperative
  band, monotonic above the threshold, and saturates on the fight (the
  keystone — and the matched-pair invariance that keeps the 100% reference);
- the transport PBRS potential is a Φ-difference, Φ = -coeff·dist, vanishing
  at the goal, with the NHR boundary convention (terminal ⇒ Φ=0);
- the new coefficients are captured by the freeze enumerator (the
  completeness guard) and the training config's transport_pbrs_coeff equals
  the env constant (parity, no drift).

The learned-mechanism check (stress p90 trending down, success leaving 0) is
the GPU smoke / train run; reliable Tier-1 shape + capture lives here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pytest

import chamber.benchmarks.cocarry_freeze as freeze
from chamber.envs.cocarry import (
    COCARRY_REWARD_STRESS_COEFF,
    COCARRY_REWARD_STRESS_SOFT_THRESHOLD_N,
    COCARRY_REWARD_TRANSPORT_PBRS_COEFF,
    COCARRY_STRESS_MAX_PROXY_N,
    cocarry_excess_stress_penalty,
)
from chamber.envs.cocarry_shaping import CoCarryTransportPBRSWrapper

# Committed Step-1 matched cooperative band (cocarry_rung2_fmax_distribution.json):
# success-stress p50=92, p90=101, p99=104.5, max=105 N. The penalty must be
# negligible across this whole band so the 100% matched reference is unchanged.
_MATCHED_BAND_N = (92.0, 101.0, 104.5, 105.0)
_FIGHT_P90_N = 854.0


class TestStressPenaltyShape:
    """Excess-internal-stress penalty shape (keystone; ADR-026 §Decision 4)."""

    def test_zero_across_matched_cooperative_band(self) -> None:
        # The whole matched band sits below the threshold (= f_max 130 N),
        # so the cooperative pair incurs ~0 penalty (reference invariant).
        for s in _MATCHED_BAND_N:
            assert float(cocarry_excess_stress_penalty(np.array([s]))[0]) == pytest.approx(
                0.0, abs=1e-9
            )

    def test_zero_at_and_below_threshold(self) -> None:
        assert (
            float(cocarry_excess_stress_penalty(np.array([COCARRY_STRESS_MAX_PROXY_N]))[0]) == 0.0
        )
        assert float(cocarry_excess_stress_penalty(np.array([0.0]))[0]) == 0.0

    def test_monotonic_nondecreasing_in_stress(self) -> None:
        xs = np.linspace(0.0, 1000.0, 50)
        ys = cocarry_excess_stress_penalty(xs)
        assert np.all(np.diff(np.asarray(ys)) >= -1e-12)

    def test_saturates_near_coeff_on_the_fight(self) -> None:
        # The incumbent fight (p90 ~854 N, excess ~724 N) saturates tanh ~ coeff.
        pen = float(cocarry_excess_stress_penalty(np.array([_FIGHT_P90_N]))[0])
        assert pen > 0.9 * COCARRY_REWARD_STRESS_COEFF
        assert pen <= COCARRY_REWARD_STRESS_COEFF

    def test_threshold_is_the_success_ceiling(self) -> None:
        # Principled tie: the penalty bites exactly the stress that violates
        # the success constraint (COCARRY_STRESS_MAX_PROXY_N).
        assert COCARRY_REWARD_STRESS_SOFT_THRESHOLD_N == COCARRY_STRESS_MAX_PROXY_N

    def test_torch_backend_returns_torch(self) -> None:
        torch = pytest.importorskip("torch")
        out = cocarry_excess_stress_penalty(torch.tensor([_FIGHT_P90_N]))
        assert hasattr(out, "detach")
        assert float(out[0]) > 0.9 * COCARRY_REWARD_STRESS_COEFF
        # Matched-band stress on the torch path is also ~0.
        assert float(cocarry_excess_stress_penalty(torch.tensor([104.5])[0])) == pytest.approx(
            0.0, abs=1e-6
        )


class _FakeTransportEnv(gym.Env):  # type: ignore[type-arg]
    """Minimal Tier-1 gym.Env fake exposing the privileged transport distance + step."""

    def __init__(self, dist: float = 0.3) -> None:
        super().__init__()
        self._dist = dist
        self._next_dist: float | None = None  # post-step distance, if the step transitions
        self._terminated = False

    def privileged_transport_distance(self) -> np.ndarray:  # type: ignore[type-arg]
        return np.array([self._dist], dtype=np.float64)

    def step(self, action: Any) -> tuple[Any, float, Any, Any, dict[str, Any]]:
        del action
        if self._next_dist is not None:  # transition s -> s' so entry/exit reads differ
            self._dist = self._next_dist
        return ({}, 1.0, np.array([self._terminated]), np.array([False]), {})

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        del kwargs
        return {}, {}


class TestTransportPBRS:
    """Policy-invariant transport PBRS (supporting term; ADR-026 §Decision 4)."""

    def test_phi_is_negative_coeff_times_distance(self) -> None:
        w = CoCarryTransportPBRSWrapper(_FakeTransportEnv(0.3), coeff=2.0, gamma=0.9)
        assert float(w._phi()[0]) == pytest.approx(-2.0 * 0.3)

    def test_phi_vanishes_at_goal(self) -> None:
        w = CoCarryTransportPBRSWrapper(_FakeTransportEnv(0.0), coeff=1.0, gamma=0.9)
        assert float(w._phi()[0]) == pytest.approx(0.0, abs=1e-12)

    def test_step_adds_gamma_phi_next_minus_phi(self) -> None:
        env = _FakeTransportEnv(0.30)
        env._next_dist = 0.20  # step transitions s(0.30) -> s'(0.20)
        w = CoCarryTransportPBRSWrapper(env, coeff=1.0, gamma=0.9)
        _, reward, _, _, _ = w.step({})
        # F = gamma*Phi(s') - Phi(s) = 0.9*(-0.20) - (-0.30) = 0.12; reward 1.0 + F.
        assert float(reward) == pytest.approx(1.0 + (0.9 * -0.20 - -0.30))

    def test_terminal_zeroes_next_potential(self) -> None:
        env = _FakeTransportEnv(0.30)
        env._terminated = True
        w = CoCarryTransportPBRSWrapper(env, coeff=1.0, gamma=0.9)
        _, reward, _, _, _ = w.step({})
        # terminated => Phi(s')=0 => F = -Phi(s) = +0.30.
        assert float(reward) == pytest.approx(1.0 + 0.30)

    def test_nonpositive_coeff_raises(self) -> None:
        with pytest.raises(ValueError, match="coeff must be > 0"):
            CoCarryTransportPBRSWrapper(_FakeTransportEnv(), coeff=0.0, gamma=0.9)

    def test_missing_accessor_raises(self) -> None:
        import gymnasium as gym

        class _NoAccessor(gym.Env):  # type: ignore[type-arg]
            pass

        with pytest.raises(TypeError, match="privileged_transport_distance"):
            CoCarryTransportPBRSWrapper(_NoAccessor(), coeff=1.0, gamma=0.9)


class TestRemediationConstantsFrozen:
    """The new coefficients are captured by the freeze completeness guard (R-2026-06-B §15)."""

    def test_new_constants_in_enumerator(self) -> None:
        consts = freeze.enumerate_cocarry_constants()
        for name in (
            "COCARRY_REWARD_STRESS_COEFF",
            "COCARRY_REWARD_STRESS_SOFT_THRESHOLD_N",
            "COCARRY_REWARD_STRESS_TANH_SCALE_N",
            "COCARRY_REWARD_TRANSPORT_PBRS_COEFF",
        ):
            assert name in consts

    def test_manifest_complete_with_new_constants(self) -> None:
        manifest = freeze.build_manifest(
            matched_reference_success_rate=1.0,
            n_seed_clusters=12,
            matched_success_stress_p99_n=104.5,
            config_path=Path("configs/training/ego_aht_happo/cocarry_matched.yaml"),
            env_module_path=Path("src/chamber/envs/cocarry.py"),
        )
        assert freeze.missing_constants(manifest) == []


class TestConfigParity:
    """The training config's transport PBRS coeff equals the env constant (no drift)."""

    def test_transport_pbrs_coeff_parity(self) -> None:
        from concerto.training.config import load_config

        cfg = load_config(config_path=Path("configs/training/ego_aht_happo/cocarry_matched.yaml"))
        assert cfg.shaping.transport_pbrs_coeff == COCARRY_REWARD_TRANSPORT_PBRS_COEFF
        assert cfg.shaping.transport_pbrs_coeff > 0.0  # the remediation is ON for this cell
