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
        # The whole matched band (max 105 N) sits below the re-grounded
        # threshold (110 N, the demonstrated cooperative ceiling), so the
        # cooperative pair incurs ~0 penalty (the 100% reference invariant).
        for s in _MATCHED_BAND_N:
            assert float(cocarry_excess_stress_penalty(np.array([s]))[0]) == pytest.approx(
                0.0, abs=1e-9
            )

    def test_zero_at_and_below_threshold(self) -> None:
        assert (
            float(
                cocarry_excess_stress_penalty(np.array([COCARRY_REWARD_STRESS_SOFT_THRESHOLD_N]))[0]
            )
            == 0.0
        )
        assert float(cocarry_excess_stress_penalty(np.array([0.0]))[0]) == 0.0

    def test_monotonic_nondecreasing_in_stress(self) -> None:
        xs = np.linspace(0.0, 1000.0, 50)
        ys = cocarry_excess_stress_penalty(xs)
        assert np.all(np.diff(np.asarray(ys)) >= -1e-12)

    def test_saturates_near_coeff_on_the_fight(self) -> None:
        # The incumbent fight (p90 ~854 N, excess ~744 N) saturates tanh ~ coeff.
        pen = float(cocarry_excess_stress_penalty(np.array([_FIGHT_P90_N]))[0])
        assert pen > 0.9 * COCARRY_REWARD_STRESS_COEFF
        assert pen <= COCARRY_REWARD_STRESS_COEFF

    def test_threshold_grounded_in_cooperative_band_below_limit(self) -> None:
        # Re-grounded 2026-06-18 (cocarry_rung2_900k_train_stop.json): the
        # threshold is the demonstrated cooperative ceiling — strictly ABOVE
        # the matched success-stress p99 (104.5 N, so the matched pair pays ~0)
        # and strictly BELOW the f_max success limit (so the policy is pushed
        # into the cooperative regime with margin, not onto the failure edge).
        assert 104.5 < COCARRY_REWARD_STRESS_SOFT_THRESHOLD_N < COCARRY_STRESS_MAX_PROXY_N

    def test_bites_before_the_failure_edge(self) -> None:
        # The regression guard for the squeeze-fragility fix: with the old
        # threshold=130 / scale=100 the penalty at the 130 N limit was ~0.20
        # and ~0.03 at the 133 N spike — no gradient, so the 900k incumbent
        # rode the edge and 2/12 seeds spiked to 133-134 N. The re-grounded
        # penalty must carry a REAL cost across the corridor BEFORE the limit:
        # comfortably positive at 120 N and strong (> half coeff) at the 130 N
        # edge, so margin-building is rewarded.
        at_120 = float(cocarry_excess_stress_penalty(np.array([120.0]))[0])
        at_limit = float(cocarry_excess_stress_penalty(np.array([COCARRY_STRESS_MAX_PROXY_N]))[0])
        assert at_120 > 0.2 * COCARRY_REWARD_STRESS_COEFF
        assert at_limit > 0.5 * COCARRY_REWARD_STRESS_COEFF
        assert at_limit < COCARRY_REWARD_STRESS_COEFF  # still below saturation at the edge

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


class TestCouplingGroundedPenalty:
    """Coupling-measure penalty re-grounding contract (ADR-026 §D4 4c re-freeze).

    Under ``stress_measure="coupling"`` the env threads a coupling-grounded
    threshold + scale into :func:`cocarry_excess_stress_penalty` so the penalty
    is ~0 across the matched-coupling cooperative band (~240-290 N) and carries a
    real cost by the coupling f_max (365.6 N) — the SAME shape the wrist
    grounding has, on the invariant measure. Without it the wrist-grounded
    default (threshold 110 N) saturates across the whole cooperative coupling
    band and would punish cooperation ("train against one constraint, judge
    against another"). The exact threshold is grounded from the Stage-1
    matched-coupling distribution under the locked rule; these representative
    values pin only the CONTRACT — the penalty honours caller threshold/scale.
    """

    # Representative coupling grounding (illustrative; Stage-1 measures the exact
    # p99 and f_max 365.6 N is held fixed per the locked re-freeze protocol).
    _COUPLING_THRESHOLD_N = 300.0
    _COUPLING_SCALE_N = 65.6
    _COUPLING_FMAX_N = 365.6
    _COUPLING_BAND_N = (240.0, 270.0, 290.0)

    def _pen(self, s: float) -> float:
        return float(
            cocarry_excess_stress_penalty(
                np.array([s]),
                threshold=self._COUPLING_THRESHOLD_N,
                scale=self._COUPLING_SCALE_N,
            )[0]
        )

    def test_zero_across_matched_coupling_band(self) -> None:
        for s in self._COUPLING_BAND_N:
            assert self._pen(s) == pytest.approx(0.0, abs=1e-9)

    def test_real_cost_by_coupling_fmax(self) -> None:
        # At the coupling f_max the excess ~ scale, so tanh(1) ~ 0.76 coeff — a
        # real gradient holding stress inside the cooperative regime (mirrors the
        # wrist grounding's behaviour at its 130 N limit).
        at_fmax = self._pen(self._COUPLING_FMAX_N)
        assert at_fmax > 0.5 * COCARRY_REWARD_STRESS_COEFF
        assert at_fmax < COCARRY_REWARD_STRESS_COEFF

    def test_default_wrist_grounding_saturates_on_coupling_band(self) -> None:
        # The motivation for the fix: the DEFAULT (wrist) grounding saturates
        # across the cooperative coupling band, which would punish cooperation.
        default_pen = float(cocarry_excess_stress_penalty(np.array([270.0]))[0])
        assert default_pen > 0.9 * COCARRY_REWARD_STRESS_COEFF
