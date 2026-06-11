# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``concerto.safety.conformal`` (T3.6).

Covers ADR-004 §Decision (Huriot & Sibai 2025 §IV update rule, partner-
swap warmup per risk-mitigation #2) and the constant-velocity predictor
stub from plan/03 §3.3.
"""

from __future__ import annotations

import numpy as np
import pytest

from concerto.safety.api import (
    DEFAULT_EPSILON,
    DEFAULT_ETA,
    DEFAULT_WARMUP_STEPS,
    SafetyState,
    make_lambda_dict,
)
from concerto.safety.cbf_qp import AgentSnapshot
from concerto.safety.conformal import (
    compute_per_pair_loss,
    constant_velocity_predict,
    reset_on_partner_swap,
    update_lambda,
)

_PAIR_AB: tuple[str, str] = ("a", "b")
_PAIR_AC: tuple[str, str] = ("a", "c")


def test_update_lambda_decreases_when_loss_exceeds_epsilon() -> None:
    state = SafetyState(lambda_={_PAIR_AB: 0.1}, epsilon=0.05, eta=0.1)
    update_lambda(state, {_PAIR_AB: 0.5}, in_warmup=False)
    # eps - l_k = 0.05 - 0.5 = -0.45; lambda += 0.1 * -0.45 = -0.045.
    assert state.lambda_[_PAIR_AB] == pytest.approx(0.1 + 0.1 * (0.05 - 0.5))
    assert state.lambda_[_PAIR_AB] < 0.1


def test_update_lambda_increases_when_loss_below_epsilon() -> None:
    state = SafetyState(lambda_={_PAIR_AB: 0.0}, epsilon=0.05, eta=0.1)
    update_lambda(state, {_PAIR_AB: 0.0}, in_warmup=False)
    assert state.lambda_[_PAIR_AB] == pytest.approx(0.1 * 0.05)
    assert state.lambda_[_PAIR_AB] > 0.0


def test_update_lambda_warmup_widens_step_size() -> None:
    """Warmup epsilon (eps * 1.5) yields larger |delta_lambda| at fixed loss."""
    state_normal = SafetyState(
        lambda_={_PAIR_AB: 0.0},
        epsilon=0.05,
        eta=0.1,
        warmup_steps_remaining=50,
    )
    state_warmup = SafetyState(
        lambda_={_PAIR_AB: 0.0},
        epsilon=0.05,
        eta=0.1,
        warmup_steps_remaining=50,
    )
    loss_zero: dict[tuple[str, str], float] = {_PAIR_AB: 0.0}
    update_lambda(state_normal, loss_zero, in_warmup=False)
    update_lambda(state_warmup, loss_zero, in_warmup=True)
    assert state_warmup.lambda_[_PAIR_AB] > state_normal.lambda_[_PAIR_AB]
    assert state_warmup.warmup_steps_remaining == 49
    assert state_normal.warmup_steps_remaining == 50  # untouched when not in warmup


def test_update_lambda_warmup_decrement_clamps_at_zero() -> None:
    state = SafetyState(
        lambda_={_PAIR_AB: 0.0},
        epsilon=0.05,
        eta=0.1,
        warmup_steps_remaining=0,
    )
    update_lambda(state, {_PAIR_AB: 0.0}, in_warmup=True)
    assert state.warmup_steps_remaining == 0


def test_update_lambda_rejects_key_set_mismatch() -> None:
    state = SafetyState(lambda_={_PAIR_AB: 0.0, _PAIR_AC: 0.0}, epsilon=0.05, eta=0.1)
    with pytest.raises(ValueError, match="key set"):
        update_lambda(state, {_PAIR_AB: 0.0}, in_warmup=False)


def test_update_lambda_default_state_uses_adr004_constants() -> None:
    """Default SafetyState picks up ADR-004 §Decision values."""
    state = SafetyState(lambda_=make_lambda_dict(("a", "b")))
    assert state.epsilon == DEFAULT_EPSILON == -0.05
    assert state.eta == DEFAULT_ETA == 0.01
    update_lambda(state, {_PAIR_AB: 0.0}, in_warmup=False)
    # eps = -0.05, eta = 0.01, l_k = 0 ⇒ lambda becomes -5e-4.
    assert state.lambda_[_PAIR_AB] == pytest.approx(0.01 * (-0.05))
    assert state.lambda_[_PAIR_AB] < 0.0  # negative epsilon biases toward tighter constraint


# ----- P1.05.7 / issue #180: symmetric λ clamp -----


class TestUpdateLambdaSymmetricClamp:
    """``lambda_bound`` kwarg clamps λ to ``[-bound, +bound]`` per step (#180)."""

    def test_clamp_none_preserves_unclamped_theorem3_behaviour(self) -> None:
        """``lambda_bound=None`` (default) ⇒ pre-P1.05.7 behaviour, no clamp."""
        state = SafetyState(lambda_={_PAIR_AB: -0.1}, epsilon=-0.05, eta=0.1)
        update_lambda(state, {_PAIR_AB: 0.0}, in_warmup=False)
        # No clamp ⇒ updated to -0.1 + 0.1 * (-0.05) = -0.105.
        assert state.lambda_[_PAIR_AB] == pytest.approx(-0.105)

    def test_clamp_pins_negative_drift_at_negative_bound(self) -> None:
        """Production case: long negative drift gets pinned at ``-lambda_bound``."""
        state = SafetyState(lambda_={_PAIR_AB: -6.9}, epsilon=-0.05, eta=0.1)
        update_lambda(state, {_PAIR_AB: 0.0}, in_warmup=False, lambda_bound=7.0)
        # Unclamped: -6.9 + 0.1*(-0.05) = -6.905; still inside the bound.
        assert state.lambda_[_PAIR_AB] == pytest.approx(-6.905)

        # Now push past the boundary: λ would drop to -7.005 — clamped to -7.0.
        state2 = SafetyState(lambda_={_PAIR_AB: -7.0}, epsilon=-0.05, eta=0.1)
        update_lambda(state2, {_PAIR_AB: 0.0}, in_warmup=False, lambda_bound=7.0)
        assert state2.lambda_[_PAIR_AB] == -7.0

    def test_clamp_pins_positive_drift_at_positive_bound(self) -> None:
        """Symmetric: positive drift gets pinned at ``+lambda_bound``."""
        # Standard Theorem 3 regime (eps>0): λ drifts positive when loss < eps.
        state = SafetyState(lambda_={_PAIR_AB: 7.0}, epsilon=0.5, eta=0.1)
        update_lambda(state, {_PAIR_AB: 0.0}, in_warmup=False, lambda_bound=7.0)
        # Unclamped would be 7.05; clamped to 7.0.
        assert state.lambda_[_PAIR_AB] == 7.0

    def test_clamp_holds_under_repeated_drift(self) -> None:
        """100 consecutive unconstrained-loss steps: λ never crosses ±bound."""
        state = SafetyState(lambda_={_PAIR_AB: 0.0}, epsilon=-0.05, eta=0.1)
        for _ in range(100):
            update_lambda(state, {_PAIR_AB: 0.0}, in_warmup=False, lambda_bound=0.3)
        # Without clamp: 100 x 0.1 x -0.05 = -0.5. With clamp at 0.3: pinned to -0.3.
        assert state.lambda_[_PAIR_AB] == -0.3

    def test_clamp_does_not_alter_in_bound_updates(self) -> None:
        """Steps that stay inside the bound are byte-identical to the unclamped path."""
        state_clamped = SafetyState(lambda_={_PAIR_AB: 0.0}, epsilon=-0.05, eta=0.01)
        state_unclamped = SafetyState(lambda_={_PAIR_AB: 0.0}, epsilon=-0.05, eta=0.01)
        for _ in range(50):  # eta*|eps|*50 = -0.025, well inside ±7.0
            update_lambda(state_clamped, {_PAIR_AB: 0.0}, in_warmup=False, lambda_bound=7.0)
            update_lambda(state_unclamped, {_PAIR_AB: 0.0}, in_warmup=False)
        assert state_clamped.lambda_[_PAIR_AB] == state_unclamped.lambda_[_PAIR_AB]

    def test_clamp_rejects_non_positive_bound(self) -> None:
        """``lambda_bound=0`` or negative ⇒ ValueError (operator-intent loud-fail)."""
        state = SafetyState(lambda_={_PAIR_AB: 0.0})
        with pytest.raises(ValueError, match="strictly positive"):
            update_lambda(state, {_PAIR_AB: 0.0}, lambda_bound=0.0)
        with pytest.raises(ValueError, match="strictly positive"):
            update_lambda(state, {_PAIR_AB: 0.0}, lambda_bound=-1.0)

    def test_clamp_applied_per_pair_not_globally(self) -> None:
        """Multi-pair: each pair's λ clamps independently."""
        state = SafetyState(lambda_={_PAIR_AB: -6.99, _PAIR_AC: 0.0}, epsilon=-0.05, eta=0.5)
        update_lambda(state, {_PAIR_AB: 0.0, _PAIR_AC: 0.0}, lambda_bound=7.0)
        # _PAIR_AB: -6.99 + 0.5*-0.05 = -7.015 → clamped to -7.0.
        # _PAIR_AC:   0.0 + 0.5*-0.05 = -0.025 (inside bound, no clamp).
        assert state.lambda_[_PAIR_AB] == -7.0
        assert state.lambda_[_PAIR_AC] == pytest.approx(-0.025)


def test_compute_per_pair_loss_positive_on_overestimate() -> None:
    pred = np.array([0.5, 0.2, 0.3], dtype=np.float64)
    actual = np.array([0.3, 0.4, 0.3], dtype=np.float64)
    loss = compute_per_pair_loss(pred, actual)
    np.testing.assert_array_equal(loss, [0.2, 0.0, 0.0])


def test_compute_per_pair_loss_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape"):
        compute_per_pair_loss(np.zeros(2, dtype=np.float64), np.zeros(3, dtype=np.float64))


def test_constant_velocity_predict_extrapolates() -> None:
    snap = AgentSnapshot(
        position=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        velocity=np.array([1.0, 2.0, -0.5], dtype=np.float64),
        radius=0.2,
    )
    pred = constant_velocity_predict(snap, dt=0.5)
    np.testing.assert_allclose(pred.position, [0.5, 1.0, -0.25])
    np.testing.assert_allclose(pred.velocity, snap.velocity)
    assert pred.radius == snap.radius


def test_constant_velocity_predict_zero_dt_is_noop() -> None:
    snap = AgentSnapshot(
        position=np.array([1.0, 2.0], dtype=np.float64),
        velocity=np.array([3.0, 4.0], dtype=np.float64),
        radius=0.1,
    )
    pred = constant_velocity_predict(snap, dt=0.0)
    np.testing.assert_allclose(pred.position, snap.position)
    np.testing.assert_allclose(pred.velocity, snap.velocity)


def test_reset_on_partner_swap_resets_lambda_and_primes_warmup() -> None:
    state = SafetyState(lambda_={_PAIR_AB: 0.5, _PAIR_AC: -0.3}, epsilon=0.05, eta=0.1)
    reset_on_partner_swap(state, uids=("a", "b", "c"), lambda_safe=0.0, n_warmup_steps=30)
    assert state.lambda_ == {_PAIR_AB: 0.0, _PAIR_AC: 0.0, ("b", "c"): 0.0}
    assert state.warmup_steps_remaining == 30


def test_reset_on_partner_swap_changes_pair_set() -> None:
    """Partner swap may add/remove agents — lambda_ key set must change."""
    state = SafetyState(
        lambda_={_PAIR_AB: 0.5, _PAIR_AC: -0.3, ("b", "c"): 0.1},
        epsilon=0.05,
        eta=0.1,
    )
    reset_on_partner_swap(state, uids=("a", "b"), lambda_safe=0.2)
    assert state.lambda_ == {_PAIR_AB: 0.2}
    assert state.warmup_steps_remaining == DEFAULT_WARMUP_STEPS


def test_full_warmup_decay_to_steady_state() -> None:
    """Running update_lambda for the full warmup window decrements to zero."""
    state = SafetyState(
        lambda_={_PAIR_AB: 0.0},
        epsilon=0.05,
        eta=0.05,
        warmup_steps_remaining=10,
    )
    for _ in range(10):
        update_lambda(state, {_PAIR_AB: 0.0}, in_warmup=True)
    assert state.warmup_steps_remaining == 0
