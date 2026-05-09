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
)
from concerto.safety.cbf_qp import AgentSnapshot
from concerto.safety.conformal import (
    compute_per_pair_loss,
    constant_velocity_predict,
    reset_on_partner_swap,
    update_lambda,
)


def test_update_lambda_decreases_when_loss_exceeds_epsilon() -> None:
    state = SafetyState(lambda_=np.array([0.1], dtype=np.float64), epsilon=0.05, eta=0.1)
    update_lambda(state, np.array([0.5], dtype=np.float64), in_warmup=False)
    # eps - l_k = 0.05 - 0.5 = -0.45; lambda += 0.1 * -0.45 = -0.045.
    assert state.lambda_[0] == pytest.approx(0.1 + 0.1 * (0.05 - 0.5))
    assert state.lambda_[0] < 0.1


def test_update_lambda_increases_when_loss_below_epsilon() -> None:
    state = SafetyState(lambda_=np.array([0.0], dtype=np.float64), epsilon=0.05, eta=0.1)
    update_lambda(state, np.array([0.0], dtype=np.float64), in_warmup=False)
    assert state.lambda_[0] == pytest.approx(0.1 * 0.05)
    assert state.lambda_[0] > 0.0


def test_update_lambda_warmup_widens_step_size() -> None:
    """Warmup epsilon (eps * 1.5) yields larger |delta_lambda| at fixed loss."""
    state_normal = SafetyState(
        lambda_=np.array([0.0], dtype=np.float64),
        epsilon=0.05,
        eta=0.1,
        warmup_steps_remaining=50,
    )
    state_warmup = SafetyState(
        lambda_=np.array([0.0], dtype=np.float64),
        epsilon=0.05,
        eta=0.1,
        warmup_steps_remaining=50,
    )
    loss_zero = np.array([0.0], dtype=np.float64)
    update_lambda(state_normal, loss_zero, in_warmup=False)
    update_lambda(state_warmup, loss_zero, in_warmup=True)
    assert state_warmup.lambda_[0] > state_normal.lambda_[0]
    assert state_warmup.warmup_steps_remaining == 49
    assert state_normal.warmup_steps_remaining == 50  # untouched when not in warmup


def test_update_lambda_warmup_decrement_clamps_at_zero() -> None:
    state = SafetyState(
        lambda_=np.array([0.0], dtype=np.float64),
        epsilon=0.05,
        eta=0.1,
        warmup_steps_remaining=0,
    )
    update_lambda(state, np.array([0.0], dtype=np.float64), in_warmup=True)
    assert state.warmup_steps_remaining == 0


def test_update_lambda_rejects_shape_mismatch() -> None:
    state = SafetyState(lambda_=np.zeros(2, dtype=np.float64), epsilon=0.05, eta=0.1)
    with pytest.raises(ValueError, match="shape"):
        update_lambda(state, np.zeros(3, dtype=np.float64), in_warmup=False)


def test_update_lambda_default_state_uses_adr004_constants() -> None:
    """Default SafetyState picks up ADR-004 §Decision values."""
    state = SafetyState(lambda_=np.zeros(1, dtype=np.float64))
    assert state.epsilon == DEFAULT_EPSILON == -0.05
    assert state.eta == DEFAULT_ETA == 0.01
    update_lambda(state, np.array([0.0], dtype=np.float64), in_warmup=False)
    # eps = -0.05, eta = 0.01, l_k = 0 ⇒ lambda becomes -5e-4.
    assert state.lambda_[0] == pytest.approx(0.01 * (-0.05))
    assert state.lambda_[0] < 0.0  # negative epsilon biases toward tighter constraint


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
    state = SafetyState(lambda_=np.array([0.5, -0.3], dtype=np.float64), epsilon=0.05, eta=0.1)
    reset_on_partner_swap(state, n_pairs=2, lambda_safe=0.0, n_warmup_steps=30)
    np.testing.assert_array_equal(state.lambda_, [0.0, 0.0])
    assert state.warmup_steps_remaining == 30


def test_reset_on_partner_swap_changes_pair_count() -> None:
    """Partner swap may add/remove agents — lambda_ length must change."""
    state = SafetyState(
        lambda_=np.array([0.5, -0.3, 0.1], dtype=np.float64),
        epsilon=0.05,
        eta=0.1,
    )
    reset_on_partner_swap(state, n_pairs=1, lambda_safe=0.2)
    assert state.lambda_.shape == (1,)
    assert state.lambda_[0] == 0.2
    assert state.warmup_steps_remaining == DEFAULT_WARMUP_STEPS


def test_full_warmup_decay_to_steady_state() -> None:
    """Running update_lambda for the full warmup window decrements to zero."""
    state = SafetyState(
        lambda_=np.zeros(1, dtype=np.float64),
        epsilon=0.05,
        eta=0.05,
        warmup_steps_remaining=10,
    )
    for _ in range(10):
        update_lambda(state, np.zeros(1, dtype=np.float64), in_warmup=True)
    assert state.warmup_steps_remaining == 0
