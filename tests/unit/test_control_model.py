# SPDX-License-Identifier: Apache-2.0
"""Unit tests for :class:`AgentControlModel` implementations (spike_004A §Per-agent control model).

Covers:

- :class:`concerto.safety.api.DoubleIntegratorControlModel` — identity
  action ↔ Cartesian-accel map (ADR-004 §Decision; spike_004A §Reduction
  to the current double-integrator case).
- :class:`concerto.safety.api.JacobianControlModel` — Stage-1 AS spike
  skeleton (ADR-007 §Stage 1 — Foundation axes); raises
  :class:`NotImplementedError` until a Jacobian callable is supplied.
"""

from __future__ import annotations

import numpy as np
import pytest

from concerto.safety.api import (
    Bounds,
    DoubleIntegratorControlModel,
    JacobianControlModel,
)
from concerto.safety.cbf_qp import AgentSnapshot


def _state(dim: int = 2) -> AgentSnapshot:
    return AgentSnapshot(
        position=np.zeros(dim, dtype=np.float64),
        velocity=np.zeros(dim, dtype=np.float64),
        radius=0.1,
    )


def _bounds(action_norm: float = 2.0) -> Bounds:
    return Bounds(action_norm=action_norm, action_rate=0.5, comm_latency_ms=1.0, force_limit=20.0)


def test_double_integrator_action_to_cartesian_accel_is_identity() -> None:
    model = DoubleIntegratorControlModel(uid="a", action_dim=2)
    action = np.array([0.3, -0.7], dtype=np.float64)
    np.testing.assert_array_equal(model.action_to_cartesian_accel(_state(2), action), action)


def test_double_integrator_cartesian_accel_to_action_is_identity() -> None:
    model = DoubleIntegratorControlModel(uid="b", action_dim=3)
    cart = np.array([1.0, 2.0, -3.0], dtype=np.float64)
    np.testing.assert_array_equal(model.cartesian_accel_to_action(_state(3), cart), cart)


def test_double_integrator_round_trip_is_identity_for_random_actions() -> None:
    rng = np.random.default_rng(0)
    model = DoubleIntegratorControlModel(uid="c", action_dim=4)
    for _ in range(20):
        action = rng.standard_normal(4).astype(np.float64)
        cart = model.action_to_cartesian_accel(_state(4), action)
        action_back = model.cartesian_accel_to_action(_state(4), cart)
        np.testing.assert_allclose(action_back, action)


def test_double_integrator_max_cartesian_accel_returns_bounds_action_norm() -> None:
    model = DoubleIntegratorControlModel(uid="a", action_dim=2)
    assert model.max_cartesian_accel(_bounds(action_norm=3.5)) == pytest.approx(3.5)


def test_double_integrator_position_dim_equals_action_dim() -> None:
    model = DoubleIntegratorControlModel(uid="a", action_dim=7)
    assert model.position_dim == model.action_dim == 7


def test_double_integrator_action_shape_mismatch_raises() -> None:
    model = DoubleIntegratorControlModel(uid="a", action_dim=2)
    with pytest.raises(ValueError, match="action shape"):
        model.action_to_cartesian_accel(_state(2), np.zeros(3, dtype=np.float64))


def test_double_integrator_cartesian_shape_mismatch_raises() -> None:
    model = DoubleIntegratorControlModel(uid="a", action_dim=2)
    with pytest.raises(ValueError, match="cartesian_accel shape"):
        model.cartesian_accel_to_action(_state(2), np.zeros(3, dtype=np.float64))


def test_jacobian_control_model_without_jacobian_fn_raises_not_implemented() -> None:
    """Stage-1 AS spike has not landed yet — the placeholder must fail loudly."""
    model = JacobianControlModel(uid="arm", action_dim=7, position_dim=3)
    with pytest.raises(NotImplementedError, match="Stage-1 AS spike"):
        model.action_to_cartesian_accel(_state(3), np.zeros(7, dtype=np.float64))
    with pytest.raises(NotImplementedError, match="Stage-1 AS spike"):
        model.cartesian_accel_to_action(_state(3), np.zeros(3, dtype=np.float64))


def test_jacobian_control_model_with_jacobian_fn_applies_linear_map() -> None:
    """When a Jacobian callable is supplied the model becomes usable.

    Stage-1 AS spike will exercise this path with a real kinematic
    model; the unit test pins the contract.
    """
    rng = np.random.default_rng(0)
    jac_const = rng.standard_normal((3, 7)).astype(np.float64)

    def jacobian_fn(state: AgentSnapshot) -> np.ndarray:
        del state
        return jac_const

    model = JacobianControlModel(
        uid="arm",
        action_dim=7,
        position_dim=3,
        jacobian_fn=jacobian_fn,
        max_cartesian_accel_value=5.0,
    )
    action = rng.standard_normal(7).astype(np.float64)
    cart = model.action_to_cartesian_accel(_state(3), action)
    np.testing.assert_allclose(cart, jac_const @ action)
    # Right-inverse round-trip: forward through pseudo-inverse and back
    # must reconstruct the Cartesian acceleration (over-actuated system,
    # so the action selection is non-unique but the Cartesian image is).
    action_back = model.cartesian_accel_to_action(_state(3), cart)
    np.testing.assert_allclose(jac_const @ action_back, cart, atol=1e-6)


def test_jacobian_control_model_max_cartesian_accel_requires_configured_value() -> None:
    model = JacobianControlModel(uid="arm", action_dim=7, position_dim=3)
    with pytest.raises(ValueError, match="max_cartesian_accel_value"):
        model.max_cartesian_accel(_bounds())


def test_jacobian_control_model_max_cartesian_accel_returns_configured_value() -> None:
    model = JacobianControlModel(
        uid="arm", action_dim=7, position_dim=3, max_cartesian_accel_value=4.2
    )
    assert model.max_cartesian_accel(_bounds()) == pytest.approx(4.2)
