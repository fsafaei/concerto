# SPDX-License-Identifier: Apache-2.0
"""Unit tests for :class:`JacobianEmergencyController` (ADR-004 risk-mitigation #1; P1.02).

The Plan-subagent design pass for P1.02 settled four contracts the
real (non-placeholder) controller must satisfy:

1. The Cartesian L2 cap ``bounds.cartesian_accel_capacity`` is enforced
   **before** the Jacobian transform — the operator-facing
   right-inverse Protocol promise (``AgentControlModel`` at
   ``api.py:309``) is honoured at the input boundary.
2. The per-joint L-infinity clip ``bounds.action_linf_component`` is
   enforced **after** the Jacobian transform — the damped J^+ minimises
   joint L2 norm, not L-infinity, so the post-transform clip is what
   guarantees per-joint hardware envelopes.
3. Near-singular configurations stay finite via the damped
   pseudoinverse (``J^+ = J^T (J J^T + damping^2 I)^{-1}``,
   Nakamura & Hanafusa 1986).

(The original Plan-subagent design pass also proposed a runtime
``isinstance(control_model, JacobianControlModel)`` guard at
construction, but pyright-strict already enforces the type at
the boundary; the runtime check would be a redundant belt-and-
braces.)

These tests exercise the controller in isolation against a
synthetic Jacobian fixture; integration via :func:`maybe_brake` is
pinned by
``tests/property/test_braking_multipair.py::test_jacobian_emergency_controller_routes_through_jacobian_control_model``.
"""

from __future__ import annotations

import numpy as np
import pytest

from concerto.safety.api import Bounds, JacobianControlModel
from concerto.safety.cbf_qp import AgentSnapshot
from concerto.safety.emergency import JacobianEmergencyController


def _bounds(*, linf: float = 5.0, cap: float = 5.0) -> Bounds:
    """Build a Bounds fixture with independent L-inf and L2 caps.

    Defaults to the conservative pattern (``linf == cap``), matching
    the post-P1.02 ``Bounds`` docstring's recommended operator
    pattern. Tests that need a tighter per-joint envelope override
    ``linf`` independently to assert the post-transform clip fires.
    """
    return Bounds(
        action_linf_component=linf,
        cartesian_accel_capacity=cap,
        action_rate=0.5,
        comm_latency_ms=1.0,
        force_limit=20.0,
    )


def _snapshot(position: np.ndarray) -> AgentSnapshot:
    return AgentSnapshot(
        position=position.astype(np.float64),
        velocity=np.zeros_like(position, dtype=np.float64),
        radius=0.05,
    )


def _well_conditioned_jacobian(*, position_dim: int = 3, action_dim: int = 7) -> np.ndarray:
    """3-Cartesian x 7-joint Jacobian with full row rank and no near-singular axes.

    First ``position_dim`` joints contribute identity-style to the
    Cartesian directions; the remaining joints have zero rows so the
    damped J^+ round-trip is exact to numerical precision (the damping
    residual collapses when the Cartesian target lies in the row
    space).
    """
    jac = np.zeros((position_dim, action_dim), dtype=np.float64)
    for k in range(position_dim):
        jac[k, k] = 1.0
    return jac


def _make_controller(*, jacobian: np.ndarray, damping: float = 1e-3) -> JacobianEmergencyController:
    model = JacobianControlModel(
        uid="arm",
        action_dim=jacobian.shape[1],
        position_dim=jacobian.shape[0],
        jacobian_fn=lambda _state: jacobian,
        damping=damping,
        max_cartesian_accel_value=5.0,
    )
    return JacobianEmergencyController(control_model=model)


# ---------------------------------------------------------------------------
# 1. Zero / cancelled aggregate -> zero joint vector
# ---------------------------------------------------------------------------


def test_zero_repulsion_returns_zero_torques() -> None:
    """Empty pairwise list returns a zero ``(action_dim,)`` joint vector."""
    jac = _well_conditioned_jacobian()
    controller = _make_controller(jacobian=jac)
    state = _snapshot(np.array([0.0, 0.0, 0.0]))
    out = controller.compute_override(state, [], _bounds())
    assert out.shape == (7,)
    assert np.all(out == 0.0)
    assert out.dtype == np.float64


def test_cancelled_aggregate_returns_zero_torques() -> None:
    """Two opposing unit vectors aggregate to ~zero -> zero joint vector (not a spurious push)."""
    jac = _well_conditioned_jacobian()
    controller = _make_controller(jacobian=jac)
    state = _snapshot(np.array([0.0, 0.0, 0.0]))
    out = controller.compute_override(
        state,
        [
            np.array([1.0, 0.0, 0.0], dtype=np.float64),
            np.array([-1.0, 0.0, 0.0], dtype=np.float64),
        ],
        _bounds(),
    )
    assert out.shape == (7,)
    assert np.all(out == 0.0)


# ---------------------------------------------------------------------------
# 3. Cartesian L2 cap fires before the Jacobian transform
# ---------------------------------------------------------------------------


def test_cartesian_l2_cap_applied_before_jacobian_transform() -> None:
    """A pre-cap aggregate magnitude of 2*cap collapses to magnitude exactly ``cap``.

    Round-trip ``J @ override`` must equal ``cap * n_hat`` (the
    saturated Cartesian target), not the raw aggregate magnitude.
    """
    jac = _well_conditioned_jacobian()
    controller = _make_controller(jacobian=jac)
    state = _snapshot(np.array([0.0, 0.0, 0.0]))
    cap = 4.0
    # Two parallel unit vectors sum to 2*n_hat with magnitude 2 - well
    # above the cap, so the saturation step must scale back to cap.
    n_hat = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    out = controller.compute_override(
        state,
        [n_hat, n_hat],
        _bounds(linf=100.0, cap=cap),  # linf high so post-clip is a no-op
    )
    realised_cart = jac @ out
    # Damped-pseudoinverse residual is O(damping^2 * cap / sigma_min^2)
    # = O(1e-3^2 * 4) = O(1e-5) for this well-conditioned fixture.
    np.testing.assert_allclose(realised_cart, cap * n_hat, atol=1e-4)
    assert float(np.linalg.norm(realised_cart)) == pytest.approx(cap, abs=1e-4)


# ---------------------------------------------------------------------------
# 4. Per-joint L-infinity clip fires after the Jacobian transform
# ---------------------------------------------------------------------------


def test_per_joint_linf_clip_applied_after_jacobian_transform() -> None:
    """A J^+ output that would exceed the per-joint envelope on one joint gets clipped.

    Constructs an over-actuated Jacobian where the damped pseudo-inverse
    of a unit-x Cartesian target concentrates the response on joint 0
    (the only joint with a non-zero x-row entry, and at unit gain).
    Setting ``cartesian_accel_capacity = 10`` and ``action_linf_component
    = 2`` makes the unclipped joint-0 torque 10; the post-transform
    L-infinity clip pins it to 2.
    """
    jac = _well_conditioned_jacobian()  # joint-0 controls x; joints 1-2 control y, z.
    controller = _make_controller(jacobian=jac)
    state = _snapshot(np.array([0.0, 0.0, 0.0]))
    n_hat = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    bounds = _bounds(linf=2.0, cap=10.0)
    out = controller.compute_override(state, [n_hat], bounds)
    assert out.shape == (7,)
    # Joint 0 is the only joint affected by an x-direction Cartesian
    # target on this Jacobian; without the clip it would equal cap = 10.
    # After the clip it equals action_linf_component = 2.
    assert out[0] == pytest.approx(2.0, abs=1e-9)
    # No torque component exceeds the L-inf envelope.
    assert np.all(np.abs(out) <= bounds.action_linf_component + 1e-12)


# ---------------------------------------------------------------------------
# 5. Singular configurations stay finite (damped pseudoinverse)
# ---------------------------------------------------------------------------


def test_near_singular_jacobian_returns_finite_torques() -> None:
    """A near-singular Jacobian (one row near zero) must not produce inf/nan torques.

    The damped pseudoinverse ``J^T (J J^T + damping^2 I)^{-1}`` keeps
    the gram matrix invertible even when ``J`` loses row rank. The
    cost of the damping is that the realised Cartesian magnitude
    can be below the requested target by a factor of
    ``damping^2 / sigma_min^2`` (well-characterised; not silent).
    """
    # Degenerate Jacobian: the x-row is at the damping floor, y/z rows
    # full rank. Without damping, ``J J^T`` would be near-singular and
    # ``np.linalg.solve`` would explode.
    jac = np.zeros((3, 7), dtype=np.float64)
    jac[0, 0] = 1e-9  # near-zero x-row -> near-singular
    jac[1, 1] = 1.0
    jac[2, 2] = 1.0
    controller = _make_controller(jacobian=jac, damping=1e-2)
    state = _snapshot(np.array([0.0, 0.0, 0.0]))
    out = controller.compute_override(
        state,
        [np.array([1.0, 0.0, 0.0], dtype=np.float64)],
        _bounds(linf=100.0, cap=5.0),
    )
    assert out.shape == (7,)
    assert np.all(np.isfinite(out))
    # The damping makes the realised x-acceleration under-deliver; the
    # contract is "finite + bounded", not "exact magnitude".
    assert float(np.linalg.norm(out)) < 1e3
