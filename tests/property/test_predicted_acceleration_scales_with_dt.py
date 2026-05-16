# SPDX-License-Identifier: Apache-2.0
"""Property test: predicted Cartesian acceleration scales with 1/dt (review P0-1).

External-review finding P0-1 (2026-05-16): the EGO_ONLY filter's
partner-disturbance term is built from
``_predicted_cartesian_accel(snap_now, snap_pred)`` which historically
returned ``snap_pred.velocity - snap_now.velocity`` (a velocity *delta*)
without dividing by ``dt``. Under the M3 constant-velocity predictor the
delta is always zero so the unit error was latent; the moment a Phase-1
predictor (AoI-conditioned, learned) supplies a nonzero forecast the CBF
RHS is wrong by a factor of ``1/dt``.

These property tests pin the corrected physics — predicted acceleration
``a = (v_pred - v_now) / dt`` — for any positive ``dt`` and any velocity
delta, *and* re-pin the Phase-0 constant-velocity invariant (zero delta
yields zero acceleration for every ``dt``). The assertion shape reflects
the correct physics, not the prior bug, so the file fails today and
passes after the fix lands.

References:
- ADR-004 §Decisions ("Predicted-acceleration units" paragraph added in
  this sprint).
- ``src/concerto/safety/cbf_qp.py::_predicted_cartesian_accel`` —
  the helper under test.
- ``src/concerto/safety/conformal.py::update_lambda_from_predictor`` —
  the wired-up caller that already supplies ``dt`` to its predictor; the
  EGO_ONLY filter must take the same parameter for consistency.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from concerto.safety.cbf_qp import AgentSnapshot, _predicted_cartesian_accel


def _snap(position: np.ndarray, velocity: np.ndarray, *, radius: float = 0.1) -> AgentSnapshot:
    """Build a per-agent snapshot with float64 arrays."""
    return AgentSnapshot(
        position=position.astype(np.float64, copy=True),
        velocity=velocity.astype(np.float64, copy=True),
        radius=radius,
    )


@given(
    dt=st.floats(min_value=1e-4, max_value=1.0, allow_nan=False, allow_infinity=False),
    delta=st.lists(
        st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        min_size=3,
        max_size=3,
    ),
)
@settings(max_examples=200, deadline=None)
def test_predicted_cartesian_accel_equals_delta_v_over_dt(dt: float, delta: list[float]) -> None:
    """For any nonzero predictor delta ``Δv``, the result is ``Δv / dt``."""
    v_now = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    v_pred = np.asarray(delta, dtype=np.float64)
    position = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    snap_now = _snap(position, v_now)
    snap_pred = _snap(position + v_pred * dt, v_pred)

    accel = _predicted_cartesian_accel(snap_now=snap_now, snap_pred=snap_pred, dt=dt)
    expected = v_pred / dt

    np.testing.assert_allclose(accel, expected, rtol=1e-9, atol=1e-12)


def test_constant_velocity_predictor_yields_zero_accel_invariant_in_dt() -> None:
    """ADR-004 §Decisions: constant-velocity stub returns zero accel for any ``dt``.

    Zero velocity delta divided by any positive ``dt`` is zero — the
    Phase-0 latent-bug regime is preserved by the corrected formula.
    """
    v = np.array([1.0, 2.0, -0.5], dtype=np.float64)
    position = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    snap_now = _snap(position, v)
    snap_pred = _snap(position + v * 0.05, v.copy())  # constant velocity

    for dt in (1e-3, 0.01, 0.05, 0.1, 1.0):
        accel = _predicted_cartesian_accel(snap_now=snap_now, snap_pred=snap_pred, dt=dt)
        np.testing.assert_array_equal(accel, np.zeros(3, dtype=np.float64))


@pytest.mark.parametrize("dt", [0.001, 0.01, 0.05, 0.1, 0.5])
def test_scaling_law_holds_under_dt_sweep(dt: float) -> None:
    """For a fixed Δv, ``accel * dt == Δv`` exactly (closed-form check).

    Pins the unit relationship explicitly: predicted acceleration times
    the lookahead horizon must reproduce the predictor's velocity delta.
    Catches any future regression that, e.g., divides by ``dt**2`` or
    by ``2 * dt``.
    """
    delta_v = np.array([3.0, -1.5, 0.25], dtype=np.float64)
    v_now = np.zeros(3, dtype=np.float64)
    position = np.array([0.5, 0.5, 0.5], dtype=np.float64)
    snap_now = _snap(position, v_now)
    snap_pred = _snap(position + delta_v * dt, delta_v)

    accel = _predicted_cartesian_accel(snap_now=snap_now, snap_pred=snap_pred, dt=dt)
    np.testing.assert_allclose(accel * dt, delta_v, rtol=1e-12, atol=1e-14)
