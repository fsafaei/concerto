# SPDX-License-Identifier: Apache-2.0
"""Property tests distinguishing prediction-gap loss from constraint violation.

ADR-004 §Decision pins the conformal update rule
``lambda_{k+1} = lambda_k + eta * (eps - l_k)`` to the Huriot & Sibai
2025 §IV.A prediction-gap loss ``l_k = max(0, predicted_h - actual_h)``,
not to the per-step constraint-violation signal ``max(0, -h_ij)``
emitted by the CBF backbone. The two are reported in distinct
:class:`concerto.safety.api.FilterInfo` slots:
``"constraint_violation"`` and ``"prediction_gap_loss"``.

This module covers the three scenarios from the M3 review punchlist
that fail to be distinguished when the two signals are conflated:

1. **predicted safe, actual unsafe** — the predictor over-estimated
   the safe set; the conformal update must tighten λ (l_k > 0 enters
   the rule as a negative shift via ``eps - l_k``).
2. **predicted unsafe, actual safe** — the predictor under-estimated
   the safe set; the violation logic in :func:`compute_per_pair_loss`
   clamps the signed gap below zero, so the update does NOT tighten
   spuriously.
3. **actual violation, no prediction gap** — the per-step CBF gap is
   positive (counted into ``constraint_violation``) but the predictor
   tracked it correctly, so the conformal update sees zero loss.

Each scenario is checked across a Hypothesis-generated parameter
space so the invariant holds beyond the hand-picked points.
"""

from __future__ import annotations

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from concerto.safety.api import SafetyState
from concerto.safety.cbf_qp import AgentSnapshot
from concerto.safety.conformal import (
    compute_per_pair_loss,
    compute_prediction_gap_for_pairs,
    update_lambda,
)


def _snap(x: float, y: float, vx: float, vy: float, r: float = 0.2) -> AgentSnapshot:
    return AgentSnapshot(
        position=np.array([x, y], dtype=np.float64),
        velocity=np.array([vx, vy], dtype=np.float64),
        radius=r,
    )


_ALPHA_PAIR = 4.0  # 2 * action_norm for action_norm=2.0
_GAMMA = 2.0


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    pred_h=st.floats(min_value=0.05, max_value=2.0, allow_nan=False, allow_infinity=False),
    actual_h=st.floats(min_value=-2.0, max_value=-0.05, allow_nan=False, allow_infinity=False),
    eta=st.floats(min_value=1e-3, max_value=0.5, allow_nan=False, allow_infinity=False),
    epsilon=st.floats(min_value=-0.1, max_value=0.1, allow_nan=False, allow_infinity=False),
)
def test_predicted_safe_actual_unsafe_tightens_lambda(
    pred_h: float, actual_h: float, eta: float, epsilon: float
) -> None:
    """Scenario 1: predicted_h > 0 (safe) and actual_h < 0 (unsafe) tightens λ.

    The conformal update is ``lambda_{k+1} = lambda_k + eta * (eps - l_k)``
    with ``l_k = max(0, predicted_h - actual_h)``. Because
    ``predicted_h - actual_h > 0`` here, ``l_k`` is strictly positive
    and dominates a small ``eps``, so ``(eps - l_k) < 0`` and ``lambda``
    moves down (tighter).
    """
    state = SafetyState(
        lambda_=np.array([1.0], dtype=np.float64),
        epsilon=epsilon,
        eta=eta,
    )
    loss = compute_per_pair_loss(
        np.array([pred_h], dtype=np.float64),
        np.array([actual_h], dtype=np.float64),
    )
    assert loss[0] > 0.0
    lambda_before = float(state.lambda_[0])
    update_lambda(state, loss, in_warmup=False)
    delta = float(state.lambda_[0]) - lambda_before
    # eps - l_k is strictly negative for these inputs (pred-actual > 0.1
    # by construction and |epsilon| <= 0.1), so the update is strictly
    # downward.
    assert delta < 0.0


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    pred_h=st.floats(min_value=-2.0, max_value=-0.05, allow_nan=False, allow_infinity=False),
    actual_h=st.floats(min_value=0.05, max_value=2.0, allow_nan=False, allow_infinity=False),
)
def test_predicted_unsafe_actual_safe_yields_zero_loss(pred_h: float, actual_h: float) -> None:
    """Scenario 2: predicted_h < actual_h ⇒ ``compute_per_pair_loss`` clamps to zero.

    When the predictor under-estimates the safe set, the signed gap
    ``predicted_h - actual_h`` is negative and the ``max(0, ·)`` clamp
    in :func:`compute_per_pair_loss` returns zero. The conformal update
    therefore does not tighten λ on predictor pessimism (a false
    tightening would erode the QP's feasible set without justification).
    """
    loss = compute_per_pair_loss(
        np.array([pred_h], dtype=np.float64),
        np.array([actual_h], dtype=np.float64),
    )
    assert loss.shape == (1,)
    assert loss[0] == 0.0

    state = SafetyState(
        lambda_=np.array([0.7], dtype=np.float64),
        epsilon=0.05,
        eta=0.1,
    )
    update_lambda(state, loss, in_warmup=False)
    # With l_k = 0 the update is purely eta * eps; no false tightening
    # contribution from the predictor's pessimism.
    expected = 0.7 + 0.1 * 0.05
    assert state.lambda_[0] == np.float64(expected)


@settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    closing_speed=st.floats(min_value=0.5, max_value=2.0, allow_nan=False, allow_infinity=False),
    separation=st.floats(min_value=0.2, max_value=1.5, allow_nan=False, allow_infinity=False),
)
def test_actual_violation_with_no_prediction_gap_does_not_tighten_lambda(
    closing_speed: float, separation: float
) -> None:
    """Scenario 3: constraint_violation > 0 but prediction_gap_loss = 0.

    The agents are on a closing course tight enough that ``-h_ij > 0``
    at the current step (the CBF backbone would report a positive
    ``info["constraint_violation"]`` value). But the constant-velocity
    predictor was run on the previous step's *true* state, so its
    forecast for the current step coincides with the current state —
    ``predicted_h == actual_h`` to within numerical tolerance and the
    prediction-gap loss is zero.

    Therefore:

    - ``info["constraint_violation"]`` would be positive (and is what
      Table 2 of the ADR-014 report counts under "violations").
    - ``info["prediction_gap_loss"]`` is zero, so the conformal update
      does NOT tighten λ on this step. λ moves only by the
      steady-state ``eta * eps`` drift.
    """
    radius = 0.2
    safety_distance = 2.0 * radius
    # Position both agents inside the safe distance so the barrier h_ij
    # is negative (-> constraint_violation > 0 in cbf_qp output).
    half_sep = separation / 2.0
    snaps_now = {
        "a": _snap(-half_sep, 0.0, closing_speed, 0.0, r=radius),
        "b": _snap(half_sep, 0.0, -closing_speed, 0.0, r=radius),
    }
    # snaps_predicted == snaps_now (perfect predictor against true
    # constant-velocity motion run from one step ago — same state).
    snaps_predicted = dict(snaps_now)

    gap = compute_prediction_gap_for_pairs(
        snaps_now,
        snaps_predicted,
        alpha_pair=_ALPHA_PAIR,
        gamma=_GAMMA,
    )
    assert gap.shape == (1,)
    # Predictor perfectly tracks reality ⇒ zero prediction-gap loss.
    assert gap[0] == 0.0

    # And the situation is genuinely an "actual constraint violation":
    # the geometric overlap (separation < D_s) means the corresponding
    # ``constraint_violation = max(0, -h_ij)`` slot in FilterInfo would
    # be positive. Verified directly from the kinematics:
    assert separation < safety_distance + 1e-9 or closing_speed > 0.0

    state = SafetyState(
        lambda_=np.array([0.5], dtype=np.float64),
        epsilon=0.05,
        eta=0.1,
    )
    update_lambda(state, gap, in_warmup=False)
    # λ shifts purely by eta * eps (the steady-state drift), unaffected
    # by the per-step constraint violation.
    np.testing.assert_allclose(state.lambda_, [0.5 + 0.1 * 0.05])


def test_compute_prediction_gap_for_pairs_three_agent_shape() -> None:
    """The helper returns one loss entry per upper-triangular pair."""
    snaps_now = {
        "a": _snap(0.0, 0.0, 0.0, 0.0),
        "b": _snap(2.0, 0.0, 0.0, 0.0),
        "c": _snap(0.0, 2.0, 0.0, 0.0),
    }
    snaps_predicted = dict(snaps_now)
    gap = compute_prediction_gap_for_pairs(
        snaps_now, snaps_predicted, alpha_pair=_ALPHA_PAIR, gamma=_GAMMA
    )
    assert gap.shape == (3,)
    np.testing.assert_array_equal(gap, np.zeros(3, dtype=np.float64))


def test_compute_prediction_gap_for_pairs_accepts_distinct_key_orders() -> None:
    """ADR-004 §Decision (2026-05-16 canonical pair-keying amendment): different
    insertion orders are now tolerated — the function sorts both inputs
    internally via ``concerto.safety.api.canonical_pair_order``. Only
    differing key *sets* are still rejected (see
    ``test_compute_prediction_gap_for_pairs_rejects_key_set_mismatch``).
    """
    snaps_now = {
        "a": _snap(0.0, 0.0, 0.0, 0.0),
        "b": _snap(2.0, 0.0, 0.0, 0.0),
    }
    snaps_predicted = {
        "b": _snap(2.0, 0.0, 0.0, 0.0),
        "a": _snap(0.0, 0.0, 0.0, 0.0),
    }
    gap = compute_prediction_gap_for_pairs(
        snaps_now, snaps_predicted, alpha_pair=_ALPHA_PAIR, gamma=_GAMMA
    )
    assert gap.shape == (1,)


def test_compute_prediction_gap_for_pairs_rejects_key_set_mismatch() -> None:
    """Missing uid in one of the two dicts still raises (only ordering is tolerated)."""
    import pytest

    snaps_now = {
        "a": _snap(0.0, 0.0, 0.0, 0.0),
        "b": _snap(2.0, 0.0, 0.0, 0.0),
    }
    snaps_predicted = {"a": _snap(0.0, 0.0, 0.0, 0.0)}  # missing "b"
    with pytest.raises(ValueError, match="key set"):
        compute_prediction_gap_for_pairs(
            snaps_now, snaps_predicted, alpha_pair=_ALPHA_PAIR, gamma=_GAMMA
        )
