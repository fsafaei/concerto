# SPDX-License-Identifier: Apache-2.0
"""Property + unit tests for ``concerto.safety.braking`` (T3.7).

Covers ADR-004 risk-mitigation #1 (Wang-Ames-Egerstedt 2017 eq. 17
hybrid braking controller as the per-step backstop independent of QP
feasibility). Plan/03 §4 T3.7 acceptance: under injected
catastrophic-violation state, ``fired_flag`` is True; otherwise silent.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from concerto.safety.api import Bounds
from concerto.safety.braking import (
    DEFAULT_TAU_BRAKE,
    compute_min_ttc,
    maybe_brake,
)
from concerto.safety.cbf_qp import AgentSnapshot


def _bounds(action_norm: float = 5.0) -> Bounds:
    return Bounds(
        action_norm=action_norm,
        action_rate=0.5,
        comm_latency_ms=1.0,
        force_limit=20.0,
    )


def _snap(x: float, y: float, vx: float, vy: float, r: float = 0.2) -> AgentSnapshot:
    return AgentSnapshot(
        position=np.array([x, y], dtype=np.float64),
        velocity=np.array([vx, vy], dtype=np.float64),
        radius=r,
    )


def test_compute_min_ttc_well_separated_returns_inf() -> None:
    snaps = {"a": _snap(0.0, 0.0, 0.0, 0.0), "b": _snap(10.0, 0.0, 0.0, 0.0)}
    ttc = compute_min_ttc(snaps)
    assert ttc[("a", "b")] == math.inf


def test_compute_min_ttc_head_on_finite_positive_root() -> None:
    """Two agents at (-1, 0) and (1, 0) closing at 2 m/s ⇒ TTC ~0.8 s for D_s=0.4."""
    snaps = {
        "a": _snap(-1.0, 0.0, 1.0, 0.0),
        "b": _snap(1.0, 0.0, -1.0, 0.0),
    }
    ttc = compute_min_ttc(snaps)
    # |Dp|=2, |Dv|=2, D_s=0.4. TTC = (2 - 0.4)/2 = 0.8 s.
    assert ttc[("a", "b")] == pytest.approx(0.8, abs=1e-6)


def test_compute_min_ttc_already_in_collision_is_zero() -> None:
    snaps = {"a": _snap(0.0, 0.0, 0.0, 0.0), "b": _snap(0.1, 0.0, 0.0, 0.0)}
    # |Dp|=0.1 < D_s=0.4 ⇒ already in collision.
    ttc = compute_min_ttc(snaps)
    assert ttc[("a", "b")] == 0.0


def test_compute_min_ttc_separating_agents_returns_inf() -> None:
    """Agents moving apart can never collide on current trajectory."""
    snaps = {
        "a": _snap(-1.0, 0.0, -1.0, 0.0),
        "b": _snap(1.0, 0.0, 1.0, 0.0),
    }
    ttc = compute_min_ttc(snaps)
    assert ttc[("a", "b")] == math.inf


def test_compute_min_ttc_perpendicular_no_collision() -> None:
    """Tangential miss — closest approach > D_s ⇒ TTC = inf."""
    snaps = {
        "a": _snap(0.0, 0.0, 1.0, 0.0),
        "b": _snap(2.0, 1.0, -1.0, 0.0),  # offset by 1 m on y; D_s=0.4
    }
    ttc = compute_min_ttc(snaps)
    assert ttc[("a", "b")] == math.inf


def test_compute_min_ttc_empty_when_single_agent() -> None:
    snaps = {"a": _snap(0.0, 0.0, 0.0, 0.0)}
    assert compute_min_ttc(snaps) == {}


def test_maybe_brake_silent_when_well_separated() -> None:
    snaps = {"a": _snap(0.0, 0.0, 0.0, 0.0), "b": _snap(10.0, 0.0, 0.0, 0.0)}
    proposed = {
        "a": np.array([0.5, 0.3], dtype=np.float64),
        "b": np.array([-0.4, 0.1], dtype=np.float64),
    }
    override, fired = maybe_brake(proposed, snaps, bounds=_bounds())
    assert override is None
    assert fired is False


def test_maybe_brake_fires_under_catastrophic_state() -> None:
    """Plan/03 §4 T3.7: injected catastrophic state ⇒ fired_flag True."""
    # Agents 0.05 m apart, closing at 2 m/s ⇒ TTC < tau_brake.
    snaps = {
        "a": _snap(-0.05, 0.0, 1.0, 0.0),
        "b": _snap(0.5, 0.0, -1.0, 0.0),
    }
    proposed = {
        "a": np.zeros(2, dtype=np.float64),
        "b": np.zeros(2, dtype=np.float64),
    }
    bounds = _bounds(action_norm=5.0)
    override, fired = maybe_brake(proposed, snaps, bounds=bounds)
    assert fired is True
    assert override is not None
    # Magnitude is bounds.action_norm in opposite directions on the
    # line between centres (push-apart).
    assert np.linalg.norm(override["a"]) == pytest.approx(bounds.action_norm)
    assert np.linalg.norm(override["b"]) == pytest.approx(bounds.action_norm)
    # Push apart: a's override has negative x (a is to the left of b),
    # b's has positive x. n_hat = (Dp / |Dp|) where Dp = p_a - p_b.
    n_hat_x = (-0.05 - 0.5) / abs(-0.05 - 0.5)  # = -1
    np.testing.assert_allclose(override["a"], [bounds.action_norm * n_hat_x, 0.0])
    np.testing.assert_allclose(override["b"], [-bounds.action_norm * n_hat_x, 0.0])


def test_maybe_brake_silent_when_separating_at_close_range() -> None:
    """Close-range agents already moving apart should NOT trigger brake."""
    snaps = {
        "a": _snap(-0.25, 0.0, -1.0, 0.0),
        "b": _snap(0.25, 0.0, 1.0, 0.0),
    }
    proposed = {
        "a": np.zeros(2, dtype=np.float64),
        "b": np.zeros(2, dtype=np.float64),
    }
    override, fired = maybe_brake(proposed, snaps, bounds=_bounds())
    assert override is None
    assert fired is False


def test_maybe_brake_fires_when_already_in_collision() -> None:
    """Already in collision (TTC = 0) is the tightest brake-fire case."""
    snaps = {
        "a": _snap(0.0, 0.0, 1.0, 0.0),
        "b": _snap(0.1, 0.0, -1.0, 0.0),  # |Dp| < D_s
    }
    proposed = {
        "a": np.zeros(2, dtype=np.float64),
        "b": np.zeros(2, dtype=np.float64),
    }
    override, fired = maybe_brake(proposed, snaps, bounds=_bounds())
    assert fired is True
    assert override is not None


def test_maybe_brake_handles_coincident_centres_without_nan() -> None:
    """Coincident centres: TTC=0 ⇒ fire; override direction falls back to +x."""
    snaps = {
        "a": _snap(0.0, 0.0, 0.0, 0.0),
        "b": _snap(0.0, 0.0, 0.0, 0.0),
    }
    proposed = {
        "a": np.zeros(2, dtype=np.float64),
        "b": np.zeros(2, dtype=np.float64),
    }
    bounds = _bounds(action_norm=5.0)
    override, fired = maybe_brake(proposed, snaps, bounds=bounds)
    assert fired is True
    assert override is not None
    assert not np.any(np.isnan(override["a"]))
    assert not np.any(np.isnan(override["b"]))
    np.testing.assert_allclose(override["a"], [bounds.action_norm, 0.0])
    np.testing.assert_allclose(override["b"], [-bounds.action_norm, 0.0])


def test_maybe_brake_three_agents_only_offending_pair_overrides() -> None:
    """Only agents in the trip-pair receive brake override; others zero."""
    snaps = {
        "a": _snap(-0.1, 0.0, 1.0, 0.0),
        "b": _snap(0.5, 0.0, -1.0, 0.0),
        "c": _snap(20.0, 0.0, 0.0, 0.0),  # far away
    }
    proposed = {
        "a": np.zeros(2, dtype=np.float64),
        "b": np.zeros(2, dtype=np.float64),
        "c": np.zeros(2, dtype=np.float64),
    }
    override, fired = maybe_brake(proposed, snaps, bounds=_bounds(action_norm=5.0))
    assert fired is True
    assert override is not None
    assert np.linalg.norm(override["a"]) == pytest.approx(5.0)
    assert np.linalg.norm(override["b"]) == pytest.approx(5.0)
    np.testing.assert_array_equal(override["c"], [0.0, 0.0])


def test_default_tau_brake_is_100ms() -> None:
    assert DEFAULT_TAU_BRAKE == 0.100


def test_maybe_brake_custom_tau_brake_threshold_is_respected() -> None:
    """A larger tau_brake makes the fallback fire earlier."""
    # TTC ~0.5s — within tau=1s but outside default 100ms.
    snaps = {
        "a": _snap(-0.4, 0.0, 1.0, 0.0),
        "b": _snap(0.6, 0.0, -1.0, 0.0),
    }
    proposed = {"a": np.zeros(2, dtype=np.float64), "b": np.zeros(2, dtype=np.float64)}
    _, fired_default = maybe_brake(proposed, snaps, bounds=_bounds())
    _, fired_relaxed = maybe_brake(proposed, snaps, bounds=_bounds(), tau_brake=1.0)
    assert fired_default is False
    assert fired_relaxed is True


@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    closing_speed=st.floats(min_value=0.5, max_value=5.0),
    initial_gap=st.floats(min_value=0.05, max_value=0.4),
)
def test_maybe_brake_fires_when_closing_fast_within_tau(
    closing_speed: float, initial_gap: float
) -> None:
    """Hypothesis: any close head-on encounter within tau_brake fires.

    Skips boundary cases ``|expected_ttc - tau_brake| < 5 ms`` where
    floating-point in the quadratic-root TTC computation can flip the
    inequality vs. an analytic ``initial_gap / closing_speed``.
    """
    radius = 0.2
    safety_distance = 2 * radius
    expected_ttc = initial_gap / closing_speed
    # Skip the boundary band where roundoff can flip the strict inequality.
    if abs(expected_ttc - DEFAULT_TAU_BRAKE) < 5e-3:
        return

    half_dp = (safety_distance + initial_gap) / 2
    snaps = {
        "a": _snap(-half_dp, 0.0, closing_speed / 2, 0.0, r=radius),
        "b": _snap(half_dp, 0.0, -closing_speed / 2, 0.0, r=radius),
    }
    proposed = {"a": np.zeros(2, dtype=np.float64), "b": np.zeros(2, dtype=np.float64)}
    bounds = _bounds(action_norm=5.0)

    expected_fire = expected_ttc < DEFAULT_TAU_BRAKE
    _, fired = maybe_brake(proposed, snaps, bounds=bounds)
    assert fired == expected_fire


def test_maybe_brake_does_not_route_through_qp() -> None:
    """ADR-004 risk-mitigation #1 + plan/03 §8: braking bypasses the QP.

    Smoke test that ``braking.py`` does not reference any QPSolver /
    cvxpy / qpsolvers symbol — the override must be computable from
    kinematic state alone.
    """
    import pathlib

    import concerto.safety.braking as br

    assert br.__file__ is not None
    text = pathlib.Path(br.__file__).read_text(encoding="utf-8")
    for forbidden in (
        "QPSolver",
        "ClarabelSolver",
        "OSQPSolver",
        "qpsolvers",
        "cvxpy",
    ):
        assert forbidden not in text, (
            f"braking.py must not depend on QP-solver symbols "
            f"(found {forbidden!r}); ADR-004 risk-mitigation #1 requires "
            "the per-step backstop be independent of QP feasibility."
        )
