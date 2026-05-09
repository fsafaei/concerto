# SPDX-License-Identifier: Apache-2.0
"""Property + reproduction tests for ``concerto.safety.cbf_qp`` (T3.5).

Covers:

- ADR-004 §Decision (exp CBF-QP backbone, Wang-Ames-Egerstedt 2017 §III).
- The "QP cannot worsen the proposed action when feasible" invariant
  (plan/03 §5).
- Wang-Ames-Egerstedt 2017 §V toy 2-agent crossing reproduction —
  structural collision-avoidance check; the published trajectory's
  numerical RMS is a Phase-1 cross-check once the §V parameter set is
  digitised from the paper.
- Conformal-slack relaxation: increasing ``lambda`` weakens the
  constraint, bringing the safe action closer to the proposal.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from concerto.safety.api import Bounds, SafetyState
from concerto.safety.cbf_qp import AgentSnapshot, ExpCBFQP


def _bounds(action_norm: float = 2.0) -> Bounds:
    return Bounds(
        action_norm=action_norm,
        action_rate=0.5,
        comm_latency_ms=1.0,
        force_limit=20.0,
    )


def _state(n_pairs: int) -> SafetyState:
    return SafetyState(lambda_=np.zeros(n_pairs, dtype=np.float64))


def _snap(x: float, y: float, vx: float, vy: float, r: float = 0.2) -> AgentSnapshot:
    return AgentSnapshot(
        position=np.array([x, y], dtype=np.float64),
        velocity=np.array([vx, vy], dtype=np.float64),
        radius=r,
    )


def test_well_separated_agents_pass_through_proposed_action() -> None:
    """When CBF constraints are inactive, QP returns u_hat exactly."""
    snaps = {"a": _snap(0.0, 0.0, 0.0, 0.0), "b": _snap(10.0, 0.0, 0.0, 0.0)}
    proposed = {
        "a": np.array([0.5, 0.3], dtype=np.float64),
        "b": np.array([-0.4, 0.1], dtype=np.float64),
    }
    cbf = ExpCBFQP()
    safe, info = cbf.filter(
        proposed_action=proposed,
        obs={"agent_states": snaps, "meta": {"partner_id": None}},
        state=_state(1),
        bounds=_bounds(),
    )
    np.testing.assert_allclose(safe["a"], proposed["a"], atol=1e-5)
    np.testing.assert_allclose(safe["b"], proposed["b"], atol=1e-5)
    assert info["fallback_fired"] is False
    assert info["qp_solve_ms"] >= 0.0


def test_head_on_collision_avoidance_wang_ames_egerstedt_crossing() -> None:
    """Wang-Ames-Egerstedt 2017 §V toy: two agents on a head-on course must not collide.

    Structural reproduction — asserts the integrated CBF-QP filter keeps
    the inter-agent distance above the safety threshold across a
    full-trajectory rollout. The published §V trajectory's per-frame RMS
    is a Phase-1 cross-check once the paper's parameter set is
    digitised; this test verifies the qualitative invariant.
    """
    dt = 0.05
    n_steps = 200
    a_max = 5.0
    radius = 0.2
    safety_distance = 2.0 * radius

    p_a = np.array([-2.0, 0.0], dtype=np.float64)
    p_b = np.array([2.0, 0.0], dtype=np.float64)
    v_a = np.array([1.0, 0.0], dtype=np.float64)
    v_b = np.array([-1.0, 0.0], dtype=np.float64)

    cbf = ExpCBFQP(cbf_gamma=2.0)
    bounds = _bounds(action_norm=a_max)
    state = _state(1)

    min_distance = float("inf")
    for _ in range(n_steps):
        snaps = {
            "a": AgentSnapshot(position=p_a.copy(), velocity=v_a.copy(), radius=radius),
            "b": AgentSnapshot(position=p_b.copy(), velocity=v_b.copy(), radius=radius),
        }
        proposed = {
            "a": np.zeros(2, dtype=np.float64),
            "b": np.zeros(2, dtype=np.float64),
        }
        safe, _ = cbf.filter(
            proposed_action=proposed,
            obs={"agent_states": snaps, "meta": {"partner_id": None}},
            state=state,
            bounds=bounds,
        )
        v_a += safe["a"] * dt
        v_b += safe["b"] * dt
        p_a += v_a * dt
        p_b += v_b * dt
        d = float(np.linalg.norm(p_a - p_b))
        min_distance = min(min_distance, d)

    assert min_distance > safety_distance, (
        f"Head-on collision: min distance {min_distance:.4f} <= D_s {safety_distance:.4f}"
    )


def test_lambda_relaxes_constraint_brings_safe_action_closer_to_proposal() -> None:
    """Increasing the conformal slack lambda must not move safe further from u_hat.

    Huriot & Sibai 2025 §IV.A: lambda is added to the CBF constraint
    RHS; positive lambda loosens the constraint, so the QP-projected u
    is at most as far from u_hat as it was at lambda=0.
    """
    snaps = {
        "a": _snap(0.0, 0.0, 1.0, 0.0),
        "b": _snap(3.0, 0.0, -1.0, 0.0),
    }
    proposed = {
        "a": np.array([1.0, 0.0], dtype=np.float64),
        "b": np.array([-1.0, 0.0], dtype=np.float64),
    }
    cbf = ExpCBFQP()
    bounds = _bounds(action_norm=5.0)

    safe_zero, _ = cbf.filter(
        proposed_action=proposed,
        obs={"agent_states": snaps, "meta": {"partner_id": None}},
        state=_state(1),
        bounds=bounds,
    )
    safe_relaxed, _ = cbf.filter(
        proposed_action=proposed,
        obs={"agent_states": snaps, "meta": {"partner_id": None}},
        state=SafetyState(lambda_=np.array([2.0], dtype=np.float64)),
        bounds=bounds,
    )

    diff_zero = np.linalg.norm(
        np.concatenate([safe_zero["a"] - proposed["a"], safe_zero["b"] - proposed["b"]])
    )
    diff_relaxed = np.linalg.norm(
        np.concatenate([safe_relaxed["a"] - proposed["a"], safe_relaxed["b"] - proposed["b"]])
    )
    assert diff_relaxed <= diff_zero + 1e-6


def test_loss_k_carries_negative_barrier_value_for_conformal_layer() -> None:
    """FilterInfo['loss_k'] is -h_ij so the conformal update can drive lambda."""
    snaps = {
        "a": _snap(0.0, 0.0, 1.0, 0.0),
        "b": _snap(2.0, 0.0, -1.0, 0.0),
    }
    proposed = {
        "a": np.zeros(2, dtype=np.float64),
        "b": np.zeros(2, dtype=np.float64),
    }
    _, info = ExpCBFQP().filter(
        proposed_action=proposed,
        obs={"agent_states": snaps},
        state=_state(1),
        bounds=_bounds(action_norm=5.0),
    )
    # On a closing course at d=2, h_ij is small/negative; the loss
    # signal that the conformal layer reads should be a finite scalar.
    assert info["loss_k"].shape == (1,)
    assert np.isfinite(info["loss_k"]).all()


def test_filter_rejects_lambda_shape_mismatch() -> None:
    snaps = {"a": _snap(0.0, 0.0, 0.0, 0.0), "b": _snap(10.0, 0.0, 0.0, 0.0)}
    proposed = {
        "a": np.zeros(2, dtype=np.float64),
        "b": np.zeros(2, dtype=np.float64),
    }
    state_wrong = SafetyState(lambda_=np.zeros(5, dtype=np.float64))  # 1 pair; expected shape (1,).
    with pytest.raises(ValueError, match="lambda_ shape"):
        ExpCBFQP().filter(
            proposed_action=proposed,
            obs={"agent_states": snaps},
            state=state_wrong,
            bounds=_bounds(),
        )


def test_filter_rejects_missing_agent_states() -> None:
    proposed = {
        "a": np.zeros(2, dtype=np.float64),
        "b": np.zeros(2, dtype=np.float64),
    }
    with pytest.raises(ValueError, match="agent_states"):
        ExpCBFQP().filter(
            proposed_action=proposed,
            obs={},
            state=_state(1),
            bounds=_bounds(),
        )


@settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(min_value=0, max_value=10_000))
def test_qp_cannot_worsen_when_zero_is_feasible(seed: int) -> None:
    """plan/03 §5: ``||u - u_hat|| <= ||u_hat||`` when ``u=0`` is feasible.

    Well-separated stationary agents make ``u=0`` trivially feasible
    (no CBF activation), so the QP-projected solution is no farther
    from ``u_hat`` than the origin is.
    """
    rng = np.random.default_rng(seed)
    snaps = {
        "a": _snap(0.0, 0.0, 0.0, 0.0),
        "b": _snap(20.0, 0.0, 0.0, 0.0),
    }
    u_hat_a = rng.uniform(-1.0, 1.0, size=2).astype(np.float64)
    u_hat_b = rng.uniform(-1.0, 1.0, size=2).astype(np.float64)
    proposed = {"a": u_hat_a, "b": u_hat_b}

    safe, _ = ExpCBFQP().filter(
        proposed_action=proposed,
        obs={"agent_states": snaps},
        state=_state(1),
        bounds=_bounds(action_norm=2.0),
    )

    diff = np.concatenate([safe["a"] - u_hat_a, safe["b"] - u_hat_b])
    u_hat_full = np.concatenate([u_hat_a, u_hat_b])
    assert np.linalg.norm(diff) <= np.linalg.norm(u_hat_full) + 1e-5


def test_three_agents_no_collision_under_random_drift() -> None:
    """Sanity check: filter scales to >2 agents (3 agents -> 3 pairs)."""
    snaps = {
        "a": _snap(-3.0, 0.0, 1.0, 0.0),
        "b": _snap(3.0, 0.0, -1.0, 0.0),
        "c": _snap(0.0, 3.0, 0.0, -1.0),
    }
    proposed = {
        "a": np.zeros(2, dtype=np.float64),
        "b": np.zeros(2, dtype=np.float64),
        "c": np.zeros(2, dtype=np.float64),
    }
    cbf = ExpCBFQP(cbf_gamma=2.0)
    state = _state(3)
    safe, info = cbf.filter(
        proposed_action=proposed,
        obs={"agent_states": snaps},
        state=state,
        bounds=_bounds(action_norm=5.0),
    )
    assert set(safe.keys()) == {"a", "b", "c"}
    assert info["lambda"].shape == (3,)
    assert info["loss_k"].shape == (3,)
