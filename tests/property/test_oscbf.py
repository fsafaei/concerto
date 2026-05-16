# SPDX-License-Identifier: Apache-2.0
"""Property + timing tests for ``concerto.safety.oscbf`` (T3.8).

Covers ADR-004 §Decision (Morton-Pavone 2025 §IV two-level QP), the
1 kHz solve-time target on 7-DOF (ADR-004 validation criterion 3),
slack relaxation under conflicting CBFs (Morton-Pavone §IV.D), and
the plan/03 §5 invariant that OSCBF output respects joint-space
limits.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from concerto.safety.oscbf import (
    DEFAULT_CBF_ALPHA,
    DEFAULT_SLACK_PENALTY,
    DEFAULT_W_JOINT,
    DEFAULT_W_OPERATIONAL,
    OSCBF,
    OSCBFConstraints,
    collision_constraint_row,
    joint_limit_constraint_row,
)


def _stack(rows: list[np.ndarray], rhs: list[float]) -> OSCBFConstraints:
    return OSCBFConstraints(a=np.vstack(rows), b=np.asarray(rhs, dtype=np.float64))


def test_constants_pinned() -> None:
    assert DEFAULT_W_JOINT == 1.0
    assert DEFAULT_W_OPERATIONAL == 1.0
    assert DEFAULT_SLACK_PENALTY == 100.0
    assert DEFAULT_CBF_ALPHA == 5.0


def test_oscbf_constraints_validates_dimensions() -> None:
    with pytest.raises(ValueError, match="row count mismatch"):
        OSCBFConstraints(
            a=np.zeros((3, 7), dtype=np.float64),
            b=np.zeros(2, dtype=np.float64),
        )
    with pytest.raises(ValueError, match="2-D"):
        OSCBFConstraints(
            a=np.zeros(7, dtype=np.float64),  # 1-D
            b=np.zeros(7, dtype=np.float64),
        )
    with pytest.raises(ValueError, match="1-D"):
        OSCBFConstraints(
            a=np.zeros((1, 7), dtype=np.float64),
            b=np.zeros((1, 1), dtype=np.float64),
        )


def test_oscbf_rejects_zero_n_joints() -> None:
    with pytest.raises(ValueError, match=">= 1"):
        OSCBF(n_joints=0)


def test_oscbf_n_joints_property() -> None:
    assert OSCBF(n_joints=7).n_joints == 7


def test_joint_limit_row_upper_returns_e_i() -> None:
    row, rhs = joint_limit_constraint_row(n_joints=7, joint_index=2, upper=1.5)
    expected = np.zeros(7, dtype=np.float64)
    expected[2] = 1.0
    np.testing.assert_array_equal(row, expected)
    assert rhs == 1.5


def test_joint_limit_row_lower_returns_neg_e_i() -> None:
    row, rhs = joint_limit_constraint_row(n_joints=7, joint_index=2, lower=-1.5)
    expected = np.zeros(7, dtype=np.float64)
    expected[2] = -1.0
    np.testing.assert_array_equal(row, expected)
    assert rhs == 1.5  # -lower = -(-1.5)


def test_joint_limit_row_rejects_both_or_neither() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        joint_limit_constraint_row(n_joints=7, joint_index=0, upper=1.0, lower=-1.0)
    with pytest.raises(ValueError, match="exactly one"):
        joint_limit_constraint_row(n_joints=7, joint_index=0)


def test_joint_limit_row_rejects_out_of_range_index() -> None:
    with pytest.raises(ValueError, match="not in"):
        joint_limit_constraint_row(n_joints=7, joint_index=10, upper=1.0)


def test_collision_row_matches_handwritten_form() -> None:
    """Manual derivation: n_hat = (1,0,0); row = -[1,0,0]; h = 0.8; rhs = alpha*h."""
    row, rhs = collision_constraint_row(
        n_joints=3,
        sphere_a_center=np.array([1.0, 0.0, 0.0], dtype=np.float64),
        sphere_b_center=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        sphere_a_radius=0.1,
        sphere_b_radius=0.1,
        sphere_a_jacobian=np.eye(3, dtype=np.float64),
        cbf_alpha=2.0,
    )
    np.testing.assert_allclose(row, [-1.0, 0.0, 0.0])
    assert rhs == pytest.approx(2.0 * (1.0 - 0.2))


def test_collision_row_with_b_jacobian_uses_difference() -> None:
    row, _ = collision_constraint_row(
        n_joints=3,
        sphere_a_center=np.array([1.0, 0.0, 0.0], dtype=np.float64),
        sphere_b_center=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        sphere_a_radius=0.1,
        sphere_b_radius=0.1,
        sphere_a_jacobian=np.eye(3, dtype=np.float64),
        sphere_b_jacobian=np.eye(3, dtype=np.float64),
    )
    # j_diff = J_a - J_b = 0; row = -n_hat^T 0 = 0.
    np.testing.assert_allclose(row, np.zeros(3))


def test_collision_row_lambda_propagates_to_rhs() -> None:
    _, rhs = collision_constraint_row(
        n_joints=3,
        sphere_a_center=np.array([1.0, 0.0, 0.0], dtype=np.float64),
        sphere_b_center=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        sphere_a_radius=0.1,
        sphere_b_radius=0.1,
        sphere_a_jacobian=np.eye(3, dtype=np.float64),
        cbf_alpha=2.0,
        lambda_ij=0.3,
    )
    assert rhs == pytest.approx(2.0 * 0.8 + 0.3)


def test_collision_row_handles_coincident_centres_without_nan() -> None:
    row, rhs = collision_constraint_row(
        n_joints=3,
        sphere_a_center=np.zeros(3, dtype=np.float64),
        sphere_b_center=np.zeros(3, dtype=np.float64),
        sphere_a_radius=0.1,
        sphere_b_radius=0.1,
        sphere_a_jacobian=np.eye(3, dtype=np.float64),
    )
    assert not np.any(np.isnan(row))
    assert not np.isnan(rhs)


def test_oscbf_well_separated_returns_q_dot_nom() -> None:
    """When CBF rows are inactive, OSCBF returns q_dot_nom (joint cost only)."""
    n = 7
    rng = np.random.default_rng(0)
    j = rng.standard_normal((6, n)) * 0.3
    q_dot_nom = np.zeros(n, dtype=np.float64)
    nu_nom = np.zeros(6, dtype=np.float64)

    rows: list[np.ndarray] = []
    rhs: list[float] = []
    for jdx in range(n):
        a, b = joint_limit_constraint_row(n_joints=n, joint_index=jdx, upper=10.0)
        rows.append(a)
        rhs.append(b)
        a, b = joint_limit_constraint_row(n_joints=n, joint_index=jdx, lower=-10.0)
        rows.append(a)
        rhs.append(b)

    oscbf = OSCBF(n_joints=n)
    q_dot = oscbf.solve(
        q_dot_nom=q_dot_nom,
        nu_nom=nu_nom,
        jacobian=j,
        constraints=_stack(rows, rhs),
    ).q_dot
    np.testing.assert_allclose(q_dot, q_dot_nom, atol=1e-5)


def test_oscbf_respects_joint_velocity_upper_bound() -> None:
    """plan/03 §5: OSCBF output respects joint-space limits."""
    n = 7
    j = np.eye(6, n, dtype=np.float64)
    q_dot_nom = np.full(n, 5.0, dtype=np.float64)  # exceeds upper of 1
    nu_nom = np.zeros(6, dtype=np.float64)

    rows: list[np.ndarray] = []
    rhs: list[float] = []
    for jdx in range(n):
        a, b = joint_limit_constraint_row(n_joints=n, joint_index=jdx, upper=1.0)
        rows.append(a)
        rhs.append(b)

    oscbf = OSCBF(n_joints=n, weight_operational=0.0, slack_penalty=1e6)
    q_dot = oscbf.solve(
        q_dot_nom=q_dot_nom,
        nu_nom=nu_nom,
        jacobian=j,
        constraints=_stack(rows, rhs),
    ).q_dot
    # All joints clamp at the upper bound 1.0 within solver tolerance.
    assert np.all(q_dot <= 1.0 + 1e-3)


def test_oscbf_solve_rejects_shape_mismatches() -> None:
    n = 7
    j = np.eye(6, n, dtype=np.float64)
    constraints = _stack(
        [np.zeros(n, dtype=np.float64)],
        [10.0],
    )
    oscbf = OSCBF(n_joints=n)

    with pytest.raises(ValueError, match="q_dot_nom"):
        oscbf.solve(
            q_dot_nom=np.zeros(n + 1, dtype=np.float64),
            nu_nom=np.zeros(6, dtype=np.float64),
            jacobian=j,
            constraints=constraints,
        )
    with pytest.raises(ValueError, match="jacobian"):
        oscbf.solve(
            q_dot_nom=np.zeros(n, dtype=np.float64),
            nu_nom=np.zeros(6, dtype=np.float64),
            jacobian=np.eye(6, n + 1, dtype=np.float64),
            constraints=constraints,
        )
    with pytest.raises(ValueError, match="nu_nom"):
        oscbf.solve(
            q_dot_nom=np.zeros(n, dtype=np.float64),
            nu_nom=np.zeros(7, dtype=np.float64),  # mismatched
            jacobian=j,
            constraints=constraints,
        )
    with pytest.raises(ValueError, match=r"constraints\.a"):
        oscbf.solve(
            q_dot_nom=np.zeros(n, dtype=np.float64),
            nu_nom=np.zeros(6, dtype=np.float64),
            jacobian=j,
            constraints=_stack([np.zeros(n + 1, dtype=np.float64)], [0.0]),
        )


def test_oscbf_slack_relaxation_keeps_qp_feasible_under_conflict() -> None:
    """Morton-Pavone 2025 §IV.D: conflicting CBFs become feasible via slack."""
    n = 7
    j = np.eye(6, n, dtype=np.float64)
    q_dot_nom = np.zeros(n, dtype=np.float64)
    nu_nom = np.zeros(6, dtype=np.float64)

    # q_dot[0] >= 1 AND q_dot[0] <= -1 — infeasible without slack.
    row_a = np.zeros(n, dtype=np.float64)
    row_a[0] = -1.0  # -q_dot[0] <= -1 ⇔ q_dot[0] >= 1
    row_b = np.zeros(n, dtype=np.float64)
    row_b[0] = 1.0  # q_dot[0] <= -1
    constraints = _stack([row_a, row_b], [-1.0, -1.0])

    oscbf = OSCBF(n_joints=n, slack_penalty=100.0)
    result = oscbf.solve(
        q_dot_nom=q_dot_nom,
        nu_nom=nu_nom,
        jacobian=j,
        constraints=constraints,
    )
    # No exception — the slack absorbed the conflict; result is finite.
    assert result.q_dot.shape == (n,)
    assert np.all(np.isfinite(result.q_dot))
    # Conflict ⇒ at least one row is slack-active (ADR-014 Table 2
    # external-review P0-3 telemetry contract; pre-fix this signal was
    # silently dropped).
    assert result.max_slack > 0.0
    assert len(result.active_rows) >= 1


def _build_seven_dof_problem(
    rng: np.random.Generator, n_collision_rows: int = 3
) -> tuple[np.ndarray, np.ndarray, np.ndarray, OSCBFConstraints]:
    n_joints = 7
    j = (rng.standard_normal((6, n_joints)) * 0.5).astype(np.float64)
    q_dot_nom = rng.uniform(-0.5, 0.5, size=n_joints).astype(np.float64)
    nu_nom = rng.uniform(-0.3, 0.3, size=6).astype(np.float64)

    rows: list[np.ndarray] = []
    rhs: list[float] = []
    for jdx in range(n_joints):
        r, b = joint_limit_constraint_row(n_joints=n_joints, joint_index=jdx, upper=2.0)
        rows.append(r)
        rhs.append(b)
        r, b = joint_limit_constraint_row(n_joints=n_joints, joint_index=jdx, lower=-2.0)
        rows.append(r)
        rhs.append(b)
    for _ in range(n_collision_rows):
        a_center = rng.standard_normal(3).astype(np.float64)
        b_center = (rng.standard_normal(3) + 1.0).astype(np.float64)
        r, b = collision_constraint_row(
            n_joints=n_joints,
            sphere_a_center=a_center,
            sphere_b_center=b_center,
            sphere_a_radius=0.05,
            sphere_b_radius=0.05,
            sphere_a_jacobian=(rng.standard_normal((3, n_joints)) * 0.3).astype(np.float64),
        )
        rows.append(r)
        rhs.append(b)

    return q_dot_nom, nu_nom, j, _stack(rows, rhs)


@pytest.mark.skipif(
    sys.gettrace() is not None,
    reason=(
        "Coverage instrumentation inflates timings 2-3x; the 1 kHz target "
        "is asserted in uncovered runs. The OSCBF correctness paths are "
        "covered by the other tests in this module."
    ),
)
def test_oscbf_mean_solve_time_under_1ms_on_7dof_arm() -> None:
    """ADR-004 validation criterion 3: < 1 ms mean solve on 7-DOF (CPU)."""
    rng = np.random.default_rng(0)
    q_dot_nom, nu_nom, jac, constraints = _build_seven_dof_problem(rng)
    oscbf = OSCBF(n_joints=7)

    # Warm-up to amortise first-call overhead in the timing window.
    for _ in range(5):
        oscbf.solve(
            q_dot_nom=q_dot_nom,
            nu_nom=nu_nom,
            jacobian=jac,
            constraints=constraints,
        )
    times: list[float] = []
    for _ in range(50):
        ms = oscbf.solve(
            q_dot_nom=q_dot_nom,
            nu_nom=nu_nom,
            jacobian=jac,
            constraints=constraints,
        ).solve_ms
        times.append(ms)
    mean_ms = float(np.mean(times))
    p95_ms = float(np.percentile(times, 95))
    assert mean_ms < 1.0, (
        f"OSCBF mean {mean_ms:.3f} ms exceeds 1 ms target "
        f"(p95={p95_ms:.3f} ms; ADR-004 validation criterion 3)."
    )


@settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(seed=st.integers(min_value=0, max_value=10_000))
def test_oscbf_random_nominal_within_action_rate_respects_joint_limits(
    seed: int,
) -> None:
    """plan/03 §5 invariant: any nominal within bounds ⇒ output respects limits."""
    rng = np.random.default_rng(seed)
    n_joints = 7
    j = (rng.standard_normal((6, n_joints)) * 0.3).astype(np.float64)

    # Nominal velocity within an action_rate envelope of 0.5.
    q_dot_nom = rng.uniform(-0.5, 0.5, size=n_joints).astype(np.float64)
    nu_nom = rng.uniform(-0.3, 0.3, size=6).astype(np.float64)

    upper, lower = 1.0, -1.0
    rows: list[np.ndarray] = []
    rhs: list[float] = []
    for jdx in range(n_joints):
        r, b = joint_limit_constraint_row(n_joints=n_joints, joint_index=jdx, upper=upper)
        rows.append(r)
        rhs.append(b)
        r, b = joint_limit_constraint_row(n_joints=n_joints, joint_index=jdx, lower=lower)
        rows.append(r)
        rhs.append(b)

    oscbf = OSCBF(n_joints=n_joints, slack_penalty=1e6)
    q_dot = oscbf.solve(
        q_dot_nom=q_dot_nom,
        nu_nom=nu_nom,
        jacobian=j,
        constraints=_stack(rows, rhs),
    ).q_dot
    # Joint-space limits respected within solver tolerance.
    assert np.all(q_dot <= upper + 1e-3)
    assert np.all(q_dot >= lower - 1e-3)
