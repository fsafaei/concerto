# SPDX-License-Identifier: Apache-2.0
"""Unit tests: ``OSCBF.solve`` returns ``OSCBFResult`` with slack telemetry (review P0-3).

External-review finding P0-3 (2026-05-16): ``OSCBF.solve()`` historically
returned ``(q_dot, solve_ms)`` and silently dropped the slack vector
``x[n:]``. For safety methodology, slack is the diagnostic signal that
distinguishes "constraints satisfied" from "constraints relaxed via
slack" — a controller that "succeeds" only via large slack is not safe
in the intended sense. ADR-014 Table 2 cannot aggregate slack stats
today; this PR makes them first-class telemetry.

These tests pin the new return shape:

- :class:`OSCBFResult` is a frozen dataclass with
  ``(q_dot, slack, solve_ms, active_rows, max_slack, slack_l2, solver_status)``.
- ``slack`` is shape ``(m,)`` where ``m`` is the constraint row count;
  values are non-negative by the QP's slack-non-negativity row.
- ``max_slack`` and ``slack_l2`` are aggregate slack statistics.
- ``active_rows`` enumerates rows whose slack exceeds a small numerical
  floor; empty under feasibility.
- Under deliberate constraint conflict (two contradictory joint-limit
  rows), slack absorbs the gap: ``max_slack > 0``, ``slack_l2 > 0``,
  ``active_rows`` is non-empty.

References:
- ADR-004 §Decision — OSCBF inner filter; slack relaxation pattern
  (Morton-Pavone 2025 §IV.D).
- ADR-014 §Decision (2026-05-16 amendment) — Table 2 ``max_slack`` and
  ``slack_l2`` columns; rationale on safety-vs-relaxation distinction.
- ``src/concerto/safety/oscbf.py`` — the helper under test.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from concerto.safety.oscbf import (
    OSCBF,
    OSCBFConstraints,
    OSCBFResult,
    joint_limit_constraint_row,
)


def _feasible_constraints(n_joints: int) -> OSCBFConstraints:
    """Single joint-limit row that the nominal `q_dot_nom=0` already satisfies."""
    row, rhs = joint_limit_constraint_row(n_joints=n_joints, joint_index=0, upper=1.0)
    return OSCBFConstraints(a=row.reshape(1, n_joints), b=np.array([rhs], dtype=np.float64))


def _conflicting_constraints(n_joints: int) -> OSCBFConstraints:
    """Two rows that are jointly infeasible without slack.

    Row 1: ``q_dot[0] <= 0.5``.
    Row 2: ``-q_dot[0] <= -1.0`` (i.e., ``q_dot[0] >= 1.0``).
    Slack absorbs the 0.5-unit gap on at least one row.
    """
    row_upper, rhs_upper = joint_limit_constraint_row(n_joints=n_joints, joint_index=0, upper=0.5)
    row_lower, rhs_lower = joint_limit_constraint_row(n_joints=n_joints, joint_index=0, lower=1.0)
    a = np.vstack([row_upper, row_lower])
    b = np.array([rhs_upper, rhs_lower], dtype=np.float64)
    return OSCBFConstraints(a=a, b=b)


def test_oscbf_solve_returns_oscbf_result_dataclass() -> None:
    """Public surface: `OSCBF.solve` returns the new OSCBFResult shape."""
    n = 3
    oscbf = OSCBF(n_joints=n)
    result = oscbf.solve(
        q_dot_nom=np.zeros(n, dtype=np.float64),
        nu_nom=np.zeros(6, dtype=np.float64),
        jacobian=np.zeros((6, n), dtype=np.float64),
        constraints=_feasible_constraints(n),
    )
    assert isinstance(result, OSCBFResult)


def test_oscbf_result_field_shapes_and_types() -> None:
    """Field-shape contract for the new OSCBFResult dataclass."""
    n = 4
    oscbf = OSCBF(n_joints=n)
    constraints = _feasible_constraints(n)
    m = constraints.a.shape[0]
    result = oscbf.solve(
        q_dot_nom=np.zeros(n, dtype=np.float64),
        nu_nom=np.zeros(6, dtype=np.float64),
        jacobian=np.zeros((6, n), dtype=np.float64),
        constraints=constraints,
    )
    assert result.q_dot.shape == (n,)
    assert result.q_dot.dtype == np.float64
    assert result.slack.shape == (m,)
    assert result.slack.dtype == np.float64
    assert isinstance(result.solve_ms, float)
    assert result.solve_ms >= 0.0
    assert isinstance(result.active_rows, tuple)
    assert all(isinstance(i, int) for i in result.active_rows)
    assert isinstance(result.max_slack, float)
    assert isinstance(result.slack_l2, float)
    assert isinstance(result.solver_status, str)


def test_oscbf_result_is_frozen() -> None:
    """OSCBFResult is a frozen dataclass — mutation raises."""
    n = 2
    oscbf = OSCBF(n_joints=n)
    result = oscbf.solve(
        q_dot_nom=np.zeros(n, dtype=np.float64),
        nu_nom=np.zeros(6, dtype=np.float64),
        jacobian=np.zeros((6, n), dtype=np.float64),
        constraints=_feasible_constraints(n),
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.max_slack = 99.0  # type: ignore[misc]


def test_oscbf_slack_zero_under_feasibility() -> None:
    """Feasible constraints ⇒ slack at solver round-off, ``active_rows`` empty, status optimal.

    The test tolerance ``1e-4`` matches the
    ``oscbf._SLACK_ACTIVE_FLOOR`` floor: any slack below the floor is
    treated as interior-point round-off, not a genuine constraint
    relaxation. Clarabel's default tolerances place feasible-interior
    slack at roughly ``5e-6`` on this QP shape; the floor is two
    orders of magnitude looser to keep the contract robust to small
    solver-version drift.
    """
    n = 3
    oscbf = OSCBF(n_joints=n)
    result = oscbf.solve(
        q_dot_nom=np.zeros(n, dtype=np.float64),
        nu_nom=np.zeros(6, dtype=np.float64),
        jacobian=np.zeros((6, n), dtype=np.float64),
        constraints=_feasible_constraints(n),
    )
    np.testing.assert_allclose(result.slack, np.zeros_like(result.slack), atol=1e-4)
    assert result.max_slack == pytest.approx(0.0, abs=1e-4)
    assert result.slack_l2 == pytest.approx(0.0, abs=1e-4)
    assert result.active_rows == ()
    assert result.solver_status == "optimal"


def test_oscbf_slack_positive_under_constraint_conflict() -> None:
    """Conflicting constraints ⇒ slack > 0 on at least one row.

    Two row directions (upper bound + lower bound on the same joint with
    no feasible overlap) force the QP to relax at least one. The
    OSCBFResult must carry the relaxation signal so ADR-014 Table 2 can
    aggregate it.
    """
    n = 3
    oscbf = OSCBF(n_joints=n, slack_penalty=10.0)  # finite rho so slack is materially used
    constraints = _conflicting_constraints(n)
    m = constraints.a.shape[0]
    result = oscbf.solve(
        q_dot_nom=np.zeros(n, dtype=np.float64),
        nu_nom=np.zeros(6, dtype=np.float64),
        jacobian=np.zeros((6, n), dtype=np.float64),
        constraints=constraints,
    )
    assert result.slack.shape == (m,)
    assert result.max_slack > 0.0
    assert result.slack_l2 > 0.0
    assert len(result.active_rows) >= 1
    assert all(0 <= i < m for i in result.active_rows)


def test_oscbf_aggregates_consistent_with_slack_vector() -> None:
    """``max_slack`` and ``slack_l2`` agree with the raw slack array."""
    n = 3
    oscbf = OSCBF(n_joints=n, slack_penalty=10.0)
    result = oscbf.solve(
        q_dot_nom=np.zeros(n, dtype=np.float64),
        nu_nom=np.zeros(6, dtype=np.float64),
        jacobian=np.zeros((6, n), dtype=np.float64),
        constraints=_conflicting_constraints(n),
    )
    assert result.max_slack == pytest.approx(float(np.max(result.slack)))
    assert result.slack_l2 == pytest.approx(float(np.linalg.norm(result.slack)))


def test_oscbf_warns_when_raw_slack_below_round_off_floor() -> None:
    """A real solver bug returning materially-negative slack triggers a UserWarning.

    The QP's ``-s <= 0`` row makes slack non-negative mathematically; the
    helper clips tiny round-off excursions silently but emits a
    :class:`UserWarning` when the excursion is below
    ``_NEGATIVE_SLACK_WARN_FLOOR`` (review P0-3 reviewer follow-up).
    The test stubs the solver to return a negative slack iterate.
    """
    from concerto.safety.oscbf import OSCBF as _OSCBF  # local rebind for stubbing

    class _NegativeSlackSolver:
        """Stub: returns an x with materially-negative slack to simulate a solver bug."""

        def solve(
            self,
            P: np.ndarray,
            q: np.ndarray,
            A: np.ndarray,
            b: np.ndarray,
            *,
            warm_start: bool = True,
        ) -> tuple[np.ndarray, float]:
            del P, q, A, b, warm_start
            n = 3
            m = 1
            x = np.zeros(n + m, dtype=np.float64)
            x[n] = -1e-3  # well below _NEGATIVE_SLACK_WARN_FLOOR = -1e-6
            return x, 0.0

    n = 3
    oscbf = _OSCBF(n_joints=n, solver=_NegativeSlackSolver())
    with pytest.warns(UserWarning, match="raw slack went below"):
        result = oscbf.solve(
            q_dot_nom=np.zeros(n, dtype=np.float64),
            nu_nom=np.zeros(6, dtype=np.float64),
            jacobian=np.zeros((6, n), dtype=np.float64),
            constraints=_feasible_constraints(n),
        )
    # The clip still fires — the public-facing slack is non-negative.
    assert result.slack.min() >= 0.0


def test_oscbf_active_rows_match_nonzero_slack_indices() -> None:
    """``active_rows`` enumerates exactly the indices with slack above the floor."""
    n = 3
    oscbf = OSCBF(n_joints=n, slack_penalty=10.0)
    result = oscbf.solve(
        q_dot_nom=np.zeros(n, dtype=np.float64),
        nu_nom=np.zeros(6, dtype=np.float64),
        jacobian=np.zeros((6, n), dtype=np.float64),
        constraints=_conflicting_constraints(n),
    )
    # The "active" floor lives inside oscbf.py; test via the public
    # contract that active_rows ⊆ {i : slack[i] > 0} and that no
    # nonzero-slack row is silently dropped beyond a tiny numerical
    # tolerance (1e-9 = the helper's own floor).
    for i in range(result.slack.shape[0]):
        if result.slack[i] > 1e-6:  # well above any reasonable floor
            assert i in result.active_rows
        if i in result.active_rows:
            assert result.slack[i] > 0.0
