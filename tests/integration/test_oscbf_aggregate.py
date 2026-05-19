# SPDX-License-Identifier: Apache-2.0
"""Integration test: per-step OSCBF slack ⇒ ADR-014 Table 2 ConditionRow (issue #142).

Pins the locked aggregation contract from ADR-014 §Revision history
2026-05-16:

- ``ConditionRow.max_slack`` = max over steps ``k`` of
  ``OSCBFResult.max_slack``.
- ``ConditionRow.slack_l2`` = mean over steps ``k`` of
  ``OSCBFResult.slack_l2``.

The conflicted-constraint pattern (upper bound + non-overlapping lower
bound on the same joint) forces the OSCBF QP to relax at least one row,
which is the only known way to drive ``slack_l2 > 0`` without injecting
synthetic OSCBFResult values. The test combines feasible and conflicted
steps inside a single condition so both columns carry non-zero values
when the aggregator runs end-to-end through
:func:`chamber.evaluation.oscbf_aggregate.make_condition_row` and then
through ``ThreeTableReport`` JSON round-tripping.
"""

from __future__ import annotations

import numpy as np
import pytest

from chamber.evaluation.oscbf_aggregate import (
    SlackAggregate,
    aggregate_oscbf_slack,
    make_condition_row,
)
from concerto.safety.oscbf import (
    OSCBF,
    OSCBFConstraints,
    joint_limit_constraint_row,
)
from concerto.safety.reporting import (
    AssumptionRow,
    ConditionRow,
    GapRow,
    ThreeTableReport,
    emit_three_tables,
    parse_three_tables,
)


def _feasible_constraints(n_joints: int) -> OSCBFConstraints:
    row, rhs = joint_limit_constraint_row(n_joints=n_joints, joint_index=0, upper=1.0)
    return OSCBFConstraints(a=row.reshape(1, n_joints), b=np.array([rhs], dtype=np.float64))


def _conflicting_constraints(n_joints: int) -> OSCBFConstraints:
    row_upper, rhs_upper = joint_limit_constraint_row(n_joints=n_joints, joint_index=0, upper=0.5)
    row_lower, rhs_lower = joint_limit_constraint_row(n_joints=n_joints, joint_index=0, lower=1.0)
    a = np.vstack([row_upper, row_lower])
    b = np.array([rhs_upper, rhs_lower], dtype=np.float64)
    return OSCBFConstraints(a=a, b=b)


def _solve(oscbf: OSCBF, n: int, constraints: OSCBFConstraints):
    return oscbf.solve(
        q_dot_nom=np.zeros(n, dtype=np.float64),
        nu_nom=np.zeros(6, dtype=np.float64),
        jacobian=np.zeros((6, n), dtype=np.float64),
        constraints=constraints,
    )


def test_aggregate_empty_returns_zero() -> None:
    """An empty iterable returns the schema-v2 ConditionRow defaults."""
    aggregate = aggregate_oscbf_slack(())
    assert aggregate == SlackAggregate(max_slack=0.0, slack_l2=0.0)


def test_aggregate_max_is_max_and_mean_is_mean() -> None:
    """The aggregator implements the locked contract: max + mean."""
    n = 3
    oscbf = OSCBF(n_joints=n, slack_penalty=10.0)
    feasible = _solve(oscbf, n, _feasible_constraints(n))
    conflicted = _solve(oscbf, n, _conflicting_constraints(n))

    aggregate = aggregate_oscbf_slack([feasible, conflicted])

    expected_max = max(feasible.max_slack, conflicted.max_slack)
    expected_mean = (feasible.slack_l2 + conflicted.slack_l2) / 2.0
    assert aggregate.max_slack == pytest.approx(expected_max)
    assert aggregate.slack_l2 == pytest.approx(expected_mean)


def test_condition_row_slack_columns_non_zero_under_conflict() -> None:
    """Issue #142 §Work to do #3: ConditionRow slack columns are populated.

    A condition that exercises a conflicted constraint pattern at least
    once produces ``max_slack > 0`` and ``slack_l2 > 0`` after running
    through the aggregator + ConditionRow constructor.
    """
    n = 3
    oscbf = OSCBF(n_joints=n, slack_penalty=10.0)
    results = [
        _solve(oscbf, n, _feasible_constraints(n)),
        _solve(oscbf, n, _conflicting_constraints(n)),
        _solve(oscbf, n, _feasible_constraints(n)),
    ]

    row = make_condition_row(
        predictor="pred",
        conformal_mode="Learn",
        vendor_compliance=None,
        n_episodes=1,
        violations=0,
        fallback_fires=0,
        oscbf_results=results,
    )

    assert row.max_slack > 0.0
    assert row.slack_l2 > 0.0


def test_condition_row_slack_columns_zero_when_no_oscbf_results() -> None:
    """EGO_ONLY default (no OSCBF calls) → schema-v2 0.0 defaults."""
    row = make_condition_row(
        predictor="gt",
        conformal_mode="noLearn",
        vendor_compliance=None,
        n_episodes=1,
        violations=0,
        fallback_fires=0,
    )
    assert row.max_slack == 0.0
    assert row.slack_l2 == 0.0


def test_aggregator_feeds_three_tables_round_trip(tmp_path) -> None:
    """End-to-end seam: aggregated slack survives ``three_tables.json`` round-trip."""
    n = 3
    oscbf = OSCBF(n_joints=n, slack_penalty=10.0)
    results = [_solve(oscbf, n, _conflicting_constraints(n)) for _ in range(4)]

    row = make_condition_row(
        predictor="pred",
        conformal_mode="Learn",
        vendor_compliance=None,
        n_episodes=4,
        violations=0,
        fallback_fires=0,
        oscbf_results=results,
    )

    report = ThreeTableReport(
        table_1=(AssumptionRow(assumption="A1", description="-", violations=0, n_steps=0),),
        table_2=(row,),
        table_3=(
            GapRow(condition="pred/Learn", lambda_mean=0.0, lambda_var=0.0, oracle_lambda_mean=0.0),
        ),
    )

    json_path, _ = emit_three_tables(out_dir=tmp_path, report=report)
    parsed = parse_three_tables(json_path)
    parsed_row = parsed.table_2[0]
    assert parsed_row.max_slack == pytest.approx(row.max_slack)
    assert parsed_row.slack_l2 == pytest.approx(row.slack_l2)
    assert parsed_row.max_slack > 0.0
    assert parsed_row.slack_l2 > 0.0


def test_aggregate_single_step_mean_equals_value() -> None:
    """Single-step condition: mean degenerates to the lone value (regression pin)."""
    n = 3
    oscbf = OSCBF(n_joints=n, slack_penalty=10.0)
    only = _solve(oscbf, n, _conflicting_constraints(n))

    aggregate = aggregate_oscbf_slack([only])
    assert aggregate.max_slack == pytest.approx(only.max_slack)
    assert aggregate.slack_l2 == pytest.approx(only.slack_l2)


def test_condition_row_round_trips_through_dict() -> None:
    """Schema-v2 columns survive the ConditionRow.to_jsonable round-trip."""
    n = 3
    oscbf = OSCBF(n_joints=n, slack_penalty=10.0)
    results = [_solve(oscbf, n, _conflicting_constraints(n))]
    row = make_condition_row(
        predictor="pred",
        conformal_mode="Learn",
        vendor_compliance=None,
        n_episodes=1,
        violations=0,
        fallback_fires=0,
        oscbf_results=results,
    )
    payload = row.to_jsonable()
    parsed = ConditionRow.from_jsonable(payload)
    assert parsed == row
