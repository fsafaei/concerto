# SPDX-License-Identifier: Apache-2.0
"""Per-condition OSCBF slack aggregation for ADR-014 Table 2 (issue #142).

ADR-014 §Revision history 2026-05-16 amends Table 2 with the
``ConditionRow.max_slack`` and ``ConditionRow.slack_l2`` columns;
PR #141 (closes external-review P0-3) ships the matching solver-side
telemetry on :class:`concerto.safety.oscbf.OSCBFResult`. The seam
between the two — aggregating per-step :class:`OSCBFResult` values into
the per-condition row at ``three_tables.json`` emission time — is the
work this module handles.

The locked aggregation contract (ADR-014 §Revision history 2026-05-16):

- ``ConditionRow.max_slack`` is the **maximum** over steps ``k`` in the
  condition of ``OSCBFResult.max_slack`` — the worst single-step
  relaxation observed across the condition.
- ``ConditionRow.slack_l2`` is the **mean** over steps ``k`` of
  ``OSCBFResult.slack_l2`` — the mean L2-norm of the per-step slack
  vector across the condition.

Empty / zero-step conditions return ``0.0`` for both quantities (the
``ConditionRow`` defaults; see
:class:`concerto.safety.reporting.ConditionRow`). This is consistent
with the Stage-1a default where the OSCBF inner filter is not
exercised (the EGO_ONLY outer filter path goes through
:class:`concerto.safety.cbf_qp.ExpCBFQP` and never calls
:meth:`concerto.safety.oscbf.OSCBF.solve` per the issue body).

Wiring contract. The Stage-1 AS spike adapter
(:mod:`chamber.benchmarks.stage1_as`) is the natural integration
point — that is where OSCBF goes load-bearing for per-arm safety on
the 7-DOF arm under Stage 1b (ADR-007 §Implementation staging). Until
that real-env rollout lands, this module is consumed by the
``three_tables.json`` emission path so the Tier-1 fake test surface
can pin the aggregation contract end-to-end.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from concerto.safety.reporting import ConditionRow

if TYPE_CHECKING:
    from collections.abc import Iterable

    from concerto.safety.oscbf import OSCBFResult


@dataclass(frozen=True)
class SlackAggregate:
    """Per-condition OSCBF slack aggregate (ADR-014 §Revision history 2026-05-16).

    Carries the two scalar columns ADR-014 Table 2 v2 adds to
    :class:`concerto.safety.reporting.ConditionRow`. The values are
    aggregated by :func:`aggregate_oscbf_slack` over the per-step
    :class:`concerto.safety.oscbf.OSCBFResult` stream within a single
    ADR-014 Table 2 condition.

    Attributes:
        max_slack: Maximum over steps ``k`` in the condition of
            ``OSCBFResult.max_slack`` — the worst single-step OSCBF
            relaxation observed across the condition. Non-negative by
            construction (the source signal is non-negative).
        slack_l2: Mean over steps ``k`` of ``OSCBFResult.slack_l2`` —
            the mean L2-norm of the per-step slack vector across the
            condition. Non-negative by construction.
    """

    max_slack: float
    slack_l2: float


def aggregate_oscbf_slack(results: Iterable[OSCBFResult]) -> SlackAggregate:
    """Aggregate per-step :class:`OSCBFResult` values into the ADR-014 contract.

    Implements the locked aggregation contract from ADR-014 §Revision
    history 2026-05-16:

    - ``max_slack`` is the max over steps of ``OSCBFResult.max_slack``.
    - ``slack_l2`` is the mean over steps of ``OSCBFResult.slack_l2``.

    A consumed empty iterable produces ``SlackAggregate(0.0, 0.0)`` so
    the result composes with the
    :class:`concerto.safety.reporting.ConditionRow` defaults (the
    Stage-1a EGO_ONLY path that does not exercise the OSCBF inner
    filter; issue #142).

    Args:
        results: Iterable of per-step
            :class:`concerto.safety.oscbf.OSCBFResult` values from the
            steps within a single ADR-014 Table 2 condition. Order is
            irrelevant; the aggregator consumes the iterable exactly
            once.

    Returns:
        The :class:`SlackAggregate` carrying the two ADR-014 v2
        ConditionRow columns.
    """
    max_slack = 0.0
    slack_l2_sum = 0.0
    n_steps = 0
    for result in results:
        max_slack = max(max_slack, result.max_slack)
        slack_l2_sum += result.slack_l2
        n_steps += 1
    slack_l2_mean = slack_l2_sum / n_steps if n_steps > 0 else 0.0
    return SlackAggregate(max_slack=max_slack, slack_l2=slack_l2_mean)


def make_condition_row(
    *,
    predictor: str,
    conformal_mode: str,
    vendor_compliance: str | None,
    n_episodes: int,
    violations: int,
    fallback_fires: int,
    oscbf_results: Iterable[OSCBFResult] = (),
) -> ConditionRow:
    """Construct a :class:`ConditionRow` with slack aggregates wired in (issue #142).

    The natural call-site at ``three_tables.json`` emission time. Routes
    the per-step :class:`concerto.safety.oscbf.OSCBFResult` stream through
    :func:`aggregate_oscbf_slack` so the v2 slack columns are populated
    per the locked ADR-014 §Revision history 2026-05-16 contract. Callers
    on the EGO_ONLY outer-filter path (the Stage-1a default) pass the
    default empty tuple and get the schema-v2 ``0.0`` defaults — exactly
    the behaviour documented on
    :class:`concerto.safety.reporting.ConditionRow`.

    Args:
        predictor: ``"gt"`` or ``"pred"`` (Huriot & Sibai 2025 Table I
            labels).
        conformal_mode: ``"noLearn"`` or ``"Learn"``.
        vendor_compliance: ADR-007 rev 3 placeholder; ``None`` in
            Phase-0.
        n_episodes: Number of episodes in this condition.
        violations: Count of CBF-constraint violations.
        fallback_fires: Count of braking-fallback fires.
        oscbf_results: Per-step :class:`OSCBFResult` stream for the
            condition. Defaults to an empty tuple — the Stage-1a
            EGO_ONLY path that does not exercise the OSCBF inner
            filter (issue #142).

    Returns:
        A :class:`ConditionRow` carrying both the Phase-0 columns and
        the ADR-014 v2 slack aggregates.
    """
    aggregate = aggregate_oscbf_slack(oscbf_results)
    return ConditionRow(
        predictor=predictor,
        conformal_mode=conformal_mode,
        vendor_compliance=vendor_compliance,
        n_episodes=n_episodes,
        violations=violations,
        fallback_fires=fallback_fires,
        max_slack=aggregate.max_slack,
        slack_l2=aggregate.slack_l2,
    )


__all__ = [
    "SlackAggregate",
    "aggregate_oscbf_slack",
    "make_condition_row",
]
