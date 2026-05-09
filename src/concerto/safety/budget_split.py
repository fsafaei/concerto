# SPDX-License-Identifier: Apache-2.0
"""Per-pair budget split strategies (ADR-004 §Decision; plan/03 §3.4).

Decentralisation of the CBF-QP follows Wang-Ames-Egerstedt 2017 §IV: each
joint pairwise safety constraint carries a budget that's split between
the two agents proportional to their actuator capacity (typically the
maximum acceleration ``alpha`` from :class:`concerto.safety.api.Bounds`).
The default ``alpha_i / (alpha_i + alpha_j)`` split handles the
homogeneous and mixed-magnitude cases; mixed-relative-degree pairs
(velocity-controlled mobile base vs torque-controlled arm) need the
``relative_degree_aware`` strategy that lands in Phase-1 (ADR-004 Open
Question #3; risk register R4 in IMPLEMENTATION_PLAN §8).

The strategy interface keeps the QP-row generator (``cbf_qp.py``)
agnostic of which split is in effect — selection is a Hydra config
setting (``safety.budget_split``). The Phase-0 stub for the
relative-degree-aware variant fails fast (``NotImplementedError``)
rather than silently falling back to the proportional split, so configs
referencing the not-yet-implemented strategy break loudly at startup.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

#: Strategy names accepted by :func:`make_budget_split`
#: (ADR-004 §"Per-pair budget split strategy").
SUPPORTED_STRATEGIES: tuple[str, ...] = ("proportional", "relative_degree_aware")


@runtime_checkable
class BudgetSplitStrategy(Protocol):
    """Strategy for splitting a pairwise CBF budget between two agents (ADR-004 §Decision).

    ADR-004 §Decision pins the ``alpha_i / (alpha_i + alpha_j)``
    Wang-Ames-Egerstedt 2017 default for Phase-0; the strategy interface
    lets a relative-degree-aware variant slot in for Phase-1
    mixed-relative-degree pairs (ADR-004 Open Question #3; risk R4).

    Implementations consume strictly positive per-agent capacities and
    return non-negative budget fractions summing to ``1.0`` within
    numerical precision. The CBF-QP row generator (``cbf_qp.py``) then
    multiplies the pairwise constraint bound by each fraction to assign
    per-agent half-rows.
    """

    name: str

    def split(self, alpha_i: float, alpha_j: float) -> tuple[float, float]:
        """Return the budget fractions ``(beta_i, beta_j)`` (ADR-004 §Decision).

        Args:
            alpha_i: Strictly positive actuator capacity for agent ``i``
                (typically maximum acceleration, ADR-006 §Decision
                ``Bounds.action_norm``).
            alpha_j: Strictly positive actuator capacity for agent ``j``.

        Returns:
            ``(beta_i, beta_j)``, each in ``[0, 1]``, summing to ``1.0``.

        Raises:
            ValueError: If either capacity is non-positive.
        """
        ...


class ProportionalBudgetSplit:
    """Default per-pair split: ``alpha_i / (alpha_i + alpha_j)`` (ADR-004 §Decision).

    Implements Wang-Ames-Egerstedt 2017 §IV: the pairwise CBF budget is
    apportioned in proportion to actuator capacity. Symmetric in
    ``(alpha_i, alpha_j)``, so equal capacities yield a 50-50 split.
    """

    name: str = "proportional"

    def split(self, alpha_i: float, alpha_j: float) -> tuple[float, float]:
        """Compute ``(alpha_i / total, alpha_j / total)`` (ADR-004 §Decision).

        Wang-Ames-Egerstedt 2017 §IV proportional budget rule.

        Args:
            alpha_i: Strictly positive capacity for agent ``i``.
            alpha_j: Strictly positive capacity for agent ``j``.

        Returns:
            The proportional split, summing to ``1.0`` exactly under
            IEEE-754 in the equal-capacity case and within
            ``2 * eps_64`` otherwise.

        Raises:
            ValueError: If either capacity is non-positive.
        """
        if alpha_i <= 0.0 or alpha_j <= 0.0:
            msg = (
                f"Both capacities must be > 0; got alpha_i={alpha_i}, "
                f"alpha_j={alpha_j} (Bounds.action_norm is strictly positive "
                "per ADR-006 §Decision)."
            )
            raise ValueError(msg)
        total = alpha_i + alpha_j
        return alpha_i / total, alpha_j / total


class RelativeDegreeAwareBudgetSplit:
    """Phase-1 stub: relative-degree-aware budget split (ADR-004 Open Question #3).

    Reserved for the Phase-1 extension that handles mixed-relative-degree
    pairs (velocity-controlled mobile base vs torque-controlled arm) per
    risk R4 in IMPLEMENTATION_PLAN §8. The Phase-0 stub raises
    :class:`NotImplementedError`; the strategy name is reserved here so
    config files referencing it fail fast rather than silently falling
    back to the proportional split.
    """

    name: str = "relative_degree_aware"

    def split(self, alpha_i: float, alpha_j: float) -> tuple[float, float]:
        """Raise :class:`NotImplementedError` (ADR-004 Open Question #3; risk R4).

        Phase-1 will weight the split by each agent's CBF relative degree
        alongside its actuator capacity. Until then this stub fails fast
        so configs cannot accidentally bind to the not-yet-implemented
        strategy.
        """
        del alpha_i, alpha_j
        msg = (
            "RelativeDegreeAwareBudgetSplit is a Phase-1 stub "
            "(ADR-004 Open Question #3 / risk R4 must resolve first). "
            "Use the 'proportional' strategy in Phase-0."
        )
        raise NotImplementedError(msg)


def make_budget_split(name: str) -> BudgetSplitStrategy:
    """Factory for budget-split strategies (ADR-004 §Decision; plan/03 §3.4).

    Args:
        name: One of :data:`SUPPORTED_STRATEGIES` —
            ``"proportional"`` (Phase-0 default) or
            ``"relative_degree_aware"`` (Phase-1 stub).

    Returns:
        A fresh strategy instance.

    Raises:
        ValueError: If ``name`` is not in :data:`SUPPORTED_STRATEGIES`.
    """
    if name == "proportional":
        return ProportionalBudgetSplit()
    if name == "relative_degree_aware":
        return RelativeDegreeAwareBudgetSplit()
    msg = f"Unknown budget-split strategy {name!r}; expected one of {SUPPORTED_STRATEGIES}"
    raise ValueError(msg)


__all__ = [
    "SUPPORTED_STRATEGIES",
    "BudgetSplitStrategy",
    "ProportionalBudgetSplit",
    "RelativeDegreeAwareBudgetSplit",
    "make_budget_split",
]
