# SPDX-License-Identifier: Apache-2.0
"""Unit + property tests for ``concerto.safety.budget_split`` (T3.4).

Covers ADR-004 §Decision (``alpha_i / (alpha_i + alpha_j)`` Phase-0
default per Wang-Ames-Egerstedt 2017 §IV), the Phase-1 stub for
relative-degree-aware splits (ADR-004 Open Question #3; risk R4), and
the strategy factory.
"""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from concerto.safety.budget_split import (
    SUPPORTED_STRATEGIES,
    BudgetSplitStrategy,
    ProportionalBudgetSplit,
    RelativeDegreeAwareBudgetSplit,
    make_budget_split,
)


def test_proportional_equal_capacities_is_half_half() -> None:
    bi, bj = ProportionalBudgetSplit().split(1.0, 1.0)
    assert bi == pytest.approx(0.5)
    assert bj == pytest.approx(0.5)


def test_proportional_skewed_capacities() -> None:
    bi, bj = ProportionalBudgetSplit().split(3.0, 1.0)
    assert bi == pytest.approx(0.75)
    assert bj == pytest.approx(0.25)


def test_proportional_is_symmetric_in_swap() -> None:
    bi1, bj1 = ProportionalBudgetSplit().split(2.0, 5.0)
    bj2, bi2 = ProportionalBudgetSplit().split(5.0, 2.0)
    assert bi1 == pytest.approx(bi2)
    assert bj1 == pytest.approx(bj2)


def test_proportional_rejects_zero_alpha() -> None:
    split = ProportionalBudgetSplit()
    with pytest.raises(ValueError, match="> 0"):
        split.split(0.0, 1.0)
    with pytest.raises(ValueError, match="> 0"):
        split.split(1.0, 0.0)


def test_proportional_rejects_negative_alpha() -> None:
    split = ProportionalBudgetSplit()
    with pytest.raises(ValueError, match="> 0"):
        split.split(-1.0, 1.0)


@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    alpha_i=st.floats(min_value=1e-3, max_value=1e3, allow_nan=False, allow_infinity=False),
    alpha_j=st.floats(min_value=1e-3, max_value=1e3, allow_nan=False, allow_infinity=False),
)
def test_proportional_split_sums_to_one(alpha_i: float, alpha_j: float) -> None:
    bi, bj = ProportionalBudgetSplit().split(alpha_i, alpha_j)
    assert bi + bj == pytest.approx(1.0)
    assert 0.0 <= bi <= 1.0
    assert 0.0 <= bj <= 1.0


def test_relative_degree_aware_is_phase1_stub() -> None:
    split = RelativeDegreeAwareBudgetSplit()
    with pytest.raises(NotImplementedError, match="Phase-1 stub"):
        split.split(1.0, 1.0)


def test_make_budget_split_returns_proportional_by_name() -> None:
    s = make_budget_split("proportional")
    assert isinstance(s, ProportionalBudgetSplit)
    assert isinstance(s, BudgetSplitStrategy)
    assert s.name == "proportional"


def test_make_budget_split_returns_relative_degree_aware_stub() -> None:
    s = make_budget_split("relative_degree_aware")
    assert isinstance(s, RelativeDegreeAwareBudgetSplit)
    assert isinstance(s, BudgetSplitStrategy)
    assert s.name == "relative_degree_aware"


def test_make_budget_split_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unknown budget-split strategy"):
        make_budget_split("greedy")


def test_supported_strategies_constant_matches_factory() -> None:
    assert set(SUPPORTED_STRATEGIES) == {"proportional", "relative_degree_aware"}
    for name in SUPPORTED_STRATEGIES:
        make_budget_split(name)


def test_protocol_runtime_checkable_for_both_strategies() -> None:
    assert isinstance(ProportionalBudgetSplit(), BudgetSplitStrategy)
    assert isinstance(RelativeDegreeAwareBudgetSplit(), BudgetSplitStrategy)
