# SPDX-License-Identifier: Apache-2.0
"""Reviewer P1-7: optimality-gap is expected shortfall, not empirical CDF.

The optimality-gap docstring and the implementation in
:func:`chamber.evaluation.bootstrap._optimality_gap` define the metric
as ``mean(max(threshold - x, 0))`` — the Agarwal et al. 2021 / rliable
"expected shortfall below threshold" definition (ADR-008 §Decision;
ADR-014 §Decision; docs/reference/evaluation.md §3.2). An earlier
version of the docs described it as "the proportion of the score
distribution below a target threshold" (i.e. the empirical CDF at the
threshold), which is a different statistic. These tests pin the
implementation to the expected-shortfall formula on hand-rolled
distributions so future drift is caught at the unit-test layer.

The companion test exercises the missing-``rliable`` branch of
:func:`chamber.evaluation.bootstrap.aggregate_metrics`: when the
optional extra is absent, the call returns ``None`` for the
``performance_profile`` slot and emits a ``RuntimeWarning`` pointing
to the install command, rather than failing the call.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from chamber.evaluation.bootstrap import _optimality_gap, aggregate_metrics


def test_optimality_gap_matches_expected_shortfall_formula() -> None:
    values = np.array([0.2, 0.5, 0.9, 1.0])
    threshold = 1.0
    expected = float(np.mean(np.maximum(threshold - values, 0.0)))
    assert _optimality_gap(values, threshold=threshold) == pytest.approx(expected)
    # Hand-checked: (0.8 + 0.5 + 0.1 + 0.0) / 4 = 0.35.
    assert _optimality_gap(values, threshold=threshold) == pytest.approx(0.35)


def test_optimality_gap_is_zero_when_all_values_meet_threshold() -> None:
    values = np.array([1.0, 1.0, 1.2, 2.0])
    assert _optimality_gap(values, threshold=1.0) == pytest.approx(0.0)


def test_optimality_gap_threshold_argument_shifts_reference() -> None:
    values = np.array([0.0, 0.25, 0.5, 0.75])
    # Lowering the threshold strictly reduces the expected shortfall.
    high = _optimality_gap(values, threshold=1.0)
    low = _optimality_gap(values, threshold=0.5)
    assert low < high
    # Hand-checked at threshold=0.5: (0.5 + 0.25 + 0.0 + 0.0) / 4 = 0.1875.
    assert low == pytest.approx(0.1875)


def test_optimality_gap_is_not_the_empirical_cdf_at_threshold() -> None:
    """Guards against regressing to the pre-correction definition.

    The empirical CDF at threshold=1.0 for this sample is 3/4 = 0.75
    (three of four values are strictly below 1.0). The expected
    shortfall is 0.35. They are distinct statistics; this test pins
    the implementation to the shortfall, not the CDF.
    """
    values = np.array([0.2, 0.5, 0.9, 1.0])
    empirical_cdf = float(np.mean(values < 1.0))
    assert _optimality_gap(values, threshold=1.0) != pytest.approx(empirical_cdf)


def test_aggregate_metrics_performance_profile_when_rliable_absent() -> None:
    """Reviewer P1-4: native fallback path is non-fatal and signposted.

    When the optional ``rliable`` extra is not installed, the
    ``performance_profile`` slot is ``None`` and a ``RuntimeWarning``
    quotes the install command. IQM and optimality gap are computed
    natively in either case.
    """
    values: dict[int, list[float]] = {1: [1.0, 1.0, 0.0, 1.0], 2: [0.5, 0.5, 1.0, 0.0]}
    if importlib.util.find_spec("rliable") is None:
        with pytest.warns(RuntimeWarning, match="rliable"):
            metrics = aggregate_metrics(values)
        assert metrics["performance_profile"] is None
    else:
        metrics = aggregate_metrics(values)
        assert metrics["performance_profile"] is not None
    assert metrics["iqm"] is not None
    assert metrics["optimality_gap"] is not None
