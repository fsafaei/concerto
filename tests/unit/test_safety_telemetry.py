# SPDX-License-Identifier: Apache-2.0
"""Unit tests for :class:`SafetyAggregator` (P1.04.5; ADR-007 §Stage 1b).

Pure-Python — no env, no SAPIEN. Pins the per-cell aggregator's
running statistics, window flush + reset, and finalise-summary
schema (the audit-gate predicate A + B inputs).
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from concerto.training.safety_telemetry import SafetyAggregator


def _f(summary: dict[str, object], key: str) -> float:
    """Coerce a finalise/flush field to float for numeric assertions (pyright-clean)."""
    return float(cast("float", summary[key]))


def _i(summary: dict[str, object], key: str) -> int:
    """Coerce a finalise/flush field to int for counter assertions (pyright-clean)."""
    return int(cast("int", summary[key]))


class TestSafetyAggregatorRunningStats:
    """Per-step observe + window flush + finalise correctness (P1.04.5)."""

    def test_empty_aggregator_finalises_to_safe_defaults(self) -> None:
        """Zero observations: lambda_* = 0.0, saturated False."""
        agg = SafetyAggregator(n_pairs=1, cartesian_accel_capacity=10.0)
        summary = agg.finalise(safety_enabled=True)
        assert summary["event"] == "safety_telemetry_final"
        assert summary["n_filter_calls"] == 0
        assert summary["lambda_mean"] == 0.0
        assert summary["lambda_var"] == 0.0
        assert summary["lambda_steady_state"] == 0.0
        assert summary["saturated"] is False

    def test_constant_lambda_run_steady_state_matches_mean(self) -> None:
        """All λ = 0.5 → lambda_steady_state = 0.5; var = 0."""
        agg = SafetyAggregator(n_pairs=1, cartesian_accel_capacity=10.0)
        for _ in range(100):
            agg.observe({("a", "b"): 0.5})
        summary = agg.finalise(safety_enabled=True)
        assert summary["n_filter_calls"] == 100
        np.testing.assert_allclose(_f(summary, "lambda_mean"), 0.5)
        np.testing.assert_allclose(_f(summary, "lambda_var"), 0.0, atol=1e-12)
        np.testing.assert_allclose(_f(summary, "lambda_steady_state"), 0.5)
        assert summary["saturated"] is False

    def test_saturation_detected_when_steady_state_crosses_threshold(self) -> None:
        """λ at 9.5, threshold 0.9 x 10 = 9.0 → saturated True."""
        agg = SafetyAggregator(n_pairs=1, cartesian_accel_capacity=10.0, saturation_threshold=0.9)
        for _ in range(50):
            agg.observe({("a", "b"): 9.5})
        summary = agg.finalise(safety_enabled=True)
        assert summary["saturated"] is True
        np.testing.assert_allclose(_f(summary, "lambda_steady_state"), 9.5)

    def test_steady_state_uses_final_tenth_of_history(self) -> None:
        """Linear ramp 0→1 over 100 steps: steady_state = mean of last 10 ≈ 0.945."""
        agg = SafetyAggregator(n_pairs=1, cartesian_accel_capacity=10.0)
        for i in range(100):
            agg.observe({("a", "b"): i / 99.0})
        summary = agg.finalise(safety_enabled=True)
        # Final 10% = last 10 samples (steps 90-99), mean = (90+91+...+99)/99 / 10
        expected_steady_state = float(np.mean([i / 99.0 for i in range(90, 100)]))
        np.testing.assert_allclose(
            _f(summary, "lambda_steady_state"), expected_steady_state, rtol=1e-9
        )
        # Full-run mean is the midpoint of the ramp ≈ 0.5
        np.testing.assert_allclose(_f(summary, "lambda_mean"), 0.5, rtol=0.05)

    def test_lambda_var_captures_variance_for_predicate_b(self) -> None:
        """Predicate B input: lambda_var > 1e-12 distinguishes adaptive from stuck."""
        agg = SafetyAggregator(n_pairs=1, cartesian_accel_capacity=10.0)
        rng = np.random.default_rng(0)
        for _ in range(100):
            agg.observe({("a", "b"): float(rng.uniform(0.4, 0.6))})
        summary = agg.finalise(safety_enabled=True)
        # Uniform(0.4, 0.6) has variance (0.2)^2/12 ≈ 0.0033
        assert _f(summary, "lambda_var") > 1e-3
        # Predicate B's "stuck" threshold is 1e-12; this run clears it.
        assert _f(summary, "lambda_var") > 1e-12


class TestSafetyAggregatorWindowFlush:
    """Per-rollout flush + reset semantics (P1.04.5)."""

    def test_flush_returns_window_aggregate_and_resets(self) -> None:
        agg = SafetyAggregator(n_pairs=1, cartesian_accel_capacity=10.0)
        for i in range(10):
            agg.observe({("a", "b"): float(i)})
        window = agg.flush_window_stats()
        assert window["n_obs"] == 10
        np.testing.assert_allclose(_f(window, "lambda_mean"), 4.5)
        # Window state cleared after flush.
        empty = agg.flush_window_stats()
        assert empty["n_obs"] == 0

    def test_window_flush_does_not_clear_total_stats(self) -> None:
        agg = SafetyAggregator(n_pairs=1, cartesian_accel_capacity=10.0)
        for _ in range(10):
            agg.observe({("a", "b"): 1.0})
        agg.flush_window_stats()  # discard window
        for _ in range(10):
            agg.observe({("a", "b"): 2.0})
        summary = agg.finalise(safety_enabled=True)
        assert summary["n_filter_calls"] == 20
        np.testing.assert_allclose(_f(summary, "lambda_mean"), 1.5)

    def test_window_flush_on_empty_aggregator_returns_zero_record(self) -> None:
        """No prior observations: flush returns a zero-filled record."""
        agg = SafetyAggregator(n_pairs=1, cartesian_accel_capacity=10.0)
        window = agg.flush_window_stats()
        assert window["n_obs"] == 0
        assert window["lambda_mean"] == 0.0
        assert window["lambda_var"] == 0.0


class TestSafetyAggregatorEventCounters:
    """Fallback + QP-infeasible counters feed the audit-gate hook."""

    def test_fallback_fires_counted_in_total_and_window(self) -> None:
        agg = SafetyAggregator(n_pairs=1, cartesian_accel_capacity=10.0)
        for i in range(10):
            agg.observe(
                {("a", "b"): 0.0},
                fallback_fired=(i % 3 == 0),
            )
        window = agg.flush_window_stats()
        assert window["n_fallback_fires"] == 4  # i=0,3,6,9

    def test_qp_infeasible_counted_separately_from_fallback(self) -> None:
        """ConcertoSafetyInfeasible is a worse signal than fallback-fired."""
        agg = SafetyAggregator(n_pairs=1, cartesian_accel_capacity=10.0)
        agg.observe({("a", "b"): 0.0}, fallback_fired=True)
        agg.observe({("a", "b"): 0.0}, qp_infeasible=True)
        agg.observe({("a", "b"): 0.0}, fallback_fired=True, qp_infeasible=True)
        summary = agg.finalise(safety_enabled=True)
        assert summary["n_fallback_fires"] == 2
        assert summary["n_qp_infeasible"] == 2


class TestSafetyAggregatorShapeValidation:
    """Loud-fail on dimension mismatch (P1.04.5)."""

    def test_observe_rejects_wrong_n_pairs(self) -> None:
        agg = SafetyAggregator(n_pairs=2, cartesian_accel_capacity=10.0)
        with pytest.raises(ValueError, match="n_pairs"):
            agg.observe({("a", "b"): 0.5})  # shape (1,), expected (2,)


class TestSafetyAggregatorBrakingFires:
    """P1.04.6: ``braking_fired`` populates ``n_braking_fires`` and ``braking_fire_rate``."""

    def test_no_braking_fires_yields_zero_fields(self) -> None:
        """When no step fired braking, both fields stay 0 / 0.0 (matches pre-P1.04.6 vintage)."""
        agg = SafetyAggregator(n_pairs=1, cartesian_accel_capacity=10.0)
        agg.observe({("a", "b"): 0.1})
        summary = agg.finalise(safety_enabled=True)
        assert summary["n_braking_fires"] == 0
        assert summary["braking_fire_rate"] == 0.0

    def test_braking_fired_counter_increments_and_rate_normalises(self) -> None:
        """3 fires over 10 steps → n_braking_fires=3, braking_fire_rate=0.3."""
        agg = SafetyAggregator(n_pairs=1, cartesian_accel_capacity=10.0)
        for k in range(10):
            agg.observe({("a", "b"): 0.1}, braking_fired=(k < 3))
        summary = agg.finalise(safety_enabled=True)
        assert summary["n_braking_fires"] == 3
        assert summary["braking_fire_rate"] == pytest.approx(0.3)

    def test_window_flush_reports_braking_fires_and_resets(self) -> None:
        """The per-rollout-window dict carries ``n_braking_fires`` and the counter resets."""
        agg = SafetyAggregator(n_pairs=1, cartesian_accel_capacity=10.0)
        for _ in range(5):
            agg.observe({("a", "b"): 0.1}, braking_fired=True)
        window_a = agg.flush_window_stats()
        assert window_a["n_braking_fires"] == 5
        # Next window starts at zero — running cell-total still climbs.
        for _ in range(2):
            agg.observe({("a", "b"): 0.1}, braking_fired=True)
        window_b = agg.flush_window_stats()
        assert window_b["n_braking_fires"] == 2
        summary = agg.finalise(safety_enabled=True)
        assert summary["n_braking_fires"] == 7

    def test_braking_fires_counted_independently_of_fallback_and_qp_infeasible(self) -> None:
        """The three event flags are independent counters on the same observe call."""
        agg = SafetyAggregator(n_pairs=1, cartesian_accel_capacity=10.0)
        agg.observe(
            {("a", "b"): 0.1}, fallback_fired=True, qp_infeasible=False, braking_fired=False
        )
        agg.observe(
            {("a", "b"): 0.1}, fallback_fired=False, qp_infeasible=True, braking_fired=False
        )
        agg.observe(
            {("a", "b"): 0.1}, fallback_fired=False, qp_infeasible=False, braking_fired=True
        )
        summary = agg.finalise(safety_enabled=True)
        assert summary["n_fallback_fires"] == 1
        assert summary["n_qp_infeasible"] == 1
        assert summary["n_braking_fires"] == 1

    def test_predictor_kind_round_trips(self) -> None:
        agg = SafetyAggregator(n_pairs=1, cartesian_accel_capacity=10.0)
        agg.observe({("a", "b"): 0.1})
        summary = agg.finalise(safety_enabled=True, predictor_kind="constant_velocity")
        assert summary["predictor_kind"] == "constant_velocity"


class TestSafetyAggregatorSafetyDisabledPath:
    """When safety_enabled=False, the summary still emits + flags the operator intent."""

    def test_safety_disabled_flag_propagates(self) -> None:
        agg = SafetyAggregator(n_pairs=1, cartesian_accel_capacity=10.0)
        summary = agg.finalise(safety_enabled=False)
        assert summary["safety_enabled"] is False
