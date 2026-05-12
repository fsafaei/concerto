# SPDX-License-Identifier: Apache-2.0
# pyright: reportAttributeAccessIssue=false
#
# Suppresses pyright's complaints about ``scipy.stats.linregress``
# attribute access (see same suppression in
# ``src/concerto/training/empirical_guarantee.py``).
"""Unit tests for ``concerto.training.empirical_guarantee`` (T4b.13).

Covers ADR-002 §Risks #1 (the trip-wire's pure-function side):

- The canonical slope test (:func:`assert_positive_learning_slope`) and
  its plotting helper.
- The legacy moving-window-of-K helper retained for backward
  comparison (issue #62 root-cause writeup).

The integration side — running the 100k-frame experiment end-to-end —
lives in ``tests/integration/test_empirical_guarantee.py``.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest
from scipy import stats

from concerto.training.ego_aht import RewardCurve
from concerto.training.empirical_guarantee import (
    DEFAULT_ALPHA,
    DEFAULT_MIN_EPISODES,
    DEFAULT_THRESHOLD,
    DEFAULT_WINDOW,
    GuaranteeReport,
    SlopeReport,
    assert_non_decreasing_window,
    assert_positive_learning_slope,
    plot_reward_curve,
    plot_slope_curve,
)

if TYPE_CHECKING:
    from pathlib import Path


def _curve(rewards: list[float], *, run_id: str = "0" * 16) -> RewardCurve:
    """Build a :class:`RewardCurve` populated only on the per-episode field."""
    return RewardCurve(run_id=run_id, per_episode_ego_rewards=list(rewards))


class TestAssertNonDecreasingWindow:
    def test_empty_curve_is_vacuously_passing(self) -> None:
        """Plan/05 §6 #4: too-short curve returns n_intervals=0, passed=True."""
        report = assert_non_decreasing_window(_curve([]))
        assert report.n_intervals == 0
        assert report.n_nondecreasing == 0
        assert report.fraction == 1.0
        assert report.passed is True

    def test_curve_shorter_than_window_is_vacuous(self) -> None:
        """Plan/05 §6 #4: len(curve) < window returns passed=True (vacuous)."""
        rewards = [0.0, 1.0, 2.0, 3.0]  # len=4 < default window 10
        report = assert_non_decreasing_window(_curve(rewards))
        assert report.n_intervals == 0
        assert report.passed is True

    def test_strictly_increasing_curve_passes_with_fraction_1(self) -> None:
        """Plan/05 §6 #4: monotone-up curve → fraction==1.0, passed=True."""
        rewards = [float(i) for i in range(40)]
        report = assert_non_decreasing_window(_curve(rewards))
        assert report.fraction == 1.0
        assert report.passed is True
        assert report.n_intervals == len(rewards) - DEFAULT_WINDOW

    def test_strictly_decreasing_curve_fails_with_fraction_0(self) -> None:
        """Plan/05 §6 #4: monotone-down curve → fraction==0.0, passed=False."""
        rewards = [-float(i) for i in range(40)]
        report = assert_non_decreasing_window(_curve(rewards))
        assert report.fraction == 0.0
        assert report.passed is False

    def test_known_hand_rolled_fraction_at_window_3(self) -> None:
        """Plan/05 §6 #4: lock the off-by-one with a hand-computed fraction.

        At window=3 over rewards=[1, 2, 3, 2, 1, 2, 3, 4, 5]:
          - means at offsets 0..6 = [2.0, 2.33, 2.0, 1.67, 2.0, 3.0, 4.0]
          - 6 adjacent-pair intervals
          - non-decreasing pairs: (2.0->2.33), (1.67->2.0), (2.0->3.0), (3.0->4.0) = 4
          - fraction = 4 / 6 ≈ 0.6667
        """
        rewards = [1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        report = assert_non_decreasing_window(_curve(rewards), window=3)
        assert report.n_intervals == 6
        assert report.n_nondecreasing == 4
        assert report.fraction == pytest.approx(4 / 6)
        # 4/6 ≈ 0.667 < default threshold 0.8 → fails the default check.
        assert report.passed is False

    def test_threshold_at_exact_fraction_passes(self) -> None:
        """passed is inclusive: fraction == threshold → passed."""
        rewards = [1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        report = assert_non_decreasing_window(_curve(rewards), window=3, threshold=4 / 6)
        assert report.passed is True

    def test_zero_window_is_treated_as_degenerate(self) -> None:
        """Defensive: a non-positive window returns the vacuous-pass branch."""
        report = assert_non_decreasing_window(_curve([1.0, 2.0, 3.0]), window=0)
        assert report.n_intervals == 0
        assert report.passed is True

    def test_default_window_and_threshold_match_plan(self) -> None:
        """Plan/05 §6 #4: defaults are window=10, threshold=0.8."""
        assert DEFAULT_WINDOW == 10
        assert DEFAULT_THRESHOLD == 0.8


class TestPlotRewardCurve:
    def test_emits_png_and_json_at_expected_paths(self, tmp_path: Path) -> None:
        """T4b.13: plotting returns (png, json) paths and both exist."""
        rewards = [float(i) for i in range(30)]
        curve = _curve(rewards, run_id="abcdef0123456789")
        report = assert_non_decreasing_window(curve)
        png_path, json_path = plot_reward_curve(curve, report=report, out_dir=tmp_path)
        assert png_path == tmp_path / "abcdef0123456789.png"
        assert json_path == tmp_path / "abcdef0123456789.json"
        assert png_path.exists()
        assert json_path.exists()
        assert png_path.stat().st_size > 0

    def test_json_sidecar_round_trips(self, tmp_path: Path) -> None:
        """T4b.13: JSON sidecar carries curve + report so the figure can be regenerated."""
        rewards = [0.0, 1.0, 2.0, 1.5, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0]
        curve = _curve(rewards)
        report = assert_non_decreasing_window(curve, window=3, threshold=0.5)
        _, json_path = plot_reward_curve(curve, report=report, out_dir=tmp_path)
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        assert payload["run_id"] == curve.run_id
        assert payload["per_episode_ego_rewards"] == rewards
        assert payload["report"]["window"] == 3
        assert payload["report"]["threshold"] == 0.5
        assert payload["report"]["n_intervals"] == report.n_intervals
        assert payload["report"]["passed"] == report.passed

    def test_handles_empty_curve_without_crashing(self, tmp_path: Path) -> None:
        """T4b.13: plotting an empty curve still produces both files."""
        curve = _curve([])
        report = assert_non_decreasing_window(curve)
        png_path, json_path = plot_reward_curve(curve, report=report, out_dir=tmp_path)
        assert png_path.exists()
        assert json_path.exists()
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        assert payload["per_episode_ego_rewards"] == []


class TestGuaranteeReport:
    def test_frozen(self) -> None:
        """Reports are immutable artefacts (project P6 reproducibility)."""
        import dataclasses

        report = GuaranteeReport(
            window=10,
            threshold=0.8,
            n_intervals=5,
            n_nondecreasing=4,
            fraction=0.8,
            passed=True,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            report.passed = False  # type: ignore[misc]


class TestAssertPositiveLearningSlope:
    def test_empty_curve_is_vacuously_passing(self) -> None:
        """ADR-002 §Risks #1: empty curve returns vacuous pass (n=0, p=1, passed=True)."""
        report = assert_positive_learning_slope(_curve([]))
        assert report.n_episodes == 0
        assert report.slope == 0.0
        assert report.p_value == 1.0
        assert report.passed is True

    def test_curve_shorter_than_min_episodes_is_vacuous(self) -> None:
        """ADR-002 §Risks #1: below DEFAULT_MIN_EPISODES → vacuous pass."""
        rewards = [float(i) for i in range(DEFAULT_MIN_EPISODES - 1)]
        report = assert_positive_learning_slope(_curve(rewards))
        assert report.n_episodes == DEFAULT_MIN_EPISODES - 1
        assert report.slope == 0.0
        assert report.passed is True

    def test_strictly_increasing_curve_passes(self) -> None:
        """Issue #62: monotone-up curve clears the slope test with overwhelming significance."""
        rewards = [float(i) for i in range(50)]
        report = assert_positive_learning_slope(_curve(rewards))
        assert report.slope == pytest.approx(1.0, abs=1e-9)
        assert report.r_squared == pytest.approx(1.0, abs=1e-9)
        assert report.p_value < 1e-50
        assert report.passed is True

    def test_strictly_decreasing_curve_fails(self) -> None:
        """Issue #62: monotone-down curve fails (slope < 0)."""
        rewards = [-float(i) for i in range(50)]
        report = assert_positive_learning_slope(_curve(rewards))
        assert report.slope == pytest.approx(-1.0, abs=1e-9)
        # slope < 0 ⇒ one-sided p_value for "slope > 0" should be ≈ 1.
        assert report.p_value > 0.9
        assert report.passed is False

    def test_constant_zero_variance_curve_fails(self) -> None:
        """Zero-variance curve: slope=0, p_value=1.0, passed=False (no learning signal)."""
        rewards = [3.5] * 50
        report = assert_positive_learning_slope(_curve(rewards))
        assert report.slope == 0.0
        assert report.p_value == 1.0
        assert report.passed is False

    def test_noisy_positive_trend_passes_with_p_value_below_alpha(self) -> None:
        """Issue #62: trip-wire's intended use — noisy curve with positive trend passes.

        Simulates the kind of curve the trainer actually produces: per-
        episode rewards with stdev comparable to the total drift over the
        run, but a real underlying slope. The slope test should pick this
        out where the moving-window assertion would fail.
        """
        rng = np.random.default_rng(0)
        n = 2000
        true_slope = 0.005
        noise = rng.normal(scale=20.0, size=n)
        rewards = [true_slope * i + float(noise[i]) for i in range(n)]
        slope_report = assert_positive_learning_slope(_curve(rewards))
        window_report = assert_non_decreasing_window(_curve(rewards))
        assert slope_report.slope == pytest.approx(true_slope, abs=5e-4)
        assert slope_report.p_value < 1e-6
        assert slope_report.passed is True
        # The legacy moving-window assertion fails on the same curve —
        # this is the empirical justification (issue #62) for replacing
        # the canonical statistic with the slope test.
        assert window_report.fraction < 0.6
        assert window_report.passed is False

    def test_one_sided_p_value_matches_scipy_two_sided_half(self) -> None:
        """One-sided p_value = scipy_two_sided / 2 when slope > 0 (locks the math)."""
        rng = np.random.default_rng(7)
        n = 100
        x = np.arange(n, dtype=np.float64)
        rewards = (0.1 * x + rng.normal(scale=0.5, size=n)).astype(float).tolist()
        report = assert_positive_learning_slope(_curve(rewards))
        scipy_result = stats.linregress(x, np.asarray(rewards, dtype=np.float64))
        # Observed slope is positive (true 0.1), so one-sided p = two-sided / 2.
        assert report.p_value == pytest.approx(scipy_result.pvalue / 2.0, rel=1e-9)

    def test_alpha_at_exact_p_value_is_strictly_failing(self) -> None:
        """Strict inequality: p_value == alpha → passed=False (not equal)."""
        rng = np.random.default_rng(42)
        n = 200
        rewards = (0.0001 * np.arange(n) + rng.normal(scale=10, size=n)).tolist()
        report = assert_positive_learning_slope(_curve(rewards), alpha=1.0)
        # alpha=1.0 means every positive-slope result clears the gate.
        assert report.passed is (report.slope > 0.0)

    def test_defaults_match_specification(self) -> None:
        """Plan/05 §6 #4: defaults are alpha=0.05, min_episodes=20."""
        assert DEFAULT_ALPHA == 0.05
        assert DEFAULT_MIN_EPISODES == 20


class TestPlotSlopeCurve:
    def test_emits_png_and_json_at_expected_paths(self, tmp_path: Path) -> None:
        """T4b.13: slope plotting returns (png, json) paths and both exist."""
        rewards = [float(i) for i in range(30)]
        curve = _curve(rewards, run_id="fedcba9876543210")
        report = assert_positive_learning_slope(curve)
        png_path, json_path = plot_slope_curve(curve, report=report, out_dir=tmp_path)
        assert png_path == tmp_path / "fedcba9876543210.png"
        assert json_path == tmp_path / "fedcba9876543210.json"
        assert png_path.exists()
        assert json_path.exists()
        assert png_path.stat().st_size > 0

    def test_json_sidecar_round_trips(self, tmp_path: Path) -> None:
        """T4b.13: JSON sidecar carries curve + slope report so the figure can be regenerated."""
        rewards = [float(i) * 0.5 for i in range(50)]
        curve = _curve(rewards)
        report = assert_positive_learning_slope(curve)
        _, json_path = plot_slope_curve(curve, report=report, out_dir=tmp_path)
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        assert payload["run_id"] == curve.run_id
        assert payload["per_episode_ego_rewards"] == rewards
        assert payload["report"]["slope"] == pytest.approx(report.slope)
        assert payload["report"]["passed"] == report.passed
        assert payload["report"]["alpha"] == report.alpha

    def test_handles_empty_curve_without_crashing(self, tmp_path: Path) -> None:
        """T4b.13: plotting an empty curve still produces both files."""
        curve = _curve([])
        report = assert_positive_learning_slope(curve)
        png_path, json_path = plot_slope_curve(curve, report=report, out_dir=tmp_path)
        assert png_path.exists()
        assert json_path.exists()
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        assert payload["per_episode_ego_rewards"] == []


class TestSlopeReport:
    def test_frozen(self) -> None:
        """Reports are immutable artefacts (project P6 reproducibility)."""
        import dataclasses

        report = SlopeReport(
            slope=0.01,
            intercept=-60.0,
            r_squared=0.1,
            p_value=1e-20,
            alpha=0.05,
            n_episodes=2000,
            passed=True,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            report.passed = False  # type: ignore[misc]
