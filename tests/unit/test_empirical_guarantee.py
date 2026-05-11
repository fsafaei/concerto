# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``concerto.training.empirical_guarantee`` (T4b.13).

Covers ADR-002 §Risks #1 (the trip-wire's pure-function side) and plan/05
§6 criterion 4 (moving-window-of-10, threshold-0.8 default). The
integration side — running the 100k-frame experiment end-to-end — lives
in ``tests/integration/test_empirical_guarantee.py``.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from concerto.training.ego_aht import RewardCurve
from concerto.training.empirical_guarantee import (
    DEFAULT_THRESHOLD,
    DEFAULT_WINDOW,
    GuaranteeReport,
    assert_non_decreasing_window,
    plot_reward_curve,
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
