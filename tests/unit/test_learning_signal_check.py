# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``concerto.training.learning_signal_check`` (closes review P1-1, P1-2).

External-review findings (2026-05-16):

- **P1-1 (misnamed assertion).** The historical
  ``assert_positive_learning_slope`` returns a report rather than
  raising, and the slope test is a *trip-wire* for ADR-002 §Risks #1
  — not a formal monotonic-improvement guarantee. The renamed
  ``check_positive_learning_slope`` drops the "assert" verb to reflect
  the contract.
- **P1-2 (vacuous-pass-on-short-curves).** The pre-amendment branch
  ``if n < min_episodes: return SlopeReport(..., passed=True)``
  returned a *passing* report on too-short curves, masking
  insufficient-data conditions in downstream evaluation. The
  replacement returns ``CheckStatus.INSUFFICIENT_DATA`` and
  ``passed=False`` so consumers can distinguish "trip-wire cleared"
  from "data too short to evaluate".

These tests pin the four-case status taxonomy:

- ``PASSED`` — slope strictly positive, p-value below ``alpha``.
- ``FAILED`` — slope non-positive (flat or regressing).
- ``INSUFFICIENT_DATA`` — fewer than ``min_episodes`` entries.
- ``INVALID`` — curve contains NaN or +/- inf.

References:
- ADR-002 §Risks #1 + §Revision history (2026-05-16 amendment).
- plan/05 §6 criterion 4 (slope test alpha = 0.05 trip-wire).
- ``src/concerto/training/learning_signal_check.py`` — module under
  test.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from concerto.training.ego_aht import RewardCurve
from concerto.training.learning_signal_check import (
    DEFAULT_ALPHA,
    CheckStatus,
    SlopeReport,
    check_positive_learning_slope,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


def _curve(rewards: Sequence[float], *, run_id: str = "0" * 16) -> RewardCurve:
    """Build a :class:`RewardCurve` populated only on the per-episode field."""
    return RewardCurve(run_id=run_id, per_episode_ego_rewards=list(rewards))


def test_check_status_insufficient_data_on_short_curve() -> None:
    """Short curve (n < min_episodes) ⇒ INSUFFICIENT_DATA, not PASSED.

    This is the headline change vs the pre-amendment vacuous-pass
    branch (external-review P1-2): callers can now distinguish
    "trip-wire cleared" from "data too short to evaluate". Pinning the
    INSUFFICIENT_DATA value here is the canonical regression guard.
    """
    short = _curve([1.0, 2.0, 3.0])  # n=3, well below DEFAULT_MIN_EPISODES=20
    report = check_positive_learning_slope(short)
    assert report.status is CheckStatus.INSUFFICIENT_DATA
    assert report.passed is False
    assert report.n_episodes == 3


def test_check_status_passed_on_clear_positive_slope() -> None:
    """Synthetic positive slope at alpha = 0.05 ⇒ PASSED.

    Linear ramp with a small additive jitter; the slope is dominant
    and ``p_value`` is astronomically below 0.05.
    """
    rng = np.random.default_rng(0)
    n = 200
    x = np.arange(n, dtype=np.float64)
    rewards = (0.1 * x + rng.normal(scale=0.05, size=n)).tolist()
    report = check_positive_learning_slope(_curve(rewards))
    assert report.status is CheckStatus.PASSED
    assert report.passed is True
    assert report.slope > 0.0
    assert report.p_value < DEFAULT_ALPHA


def test_check_status_failed_on_flat_or_negative_slope() -> None:
    """Flat or negative-slope curve ⇒ FAILED.

    Two synthetic curves:
    - Flat: zero-variance returns FAILED (the existing zero-variance
      branch is preserved, but ``status`` is now set rather than only
      ``passed=False``).
    - Strictly decreasing: negative slope returns FAILED.
    """
    flat = _curve([1.0] * 50)
    flat_report = check_positive_learning_slope(flat)
    assert flat_report.status is CheckStatus.FAILED
    assert flat_report.passed is False

    n = 100
    decreasing = _curve((np.linspace(10.0, 0.0, n)).tolist())
    decreasing_report = check_positive_learning_slope(decreasing)
    assert decreasing_report.status is CheckStatus.FAILED
    assert decreasing_report.passed is False
    assert decreasing_report.slope < 0.0


def test_check_status_invalid_on_nan_or_inf() -> None:
    """NaN or inf in the reward curve ⇒ INVALID.

    The pre-amendment code would either crash inside
    ``scipy.stats.linregress`` (on NaN) or silently propagate inf
    through the slope. The new branch detects them up front and
    returns INVALID so downstream evaluation does not consume garbage
    statistics.
    """
    with_nan = _curve([1.0, 2.0, float("nan"), 4.0] + [5.0] * 30)
    nan_report = check_positive_learning_slope(with_nan)
    assert nan_report.status is CheckStatus.INVALID
    assert nan_report.passed is False

    with_pos_inf = _curve([1.0, 2.0, float("inf"), 4.0] + [5.0] * 30)
    pos_inf_report = check_positive_learning_slope(with_pos_inf)
    assert pos_inf_report.status is CheckStatus.INVALID
    assert pos_inf_report.passed is False

    with_neg_inf = _curve([1.0, 2.0, float("-inf"), 4.0] + [5.0] * 30)
    neg_inf_report = check_positive_learning_slope(with_neg_inf)
    assert neg_inf_report.status is CheckStatus.INVALID
    assert neg_inf_report.passed is False


def test_passed_is_derived_from_status() -> None:
    """The deprecated ``passed`` field is exactly ``status is CheckStatus.PASSED``."""
    rng = np.random.default_rng(1)
    rewards = (0.05 * np.arange(50, dtype=np.float64) + rng.normal(scale=0.01, size=50)).tolist()
    report = check_positive_learning_slope(_curve(rewards))
    assert report.passed == (report.status is CheckStatus.PASSED)


def test_slope_report_is_frozen() -> None:
    """SlopeReport remains a frozen dataclass."""
    report = SlopeReport(
        slope=0.1,
        intercept=0.0,
        r_squared=0.9,
        p_value=0.001,
        alpha=0.05,
        n_episodes=50,
        status=CheckStatus.PASSED,
        passed=True,
    )
    import dataclasses

    with pytest.raises(dataclasses.FrozenInstanceError):
        report.slope = 99.0  # type: ignore[misc]


def test_check_status_enum_completeness() -> None:
    """The enum has exactly the four named values (no silent drift)."""
    assert {member.name for member in CheckStatus} == {
        "PASSED",
        "FAILED",
        "INSUFFICIENT_DATA",
        "INVALID",
    }


def test_deprecated_path_re_exports_with_warning() -> None:
    """`concerto.training.empirical_guarantee.assert_positive_learning_slope`
    still resolves (back-compat shim) and emits a DeprecationWarning that
    points consumers to the new path. Removal target: v0.6.0.
    """
    with pytest.warns(DeprecationWarning, match=r"assert_positive_learning_slope"):
        from concerto.training.empirical_guarantee import (
            assert_positive_learning_slope,
        )
    # The alias resolves to the renamed function so call-sites continue
    # to receive the new behaviour (INSUFFICIENT_DATA on short curves).
    # The shim returns the symbol through ``__getattr__`` which pyright
    # types as ``object``; the runtime behaviour is verified below.
    short_report = assert_positive_learning_slope(_curve([1.0, 2.0]))  # type: ignore[operator]
    assert short_report.status is CheckStatus.INSUFFICIENT_DATA
