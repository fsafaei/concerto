# SPDX-License-Identifier: Apache-2.0
"""CONCERTO training stack — ego-AHT HAPPO wrapper + seeding + logging.

Implements the frozen-partner ego-AHT training algorithm from ADR-002
using the HARL fork (ADR-002 §"HARL dependency"). The seeding module
provides the determinism harness required by project principle P6.
"""

from concerto.training.learning_signal_check import (
    DEFAULT_ALPHA,
    DEFAULT_MIN_EPISODES,
    DEFAULT_THRESHOLD,
    DEFAULT_WINDOW,
    CheckStatus,
    GuaranteeReport,
    SlopeReport,
    assert_non_decreasing_window,
    check_positive_learning_slope,
    plot_reward_curve,
    plot_slope_curve,
)

# Note: the pre-rename ``assert_positive_learning_slope`` is intentionally
# NOT re-exported here so new code resolves to the canonical
# ``check_positive_learning_slope`` (external-review P1-1, 2026-05-16).
# Legacy callers can still reach the old name via
# :mod:`concerto.training.empirical_guarantee` which emits a
# DeprecationWarning on first access; removal target v0.6.0.
__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_MIN_EPISODES",
    "DEFAULT_THRESHOLD",
    "DEFAULT_WINDOW",
    "CheckStatus",
    "GuaranteeReport",
    "SlopeReport",
    "assert_non_decreasing_window",
    "check_positive_learning_slope",
    "plot_reward_curve",
    "plot_slope_curve",
]
