# SPDX-License-Identifier: Apache-2.0
"""CONCERTO training stack — ego-AHT HAPPO wrapper + seeding + logging.

Implements the frozen-partner ego-AHT training algorithm from ADR-002
using the HARL fork (ADR-002 §"HARL dependency"). The seeding module
provides the determinism harness required by project principle P6.
"""

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

__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_MIN_EPISODES",
    "DEFAULT_THRESHOLD",
    "DEFAULT_WINDOW",
    "GuaranteeReport",
    "SlopeReport",
    "assert_non_decreasing_window",
    "assert_positive_learning_slope",
    "plot_reward_curve",
    "plot_slope_curve",
]
