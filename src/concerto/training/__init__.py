# SPDX-License-Identifier: Apache-2.0
"""CONCERTO training stack — ego-AHT HAPPO wrapper + seeding + logging.

Implements the frozen-partner ego-AHT training algorithm from ADR-002
using the HARL fork (ADR-002 §"HARL dependency"). The seeding module
provides the determinism harness required by project principle P6.
"""

from concerto.training.empirical_guarantee import (
    DEFAULT_THRESHOLD,
    DEFAULT_WINDOW,
    GuaranteeReport,
    assert_non_decreasing_window,
    plot_reward_curve,
)

__all__ = [
    "DEFAULT_THRESHOLD",
    "DEFAULT_WINDOW",
    "GuaranteeReport",
    "assert_non_decreasing_window",
    "plot_reward_curve",
]
