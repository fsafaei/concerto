# SPDX-License-Identifier: Apache-2.0
"""Deprecated re-export shim for the pre-rename empirical-guarantee module.

The module's contents moved to
:mod:`concerto.training.learning_signal_check` on 2026-05-16
(external-review P1-1: the "empirical-guarantee" label overclaimed the
contract — the slope test is a *trip-wire* for ADR-002 §Risks #1, not a
formal monotonic-improvement guarantee). This shim preserves the old
import path for v0.5.x consumers; removal target v0.6.0.

The shim re-exports every public symbol from the new module under its
new name *and* serves the deprecated function name
``assert_positive_learning_slope`` via a module-level
:func:`__getattr__` that emits a :class:`DeprecationWarning` on first
access. Direct attribute imports (``from
concerto.training.empirical_guarantee import GuaranteeReport``) resolve
to the new-path symbol without a warning.

References:
- ADR-002 §Revision history (2026-05-16) — module rename + function
  rename rationale.
- :mod:`concerto.training.learning_signal_check` — the new canonical
  home.
"""

from __future__ import annotations

import warnings

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

#: Names served lazily via :func:`__getattr__` with a
#: :class:`DeprecationWarning`. The canonical replacement is
#: :func:`concerto.training.learning_signal_check.check_positive_learning_slope`
#: (the renamed function); removal target v0.6.0.
_DEPRECATED_NAMES: frozenset[str] = frozenset({"assert_positive_learning_slope"})


def __getattr__(name: str) -> object:
    """Emit a :class:`DeprecationWarning` on first access of the legacy alias.

    Routes the deprecated function name
    ``assert_positive_learning_slope`` to the renamed
    :func:`concerto.training.learning_signal_check.check_positive_learning_slope`.
    Other attribute reads on this module are delegated to the standard
    Python import machinery (returning :class:`AttributeError` for
    names not re-exported above).
    """
    if name in _DEPRECATED_NAMES:
        warnings.warn(
            f"concerto.training.empirical_guarantee.{name} is deprecated "
            "(external-review P1-1, 2026-05-16); use "
            "concerto.training.learning_signal_check.check_positive_learning_slope "
            "(or concerto.training.check_positive_learning_slope) instead. "
            "Removal target: v0.6.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        return check_positive_learning_slope
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


# ``assert_positive_learning_slope`` is intentionally NOT placed in
# ``__all__`` so ``from concerto.training.empirical_guarantee import *``
# does not silently re-introduce the legacy name into downstream
# modules; the deprecated name is reachable only via the module-level
# :func:`__getattr__` shim (which emits the deprecation warning).
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
