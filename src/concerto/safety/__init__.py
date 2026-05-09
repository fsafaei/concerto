# SPDX-License-Identifier: Apache-2.0
"""CONCERTO safety stack — exp CBF-QP + conformal overlay + OSCBF + braking.

Implements the three-layer safety filter decided in ADR-004
(Wang-Ames-Egerstedt 2017 + Huriot & Sibai 2025 + Morton & Pavone 2025),
the per-task numeric bounds plumbing from ADR-006 §Decision, and the
three-table reporting format from ADR-014 §Decision.

Public surface (see :mod:`concerto.safety.api` for full docstrings):

- :class:`Bounds` — per-task numeric envelope (ADR-006).
- :class:`SafetyState` — mutable conformal-CBF state (ADR-004).
- :class:`FilterInfo` — telemetry payload (ADR-014).
- :class:`SafetyFilter` — integrated-stack Protocol (ADR-004).
- :class:`ConcertoSafetyInfeasible` — QP-infeasibility error (ADR-004).
- :data:`DEFAULT_EPSILON`, :data:`DEFAULT_ETA`, :data:`DEFAULT_WARMUP_STEPS` —
  conformal defaults (ADR-004 §Decision).

The full filter lands in M3; M2 ships only the no-op
:func:`solve_qp_stub` below so the QP-saturation guard property test
(T2.7) can run from M2 — the test exists from now on so M3 cannot
regress it. T3.10 (PR10) replaces the stub with the real interface.
"""

from __future__ import annotations

from concerto.safety.api import (
    DEFAULT_EPSILON,
    DEFAULT_ETA,
    DEFAULT_WARMUP_STEPS,
    Bounds,
    FilterInfo,
    FloatArray,
    SafetyFilter,
    SafetyState,
)
from concerto.safety.errors import ConcertoSafetyInfeasible


def solve_qp_stub(*args: object, **kwargs: object) -> tuple[float, float]:
    """No-op stand-in for the M3 inner CBF QP solver (ADR-004 §"OSCBF target").

    Returns ``(0.0, 1e-7)`` so the QP-saturation guard property test
    (T2.7) can run from M2 without depending on the real solver. T3.10
    (M3 PR10) replaces this with the real exp CBF + OSCBF stack; the
    saturation guard test must remain green afterwards (ADR-006 §Risks
    R5).

    Args:
        *args: Forwarded to the real solver in M3 (ignored here).
        **kwargs: Forwarded to the real solver in M3 (ignored here).

    Returns:
        A ``(decision, elapsed_seconds)`` pair: ``0.0`` for the QP
        solution stand-in and ``1e-7`` seconds as the trivial timing.
    """
    del args, kwargs
    return (0.0, 1e-7)


__all__ = [
    "DEFAULT_EPSILON",
    "DEFAULT_ETA",
    "DEFAULT_WARMUP_STEPS",
    "Bounds",
    "ConcertoSafetyInfeasible",
    "FilterInfo",
    "FloatArray",
    "SafetyFilter",
    "SafetyState",
    "solve_qp_stub",
]
