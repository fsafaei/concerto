# SPDX-License-Identifier: Apache-2.0
"""CONCERTO safety stack — exp CBF-QP + conformal overlay + OSCBF + braking.

Implements the safety filter architecture decided in ADR-004 and the
three-table safety reporting format from ADR-014.

The full filter lands in M3; M2 ships only the no-op QP stub below so the
QP-saturation guard property test (T2.7) can run from M2 — the test exists
from now on so M3 cannot regress it.
"""

from __future__ import annotations


def solve_qp_stub(*args: object, **kwargs: object) -> tuple[float, float]:
    """No-op stand-in for the M3 inner CBF QP solver (ADR-004 §"OSCBF target").

    Returns ``(0.0, 1e-7)`` so the QP-saturation guard property test
    (T2.7) can run from M2 without depending on the real solver. M3
    replaces this with the real exp CBF + OSCBF stack; the saturation
    guard test must remain green afterwards (ADR-006 §Risks R5).

    Args:
        *args: Forwarded to the real solver in M3 (ignored here).
        **kwargs: Forwarded to the real solver in M3 (ignored here).

    Returns:
        A ``(decision, elapsed_seconds)`` pair: ``0.0`` for the QP
        solution stand-in and ``1e-7`` seconds as the trivial timing.
    """
    del args, kwargs
    return (0.0, 1e-7)


__all__ = ["solve_qp_stub"]
