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
- :func:`solve_qp_stub` — small Clarabel QP benchmark used by the M2
  saturation guard test (T3.10 replaced the no-op M2 stub with a real
  solve at a name-preserved entry point).
"""

from __future__ import annotations

import numpy as np

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
from concerto.safety.solvers import ClarabelSolver

#: Module-level solver for :func:`solve_qp_stub`. A single instance is
#: reused so per-call instantiation overhead is amortised across the
#: M2 saturation-guard test's parametrised invocations (ADR-004
#: §"OSCBF target"; ADR-006 §Risks R5).
_BENCH_SOLVER: ClarabelSolver = ClarabelSolver()

#: Pre-built benchmark QP matrices, sized to the smallest representative
#: shape that exercises the QP path (n=2 box-constrained quadratic
#: ``min 1/2 x^T I x + q^T x s.t. ||x||_inf <= 1``). Module-level
#: constants avoid per-call allocation; the QP is the same on every
#: call so any solver caches stay warm.
_BENCH_P: FloatArray = np.eye(2, dtype=np.float64)
_BENCH_Q: FloatArray = np.array([0.1, -0.2], dtype=np.float64)
_BENCH_A: FloatArray = np.vstack(
    [np.eye(2, dtype=np.float64), -np.eye(2, dtype=np.float64)]
).astype(np.float64, copy=False)
_BENCH_B: FloatArray = np.ones(4, dtype=np.float64)


def solve_qp_stub(*args: object, **kwargs: object) -> tuple[float, float]:
    """Real CBF-QP benchmark solve (T3.10; ADR-004 §"OSCBF target").

    T3.10 replaces M2's no-op stub with an actual Clarabel solve on a
    small representative QP. The function name is preserved for
    backward compatibility with ``tests/property/test_qp_saturation.py``
    (the M2 acceptance gate) per plan/03 §8 + the M3 brief Hard Rule
    #4: the saturation guard's regime check (drop_rate >= 10 % or
    latency >= 100 ms; ADR-006 §Risks R5) is independent of the QP
    body, so after replacement it still fires only at the URLLC
    ``saturation`` profile, not at ``lossy``.

    The QP is pre-built at module load and the solver instance is
    module-level so the per-call hot path is just the Clarabel solve
    — no allocation or instantiation overhead. The ``*args`` /
    ``**kwargs`` signature is preserved for the ``saturation_guard``
    contract (callers pass no args; the placeholder forwarding kept
    the signature open for future per-call inputs).

    Args:
        *args: Ignored (M2 backward-compat surface).
        **kwargs: Ignored.

    Returns:
        ``(decision_norm, elapsed_seconds)`` matching the M2 stub
        contract; the M2 saturation guard reads only the timing
        envelope so the decision value is informational.
    """
    del args, kwargs
    x, ms = _BENCH_SOLVER.solve(_BENCH_P, _BENCH_Q, _BENCH_A, _BENCH_B)
    return float(np.linalg.norm(x)), ms / 1000.0


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
