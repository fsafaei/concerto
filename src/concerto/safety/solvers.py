# SPDX-License-Identifier: Apache-2.0
"""QP solver strategies (ADR-004 §Decision; plan/03 §3.7).

Implements the strategy interface from plan/03 §3.7: a Protocol plus two
concrete solvers (Clarabel default, OSQP fallback). Both consume QP
problems in the canonical form

    min   1/2 * x^T P x + q^T x
    s.t.  A x <= b

and return ``(solution, solve_time_ms)``. Infeasibility raises
:class:`concerto.safety.errors.ConcertoSafetyInfeasible`; the caller
MUST route to the braking fallback (ADR-004 risk-mitigation #1) — the
conformal QP is not a valid recovery path because Theorem 3
(Huriot & Sibai 2025) only bounds the *average* loss.

The two solvers are kept behind a strategy interface so a future ADR can
swap them without touching the CBF-QP / OSCBF call sites. Selection is
via Hydra config ``safety.solver = "clarabel" | "osqp"``.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np
from qpsolvers import solve_qp  # pyright: ignore[reportUnknownVariableType]
from scipy.sparse import csc_matrix  # pyright: ignore[reportMissingTypeStubs]

from concerto.safety.errors import ConcertoSafetyInfeasible

if TYPE_CHECKING:
    from concerto.safety.api import FloatArray

#: Solver names accepted by :func:`make_solver` (ADR-004 §Decision).
SUPPORTED_SOLVERS: tuple[str, ...] = ("clarabel", "osqp")

#: OSQP convergence overrides for 1e-6-level agreement with Clarabel.
#: Defaults (eps_abs=1e-3) are too loose for the T3.2 acceptance test.
_OSQP_TIGHT_SETTINGS: dict[str, Any] = {
    "eps_abs": 1e-9,
    "eps_rel": 1e-9,
    "max_iter": 50_000,
    "polishing": True,
    "verbose": False,
}

#: Clarabel verbosity override; tolerances are already tight at default.
_CLARABEL_QUIET: dict[str, Any] = {"verbose": False}


@runtime_checkable
class QPSolver(Protocol):
    """Strategy interface for a CBF-QP solver (plan/03 §3.7; ADR-004 §Decision).

    Implementations consume a convex QP in the canonical form
    ``min 1/2 x^T P x + q^T x s.t. A x <= b`` and return the optimal
    primal ``x`` together with the wall-clock solve time. Infeasibility
    raises :class:`ConcertoSafetyInfeasible`; the caller routes to the
    braking fallback (ADR-004 risk-mitigation #1).
    """

    def solve(
        self,
        P: FloatArray,
        q: FloatArray,
        A: FloatArray,
        b: FloatArray,
        *,
        warm_start: bool = True,
    ) -> tuple[FloatArray, float]:
        """Solve a convex QP (plan/03 §3.7; ADR-004 §Decision).

        Args:
            P: Symmetric positive-definite cost matrix, shape ``(n, n)``.
            q: Linear-cost vector, shape ``(n,)``.
            A: Inequality-constraint matrix, shape ``(m, n)``.
            b: Inequality-constraint bound, shape ``(m,)``.
            warm_start: If ``True`` and the implementation supports it,
                seed the solver with the previous solution. ADR-004
                §Decision pins this on for OSQP; Clarabel is interior
                point and ignores the flag.

        Returns:
            Tuple ``(x, solve_time_ms)`` where ``x`` is the primal optimum.

        Raises:
            ConcertoSafetyInfeasible: When the QP is infeasible. The
                caller MUST route to the braking fallback in this case
                (ADR-004 risk-mitigation #1).
        """
        ...


def _solve_canonical(
    *,
    solver_name: str,
    extra: dict[str, Any],
    P: FloatArray,
    q: FloatArray,
    A: FloatArray,
    b: FloatArray,
    initvals: FloatArray | None,
) -> tuple[FloatArray, float]:
    """Forward to ``qpsolvers.solve_qp`` and time the call.

    Internal shared backbone for the two concrete solver classes; not
    part of the public surface (no ADR ref needed — leading underscore).
    """
    P_csc = csc_matrix(P)
    A_csc = csc_matrix(A)
    start = time.perf_counter()
    raw: np.ndarray[Any, np.dtype[Any]] | None = solve_qp(  # pyright: ignore[reportUnknownVariableType]
        P=P_csc,
        q=q,
        G=A_csc,
        h=b,
        solver=solver_name,
        initvals=initvals,
        **extra,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    if raw is None:
        msg = (
            f"{solver_name}: QP infeasible (P shape {P.shape}, A shape {A.shape}). "
            "Caller MUST route to braking fallback (ADR-004 risk-mitigation #1)."
        )
        raise ConcertoSafetyInfeasible(msg)
    x: FloatArray = np.asarray(raw, dtype=np.float64)
    return x, elapsed_ms


class ClarabelSolver:
    """Default QP solver: Clarabel interior-point (ADR-004 §Decision).

    Clarabel is a deterministic, GPU-friendly interior-point solver and
    the Phase-0 default per plan/03 §"QP solver". Its native tolerances
    are already tight enough for the T3.2 1e-6 agreement target so no
    tolerance overrides are passed; the ``warm_start`` flag is accepted
    but ignored (interior-point methods do not benefit from warm starts).
    """

    name: str = "clarabel"

    def __init__(self) -> None:
        """Construct a fresh Clarabel solver instance (ADR-004 §Decision)."""
        self._last_x: FloatArray | None = None

    def solve(
        self,
        P: FloatArray,
        q: FloatArray,
        A: FloatArray,
        b: FloatArray,
        *,
        warm_start: bool = True,
    ) -> tuple[FloatArray, float]:
        """Solve via Clarabel; ``warm_start`` is accepted but ignored (ADR-004 §Decision).

        See :meth:`QPSolver.solve`. Clarabel is interior-point; warm
        starts have no measurable effect, so we accept the flag for
        Protocol compatibility but discard it.
        """
        del warm_start
        x, elapsed_ms = _solve_canonical(
            solver_name="clarabel",
            extra=_CLARABEL_QUIET,
            P=P,
            q=q,
            A=A,
            b=b,
            initvals=None,
        )
        self._last_x = x
        return x, elapsed_ms


class OSQPSolver:
    """Fallback QP solver: OSQP operator-splitting (ADR-004 §Decision).

    OSQP is an operator-splitting method that benefits substantially
    from warm-starting on partner-swap and constraint-graph stability —
    plan/03 §8 calls it out as the preferred response if URLLC
    saturation re-emerges after T3.10. Native tolerances are loose
    (``eps_abs=1e-3``); this wrapper tightens to ``1e-9`` so the T3.2
    cross-solver agreement check holds at 1e-6.
    """

    name: str = "osqp"

    def __init__(self) -> None:
        """Construct a fresh OSQP solver instance (ADR-004 §Decision)."""
        self._last_x: FloatArray | None = None

    def solve(
        self,
        P: FloatArray,
        q: FloatArray,
        A: FloatArray,
        b: FloatArray,
        *,
        warm_start: bool = True,
    ) -> tuple[FloatArray, float]:
        """Solve via OSQP, warm-starting from the previous solution when shape matches.

        See :meth:`QPSolver.solve` (ADR-004 §Decision). The shape-match
        check guards against feeding a stale ``initvals`` into a QP of
        different dimension after a partner swap (ADR-006 risk #3).
        """
        initvals: FloatArray | None = None
        if warm_start and self._last_x is not None and self._last_x.shape == q.shape:
            initvals = self._last_x
        x, elapsed_ms = _solve_canonical(
            solver_name="osqp",
            extra=_OSQP_TIGHT_SETTINGS,
            P=P,
            q=q,
            A=A,
            b=b,
            initvals=initvals,
        )
        self._last_x = x
        return x, elapsed_ms


def make_solver(name: str) -> QPSolver:
    """Construct a QP solver by name (plan/03 §3.7; ADR-004 §Decision).

    Args:
        name: One of :data:`SUPPORTED_SOLVERS` — ``"clarabel"`` (default
            interior-point) or ``"osqp"`` (operator-splitting fallback).

    Returns:
        A fresh :class:`QPSolver` instance.

    Raises:
        ValueError: When ``name`` is not in :data:`SUPPORTED_SOLVERS`.
    """
    if name == "clarabel":
        return ClarabelSolver()
    if name == "osqp":
        return OSQPSolver()
    msg = f"Unknown solver {name!r}; expected one of {SUPPORTED_SOLVERS}"
    raise ValueError(msg)


__all__ = [
    "SUPPORTED_SOLVERS",
    "ClarabelSolver",
    "OSQPSolver",
    "QPSolver",
    "make_solver",
]
