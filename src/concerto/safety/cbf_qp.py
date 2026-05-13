# SPDX-License-Identifier: Apache-2.0
"""Exponential CBF-QP backbone (Wang-Ames-Egerstedt 2017 §III; ADR-004 §Decision).

Pairwise per-agent QP filter::

    min_{u}  ||u - u_hat||^2
    s.t.     A_pair u <= b_pair + lambda_pair    (per pair (i, j))
             -alpha_i <= u_i[k] <= alpha_i       (per agent component, L-infty box)

For double-integrator dynamics with bounded acceleration, the
Safety Barrier Certificate (Wang-Ames-Egerstedt 2017 eq. 9) defines a
relative-degree-1 barrier ``h_ij(p, v)`` so the constraint is linear in
``u``. The barrier::

    h_ij = sqrt(2 * alpha_pair * max(|Dp| - D_s, 0)) + (Dp^T / |Dp|) Dv

ensures the closing speed stays below the brakeable speed. The
relative-degree-1 derivative ``ḣ + gamma h + lambda >= 0`` becomes the
linear row in the CBF-QP, with the conformal slack ``lambda`` added on
the right (Huriot & Sibai 2025 §IV; ADR-004 §Decision).

This module composes :mod:`concerto.safety.solvers` and
:mod:`concerto.safety.budget_split`; it does not import from the
``chamber.*`` benchmark side (plan/10 §2 dependency rule). The
``obs["agent_states"]`` consumer contract is the integration boundary
the env wrapper supplies — see :class:`AgentSnapshot`. The
``obs["meta"]["partner_id"]`` consumer side is documented on
:class:`SafetyFilter`; the producer side lands in M4.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from concerto.safety.errors import ConcertoSafetyInfeasible
from concerto.safety.solvers import ClarabelSolver

if TYPE_CHECKING:
    from concerto.safety.api import Bounds, FilterInfo, FloatArray, SafetyState
    from concerto.safety.solvers import QPSolver

#: Default class-K function gain ``gamma`` for ``ḣ + gamma h >= 0``
#: (ADR-004 §Decision; Wang-Ames-Egerstedt 2017 §III).
DEFAULT_CBF_GAMMA: float = 5.0

#: Numerical floor for ``sqrt(2 alpha (|Dp| - D_s))`` so the gradient
#: ``alpha / psi`` does not blow up when agents are at the safety
#: boundary (ADR-004 §Decision).
_PSI_FLOOR: float = 1e-3

#: Numerical floor for ``|Dp|`` so the unit normal ``Dp / |Dp|`` is
#: well-defined on near-coincident centres (ADR-004 §Decision).
_NORM_FLOOR: float = 1e-9

#: Conservative upper bound on the per-pair RHS when a pair is in
#: penetration (``|Dp| < D_s``); the QP constraint forces braking
#: deceleration in this regime (ADR-004 risk-mitigation #1 guides the
#: hard fallback that bypasses the QP for the truly catastrophic case).
_PENETRATION_RHS: float = -1e3


@dataclass(frozen=True)
class AgentSnapshot:
    """Per-agent kinematic state at one control step (ADR-004 §Decision).

    The integration boundary the env wrapper supplies under
    ``obs["agent_states"][uid]``. Position and velocity must share the
    same Cartesian dimension (typically 2 for the §V toy crossing,
    3 for manipulation tasks).

    Attributes:
        position: Cartesian centre, shape ``(d,)``, dtype ``float64``.
        velocity: Cartesian velocity, shape ``(d,)``, dtype ``float64``.
        radius: Collision-sphere radius for the pairwise safety distance
            ``D_s = radius_i + radius_j``.
    """

    position: FloatArray
    velocity: FloatArray
    radius: float


def pair_h_value(
    snap_i: AgentSnapshot,
    snap_j: AgentSnapshot,
    *,
    alpha_pair: float,
) -> float:
    """Wang-Ames-Egerstedt 2017 §III pairwise barrier value ``h_ij`` (ADR-004 §Decision).

    Computes the relative-degree-1 safety barrier
    ``h_ij = sqrt(2 * alpha_pair * max(|Dp| - D_s, 0)) + (Dp^T / |Dp|) Dv``
    for one pair. Exposed (rather than kept under a leading underscore)
    so the conformal layer
    (:func:`concerto.safety.conformal.compute_prediction_gap_for_pairs`)
    can compute predicted vs. actual barrier values using the identical
    geometry the CBF-QP rows are built from. Intended for cross-module
    use within :mod:`concerto.safety`; external callers should reach for
    the higher-level conformal helpers instead.

    In the penetration regime (``|Dp| <= D_s``) the returned value is
    the negative-saturation surrogate ``-safety_distance`` (coincident-
    centre case) or ``closing_inner`` alone (within-radius case),
    matching the rows the QP would emit.
    """
    delta_p = snap_i.position - snap_j.position
    delta_v = snap_i.velocity - snap_j.velocity
    safety_distance = snap_i.radius + snap_j.radius
    norm_dp = float(np.linalg.norm(delta_p))

    if norm_dp < _NORM_FLOOR:
        return -safety_distance

    n_hat = (delta_p / norm_dp).astype(np.float64, copy=False)
    excess = norm_dp - safety_distance
    closing_inner = float(np.dot(n_hat, delta_v))
    if excess <= 0.0:
        return closing_inner
    psi = float(np.sqrt(2.0 * alpha_pair * excess))
    psi = max(psi, _PSI_FLOOR)
    return psi + closing_inner


def _pair_constraint_row(
    snap_i: AgentSnapshot,
    snap_j: AgentSnapshot,
    *,
    alpha_pair: float,
    gamma: float,
    lambda_ij: float,
    n_u_per_agent: int,
    slot_i: int,
    slot_j: int,
    n_total: int,
) -> tuple[FloatArray, float, float]:
    """Build one CBF-QP row + RHS for pair (i, j) (ADR-004 §Decision).

    Returns ``(row, rhs, h_ij)`` where ``row`` is the constraint vector
    (length ``n_total``), ``rhs`` is the right-hand-side scalar including
    the conformal slack ``lambda_ij``, and ``h_ij`` is the barrier value
    matching :func:`pair_h_value`.
    """
    delta_p = snap_i.position - snap_j.position
    delta_v = snap_i.velocity - snap_j.velocity
    safety_distance = snap_i.radius + snap_j.radius
    norm_dp = float(np.linalg.norm(delta_p))

    if norm_dp < _NORM_FLOOR:
        # Coincident centres — fall back to a stable normal so the QP
        # still produces a deterministic row; the braking fallback
        # (ADR-004 risk-mitigation #1) is the intended recovery.
        n_hat: FloatArray = np.zeros(n_u_per_agent, dtype=np.float64)
        n_hat[0] = 1.0
        row: FloatArray = np.zeros(n_total, dtype=np.float64)
        row[slot_i : slot_i + n_u_per_agent] = -n_hat
        row[slot_j : slot_j + n_u_per_agent] = n_hat
        return row, _PENETRATION_RHS + lambda_ij, -safety_distance

    n_hat = (delta_p / norm_dp).astype(np.float64, copy=False)
    excess = norm_dp - safety_distance
    if excess <= 0.0:
        # Penetration — most-restrictive RHS to force braking; the
        # caller's braking fallback (PR7) will bypass the QP if min
        # time-to-collision is below tau_brake.
        psi = _PSI_FLOOR
        excess_eff = 0.0
    else:
        psi = float(np.sqrt(2.0 * alpha_pair * excess))
        psi = max(psi, _PSI_FLOOR)
        excess_eff = excess

    closing_inner = float(np.dot(n_hat, delta_v))  # = Dp^T Dv / |Dp|
    h_ij = psi + closing_inner if excess_eff > 0.0 else closing_inner

    # ḣ has two state-dependent terms (drift) and one control-dependent
    # term (input). The drift sums into the RHS; the input term forms
    # the row.
    drift_psi = (alpha_pair / psi) * closing_inner if excess_eff > 0.0 else 0.0
    perp_v_sq = float(np.dot(delta_v, delta_v) - closing_inner * closing_inner)
    drift_perp = perp_v_sq / max(norm_dp, _NORM_FLOOR)
    drift = drift_psi + drift_perp

    row = np.zeros(n_total, dtype=np.float64)
    row[slot_i : slot_i + n_u_per_agent] = -n_hat
    row[slot_j : slot_j + n_u_per_agent] = n_hat

    rhs = drift + gamma * h_ij + lambda_ij
    return row, rhs, h_ij


class ExpCBFQP:
    """Exp CBF-QP filter (Wang-Ames-Egerstedt 2017 §III; ADR-004 §Decision).

    Implements the integrated outer-layer safety filter: builds pairwise
    CBF rows from :class:`AgentSnapshot` data in
    ``obs["agent_states"]``, stacks them with per-agent L-infinity action
    bounds, and projects ``proposed_action`` onto the feasible set via
    :class:`concerto.safety.solvers.QPSolver`. The conformal slack
    ``state.lambda_`` is added per-pair to the right-hand side
    (Huriot & Sibai 2025 §IV; ADR-004 §Decision).

    This module computes the per-step CBF gap and emits it as
    :class:`FilterInfo` ``"constraint_violation"`` (per pair,
    ``max(0, -h_ij)``). It does NOT compute the Huriot & Sibai §IV.A
    prediction-gap loss that drives the conformal update — that signal
    requires a partner-trajectory predictor and lives in
    ``concerto.safety.conformal`` (see
    :func:`concerto.safety.conformal.update_lambda_from_predictor` and
    :func:`concerto.safety.conformal.compute_prediction_gap_for_pairs`).
    ``FilterInfo["prediction_gap_loss"]`` is left :data:`None` here;
    callers wiring a predictor populate it via the conformal helpers.
    """

    def __init__(
        self,
        *,
        solver: QPSolver | None = None,
        cbf_gamma: float = DEFAULT_CBF_GAMMA,
    ) -> None:
        """Construct an exp CBF-QP filter (ADR-004 §Decision).

        Args:
            solver: QP solver strategy (default :class:`ClarabelSolver`).
            cbf_gamma: Class-K function gain in ``hdot + gamma h >= 0``;
                larger values tighten the constraint (default
                :data:`DEFAULT_CBF_GAMMA`).

        The per-pair ``budget_split`` strategy from
        :mod:`concerto.safety.budget_split` is **not** used by this
        centralized QP — Wang-Ames-Egerstedt's pair constraint involves
        both ``u_i`` and ``u_j`` jointly, so no per-agent decomposition
        is needed. The split is consumed by the OSCBF inner filter (PR8)
        for per-link decentralised within-arm constraints.
        """
        self._solver: QPSolver = solver if solver is not None else ClarabelSolver()
        self._cbf_gamma: float = cbf_gamma

    def reset(self, *, seed: int | None = None) -> None:
        """Reset filter state at episode start (ADR-004 §Decision).

        The exp CBF-QP is stateless across episodes — no internal RNG to
        re-seed. The conformal layer (PR6) is the stateful component;
        it is reset via :class:`SafetyState` directly.
        """
        del seed

    def filter(
        self,
        proposed_action: dict[str, FloatArray],
        obs: dict[str, object],
        state: SafetyState,
        bounds: Bounds,
    ) -> tuple[dict[str, FloatArray], FilterInfo]:
        """Project ``proposed_action`` onto the CBF-safe set (ADR-004 §Decision).

        Args:
            proposed_action: Per-agent nominal accelerations, keyed by
                uid; uid order is preserved in the QP variable layout.
            obs: Observation dict carrying ``obs["agent_states"]: dict[uid,
                AgentSnapshot]``. ``obs["meta"]["partner_id"]`` (M4
                contract) is consumed by the conformal layer (PR6); this
                module is partner-id-agnostic.
            state: Mutable :class:`SafetyState`; ``state.lambda_`` is
                read per-pair on the constraint RHS but not mutated here
                — the conformal update happens in PR6.
            bounds: Per-task :class:`Bounds`; ``bounds.action_norm``
                drives the per-agent L-infinity action bound.

        Returns:
            ``(safe_action, info)`` — the QP-projected per-agent actions
            and a :class:`FilterInfo` payload. ``"lambda"`` carries the
            current slack vector. ``"constraint_violation"`` carries the
            per-pair non-negative CBF gap ``max(0, -h_ij)`` (zero when
            the barrier holds; positive when it does not).
            ``"prediction_gap_loss"`` is :data:`None` here — the
            conformal layer (PR6) populates it from a predictor.
            ``"fallback_fired"`` is False here — PR7 wires the braking
            fallback. ``"qp_solve_ms"`` reports the solver time.

        Raises:
            ConcertoSafetyInfeasible: When the QP is infeasible. The
                caller MUST route to the braking fallback (ADR-004
                risk-mitigation #1).
        """
        snaps_obj = obs.get("agent_states")
        if not isinstance(snaps_obj, dict):
            msg = (
                "obs['agent_states'] must be dict[uid, AgentSnapshot]; "
                f"got {type(snaps_obj).__name__}"
            )
            raise ValueError(msg)
        uids = list(proposed_action.keys())
        snaps: dict[str, AgentSnapshot] = {uid: snaps_obj[uid] for uid in uids}

        # Per-agent control dimension (assumed homogeneous for Phase-0).
        n_u_per_agent = int(proposed_action[uids[0]].shape[0])
        n_total = n_u_per_agent * len(uids)
        slot_of = {uid: i * n_u_per_agent for i, uid in enumerate(uids)}

        u_hat = np.concatenate([proposed_action[uid] for uid in uids]).astype(
            np.float64, copy=False
        )

        # Pairwise CBF rows.
        n_pairs = (len(uids) * (len(uids) - 1)) // 2
        if state.lambda_.shape != (n_pairs,):
            msg = (
                f"state.lambda_ shape mismatch: expected ({n_pairs},), "
                f"got {state.lambda_.shape}. Reset SafetyState on partner-set "
                "change per ADR-004 risk-mitigation #2."
            )
            raise ValueError(msg)

        rows: list[FloatArray] = []
        rhs_list: list[float] = []
        constraint_violation = np.zeros(n_pairs, dtype=np.float64)
        pair_idx = 0
        # alpha_pair is the joint braking capacity (alpha_i + alpha_j) per
        # Wang-Ames-Egerstedt 2017 §III; symmetric agents share action_norm.
        alpha_pair = 2.0 * bounds.action_norm
        for a, uid_i in enumerate(uids):
            for uid_j in uids[a + 1 :]:
                row, rhs, h_ij = _pair_constraint_row(
                    snaps[uid_i],
                    snaps[uid_j],
                    alpha_pair=alpha_pair,
                    gamma=self._cbf_gamma,
                    lambda_ij=float(state.lambda_[pair_idx]),
                    n_u_per_agent=n_u_per_agent,
                    slot_i=slot_of[uid_i],
                    slot_j=slot_of[uid_j],
                    n_total=n_total,
                )
                rows.append(row)
                rhs_list.append(rhs)
                # Per-step CBF gap: max(0, -h_ij). This is the
                # constraint-violation signal (zero when h >= 0), NOT
                # the Huriot-Sibai §IV.A prediction-gap loss that drives
                # the conformal update — see module docstring.
                constraint_violation[pair_idx] = max(0.0, -h_ij)
                pair_idx += 1

        # Per-agent L-infinity bounds on u: -alpha <= u[k] <= alpha,
        # i.e., I u <= alpha and -I u <= alpha. Stack into 2 * n_total rows.
        identity = np.eye(n_total, dtype=np.float64)
        bound_rows = np.vstack([identity, -identity])
        bound_rhs = np.full(2 * n_total, bounds.action_norm, dtype=np.float64)

        if rows:
            cbf_rows = np.vstack(rows)
            cbf_rhs = np.asarray(rhs_list, dtype=np.float64)
            constraint_a = np.vstack([cbf_rows, bound_rows])
            constraint_b = np.concatenate([cbf_rhs, bound_rhs])
        else:
            constraint_a = bound_rows
            constraint_b = bound_rhs

        # min ||u - u_hat||^2 = min 1/2 u^T (2 I) u + (-2 u_hat)^T u + const.
        # The constant drops; QP form uses 1/2 x^T P x + q^T x.
        cost_p = 2.0 * np.eye(n_total, dtype=np.float64)
        cost_q = -2.0 * u_hat

        start = time.perf_counter()
        try:
            x_safe, _ = self._solver.solve(cost_p, cost_q, constraint_a, constraint_b)
        except ConcertoSafetyInfeasible:
            raise
        qp_solve_ms = (time.perf_counter() - start) * 1000.0

        safe_action: dict[str, FloatArray] = {}
        for uid in uids:
            slot = slot_of[uid]
            safe_action[uid] = x_safe[slot : slot + n_u_per_agent].astype(np.float64, copy=True)

        info: FilterInfo = {
            "lambda": state.lambda_.copy(),
            "constraint_violation": constraint_violation,
            "prediction_gap_loss": None,
            "fallback_fired": False,
            "qp_solve_ms": qp_solve_ms,
        }
        return safe_action, info


__all__ = [
    "DEFAULT_CBF_GAMMA",
    "AgentSnapshot",
    "ExpCBFQP",
    "pair_h_value",
]
