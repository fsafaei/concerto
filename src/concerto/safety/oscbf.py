# SPDX-License-Identifier: Apache-2.0
"""Operational-Space CBF two-level QP (Morton-Pavone 2025 §IV; ADR-004 §Decision).

ADR-004 §Decision pins the OSCBF inner filter as Morton & Pavone 2025
§IV: a two-level QP that projects nominal joint-velocity and
operational-space (end-effector) commands onto the safe set defined by
linear CBF rows (collision avoidance, joint-velocity limits, custom
within-arm constraints).

The QP::

    min   || W_j (q_dot - q_dot_nom) ||^2
        + || W_o (J q_dot - nu_nom) ||^2
        + rho * || s ||^2
    s.t.   A q_dot - I s <= b      (CBF rows; slack absorbs infeasibility)
           -s <= 0                  (slack non-negative)

The slack variable ``s`` follows Morton-Pavone §IV.D's relaxation
pattern: the QP stays feasible even when CBF rows conflict, with
``rho >> 1`` keeping ``s`` near zero in the unconflicted case. The
optimisation is solved through :class:`concerto.safety.solvers.QPSolver`
(Clarabel default per ADR-004 §Decision).

Re-implementation per plan/03 §8 / M3 brief Hard Rule #1: the public
Morton-Pavone code is **not** vendored — the QP is re-derived from the
manuscript + reading note 44.

Plan/03 §3.4 also mentions cvxpy for symbolic problem definition. This
Phase-0 implementation builds raw ``P``/``q``/``A``/``b`` matrices to
hit the 1 kHz solve-time target on 7-DOF (Morton-Pavone §V; ADR-004
validation criterion 3). cvxpy is a Phase-1 reformulation candidate if
symbolic flexibility matters more than speed.

The kinematic Jacobian ``J`` (6 x n_joints) and the per-sphere
geometry Jacobians (3 x n_joints) are supplied by the env-side robot
model; this module is JS+OS-symbolic but kinematics-agnostic. See
:func:`collision_constraint_row` and :func:`joint_limit_constraint_row`
for the row builders that turn high-level safety specs into linear
``a_i^T q_dot <= b_i`` rows.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from concerto.safety.solvers import ClarabelSolver

if TYPE_CHECKING:
    from concerto.safety.api import FloatArray
    from concerto.safety.solvers import QPSolver

#: Default penalty weight on the slack variable ``rho`` (Morton-Pavone 2025 §IV.D).
#: Large enough to keep slack negligible in the unconflicted case;
#: small enough that conflicting CBFs don't make the QP ill-conditioned.
DEFAULT_SLACK_PENALTY: float = 100.0

#: Default joint-space cost weight ``W_j`` (scalar applied as ``W_j * I``).
DEFAULT_W_JOINT: float = 1.0

#: Default operational-space cost weight ``W_o``.
DEFAULT_W_OPERATIONAL: float = 1.0

#: Default class-K function gain for ``hdot + alpha * h >= 0`` (ADR-004 §Decision).
DEFAULT_CBF_ALPHA: float = 5.0

#: Numerical floor for ``|c_a - c_b|`` in :func:`collision_constraint_row`
#: (mirrors the geometry module's coincident-centre handling).
_COINCIDENT_TOL: float = 1e-9

#: Numerical floor for declaring a row "slack-active" in :class:`OSCBFResult`
#: (ADR-014 Table 2; external-review P0-3, 2026-05-16). Slack values
#: below this threshold are treated as interior-point round-off and
#: excluded from the ``active_rows`` enumeration so the telemetry
#: signal is not dominated by solver noise.
#:
#: Calibration: Clarabel's observed feasible-interior round-off on the
#: OSCBF-shape QPs in ``tests/unit/test_oscbf_returns_slack.py`` is
#: ~5e-6; OSQP under :data:`concerto.safety.solvers._OSQP_TIGHT_SETTINGS`
#: (``eps_abs = eps_rel = 1e-9``, polishing on) converges to a similar
#: order of magnitude. The 1e-4 floor sits one to two orders of
#: magnitude above this round-off ceiling and the same value works
#: for both solvers; a real Phase-1 slack-active row will be orders of
#: magnitude above the floor (the QP penalty cost grows quadratically
#: in slack so meaningful relaxation is loud).
_SLACK_ACTIVE_FLOOR: float = 1e-4

#: Threshold for surfacing a "raw slack went negative" warning before
#: the non-negativity clip in :meth:`OSCBF.solve`. The QP's ``-s <= 0``
#: row guarantees slack >= 0 mathematically; tiny negative excursions
#: are interior-point round-off below the constraint surface and are
#: clipped silently to zero. A negative excursion *below* this floor
#: indicates a solver bug or numerical pathology and is logged so the
#: clip does not silently mask it (review P0-3 reviewer follow-up).
_NEGATIVE_SLACK_WARN_FLOOR: float = -1e-6


@dataclass(frozen=True)
class OSCBFResult:
    """OSCBF QP-solve telemetry payload (ADR-004 §Decision; ADR-014 Table 2).

    Returned by :meth:`OSCBF.solve`. Carries the QP-projected
    joint-velocity command together with the per-row slack vector and
    aggregate slack statistics, so the ADR-014 three-table report can
    distinguish constraint-*satisfaction* from constraint-*relaxation
    via slack* (external-review P0-3, 2026-05-16). The pre-amendment
    return type ``(q_dot, solve_ms)`` silently dropped the slack
    vector; a controller that "succeeds" only via large slack is not
    safe in the intended sense, and aggregating slack at the Table 2
    row was previously impossible.

    Attributes:
        q_dot: QP-projected joint velocity, shape ``(n_joints,)``,
            dtype ``float64``.
        slack: Per-row slack values from the QP, shape ``(m,)`` where
            ``m`` is the number of CBF rows in
            :class:`OSCBFConstraints`. Non-negative by construction
            (the slack-non-negativity row is part of the QP).
        solve_ms: Wall-clock QP solve time in milliseconds.
        active_rows: Indices of constraint rows whose slack exceeds
            :data:`_SLACK_ACTIVE_FLOOR` (i.e., the QP relaxed the row
            rather than satisfying it). Empty when every row holds
            within numerical tolerance.
        max_slack: ``max(slack)``, in the slack variable's units.
        slack_l2: ``||slack||_2``, in the slack variable's units.
        solver_status: ``"optimal"`` on successful solve. Reserved for
            future non-exception-based solver variants; the current
            Clarabel / OSQP path raises
            :class:`concerto.safety.errors.ConcertoSafetyInfeasible`
            instead of returning a non-optimal status.
    """

    q_dot: FloatArray
    slack: FloatArray
    solve_ms: float
    active_rows: tuple[int, ...]
    max_slack: float
    slack_l2: float
    solver_status: str


@dataclass(frozen=True)
class OSCBFConstraints:
    """Stacked linear CBF rows for the OSCBF QP (ADR-004 §Decision).

    Each row encodes ``a_i^T q_dot <= b_i + s_i`` where ``s_i`` is the
    per-row slack variable. Callers stack joint-velocity-limit rows
    (:func:`joint_limit_constraint_row`), sphere-pair collision rows
    (:func:`collision_constraint_row`), and custom rows as needed.

    Attributes:
        a: Constraint matrix, shape ``(m, n_joints)``, dtype ``float64``.
        b: Constraint bound, shape ``(m,)``, dtype ``float64``.
    """

    a: FloatArray
    b: FloatArray

    def __post_init__(self) -> None:
        """Validate row shapes (ADR-004 §Decision)."""
        expected_a_dim = 2
        expected_b_dim = 1
        if self.a.ndim != expected_a_dim:
            msg = f"a must be 2-D, got shape {self.a.shape}"
            raise ValueError(msg)
        if self.b.ndim != expected_b_dim:
            msg = f"b must be 1-D, got shape {self.b.shape}"
            raise ValueError(msg)
        if self.a.shape[0] != self.b.shape[0]:
            msg = (
                f"row count mismatch: a has {self.a.shape[0]} rows, b has {self.b.shape[0]} entries"
            )
            raise ValueError(msg)


def joint_limit_constraint_row(
    *,
    n_joints: int,
    joint_index: int,
    upper: float | None = None,
    lower: float | None = None,
) -> tuple[FloatArray, float]:
    """Build a single-joint velocity-limit CBF row (ADR-004 §Decision).

    Specify exactly one of ``upper`` or ``lower``:

    - ``upper``: yields row ``e_{joint_index}^T``, RHS ``upper`` so the
      stacked constraint reads ``q_dot[joint_index] <= upper``.
    - ``lower``: yields row ``-e_{joint_index}^T``, RHS ``-lower`` so
      the stacked constraint reads ``q_dot[joint_index] >= lower``.

    Returns one row at a time so the caller can mix per-joint bounds
    with per-pair sphere-collision rows.

    Args:
        n_joints: Total number of joints (sets the row length).
        joint_index: Index of the joint this row constrains, in
            ``[0, n_joints)``.
        upper: Upper-bound value (mutually exclusive with ``lower``).
        lower: Lower-bound value (mutually exclusive with ``upper``).

    Returns:
        ``(row, rhs)`` ready to stack into :class:`OSCBFConstraints`.

    Raises:
        ValueError: If neither or both of ``upper``/``lower`` are
            given, or ``joint_index`` is out of range.
    """
    if not 0 <= joint_index < n_joints:
        msg = f"joint_index {joint_index} not in [0, {n_joints})"
        raise ValueError(msg)
    if (upper is None) == (lower is None):
        msg = "Specify exactly one of upper or lower"
        raise ValueError(msg)
    row: FloatArray = np.zeros(n_joints, dtype=np.float64)
    if upper is not None:
        row[joint_index] = 1.0
        return row, float(upper)
    assert lower is not None  # noqa: S101 (mypy/pyright narrowing)
    row[joint_index] = -1.0
    return row, -float(lower)


def collision_constraint_row(
    *,
    n_joints: int,
    sphere_a_center: FloatArray,
    sphere_b_center: FloatArray,
    sphere_a_radius: float,
    sphere_b_radius: float,
    sphere_a_jacobian: FloatArray,
    sphere_b_jacobian: FloatArray | None = None,
    cbf_alpha: float = DEFAULT_CBF_ALPHA,
    lambda_ij: float = 0.0,
) -> tuple[FloatArray, float]:
    """Build a sphere-pair collision-avoidance CBF row (ADR-004 §Decision).

    Barrier: ``h = ||c_a - c_b|| - (r_a + r_b) >= 0``. With sphere
    centres tracking joint configuration via Jacobians ``J_a``, ``J_b``::

        h_dot = (n_hat^T (J_a - J_b)) q_dot
        constraint: h_dot + alpha * h + lambda_ij >= 0
                 => -(n_hat^T (J_a - J_b)) q_dot <= alpha * h + lambda_ij

    where ``n_hat = (c_a - c_b) / ||c_a - c_b||``. The ``lambda_ij``
    term carries the conformal slack from the outer layer
    (Huriot & Sibai 2025 §IV); pass ``0.0`` for unconformal rows.

    On near-coincident centres the row falls back to zeros and the RHS
    forces the slack to absorb (the QP path stays feasible; the
    braking fallback (PR7) is the per-step backstop for catastrophic
    states the QP cannot recover from on its own).

    Args:
        n_joints: Number of joint variables.
        sphere_a_center: Centre of sphere A, shape ``(3,)``.
        sphere_b_center: Centre of sphere B, shape ``(3,)``.
        sphere_a_radius: Radius of sphere A.
        sphere_b_radius: Radius of sphere B.
        sphere_a_jacobian: ``d c_a / d q``, shape ``(3, n_joints)``.
        sphere_b_jacobian: ``d c_b / d q``, shape ``(3, n_joints)``;
            ``None`` for static obstacles.
        cbf_alpha: Class-K gain in ``hdot + alpha * h >= 0`` (default
            :data:`DEFAULT_CBF_ALPHA`).
        lambda_ij: Conformal slack on the constraint RHS (Huriot & Sibai
            2025 §IV; default 0.0).

    Returns:
        ``(row, rhs)`` ready to stack into :class:`OSCBFConstraints`.
    """
    delta = sphere_a_center - sphere_b_center
    norm_dp = float(np.linalg.norm(delta))
    if norm_dp < _COINCIDENT_TOL:
        row: FloatArray = np.zeros(n_joints, dtype=np.float64)
        return row, lambda_ij - 1.0

    n_hat = delta / norm_dp
    h = norm_dp - (sphere_a_radius + sphere_b_radius)

    if sphere_b_jacobian is None:
        j_diff = sphere_a_jacobian
    else:
        j_diff = sphere_a_jacobian - sphere_b_jacobian

    row = (-(n_hat @ j_diff)).astype(np.float64, copy=False)
    rhs = cbf_alpha * h + lambda_ij
    return row, rhs


class OSCBF:
    """Operational-Space CBF two-level QP (Morton-Pavone 2025 §IV; ADR-004 §Decision).

    Projects nominal joint-velocity ``q_dot_nom`` and operational-space
    velocity ``nu_nom`` onto the safe set defined by linear CBF rows in
    :class:`OSCBFConstraints`. Slack relaxation keeps the QP feasible
    even when CBFs conflict (Morton-Pavone §IV.D); ``slack_penalty``
    >> 1 keeps slack negligible otherwise.
    """

    def __init__(
        self,
        *,
        n_joints: int,
        weight_joint: float = DEFAULT_W_JOINT,
        weight_operational: float = DEFAULT_W_OPERATIONAL,
        slack_penalty: float = DEFAULT_SLACK_PENALTY,
        solver: QPSolver | None = None,
    ) -> None:
        """Construct an OSCBF instance (ADR-004 §Decision).

        Args:
            n_joints: Number of joint-space variables (typically 7 for
                the Franka 7-DOF arm; ADR-004 §"OSCBF target").
            weight_joint: Joint-space cost weight ``W_j`` applied as
                ``W_j * I`` (scalar default; per-joint matrix can be
                wired in a future extension).
            weight_operational: Operational-space cost weight ``W_o``
                applied as ``W_o * I``.
            slack_penalty: Penalty ``rho`` on ``|| s ||^2`` in the
                cost (Morton-Pavone §IV.D; default
                :data:`DEFAULT_SLACK_PENALTY` = 100.0).
            solver: QP solver strategy; default is the ADR-004 Phase-0
                Clarabel interior-point solver from PR2.
        """
        if n_joints < 1:
            msg = f"n_joints must be >= 1, got {n_joints}"
            raise ValueError(msg)
        self._n_joints: int = n_joints
        self._w_j: float = weight_joint
        self._w_o: float = weight_operational
        self._rho: float = slack_penalty
        self._solver: QPSolver = solver if solver is not None else ClarabelSolver()

    @property
    def n_joints(self) -> int:
        """Number of joint-space variables (ADR-004 §Decision)."""
        return self._n_joints

    def solve(
        self,
        *,
        q_dot_nom: FloatArray,
        nu_nom: FloatArray,
        jacobian: FloatArray,
        constraints: OSCBFConstraints,
    ) -> OSCBFResult:
        """Solve the two-level QP, return an :class:`OSCBFResult` (ADR-004 §Decision).

        Variable layout: ``x = [q_dot; s]`` where ``s`` is the per-row
        slack vector of length ``m = constraints.a.shape[0]``. Cost::

            P_block = block_diag(2 * (W_j^2 I + W_o^2 J^T J),
                                 2 * rho * I_m)
            q_block = [-2 (W_j^2 q_dot_nom + W_o^2 J^T nu_nom); 0]

        Constraints::

            [A - I_m][q_dot] <= [b]
            [0 - I_m][s] <= [0]

        Args:
            q_dot_nom: Nominal joint-velocity command, shape ``(n_joints,)``.
            nu_nom: Nominal operational-space velocity (``[v; omega]``
                for a 6-DoF end-effector), shape ``(6,)``.
            jacobian: Operational-space Jacobian, shape
                ``(6, n_joints)``.
            constraints: Stacked CBF rows.

        Returns:
            An :class:`OSCBFResult` carrying the QP-projected
            joint-velocity, the per-row slack vector with aggregate
            statistics (``max_slack``, ``slack_l2``), the indices of
            slack-active rows, the wall-clock solve time, and the
            solver status. The slack telemetry is the diagnostic
            signal ADR-014 Table 2 aggregates to distinguish
            constraint-*satisfaction* from constraint-*relaxation*
            (external-review P0-3, 2026-05-16; the pre-amendment
            ``(q_dot, solve_ms)`` return silently dropped the slack
            vector).

        Raises:
            ValueError: If input shapes are inconsistent.
            concerto.safety.errors.ConcertoSafetyInfeasible: When the QP
                is infeasible even with slack — should not happen with
                positive ``slack_penalty`` and finite RHS, so it
                indicates a malformed input. The braking fallback
                (PR7) is the per-step recovery (ADR-004
                risk-mitigation #1).
        """
        n = self._n_joints
        if q_dot_nom.shape != (n,):
            msg = f"q_dot_nom shape {q_dot_nom.shape} != ({n},)"
            raise ValueError(msg)
        if jacobian.shape[1] != n:
            msg = f"jacobian must have {n} columns; got shape {jacobian.shape}"
            raise ValueError(msg)
        if nu_nom.shape != (jacobian.shape[0],):
            msg = f"nu_nom shape {nu_nom.shape} must match jacobian rows {jacobian.shape[0]}"
            raise ValueError(msg)
        if constraints.a.shape[1] != n:
            msg = f"constraints.a column count {constraints.a.shape[1]} != n_joints {n}"
            raise ValueError(msg)

        m = constraints.a.shape[0]
        wj_sq = self._w_j * self._w_j
        wo_sq = self._w_o * self._w_o

        # Cost block for q_dot: 2 (W_j^2 I + W_o^2 J^T J).
        p_q = wj_sq * np.eye(n, dtype=np.float64) + wo_sq * (jacobian.T @ jacobian)
        p_full: FloatArray = np.zeros((n + m, n + m), dtype=np.float64)
        p_full[:n, :n] = 2.0 * p_q
        p_full[n:, n:] = 2.0 * self._rho * np.eye(m, dtype=np.float64)

        # Linear-cost block.
        q_q = -2.0 * (wj_sq * q_dot_nom + wo_sq * (jacobian.T @ nu_nom))
        q_full: FloatArray = np.concatenate([q_q, np.zeros(m, dtype=np.float64)]).astype(
            np.float64, copy=False
        )

        # Constraints: [A -I; 0 -I] [q_dot; s] <= [b; 0].
        eye_m = np.eye(m, dtype=np.float64)
        a_top = np.hstack([constraints.a, -eye_m])
        a_bot = np.hstack([np.zeros((m, n), dtype=np.float64), -eye_m])
        a_full: FloatArray = np.vstack([a_top, a_bot]).astype(np.float64, copy=False)
        b_full: FloatArray = np.concatenate([constraints.b, np.zeros(m, dtype=np.float64)]).astype(
            np.float64, copy=False
        )

        start = time.perf_counter()
        x, _ = self._solver.solve(p_full, q_full, a_full, b_full)
        solve_ms = (time.perf_counter() - start) * 1000.0
        q_dot: FloatArray = x[:n].astype(np.float64, copy=False)
        # Slack is at x[n:n+m]; clip tiny negatives that can occur as
        # solver round-off below the ``-s <= 0`` constraint surface so
        # ``max_slack`` and ``slack_l2`` are always non-negative. A
        # negative excursion *below* :data:`_NEGATIVE_SLACK_WARN_FLOOR`
        # is too large to be round-off and indicates a solver bug; we
        # log it via :class:`UserWarning` so the silent clip doesn't
        # mask it (review P0-3 reviewer follow-up).
        slack_raw: FloatArray = x[n : n + m].astype(np.float64, copy=False)
        if m > 0:
            min_raw = float(slack_raw.min())
            if min_raw < _NEGATIVE_SLACK_WARN_FLOOR:
                warnings.warn(
                    "OSCBF: raw slack went below the round-off floor "
                    f"(min={min_raw!r} < {_NEGATIVE_SLACK_WARN_FLOOR!r}); "
                    "clipping to zero. This indicates the QP solver "
                    "returned an infeasible iterate — investigate before "
                    "trusting the OSCBFResult.slack telemetry.",
                    UserWarning,
                    stacklevel=2,
                )
        slack: FloatArray = np.maximum(slack_raw, 0.0)
        active_rows: tuple[int, ...] = tuple(
            int(i) for i in np.flatnonzero(slack > _SLACK_ACTIVE_FLOOR).tolist()
        )
        max_slack: float = float(slack.max()) if m > 0 else 0.0
        slack_l2: float = float(np.linalg.norm(slack)) if m > 0 else 0.0
        return OSCBFResult(
            q_dot=q_dot,
            slack=slack,
            solve_ms=solve_ms,
            active_rows=active_rows,
            max_slack=max_slack,
            slack_l2=slack_l2,
            solver_status="optimal",
        )


__all__ = [
    "DEFAULT_CBF_ALPHA",
    "DEFAULT_SLACK_PENALTY",
    "DEFAULT_W_JOINT",
    "DEFAULT_W_OPERATIONAL",
    "OSCBF",
    "OSCBFConstraints",
    "OSCBFResult",
    "collision_constraint_row",
    "joint_limit_constraint_row",
]
