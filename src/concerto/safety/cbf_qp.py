# SPDX-License-Identifier: Apache-2.0
"""Exponential CBF-QP backbone (Wang-Ames-Egerstedt 2017 §III; ADR-004 §Decision).

Pairwise multi-mode CBF-QP filter::

    min_{u}  || u - u_hat ||^2
    s.t.     A_pair u <= b_pair + lambda_pair    (per pair (i, j))
             -alpha_i_action <= u_i[k] <= alpha_i_action  (per agent slot)

The pairwise CBF row is built in **Cartesian / safety space** from a
Wang-Ames-Egerstedt 2017 §III relative-degree-1 barrier::

    h_ij = sqrt(2 * alpha_pair * max(|Dp| - D_s, 0)) + (Dp^T / |Dp|) Dv

and then projected to per-agent action-space coefficients via each
agent's :class:`concerto.safety.api.AgentControlModel`. For an
exactly-actuated double-integrator agent the projection is the
identity (the Jacobian is ``I``); for a 7-DOF arm the projection
uses the Jacobian-aware control model. This is the
heterogeneity-aware refactor that spike_004A specifies — replacing
the prior "first agent's shape" homogeneous assumption with explicit
per-uid control maps.

The QP variable layout and the partner role depend on the
:class:`concerto.safety.api.SafetyMode` selected at construction:

- :attr:`SafetyMode.EGO_ONLY` (default): one decision variable of
  length ``ego.action_dim``; the partner's predicted Cartesian
  acceleration enters the constraint RHS as a known drift term.
  This is the deployment / ad-hoc / black-box-partner mode (ADR-006
  §Decision; reviewer P0-3).
- :attr:`SafetyMode.CENTRALIZED`: concatenated per-uid action
  vectors with slot widths from
  ``control_models[uid].action_dim``. The oracle / ablation
  baseline (ADR-014 three-table report); never used at evaluation
  against ad-hoc partners.
- :attr:`SafetyMode.SHARED_CONTROL`: same shape as
  :attr:`SafetyMode.CENTRALIZED` with an additional per-call
  ``partner_action_bound`` that constrains the partner's adjustment
  magnitude (reviewer P0-3 mode 3; lab-only baseline).

This module composes :mod:`concerto.safety.solvers` and
:mod:`concerto.safety.budget_split`; it does not import from the
``chamber.*`` benchmark side (plan/10 §2 dependency rule).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast, overload

import numpy as np

from concerto.safety.api import SafetyMode
from concerto.safety.budget_split import ProportionalBudgetSplit
from concerto.safety.solvers import ClarabelSolver

if TYPE_CHECKING:
    from collections.abc import Mapping

    from concerto.safety.api import (
        AgentControlModel,
        Bounds,
        EgoOnlySafetyFilter,
        FilterInfo,
        FloatArray,
        JointSafetyFilter,
        SafetyState,
    )
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


@dataclass(frozen=True)
class _CartesianPairRow:
    """One pair's CBF row built in Cartesian / safety space (ADR-004 §Decision; spike_004A).

    Internal helper. The row is ``n_hat^T (ddot p_i - ddot p_j) >=
    -drift - gamma * h_ij - lambda_ij``; downstream code projects
    ``ddot p`` into action space via the per-uid
    :class:`concerto.safety.api.AgentControlModel`.

    Attributes:
        n_hat: Cartesian unit normal from agent ``j`` to agent ``i``,
            shape ``(position_dim,)``.
        rhs: Right-hand side scalar ``drift + gamma * h_ij + lambda_ij``.
        h_ij: Barrier value matching :func:`pair_h_value`.
        partner_disturbance: For :attr:`SafetyMode.EGO_ONLY`, the
            Cartesian acceleration of the partner enters the RHS as a
            known drift term ``-n_hat^T ddot p_partner``. Zero for
            :attr:`SafetyMode.CENTRALIZED` / :attr:`SafetyMode.SHARED_CONTROL`
            where the partner's acceleration is itself a decision
            variable.
    """

    n_hat: FloatArray
    rhs: float
    h_ij: float
    partner_disturbance: float


def _build_cartesian_pair_row(
    snap_i: AgentSnapshot,
    snap_j: AgentSnapshot,
    *,
    alpha_pair: float,
    gamma: float,
    lambda_ij: float,
) -> _CartesianPairRow:
    """Build one pair's CBF row in Cartesian / safety space (ADR-004 §Decision; spike_004A).

    The row coefficients on the Cartesian acceleration variables
    ``ddot p_i``, ``ddot p_j`` are ``-n_hat`` and ``+n_hat``
    respectively. The RHS carries the drift terms from differentiating
    ``h_ij`` once plus the class-K and conformal-slack contributions.
    Projection to action-space slots happens at the caller, using each
    uid's :class:`AgentControlModel.cartesian_accel_to_action` map.
    """
    delta_p = snap_i.position - snap_j.position
    delta_v = snap_i.velocity - snap_j.velocity
    safety_distance = snap_i.radius + snap_j.radius
    norm_dp = float(np.linalg.norm(delta_p))
    position_dim = int(snap_i.position.shape[0])

    if norm_dp < _NORM_FLOOR:
        # Coincident centres — fall back to a stable normal so the QP
        # still produces a deterministic row; the braking fallback
        # (ADR-004 risk-mitigation #1) is the intended recovery.
        n_hat: FloatArray = np.zeros(position_dim, dtype=np.float64)
        n_hat[0] = 1.0
        return _CartesianPairRow(
            n_hat=n_hat,
            rhs=_PENETRATION_RHS + lambda_ij,
            h_ij=-safety_distance,
            partner_disturbance=0.0,
        )

    n_hat = (delta_p / norm_dp).astype(np.float64, copy=False)
    excess = norm_dp - safety_distance
    if excess <= 0.0:
        psi = _PSI_FLOOR
        excess_eff = 0.0
    else:
        psi = float(np.sqrt(2.0 * alpha_pair * excess))
        psi = max(psi, _PSI_FLOOR)
        excess_eff = excess

    closing_inner = float(np.dot(n_hat, delta_v))
    h_ij = psi + closing_inner if excess_eff > 0.0 else closing_inner

    drift_psi = (alpha_pair / psi) * closing_inner if excess_eff > 0.0 else 0.0
    perp_v_sq = float(np.dot(delta_v, delta_v) - closing_inner * closing_inner)
    drift_perp = perp_v_sq / max(norm_dp, _NORM_FLOOR)
    drift = drift_psi + drift_perp

    rhs = drift + gamma * h_ij + lambda_ij
    return _CartesianPairRow(
        n_hat=n_hat,
        rhs=rhs,
        h_ij=h_ij,
        partner_disturbance=0.0,
    )


# _internal: not part of the public surface — see ADR-004 §Risks #4.
def _build_ego_only_row(
    ego_snap: AgentSnapshot,
    partner_snap: AgentSnapshot,
    partner_predicted_accel: FloatArray,
    *,
    alpha_pair: float,
    gamma: float,
    lambda_ij: float,
) -> tuple[FloatArray, float, float]:
    """Assemble the Cartesian CBF row + RHS + ``h_ij`` for one EGO_ONLY pair (ADR-004 §Risks #4).

    Internal seam exposed for the analytic sign-convention test suite
    (``tests/unit/test_cbf_qp_ego_only_signs.py``). A sign error in any
    of three sensitive transformations — the ego-acceleration row sign,
    the partner-disturbance subtraction on the RHS, or the closing-
    velocity sign in ``h_ij`` — would silently invert the pairwise
    constraint and make the entire safety filter unsafe (reviewer
    P0-3). Tests pin each sign analytically against the closed-form
    row rather than running the QP solve, sidestepping solver
    tolerance noise.

    The returned ``row`` is the coefficient on the ego's *Cartesian*
    acceleration variable: ``-n_hat``. For
    :class:`~concerto.safety.api.DoubleIntegratorControlModel` (whose
    action space coincides with Cartesian acceleration) this is the
    action-space row the QP stacker uses; for Jacobian-mediated
    embodiments the caller composes the row with the ego's
    :class:`~concerto.safety.api.AgentControlModel` Jacobian.
    :meth:`ExpCBFQP._filter_ego_only` consumes this helper directly,
    so the row coefficients tested here are the same ones the QP sees.
    """
    pair_row = _build_cartesian_pair_row(
        snap_i=ego_snap,
        snap_j=partner_snap,
        alpha_pair=alpha_pair,
        gamma=gamma,
        lambda_ij=lambda_ij,
    )
    partner_disturbance = float(
        pair_row.n_hat @ partner_predicted_accel.astype(np.float64, copy=False)
    )
    row: FloatArray = (-pair_row.n_hat).astype(np.float64, copy=True)
    rhs = pair_row.rhs - partner_disturbance
    return row, rhs, pair_row.h_ij


def _project_cartesian_row_to_action(
    pair_row: _CartesianPairRow,
    *,
    snap_i: AgentSnapshot,
    snap_j: AgentSnapshot,
    model_i: AgentControlModel,
    model_j: AgentControlModel | None,
    n_total: int,
    slot_i: int,
    slot_j: int | None,
) -> FloatArray:
    """Project a Cartesian pair row to action-space coefficients (ADR-004 §Decision; spike_004A).

    The Cartesian row reads ``-n_hat^T ddot p_i + n_hat^T ddot p_j <=
    -RHS``; substituting ``ddot p = J(state) action`` for each agent
    yields the action-space row ``(-n_hat^T J_i) action_i + (n_hat^T
    J_j) action_j <= -RHS``. The Jacobian ``J_i`` is the linearisation
    of :meth:`AgentControlModel.action_to_cartesian_accel` at the
    current state; for :class:`DoubleIntegratorControlModel` it is the
    identity (the row coefficients are ``-n_hat`` / ``+n_hat``,
    matching the pre-refactor backbone byte-for-byte).

    Args:
        pair_row: Output of :func:`_build_cartesian_pair_row`.
        snap_i: Current snapshot of agent ``i``.
        snap_j: Current snapshot of agent ``j``.
        model_i: Control model for agent ``i``.
        model_j: Control model for agent ``j``; :data:`None` when the
            caller is in :attr:`SafetyMode.EGO_ONLY` mode and agent
            ``j`` is the partner (its action is not a decision variable).
        n_total: Total length of the QP variable vector.
        slot_i: Index of the first action component of agent ``i``.
        slot_j: Index of the first action component of agent ``j``;
            :data:`None` mirrors ``model_j``.

    Returns:
        Length-``n_total`` row of action-space coefficients.
    """
    row: FloatArray = np.zeros(n_total, dtype=np.float64)
    j_i = _agent_jacobian(model_i, snap_i)
    # Cartesian row: -n_hat^T ddot p_i + n_hat^T ddot p_j <= -RHS.
    row[slot_i : slot_i + model_i.action_dim] = -(pair_row.n_hat @ j_i)
    if model_j is not None and slot_j is not None:
        j_j = _agent_jacobian(model_j, snap_j)
        row[slot_j : slot_j + model_j.action_dim] = pair_row.n_hat @ j_j
    return row


def _agent_jacobian(model: AgentControlModel, state: AgentSnapshot) -> FloatArray:
    """Linearise ``action_to_cartesian_accel`` at ``state`` (ADR-004 §Decision; spike_004A).

    The control map is linear in action for every implementation
    Phase-0 supports (:class:`DoubleIntegratorControlModel` is the
    identity; :class:`JacobianControlModel` is ``J(state) @ action``).
    The Jacobian is obtained column-by-column from the action-space
    standard basis; this is exact (not numerical differentiation) for
    any linear map and is O(action_dim * position_dim) work per row.

    Args:
        model: Per-agent control model.
        state: Current snapshot.

    Returns:
        Jacobian matrix of shape ``(position_dim, action_dim)``,
        dtype ``float64``.
    """
    jac = np.zeros((model.position_dim, model.action_dim), dtype=np.float64)
    basis = np.zeros(model.action_dim, dtype=np.float64)
    for k in range(model.action_dim):
        basis[k] = 1.0
        jac[:, k] = model.action_to_cartesian_accel(state, basis)
        basis[k] = 0.0
    return jac


class ExpCBFQP:
    """Exp CBF-QP filter (Wang-Ames-Egerstedt 2017 §III; ADR-004 §Decision).

    Implements the integrated outer-layer safety filter under the
    spike_004A taxonomy: builds pairwise CBF rows in Cartesian / safety
    space from :class:`AgentSnapshot` data in ``obs["agent_states"]``,
    projects each row through the per-uid
    :class:`concerto.safety.api.AgentControlModel` to obtain action-
    space coefficients, stacks them with per-agent action-space
    L-infinity bounds (derived from each control model's Cartesian
    capacity and the proportional Wang-Ames-Egerstedt 2017 §IV budget
    split), and projects ``proposed_action`` onto the feasible set via
    :class:`concerto.safety.solvers.QPSolver`. The conformal slack
    ``state.lambda_`` is added per-pair to the right-hand side
    (Huriot & Sibai 2025 §IV; ADR-004 §Decision).

    Mode is set at construction (:class:`SafetyMode`); flipping it
    mid-episode would silently invalidate the conformal state and is
    not supported. A mode change requires a fresh filter + a fresh
    :class:`SafetyState`.

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
        control_models: Mapping[str, AgentControlModel],
        mode: SafetyMode = SafetyMode.EGO_ONLY,
        solver: QPSolver | None = None,
        cbf_gamma: float = DEFAULT_CBF_GAMMA,
    ) -> None:
        """Low-level constructor (ADR-004 §Decision; spike_004A).

        Prefer the mode-specific typed constructors
        :meth:`ExpCBFQP.ego_only`, :meth:`ExpCBFQP.centralized`, and
        :meth:`ExpCBFQP.shared_control` when invoking from typed code
        — their return-type annotations resolve to
        :class:`~concerto.safety.api.EgoOnlySafetyFilter` /
        :class:`~concerto.safety.api.JointSafetyFilter` respectively,
        so static checkers (pyright strict) can see the
        :meth:`filter` signature that matches the constructor's mode.
        This direct ``__init__`` is retained as the low-level entry
        point for tooling, deserialisation, and tests that need to
        exercise the constructor's preconditions directly.

        Args:
            control_models: Per-uid :class:`AgentControlModel` map.
                Required; there is no implicit "first uid's shape"
                fallback (reviewer P0-2). Every uid that appears in
                any subsequent :meth:`filter` call MUST be present
                here; missing uids raise :class:`KeyError` at filter
                time, not at construction (the uid set is allowed to
                shrink between calls — e.g., partner swap — but never
                to include uids the filter has not been told about).
            mode: :class:`SafetyMode`. Default
                :attr:`SafetyMode.EGO_ONLY` (deployment / ad-hoc /
                black-box partner; ADR-006 §Decision).
                :attr:`SafetyMode.CENTRALIZED` is the oracle baseline;
                :attr:`SafetyMode.SHARED_CONTROL` is the lab-only
                co-control baseline.
            solver: QP solver strategy (default :class:`ClarabelSolver`).
            cbf_gamma: Class-K function gain in ``hdot + gamma h >= 0``;
                larger values tighten the constraint (default
                :data:`DEFAULT_CBF_GAMMA`).

        The proportional Wang-Ames-Egerstedt 2017 §IV per-pair budget
        split sizes the per-agent action-space L-infinity bound from
        each control model's Cartesian acceleration capacity. The
        relative-degree-aware variant is a Phase-1 stub (ADR-004 Open
        Question #3); :class:`concerto.safety.budget_split.ProportionalBudgetSplit`
        is the Phase-0 default.
        """
        if not control_models:
            msg = (
                "control_models must be a non-empty dict[uid, AgentControlModel] "
                "(spike_004A §Per-agent control model; reviewer P0-2)."
            )
            raise ValueError(msg)
        self._control_models: dict[str, AgentControlModel] = dict(control_models)
        self._mode: SafetyMode = mode
        self._solver: QPSolver = solver if solver is not None else ClarabelSolver()
        self._cbf_gamma: float = cbf_gamma
        self._budget_split = ProportionalBudgetSplit()

    @classmethod
    def ego_only(
        cls,
        *,
        control_models: Mapping[str, AgentControlModel],
        solver: QPSolver | None = None,
        cbf_gamma: float = DEFAULT_CBF_GAMMA,
    ) -> EgoOnlySafetyFilter:
        """Typed constructor for :attr:`SafetyMode.EGO_ONLY` (ADR-004 §Public API; spike_004A).

        Returns the filter through the
        :class:`~concerto.safety.api.EgoOnlySafetyFilter` Protocol so
        callers get the mode-specific :meth:`filter` signature (single
        :class:`FloatArray` ego action + ``partner_predicted_states``)
        at static-check time. This is the deployment / ad-hoc /
        black-box-partner default (ADR-006 §Decision; reviewer P0-3).

        Args:
            control_models: See :meth:`__init__`.
            solver: See :meth:`__init__`.
            cbf_gamma: See :meth:`__init__`.

        Returns:
            An :class:`ExpCBFQP` instance whose static type narrows
            to :class:`~concerto.safety.api.EgoOnlySafetyFilter`.
        """
        return cls(
            control_models=control_models,
            mode=SafetyMode.EGO_ONLY,
            solver=solver,
            cbf_gamma=cbf_gamma,
        )

    @classmethod
    def centralized(
        cls,
        *,
        control_models: Mapping[str, AgentControlModel],
        solver: QPSolver | None = None,
        cbf_gamma: float = DEFAULT_CBF_GAMMA,
    ) -> JointSafetyFilter:
        """Typed constructor for :attr:`SafetyMode.CENTRALIZED` (ADR-004 §Public API; spike_004A).

        Returns the filter through the
        :class:`~concerto.safety.api.JointSafetyFilter` Protocol so
        callers get the joint-action :meth:`filter` signature
        (per-uid ``dict[uid, FloatArray]`` in and out). Oracle /
        ablation baseline only — never used at evaluation against
        ad-hoc partners (ADR-006 §Decision; ADR-014 §Decision).

        Args:
            control_models: See :meth:`__init__`.
            solver: See :meth:`__init__`.
            cbf_gamma: See :meth:`__init__`.

        Returns:
            An :class:`ExpCBFQP` instance whose static type narrows
            to :class:`~concerto.safety.api.JointSafetyFilter`.
        """
        return cls(
            control_models=control_models,
            mode=SafetyMode.CENTRALIZED,
            solver=solver,
            cbf_gamma=cbf_gamma,
        )

    @classmethod
    def shared_control(
        cls,
        *,
        control_models: Mapping[str, AgentControlModel],
        solver: QPSolver | None = None,
        cbf_gamma: float = DEFAULT_CBF_GAMMA,
    ) -> JointSafetyFilter:
        """Typed constructor for :attr:`SafetyMode.SHARED_CONTROL` (ADR-004 §Public API).

        Returns the filter through the
        :class:`~concerto.safety.api.JointSafetyFilter` Protocol; the
        ``partner_action_bound`` argument to :meth:`filter` is
        required at call time. Lab-only baseline measuring how much
        of the centralised headroom comes from co-control budget vs
        partner co-operation (reviewer P0-3 mode 3).

        Args:
            control_models: See :meth:`__init__`.
            solver: See :meth:`__init__`.
            cbf_gamma: See :meth:`__init__`.

        Returns:
            An :class:`ExpCBFQP` instance whose static type narrows
            to :class:`~concerto.safety.api.JointSafetyFilter`.
        """
        return cls(
            control_models=control_models,
            mode=SafetyMode.SHARED_CONTROL,
            solver=solver,
            cbf_gamma=cbf_gamma,
        )

    @property
    def mode(self) -> SafetyMode:
        """Current safety mode (ADR-004 §Decision; spike_004A §Three-mode taxonomy)."""
        return self._mode

    def reset(self, *, seed: int | None = None) -> None:
        """Reset filter state at episode start (ADR-004 §Decision).

        The exp CBF-QP is stateless across episodes — no internal RNG to
        re-seed. The conformal layer (PR6) is the stateful component;
        it is reset via :class:`SafetyState` directly.
        """
        del seed

    @overload
    def filter(  # noqa: D418  # ADR-004 §Public API typed overload; impl below.
        self,
        proposed_action: FloatArray,
        obs: dict[str, object],
        state: SafetyState,
        bounds: Bounds,
        *,
        ego_uid: str,
        partner_predicted_states: dict[str, AgentSnapshot],
        dt: float,
    ) -> tuple[FloatArray, FilterInfo]:
        """EGO_ONLY overload (ADR-004 §Public API; spike_004A §Three-mode taxonomy)."""

    @overload
    def filter(  # noqa: D418  # ADR-004 §Public API typed overload; impl below.
        self,
        proposed_action: dict[str, FloatArray],
        obs: dict[str, object],
        state: SafetyState,
        bounds: Bounds,
        *,
        ego_uid: str | None = ...,
        partner_action_bound: float | None = ...,
    ) -> tuple[dict[str, FloatArray], FilterInfo]:
        """CENTRALIZED / SHARED_CONTROL overload (ADR-004 §Public API; spike_004A)."""

    def filter(
        self,
        proposed_action: dict[str, FloatArray] | FloatArray,
        obs: dict[str, object],
        state: SafetyState,
        bounds: Bounds,
        *,
        ego_uid: str | None = None,
        partner_predicted_states: dict[str, AgentSnapshot] | None = None,
        partner_action_bound: float | None = None,
        dt: float | None = None,
    ) -> tuple[dict[str, FloatArray] | FloatArray, FilterInfo]:
        """Project ``proposed_action`` onto the CBF-safe set (ADR-004 §Decision).

        The signature depends on :attr:`mode`:

        - :attr:`SafetyMode.EGO_ONLY`: ``proposed_action`` is a single
          :class:`FloatArray` for the ego; ``ego_uid`` and
          ``partner_predicted_states`` are required. The QP decides
          only the ego's action; partner motion enters as a known
          drift term on the constraint RHS. Returns
          ``(safe_ego_action, info)`` where ``safe_ego_action`` is a
          :class:`FloatArray` of shape
          ``(control_models[ego_uid].action_dim,)``.
        - :attr:`SafetyMode.CENTRALIZED`: ``proposed_action`` is a
          ``dict[uid, FloatArray]``; ``ego_uid`` and partner-related
          arguments are ignored. Returns ``(dict_of_safe_actions, info)``
          with one entry per input uid.
        - :attr:`SafetyMode.SHARED_CONTROL`: same shape as
          :attr:`SafetyMode.CENTRALIZED`. ``partner_action_bound`` is
          required: every uid *other* than ``ego_uid`` is constrained
          to lie within an L-infinity ball of ``partner_action_bound``
          around its proposed action.

        Args:
            proposed_action: Either a per-uid dict (CENTRALIZED /
                SHARED_CONTROL) or a single ego action (EGO_ONLY).
            obs: Observation dict carrying ``obs["agent_states"]:
                dict[uid, AgentSnapshot]``. ``obs["meta"]["partner_id"]``
                (M4 contract) is consumed by the conformal layer
                (PR6); this module is partner-id-agnostic.
            state: Mutable :class:`SafetyState`; ``state.lambda_`` is
                read per-pair on the constraint RHS but not mutated
                here — the conformal update happens in PR6.
            bounds: Per-task :class:`Bounds`.
            ego_uid: Required for :attr:`SafetyMode.EGO_ONLY` and
                :attr:`SafetyMode.SHARED_CONTROL`; identifies which
                uid in ``obs["agent_states"]`` is the ego.
            partner_predicted_states: Required for
                :attr:`SafetyMode.EGO_ONLY`. Per-partner-uid predicted
                :class:`AgentSnapshot` at the *next* control step,
                used to evaluate the partner's predicted Cartesian
                acceleration; this enters the constraint RHS as a
                known drift term. The Phase-0 caller derives this
                from :func:`concerto.safety.conformal.constant_velocity_predict`.
            partner_action_bound: Required for
                :attr:`SafetyMode.SHARED_CONTROL`. Strictly positive
                L-infinity ball radius around each partner's proposed
                action.
            dt: Required for :attr:`SafetyMode.EGO_ONLY`. The
                predictor's lookahead horizon in seconds (typically the
                env's control-step duration), used to convert the
                predicted velocity delta into a Cartesian acceleration
                ``(v_pred - v_now) / dt`` on the constraint RHS
                (ADR-004 §Decisions, "Predicted-acceleration units";
                external-review P0-1, 2026-05-16). Must be strictly
                positive. Ignored in :attr:`SafetyMode.CENTRALIZED`
                and :attr:`SafetyMode.SHARED_CONTROL` (no partner-
                disturbance term).

        Returns:
            ``(safe_action, info)`` — shape matches the input
            ``proposed_action`` (single array in
            :attr:`SafetyMode.EGO_ONLY`; dict in
            :attr:`SafetyMode.CENTRALIZED` / :attr:`SafetyMode.SHARED_CONTROL`).
            ``info`` is a :class:`FilterInfo` payload.

        Raises:
            ConcertoSafetyInfeasible: When the QP is infeasible. The
                caller MUST route to the braking fallback (ADR-004
                risk-mitigation #1).
            ValueError: On mode-specific argument-shape problems
                (missing ``ego_uid`` in :attr:`SafetyMode.EGO_ONLY`;
                missing ``partner_action_bound`` in
                :attr:`SafetyMode.SHARED_CONTROL`; etc.).
        """
        snaps_obj = obs.get("agent_states")
        if not isinstance(snaps_obj, dict):
            msg = (
                "obs['agent_states'] must be dict[uid, AgentSnapshot]; "
                f"got {type(snaps_obj).__name__}"
            )
            raise ValueError(msg)
        # The obs['agent_states'] payload is the env adapter's
        # integration boundary (ADR-004 §Decision); the env produces
        # AgentSnapshot per uid, but obs is typed as
        # ``dict[str, object]`` at the SafetyFilter API boundary so the
        # value type must be cast here.
        snaps: dict[str, AgentSnapshot] = cast("dict[str, AgentSnapshot]", snaps_obj)

        if self._mode is SafetyMode.EGO_ONLY:
            return self._filter_ego_only(
                proposed_action=proposed_action,
                snaps=snaps,
                state=state,
                bounds=bounds,
                ego_uid=ego_uid,
                partner_predicted_states=partner_predicted_states,
                dt=dt,
            )
        return self._filter_joint(
            proposed_action=proposed_action,
            snaps=snaps,
            state=state,
            bounds=bounds,
            ego_uid=ego_uid,
            partner_action_bound=partner_action_bound,
        )

    def _filter_ego_only(  # noqa: PLR0915
        # Single QP-build flow + validation guards; splitting it would
        # force shared state through call arguments (premature
        # abstraction per the project rules). The statement count is
        # the price of an end-to-end CBF row build + projection in
        # one place.
        self,
        *,
        proposed_action: dict[str, FloatArray] | FloatArray,
        snaps: dict[str, AgentSnapshot],
        state: SafetyState,
        bounds: Bounds,
        ego_uid: str | None,
        partner_predicted_states: dict[str, AgentSnapshot] | None,
        dt: float | None,
    ) -> tuple[FloatArray, FilterInfo]:
        """EGO_ONLY filter implementation (ADR-004 §Decision; spike_004A §Three-mode taxonomy).

        Internal helper. See :meth:`filter` for the public contract.
        """
        if ego_uid is None:
            msg = "ego_uid is required in SafetyMode.EGO_ONLY (spike_004A §Three-mode taxonomy)."
            raise ValueError(msg)
        if partner_predicted_states is None:
            msg = (
                "partner_predicted_states is required in SafetyMode.EGO_ONLY "
                "(spike_004A §Three-mode taxonomy): the partner enters as a "
                "predicted disturbance on the constraint RHS."
            )
            raise ValueError(msg)
        if dt is None:
            msg = (
                "dt is required in SafetyMode.EGO_ONLY: the predicted "
                "Cartesian acceleration is (v_pred - v_now) / dt and the "
                "filter cannot infer the predictor's lookahead horizon "
                "(ADR-004 §Decisions, 'Predicted-acceleration units'; "
                "external-review P0-1, 2026-05-16)."
            )
            raise ValueError(msg)
        # ``not (dt > 0)`` catches NaN, 0.0, and negative values in one shot;
        # the separate ``isfinite`` rejects ``+inf`` (which would silently
        # mask the partner-disturbance term to zero on the constraint RHS).
        if not (dt > 0.0) or not np.isfinite(dt):
            msg = f"dt must be a finite strictly-positive float in SafetyMode.EGO_ONLY; got {dt!r}."
            raise ValueError(msg)
        if not isinstance(proposed_action, np.ndarray):
            msg = (
                "proposed_action must be a numpy ndarray for the ego in "
                "SafetyMode.EGO_ONLY; got a dict (use SafetyMode.CENTRALIZED "
                "for the joint-action interface)."
            )
            raise TypeError(msg)
        if ego_uid not in snaps:
            msg = f"ego_uid {ego_uid!r} not in obs['agent_states'] keys {list(snaps.keys())!r}"
            raise ValueError(msg)
        if ego_uid not in self._control_models:
            msg = f"ego_uid {ego_uid!r} not in control_models keys {list(self._control_models)!r}"
            raise KeyError(msg)

        ego_model = self._control_models[ego_uid]
        partner_uids = [uid for uid in snaps if uid != ego_uid]
        n_pairs = len(partner_uids)
        if state.lambda_.shape != (n_pairs,):
            msg = (
                f"state.lambda_ shape mismatch: expected ({n_pairs},), "
                f"got {state.lambda_.shape}. Reset SafetyState on partner-set "
                "change per ADR-004 risk-mitigation #2."
            )
            raise ValueError(msg)

        n_total = ego_model.action_dim
        u_hat = proposed_action.astype(np.float64, copy=False)
        if u_hat.shape != (n_total,):
            msg = (
                f"proposed_action shape {u_hat.shape} mismatches "
                f"control_models[{ego_uid!r}].action_dim={n_total}"
            )
            raise ValueError(msg)

        rows: list[FloatArray] = []
        rhs_list: list[float] = []
        constraint_violation = np.zeros(n_pairs, dtype=np.float64)
        ego_max_cart = ego_model.max_cartesian_accel(bounds)
        for pair_idx, partner_uid in enumerate(partner_uids):
            if partner_uid not in self._control_models:
                msg = (
                    f"partner_uid {partner_uid!r} not in control_models "
                    f"keys {list(self._control_models)!r}"
                )
                raise KeyError(msg)
            if partner_uid not in partner_predicted_states:
                msg = (
                    f"partner_uid {partner_uid!r} not in partner_predicted_states "
                    f"keys {list(partner_predicted_states)!r}"
                )
                raise KeyError(msg)
            partner_model = self._control_models[partner_uid]
            partner_max_cart = partner_model.max_cartesian_accel(bounds)
            alpha_pair = ego_max_cart + partner_max_cart

            # Partner's predicted Cartesian acceleration enters as a
            # known drift term: the row becomes
            # -n_hat^T ddot p_ego <= -RHS - n_hat^T ddot p_partner_pred.
            partner_pred_accel = _predicted_cartesian_accel(
                snap_now=snaps[partner_uid],
                snap_pred=partner_predicted_states[partner_uid],
                dt=dt,
            )
            cart_row, rhs, h_ij = _build_ego_only_row(
                ego_snap=snaps[ego_uid],
                partner_snap=snaps[partner_uid],
                partner_predicted_accel=partner_pred_accel,
                alpha_pair=alpha_pair,
                gamma=self._cbf_gamma,
                lambda_ij=float(state.lambda_[pair_idx]),
            )
            # Project the Cartesian row to action space via the ego's
            # control-model Jacobian. ``cart_row == -n_hat``; for
            # DoubleIntegratorControlModel J_ego is the identity and
            # ``cart_row @ J_ego`` equals ``cart_row`` unchanged, matching
            # the pre-refactor row coefficients byte-for-byte.
            j_ego = _agent_jacobian(ego_model, snaps[ego_uid])
            action_row: FloatArray = np.zeros(n_total, dtype=np.float64)
            action_row[: ego_model.action_dim] = cart_row @ j_ego
            rows.append(action_row)
            rhs_list.append(rhs)
            constraint_violation[pair_idx] = max(0.0, -h_ij)

        # Per-component action-space L-infinity bound; the Cartesian
        # capacity drives ``alpha_pair`` only — the action-space envelope
        # is the agent's own component-wise bound (Bounds.action_norm in
        # Phase-0; future models may override this).
        identity = np.eye(n_total, dtype=np.float64)
        ego_action_bound = np.full(n_total, bounds.action_norm, dtype=np.float64)
        bound_rows = np.vstack([identity, -identity])
        bound_rhs = np.concatenate([ego_action_bound, ego_action_bound])

        constraint_a, constraint_b = _stack_constraints(rows, rhs_list, bound_rows, bound_rhs)
        cost_p = 2.0 * np.eye(n_total, dtype=np.float64)
        cost_q = -2.0 * u_hat

        start = time.perf_counter()
        x_safe, _ = self._solver.solve(cost_p, cost_q, constraint_a, constraint_b)
        qp_solve_ms = (time.perf_counter() - start) * 1000.0

        info: FilterInfo = {
            "lambda": state.lambda_.copy(),
            "constraint_violation": constraint_violation,
            "prediction_gap_loss": None,
            "fallback_fired": False,
            "qp_solve_ms": qp_solve_ms,
        }
        return x_safe[:n_total].astype(np.float64, copy=True), info

    def _filter_joint(  # noqa: PLR0912, PLR0915
        # CENTRALIZED + SHARED_CONTROL share the same QP-build flow with
        # one branch on partner-bound restriction; the alternative is
        # two near-identical helpers, which the project rules call out
        # as premature abstraction. Branches/statements above the
        # default cap are validation guards + a single inner per-uid
        # loop.
        self,
        *,
        proposed_action: dict[str, FloatArray] | FloatArray,
        snaps: dict[str, AgentSnapshot],
        state: SafetyState,
        bounds: Bounds,
        ego_uid: str | None,
        partner_action_bound: float | None,
    ) -> tuple[dict[str, FloatArray], FilterInfo]:
        """CENTRALIZED and SHARED_CONTROL filter implementation (ADR-004 §Decision; spike_004A).

        Internal helper. See :meth:`filter` for the public contract.
        """
        if not isinstance(proposed_action, dict):
            msg = (
                f"proposed_action must be a dict in SafetyMode.{self._mode.name}; "
                f"got {type(proposed_action).__name__}"
            )
            raise TypeError(msg)
        if self._mode is SafetyMode.SHARED_CONTROL:
            if partner_action_bound is None or partner_action_bound <= 0.0:
                msg = (
                    "partner_action_bound must be a positive float in "
                    "SafetyMode.SHARED_CONTROL (spike_004A §Three-mode taxonomy)."
                )
                raise ValueError(msg)
            if ego_uid is None or ego_uid not in proposed_action:
                msg = (
                    "ego_uid is required in SafetyMode.SHARED_CONTROL and must "
                    "appear in proposed_action keys."
                )
                raise ValueError(msg)

        uids = list(proposed_action.keys())
        slot_of: dict[str, int] = {}
        offset = 0
        for uid in uids:
            if uid not in self._control_models:
                msg = f"uid {uid!r} not in control_models keys {list(self._control_models)!r}"
                raise KeyError(msg)
            if uid not in snaps:
                msg = f"uid {uid!r} not in obs['agent_states'] keys {list(snaps.keys())!r}"
                raise ValueError(msg)
            slot_of[uid] = offset
            offset += self._control_models[uid].action_dim
        n_total = offset

        u_hat_parts: list[FloatArray] = []
        for uid in uids:
            model = self._control_models[uid]
            arr = proposed_action[uid].astype(np.float64, copy=False)
            if arr.shape != (model.action_dim,):
                msg = (
                    f"proposed_action[{uid!r}] shape {arr.shape} mismatches "
                    f"control_models[{uid!r}].action_dim={model.action_dim}"
                )
                raise ValueError(msg)
            u_hat_parts.append(arr)
        u_hat = np.concatenate(u_hat_parts).astype(np.float64, copy=False)

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
        for a, uid_i in enumerate(uids):
            for uid_j in uids[a + 1 :]:
                model_i = self._control_models[uid_i]
                model_j = self._control_models[uid_j]
                alpha_i_cart = model_i.max_cartesian_accel(bounds)
                alpha_j_cart = model_j.max_cartesian_accel(bounds)
                alpha_pair = alpha_i_cart + alpha_j_cart

                pair_row = _build_cartesian_pair_row(
                    snap_i=snaps[uid_i],
                    snap_j=snaps[uid_j],
                    alpha_pair=alpha_pair,
                    gamma=self._cbf_gamma,
                    lambda_ij=float(state.lambda_[pair_idx]),
                )
                action_row = _project_cartesian_row_to_action(
                    pair_row,
                    snap_i=snaps[uid_i],
                    snap_j=snaps[uid_j],
                    model_i=model_i,
                    model_j=model_j,
                    n_total=n_total,
                    slot_i=slot_of[uid_i],
                    slot_j=slot_of[uid_j],
                )
                rows.append(action_row)
                # Sign convention: -n_hat^T (a_i - a_j) <= pair_row.rhs;
                # action-space substitution preserves the RHS unchanged.
                rhs_list.append(pair_row.rhs)
                constraint_violation[pair_idx] = max(0.0, -pair_row.h_ij)
                pair_idx += 1

        bound_rows_list: list[FloatArray] = []
        bound_rhs_list: list[float] = []
        for uid in uids:
            model = self._control_models[uid]
            # Per-component action-space L-infinity bound, matching the
            # pre-refactor convention; the per-agent Cartesian capacity
            # affects ``alpha_pair`` only.
            action_bound = np.full(model.action_dim, bounds.action_norm, dtype=np.float64)
            if (
                self._mode is SafetyMode.SHARED_CONTROL
                and uid != ego_uid
                and partner_action_bound is not None
            ):
                # Partner is co-controllable but restricted to an
                # L-infinity ball of partner_action_bound around its
                # proposed action. partner_action_bound is guaranteed
                # non-None by the SHARED_CONTROL preamble above; the
                # explicit narrowing condition keeps pyright happy
                # without an assert.
                centre = proposed_action[uid].astype(np.float64, copy=False)
                partner_radius = np.full(
                    model.action_dim,
                    float(partner_action_bound),
                    dtype=np.float64,
                )
                upper_eff = np.minimum(action_bound, centre + partner_radius)
                lower_eff = np.maximum(-action_bound, centre - partner_radius)
            else:
                upper_eff = action_bound
                lower_eff = -action_bound
            for k in range(model.action_dim):
                upper_row: FloatArray = np.zeros(n_total, dtype=np.float64)
                upper_row[slot_of[uid] + k] = 1.0
                bound_rows_list.append(upper_row)
                bound_rhs_list.append(float(upper_eff[k]))
                lower_row: FloatArray = np.zeros(n_total, dtype=np.float64)
                lower_row[slot_of[uid] + k] = -1.0
                bound_rows_list.append(lower_row)
                bound_rhs_list.append(float(-lower_eff[k]))

        bound_rows = (
            np.vstack(bound_rows_list)
            if bound_rows_list
            else np.zeros((0, n_total), dtype=np.float64)
        )
        bound_rhs = np.asarray(bound_rhs_list, dtype=np.float64)

        constraint_a, constraint_b = _stack_constraints(rows, rhs_list, bound_rows, bound_rhs)
        cost_p = 2.0 * np.eye(n_total, dtype=np.float64)
        cost_q = -2.0 * u_hat

        start = time.perf_counter()
        x_safe, _ = self._solver.solve(cost_p, cost_q, constraint_a, constraint_b)
        qp_solve_ms = (time.perf_counter() - start) * 1000.0

        safe_action: dict[str, FloatArray] = {}
        for uid in uids:
            slot = slot_of[uid]
            width = self._control_models[uid].action_dim
            safe_action[uid] = x_safe[slot : slot + width].astype(np.float64, copy=True)

        info: FilterInfo = {
            "lambda": state.lambda_.copy(),
            "constraint_violation": constraint_violation,
            "prediction_gap_loss": None,
            "fallback_fired": False,
            "qp_solve_ms": qp_solve_ms,
        }
        return safe_action, info


def _predicted_cartesian_accel(
    *,
    snap_now: AgentSnapshot,
    snap_pred: AgentSnapshot,
    dt: float,
) -> FloatArray:
    """Estimate the partner's Cartesian acceleration from successive snapshots (ADR-004 §Decision).

    Internal helper. Approximates ``ddot p_partner`` as the velocity
    difference between the predicted snapshot at step ``k+1`` and the
    current snapshot at step ``k``, divided by the lookahead horizon
    ``dt``: ``a = (v_pred - v_now) / dt`` (forward-difference Cartesian
    acceleration consistent with the predictor's extrapolation
    over ``dt``; cf. the same ``dt`` parameter on
    :func:`concerto.safety.conformal.update_lambda_from_predictor` and
    :func:`concerto.safety.conformal.constant_velocity_predict`).

    With the constant-velocity predictor stub (Phase-0) the velocity
    delta is zero and the result is zero for every ``dt``; a smarter
    Phase-1 predictor (AoI-conditioned, learned) produces non-zero
    accelerations and the row RHS carries the corresponding drift at
    the correct physical scale.

    Args:
        snap_now: Partner snapshot at the current step ``k``.
        snap_pred: Partner snapshot predicted for the next step ``k+1``.
        dt: Lookahead horizon used by the predictor, in seconds.
            Strictly positive; the same value the predictor was called
            with (typically the env's control-step duration).

    Returns:
        Predicted Cartesian acceleration of shape
        ``(position_dim,)``, dtype ``float64``.

    Notes:
        Closes external-review P0-1 (2026-05-16). The pre-fix form
        returned the raw velocity delta without dividing by ``dt``,
        making the CBF RHS wrong by a factor of ``1/dt`` for any
        nonzero predictor; the bug was latent under the constant-
        velocity stub. See
        ``tests/property/test_predicted_acceleration_scales_with_dt.py``
        for the pinned scaling law.
    """
    return ((snap_pred.velocity - snap_now.velocity) / dt).astype(np.float64, copy=False)


def _stack_constraints(
    cbf_rows: list[FloatArray],
    cbf_rhs_list: list[float],
    bound_rows: FloatArray,
    bound_rhs: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Stack CBF rows + action-bound rows into one ``(A, b)`` (ADR-004 §Decision).

    Internal helper.
    """
    if cbf_rows:
        cbf_a = np.vstack(cbf_rows)
        cbf_b = np.asarray(cbf_rhs_list, dtype=np.float64)
        return np.vstack([cbf_a, bound_rows]), np.concatenate([cbf_b, bound_rhs])
    return bound_rows, bound_rhs


__all__ = [
    "DEFAULT_CBF_GAMMA",
    "AgentSnapshot",
    "ExpCBFQP",
    "pair_h_value",
]
