# SPDX-License-Identifier: Apache-2.0
"""Public safety-stack API: Protocol + Bounds + SafetyState (ADR-004 + ADR-006 + ADR-014).

ADR-004 §Decision pins the three-layer architecture (exp CBF-QP backbone +
conformal slack overlay + OSCBF inner filter) plus a hard braking fallback
(ADR-004 risk-mitigation #1, Wang-Ames-Egerstedt 2017 eq. 17). This module
declares the contracts every layer's caller depends on:

- :class:`Bounds` — per-task numeric envelope (ADR-006 §Decision Option C).
- :class:`SafetyState` — mutable conformal-CBF state (Huriot & Sibai 2025
  §IV; ADR-004 risk-mitigation #2 governs the partner-swap warmup).
- :class:`FilterInfo` — telemetry payload returned alongside the safe
  action; consumed by the three-table renderer (ADR-014 §Decision).
- :class:`EgoOnlySafetyFilter` and :class:`JointSafetyFilter` — the two
  mode-specific Protocols the integrated stack implements (ADR-004
  §Public API; spike_004A §Three-mode taxonomy; reviewer P0-2). The
  EGO_ONLY filter takes a single ego :class:`FloatArray`; the
  CENTRALIZED / SHARED_CONTROL filters take the joint per-uid dict.
  The pre-refactor :data:`SafetyFilter` name is preserved as a
  deprecated union alias; see :func:`__getattr__` for the deprecation
  trigger and removal target (0.3.0).

The module imports nothing from ``chamber.*`` — the dependency-direction
rule (plan/10 §2) keeps the method side stand-alone and testable against a
stub env. The partner-swap contract reads ``obs["meta"]["partner_id"]``
(M4 will produce the hash; M3 ships the consumer side, gated to
single-partner mode when the field is ``None``).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Protocol, TypedDict, runtime_checkable

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from concerto.safety.cbf_qp import AgentSnapshot

#: Per-agent action vector alias. Public so Protocol signatures stay readable.
FloatArray = npt.NDArray[np.float64]


def canonical_pair_order(uids: Iterable[str]) -> list[str]:
    """Canonical lexicographic UID order for pair iteration (ADR-004 §Decision).

    Public utility used by the three pair-iteration entry points
    (:func:`concerto.safety.conformal.compute_prediction_gap_for_pairs`,
    :meth:`concerto.safety.cbf_qp.ExpCBFQP._filter_ego_only`, and
    :meth:`concerto.safety.cbf_qp.ExpCBFQP._filter_joint`); third-party
    consumers building their own pair-iteration code over the safety
    stack should call this helper to stay aligned with the same
    canonical ordering :class:`SafetyState` ``lambda_`` is indexed
    against. Returns
    a list of the UIDs sorted lexicographically; downstream code
    iterates upper-triangular pairs ``(uids[i], uids[j]) for i < j``
    so the pair index for any ``(uid_a, uid_b)`` (with
    ``uid_a < uid_b`` lexicographically) is stable across callers and
    across dict reconstruction (external-review P1, 2026-05-16).

    Pre-amendment, each entry point used ``list(<dict>.keys())`` and
    relied on Python 3.7+'s insertion-order guarantee — but the
    invariant was implicit, and any caller that reconstructed the
    snapshot dict in a different order silently misaligned the
    per-pair conformal-slack vector ``SafetyState.lambda_`` with the
    pairwise CBF constraints. The canonical sort makes the pair index
    a pure function of the uid set, eliminating the cross-call
    misalignment failure mode.

    The structurally cleanest fix (promoting ``lambda_`` to
    ``dict[tuple[str, str], float]``) is deferred to Phase-1 — it
    requires bumping the ADR-014 ``FilterInfo["lambda"]`` wire
    contract and the three-table renderer at the same time, which
    busts the Phase-0 diff budget. Tracking issue is opened alongside
    the present PR; the Stage-1 AS spike is the natural integration
    point.

    Args:
        uids: An iterable of agent UIDs.

    Returns:
        A new list containing the UIDs sorted lexicographically.

    Raises:
        ValueError: If the iterable contains duplicates (a partner-set
            invariant the upstream filter already enforces, surfaced
            here as a defensive check).
    """
    out = sorted(uids)
    if len(set(out)) != len(out):
        msg = f"duplicate uids in pair iteration: {out!r}"
        raise ValueError(msg)
    return out


@dataclass(frozen=True)
class Bounds:
    r"""Per-task numeric bounds for the safety stack (ADR-006 §Decision Option C).

    The ``Bounds`` envelope is the explicit-numeric half of ADR-006's hybrid
    decision: bounds gate QP feasibility and reportability while the
    conformal slack (:class:`SafetyState`) governs online adaptation. The
    enumerated values track ADR-006 §Consequences — the comm-latency bound
    is anchored to the URLLC-3GPP-R17 sweep table from M2; ``action_norm``
    and ``action_rate`` are task-specific (Phase-1 fills them; Phase-0
    ships sane defaults). The ``force_limit`` field is a Phase-0 stub for
    ADR-007 Open Question #4 (per-vendor decomposition, ADR-004 Open
    Question #5) — a strategy-interfaced per-vendor handler slots in once
    Stage-3 SA resolves the open question.

    **Known semantic inconsistency (2026-05-16; external-review P1-3):**
    ``action_norm`` is consumed by two safety layers with mismatched
    semantics. The exponential CBF-QP outer filter
    (:class:`concerto.safety.cbf_qp.ExpCBFQP`) enforces it as a
    per-component **L-infinity** bound (``|u_i[k]| <= action_norm`` for
    every component ``k``). The Cartesian emergency controller
    (:mod:`concerto.safety.emergency`, around line 174) reads it as an
    **L2** magnitude cap on the emergency acceleration. For a ``d``-
    dimensional action these are inconsistent: an action with
    ``||u||_inf <= action_norm`` can have
    ``||u||_2 ~ sqrt(d) * action_norm``. The CBF derivation may
    therefore authorise an action the emergency fallback cannot deliver.
    This is tracked as Phase-1 safety-critical work in ADR-004 §Open
    questions (issue #146); the correct fix is to split ``Bounds`` into
    two fields with explicit semantics (``action_linf_component`` +
    ``cartesian_accel_capacity``). **Until the split lands, callers
    should treat ``action_norm`` as the stricter L2 cap (i.e. set
    ``action_norm = capacity / sqrt(d)``) so the emergency-fallback
    constraint is preserved.**

    The inconsistency is pinned by
    ``tests/property/test_bounds_action_norm_inconsistency_documented.py``
    with ``@pytest.mark.xfail(strict=True)`` so that when the field
    split lands, the test flips from ``xfail`` to ``xpass`` and forces
    a follow-up PR to remove the marker — a built-in regression flag
    for the safety fix.

    Attributes:
        action_norm: **Deprecated semantics — see "Known semantic
            inconsistency" above.** The current consumers split as:
            ``ExpCBFQP`` reads this as per-component L-infinity; the
            emergency controller reads it as an L2 magnitude cap. The
            safe operator pattern is ``action_norm = capacity / sqrt(d)``
            for ``d``-dim actions so both layers agree on the stricter
            envelope. Removal/split target: tracked in issue #146.
        action_rate: Maximum :math:`\lVert u_k - u_{k-1} \rVert_2` per
            agent, per task. Bounds the actuator-rate slack the conformal
            CBF must remain feasible under (ADR-006 §Decision).
        comm_latency_ms: Default mean comm latency, in milliseconds. Set
            from the URLLC profile selected for the run (ADR-006
            §Consequences; ``chamber.comm.URLLC_3GPP_R17``).
        force_limit: Global force threshold for the per-pair contact-force
            constraint (ADR-004 Open Question #5; per-vendor handler slots
            in once Stage-3 SA resolves the decomposition).
    """

    action_norm: float
    action_rate: float
    comm_latency_ms: float
    force_limit: float


#: Default conformal target loss in the conservative manipulation regime
#: (ADR-004 §Decision; Huriot & Sibai 2025 §VI).
DEFAULT_EPSILON: float = -0.05

#: Default conformal learning rate eta for ``lambda_{k+1} = lambda_k + eta * (eps - l_k)``
#: (ADR-004 §Decision; Huriot & Sibai 2025 Theorem 3).
DEFAULT_ETA: float = 0.01

#: Default partner-swap warmup window length in control steps
#: (ADR-004 risk-mitigation #2; ADR-006 risk #3).
DEFAULT_WARMUP_STEPS: int = 50


@dataclass
class SafetyState:
    """Mutable conformal CBF state (Huriot & Sibai 2025 §IV; ADR-004 §Decision).

    The conformal slack vector ``lambda_`` carries one entry per agent
    pair ``(i, j)``; it is updated each control step by
    ``concerto.safety.conformal.update_lambda`` according to Theorem 3's
    rule ``lambda_{k+1} = lambda_k + eta * (eps - l_k)``. ADR-004
    risk-mitigation #2 motivates the partner-swap warmup window: on
    partner identity change (detected via ``obs["meta"]["partner_id"]``)
    the state is reset to ``lambda_safe`` (the value guaranteeing QP
    feasibility under the worst-case bounded prediction error per ADR-006
    Assumption A2) and the next ``warmup_steps_remaining`` steps run with
    a tighter eps.

    Attributes:
        lambda_: Per-pair slack values, shape ``(N_pairs,)`` and dtype
            ``float64``. Initialised to ``lambda_safe``; mutated in place.
        epsilon: Target average loss. ADR-004 §Decision pins the default
            to ``-0.05`` (conservative manipulation regime).
        eta: Conformal learning rate (default ``0.01``).
        warmup_steps_remaining: Decremented each step while in warmup;
            zero outside the warmup window (ADR-004 risk-mitigation #2).
    """

    lambda_: FloatArray
    epsilon: float = DEFAULT_EPSILON
    eta: float = DEFAULT_ETA
    warmup_steps_remaining: int = 0


# Functional TypedDict form: plan/03 §3.1 fixes the wire-key as ``"lambda"``
# (a Python keyword), and the three-table renderer (ADR-014) reads
# ``info["lambda"]`` directly. The functional form lets us keep the spec key.
FilterInfo = TypedDict(
    "FilterInfo",
    {
        "lambda": FloatArray,
        "constraint_violation": FloatArray,
        "prediction_gap_loss": FloatArray | None,
        "fallback_fired": bool,
        "qp_solve_ms": float,
    },
)
"""Telemetry payload returned by the filter Protocols (ADR-014 §Decision).

Returned by :meth:`EgoOnlySafetyFilter.filter` and
:meth:`JointSafetyFilter.filter`.

The fields populate the three-table renderer's row data:

- ``lambda`` — current conformal slack vector; feeds Table 3
  (conservativeness gap λ mean/var vs. oracle gt/noLearn).
- ``constraint_violation`` — per-pair non-negative violation signal
  ``max(0, -h_ij)`` from the CBF backbone. Zero when the constraint is
  satisfied; positive when the barrier is in the violated half-space.
  Counted into Table 2's per-condition violation rates.
- ``prediction_gap_loss`` — per-pair Huriot & Sibai 2025 §IV.A loss
  ``max(0, predicted_h - actual_h)`` against a partner-trajectory
  predictor. This is the signal that drives the conformal update
  (``update_lambda``); :data:`None` when no predictor is wired (the
  CBF backbone alone cannot compute it — see
  ``concerto.safety.conformal.update_lambda_from_predictor``).
- ``fallback_fired`` — feeds Table 2's "fallback fired" column.
- ``qp_solve_ms`` — feeds the OSCBF 1 kHz target check
  (ADR-004 §"OSCBF target").

The constraint-violation and prediction-gap loss are reported as
separate cells of the ADR-014 three-table report because they answer
different questions: ``constraint_violation`` is a per-step CBF gap
(did the QP keep us in the safe set this step?), and
``prediction_gap_loss`` is the conformal calibration signal (was the
predictor's forecast of the safe set wrong?).
"""


class SafetyMode(Enum):
    """Safety-filter operating mode (ADR-004 §Decision; spike_004A §Three-mode taxonomy).

    Three modes make the filter's *who is being optimised* contract
    explicit at the constructor site, so callsites cannot accidentally
    request the oracle solve at deployment (reviewer P0-3):

    - :attr:`EGO_ONLY` — **deployment default.** QP decision variable is
      the ego agent's action only; the partner's motion enters as a
      predicted disturbance on the constraint RHS. This is the
      ad-hoc / black-box-partner setting ADR-006 §Decision pins.
    - :attr:`CENTRALIZED` — oracle / ablation only. QP decision variable
      is the concatenation of every uid's action, with per-uid slot
      widths set by ``control_models[uid].action_dim`` (no implicit
      "first uid's shape" fallback). Used as the upper-bound baseline
      in ADR-014's three-table report. Never used at evaluation
      against ad-hoc partners.
    - :attr:`SHARED_CONTROL` — lab-only baseline (reviewer P0-3 mode 3).
      Same shape as :attr:`CENTRALIZED` with an additional
      ``partner_action_bound`` parameter that restricts the partner's
      adjustment magnitude. Measures how much of the centralised
      headroom comes from co-control budget vs from partner
      co-operation.

    Mode is a **constructor argument**, not a per-call flag: the
    conformal slack vector ``SafetyState.lambda_`` is sized per pair,
    and the pair set differs between :attr:`EGO_ONLY` (one row per
    partner) and :attr:`CENTRALIZED` (full pairwise table). Flipping
    mode mid-episode would silently invalidate the conformal state.
    A mode change requires constructing a new filter with a fresh
    :class:`SafetyState`.
    """

    EGO_ONLY = "ego_only"
    CENTRALIZED = "centralized"
    SHARED_CONTROL = "shared_control"


@runtime_checkable
class AgentControlModel(Protocol):
    """Per-agent action ↔ Cartesian-acceleration map (ADR-004 §Decision; spike_004A).

    The CBF row generator builds pairwise constraints in Cartesian /
    safety space, then projects to per-agent action-space coefficients
    via this Protocol. The Protocol mediates between each agent's
    *action space* (joint torques for a 7-DOF arm, wheel velocities
    for a 2-DOF base, Cartesian accelerations for a point mass) and
    the shared *safety space* (Cartesian acceleration of the agent's
    safety body).

    Implementations MUST be deterministic on identical ``(state,
    action)`` inputs (P6; seeding contract) and MUST be consistent
    with each other: for any ``state`` and any Cartesian acceleration
    ``a`` in the image of :meth:`action_to_cartesian_accel`,
    ``action_to_cartesian_accel(state,
    cartesian_accel_to_action(state, a)) == a`` to within floating-
    point tolerance. The right-inverse exists by construction for
    over-actuated agents (Jacobian has more columns than rows); for
    exactly-actuated agents the right-inverse is the unique inverse.

    Attributes:
        uid: Unique identifier of the agent whose model this is.
        action_dim: Dimensionality of the agent's action vector
            (e.g. 7 for a 7-DOF arm; 2 for a planar double integrator).
        position_dim: Cartesian dimension of the agent's safety body
            (typically 2 for the §V toy crossing, 3 for manipulation).
    """

    @property
    def uid(self) -> str:
        """Unique identifier of the agent (ADR-004 §Decision)."""
        ...

    @property
    def action_dim(self) -> int:
        """Dimensionality of the action vector (ADR-004 §Decision)."""
        ...

    @property
    def position_dim(self) -> int:
        """Cartesian dimension of the safety body (ADR-004 §Decision)."""
        ...

    def action_to_cartesian_accel(self, state: AgentSnapshot, action: FloatArray) -> FloatArray:
        """Map a control-space action to a Cartesian acceleration (ADR-004 §Decision).

        Args:
            state: Current :class:`concerto.safety.cbf_qp.AgentSnapshot`.
                The mapping is state-dependent for any embodiment whose
                Jacobian depends on configuration (e.g. a 7-DOF arm).
            action: Action vector of shape ``(action_dim,)``, dtype
                ``float64``.

        Returns:
            Cartesian acceleration of shape ``(position_dim,)``, dtype
            ``float64``.
        """
        ...

    def cartesian_accel_to_action(
        self, state: AgentSnapshot, cartesian_accel: FloatArray
    ) -> FloatArray:
        """Right-inverse for QP projection back into action space (ADR-004 §Decision).

        Used by the CBF-QP to express the per-agent action-space slot
        coefficients of a Cartesian constraint row. For an
        over-actuated agent the right-inverse selects one of the many
        actions that realise the same Cartesian acceleration; the
        damped-least-squares pseudo-inverse is the canonical choice
        (see :class:`JacobianControlModel`).

        Args:
            state: Current :class:`concerto.safety.cbf_qp.AgentSnapshot`.
            cartesian_accel: Cartesian acceleration of shape
                ``(position_dim,)``, dtype ``float64``.

        Returns:
            Action vector of shape ``(action_dim,)``, dtype ``float64``.
        """
        ...

    def max_cartesian_accel(self, bounds: Bounds) -> float:
        """Per-agent Cartesian acceleration capacity (ADR-004 §Decision; spike_004A).

        Used by the QP-row generator to size the per-pair
        ``alpha_pair = alpha_i_cart + alpha_j_cart`` and to drive the
        proportional Wang-Ames-Egerstedt 2017 §IV budget split for
        heterogeneous embodiments. For a double-integrator agent
        whose action space *is* Cartesian acceleration this is just
        ``bounds.action_norm``.

        Args:
            bounds: Per-task :class:`Bounds` envelope.

        Returns:
            Strictly positive scalar acceleration capacity.
        """
        ...


@dataclass(frozen=True)
class DoubleIntegratorControlModel:
    """Identity action ↔ Cartesian-acceleration map (ADR-004 §Decision; spike_004A §Reduction).

    Default :class:`AgentControlModel` for double-integrator agents
    whose action space coincides with Cartesian acceleration (the
    Wang-Ames-Egerstedt 2017 §V toy crossing and every Phase-0
    homogeneous test). The Jacobian ``J_i = d cartesian_accel /
    d action`` is the identity, so the CBF row coefficients projected
    through this model are element-wise identical to the rows the
    pre-refactor ``_pair_constraint_row`` emitted. This is the
    migration shim: tests that construct homogeneous 2-D pairs pass
    a :class:`DoubleIntegratorControlModel` per uid and run unchanged
    through :class:`ExpCBFQP` in :attr:`SafetyMode.CENTRALIZED` mode.

    Attributes:
        uid: Unique identifier of the agent.
        action_dim: Dimensionality of the agent's action vector
            (equal to ``position_dim`` by construction).
    """

    uid: str
    action_dim: int

    @property
    def position_dim(self) -> int:
        """Cartesian dimension of the safety body (ADR-004 §Decision).

        For a double integrator the action and position spaces have
        the same dimension by definition.
        """
        return self.action_dim

    def action_to_cartesian_accel(self, state: AgentSnapshot, action: FloatArray) -> FloatArray:
        """Return ``action`` unchanged (ADR-004 §Decision; spike_004A §Reduction).

        The double-integrator identity map.
        """
        del state
        if action.shape != (self.action_dim,):
            msg = (
                f"action shape {action.shape} mismatches "
                f"DoubleIntegratorControlModel(uid={self.uid!r}).action_dim={self.action_dim}"
            )
            raise ValueError(msg)
        return action.astype(np.float64, copy=False)

    def cartesian_accel_to_action(
        self, state: AgentSnapshot, cartesian_accel: FloatArray
    ) -> FloatArray:
        """Return ``cartesian_accel`` unchanged (ADR-004 §Decision; spike_004A §Reduction).

        The double-integrator identity right-inverse.
        """
        del state
        if cartesian_accel.shape != (self.action_dim,):
            msg = (
                f"cartesian_accel shape {cartesian_accel.shape} mismatches "
                f"DoubleIntegratorControlModel(uid={self.uid!r}).position_dim={self.action_dim}"
            )
            raise ValueError(msg)
        return cartesian_accel.astype(np.float64, copy=False)

    def max_cartesian_accel(self, bounds: Bounds) -> float:
        """Return ``bounds.action_norm`` (ADR-004 §Decision; spike_004A §Per-agent alpha).

        For a double integrator the action-space L-infinity bound *is*
        the Cartesian acceleration capacity.
        """
        return float(bounds.action_norm)


#: Default damping parameter for the damped-least-squares pseudo-inverse
#: in :class:`JacobianControlModel` (Nakamura & Hanafusa 1986). Keeps
#: the right-inverse well-defined near kinematic singularities; ADR-004
#: §Decision; spike_004A §Per-agent control model.
DEFAULT_DLS_DAMPING: float = 1e-3


@dataclass
class JacobianControlModel:
    """Jacobian-aware action ↔ Cartesian map skeleton (ADR-004 §Decision; ADR-007 Stage 1 AS).

    The 7-DOF arm case for the AS axis. Takes a Jacobian callable
    ``jacobian_fn(state) -> J`` (Cartesian-to-action shape
    ``(position_dim, action_dim)``) and a damped-least-squares
    pseudo-inverse for the right-inverse (Nakamura & Hanafusa 1986;
    the singularity-robust variant the OSCBF inner filter already
    relies on per Morton & Pavone 2025 §IV).

    Stage-1 AS spike will exercise this on the 7-DOF arm; the
    skeleton raises :class:`NotImplementedError` on
    :meth:`action_to_cartesian_accel` unless a Jacobian callable is
    supplied via ``jacobian_fn``. The same loud-fail discipline
    ADR-004 §Risks established for
    :class:`concerto.safety.emergency.JacobianEmergencyController`
    applies here: 7-DOF uids must not silently receive a Cartesian-
    shaped action by accident.

    Attributes:
        uid: Unique identifier of the agent.
        action_dim: Dimensionality of the action vector (e.g. 7 for
            a Franka arm).
        position_dim: Cartesian dimension of the safety body
            (typically 3 for manipulation).
        jacobian_fn: Optional callable mapping
            :class:`AgentSnapshot` to the Jacobian matrix of shape
            ``(position_dim, action_dim)``. When :data:`None`, both
            mapping methods raise :class:`NotImplementedError`.
        damping: Damped-least-squares parameter; default
            :data:`DEFAULT_DLS_DAMPING`.
        max_cartesian_accel_value: Pre-computed scalar capacity
            (kg-free acceleration units). Defaults to
            :data:`numpy.nan` so the placeholder fails loudly on
            attempted use; supply the embodiment-specific value at
            construction time.
    """

    uid: str
    action_dim: int
    position_dim: int
    jacobian_fn: Callable[[AgentSnapshot], FloatArray] | None = None
    damping: float = DEFAULT_DLS_DAMPING
    max_cartesian_accel_value: float = float("nan")

    def action_to_cartesian_accel(self, state: AgentSnapshot, action: FloatArray) -> FloatArray:
        """Apply ``J(state) @ action`` (ADR-004 §Decision; ADR-007 Stage 1 AS).

        Raises:
            NotImplementedError: When ``jacobian_fn`` was not supplied
                at construction time. Stage-1 AS spike delivers the
                concrete kinematic model; until then 7-DOF uids fail
                loudly per ADR-004 §Risks.
        """
        if self.jacobian_fn is None:
            msg = (
                f"JacobianControlModel(uid={self.uid!r}) has no jacobian_fn; "
                "Stage-1 AS spike (ADR-007 §Stage 1) supplies the kinematic "
                "model. Until then 7-DOF uids must not be routed through the "
                "CBF-QP."
            )
            raise NotImplementedError(msg)
        jac = np.asarray(self.jacobian_fn(state), dtype=np.float64)
        return (jac @ action.astype(np.float64, copy=False)).astype(np.float64, copy=False)

    def cartesian_accel_to_action(
        self, state: AgentSnapshot, cartesian_accel: FloatArray
    ) -> FloatArray:
        """Apply the damped-least-squares pseudo-inverse (ADR-004 §Decision; ADR-007 Stage 1 AS).

        ``J^T (J J^T + damping^2 I)^{-1} a`` — Nakamura & Hanafusa
        1986 singularity-robust right-inverse, identical in form to
        the OSCBF inner filter's Jacobian handling.

        Raises:
            NotImplementedError: When ``jacobian_fn`` was not supplied
                at construction time.
        """
        if self.jacobian_fn is None:
            msg = (
                f"JacobianControlModel(uid={self.uid!r}) has no jacobian_fn; "
                "Stage-1 AS spike (ADR-007 §Stage 1) supplies the kinematic "
                "model. Until then 7-DOF uids must not be routed through the "
                "CBF-QP."
            )
            raise NotImplementedError(msg)
        jac = np.asarray(self.jacobian_fn(state), dtype=np.float64)
        gram = jac @ jac.T + (self.damping * self.damping) * np.eye(
            self.position_dim, dtype=np.float64
        )
        rhs = cartesian_accel.astype(np.float64, copy=False)
        return (jac.T @ np.linalg.solve(gram, rhs)).astype(np.float64, copy=False)

    def max_cartesian_accel(self, bounds: Bounds) -> float:
        """Return the pre-computed Cartesian acceleration capacity (ADR-004 §Decision).

        Raises:
            ValueError: When ``max_cartesian_accel_value`` was left at
                its placeholder :data:`numpy.nan`. Stage-1 AS spike
                supplies the embodiment-specific value.
        """
        del bounds
        if not np.isfinite(self.max_cartesian_accel_value):
            msg = (
                f"JacobianControlModel(uid={self.uid!r}).max_cartesian_accel_value "
                "was not configured. Stage-1 AS spike (ADR-007 §Stage 1) supplies "
                "the embodiment-specific value."
            )
            raise ValueError(msg)
        return float(self.max_cartesian_accel_value)


@runtime_checkable
class EgoOnlySafetyFilter(Protocol):
    """Contract for an :attr:`SafetyMode.EGO_ONLY` safety-filter pipeline (ADR-004 §Public API).

    The deployment / ad-hoc / black-box-partner mode (ADR-006 §Decision;
    reviewer P0-3). The QP decides only the ego agent's action; the
    partner's motion enters as a predicted disturbance on the
    constraint right-hand side. The structural difference from
    :class:`JointSafetyFilter` is wire-visible: ``proposed_action`` is a
    single :class:`FloatArray` (the ego's nominal control), and the
    return value's first element is likewise an array — never a dict.

    Implementations compose the three layers from ADR-004 — exp CBF-QP
    backbone (Wang-Ames-Egerstedt 2017), conformal slack overlay
    (Huriot & Sibai 2025), OSCBF inner filter (Morton & Pavone 2025) —
    plus the hard braking fallback (ADR-004 risk-mitigation #1) into a
    single propose-check-replan boundary that wraps any nominal
    controller without accessing its internals (the property the
    black-box-partner setting requires per ADR-006 §Decision).

    The canonical implementation is
    :meth:`concerto.safety.cbf_qp.ExpCBFQP.ego_only`; third-party
    plugin filters that target ego-only deployment SHOULD subclass /
    structurally-conform to this Protocol so static checkers can
    catch shape mismatches at construction time (spike_004A
    §Three-mode taxonomy; reviewer P0-2).
    """

    def reset(self, *, seed: int | None = None) -> None:
        """Reset filter state at episode start (ADR-004 §Decision).

        Args:
            seed: Optional root seed for the filter's deterministic RNG
                substream (P6). Implementations route this through
                ``concerto.training.seeding.derive_substream`` so two
                CPU runs with the same seed produce byte-identical
                outputs.
        """
        ...

    def filter(
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
        """Project the nominal ego action onto the safe set (ADR-004 §Public API).

        Args:
            proposed_action: Ego nominal control input of shape
                ``(control_models[ego_uid].action_dim,)``. The QP
                minimises ``||u - u_hat||^2`` with this as the reference
                (Wang-Ames-Egerstedt 2017 eq. 9).
            obs: Observation dict; the filter reads ``obs["comm"]`` (M2
                contract — see ``chamber.comm.api.CommPacket``) and
                ``obs["meta"]["partner_id"]`` (M4 contract; ``None`` ⇒
                single-partner mode, no hard fail; identity change ⇒
                reset ``lambda`` to ``lambda_safe`` and enter warmup
                per ADR-004 risk-mitigation #2).
            state: Mutable :class:`SafetyState`; updated in place by the
                conformal layer each control step.
            bounds: Per-task :class:`Bounds` envelope (ADR-006 §Decision).
            ego_uid: Identifies which uid in ``obs["agent_states"]`` is
                the ego.
            partner_predicted_states: Per-partner-uid predicted
                :class:`concerto.safety.cbf_qp.AgentSnapshot` at the
                *next* control step, used to evaluate the partner's
                predicted Cartesian acceleration; this enters the
                constraint RHS as a known drift term.
            dt: Predictor lookahead horizon in seconds (typically the
                env's control-step duration). Used to convert the
                predicted velocity delta into a Cartesian acceleration
                ``(v_pred - v_now) / dt`` on the constraint RHS
                (ADR-004 §Decisions, "Predicted-acceleration units";
                external-review P0-1, 2026-05-16). Must be strictly
                positive and consistent with the value passed to the
                partner-trajectory predictor (e.g.
                :func:`concerto.safety.conformal.constant_velocity_predict`).

        Returns:
            A pair ``(safe_action, info)`` where ``safe_action`` is the
            QP-projected ego action and ``info`` is a
            :class:`FilterInfo` telemetry payload feeding the ADR-014
            three-table renderer.

        Raises:
            ConcertoSafetyInfeasible: When the QP is infeasible even after
                slack relaxation (ADR-004 §Decision; ADR-006 §Risks). The
                caller MUST route to the braking fallback in this case.
        """
        ...


@runtime_checkable
class JointSafetyFilter(Protocol):
    """Joint-action filter contract for CENTRALIZED / SHARED_CONTROL (ADR-004 §Public API).

    The oracle / ablation baseline mode plus the lab-only co-control
    mode (ADR-014 three-table report upper-bound row; reviewer P0-3
    mode 3). The QP decides every uid's action; per-uid slot widths
    come from ``control_models[uid].action_dim`` (no implicit "first
    uid's shape" fallback per reviewer P0-2). The structural
    difference from :class:`EgoOnlySafetyFilter` is wire-visible:
    ``proposed_action`` is a ``dict[uid, FloatArray]``, and the
    return value's first element is likewise a dict.

    The canonical implementations are
    :meth:`concerto.safety.cbf_qp.ExpCBFQP.centralized` and
    :meth:`concerto.safety.cbf_qp.ExpCBFQP.shared_control`.
    """

    def reset(self, *, seed: int | None = None) -> None:
        """Reset filter state at episode start (ADR-004 §Decision).

        See :meth:`EgoOnlySafetyFilter.reset` for the determinism
        contract (P6 seeding).
        """
        ...

    def filter(
        self,
        proposed_action: dict[str, FloatArray],
        obs: dict[str, object],
        state: SafetyState,
        bounds: Bounds,
        *,
        ego_uid: str | None = None,
        partner_action_bound: float | None = None,
    ) -> tuple[dict[str, FloatArray], FilterInfo]:
        """Project the per-uid nominal actions onto the safe set (ADR-004 §Public API).

        Args:
            proposed_action: Per-agent nominal control inputs, keyed by
                uid. The QP minimises ``||u - u_hat||^2`` with this as
                the reference (Wang-Ames-Egerstedt 2017 eq. 9).
            obs: Observation dict; same contract as
                :meth:`EgoOnlySafetyFilter.filter`.
            state: Mutable :class:`SafetyState`; updated in place by the
                conformal layer each control step.
            bounds: Per-task :class:`Bounds` envelope (ADR-006 §Decision).
            ego_uid: Required for :attr:`SafetyMode.SHARED_CONTROL`,
                ignored for :attr:`SafetyMode.CENTRALIZED`. Identifies
                which uid in ``proposed_action`` is the ego whose
                action is unconstrained by ``partner_action_bound``.
            partner_action_bound: Required for
                :attr:`SafetyMode.SHARED_CONTROL`, ignored for
                :attr:`SafetyMode.CENTRALIZED`. Strictly positive
                L-infinity ball radius around each non-ego uid's
                proposed action.

        Returns:
            A pair ``(safe_action, info)`` matching the input shape
            (one entry per uid in ``proposed_action``).

        Raises:
            ConcertoSafetyInfeasible: When the QP is infeasible even after
                slack relaxation (ADR-004 §Decision; ADR-006 §Risks).
        """
        ...


#: Deprecated union alias preserving the pre-refactor :data:`SafetyFilter`
#: name (ADR-004 §Public API). Resolves to
#: ``Union[EgoOnlySafetyFilter, JointSafetyFilter]``; the runtime
#: deprecation warning is emitted by the module-level :func:`__getattr__`
#: below. Targeted for removal in 0.3.0.
_SafetyFilterAlias = EgoOnlySafetyFilter | JointSafetyFilter

_DEPRECATED_NAMES: frozenset[str] = frozenset({"SafetyFilter"})


def __getattr__(name: str) -> object:
    """Emit a :class:`DeprecationWarning` on first use of the legacy alias (ADR-004 §Public API).

    The pre-refactor :data:`SafetyFilter` Protocol was a single
    dict-in / dict-out contract that the deployment-default
    :attr:`SafetyMode.EGO_ONLY` implementation did not satisfy
    (reviewer P0-2). The Protocol now splits into
    :class:`EgoOnlySafetyFilter` and :class:`JointSafetyFilter`;
    :data:`SafetyFilter` remains as a deprecated union alias and will
    be removed in 0.3.0.
    """
    if name in _DEPRECATED_NAMES:
        warnings.warn(
            "concerto.safety.api.SafetyFilter is deprecated as of the "
            "ADR-004 §Public API split (spike_004A §Three-mode taxonomy; "
            "reviewer P0-2); use EgoOnlySafetyFilter or JointSafetyFilter "
            "instead. Removal target: 0.3.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        return _SafetyFilterAlias
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


# Note: the deprecated :data:`SafetyFilter` alias is intentionally
# excluded from ``__all__`` so ``from concerto.safety.api import *``
# does not silently re-introduce the legacy name into downstream
# modules. The alias is still importable via the module-level
# :func:`__getattr__` shim (which emits the :class:`DeprecationWarning`).
__all__ = [
    "DEFAULT_DLS_DAMPING",
    "DEFAULT_EPSILON",
    "DEFAULT_ETA",
    "DEFAULT_WARMUP_STEPS",
    "AgentControlModel",
    "Bounds",
    "DoubleIntegratorControlModel",
    "EgoOnlySafetyFilter",
    "FilterInfo",
    "FloatArray",
    "JacobianControlModel",
    "JointSafetyFilter",
    "SafetyMode",
    "SafetyState",
    "canonical_pair_order",
]
