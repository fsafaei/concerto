# SPDX-License-Identifier: Apache-2.0
"""Embodiment-aware emergency-override controllers (ADR-004 risk-mitigation #1).

The hard braking fallback (Wang-Ames-Egerstedt 2017 eq. 17) is ADR-004
risk-mitigation #1's per-step backstop to Theorem 3's *average-loss*
guarantee (Huriot & Sibai 2025). The fallback computes a push-apart
direction in Cartesian space from each agent's kinematic snapshot, but
the action it writes lives in the agent's *control* space — and those
two spaces only coincide for double-integrator agents whose control
dimension equals the Cartesian dimension. For a 7-DOF arm the control
space is joint torques, and a Cartesian-shaped vector cannot be written
into a 7-vector slot without either crashing at integration time or
silently corrupting the action.

This module makes the embodiment hook explicit. :class:`EmergencyController`
is the per-uid Protocol that :func:`concerto.safety.braking.maybe_brake`
consults to translate the per-uid *aggregate* repulsion (sum of unit
vectors from every dangerous pair the uid is involved in) into a
control-space override action. Two implementations ship in Phase-0:

- :class:`CartesianAccelEmergencyController` — the default. Aggregates
  the pairwise Cartesian repulsion vectors by summation, saturates the
  resulting Cartesian acceleration to ``bounds.cartesian_accel_capacity``,
  and returns an action whose shape matches the agent's Cartesian
  ``position.shape``. Correct for the toy double-integrator agents
  used in §V of Wang-Ames-Egerstedt 2017 and the Phase-0 smoke tests;
  **not** correct for 7-DOF arms.

- :class:`JacobianEmergencyController` — the manipulator-aware override.
  Takes a :class:`~concerto.safety.api.JacobianControlModel` collaborator
  at construction and delegates the Cartesian-to-joint transform to its
  damped-pseudoinverse math (Nakamura & Hanafusa 1986 singularity-robust
  right-inverse), giving single-source-of-truth with the outer CBF row
  projection (``cbf_qp._agent_jacobian``). Saturation order: Cartesian
  L2 cap ``bounds.cartesian_accel_capacity`` applies **before** the
  Jacobian transform (operator-facing contract; matches the
  ``AgentControlModel`` right-inverse Protocol promise); per-joint
  L-infinity clip ``bounds.action_linf_component`` applies **after**
  the transform (hardware-faithful; the damped J^+ minimises joint L2
  norm, not L-infinity). The chamber-side wiring of a real Jacobian
  (panda URDF or SAPIEN-side derivative) lands in P1.03
  (``chamber.envs.stage1_pickplace``); P1.02 ships controller +
  synthetic-Jacobian-fixture tests.

The pairwise repulsion aggregation step is critical: prior to this
module, :func:`maybe_brake` wrote one override per dangerous pair, and
the last pair processed in dict-iteration order won. When uid_0 was in
two dangerous pairs in *opposite* directions, the surviving override
reflected only the last pair — losing superposition entirely. The
:class:`EmergencyController` contract takes a full ``list`` of
per-pair repulsion vectors so the aggregation strategy is a first-class
concern, not an accident of iteration order.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from concerto.safety.api import Bounds, FloatArray, JacobianControlModel
    from concerto.safety.cbf_qp import AgentSnapshot

#: Numerical floor on the aggregate Cartesian repulsion norm. Below this
#: the saturated direction is ill-defined (every pairwise unit vector
#: cancelled out), so the override falls back to a zero action — no
#: spurious push (ADR-004 risk-mitigation #1).
_AGGREGATE_NORM_FLOOR: float = 1e-12


@runtime_checkable
class EmergencyController(Protocol):
    """Per-uid emergency-override hook (ADR-004 risk-mitigation #1).

    Translates the aggregate per-uid repulsion (summed across every
    dangerous pair the uid is involved in) into a control-space override
    action. The braking fallback consults one controller per uid so
    heterogeneous embodiments (double-integrator base, 7-DOF arm,
    manipulator end-effector) can supply their own Cartesian-to-control
    mapping without :func:`concerto.safety.braking.maybe_brake` needing
    to know the embodiment class.

    Implementations MUST be deterministic on identical inputs (P6;
    seeding contract) and MUST return an array whose shape matches the
    control-action shape the env expects for this uid — *not*
    necessarily the Cartesian shape. A 7-DOF arm controller returns a
    7-vector; a 2D double-integrator controller returns a 2-vector.
    """

    def compute_override(
        self,
        agent_state: AgentSnapshot,
        pairwise_repulsion_vectors: list[FloatArray],
        bounds: Bounds,
    ) -> FloatArray:
        """Return the embodiment-specific emergency override (ADR-004 risk-mitigation #1).

        Args:
            agent_state: This uid's :class:`AgentSnapshot` at the current
                control step. Implementations may read position and
                velocity to build a Jacobian or any embodiment-specific
                mapping. The Cartesian default ignores it (the repulsion
                vectors already carry the direction information).
            pairwise_repulsion_vectors: List of unit-magnitude Cartesian
                repulsion vectors, one per dangerous pair this uid is
                involved in (sign convention: each vector points *away*
                from the partner in that pair). The list has at least
                one element — uids with zero dangerous pairs are not
                routed through this hook.
            bounds: Per-task :class:`Bounds`; how each field caps the
                override is up to the implementation. The Cartesian
                controller uses ``bounds.cartesian_accel_capacity`` to
                L2-cap the saturated direction; the Jacobian controller
                additionally uses ``bounds.action_linf_component`` for
                a post-transform per-joint clip.

        Returns:
            Override action in this uid's control space, ``float64``.
        """
        ...


class CartesianAccelEmergencyController:
    """Cartesian-acceleration emergency override (ADR-004 risk-mitigation #1).

    Default :class:`EmergencyController` for double-integrator agents
    whose control dimension equals the Cartesian dimension (2D and 3D
    point masses; the toy crossing example in Wang-Ames-Egerstedt 2017
    §V). The aggregation rule is:

    1. Sum the per-pair Cartesian repulsion unit vectors.
    2. Saturate the result to ``bounds.cartesian_accel_capacity`` in
       the *net* push-apart direction: when the sum has non-trivial
       magnitude, rescale it to ``bounds.cartesian_accel_capacity``
       (max emergency push); when it cancels out, return zero.
    3. Return the saturated vector — its shape matches
       ``agent_state.position.shape`` by construction.

    Single-pair case (one dangerous partner): the sum is a single unit
    vector with norm 1; the output magnitude is
    ``bounds.cartesian_accel_capacity``, matching the original
    toy-crossing contract. Two opposing pairs (uid squeezed in the
    middle): the sum cancels to ~zero and the output is a zero action
    — there is no preferred direction to push in, and a spurious push
    would be worse than a no-op. Two non-opposing pairs: the sum
    points in the net escape direction and the output is
    ``bounds.cartesian_accel_capacity`` along that direction.

    Not correct for 7-DOF manipulators: see
    :class:`JacobianEmergencyController` and the ADR-007 Stage-1 AS
    spike scope.
    """

    def compute_override(
        self,
        agent_state: AgentSnapshot,
        pairwise_repulsion_vectors: list[FloatArray],
        bounds: Bounds,
    ) -> FloatArray:
        """Sum-and-saturate Cartesian repulsion vectors (ADR-004 risk-mitigation #1).

        Args:
            agent_state: This uid's snapshot; only its ``position.shape``
                is used (the override's output shape).
            pairwise_repulsion_vectors: At least one Cartesian unit
                vector pointing away from each dangerous partner.
            bounds: Per-task :class:`Bounds`; the saturation magnitude
                is ``bounds.cartesian_accel_capacity`` (the L2 magnitude
                cap on Cartesian acceleration). Post-P1.02 the field
                split closes the prior L-infinity / L2 ambiguity that
                lived on the unified ``action_norm`` field — the
                per-component L-infinity envelope now lives on
                ``bounds.action_linf_component`` and is consumed only
                by the CBF-QP outer filter and (post-Jacobian) by
                :class:`JacobianEmergencyController`.

        Returns:
            Saturated Cartesian acceleration along the net push-apart
            direction with magnitude ``bounds.cartesian_accel_capacity``;
            or zero when the pairwise vectors cancel out. Shape matches
            ``agent_state.position.shape``, dtype ``float64``.
        """
        if not pairwise_repulsion_vectors:
            return np.zeros_like(agent_state.position, dtype=np.float64)

        aggregate = np.sum(
            np.stack(pairwise_repulsion_vectors, axis=0).astype(np.float64, copy=False),
            axis=0,
        )
        norm = float(np.linalg.norm(aggregate))
        if norm < _AGGREGATE_NORM_FLOOR:
            return np.zeros_like(agent_state.position, dtype=np.float64)
        return ((aggregate / norm) * bounds.cartesian_accel_capacity).astype(np.float64, copy=False)


class JacobianEmergencyController:
    """Jacobian-aware emergency override for manipulator embodiments (ADR-004 risk-mitigation #1).

    7-DOF manipulator embodiments need a Cartesian-to-joint mapping to
    translate the aggregate Cartesian repulsion into joint-space
    torques. This controller delegates the kinematic transform to a
    :class:`~concerto.safety.api.JacobianControlModel` collaborator
    supplied at construction, giving single-source-of-truth with the
    outer CBF row projection (``cbf_qp._agent_jacobian`` routes the
    constraint coefficient through the same model). The math is
    Nakamura & Hanafusa 1986's damped-least-squares pseudo-inverse
    ``J^+ = J^T (J J^T + damping^2 I)^{-1}``, singularity-robust for
    near-singular configurations.

    Saturation order (Plan-subagent design pass; P1.02):

    1. **Aggregate** the per-pair Cartesian repulsion unit vectors
       (same as :class:`CartesianAccelEmergencyController`); below
       :data:`_AGGREGATE_NORM_FLOOR` the direction is ill-defined and
       the controller returns a zero joint vector.
    2. **Cartesian L2 cap before transform** -- scale the aggregate
       to magnitude ``bounds.cartesian_accel_capacity``. The
       operator-facing L2 contract (matches the
       ``AgentControlModel.cartesian_accel_to_action`` right-inverse
       Protocol promise) is honoured at the input boundary so the
       Cartesian magnitude the CBF derivation reasoned about and the
       magnitude the emergency stage actually delivers agree.
    3. **Jacobian transform** -- delegate to
       ``control_model.cartesian_accel_to_action(state, cartesian)``.
       The damped-pseudoinverse math is reused, not duplicated.
    4. **Per-joint L-infinity clip after transform** -- element-wise
       clip the resulting joint torques to
       ``[-bounds.action_linf_component, +bounds.action_linf_component]``.
       Hardware-faithful (panda joint torque envelopes are per-joint;
       the damped J^+ minimises joint L2 norm, not L-infinity, so a
       Cartesian-feasible target near certain poses can still demand
       torques that exceed a joint's limit). The scalar
       ``action_linf_component`` is the **homogeneous per-joint
       envelope** for the embodiment; per-joint heterogeneous
       envelopes (e.g. the panda's (87, 87, 87, 87, 12, 12, 12) Nm
       URDF row) are a Phase-1 follow-up (extend ``Bounds`` with an
       optional per-joint vector; out of scope for P1.02).

    The chamber-side wiring (constructing a
    :class:`~concerto.safety.api.JacobianControlModel` with a real
    panda Jacobian from the SAPIEN ManiSkill v3 env) lives in P1.03
    (``chamber.envs.stage1_pickplace``); P1.02 tests against a
    synthetic Jacobian fixture (see
    ``tests/unit/test_jacobian_emergency_controller.py``).

    Attributes:
        control_model: The :class:`JacobianControlModel` collaborator
            carrying the per-uid ``jacobian_fn``, ``action_dim``,
            ``position_dim``, and damping factor. Same instance the
            outer CBF row projection uses, so the Jacobian definition
            (frame, sign convention, units) is configured once at
            chamber-side wiring time.
    """

    def __init__(self, *, control_model: JacobianControlModel) -> None:
        """Build the Jacobian-aware controller (ADR-004 risk-mitigation #1; P1.02).

        Args:
            control_model: A configured
                :class:`~concerto.safety.api.JacobianControlModel`
                with a non-``None`` ``jacobian_fn``. The fallback
                math (``cartesian_accel_to_action``) is delegated to
                this instance, so the same kinematic convention the
                outer CBF uses is what the emergency stage applies.
                The pyright-strict type annotation is the loud-fail
                contract; dynamic callers that pass ``None`` will
                raise ``AttributeError`` on the first
                ``self._control_model.action_dim`` access in
                :meth:`compute_override`.
        """
        self._control_model = control_model

    def compute_override(
        self,
        agent_state: AgentSnapshot,
        pairwise_repulsion_vectors: list[FloatArray],
        bounds: Bounds,
    ) -> FloatArray:
        """Aggregate -> Cartesian-cap -> damped J^+ -> per-joint clip (ADR-004 risk-mitigation #1).

        See the class docstring for the saturation order rationale.

        Args:
            agent_state: This uid's :class:`AgentSnapshot`. Passed
                verbatim to ``control_model.cartesian_accel_to_action``;
                the Jacobian callable consumes whatever the chamber-side
                wiring populates (joint positions for a panda, etc.).
            pairwise_repulsion_vectors: Cartesian unit vectors pointing
                away from each dangerous partner.
            bounds: Per-task :class:`Bounds`;
                ``bounds.cartesian_accel_capacity`` caps the pre-transform
                Cartesian magnitude; ``bounds.action_linf_component`` caps
                the post-transform per-joint torque magnitude.

        Returns:
            Joint-space override of shape ``(control_model.action_dim,)``,
            dtype ``float64``. Zero vector when the aggregate cancels.
        """
        if not pairwise_repulsion_vectors:
            return np.zeros(self._control_model.action_dim, dtype=np.float64)

        aggregate = np.sum(
            np.stack(pairwise_repulsion_vectors, axis=0).astype(np.float64, copy=False),
            axis=0,
        )
        norm = float(np.linalg.norm(aggregate))
        if norm < _AGGREGATE_NORM_FLOOR:
            return np.zeros(self._control_model.action_dim, dtype=np.float64)

        # Step 2: Cartesian L2 cap before the Jacobian transform.
        cartesian_target = (aggregate / norm) * bounds.cartesian_accel_capacity

        # Step 3: damped-pseudoinverse via the shared JacobianControlModel.
        joint_torques = self._control_model.cartesian_accel_to_action(agent_state, cartesian_target)

        # Step 4: per-joint L-infinity clip after the transform.
        return np.clip(
            joint_torques,
            -bounds.action_linf_component,
            +bounds.action_linf_component,
        ).astype(np.float64, copy=False)


__all__ = [
    "CartesianAccelEmergencyController",
    "EmergencyController",
    "JacobianEmergencyController",
]
