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
  resulting Cartesian acceleration to ``bounds.action_norm``, and
  returns an action whose shape matches the agent's Cartesian
  ``position.shape``. Correct for the toy double-integrator agents
  used in §V of Wang-Ames-Egerstedt 2017 and the Phase-0 smoke tests;
  **not** correct for 7-DOF arms.

- :class:`JacobianEmergencyController` — a placeholder that raises
  :class:`NotImplementedError`. The Jacobian-aware override for
  manipulator embodiments is a Stage-1 deliverable flagged in ADR-007
  §"Stage 1 — Foundation axes" AS spike scope (7-DOF arm vs 2-DOF
  diff-drive base). The placeholder is wired in so 7-DOF uids fail
  *loudly* at the fallback boundary rather than silently corrupting
  actions; the gap is documented but not faked.

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
    from concerto.safety.api import Bounds, FloatArray
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
            bounds: Per-task :class:`Bounds`; ``bounds.action_norm`` caps
                the override magnitude.

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
    2. Saturate the result to ``bounds.action_norm`` in the *net*
       push-apart direction: when the sum has non-trivial magnitude,
       rescale it to ``bounds.action_norm`` (max emergency push); when
       it cancels out, return zero.
    3. Return the saturated vector — its shape matches
       ``agent_state.position.shape`` by construction.

    Single-pair case (one dangerous partner): the sum is a single unit
    vector with norm 1; the output magnitude is ``bounds.action_norm``,
    matching the original toy-crossing contract. Two opposing pairs
    (uid squeezed in the middle): the sum cancels to ~zero and the
    output is a zero action — there is no preferred direction to push
    in, and a spurious push would be worse than a no-op. Two
    non-opposing pairs: the sum points in the net escape direction and
    the output is ``bounds.action_norm`` along that direction.

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
            bounds: Per-task :class:`Bounds`; ``bounds.action_norm`` is
                the saturated output magnitude. **Note: this is the L2
                magnitude cap consumer of the field with the documented
                L-infinity / L2 semantic inconsistency** (see
                :class:`concerto.safety.api.Bounds` "Known semantic
                inconsistency"; ADR-004 §Open questions; tracking
                issue #146). Operators wanting the CBF-side and
                emergency-side envelopes to agree should set
                ``action_norm = capacity / sqrt(d)`` (where ``d`` is
                the action dimension) until the field-split lands.

        Returns:
            Saturated Cartesian acceleration along the net push-apart
            direction with magnitude ``bounds.action_norm``; or zero
            when the pairwise vectors cancel out. Shape matches
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
        return ((aggregate / norm) * bounds.action_norm).astype(np.float64, copy=False)


class JacobianEmergencyController:
    """Jacobian-aware emergency override placeholder (ADR-004 risk-mitigation #1).

    7-DOF manipulator embodiments need a Cartesian-to-joint mapping
    (typically ``J^T(q)`` or ``J^+(q)``, where ``J`` is the end-effector
    Jacobian) to translate the aggregate repulsion into joint-space
    torques or velocity setpoints. That mapping requires the same
    full-dynamics model the OSCBF inner filter consumes (Morton & Pavone
    2025), so its arrival is sequenced with the Stage-1 AS spike scope
    in ADR-007 (7-DOF arm vs 2-DOF diff-drive base).

    The placeholder exists so the embodiment dispatch is wired in M3 and
    7-DOF uids fail *loudly* at the fallback boundary rather than
    silently corrupting joint commands with a Cartesian-shaped write.
    Stage-1 lands the real Jacobian-aware controller; the public API of
    this class is the contract that controller must satisfy.
    """

    def compute_override(
        self,
        agent_state: AgentSnapshot,
        pairwise_repulsion_vectors: list[FloatArray],
        bounds: Bounds,
    ) -> FloatArray:
        """Raise :class:`NotImplementedError` (ADR-004 risk-mitigation #1).

        Args:
            agent_state: Ignored; the placeholder never inspects state.
            pairwise_repulsion_vectors: Ignored.
            bounds: Ignored.

        Raises:
            NotImplementedError: Always. The Stage-1 AS spike delivers
                the real Jacobian-aware controller; until then 7-DOF
                uids must not be routed through the braking fallback.
        """
        del agent_state, pairwise_repulsion_vectors, bounds
        msg = (
            "Jacobian-aware emergency override is a Stage-1 deliverable; "
            "see ADR-004 risk-mitigation #1 follow-up."
        )
        raise NotImplementedError(msg)


__all__ = [
    "CartesianAccelEmergencyController",
    "EmergencyController",
    "JacobianEmergencyController",
]
