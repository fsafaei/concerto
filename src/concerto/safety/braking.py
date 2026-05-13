# SPDX-License-Identifier: Apache-2.0
"""Hard braking fallback (Wang-Ames-Egerstedt 2017 eq. 17; ADR-004 risk-mitigation #1).

ADR-004 §Decision risk-mitigation #1 (master plan risk register R1)
makes this load-bearing: the conformal CBF-QP carries an *average-loss*
guarantee (Huriot & Sibai 2025 Theorem 3), not per-step. In contact-rich
manipulation a single step of constraint violation can cause irreversible
hardware damage, so the QP alone is not a sufficient safety stack — a
per-step backstop that does not depend on QP feasibility is required.

This module is **not** routed through the QP. Plan/03 §8 + the M3
brief's Hard Rule #2 explicitly forbid "improving" the fallback by
passing through the QP — the whole point is independence from QP
feasibility. The override is computed from the kinematic state alone:

1. Compute per-pair time-to-collision ``ttc_ij`` from current positions
   and velocities (sphere-pair quadratic root).
2. If any ``ttc_ij < tau_brake``, accumulate the per-pair push-apart
   unit vectors per uid, then dispatch each uid's *aggregate*
   repulsion to its :class:`concerto.safety.emergency.EmergencyController`
   for embodiment-specific saturation. Signal ``fired=True`` iff at
   least one uid received an override.
3. Otherwise, return ``(None, False)`` — the QP path stays in charge.

The aggregation step is load-bearing: a uid in two simultaneous
dangerous pairs (e.g. uid_0 squeezed between uid_1 on the left and
uid_2 on the right) gets an override reflecting the *sum* of the
two push-apart unit vectors, not the last-processed pair only. The
prior "write override[uid] per pair" path lost superposition and was
non-deterministic across dict-iteration order (ADR-004
risk-mitigation #1).

The embodiment hook (the :class:`EmergencyController` Protocol)
turns the Cartesian-to-control mapping into a first-class concern.
The default :class:`CartesianAccelEmergencyController` is correct for
double-integrator agents whose control dim equals the Cartesian dim;
a 7-DOF arm needs a Jacobian-aware override (Stage-1 deliverable,
flagged in ADR-007 §"Stage 1 — Foundation axes" AS spike scope).

The ``fired`` flag propagates into ``FilterInfo["fallback_fired"]`` so
the ADR-014 three-table renderer's "fallback fired" column can count
events at the per-condition granularity (PR9 emits the column; M5/M6
spike runners populate the rows).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from concerto.safety.emergency import (
    CartesianAccelEmergencyController,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from concerto.safety.api import Bounds, FloatArray
    from concerto.safety.cbf_qp import AgentSnapshot
    from concerto.safety.emergency import EmergencyController

#: Default time-to-collision threshold in seconds. Plan/03 §2 Decision
#: row "Braking fallback trigger" pins 100 ms; configurable per task.
DEFAULT_TAU_BRAKE: float = 0.100

#: Numerical floor on relative velocity squared. Below this the
#: quadratic ``a`` coefficient is treated as zero ("no relative
#: motion") and the TTC is reported as ``+inf`` (ADR-004 §Decision).
_RELATIVE_VEL_FLOOR: float = 1e-12

#: Numerical floor on ``|Δp|`` so the push-apart unit vector is
#: well-defined when centres are nearly coincident (ADR-004 §Decision).
_NORM_FLOOR: float = 1e-9


def _pair_time_to_collision(snap_i: AgentSnapshot, snap_j: AgentSnapshot) -> float:
    """Solve the sphere-pair TTC quadratic (ADR-004 risk-mitigation #1).

    For relative position ``Dp = p_i - p_j``, relative velocity
    ``Dv = v_i - v_j`` and combined safety distance ``D_s``, collisions
    happen at the smaller positive root of

        ||Dp + t Dv||^2 = D_s^2
        ||Dv||^2 t^2 + 2 (Dp·Dv) t + (||Dp||^2 - D_s^2) = 0

    Returns ``0.0`` if already in collision (``||Dp|| <= D_s``);
    ``+inf`` if no positive real root (no collision on the current
    trajectory).
    """
    delta_p = snap_i.position - snap_j.position
    delta_v = snap_i.velocity - snap_j.velocity
    safety_distance = snap_i.radius + snap_j.radius

    c = float(np.dot(delta_p, delta_p)) - safety_distance * safety_distance
    if c <= 0.0:
        return 0.0  # Already inside the safety sphere.

    a = float(np.dot(delta_v, delta_v))
    if a < _RELATIVE_VEL_FLOOR:
        return float("inf")  # No relative motion ⇒ no collision.

    b = 2.0 * float(np.dot(delta_p, delta_v))
    discriminant = b * b - 4.0 * a * c
    if discriminant < 0.0:
        return float("inf")  # Trajectories don't intersect the sphere.

    sqrt_disc = float(np.sqrt(discriminant))
    t1 = (-b - sqrt_disc) / (2.0 * a)
    if t1 > 0.0:
        return t1
    t2 = (-b + sqrt_disc) / (2.0 * a)
    if t2 > 0.0:
        return t2
    return float("inf")  # Both roots non-positive — past closest approach.


def compute_min_ttc(
    snaps: dict[str, AgentSnapshot],
) -> dict[tuple[str, str], float]:
    """Per-pair minimum time-to-collision (ADR-004 risk-mitigation #1).

    Wang-Ames-Egerstedt 2017 eq. 17's prerequisite: the hybrid braking
    controller fires when the smallest pairwise TTC drops below
    ``tau_brake``. This function builds the per-pair map for use by
    :func:`maybe_brake`.

    Args:
        snaps: Per-agent kinematic state, keyed by uid (uses
            :class:`concerto.safety.cbf_qp.AgentSnapshot`).

    Returns:
        Mapping from ``(uid_i, uid_j)`` (lexicographic uid order
        preserved from input dict insertion) to TTC in seconds. Empty
        dict when fewer than two agents are present. ``0.0`` indicates
        already in collision; ``+inf`` indicates no collision on the
        current trajectory.
    """
    uids = list(snaps.keys())
    ttc: dict[tuple[str, str], float] = {}
    for i, uid_i in enumerate(uids):
        for uid_j in uids[i + 1 :]:
            ttc[(uid_i, uid_j)] = _pair_time_to_collision(snaps[uid_i], snaps[uid_j])
    return ttc


def _push_apart_unit_vector(snap_i: AgentSnapshot, snap_j: AgentSnapshot) -> FloatArray:
    """Unit vector from ``j`` toward ``i`` (ADR-004 risk-mitigation #1).

    On near-coincident centres the direction is ill-defined; falls back
    to ``+x`` so the override action stays deterministic and finite.
    """
    delta_p = snap_i.position - snap_j.position
    norm_dp = float(np.linalg.norm(delta_p))
    if norm_dp < _NORM_FLOOR:
        n_hat: FloatArray = np.zeros_like(snap_i.position, dtype=np.float64)
        n_hat[0] = 1.0
        return n_hat
    return (delta_p / norm_dp).astype(np.float64, copy=False)


def maybe_brake(
    proposed_action: dict[str, FloatArray],
    snaps: dict[str, AgentSnapshot],
    *,
    bounds: Bounds,
    tau_brake: float = DEFAULT_TAU_BRAKE,
    emergency_controllers: Mapping[str, EmergencyController] | None = None,
) -> tuple[dict[str, FloatArray] | None, bool]:
    """Per-step braking-fallback override (ADR-004 risk-mitigation #1).

    Wang-Ames-Egerstedt 2017 eq. 17 hybrid braking controller.

    **Bypasses the conformal QP by design** — plan/03 §8 + the M3
    brief's Hard Rule #2 forbid routing this through the QP, because
    the whole point is a per-step backstop that does not depend on QP
    feasibility. Theorem 3 (Huriot & Sibai 2025) is *average-loss*, not
    per-step; in contact-rich manipulation a single bad step can damage
    hardware.

    The override is computed in two phases. First, for every pair whose
    TTC is below ``tau_brake``, the Cartesian push-apart unit vector is
    appended to *each involved uid's* repulsion list (positive on uid_i,
    negative on uid_j). Second, each uid with at least one repulsion
    vector is dispatched to its
    :class:`concerto.safety.emergency.EmergencyController`, which
    aggregates the per-pair vectors and produces an embodiment-specific
    override action. uids with zero repulsion vectors retain
    ``proposed_action[uid]`` unchanged — no spurious override.

    The dispatch step is what makes the fallback embodiment-aware:
    double-integrator agents get the default
    :class:`concerto.safety.emergency.CartesianAccelEmergencyController`
    (Cartesian sum-and-saturate); 7-DOF arms get a Jacobian-aware
    controller (Stage-1 deliverable, see ADR-007 §"Stage 1 — Foundation
    axes" AS spike scope).

    Args:
        proposed_action: The QP-projected (or nominal) per-agent action;
            used as the no-override fallback for uids in zero dangerous
            pairs.
        snaps: Per-agent kinematic state at the current step.
        bounds: Per-task :class:`Bounds`; the per-uid override magnitude
            is capped at ``bounds.action_norm`` inside each
            controller's saturation step.
        tau_brake: TTC threshold in seconds (default
            :data:`DEFAULT_TAU_BRAKE` = 100 ms; ADR-004 §Decision row
            "Braking fallback trigger").
        emergency_controllers: Per-uid embodiment dispatch map. When
            :data:`None`, a default
            :class:`CartesianAccelEmergencyController` is built per uid;
            this preserves the toy-crossing behaviour for the Phase-0
            smoke tests. Callers integrating a 7-DOF arm uid MUST pass
            an entry that is *not* the Cartesian default (the
            :class:`JacobianEmergencyController` placeholder will raise
            until Stage-1 lands the real controller).

    Returns:
        ``(override, fired)``: when ``fired=True``, ``override`` is a
        dict matching the keys of ``proposed_action`` carrying the
        per-uid override actions (overridden or unchanged); the QP path
        is then skipped. When ``fired=False``, ``override`` is
        ``None`` and the QP solution stays in charge.
    """
    ttc_per_pair = compute_min_ttc(snaps)
    if not ttc_per_pair:
        return None, False
    if min(ttc_per_pair.values()) >= tau_brake:
        return None, False

    controllers: Mapping[str, EmergencyController] = (
        {uid: CartesianAccelEmergencyController() for uid in proposed_action}
        if emergency_controllers is None
        else emergency_controllers
    )

    repulsion_per_uid: dict[str, list[FloatArray]] = {uid: [] for uid in proposed_action}
    for (uid_i, uid_j), ttc in ttc_per_pair.items():
        if ttc < tau_brake:
            n_hat = _push_apart_unit_vector(snaps[uid_i], snaps[uid_j])
            repulsion_per_uid[uid_i].append(+n_hat)
            repulsion_per_uid[uid_j].append(-n_hat)

    override: dict[str, FloatArray] = {}
    fired = False
    for uid, action in proposed_action.items():
        vectors = repulsion_per_uid[uid]
        if not vectors:
            override[uid] = action
            continue
        override[uid] = controllers[uid].compute_override(snaps[uid], vectors, bounds)
        fired = True

    if not fired:
        return None, False
    return override, True


__all__ = [
    "DEFAULT_TAU_BRAKE",
    "compute_min_ttc",
    "maybe_brake",
]
