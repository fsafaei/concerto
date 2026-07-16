# SPDX-License-Identifier: Apache-2.0
# pyright: reportPrivateImportUsage=false
#
# Same suppression rationale as :mod:`chamber.envs.coinsert`: ``torch.zeros`` /
# ``torch.from_numpy`` are exported but not advertised in the torch stub's
# ``__all__``. Suppressed file-locally so the scene / telemetry logic stays
# free of per-line ``type: ignore`` noise.
r"""Co-hold-secure env — the Gate-0 fixation-tooling task (ADR-029 §Decision).

**STATUS: Gate-0 in progress (PR-A; ADR-029).** The v1.1 second-discriminating-
task candidate: the **holder** (partner, black-box) presents a part carrying a
shallow chamfered receptacle; the **securing operator** (ego) pushes a plug to
seat against a detent ("click") resistance over the final travel — the
fixtureless-fastening / connector-seat structure the task card sources. The
securing process load is what makes the coupling binding: the part is held
only by the holder, and the load axis is deliberately **non-vertical** so no
passive support can react it (ADR-029 §Decision — the two-robot-necessity
geometry).

**The wedge-inverted design rule (ADR-029 §Decision).** The co-insert closure
measured the geometry of its own failure: seating needs the relative plug-bore
tilt below the two-point wedge limit ``theta_wedge ~= arctan(per_side_clearance
/ engagement_depth)`` (< ~0.7 deg at 0.5 mm/side and 38 mm), while contact
itself cocks the plug to ~0.9-2.8 deg. This module inverts that into a
constructive constraint: the engagement depth (10 mm) and per-side clearance
band ({1.5, 2.5, 3.5} mm) are chosen so the wedge-limit tilt at full engagement
is at least **2x** the measured achievable-control ceiling (2.8 deg) on every
cell — 8.5 / 14.0 / 19.3 deg. :func:`cohold_wedge_limit_deg` and
:func:`cohold_design_window_ok` carry the rule as code.

Rig lineage (the banked co-insert instrument, ADR-029 §Rationale): the part is
a rigid **fixed link** of the holder articulation (never ``create_drive`` —
the S2 over-constraint lesson: 573-1306 N phantom preloads vs 0.2 N for the
fixed link), the plug is welded to the ego gripper by the two-anchor drive
attach, the cooperation-cost instrument is the friction-inclusive holder
workpiece-wrench (``get_link_incoming_joint_forces``), and the success
predicate machinery extends the co-insert conjunction with the **pose_held**
conjunct — the part must stay within the declared pose tolerance under the
securing load (the industrial requirement that makes a limp holder fail
honestly rather than cosmetically; ADR-029 §Consequences).

**The detent instrument.** The seat-click resistance is an analytic
action-reaction force pair applied inside the physics loop (plug pushed back,
part pushed forward, both along the bore axis) over the final
:data:`COHOLD_DETENT_TRAVEL_M` of travel — deliberately NOT a geometric
interference feature, because the S2 archives showed constraint-fidelity
artifacts when rigid geometry emulates what is physically a compliant feature
(ADR-029 §Risks). The precheck's P4 bound (monotone holder-wrench response to
a detent sweep, bounded peak) is the instrument check.

**Controls, constructible by flag** (the ``control`` factory argument;
ADR-029 §Decision; the precheck cells):

- ``"matched"`` — cooperative holder (fixed-link part; drive the holder seat
  with :class:`chamber.partners.coinsert_impedance.CoInsertReferenceHolder`).
- ``"limp"`` — identical build to ``"matched"``; the holder seat is driven by
  the registered zero-action instrument
  (:class:`chamber.partners.ablation.PartnerAblatedZero`) — grasp maintained,
  zero corrective action (the PR #298 C2-style coupling-liveness control).
- ``"none"`` — no holder participation: the holder arm is retracted, the part
  is a **free** actor resting on a passive stand at the nominal pose,
  unrestrained (the two-robot-necessity control; the securing load's lateral
  component exceeds what stand friction can react).
- ``"fixture"`` — the part is world-fixed (kinematic) at the nominal pose:
  the A2 passive-fixture rehearsal, reported honestly beside the team cells
  (ADR-029 §Decision, route (i)).

**Multi-pose hook (route (ii), dormant).** ``pose_sequence_len`` defaults to
1; values > 1 (the holder re-presents the part between securing operations,
which a static fixture cannot serve) raise ``NotImplementedError`` — enabling
the hook is a task change (version bump + ADR review; ADR-029 §Open questions).

Tier-1 / Tier-2 split (mirrors :mod:`chamber.envs.coinsert`): module-top-level
imports are Tier-1-safe; ManiSkill / SAPIEN imports happen inside
:func:`make_co_hold_secure_env` so ``python -c "import
chamber.envs.co_hold_secure"`` succeeds on a Vulkan-less host (ADR-001 §Risks
/ P2). The pure surface (control resolver, success predicate, wedge rule,
geometry helpers, constants) is Tier-1-tested; env construction and the
telemetry contract are Tier-2 (GPU-host integration tests).

References:
- ADR-029 §Decision / §Consequences / §Risks (this program; the design rule;
  the A2 posture; the staged gates).
- ADR-027 §Admission (the A1-A4 gates the task will face at PR-B).
- ADR-026 §Decision (coupling-validity criterion the precheck rehearses).
- ADR-009 §Decision (the frozen black-box partner contract for the holder).
- ADR-001 §Risks (wrapper-only; lazy SAPIEN import; no mesh assets).
- ADR-002 / P6 (determinism; :func:`derive_substream`).
- :mod:`chamber.envs.coinsert` (the banked rig this adapts).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple

import numpy as np

from chamber.envs.errors import ChamberEnvCompatibilityError
from concerto.training.seeding import derive_substream

if TYPE_CHECKING:  # pragma: no cover
    import gymnasium as gym
    import torch
    from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Frozen-able task parameters (ADR-029 §Decision). Declared as named module
# constants now (the co-insert / co-carry discipline); they are frozen at the
# Gate-0 pre-registration (PR-B, founder-signed tag) — nothing here is
# pre-registration-locked yet. The geometry triple (depth / clearance band /
# chamfer) and the detent numbers are the founder-confirmed anchors checked
# against the wedge derivation (ADR-029 §Decision).
# ---------------------------------------------------------------------------

#: Plug diameter, metres (16 mm — the co-insert peg scale; within Panda payload
#: and gripper span).
COHOLD_PLUG_DIAMETER_M: float = 0.016

#: Graded **per-side** clearance set, metres: {1.5, 2.5, 3.5} mm. NOTE the
#: convention change from :data:`chamber.envs.coinsert.COINSERT_CLEARANCE_SET_M`
#: (which is diametral): the wedge rule is stated per-side, so this module
#: declares per-side numbers directly (ADR-029 §Decision). The difficulty knob
#: stays physically meaningful and monotone (ADR-026 §Decision).
COHOLD_CLEARANCE_SIDE_SET_M: tuple[float, float, float] = (1.5e-3, 2.5e-3, 3.5e-3)

#: Engagement (seating) depth target, metres (10 mm — shallow by design: the
#: wedge-inverted rule trades the co-insert 40 mm bore for a seatable region
#: that contains the achievable-control region with margin; ADR-029 §Decision).
COHOLD_ENGAGE_DEPTH_M: float = 0.010

#: Receptacle lead-in chamfer, metres (2 mm x 45 deg — a generous funnel that,
#: with the clearance band, keeps the capture radius well above the base
#: securer's alignment tolerance; ADR-029 §Decision).
COHOLD_CHAMFER_M: float = 0.002

#: Detent ("seat-click") resistance, Newtons — the securing process load. The
#: resistance ramps linearly from 0 at the window entry to this peak just
#: before the click (a cam-ramp detent, not a step discontinuity), then
#: releases (ADR-029 §Decision). The precheck's P4 sweeps {20, 40, 60} N
#: around this anchor.
COHOLD_DETENT_FORCE_N: float = 40.0

#: Detent travel, metres — the final travel window over which the detent
#: resists; past it the plug is "clicked" home and the bore floor carries the
#: load (ADR-029 §Decision).
COHOLD_DETENT_TRAVEL_M: float = 0.002

#: Click release margin, metres: within this of the engagement depth the
#: detent latches free (the "click" — a snap-fit releases once the cam is
#: past; a resistance that never released would push the seated plug back out
#: against the securer's terminal hold). The latch re-arms only if the plug
#: backs fully out of the detent window. Smaller than
#: :data:`COHOLD_DEPTH_EPS_M` so ``seated`` still certifies a genuine
#: click-through.
COHOLD_DETENT_CLICK_EPS_M: float = 0.0003

#: Part (held workpiece) mass, kilograms (the co-carry / co-insert scale).
COHOLD_PART_MASS_KG: float = 0.5

#: Part outer half-extent around the bore axis, metres (the receptacle ring).
COHOLD_PART_OUTER_HALF_M: float = 0.030

#: Bore floor thickness, metres (a *blind* shallow receptacle — the plug
#: bottoms out exactly at the engagement depth, which is what makes the
#: "click" boundary stable; ADR-029 §Risks).
COHOLD_PART_FLOOR_M: float = 0.008

#: Securing-axis tilt from vertical, degrees. **Deliberately non-vertical**
#: (ADR-029 §Decision): a vertical push onto a passively supported part is
#: braced by the support, which would defeat the two-robot-necessity control
#: by construction. At 60 deg the detent load's lateral component (F sin 60)
#: exceeds stand friction (mu (F cos 60 + m g)) for any mu < ~1.4.
COHOLD_AXIS_TILT_FROM_VERTICAL_DEG: float = 60.0

#: Coulomb friction on the plug-part pair (the frozen co-insert value; one
#: declared contact model).
COHOLD_PLUG_PART_FRICTION: float = 0.5

# ---------------------------------------------------------------------------
# The wedge-inverted design rule (ADR-029 §Decision) — the S2-measured inputs
# and the margin requirement, as code.
# ---------------------------------------------------------------------------

#: The measured achievable-control tilt ceiling under contact, degrees — the
#: upper end of the co-insert S2 ~0.9-2.8 deg contact-cocking band
#: (``spikes/results/coinsert/s2/``; ADR-029 §Decision). An S2-measured input,
#: never re-tuned here.
COHOLD_ACHIEVABLE_TILT_CEILING_DEG: float = 2.8

#: Required design-window margin: the wedge-limit tilt at full engagement must
#: be at least this multiple of the achievable-control ceiling (ADR-029
#: §Decision).
COHOLD_WEDGE_MARGIN_MIN: float = 2.0

# ---------------------------------------------------------------------------
# Success predicate thresholds (ADR-029 §Consequences; the co-insert
# ``seated ∧ within_force ∧ static ∧ settled`` pattern + the new pose_held
# conjunct). Force budgets are ``None`` placeholders — derived from measured
# matched-pair distributions at Gate-0 and frozen by the signed prereg, never
# asserted from intuition (the co-insert / co-carry derivation discipline).
# ---------------------------------------------------------------------------

#: Seating depth tolerance, metres. Deliberately SMALLER than
#: :data:`COHOLD_DETENT_TRAVEL_M` so ``seated`` certifies the click was pushed
#: through, not merely approached (ADR-029 §Consequences).
COHOLD_DEPTH_EPS_M: float = 0.0005

#: Seating axis-alignment tolerance, degrees (the co-insert convention).
COHOLD_AXIS_ALIGN_TOL_DEG: float = 5.0

#: ``pose_held`` translation tolerance, metres: the part origin must stay
#: within this of its settled reference pose under the securing load. Sits
#: between the matched holder's corrected excursion and a limp holder's
#: uncorrected escape under the detent load (ADR-029 §Consequences); frozen at
#: Gate-0.
COHOLD_POSE_HELD_TRANS_TOL_M: float = 0.005

#: ``pose_held`` tilt tolerance, degrees (same reference; same freeze).
COHOLD_POSE_HELD_TILT_TOL_DEG: float = 5.0

#: Peak plug-part (securing) contact-force budget, Newtons — derived from the
#: matched-pair success distribution at Gate-0; ``None`` until then.
COHOLD_SECURE_FORCE_MAX_N: float | None = None

#: Peak holder workpiece-wrench budget, Newtons — the stress channel budget;
#: derived at Gate-0; ``None`` until then.
COHOLD_COUPLE_FORCE_MAX_N: float | None = None

#: Joint-velocity threshold (rad/s) for the per-arm ``is_static`` test.
COHOLD_STATIC_QVEL_THRESH: float = 0.2

#: Settle window, env ticks, excluded from force maxima, pose_held tracking,
#: and success eligibility (the co-carry settle discipline).
COHOLD_SETTLE_WINDOW_STEPS: int = 15

#: Part linear-velocity threshold (m/s) for the ``settled`` conjunct.
COHOLD_SETTLE_VEL_THRESH: float = 0.05

# ---------------------------------------------------------------------------
# Control-rate / horizon (the co-insert quasi-static band).
# ---------------------------------------------------------------------------

#: Ego control rate, Hz.
COHOLD_CONTROL_HZ: int = 20

#: Control mode — the CHAMBER / co-carry / co-insert control surface.
COHOLD_CONTROL_MODE: str = "pd_joint_delta_pos"

#: Episode horizon bounds + default, env ticks (quasi-static press).
COHOLD_EPISODE_LENGTH_MIN: int = 250
COHOLD_EPISODE_LENGTH_MAX: int = 320
COHOLD_DEFAULT_EPISODE_LENGTH: int = 320

# ---------------------------------------------------------------------------
# Robot / scene wiring. Ready poses solved offline on the ADR-004 FK chain
# (:class:`chamber.agents.panda_jacobian.PandaJacobianProvider`) for the
# 60-deg securing axis; verified live at bring-up. Recompute rule: solve the
# planar family ``j6 = tilt - j4 + j2`` (hand approach axis in the x-z plane)
# for the TCP targets below, then re-derive the part fixed-joint origin from
# the live holder hand pose (the S2-documented procedure).
# ---------------------------------------------------------------------------

#: Number of Panda arm DOF (excludes the two gripper fingers).
_PANDA_ARM_DOF: int = 7

#: Ego (securing operator) uid — the Panda incumbent.
_EGO_UID: str = "panda_wristcam"

#: Holder uid — the no-camera ``panda_v2.urdf`` partner body.
_PARTNER_UID: str = "panda_partner"

#: Panda hand link name (grip frame; wrench instrument anchor).
_HAND_LINK_NAME: str = "panda_hand"

#: ``panda_hand_tcp`` offset down the hand z-axis, metres (mani-skill==3.0.1).
_GRIP_OFFSET_Z_M: float = 0.1034

#: Half the base-to-base separation along world x. Ego at ``(-x, 0, 0)``
#: facing +x; holder at ``(+x, 0, 0)`` yawed pi.
_BASE_X_M: float = 0.62

#: Substream name for the env's deterministic RNG (P6 / ADR-002).
_SUBSTREAM_NAME: str = "env.co_hold_secure"

#: Ego ready qpos (7 arm + 2 fingers): forward reach, hand approach axis on
#: the 60-deg securing axis, plug tip ~40 mm short of the part mouth.
_PANDA_READY_QPOS_EGO: NDArray[np.float64] = np.array(
    [0.0, -0.2992, 0.0, -2.6496, 0.0, 3.3976, 0.785, 0.04, 0.04]
)

#: Holder ready qpos: mirrored forward reach; the part rides a straight
#: fixed-link stem along the hand approach axis, mouth toward the ego.
_PANDA_READY_QPOS_HOLDER: NDArray[np.float64] = np.array(
    [0.0, -0.3606, 0.0, -2.6185, 0.0, 3.3051, 0.785, 0.04, 0.04]
)

#: Retracted qpos for the no-holder controls (folded back out of the
#: workspace; the co-insert positive-control pose).
_PANDA_RETRACTED_QPOS: NDArray[np.float64] = np.array(
    [0.0, -np.pi / 4, 0.0, -np.pi * 7 / 8, 0.0, np.pi * 5 / 8, -np.pi / 4, 0.04, 0.04]
)

#: Plug weld (ego): identity at the grip offset — plug local +z (the securing
#: tip) extends along the gripper approach axis.
_PLUG_GRIP_IN_HAND_P: tuple[float, float, float] = (0.0, 0.0, _GRIP_OFFSET_Z_M)
_PLUG_GRIP_IN_HAND_Q: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

#: Part fixed-joint origin (holder hand frame): a straight 0.2234 m stem down
#: the hand approach axis, with a 60-deg pitch flip so the bore opening (part
#: local +z) faces back up the securing axis toward the ego. **Tied to the two
#: ready qposes** — recompute per the module recompute rule if they change.
_PART_GRIP_IN_HAND_P: tuple[float, float, float] = (0.0, 0.0, 0.2234)
_PART_GRIP_IN_HAND_Q: tuple[float, float, float, float] = (0.8660254, 0.0, 0.5, 0.0)

#: Span (metres) between the two plug weld anchors along the plug axis (the
#: S2 two-anchor orientation-locking attach).
_WELD_ANCHOR_SPAN_M: float = 0.06

#: Name of the part link injected into the holder URDF (fixed-link attach).
_PART_LINK_NAME: str = "cohold_part"

#: Number of tangential wall facets approximating the round bore (S2 round
#: geometry — the cylinder plug self-centres and frees yaw).
_ROUND_BORE_FACETS: int = 12

#: Nominal part world pose ``(xyz, quat wxyz)`` — the fixed-link part pose at
#: the holder ready qpos (mouth at the origin of the part frame; opening
#: toward the ego, 60 deg from vertical). The ``"none"`` / ``"fixture"``
#: controls place the free / world-fixed part actor here so all four controls
#: share one nominal presentation. Derived from the ready poses; verified live
#: at bring-up (same recompute rule).
_NOMINAL_PART_POS_W: tuple[float, float, float] = (-0.02504, 0.0, 0.38901)
_NOMINAL_PART_QUAT_W: tuple[float, float, float, float] = (0.8660254, 0.0, -0.5, 0.0)

#: Stand (the ``"none"`` control): footprint half-extents of the kinematic
#: pedestal the free part rests on, and the pad half-extents of the part's
#: flat-bottom base pad (world-axis-aligned at the nominal pose).
_STAND_HALF_XY_M: tuple[float, float] = (0.06, 0.05)
_PAD_HALF_M: tuple[float, float, float] = (0.045, 0.035, 0.008)

#: Radial gap between the receptacle ring's outer surface and the base pad's
#: inner face along world-down at the nominal pose.
_PAD_RING_GAP_M: float = 0.0


def cohold_bore_inner_radius(
    clearance_side_m: float,
    *,
    plug_diameter_m: float = COHOLD_PLUG_DIAMETER_M,
) -> float:
    """Bore inner radius for a given per-side clearance (ADR-029 §Decision).

    The bore is a round (N-gon facet) cavity around a cylindrical plug; the
    inscribed radius is ``plug_radius + clearance_side_m`` so the per-side gap
    equals the declared per-side clearance directly (NOTE: per-side, unlike
    the co-insert diametral convention). Pure Tier-1 function; the URDF part
    builder and the actor part builder both derive their geometry from it.

    Args:
        clearance_side_m: Per-side clearance, metres.
        plug_diameter_m: Plug diameter (default :data:`COHOLD_PLUG_DIAMETER_M`).

    Returns:
        The bore inscribed radius, metres.
    """
    return float(plug_diameter_m) / 2.0 + float(clearance_side_m)


def cohold_wedge_limit_deg(
    clearance_side_m: float,
    engagement_depth_m: float = COHOLD_ENGAGE_DEPTH_M,
) -> float:
    """Two-point wedge-limit tilt, degrees (ADR-029 §Decision — the design rule).

    The empirically-consistent form of the co-insert S2 tilt-wedge finding:
    ``theta_wedge ~= arctan(per_side_clearance / engagement_depth)`` (the S2
    archives record seatable tilt < ~0.7 deg at 0.5 mm/side and 38 mm;
    ``arctan(0.5/38) = 0.75 deg``). Pure Tier-1 function.

    Args:
        clearance_side_m: Per-side clearance, metres.
        engagement_depth_m: Engagement depth, metres (default
            :data:`COHOLD_ENGAGE_DEPTH_M`).

    Returns:
        The wedge-limit tilt at full engagement, degrees.
    """
    return float(np.degrees(np.arctan2(float(clearance_side_m), float(engagement_depth_m))))


def cohold_design_window_ok(
    clearance_side_m: float,
    engagement_depth_m: float = COHOLD_ENGAGE_DEPTH_M,
    *,
    achievable_ceiling_deg: float = COHOLD_ACHIEVABLE_TILT_CEILING_DEG,
    margin_min: float = COHOLD_WEDGE_MARGIN_MIN,
) -> bool:
    """Whether a (clearance, depth) cell sits in the design window (ADR-029 §Decision).

    The wedge-inverted rule: the seatable region must contain the
    achievable-control region with margin —
    ``wedge_limit >= margin_min * achievable_ceiling``. Every cell of
    :data:`COHOLD_CLEARANCE_SIDE_SET_M` at :data:`COHOLD_ENGAGE_DEPTH_M`
    satisfies it (8.5 / 14.0 / 19.3 deg >= 5.6 deg); Tier-1 tests pin this so
    a geometry edit that silently leaves the window fails CI. Pure Tier-1
    function.

    Args:
        clearance_side_m: Per-side clearance, metres.
        engagement_depth_m: Engagement depth, metres.
        achievable_ceiling_deg: The S2-measured achievable-control tilt
            ceiling (default :data:`COHOLD_ACHIEVABLE_TILT_CEILING_DEG`).
        margin_min: Required margin multiple (default
            :data:`COHOLD_WEDGE_MARGIN_MIN`).

    Returns:
        ``True`` iff the cell is inside the design window.
    """
    limit = cohold_wedge_limit_deg(clearance_side_m, engagement_depth_m)
    return limit >= float(margin_min) * float(achievable_ceiling_deg)


class CoHoldControl(NamedTuple):
    """Resolved co-hold-secure control configuration (ADR-029 §Decision).

    Pure-Python NamedTuple so Tier-1 tests construct / assert without SAPIEN.
    The four controls are the precheck cells; the holder *identity* (matched
    vs limp) enters through the :class:`chamber.partners.api.FrozenPartner`
    interface (ADR-009), not through the env build — ``"matched"`` and
    ``"limp"`` are the identical rig.

    Attributes:
        control_id: Verbatim control string.
        holder_active: Whether the holder arm participates (fixed-link part).
        part_mode: ``"fixed_link"`` (part is a holder articulation link),
            ``"free_stand"`` (free actor resting on the passive stand), or
            ``"world_fixed"`` (kinematic actor — the A2 rehearsal).
        expected_partner: The partner class expected to drive the holder seat
            (documentation for drivers; the env does not enforce it).
    """

    control_id: str
    holder_active: bool
    part_mode: str
    expected_partner: str


#: control_id -> resolved configuration (the four precheck cells; ADR-029).
_CONTROL_TABLE: dict[str, CoHoldControl] = {
    "matched": CoHoldControl(
        control_id="matched",
        holder_active=True,
        part_mode="fixed_link",
        expected_partner="coinsert_reference_holder",
    ),
    "limp": CoHoldControl(
        control_id="limp",
        holder_active=True,
        part_mode="fixed_link",
        expected_partner="partner_ablated_zero",
    ),
    "none": CoHoldControl(
        control_id="none",
        holder_active=False,
        part_mode="free_stand",
        expected_partner="partner_ablated_zero",
    ),
    "fixture": CoHoldControl(
        control_id="fixture",
        holder_active=False,
        part_mode="world_fixed",
        expected_partner="partner_ablated_zero",
    ),
}


def resolve_cohold_control(control: str) -> CoHoldControl:
    """Resolve a co-hold-secure ``control`` flag to its config (ADR-029 §Decision).

    Pure Tier-1 function — no SAPIEN dependency.

    Args:
        control: One of ``"matched"`` / ``"limp"`` / ``"none"`` /
            ``"fixture"`` (the precheck cells).

    Returns:
        The resolved :class:`CoHoldControl`.

    Raises:
        ValueError: If ``control`` is unknown. The message lists the valid
            options and cites ADR-029 §Decision.
    """
    try:
        return _CONTROL_TABLE[control]
    except KeyError as exc:
        msg = (
            f"CoHoldSecureEnv: control {control!r} is not one of the co-hold-secure "
            f"controls {sorted(_CONTROL_TABLE)!r} (ADR-029 §Decision — the precheck cells)."
        )
        raise ValueError(msg) from exc


def evaluate_cohold_success(
    *,
    seated_depth_m: NDArray[np.floating] | float,
    axis_align_deg: NDArray[np.floating] | float,
    pose_excursion_m: NDArray[np.floating] | float,
    pose_tilt_deg: NDArray[np.floating] | float,
    peak_secure_force_n: NDArray[np.floating] | float,
    peak_couple_wrench_n: NDArray[np.floating] | float,
    both_static: NDArray[np.bool_] | bool,
    settled: NDArray[np.bool_] | bool,
    f_secure_max: float,
    f_couple_max: float,
    depth_target_m: float = COHOLD_ENGAGE_DEPTH_M,
    depth_eps_m: float = COHOLD_DEPTH_EPS_M,
    axis_align_tol_deg: float = COHOLD_AXIS_ALIGN_TOL_DEG,
    pose_trans_tol_m: float = COHOLD_POSE_HELD_TRANS_TOL_M,
    pose_tilt_tol_deg: float = COHOLD_POSE_HELD_TILT_TOL_DEG,
) -> NDArray[np.bool_]:
    """Joint co-hold-secure success predicate (ADR-029 §Consequences).

    Success iff **all** of (the co-insert conjunction + the new pose_held
    conjunct):

    - **seated**: plug-tip depth >= ``depth_target_m - depth_eps_m`` (with
      ``depth_eps_m`` < the detent travel, so seated certifies the click) AND
      plug-bore axes aligned within ``axis_align_tol_deg``;
    - **pose_held**: the part's maximum post-settle pose excursion from its
      settled reference stays within ``pose_trans_tol_m`` /
      ``pose_tilt_tol_deg`` under the securing load — the conjunct that makes
      a limp holder fail honestly (part-pose escape) even if a plug still
      finds the bore;
    - **within_force**: peak securing contact force <= ``f_secure_max`` AND
      peak holder workpiece-wrench <= ``f_couple_max``;
    - **static**: participating arms below the static qvel threshold;
    - **settled**: past the settle window with the part quasi-stationary.

    ``f_secure_max`` / ``f_couple_max`` are **required** (no default): the
    module placeholders are ``None`` until the Gate-0 derivation, so callers
    must pass measured values (or ``inf`` for a raw-distribution run) —
    nothing here asserts an unmeasured number. Pure Tier-1 function.

    Args:
        seated_depth_m: Plug-tip insertion depth, metres.
        axis_align_deg: Plug-to-bore axis misalignment, degrees.
        pose_excursion_m: Max post-settle part translation excursion, metres.
        pose_tilt_deg: Max post-settle part tilt excursion, degrees.
        peak_secure_force_n: Episode-peak plug-part contact force, Newtons.
        peak_couple_wrench_n: Episode-peak holder workpiece-wrench, Newtons.
        both_static: Whether the participating arms are static.
        settled: Whether the part is settled past the settle window.
        f_secure_max: Securing contact-force budget (required).
        f_couple_max: Workpiece-wrench budget (required).
        depth_target_m: Engagement depth target.
        depth_eps_m: Seating depth tolerance.
        axis_align_tol_deg: Seating alignment tolerance.
        pose_trans_tol_m: pose_held translation tolerance.
        pose_tilt_tol_deg: pose_held tilt tolerance.

    Returns:
        Boolean array (broadcast of the inputs) — the per-env success.
    """
    seated = (np.asarray(seated_depth_m) >= (depth_target_m - depth_eps_m)) & (
        np.asarray(axis_align_deg) <= axis_align_tol_deg
    )
    pose_held = (np.asarray(pose_excursion_m) <= pose_trans_tol_m) & (
        np.asarray(pose_tilt_deg) <= pose_tilt_tol_deg
    )
    within_force = (np.asarray(peak_secure_force_n) <= f_secure_max) & (
        np.asarray(peak_couple_wrench_n) <= f_couple_max
    )
    static = np.asarray(both_static, dtype=bool)
    is_settled = np.asarray(settled, dtype=bool)
    return np.asarray(seated & pose_held & within_force & static & is_settled, dtype=bool)


def _rot_from_quat_wxyz(q: tuple[float, float, float, float]) -> NDArray[np.float64]:
    """Rotation matrix from a wxyz quaternion (SAPIEN convention; pure NumPy)."""
    qa = np.asarray(q, dtype=np.float64)
    w, x, y, z = qa / max(float(np.linalg.norm(qa)), 1e-12)
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


#: Trace threshold below which the quaternion conversion switches to the
#: dominant-diagonal branch (numerical stability near a pi rotation).
_QUAT_TRACE_SWITCH: float = -0.99


def _quat_wxyz_from_rot(rot: NDArray[np.float64]) -> tuple[float, float, float, float]:
    """Build the wxyz quaternion of a rotation matrix (stable for rig frames)."""
    t = float(np.trace(rot))
    if t > _QUAT_TRACE_SWITCH:
        w = float(np.sqrt(max(0.0, 1.0 + t))) / 2.0
        x = float(rot[2, 1] - rot[1, 2]) / (4.0 * w)
        y = float(rot[0, 2] - rot[2, 0]) / (4.0 * w)
        z = float(rot[1, 0] - rot[0, 1]) / (4.0 * w)
        return (w, x, y, z)
    # trace near -1: pick the dominant diagonal axis (rig frames never hit this
    # in practice, but keep the conversion total).
    i = int(np.argmax(np.diag(rot)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = float(np.sqrt(max(0.0, 1.0 + rot[i, i] - rot[j, j] - rot[k, k]))) * 2.0
    axis = [0.0, 0.0, 0.0]
    axis[i] = s / 4.0
    axis[j] = float(rot[j, i] + rot[i, j]) / s
    axis[k] = float(rot[k, i] + rot[i, k]) / s
    w = float(rot[k, j] - rot[j, k]) / s
    return (w, axis[0], axis[1], axis[2])


def _quat_wxyz_to_rpy(q: tuple[float, float, float, float]) -> tuple[float, float, float]:
    """URDF roll-pitch-yaw (extrinsic XYZ) from a wxyz quaternion (S2 helper)."""
    w, x, y, z = q
    roll = float(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
    pitch = float(np.arcsin(max(-1.0, min(1.0, 2 * (w * y - z * x)))))
    yaw = float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    return roll, pitch, yaw


def _part_boxes(
    clearance_side_m: float,
) -> list[tuple[list[float], list[float], tuple[float, float, float, float]]]:
    """Part geometry as convex boxes: floor + bore ring + chamfer + base pad.

    Returns ``(half_size[xyz], centre[xyz], quat_wxyz)`` in the **part frame**
    (mouth at the origin; bore axis = local +z pointing out of the mouth;
    cavity descends to ``-COHOLD_ENGAGE_DEPTH_M``; the 45-deg chamfer flares
    above the mouth). Composed from convex primitives (a cavity is non-convex;
    ADR-001 §Risks — no mesh asset), shared by the fixed-link URDF part and
    the actor part so all controls are dimensionally identical:

    - the bore **floor** at exactly the engagement depth (the plug bottoms out
      there — the stable "clicked" boundary);
    - :data:`_ROUND_BORE_FACETS` tangential **wall** facets of inscribed
      radius :func:`cohold_bore_inner_radius` (per-side clearance);
    - the same count of **chamfer** facets, tilted 45 deg, flaring from the
      bore radius at the mouth to +:data:`COHOLD_CHAMFER_M` above it;
    - a **base pad** whose faces are world-axis-aligned at the nominal part
      pose (world-down is a known fixed direction in the part frame), giving
      the free part a flat, statically stable rest on the ``"none"`` stand.
    """
    r_in = cohold_bore_inner_radius(clearance_side_m)
    w_out = COHOLD_PART_OUTER_HALF_M
    depth = COHOLD_ENGAGE_DEPTH_M
    t_floor = COHOLD_PART_FLOOR_M
    n = _ROUND_BORE_FACETS
    t = w_out - r_in  # radial wall thickness
    half_seg = (r_in + t) * float(np.tan(np.pi / n)) + 0.002
    out: list[tuple[list[float], list[float], tuple[float, float, float, float]]] = [
        (
            [w_out, w_out, t_floor / 2.0],
            [0.0, 0.0, -depth - t_floor / 2.0],
            (1.0, 0.0, 0.0, 0.0),
        ),
    ]
    for i in range(n):
        theta = 2.0 * np.pi * i / n
        c, s = float(np.cos(theta)), float(np.sin(theta))
        yaw_quat = (float(np.cos(theta / 2.0)), 0.0, 0.0, float(np.sin(theta / 2.0)))
        # Wall facet: thin box tangent to the bore circle, spanning the cavity.
        r_c = r_in + t / 2.0
        out.append(([t / 2.0, half_seg, depth / 2.0], [r_c * c, r_c * s, -depth / 2.0], yaw_quat))
        # Chamfer facet: the same tangential slab tilted 45 deg about the
        # local tangential axis so its inner face is the lead-in cone flaring
        # from r_in at the mouth to r_in + chamfer at +chamfer above it. The
        # facet frame is Rz(theta) @ Ry(45 deg); its centre sits half the slab
        # thickness outward-normal of the mid-slant point.
        ch = COHOLD_CHAMFER_M
        slant_half = ch * float(np.sqrt(2.0)) / 2.0 + 0.001
        t_ch = 0.004
        rz = _rot_from_quat_wxyz(yaw_quat)
        ry45 = _rot_from_quat_wxyz(
            (float(np.cos(np.pi / 8.0)), 0.0, float(np.sin(np.pi / 8.0)), 0.0)
        )
        r_facet = rz @ ry45
        mid_slant = np.array([r_in + ch / 2.0, 0.0, ch / 2.0], dtype=np.float64)
        # outward normal of the 45-deg cone surface in the yaw-0 frame:
        n_out = np.array([1.0, 0.0, -1.0], dtype=np.float64) / np.sqrt(2.0)
        centre = rz @ (mid_slant + (t_ch / 2.0) * n_out)
        out.append(
            (
                [t_ch / 2.0, half_seg, slant_half],
                [float(centre[0]), float(centre[1]), float(centre[2])],
                _quat_wxyz_from_rot(r_facet),
            )
        )
    # Base pad: world-axis-aligned at the nominal pose. World-down in the part
    # frame is fixed by the nominal orientation; the pad hangs off the ring's
    # world-underside with its bottom face horizontal.
    r_nom = _rot_from_quat_wxyz(_NOMINAL_PART_QUAT_W)
    down_local = r_nom.T @ np.array([0.0, 0.0, -1.0], dtype=np.float64)
    ring_centre = np.array([0.0, 0.0, -(depth + t_floor) / 2.0], dtype=np.float64)
    pad_centre = ring_centre + (w_out + _PAD_RING_GAP_M + _PAD_HALF_M[2]) * down_local
    out.append(
        (
            list(_PAD_HALF_M),
            [float(pad_centre[0]), float(pad_centre[1]), float(pad_centre[2])],
            _quat_wxyz_from_rot(r_nom.T),
        )
    )
    return out


def cohold_nominal_pad_bottom_z_w() -> float:
    """World z of the part base pad's bottom face at the nominal pose (ADR-029).

    The ``"none"`` control's kinematic stand top is placed here so the free
    part rests statically stable at the nominal presentation before the
    securing load arrives. Pure Tier-1 function (constants + linear algebra).

    Returns:
        The pad bottom-face world z at :data:`_NOMINAL_PART_POS_W`.
    """
    half, centre, _quat = _part_boxes(COHOLD_CLEARANCE_SIDE_SET_M[0])[-1]
    r_nom = _rot_from_quat_wxyz(_NOMINAL_PART_QUAT_W)
    centre_w = np.asarray(_NOMINAL_PART_POS_W) + r_nom @ np.asarray(centre)
    return float(centre_w[2]) - float(half[2])


def _augmented_part_holder_urdf(clearance_side_m: float) -> str:
    """Write a holder URDF = panda_v2 + the part as a fixed child of panda_hand.

    The fixed-link attach (the banked co-insert finding 1; ADR-029
    §Rationale): the held part is a rigid LINK in the holder articulation (a
    fixed joint off ``panda_hand``), NOT a ``create_drive`` inter-body weld —
    the reduced-coordinate solver holds it without phantom preload and the
    holder joint impedance bears the securing reaction. Geometry from
    :func:`_part_boxes` (shared with the actor part). Mesh paths are
    absolutised so the generated URDF can live in a temp dir. Returns the
    temp URDF path.
    """
    import tempfile

    from mani_skill import PACKAGE_ASSET_DIR

    panda_dir = f"{PACKAGE_ASSET_DIR}/robots/panda"
    base = open(f"{panda_dir}/panda_v2.urdf", encoding="utf-8").read()  # noqa: SIM115
    base = base.replace(
        'filename="franka_description/', f'filename="{panda_dir}/franka_description/'
    )
    blocks: list[str] = []
    for half, centre, quat in _part_boxes(clearance_side_m):
        r, p, y = _quat_wxyz_to_rpy(quat)
        origin = f'<origin xyz="{centre[0]} {centre[1]} {centre[2]}" rpy="{r} {p} {y}"/>'
        geometry = f'<geometry><box size="{2 * half[0]} {2 * half[1]} {2 * half[2]}"/></geometry>'
        blocks.append(f"    <collision>{origin}{geometry}</collision>")
        blocks.append(f"    <visual>{origin}{geometry}</visual>")
    geom = "\n".join(blocks)
    rpy = _quat_wxyz_to_rpy(_PART_GRIP_IN_HAND_Q)
    xyz = _PART_GRIP_IN_HAND_P
    block = (
        f'\n  <link name="{_PART_LINK_NAME}">\n{geom}\n'
        f'    <inertial><mass value="{COHOLD_PART_MASS_KG}"/>'
        '<inertia ixx="5e-4" ixy="0" ixz="0" iyy="5e-4" iyz="0" izz="5e-4"/></inertial>\n'
        f'  </link>\n  <joint name="{_PART_LINK_NAME}_joint" type="fixed">\n'
        f'    <origin xyz="{xyz[0]} {xyz[1]} {xyz[2]}" rpy="{rpy[0]} {rpy[1]} {rpy[2]}"/>\n'
        f'    <parent link="{_HAND_LINK_NAME}"/>\n'
        f'    <child link="{_PART_LINK_NAME}"/>\n  </joint>\n'
    )
    out = base.replace("</robot>", block + "</robot>")
    fd, path = tempfile.mkstemp(suffix="_cohold_holder.urdf")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(out)
    return path


#: Per-uid initial base pose ``(xyz, quat_wxyz)`` — mirrored, deliberate poses.
_BASE_POSE_BY_UID: dict[
    str, tuple[tuple[float, float, float], tuple[float, float, float, float]]
] = {
    _EGO_UID: ((-_BASE_X_M, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
    _PARTNER_UID: ((+_BASE_X_M, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
}


def make_co_hold_secure_env(
    *,
    control: str = "matched",
    clearance_side_m: float = COHOLD_CLEARANCE_SIDE_SET_M[1],
    detent_force_n: float = COHOLD_DETENT_FORCE_N,
    detent_travel_m: float = COHOLD_DETENT_TRAVEL_M,
    episode_length: int = COHOLD_DEFAULT_EPISODE_LENGTH,
    root_seed: int = 0,
    num_envs: int = 1,
    render_mode: str | None = None,
    render_backend: str | None = None,
    plug_part_friction: float = COHOLD_PLUG_PART_FRICTION,
    f_secure_max: float | None = None,
    f_couple_max: float | None = None,
    pose_sequence_len: int = 1,
) -> gym.Env[Any, Any]:
    """Build a :class:`CoHoldSecureEnv` instance (ADR-029 §Decision).

    Factory entry point (the ``chamber.tasks`` registry dotted path). SAPIEN /
    ManiSkill imports are deferred to the body so ``python -c "import
    chamber.envs.co_hold_secure"`` works on a Vulkan-less host (ADR-001 §Risks
    / P2). The :class:`CoHoldSecureEnv` class is defined inside the factory
    body — the :func:`chamber.envs.coinsert.make_coinsert_env` pattern.

    Args:
        control: Precheck cell — ``"matched"`` / ``"limp"`` / ``"none"`` /
            ``"fixture"`` (see the module docstring; ADR-029 §Decision).
        clearance_side_m: Per-side bore clearance, metres; one of
            :data:`COHOLD_CLEARANCE_SIDE_SET_M` for the graded cells (default
            the middle cell, 2.5 mm). Any positive value constructs (the
            design-window check is a precheck/prereg rule, not a constructor
            gate).
        detent_force_n: Seat-click resistance, Newtons (default
            :data:`COHOLD_DETENT_FORCE_N`; the P4 sweep passes {20, 40, 60}).
        detent_travel_m: Final travel window the detent resists over.
        episode_length: Truncation horizon, env ticks (the quasi-static band
            [250, 320]).
        root_seed: Project root seed for the env's :func:`derive_substream`
            substream (P6 / ADR-002).
        num_envs: ManiSkill vectorisation count; default 1.
        render_mode: ManiSkill ``render_mode`` (``None`` disables).
        render_backend: ManiSkill ``render_backend``; ``"none"`` enables the
            headless URDF-material strip.
        plug_part_friction: Coulomb friction on the plug-part pair.
        f_secure_max: Securing contact-force budget for ``within_force``;
            ``None`` (default) => ``+inf`` (the conjunct is vacuous so a
            measurement run collects the raw distribution; derived + frozen at
            Gate-0).
        f_couple_max: Holder workpiece-wrench budget; same ``None`` semantics.
        pose_sequence_len: Route-(ii) multi-pose hook (ADR-029 §Decision).
            Must be 1 in PR-A; values > 1 raise ``NotImplementedError``.

    Returns:
        A :class:`CoHoldSecureEnv` ready to ``reset(seed=K)``.

    Raises:
        ValueError: If ``control`` is unknown, ``clearance_side_m`` is not
            positive, or ``episode_length`` is outside [250, 320].
        NotImplementedError: If ``pose_sequence_len > 1`` (the dormant hook).
        ChamberEnvCompatibilityError: If ManiSkill / SAPIEN / Vulkan
            initialisation fails.
    """
    config = resolve_cohold_control(control)
    if int(num_envs) != 1:
        msg = (
            "CoHoldSecureEnv: the telemetry instruments (securing contact force, "
            "episode-reset detection, detent latch, pose_held reference) are scalar "
            "per-env-0 in PR-A; num_envs != 1 would silently mis-attribute channels "
            "(ADR-029 §Risks). Vectorisation is PR-B harness work — fail loudly here."
        )
        raise ValueError(msg)
    if int(pose_sequence_len) != 1:
        msg = (
            "CoHoldSecureEnv: pose_sequence_len > 1 is the dormant route-(ii) multi-pose "
            "securing hook (ADR-029 §Decision); enabling it is a task change (version bump "
            "+ ADR review), not a constructor flag."
        )
        raise NotImplementedError(msg)
    if not float(clearance_side_m) > 0.0:
        msg = f"CoHoldSecureEnv: clearance_side_m must be positive; got {clearance_side_m!r}."
        raise ValueError(msg)
    if not (COHOLD_EPISODE_LENGTH_MIN <= int(episode_length) <= COHOLD_EPISODE_LENGTH_MAX):
        msg = (
            f"CoHoldSecureEnv: episode_length {episode_length} is outside the quasi-static "
            f"band [{COHOLD_EPISODE_LENGTH_MIN}, {COHOLD_EPISODE_LENGTH_MAX}]."
        )
        raise ValueError(msg)
    try:
        import mani_skill.envs  # noqa: F401 - registers ManiSkill env IDs
        import sapien
        import torch
        from mani_skill.envs.sapien_env import BaseEnv
        from mani_skill.utils.building import actors
        from mani_skill.utils.scene_builder.table import TableSceneBuilder
        from mani_skill.utils.structs.pose import Pose

        import chamber.agents  # noqa: F401 - registers PandaPartner via @register_agent
        from chamber.envs._sapien_compat import (
            load_agent_with_bare_uids,
            patch_sapien_urdf_no_visual_material,
        )
    except ImportError as exc:
        raise ChamberEnvCompatibilityError(
            "CoHoldSecureEnv requires mani_skill / sapien in the active venv. "
            "Install per pyproject.toml; see ADR-001 §Risks and ADR-005 §Decision."
        ) from exc

    if render_backend == "none":
        patch_sapien_urdf_no_visual_material()

    # The inner env class is SAPIEN/Vulkan-only — it subclasses ManiSkill's
    # ``BaseEnv`` and every method drives the physx scene, so it cannot execute
    # on the CPU-only coverage-gate runner. It is exercised by
    # ``tests/integration/test_co_hold_secure_real.py`` on a GPU host; the pure
    # module-level surface stays covered by the Tier-1 tests.
    class CoHoldSecureEnv(BaseEnv):  # type: ignore[misc, valid-type]  # pragma: no cover
        """Two-robot hold-and-secure env (ADR-029 §Decision).

        Ego = securing operator holding the welded plug; partner = black-box
        holder carrying the part as a fixed articulation link. See the module
        docstring for the design rule, the detent instrument, and the four
        controls.
        """

        SUPPORTED_ROBOTS: ClassVar[list[tuple[str, str]]] = [  # type: ignore[assignment]
            (_EGO_UID, _PARTNER_UID),
        ]

        def __init__(self) -> None:
            self._config: CoHoldControl = config
            self._episode_length: int = int(episode_length)
            self._root_seed: int = int(root_seed)
            self._clearance_side: float = float(clearance_side_m)
            self._detent_force: float = float(detent_force_n)
            self._detent_travel: float = float(detent_travel_m)
            self._friction: float = float(plug_part_friction)
            self._f_secure_max: float = (
                float("inf") if f_secure_max is None else float(f_secure_max)
            )
            self._f_couple_max: float = (
                float("inf") if f_couple_max is None else float(f_couple_max)
            )
            self._rng: np.random.Generator = derive_substream(
                _SUBSTREAM_NAME, root_seed=self._root_seed
            ).default_rng()
            self._drives: list[Any] = []
            self._hand_link_index: int | None = None
            # Episode buffers (reset lazily; the co-insert running-maxima
            # pattern + the pose_held reference/excursion trackers).
            self._peak_secure_n: Any = None
            self._peak_couple_n: Any = None
            self._pose_ref_p: Any = None
            self._pose_ref_q: Any = None
            self._max_excursion_m: Any = None
            self._max_tilt_deg: Any = None
            self._eval_prev_step: int = -1
            self._detent_latched: list[bool] | None = None
            self._holder_urdf_path: str | None = None
            orig_holder_urdf: str | None = None
            if self._config.part_mode == "fixed_link":
                from chamber.agents.panda_partner import PandaPartner

                self._holder_urdf_path = _augmented_part_holder_urdf(self._clearance_side)
                orig_holder_urdf = PandaPartner.urdf_path
                PandaPartner.urdf_path = self._holder_urdf_path  # type: ignore[assignment]
            try:
                super().__init__(
                    robot_uids=(_EGO_UID, _PARTNER_UID),  # type: ignore[arg-type]
                    num_envs=num_envs,
                    obs_mode="state_dict",
                    control_mode=COHOLD_CONTROL_MODE,
                    render_mode=render_mode,
                    render_backend=render_backend if render_backend is not None else "gpu",
                )
            except RuntimeError as exc:
                raise ChamberEnvCompatibilityError(
                    "SAPIEN/Vulkan initialisation failed during "
                    f"CoHoldSecureEnv(control={control!r}) build: {exc}\n"
                    'Set render_backend="none" on CUDA-only hosts; see ADR-001 §Risks.'
                ) from exc
            finally:
                if orig_holder_urdf is not None:
                    from chamber.agents.panda_partner import PandaPartner

                    PandaPartner.urdf_path = orig_holder_urdf  # type: ignore[assignment]

        # ----- ManiSkill v3 BaseEnv hooks -----

        @property
        def _default_human_render_camera_configs(self) -> Any:  # noqa: ANN401 - ManiSkill CameraConfig has no project type
            """Third-person render camera for rollout visualisation (ADR-029)."""
            from mani_skill.sensors.camera import CameraConfig
            from mani_skill.utils import sapien_utils

            pose = sapien_utils.look_at(eye=[0.0, 0.9, 0.7], target=[0.0, 0.0, 0.3])
            return CameraConfig(
                "render_camera", pose=pose, width=640, height=480, fov=1.0, near=0.01, far=100.0
            )

        def _load_agent(  # type: ignore[override]
            self,
            options: dict[str, Any],
            initial_agent_poses: object = None,
            build_separate: bool = False,
        ) -> None:
            """Ego + holder ``_load_agent`` with mirrored base poses (ADR-029)."""
            if initial_agent_poses is None:
                poses: list[object] = []
                for uid in self.robot_uids:  # type: ignore[union-attr]
                    xyz, quat = _BASE_POSE_BY_UID[uid]
                    poses.append(sapien.Pose(p=list(xyz), q=list(quat)))
                initial_agent_poses = poses
            load_agent_with_bare_uids(
                self,
                options,
                initial_agent_poses=initial_agent_poses,
                build_separate=build_separate,
            )

        def _load_scene(self, options: dict[str, Any]) -> None:
            """Build table + welded plug + part (per control) + goal (ADR-029).

            The plug is a cylinder welded to the ego gripper (two-anchor
            attach). The part is (a) a fixed link of the holder articulation
            (``matched`` / ``limp``), (b) a free actor over a kinematic stand
            (``none``), or (c) a kinematic actor (``fixture``) — all built
            from the same :func:`_part_boxes` geometry.
            """
            del options
            self.table_scene = TableSceneBuilder(self)
            self.table_scene.build()

            contact_mat = sapien.physx.PhysxMaterial(self._friction, self._friction, 0.0)

            plug_radius = COHOLD_PLUG_DIAMETER_M / 2.0
            plug_half_len = 0.020
            self._plug_half_len: float = plug_half_len
            plug_builder = self.scene.create_actor_builder()
            plug_colour = sapien.render.RenderMaterial(base_color=[0.30, 0.30, 0.35, 1.0])
            # SAPIEN cylinder primitives lie along local x; rotate -90 deg
            # about y so the cylinder axis is the plug's local +z (the securing
            # axis) — the S2 round-geometry convention.
            cyl_pose = sapien.Pose(q=[0.7071067811865476, 0.0, -0.7071067811865476, 0.0])
            plug_builder.add_cylinder_collision(
                pose=cyl_pose,
                radius=plug_radius,
                half_length=plug_half_len,
                material=contact_mat,
                density=1000.0,
            )
            plug_builder.add_cylinder_visual(
                pose=cyl_pose, radius=plug_radius, half_length=plug_half_len, material=plug_colour
            )
            plug_builder.initial_pose = sapien.Pose(p=[-_BASE_X_M + 0.2, 0.0, 0.4])  # type: ignore[assignment]
            self.plug = plug_builder.build(name="cohold_plug")

            if self._config.part_mode == "fixed_link":
                self.part = self.agent.agents_dict[  # type: ignore[attr-defined]
                    _PARTNER_UID
                ].robot.links_map[_PART_LINK_NAME]
                self._stand = None
            else:
                self.part = self._build_part_actor(
                    contact_mat, kinematic=self._config.part_mode == "world_fixed"
                )
                if self._config.part_mode == "free_stand":
                    stand_top = cohold_nominal_pad_bottom_z_w()
                    stand_builder = self.scene.create_actor_builder()
                    half_z = stand_top / 2.0
                    stand_builder.add_box_collision(
                        half_size=(_STAND_HALF_XY_M[0], _STAND_HALF_XY_M[1], half_z),
                    )
                    stand_builder.add_box_visual(
                        half_size=(_STAND_HALF_XY_M[0], _STAND_HALF_XY_M[1], half_z),
                        material=sapien.render.RenderMaterial(base_color=[0.4, 0.4, 0.4, 1.0]),
                    )
                    stand_builder.initial_pose = sapien.Pose(  # type: ignore[assignment]
                        p=[
                            float(_NOMINAL_PART_POS_W[0]),
                            float(_NOMINAL_PART_POS_W[1]),
                            half_z,
                        ]
                    )
                    stand_builder.set_physx_body_type("kinematic")
                    self._stand = stand_builder.build(name="cohold_stand")
                else:
                    self._stand = None

            self.goal_site = actors.build_sphere(
                self.scene,
                radius=0.02,
                color=[0, 1, 0, 1],
                name="goal_site",
                body_type="kinematic",
                add_collision=False,
                initial_pose=sapien.Pose(),
            )
            self._hidden_objects.append(self.goal_site)  # type: ignore[attr-defined]

            # Holder wrist-link index for the friction-inclusive wrench
            # instrument (only meaningful when the holder participates).
            links = self.agent.agents_dict[_PARTNER_UID].robot.links  # type: ignore[attr-defined]
            self._hand_link_index = [ln.name for ln in links].index(_HAND_LINK_NAME)

            # Weld the plug to the ego gripper (two-anchor attach). The part
            # is NEVER create_drive-welded: fixed link (matched/limp), free
            # (none), or kinematic (fixture) — ADR-029 §Rationale.
            self._drives = []
            self._add_plug_weld()

        def _build_part_actor(
            self,
            contact_mat: Any,  # noqa: ANN401 - sapien material
            *,
            kinematic: bool,
        ) -> Any:  # noqa: ANN401 - sapien actor
            """Compose the part actor from :func:`_part_boxes` (ADR-001 §Risks no-mesh)."""
            specs = _part_boxes(self._clearance_side)
            volume = sum(8.0 * h[0] * h[1] * h[2] for h, _, _ in specs)
            density = COHOLD_PART_MASS_KG / volume
            builder = self.scene.create_actor_builder()
            colour = sapien.render.RenderMaterial(base_color=[0.55, 0.42, 0.20, 1.0])
            for half, centre, quat in specs:
                pose = sapien.Pose(p=centre, q=list(quat))
                builder.add_box_collision(
                    pose=pose,
                    half_size=tuple(half),  # type: ignore[arg-type]
                    material=contact_mat,
                    density=density,
                )
                builder.add_box_visual(pose=pose, half_size=tuple(half), material=colour)  # type: ignore[arg-type]
            builder.initial_pose = sapien.Pose(  # type: ignore[assignment]
                p=list(_NOMINAL_PART_POS_W), q=list(_NOMINAL_PART_QUAT_W)
            )
            if kinematic:
                builder.set_physx_body_type("kinematic")
            return builder.build(name="cohold_part")

        def _add_plug_weld(self) -> None:
            """Weld the plug into the ego grip frame (the S2 two-anchor attach).

            Two translation-locked anchors spread :data:`_WELD_ANCHOR_SPAN_M`
            along the plug axis lock position + both tilt DOF, leaving the
            harmless twist free — the S2 orientation-locking attach that
            avoids both the flopping single-point drive and the
            over-constrained 3-point weld. Public ``set_limit_{x,y,z}`` API
            only (ADR-001 §Risks / P2).
            """
            grip_p = np.asarray(_PLUG_GRIP_IN_HAND_P, dtype=np.float64)
            hand = self.agent.agents_dict[_EGO_UID].robot.links_map[_HAND_LINK_NAME]  # type: ignore[attr-defined]
            for d_body in ((0.0, 0.0, 0.0), (0.0, 0.0, _WELD_ANCHOR_SPAN_M)):
                d = np.asarray(d_body, dtype=np.float64)
                hand_anchor = grip_p + d  # identity weld orientation
                drive = self.scene.create_drive(
                    hand, sapien.Pose(p=hand_anchor.tolist()), self.plug, sapien.Pose(p=d.tolist())
                )
                for axis in ("x", "y", "z"):
                    getattr(drive, f"set_drive_property_{axis}")(2.0e4, 2.0e3)
                    getattr(drive, f"set_limit_{axis}")(0.0, 0.0)
                self._drives.append(drive)

        def _initialize_episode(self, env_idx: torch.Tensor, options: dict[str, Any]) -> None:
            """Reset ready/retracted poses, warm-start the plug, set the goal (P6).

            All randomisation routes through the P6 RNG (never ``torch.rand``).
            Physical state is deterministic per reset (fixed ready poses, no
            init noise), so two fresh envs with the same ``root_seed`` are
            byte-identical; within one instance the goal-jitter substream
            advances across resets (the co-insert convention — the goal is
            cosmetic telemetry the controllers do not read).
            """
            del options
            with torch.device(self.device):  # type: ignore[attr-defined]
                b = len(env_idx)
                self.table_scene.initialize(env_idx)
                for uid in self.robot_uids:  # type: ignore[union-attr]
                    if uid == _PARTNER_UID and not self._config.holder_active:
                        ready = _PANDA_RETRACTED_QPOS
                    elif uid == _EGO_UID:
                        ready = _PANDA_READY_QPOS_EGO
                    else:
                        ready = _PANDA_READY_QPOS_HOLDER
                    qpos = torch.from_numpy(np.tile(ready, (b, 1)).astype(np.float32)).to(
                        self.device
                    )
                    self.agent.agents_dict[uid].reset(qpos)  # type: ignore[attr-defined]

                # Warm-start the plug at its welded grip (zero initial weld
                # stress — the co-carry discipline).
                ego_hand = self.agent.agents_dict[_EGO_UID].robot.links_map[_HAND_LINK_NAME]  # type: ignore[attr-defined]
                plug_grip = ego_hand.pose * Pose.create_from_pq(
                    p=torch.tensor(_PLUG_GRIP_IN_HAND_P, device=self.device),
                    q=torch.tensor(_PLUG_GRIP_IN_HAND_Q, device=self.device),
                )
                self.plug.set_pose(plug_grip)

                # Free / fixed part actors start at the nominal presentation
                # (the fixed-link part follows the holder hand automatically).
                if self._config.part_mode != "fixed_link":
                    pos = torch.tensor(
                        _NOMINAL_PART_POS_W, dtype=torch.float32, device=self.device
                    ).reshape(1, 3)
                    quat = torch.tensor(
                        _NOMINAL_PART_QUAT_W, dtype=torch.float32, device=self.device
                    ).reshape(1, 4)
                    self.part.set_pose(Pose.create_from_pq(pos.expand(b, 3), quat.expand(b, 4)))

                # Goal: the nominal mouth + a tiny P6 jitter (non-degenerate
                # distribution; the controllers are closed-loop on the
                # observed poses and do not read it).
                jitter = self._rng.uniform(-0.005, 0.005, size=(b, 3))
                goal = np.asarray(_NOMINAL_PART_POS_W, dtype=np.float64)[None, :] + jitter
                self.goal_site.set_pose(
                    Pose.create_from_pq(torch.from_numpy(goal.astype(np.float32)).to(self.device))
                )
                # Clear episode buffers (peaks + pose_held trackers).
                self._peak_secure_n = None
                self._peak_couple_n = None
                self._pose_ref_p = None
                self._pose_ref_q = None
                self._max_excursion_m = None
                self._max_tilt_deg = None
                self._eval_prev_step = -1
                self._detent_latched = None

        def step(self, action: Any) -> tuple[Any, Any, Any, Any, dict[str, Any]]:  # type: ignore[override]  # noqa: ANN401
            """Enforce ``episode_length`` time-limit truncation (ADR-029)."""
            obs, reward, terminated, truncated, info = super().step(action)
            truncated = truncated | (self.elapsed_steps >= self._episode_length)
            return obs, reward, terminated, truncated, info

        def _before_simulation_step(self) -> None:
            """Apply the detent action-reaction force pair (ADR-029 §Decision).

            While the plug tip is inside the final :data:`detent_travel`
            window of the bore and the click has not yet latched, a resistance
            ramping linearly to ``detent_force`` pushes the plug back up the
            bore axis and its reaction pushes the part forward — an analytic
            compliant feature, not rigid interference geometry (the S2
            constraint-fidelity lesson; ADR-029 §Risks). At
            :data:`COHOLD_DETENT_CLICK_EPS_M` short of the engagement depth
            the detent latches free (the "click"); it re-arms only if the plug
            backs fully out of the window. Applied every physics substep
            (PhysX clears external forces after each step), so the resistance
            is a force profile, not a once-per-control-tick impulse.
            """
            depth, _align, _lateral = self._plug_part_geometry()
            axis = self._part_axis_world()  # (b, 3): bore axis, out of mouth
            tip = self._plug_tip_world()  # (b, 3)
            mouth = self.part.pose.p
            depth_np = depth.detach().cpu().numpy().reshape(-1)
            axis_np = axis.detach().cpu().numpy().reshape(-1, 3)
            tip_np = tip.detach().cpu().numpy().reshape(-1, 3)
            mouth_np = mouth.detach().cpu().numpy().reshape(-1, 3)
            target = COHOLD_ENGAGE_DEPTH_M
            entry = target - self._detent_travel
            click = target - COHOLD_DETENT_CLICK_EPS_M
            if self._detent_latched is None or len(self._detent_latched) != len(depth_np):
                self._detent_latched = [False] * len(depth_np)
            plug_bodies = self._rigid_bodies(self.plug)
            part_bodies = self._rigid_bodies(self.part)
            for i, d in enumerate(depth_np):
                if d < entry:
                    self._detent_latched[i] = False  # re-arm on full back-out
                    continue
                if d >= click:
                    self._detent_latched[i] = True  # the click — cam is past
                if self._detent_latched[i]:
                    continue
                frac = min(1.0, max(0.0, (d - entry) / max(click - entry, 1e-9)))
                force = (self._detent_force * frac * axis_np[i]).astype(np.float32)
                if i < len(plug_bodies) and plug_bodies[i] is not None:
                    plug_bodies[i].add_force_at_point(force, tip_np[i].astype(np.float32))
                # Kinematic part bodies (the "fixture" control) ignore external
                # forces by definition — skip rather than let PhysX reject the
                # call loudly (behaviour-identical; an infinitely rigid fixture
                # absorbs the reaction).
                if (
                    i < len(part_bodies)
                    and part_bodies[i] is not None
                    and not getattr(part_bodies[i], "kinematic", False)
                ):
                    part_bodies[i].add_force_at_point(-force, mouth_np[i].astype(np.float32))

        @staticmethod
        def _rigid_bodies(entity: Any) -> list[Any]:  # noqa: ANN401 - maniskill struct
            """Per-sub-scene physx rigid-body components of an Actor / Link.

            ManiSkill ``Actor``/``Link`` structs both expose ``_bodies``
            (physx rigid components); kinematic bodies accept and ignore
            external forces, so the fixture control needs no special case.
            """
            bodies = getattr(entity, "_bodies", None)
            if bodies is not None:
                return list(bodies)
            objs = getattr(entity, "_objs", [])
            out: list[Any] = []
            for obj in objs:
                if hasattr(obj, "add_force_at_point"):
                    out.append(obj)
                    continue
                comp = None
                if hasattr(obj, "find_component_by_type"):
                    comp = obj.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
                out.append(comp)
            return out

        # ----- Observation / geometry / telemetry -----

        def _get_obs_extra(self, info: dict[str, Any]) -> dict[str, Any]:
            """Task obs: goal + plug pose + part pose (ADR-029 §Decision).

            The pose keys keep the co-insert controller obs contract
            (``peg_pose`` = the ego-held plug, ``receptacle_pose`` = the held
            part) so the banked S2 controllers run unmodified (ADR-029
            §Rationale — reuse-as-is).
            """
            del info
            return {
                "goal_pos": self.goal_site.pose.p,
                "peg_pose": self.plug.pose.raw_pose,
                "receptacle_pose": self.part.pose.raw_pose,
            }

        def _part_axis_world(self) -> Any:  # noqa: ANN401 - torch.Tensor
            """Bore axis (part local +z, out of the mouth) in world frame, per env."""
            import torch as _torch

            q = self.part.pose.q
            w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
            return _torch.stack(
                [2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], dim=1
            )

        def _plug_tip_world(self) -> Any:  # noqa: ANN401 - torch.Tensor
            """World position of the plug's securing tip (local +z end)."""
            import torch as _torch

            q = self.plug.pose.q
            w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
            plug_axis = _torch.stack(
                [2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], dim=1
            )
            return self.plug.pose.p + self._plug_half_len * plug_axis

        def _plug_part_geometry(self) -> tuple[Any, Any, Any]:
            """Per-env (plug-tip depth m, axis misalignment deg, lateral offset m).

            Computed from the live plug + part poses (no hard-coded geometry).
            Depth is the tip's penetration along the bore axis (> 0 once
            engaged); alignment is the angle between the plug's securing
            direction and the bore; lateral is the tip's off-axis offset.
            """
            import torch as _torch

            tip = self._plug_tip_world()
            sock_axis = self._part_axis_world()
            q = self.plug.pose.q
            w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
            plug_axis = _torch.stack(
                [2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], dim=1
            )
            rel = tip - self.part.pose.p
            depth = -_torch.sum(rel * sock_axis, dim=1)
            lateral = _torch.linalg.norm(
                rel - _torch.sum(rel * sock_axis, dim=1, keepdim=True) * sock_axis, dim=1
            )
            cos = _torch.clamp(_torch.sum(plug_axis * (-sock_axis), dim=1), -1.0, 1.0)
            align_deg = _torch.rad2deg(_torch.arccos(cos))
            return depth, align_deg, lateral

        def _both_static(self) -> Any:  # noqa: ANN401 - torch.Tensor
            """Whether the participating arms are below the static threshold."""
            import torch as _torch

            uids = [_EGO_UID, _PARTNER_UID] if self._config.holder_active else [_EGO_UID]
            norms = []
            for uid in uids:
                qvel = self.agent.agents_dict[uid].robot.get_qvel()[..., :_PANDA_ARM_DOF]  # type: ignore[attr-defined]
                norms.append(_torch.linalg.norm(qvel, dim=1))
            return _torch.stack(norms, dim=0).amax(dim=0) < COHOLD_STATIC_QVEL_THRESH

        def holder_workpiece_wrench(self) -> Any:  # noqa: ANN401 - torch.Tensor
            """Friction-inclusive holder workpiece-wrench magnitude, N (the stress channel).

            The pinned stress instrument (ADR-029 §Consequences): the holder
            articulation's wrist incoming joint force via
            ``get_link_incoming_joint_forces`` — the full reduced-coordinate
            solver force (friction-inclusive, unlike the GPU contact-pair
            impulse). Returns zeros when the holder does not participate
            (``none`` / ``fixture`` — the channel is defined per-control and
            reported as absent, never faked).
            """
            import torch as _torch

            b = self.plug.pose.p.shape[0]
            if not self._config.holder_active or self._hand_link_index is None:
                return _torch.zeros(b, device=self.device)
            robot = self.agent.agents_dict[_PARTNER_UID].robot  # type: ignore[attr-defined]
            forces = robot.get_link_incoming_joint_forces()  # (b, n_links, 6)
            return _torch.linalg.norm(forces[:, self._hand_link_index, :3], axis=1)

        def plug_part_contact_force(self) -> float:
            """Plug-part contact-force magnitude, N (the securing contact channel; ADR-029).

            Sums the contact-pair impulses between the plug and the part and
            divides by the PhysX timestep (the S1 instrument convention).
            Returns 0.0 when there is no contact.
            """
            import torch as _torch

            impulses = self.scene.get_pairwise_contact_impulses(self.plug, self.part)  # type: ignore[attr-defined]
            imp = _torch.as_tensor(impulses)
            return float(_torch.linalg.norm(imp).item()) / float(self.sim_timestep)

        def _accumulate_episode_buffers(self) -> tuple[Any, Any, Any, Any]:
            """Update + return (peak secure N, peak couple N, max excursion m, max tilt deg).

            Resets lazily at episode start (``elapsed_steps`` regression) and
            excludes the first :data:`COHOLD_SETTLE_WINDOW_STEPS` ticks (the
            reset transient is not cooperation cost). The pose_held reference
            pose is the part pose captured at the settle-window boundary; the
            excursion trackers are running maxima after it. The tilt tracker
            is the full quaternion geodesic angle — it includes spin about the
            bore axis, so it is a *conservative* pose_held conjunct (noted for
            the Gate-0 tolerance freeze; ADR-029 §Consequences).
            """
            import torch as _torch

            elapsed = int(_torch.as_tensor(self.elapsed_steps).reshape(-1)[0].item())
            b = self.plug.pose.p.shape[0]
            if self._peak_secure_n is None or elapsed < self._eval_prev_step:
                self._peak_secure_n = _torch.zeros(b, device=self.device)
                self._peak_couple_n = _torch.zeros(b, device=self.device)
                self._pose_ref_p = None
                self._pose_ref_q = None
                self._max_excursion_m = _torch.zeros(b, device=self.device)
                self._max_tilt_deg = _torch.zeros(b, device=self.device)
            elif elapsed == self._eval_prev_step:
                # Same-tick re-evaluation (a harness calling evaluate() on top
                # of step()'s internal get_info) must be a no-op, not a silent
                # episode restart that zeroes the peaks and re-captures the
                # pose_held reference.
                return (
                    self._peak_secure_n,
                    self._peak_couple_n,
                    self._max_excursion_m,
                    self._max_tilt_deg,
                )
            self._eval_prev_step = elapsed
            if elapsed >= COHOLD_SETTLE_WINDOW_STEPS:
                if self._pose_ref_p is None:
                    self._pose_ref_p = self.part.pose.p.clone()
                    self._pose_ref_q = self.part.pose.q.clone()
                contact = _torch.full(
                    (b,), float(self.plug_part_contact_force()), device=self.device
                )
                wrench = self.holder_workpiece_wrench()
                self._peak_secure_n = _torch.maximum(self._peak_secure_n, contact)
                self._peak_couple_n = _torch.maximum(self._peak_couple_n, wrench)
                trans = _torch.linalg.norm(self.part.pose.p - self._pose_ref_p, dim=1)
                dot = _torch.clamp(
                    _torch.abs(_torch.sum(self.part.pose.q * self._pose_ref_q, dim=1)),
                    max=1.0,
                )
                tilt = _torch.rad2deg(2.0 * _torch.arccos(dot))
                self._max_excursion_m = _torch.maximum(self._max_excursion_m, trans)
                self._max_tilt_deg = _torch.maximum(self._max_tilt_deg, tilt)
            return (
                self._peak_secure_n,
                self._peak_couple_n,
                self._max_excursion_m,
                self._max_tilt_deg,
            )

        def evaluate(self) -> dict[str, Any]:
            """Joint co-hold-secure success + raw telemetry (ADR-029 §Consequences).

            ``success = seated ∧ pose_held ∧ within_force ∧ static ∧ settled``
            (:func:`evaluate_cohold_success` semantics computed live), with
            the raw geometry / peaks / excursions surfaced so the precheck and
            the later Gate-0 harness recompute conjuncts from the archived
            channels (the co-carry "env exposes raw maxima, the script applies
            the threshold" pattern). ``success_geom`` is the force-agnostic
            ``seated ∧ pose_held ∧ static ∧ settled``.
            """
            import torch as _torch

            depth, align_deg, lateral = self._plug_part_geometry()
            peak_secure, peak_couple, excursion, tilt = self._accumulate_episode_buffers()
            seated = (depth >= (COHOLD_ENGAGE_DEPTH_M - COHOLD_DEPTH_EPS_M)) & (
                align_deg <= COHOLD_AXIS_ALIGN_TOL_DEG
            )
            pose_held = (excursion <= COHOLD_POSE_HELD_TRANS_TOL_M) & (
                tilt <= COHOLD_POSE_HELD_TILT_TOL_DEG
            )
            within_force = (peak_secure <= self._f_secure_max) & (peak_couple <= self._f_couple_max)
            both_static = self._both_static()
            elapsed = int(_torch.as_tensor(self.elapsed_steps).reshape(-1)[0].item())
            part_vel = _torch.linalg.norm(
                _torch.as_tensor(self.part.get_linear_velocity()), dim=-1
            ).reshape(-1)
            settled = _torch.as_tensor(
                elapsed >= COHOLD_SETTLE_WINDOW_STEPS, device=self.device
            ) & (part_vel < COHOLD_SETTLE_VEL_THRESH)
            success_geom = seated & pose_held & both_static & settled
            success = success_geom & within_force
            return {
                "success": success,
                "success_geom": success_geom,
                "seated": seated,
                "pose_held": pose_held,
                "within_force": within_force,
                "both_static": both_static,
                "settled": settled,
                "seated_depth_m": depth,
                "axis_align_deg": align_deg,
                "lateral_offset_m": lateral,
                "pose_excursion_m": excursion,
                "pose_tilt_deg": tilt,
                "peak_secure_force_n": peak_secure,
                "peak_couple_wrench_n": peak_couple,
            }

        def compute_normalized_dense_reward(
            self,
            obs: Any,  # noqa: ANN401
            action: Any,  # noqa: ANN401
            info: dict[str, Any],
        ) -> Any:  # noqa: ANN401
            """Dense securing reward: align + depth + seat bonus (ADR-029).

            The co-insert shaping pattern kept minimal (no learned consumer
            exists in PR-A — the learned-ladder plan is a later, separately
            ratified slice; ADR-029 §Decision). Present so the ManiSkill
            reward plumbing works without a special reward_mode.
            """
            del obs, action
            import torch as _torch

            depth, _align_deg, lateral = self._plug_part_geometry()
            align = 1.0 - _torch.tanh(50.0 * lateral)
            depth_prog = _torch.clamp(depth / COHOLD_ENGAGE_DEPTH_M, min=0.0, max=1.0)
            reward = align + depth_prog + 3.0 * info["seated"].to(depth.dtype)
            reward = _torch.where(info["success"], _torch.full_like(reward, 5.0), reward)
            return reward / 5.0

        # ----- Public read-only API -----

        def close(self) -> None:  # type: ignore[override]
            """Close the env and remove the generated holder URDF temp file (ADR-029)."""
            try:
                super().close()
            finally:
                if self._holder_urdf_path is not None:
                    import contextlib

                    with contextlib.suppress(OSError):
                        os.unlink(self._holder_urdf_path)
                    self._holder_urdf_path = None

        @property
        def control_id(self) -> str:
            """The control cell this env was built for (ADR-029 §Decision)."""
            return self._config.control_id

        @property
        def control_config(self) -> CoHoldControl:
            """The resolved :class:`CoHoldControl` (read-only; ADR-029)."""
            return self._config

        @property
        def clearance_side_m(self) -> float:
            """The per-side bore clearance this env was built with (ADR-029)."""
            return self._clearance_side

        @property
        def detent_force_n(self) -> float:
            """The detent (seat-click) resistance, Newtons (ADR-029 §Decision)."""
            return self._detent_force

        @property
        def ego_uid(self) -> str:
            """The ego (securing operator) uid (ADR-029 §Decision)."""
            return _EGO_UID

        @property
        def partner_uid(self) -> str:
            """The holder partner uid (ADR-029 §Decision)."""
            return _PARTNER_UID

    try:
        return CoHoldSecureEnv()
    except RuntimeError as exc:  # pragma: no cover - host-dependent SAPIEN failure
        raise ChamberEnvCompatibilityError(
            f"CoHoldSecureEnv construction failed: {exc}; see ADR-001 §Risks."
        ) from exc


__all__ = [
    "COHOLD_ACHIEVABLE_TILT_CEILING_DEG",
    "COHOLD_AXIS_TILT_FROM_VERTICAL_DEG",
    "COHOLD_CHAMFER_M",
    "COHOLD_CLEARANCE_SIDE_SET_M",
    "COHOLD_COUPLE_FORCE_MAX_N",
    "COHOLD_DETENT_CLICK_EPS_M",
    "COHOLD_DETENT_FORCE_N",
    "COHOLD_DETENT_TRAVEL_M",
    "COHOLD_ENGAGE_DEPTH_M",
    "COHOLD_PART_MASS_KG",
    "COHOLD_PLUG_DIAMETER_M",
    "COHOLD_POSE_HELD_TILT_TOL_DEG",
    "COHOLD_POSE_HELD_TRANS_TOL_M",
    "COHOLD_SECURE_FORCE_MAX_N",
    "COHOLD_WEDGE_MARGIN_MIN",
    "CoHoldControl",
    "cohold_bore_inner_radius",
    "cohold_design_window_ok",
    "cohold_nominal_pad_bottom_z_w",
    "cohold_wedge_limit_deg",
    "evaluate_cohold_success",
    "make_co_hold_secure_env",
    "resolve_cohold_control",
]
