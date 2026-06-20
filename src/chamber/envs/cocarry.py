# SPDX-License-Identifier: Apache-2.0
# pyright: reportPrivateImportUsage=false
#
# Same suppression rationale as :mod:`chamber.envs.stage1_pickplace`:
# ``torch.zeros`` / ``torch.from_numpy`` are exported but not advertised in
# the stub's ``__all__``. Suppressed file-locally so the multi-agent
# telemetry + reward logic stays free of per-line ``type: ignore`` noise.
r"""Co-carry env — coupling-valid two-arm transport task (ADR-026 §Decision 1-2).

Phase-2 / post-M10 forward design. Two manipulators rigidly carry one
shared rigid bar to a goal while holding it level. The task is
**infeasible for one robot**, so the success score measures cooperation
rather than a single ego's competence — the construct defect ADR-026
diagnosed in the Stage-1 AS pick-place task (ego-solvable; partner a
non-participating nuisance) does not recur here.

The coupling criterion (ADR-026 §Decision 1): *a heterogeneity axis is
informative about cooperation only if the manipulated heterogeneity is
coupled to the task outcome through the cooperation the task demands.*
The shared rigid bar is a **continuous physical coupling** with no
hand-off interface that could launder embodiment away; the level / stress
limits bind only when two holds share the load.

Scope of this module (R-2026-06-B Rungs 0-1): build the rig and prove the
coupling is real, using a **matched (identical) Panda pair** only. The
learned incumbent (Rung 2), the policy-shift teammates (Rung 3), and the
embodiment-shift xArm6 (Rung 4) are later, separately pre-registered
slices — **not** built here.

Governance (ADR-026 §Decision 4; invariants I1/I8/I9):
this module adds nothing to the Phase-1 gate, does not touch any Stage-1
env / condition / pre-registration, does not edit any immutable archive
under ``spikes/results/**``, and triggers no schema bump. It reuses the
existing wrapper-only / determinism / partner-interface contracts.

The dual-hold attach (R-2026-06-B "rigid-joint stability check"):
each gripper's ``panda_hand`` link is rigidly coupled to its bar end via
a high-stiffness/damping 6-DOF SAPIEN drive (PhysX ``PhysxDriveComponent``
created through the public ``ManiSkillScene.create_drive`` surface;
linear axes locked via ``set_limit_{x,y,z}(0, 0)`` plus high-stiffness
drives, angular axes locked via the component's
``set_drive_property_{twist,swing}``). This is an *attach*, not a free
grasp — it removes grasp slip as a variable so failures reflect
coordination, not fingertip friction. Real grasping is a deliberate
later relaxation (out of scope here).

The stress proxy (R-2026-06-B "stress proxy defined physically"):
SAPIEN's free-drive (``PhysxDriveComponent``) does **not** expose a
constraint-solver force on its public surface (verified against
``sapien==3.0.3``: no force getter on ``PhysxDriveComponent`` /
``PhysxJointComponent``; only ``PhysxArticulation.get_link_incoming_
joint_forces`` exposes a reduced-coordinate solver force). The stress
proxy is therefore the **wrist constraint-solver force**: the 6-axis
spatial force the articulation solver transmits through each holding
arm's ``panda_hand`` incoming joint, read via
``robot.get_link_incoming_joint_forces()[:, HAND_LINK_INDEX]``. This is
a genuine PhysX constraint-solver output (the articulation joint
reaction), monotone in the bar load it must react. ``f_max`` is derived
from the *measured* distribution of this proxy on matched successful
episodes at Rung 1 (a justified high percentile) and recorded for the
Rung-3+ pre-statement; the module ships a documented placeholder
(:data:`COCARRY_STRESS_MAX_PROXY_N`) calibrated by that measurement.

Tier-1 / Tier-2 split (mirrors :mod:`chamber.envs.stage1_pickplace`):
module-top-level imports are Tier-1-safe (``numpy``, ``gymnasium``,
``concerto.training.seeding``). ManiSkill / SAPIEN imports happen inside
:func:`make_cocarry_env` so ``python -c "import chamber.envs.cocarry"``
succeeds on a Vulkan-less host. The pure-function Tier-1 surface
(:func:`resolve_cocarry_condition`, :func:`tilt_deg_from_quaternion`,
:func:`evaluate_cocarry_success`, the reward / geometry constants) is
what the Tier-1 tests exercise; Tier-2 SAPIEN-gated tests cover real env
construction, the Rung-0 attach stability, and the Rung-1 matched
competence + coupling positive-control.

References:
- ADR-026 §Decision 1-2 (the coupling-validity criterion; the
  falsifiable coupling positive-control + pre-committed null rule).
- ADR-007 §Stage 1b (env / determinism conventions mirrored here);
  ADR-007 §Implementation staging (the fail-cheap rung ladder).
- ADR-005 §Decision (ManiSkill v3 / SAPIEN 3; ``mani-skill==3.0.1``).
- ADR-009 §Decision (the frozen black-box partner contract the matched
  controller is routed through, so the rig is forward-compatible with
  the Rung-2 frozen incumbent).
- R-2026-06-B Rungs 0-1 (the structured-review conditions: rigid-joint
  stability gate, physical stress proxy, geometric single-arm
  infeasibility, pre-registration-grade reward coefficients).
- :mod:`chamber.envs.stage1_pickplace` (the factory / ``_load_agent`` /
  ``_initialize_episode`` / determinism template).
- :class:`chamber.agents.panda_jacobian.PandaJacobianProvider` (the
  Jacobian/FK the matched impedance controller drives the arms with).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple

import numpy as np

from chamber.envs.errors import ChamberEnvCompatibilityError
from concerto.training.seeding import derive_substream

if TYPE_CHECKING:  # pragma: no cover
    import gymnasium as gym
    import torch
    from numpy.typing import ArrayLike, NDArray

# ---------------------------------------------------------------------------
# Geometry constants (R-2026-06-B "single-arm infeasibility verified
# geometrically before freezing geometry"). These are deliberate, documented
# module constants — NOT the pick-place placeholders. The base separation +
# bar length are chosen so the goal is reachable only jointly (see the
# Rung-1 geometric positive-control in scripts/repro/cocarry_rung1_matched.sh).
# ---------------------------------------------------------------------------

#: Half the base-to-base separation along world x, in metres. The ego
#: Panda mounts at ``(-COCARRY_BASE_X_M, 0, 0)`` facing +x; the partner
#: mounts at ``(+COCARRY_BASE_X_M, 0, 0)`` facing -x (yaw pi about z).
#: 0.50 m places the two ``panda_hand_tcp`` frames ~0.23 m apart at the
#: canonical ready pose (measured: TCPs at x = +-0.115 m, z = 0.170 m),
#: so the bar spans the gap with both grippers reaching across the centre.
COCARRY_BASE_X_M: float = 0.50

#: Rigid bar long-axis length, in metres. Matched to the ready-pose TCP
#: gap so the dual-hold attach starts at zero internal stress (bar ends
#: coincide with the two TCP grip frames). Load-bearing for the geometric
#: single-arm infeasibility argument (a single arm cannot span + level
#: this length about a cantilevered free end without the tilt/stress
#: constraints binding); frozen-grade.
COCARRY_BAR_LENGTH_M: float = 0.23

#: Bar square cross-section side length, in metres (2.5 cm). Sets the
#: bar's slenderness; small enough that the bar reads as a slender beam,
#: large enough to give the box collider stable contact margins.
COCARRY_BAR_CROSS_SECTION_M: float = 0.025

#: Bar mass, in kilograms. 0.4 kg is the load-bearing calibration of the
#: coupling (R-2026-06-B; verified at Rung 1). It is heavy enough that a
#: single arm towing the free-pinned bar quasi-statically still lags it
#: far past the tilt limit (single-arm max tilt 72-89 deg over 15 seeds —
#: a free pin gives one arm zero orientation authority, so it cannot keep
#: the bar level), yet light enough that the matched pair's compliant
#: joint-delta controller places the bar inside the goal radius (matched
#: 15/15). A lighter bar is towable at low tilt and lets a single arm
#: succeed — the coupling is mass-calibrated, not purely geometric.
#: Frozen-grade.
COCARRY_BAR_MASS_KG: float = 0.4

#: Grip-frame offset along the ``panda_hand`` link z-axis, in metres.
#: The ``panda_hand_tcp`` frame sits 0.1034 m down the hand z-axis
#: (measured exactly for ``mani-skill==3.0.1`` panda); the dual-hold drive
#: welds the bar end to this frame in the hand link's local coordinates
#: (so the weld is independent of arm configuration) and the bar ends are
#: initialised at the two TCPs, giving the attach zero initial stress.
COCARRY_GRIP_OFFSET_Z_M: float = 0.1034

#: Goal centroid (bar-centre target) xyz, in metres, before per-episode
#: jitter. The bar starts level between the grippers at ~(0, 0, 0.17); the
#: goal is forward (+y) and up so the matched pair must **carry** the bar
#: level through a non-trivial transport (not just a vertical stretch,
#: which sits near the arms' ill-conditioned reach boundary). Chosen with
#: the Rung-1 geometric positive-control so a single arm holding one end
#: cannot bring the centroid here while keeping the bar level (the free
#: end pendulums past the tilt limit).
COCARRY_GOAL_CENTROID_XYZ: tuple[float, float, float] = (0.0, 0.12, 0.28)

#: Per-episode uniform jitter half-width on the goal centroid xyz, in
#: metres. Small (2 cm) so the task stays inside the jointly-reachable
#: envelope while giving the matched pair a non-degenerate goal
#: distribution. Routed through the P6 RNG (never ``torch.rand``).
COCARRY_GOAL_JITTER_HALF_M: float = 0.02

# ---------------------------------------------------------------------------
# Success thresholds (ADR-026 §Decision 1 "success must depend on the joint
# outcome"). Byte-frozen before any measurement.
# ---------------------------------------------------------------------------

#: Bar-centroid-to-goal success radius, in metres. 0.10 m is ~0.43x the
#: 0.23 m bar length. Calibrated so the matched pair places with margin
#: (Rung-1 matched centroid distances 0.087-0.10 m, all inside) while the
#: single-arm centroid (median 0.15 m) stays outside — though the coupling
#: is carried overwhelmingly by the tilt conjunct (matched ~5 deg vs
#: single-arm 72-89 deg), which a single arm fails by a wide margin
#: regardless of the distance radius.
COCARRY_GOAL_THRESH_M: float = 0.10

#: Maximum bar tilt from level over the episode, in degrees. A success
#: requires ``max_t theta(t) < COCARRY_TILT_MAX_DEG``. The level
#: constraint is one of the two coupling constraints that must *bind* on
#: matched successes (R-2026-06-B positive-control); a single arm cannot
#: hold a cantilevered bar below this tilt.
COCARRY_TILT_MAX_DEG: float = 15.0

#: Over-stress ceiling on the wrist constraint-solver force proxy, in
#: Newtons. A success requires the per-step proxy stay below this value
#: for the whole episode. **Derived from the Rung-1 matched-success
#: distribution** (15 seeds): the post-settle proxy on matched successes
#: ran at p50=90 N, p99=104 N, max=105 N, so f_max = 130 N is ~1.25x the
#: matched p99 — the constraint is active (matched successes run at
#: ~70-80% of it) without false-failing the reference, and it catches a
#: genuine over-stress event. Recorded here for the Rung-3+ pre-statement
#: (R-2026-06-B "stress proxy"). The tilt conjunct is the primary binding
#: coupling constraint; the stress proxy is the secondary physical guard.
#: Frozen-grade.
COCARRY_STRESS_MAX_PROXY_N: float = 130.0

#: Joint-velocity threshold (rad/s) for the per-arm ``is_static`` test in
#: the success predicate. Mirrors the upstream / Stage-1b ``is_static(0.2)``
#: convention; success requires **both** arms static at termination.
COCARRY_STATIC_QVEL_THRESH: float = 0.2

#: Settle window, in env ticks, excluded from the episode tilt/stress
#: maxima and from success eligibility (R-2026-06-B "name any settle term
#: as a frozen-able coefficient"). The dual-hold attach is created with
#: the bar warm-started at the two grip frames, but the arms ring for a
#: few control steps as the joint controllers settle from the reset pose;
#: that startup transient is an artifact of instantaneous placement, not
#: cooperative load, so the byte-frozen predicate ignores the first
#: ``COCARRY_SETTLE_WINDOW_STEPS`` ticks when accumulating ``max_tilt`` /
#: ``max_stress`` and when declaring success. Pre-registration-grade.
COCARRY_SETTLE_WINDOW_STEPS: int = 15

# ---------------------------------------------------------------------------
# Dual-hold drive (attach) stiffness/damping (R-2026-06-B "rigid-joint
# stability check gates Phase A"). High enough to read as rigid, damped
# enough that the Rung-0 stability smoke shows no solver blow-up.
# ---------------------------------------------------------------------------

#: Linear drive stiffness for the dual-hold attach, in N/m. The Rung-0..4
#: **rigid** coupling (the committed ladder on ``main``). The Rung-4b
#: compliant-coupling variant overrides this per-build via
#: ``make_cocarry_env(drive_stiffness=...)`` — see :func:`cocarry_coupling`.
COCARRY_DRIVE_LINEAR_STIFFNESS: float = 2.0e4

#: Linear drive damping for the dual-hold attach, in N*s/m (the rigid ladder).
COCARRY_DRIVE_LINEAR_DAMPING: float = 2.0e3

# ---------------------------------------------------------------------------
# Rung-4b compliant-coupling parameter (ADR-026 §Decision 4; R-2026-06-B §15
# Rung 4b; design COCARRY_RUNG4B_COMPLIANT_COUPLING). The rigid weld
# (20000 N/m) over-couples embodiment — a different-bodied partner fights the
# bar at ~518 N (Rung-4 feasibility wall, spikes/results/cocarry/rung4). The
# compliant variant LOWERS the linear-drive stiffness (Variant A) so the
# kinematic-mismatch fight force (= K_c x deflection) drops below f_max while
# the matched pair's tiny gravity deflection (~weight/2/K_c) stays << the
# 0.10 m radius. It is a PASSIVE spring-damper (no grasp, no learning); the
# action/obs spaces, success predicate, f_max, radius, bar, and goal are
# UNCHANGED. The chosen value is recorded in the rung4b freeze manifest, not
# frozen as a module constant (so the rigid ladder's constants are untouched).
# ---------------------------------------------------------------------------

#: Damping/stiffness ratio of the rigid attach (2000/20000 = 0.1, N*s/m per
#: N/m). The compliant sweep scales damping with stiffness at this ratio so a
#: lower-stiffness drive keeps the rig's proven overdamped, no-blow-up
#: character (the design's "damping scaled to ~critical" — this overdamped
#: ratio is conservative, well above the critical c=2*sqrt(K*m_eff) for the
#: ~0.2 kg per-hold effective mass at every swept K_c).
COCARRY_DRIVE_DAMPING_RATIO: float = COCARRY_DRIVE_LINEAR_DAMPING / COCARRY_DRIVE_LINEAR_STIFFNESS


#: PhysX default drive force-limit (effectively unbounded), N. The rigid
#: ladder + the Variant-A linear compliant sweep use this (no force cap).
COCARRY_DRIVE_FORCE_LIMIT_UNBOUNDED: float = 3.4028234663852886e38


def cocarry_coupling(
    drive_stiffness: float | None = None,
    drive_damping: float | None = None,
    drive_force_limit: float | None = None,
) -> tuple[float, float, float]:
    """Resolve the (stiffness, damping, force_limit) of the dual-hold drive (ADR-026 §D4; Rung 4b).

    Pure Tier-1 helper — the single source of truth for the coupling the env's
    :meth:`_add_weld` applies. ``drive_stiffness=None`` selects the **rigid**
    ladder default (:data:`COCARRY_DRIVE_LINEAR_STIFFNESS`, hard-locked axis);
    a float selects a Rung-4b **compliant** (freed-axis spring) variant.
    ``drive_damping=None`` derives damping as ``drive_stiffness *
    COCARRY_DRIVE_DAMPING_RATIO`` (keeps the rig's stable overdamped character
    at any stiffness); a float overrides it.

    Two compliant variants share this surface:

    - **Variant A** (linear): a lower stiffness, ``drive_force_limit=None``
      (unbounded) — the force is ``K_c x deflection`` throughout.
    - **Variant B** (force-saturated / bilinear): a stiff ``drive_stiffness``
      with a finite ``drive_force_limit`` near f_max — near-rigid for the
      small-force matched pair (stiff slope below the cap → small deflection,
      good placement) but the force SATURATES at ``drive_force_limit`` for a
      large kinematic mismatch (the cross-embodiment fight), so it cannot
      exceed the cap. This is the passive progressive law the design names; the
      force-limit is the knee.

    Args:
        drive_stiffness: Drive stiffness (N/m); ``None`` ⇒ rigid.
        drive_damping: Drive damping (N*s/m); ``None`` ⇒ derived.
        drive_force_limit: Max drive force (N); ``None`` ⇒ unbounded.

    Returns:
        ``(stiffness, damping, force_limit)`` for ``set_drive_property``.
    """
    k = COCARRY_DRIVE_LINEAR_STIFFNESS if drive_stiffness is None else float(drive_stiffness)
    c = float(k * COCARRY_DRIVE_DAMPING_RATIO) if drive_damping is None else float(drive_damping)
    fl = (
        COCARRY_DRIVE_FORCE_LIMIT_UNBOUNDED
        if drive_force_limit is None
        else float(drive_force_limit)
    )
    return k, c, fl


# ---------------------------------------------------------------------------
# Dense-reward shaping coefficients (R-2026-06-B "reward coefficients are
# pre-registration-grade"). EVERY shaping coefficient is a named, frozen-able
# module constant NOW; they get frozen at Rung 2 before any shifted teammate
# is seen. This closes the exact gap that contaminated AS — no coefficient is
# left implicit. Do not tune these against anything but the matched pair here.
# ---------------------------------------------------------------------------

#: Weight on the transport term (drive bar centroid toward the goal).
COCARRY_REWARD_TRANSPORT_COEFF: float = 1.0

#: Weight on the level term (keep the bar tilt small).
COCARRY_REWARD_LEVEL_COEFF: float = 1.0

#: Weight on the settle term (both arms static once the bar is placed).
COCARRY_REWARD_SETTLE_COEFF: float = 1.0

#: Terminal bonus added (and the reward saturated to) on a joint success.
COCARRY_REWARD_SUCCESS_BONUS: float = 5.0

#: ``tanh`` saturation scale shared by the transport + settle shaping
#: terms (per-metre / per-rad-per-s), mirroring the upstream pick-place
#: ``1 - tanh(5 * d)`` convention.
COCARRY_REWARD_TANH_SCALE: float = 5.0

#: ``tanh`` saturation scale for the level term, per radian of tilt.
COCARRY_REWARD_LEVEL_TANH_SCALE: float = 5.0

#: Normaliser dividing the summed reward so the output lands in roughly
#: ``[0, 1]`` (the success bonus saturates the numerator). Equals the
#: success bonus so a success maps to 1.0.
COCARRY_REWARD_NORMALIZER: float = 5.0

# ---------------------------------------------------------------------------
# Rung-2 reward remediation (COCARRY_RUNG2_REMEDIATION_2026-06-16). The
# Rung-2 train-to-reference STOP showed the learned ego climbs the dense
# reward (significant +slope) yet reaches 0% joint success because it
# *fights* the frozen partner — wrist constraint-solver stress p90 ~854 N,
# ~6.5x the f_max ceiling the success predicate gates on — and never
# transports the bar (centroid p50 0.37 m). Root cause: the dense reward
# rewarded transport+level+settle+success-bonus but had NO internal-stress
# term, so there was no gradient against the fight (reward-up / success-flat).
# These two additions are partner-agnostic (they penalise / shape physical
# quantities — internal force, distance-to-goal — never partner identity),
# named, and captured by the freeze enumerator. Set on a principled basis
# from the committed Step-1 distribution, NOT tuned to a target.
# ---------------------------------------------------------------------------

#: Weight on the excess-internal-stress penalty. Parity with the
#: transport/level/settle coefficients (1.0): the antagonistic fight is
#: penalised on the same scale cooperation is rewarded. Subtracted from the
#: reward sum (pre-normaliser).
COCARRY_REWARD_STRESS_COEFF: float = 1.0

#: Soft threshold (N) above which wrist constraint-solver stress is
#: penalised. **Re-grounded 2026-06-18** (cocarry_rung2_900k_train_stop.json):
#: the original tie to the success ceiling :data:`COCARRY_STRESS_MAX_PROXY_N`
#: (= f_max 130 N) bit only AT the failure edge — with the old 100 N tanh
#: scale the penalty at 133 N was ~0.03, no gradient to build margin, so the
#: budget-only 900k incumbent rode the edge and 2/12 seeds spiked the bar to
#: 133-134 N (on-goal but FAILING the unstressed conjunct). Now grounded at
#: the *demonstrated cooperative ceiling*: just above the committed Step-1
#: matched success-stress p99 (104.5 N) / max (104.9 N), within the pre-stated
#: matched-p99 band [85, 120] N, and 20 N below the 130 N success limit. The
#: matched pair (max 104.9 N < 110 N) still incurs ~0 penalty (the 100%
#: reference is unchanged); the policy is now pushed to stay INSIDE the
#: cooperative regime with margin rather than at the limit. Principled, set
#: once on the committed distribution — NOT tuned to a success target.
COCARRY_REWARD_STRESS_SOFT_THRESHOLD_N: float = 110.0

#: ``tanh`` saturation scale (N) for the excess-stress penalty:
#: ``penalty = COEFF * tanh(relu(stress - threshold) / scale)``. **Sharpened
#: 2026-06-18 (100 N -> 20 N)** to span the cooperative-ceiling -> success-limit
#: corridor (threshold 110 N -> limit 130 N = 20 N): the penalty now rises to
#: ~tanh(1) = 0.76*COEFF at the 130 N edge (was ~0.20) and ~0.46*COEFF at
#: 120 N, giving a real gradient that holds stress in the cooperative band
#: instead of letting it ride the edge. The fight excess (incumbent p90 ~854 N)
#: still saturates tanh ~ COEFF. Bounded => safe under the reward normaliser;
#: COEFF held at parity (1.0) so the bounded penalty never dominates the
#: cooperation terms (a large COEFF could make dropping the bar look optimal).
COCARRY_REWARD_STRESS_TANH_SCALE_N: float = 20.0

#: Weight on the policy-invariant transport PBRS potential
#: ``Phi = -coeff * dist(bar_centroid, goal)`` (Ng-Harada-Russell 1999;
#: applied as a training-time wrapper, :mod:`chamber.envs.cocarry_shaping`,
#: with ``F = gamma * Phi(s') - Phi(s)``). PBRS cannot change the optimum
#: (policy-invariant) — it only sharpens the gradient toward goal-directed
#: transport. Parity 1.0; bounded (the workspace distance is bounded). The
#: training config's ``shaping.transport_pbrs_coeff`` MUST equal this
#: constant (a Tier-1 parity test pins it; the freeze enumerator captures it).
COCARRY_REWARD_TRANSPORT_PBRS_COEFF: float = 1.0

# ---------------------------------------------------------------------------
# Robot / link wiring.
# ---------------------------------------------------------------------------

#: Index of the ``panda_hand`` link in the Panda articulation's ordered
#: link list (verified for ``mani-skill==3.0.1`` panda_wristcam /
#: panda_partner: ``[panda_link0..7, panda_link8, panda_hand(=9),
#: panda_hand_tcp, panda_leftfinger, panda_rightfinger, camera_base_link,
#: camera_link]``). Used to slice the wrist constraint-solver force from
#: ``get_link_incoming_joint_forces``. Resolved by name at construction
#: (this constant is the documented expectation, not the lookup path).
_HAND_LINK_NAME: str = "panda_hand"

#: xArm6 grip/TCP link the bar -x end is welded to in the embodiment-shifted
#: condition (``eef`` is the tool-center-point between the Robotiq fingers).
_XARM6_GRIP_LINK_NAME: str = "eef"

#: xArm6 "hand"/wrist link whose incoming joint force is its stress proxy (the
#: Robotiq gripper base — the analogue of ``panda_hand``).
_XARM6_HAND_LINK_NAME: str = "robotiq_arg2f_base_link"

#: Ego + partner uids (AS-homo convention from
#: :mod:`chamber.envs.stage1_pickplace`; the matched pair shares the
#: Panda body, the partner mounts the no-camera ``panda_v2.urdf`` via
#: :class:`chamber.agents.panda_partner.PandaPartner`).
_EGO_UID: str = "panda_wristcam"
_PARTNER_UID: str = "panda_partner"

#: Per-uid initial base pose: ``(xyz, quat_wxyz)``. The partner is yawed
#: pi about z so it faces the ego across the table (mirrored, deliberate
#: poses — R-2026-06-B). NOT the pick-place placeholders.
_AGENT_BASE_POSE: dict[
    str, tuple[tuple[float, float, float], tuple[float, float, float, float]]
] = {
    _EGO_UID: ((-COCARRY_BASE_X_M, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
    _PARTNER_UID: ((+COCARRY_BASE_X_M, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
}

#: Canonical ready qpos for each Panda (7 arm joints + 2 prismatic
#: fingers open at 0.04). Mirrors :mod:`chamber.envs.stage1_pickplace`'s
#: ``_PANDA_READY_QPOS``; the TableSceneBuilder has no branch for the
#: ``(panda_wristcam, panda_partner)`` tuple, so the env sets this
#: explicitly each reset (deterministic; no init noise — P6 / ADR-002).
_PANDA_READY_QPOS: NDArray[np.float64] = np.array(
    [0.0, np.pi / 8, 0.0, -np.pi * 5 / 8, 0.0, np.pi * 3 / 4, -np.pi / 4, 0.04, 0.04]
)

#: Retracted partner qpos for the single-arm positive-control: the
#: partner is disabled (no drive) and folded back out of the workspace so
#: it cannot incidentally support the bar. Arm pulled up + back; gripper
#: open. Only used when ``single_arm=True``.
_PANDA_RETRACTED_QPOS: NDArray[np.float64] = np.array(
    [0.0, -np.pi / 4, 0.0, -np.pi * 7 / 8, 0.0, np.pi * 5 / 8, -np.pi / 4, 0.04, 0.04]
)

#: Number of Panda arm DOF (excludes the two gripper fingers).
_PANDA_ARM_DOF: int = 7

#: Substream name routed through :func:`derive_substream` for the env's
#: deterministic RNG (P6 / ADR-002 determinism rule).
_SUBSTREAM_NAME: str = "env.cocarry"

# ---------------------------------------------------------------------------
# Rung-4 embodiment-shift partner: xArm6 + Robotiq 2F-85 (ADR-005 §Decision;
# ADR-026 §Decision 4; R-2026-06-B §15 Rung 4). The ego seat stays the Panda
# incumbent; the PARTNER seat is embodiment-configurable — Panda (the matched
# reference) or this xArm6 (the embodiment-shifted condition). Only the
# partner's body changes; the bar, goal, success predicate, and ego side stay
# byte-identical, so a measured drop isolates the embodiment.
# ---------------------------------------------------------------------------

#: The xArm6 + Robotiq 2F-85 partner uid (the built-in ManiSkill agent;
#: ADR-005). 6 arm joints + a Robotiq 2F-85 (12 qpos total, 7-D action).
_XARM6_PARTNER_UID: str = "xarm6_robotiq"

#: Number of xArm6 arm DOF (excludes the Robotiq gripper joints).
_XARM6_ARM_DOF: int = 6

#: xArm6 base x position, metres. The xArm6 faces -x (yaw pi about z) like the
#: Panda partner, but is mounted CLOSER to the workspace centre than the Panda
#: (0.35 m vs the Panda partner's 0.50 m) — a deliberate, geometry-verified
#: accommodation of the xArm6's shorter reach (R-2026-06-B §15 Rung 4 / spec
#: §2.1: "set the base so it can reach its bar end"). At 0.50 m the goal-region
#: bar-end target (~0.69 m away) sits at the xArm6's reach limit, so the
#: compliant controller tracks through ill-conditioned configurations and the
#: bar tilts; at 0.35 m the whole lift+transport path stays well-conditioned
#: (the longer-reach Panda needs no such move — that reach difference IS part
#: of the embodiment). The bar still starts identically (the ready ``eef`` is
#: at the same -x bar end), the goal and ego side are unchanged.
_XARM6_BASE_X_M: float = 0.35

#: xArm6 base pose — same facing as the Panda partner (yaw pi about z), mounted
#: at :data:`_XARM6_BASE_X_M`.
_XARM6_BASE_POSE: tuple[tuple[float, float, float], tuple[float, float, float, float]] = (
    (_XARM6_BASE_X_M, 0.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)

#: xArm6 ready qpos: 6 arm joints + 6 Robotiq joints. The arm joints are the
#: position-IK solution (for the :data:`_XARM6_BASE_X_M` base) placing the
#: ``eef`` tool-center-point at the partner -x bar end (world (-0.115, 0,
#: 0.1698) — the same point the Panda partner's TCP reaches), so the dual-hold
#: weld starts at ~zero stress with the bar spanning the identical 0.23 m gap.
#: The Robotiq joints are 0 (the gripper is inert — the bar is welded). Solved
#: offline via the xArm6 :class:`chamber.agents.xarm6_jacobian.XArm6JacobianProvider`
#: chain; deterministic, no init noise (P6 / ADR-002).
_XARM6_READY_QPOS: NDArray[np.float64] = np.array(
    [0.0, -0.084, -0.8, 0.0, 0.692, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
)

#: Per-uid initial base pose (ego Panda, partner Panda, partner xArm6).
_BASE_POSE_BY_UID: dict[
    str, tuple[tuple[float, float, float], tuple[float, float, float, float]]
] = {
    _EGO_UID: _AGENT_BASE_POSE[_EGO_UID],
    _PARTNER_UID: _AGENT_BASE_POSE[_PARTNER_UID],
    _XARM6_PARTNER_UID: _XARM6_BASE_POSE,
}

#: Per-uid ready qpos (Panda arms are 9-D; the xArm6 is 12-D).
_READY_QPOS_BY_UID: dict[str, NDArray[np.float64]] = {
    _EGO_UID: _PANDA_READY_QPOS,
    _PARTNER_UID: _PANDA_READY_QPOS,
    _XARM6_PARTNER_UID: _XARM6_READY_QPOS,
}

#: Per-uid link the bar end is rigidly welded to (the grip frame). The Panda
#: welds the ``panda_hand`` link offset to the TCP; the xArm6 welds the ``eef``
#: link (which IS its TCP, so the offset is identity).
_WELD_LINK_BY_UID: dict[str, str] = {
    _EGO_UID: _HAND_LINK_NAME,
    _PARTNER_UID: _HAND_LINK_NAME,
    _XARM6_PARTNER_UID: _XARM6_GRIP_LINK_NAME,
}

#: Per-uid grip-frame offset along the weld link's local z (m). Panda TCP is
#: 0.1034 m down the hand link; the xArm6 ``eef`` is itself the TCP (offset 0).
_GRIP_OFFSET_Z_BY_UID: dict[str, float] = {
    _EGO_UID: COCARRY_GRIP_OFFSET_Z_M,
    _PARTNER_UID: COCARRY_GRIP_OFFSET_Z_M,
    _XARM6_PARTNER_UID: 0.0,
}

#: Per-uid link whose incoming joint force is the wrist constraint-solver
#: stress proxy. Panda reads ``panda_hand``; the xArm6 reads the Robotiq
#: gripper base (its "hand" — the bar load flows through its incoming joint),
#: the direct analogue, so the proxy stays comparable in Newtons to the
#: Panda-derived f_max (130.6 N) the success predicate gates on.
_STRESS_LINK_BY_UID: dict[str, str] = {
    _EGO_UID: _HAND_LINK_NAME,
    _PARTNER_UID: _HAND_LINK_NAME,
    _XARM6_PARTNER_UID: _XARM6_HAND_LINK_NAME,
}


class CoCarryCondition(NamedTuple):
    """Resolved co-carry condition configuration (ADR-026 §Decision 1).

    Pure-Python NamedTuple so Tier-1 tests can construct / assert without
    any SAPIEN dependency.

    Attributes:
        condition_id: Verbatim condition string.
        agent_uids: Ordered ``(ego_uid, partner_uid)`` tuple. For the
            matched pair both run a copy of the hand-written co-carry
            impedance controller; the partner is routed through the
            partner interface (ADR-009) so the rig is forward-compatible
            with the Rung-2 frozen incumbent.
        single_arm: ``True`` only for the coupling positive-control
            condition — the partner is disabled (no dual-hold drive,
            retracted out of the workspace) so a single arm attempts the
            task. Expected success ~= 0 (R-2026-06-B; ADR-026 §Decision 2).
    """

    condition_id: str
    agent_uids: tuple[str, str]
    single_arm: bool


#: condition_id -> resolved configuration. Two conditions in this slice:
#: the matched pair (the honest high reference) and the single-arm
#: positive-control (expected ~= 0). Both share the identical Panda body
#: and the byte-frozen success predicate; the positive-control differs
#: only in whether the partner participates (ADR-026 §Decision 2 — the
#: matched-pair-vs-single-robot contrast that makes the criterion
#: falsifiable).
_CONDITION_TABLE: dict[str, CoCarryCondition] = {
    "cocarry_matched_panda_pair": CoCarryCondition(
        condition_id="cocarry_matched_panda_pair",
        agent_uids=(_EGO_UID, _PARTNER_UID),
        single_arm=False,
    ),
    "cocarry_single_arm_positive_control": CoCarryCondition(
        condition_id="cocarry_single_arm_positive_control",
        agent_uids=(_EGO_UID, _PARTNER_UID),
        single_arm=True,
    ),
    # Rung-4 embodiment shift (ADR-026 §Decision 4; ADR-005; R-2026-06-B §15
    # Rung 4): the Panda ego incumbent + an xArm6 + Robotiq 2F-85 partner. The
    # partner body is the ONLY change vs the matched pair — same bar, goal,
    # success predicate, ego side. agent_uids[1] is the partner uid the env
    # resolves the body / base pose / weld / stress link from.
    "cocarry_xarm6_partner": CoCarryCondition(
        condition_id="cocarry_xarm6_partner",
        agent_uids=(_EGO_UID, _XARM6_PARTNER_UID),
        single_arm=False,
    ),
}


def resolve_cocarry_condition(condition_id: str) -> CoCarryCondition:
    """Resolve a co-carry ``condition_id`` to its config (ADR-026 §Decision 1).

    Pure Tier-1 function — no SAPIEN dependency.

    Args:
        condition_id: ``"cocarry_matched_panda_pair"`` (the matched
            reference) or ``"cocarry_single_arm_positive_control"`` (the
            coupling positive-control).

    Returns:
        The resolved :class:`CoCarryCondition`.

    Raises:
        ValueError: If ``condition_id`` is unknown. The message lists the
            valid options and cites ADR-026 §Decision.
    """
    try:
        return _CONDITION_TABLE[condition_id]
    except KeyError as exc:
        msg = (
            f"CoCarryEnv: condition_id {condition_id!r} is not one of the co-carry "
            f"conditions {sorted(_CONDITION_TABLE)!r}. This Rung-0/1 slice ships the "
            "matched pair + the single-arm coupling positive-control only "
            "(ADR-026 §Decision 1-2; R-2026-06-B Rungs 0-1)."
        )
        raise ValueError(msg) from exc


def tilt_deg_from_quaternion(quat_wxyz: ArrayLike) -> NDArray[np.float64]:
    r"""Bar tilt from level, in degrees, from the bar pose quaternion (ADR-026 §Decision 1).

    The bar's long axis is its local x-axis. Tilt is the angle between
    that axis and the world horizontal plane:
    ``theta = arcsin(|world_long_axis_z|)``. Level => 0 deg; vertical =>
    90 deg.

    Pure Tier-1 function (numpy); the env's per-step telemetry calls the
    torch path inline. Accepts a single ``(4,)`` quaternion or a batched
    ``(..., 4)`` array in ManiSkill's ``wxyz`` order.

    Args:
        quat_wxyz: Quaternion(s) in ``(w, x, y, z)`` order, shape
            ``(4,)`` or ``(..., 4)``.

    Returns:
        Tilt in degrees, shape ``()`` (0-d) for a single quaternion or
        ``(...)`` for a batch.
    """
    q = np.asarray(quat_wxyz, dtype=np.float64)
    w = q[..., 0]
    x = q[..., 1]
    y = q[..., 2]
    z = q[..., 3]
    # World z-component of the body local x-axis (first column of the
    # rotation matrix): R[2, 0] = 2 (x z - w y).
    long_axis_z = 2.0 * (x * z - w * y)
    # Norm of the full local-x world vector for a defensive renormalise
    # (unit quaternions give |long_axis| = 1; guard tiny drift).
    ax_x = 1.0 - 2.0 * (y * y + z * z)
    ax_y = 2.0 * (x * y + w * z)
    norm = np.sqrt(ax_x * ax_x + ax_y * ax_y + long_axis_z * long_axis_z)
    norm = np.where(norm > 0.0, norm, 1.0)
    sin_tilt = np.clip(np.abs(long_axis_z) / norm, 0.0, 1.0)
    return np.degrees(np.arcsin(sin_tilt))


def evaluate_cocarry_success(
    *,
    centroid_to_goal_dist: NDArray[np.floating] | float,
    max_tilt_deg: NDArray[np.floating] | float,
    max_stress_proxy: NDArray[np.floating] | float,
    both_static: NDArray[np.bool_] | bool,
    goal_thresh: float = COCARRY_GOAL_THRESH_M,
    tilt_max_deg: float = COCARRY_TILT_MAX_DEG,
    stress_max: float = COCARRY_STRESS_MAX_PROXY_N,
) -> NDArray[np.bool_]:
    """Joint co-carry success predicate (ADR-026 §Decision 1; byte-frozen).

    Success iff **all** of:

    - the bar centroid is within ``goal_thresh`` of the goal;
    - the maximum bar tilt over the episode is ``< tilt_max_deg``;
    - no over-stress event occurred (episode-max stress proxy ``<
      stress_max``);
    - **both** robots are static at termination.

    Unlike the AS pick-place predicate (which read only the ego), this is
    a **joint** outcome: the tilt + stress conjuncts can only be held
    when two holds share the load, so the score measures cooperation
    (the coupling criterion). Pure Tier-1 function — the Tier-1 tests
    pin this logic on synthetic poses.

    Args:
        centroid_to_goal_dist: Bar-centroid-to-goal distance, metres.
        max_tilt_deg: Episode-max bar tilt, degrees.
        max_stress_proxy: Episode-max wrist constraint-solver force
            proxy, Newtons.
        both_static: Whether both arms are below the static qvel
            threshold at the call.
        goal_thresh: Placement radius (default
            :data:`COCARRY_GOAL_THRESH_M`).
        tilt_max_deg: Tilt ceiling (default :data:`COCARRY_TILT_MAX_DEG`).
        stress_max: Stress ceiling (default
            :data:`COCARRY_STRESS_MAX_PROXY_N`).

    Returns:
        Boolean array (broadcast of the inputs) — the per-env success.
    """
    placed = np.asarray(centroid_to_goal_dist) <= goal_thresh
    level = np.asarray(max_tilt_deg) < tilt_max_deg
    unstressed = np.asarray(max_stress_proxy) < stress_max
    static = np.asarray(both_static, dtype=bool)
    return np.asarray(placed & level & unstressed & static, dtype=bool)


def cocarry_excess_stress_penalty(
    stress: ArrayLike,
    *,
    coeff: float = COCARRY_REWARD_STRESS_COEFF,
    threshold: float = COCARRY_REWARD_STRESS_SOFT_THRESHOLD_N,
    scale: float = COCARRY_REWARD_STRESS_TANH_SCALE_N,
) -> Any:  # noqa: ANN401 - returns torch.Tensor or np.ndarray mirroring the input backend
    """Excess-internal-stress reward penalty (ADR-026 §Decision 4; Rung-2 remediation).

    ``penalty = coeff * tanh(relu(stress - threshold) / scale)`` — zero at or
    below the soft threshold :data:`COCARRY_REWARD_STRESS_SOFT_THRESHOLD_N`
    (110 N, the demonstrated cooperative ceiling; re-grounded 2026-06-18),
    rising monotonically and saturating at ``coeff`` for large excess. By
    construction the matched cooperative band (Step-1 success-stress p99 ≈
    104.5 N, max ≈ 104.9 N < threshold 110 N) incurs ≈0 penalty (so the 100%
    reference is unchanged); with the 20 N tanh scale it rises to ≈0.76·coeff
    at the 130 N success limit — a real gradient holding stress inside the
    cooperative regime, not at the edge — and saturates on the antagonistic
    fight (incumbent p90 ≈ 854 N). Partner-agnostic — penalises a physical
    constraint force, never partner identity (spec §11 anti-leakage).

    Backend-agnostic: a torch tensor in (the env reward path) returns a torch
    tensor; a numpy/array-like in (the Tier-1 shape test) returns an ndarray.
    A single formula, so the env reward and the Tier-1 shape guard cannot
    drift.
    """
    if hasattr(stress, "detach"):  # torch.Tensor (the env reward path)
        import torch

        excess = torch.clamp(stress - threshold, min=0.0)  # type: ignore[operator]
        return coeff * torch.tanh(excess / scale)
    arr = np.asarray(stress, dtype=np.float64)
    excess = np.maximum(arr - threshold, 0.0)
    return coeff * np.tanh(excess / scale)


def cocarry_matched_controller_specs() -> dict[str, dict[str, str]]:
    """Per-uid ``spec.extra`` dicts for the matched co-carry controllers (ADR-026 §Decision 1).

    Pure Tier-1 function — derives the geometry the matched
    :class:`chamber.partners.cocarry_impedance.CoCarryImpedancePartner`
    pair needs (base pose, bar-end sign, half-length) from this module's
    constants, so the controller geometry has a single source of truth at
    the env layer. Both arms hold the bar: the ego holds the ``+x`` end
    (``end_sign="1"``), the partner the ``-x`` end (``end_sign="-1"``).
    The Rung-0/1 runner and tests pass these straight into
    :class:`chamber.partners.api.PartnerSpec`'s ``extra``.

    Returns:
        ``{uid: extra_dict}`` for the ego + partner uids.
    """
    half = repr(COCARRY_BAR_LENGTH_M / 2.0)
    ego_xyz, _ = _AGENT_BASE_POSE[_EGO_UID]
    partner_xyz, _ = _AGENT_BASE_POSE[_PARTNER_UID]
    return {
        _EGO_UID: {
            "uid": _EGO_UID,
            "base_xyz": f"{ego_xyz[0]},{ego_xyz[1]},{ego_xyz[2]}",
            "base_yaw_deg": "0",
            "end_sign": "1",
            "bar_half_len": half,
        },
        _PARTNER_UID: {
            "uid": _PARTNER_UID,
            "base_xyz": f"{partner_xyz[0]},{partner_xyz[1]},{partner_xyz[2]}",
            "base_yaw_deg": "180",
            "end_sign": "-1",
            "bar_half_len": half,
        },
    }


def cocarry_xarm6_controller_spec() -> dict[str, str]:
    """``spec.extra`` for the Rung-4 xArm6 partner controller (ADR-026 §Decision 4; ADR-005).

    Pure Tier-1 function — derives the xArm6 partner-seat geometry (its base
    pose, the -x bar-end sign, half-length) from this module's constants, so
    the :class:`chamber.partners.cocarry_xarm6.CoCarryXArm6Partner` controller's
    frame transform uses the SAME base pose the env mounts the xArm6 at
    (:data:`_XARM6_BASE_X_M` = 0.35 m, yaw pi) — the single source of truth.
    The xArm6 mounts closer than the Panda partner (its shorter reach), so this
    base x differs from :func:`cocarry_matched_controller_specs`'s partner.

    Returns:
        The ``extra`` dict for the ``xarm6_robotiq`` partner seat.
    """
    return {
        "uid": _XARM6_PARTNER_UID,
        "base_xyz": f"{_XARM6_BASE_X_M},0.0,0.0",
        "base_yaw_deg": "180",
        "end_sign": "-1",
        "bar_half_len": repr(COCARRY_BAR_LENGTH_M / 2.0),
    }


#: Default episode horizon, in env ticks. The quasi-static carry (lift +
#: forward transport + settle) converges inside ~320 control steps at the
#: 20 Hz control rate; sized so the matched pair's PI controller closes
#: the load offset before truncation.
COCARRY_DEFAULT_EPISODE_LENGTH: int = 320


def make_cocarry_env(
    *,
    condition_id: str = "cocarry_matched_panda_pair",
    episode_length: int = COCARRY_DEFAULT_EPISODE_LENGTH,
    root_seed: int = 0,
    num_envs: int = 1,
    render_mode: str | None = None,
    render_backend: str | None = None,
    goal_centroid: tuple[float, float, float] | None = None,
    drive_stiffness: float | None = None,
    drive_damping: float | None = None,
    drive_force_limit: float | None = None,
) -> gym.Env[Any, Any]:
    """Build a :class:`CoCarryEnv` instance (ADR-026 §Decision 1-2; R-2026-06-B Rungs 0-1).

    Factory entry point. SAPIEN / ManiSkill imports are deferred to the
    body so ``python -c "import chamber.envs.cocarry"`` works on a
    Vulkan-less host (Tier-1 contract). The :class:`CoCarryEnv` class is
    defined inside the factory body — same pattern as
    :func:`chamber.envs.stage1_pickplace.make_stage1_pickplace_env`.

    Args:
        condition_id: ``"cocarry_matched_panda_pair"`` (matched
            reference) or ``"cocarry_single_arm_positive_control"`` (the
            partner is disabled + retracted; expected success ~= 0).
        episode_length: Truncation horizon, in env ticks. 120 gives the
            quasi-static carry room to lift + transport + settle.
        root_seed: Project root seed routed to the env's
            :func:`derive_substream` substream (P6 / ADR-002).
        num_envs: ManiSkill vectorisation count; default 1.
        render_mode: ManiSkill ``render_mode`` (``None`` disables).
        render_backend: ManiSkill ``render_backend``; ``"none"`` enables
            the headless URDF-material strip.
        goal_centroid: Optional override of the goal centroid xyz (before
            jitter). ``None`` uses :data:`COCARRY_GOAL_CENTROID_XYZ`.
        drive_stiffness: Dual-hold drive stiffness (N/m); ``None`` ⇒ the rigid
            ladder default. A lower value selects the Rung-4b compliant
            coupling (:func:`cocarry_coupling`; ADR-026 §D4 Rung 4b).
        drive_damping: Dual-hold drive damping (N*s/m); ``None`` ⇒ derived from
            the stiffness at the rig's damping ratio.
        drive_force_limit: Max drive force (N); ``None`` ⇒ unbounded. A finite
            value near f_max selects the Variant-B force-saturated coupling.

    Returns:
        A :class:`CoCarryEnv` ready to ``reset(seed=K)``.

    Raises:
        ValueError: If ``condition_id`` is unknown.
        ChamberEnvCompatibilityError: If ManiSkill / SAPIEN / Vulkan
            initialisation fails.
    """
    config = resolve_cocarry_condition(condition_id)
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
            "CoCarryEnv requires mani_skill / sapien in the active venv. "
            "Install per pyproject.toml; see ADR-001 §Risks and ADR-005 §Decision."
        ) from exc

    if render_backend == "none":
        patch_sapien_urdf_no_visual_material()

    goal_xyz = COCARRY_GOAL_CENTROID_XYZ if goal_centroid is None else goal_centroid
    # Resolve the dual-hold coupling once (rigid default, or the Rung-4b
    # compliant override); _add_weld closes over these (ADR-026 §D4 Rung 4b).
    # A stiffness override selects the compliant (freed-axis spring) weld; the
    # default keeps the rigid (hard-locked) weld byte-identical.
    weld_compliant = drive_stiffness is not None
    weld_stiffness, weld_damping, weld_force_limit = cocarry_coupling(
        drive_stiffness, drive_damping, drive_force_limit
    )

    class CoCarryEnv(BaseEnv):  # type: ignore[misc, valid-type]
        """Two-arm rigid-bar co-carry env (ADR-026 §Decision 1-2; R-2026-06-B Rungs 0-1).

        Subclasses :class:`mani_skill.envs.sapien_env.BaseEnv` directly
        (matching :class:`chamber.envs.stage1_pickplace.Stage1PickPlaceEnv`).
        See the module docstring for the coupling rationale, the
        dual-hold attach, and the wrist constraint-solver stress proxy.
        """

        SUPPORTED_ROBOTS: ClassVar[list[tuple[str, str]]] = [  # type: ignore[assignment]
            (_EGO_UID, config.agent_uids[1]),
        ]

        def __init__(self) -> None:
            self._config: CoCarryCondition = config
            self._episode_length: int = int(episode_length)
            self._root_seed: int = int(root_seed)
            self._single_arm: bool = config.single_arm
            # The ego seat is always the Panda incumbent; the PARTNER seat is
            # embodiment-configurable (Panda or xArm6) — agent_uids[1] resolves
            # the body, base pose, weld, and stress link (ADR-026 §D4 Rung 4).
            self._partner_uid: str = config.agent_uids[1]
            self._goal_centroid_base: tuple[float, float, float] = goal_xyz
            self._rng: np.random.Generator = derive_substream(
                _SUBSTREAM_NAME, root_seed=self._root_seed
            ).default_rng()
            # Per-episode goal centroid (num_envs, 3), running maxima, and the
            # cached per-uid hand-link indices are populated below / at reset.
            # Assigned before super().__init__ because BaseEnv.__init__
            # calls reset() -> _initialize_episode during super-init.
            self._hand_link_index_by_uid: dict[str, int] = {}
            self._drives: list[Any] = []
            try:
                super().__init__(
                    robot_uids=(_EGO_UID, self._partner_uid),  # type: ignore[arg-type]
                    num_envs=num_envs,
                    obs_mode="state_dict",
                    control_mode="pd_joint_delta_pos",
                    render_mode=render_mode,
                    render_backend=render_backend if render_backend is not None else "gpu",
                )
            except RuntimeError as exc:
                raise ChamberEnvCompatibilityError(
                    "SAPIEN/Vulkan initialisation failed during "
                    f"CoCarryEnv(condition_id={condition_id!r}) build: {exc}\n"
                    'Set render_backend="none" on CUDA-only hosts; see ADR-001 §Risks.'
                ) from exc

        # ----- ManiSkill v3 BaseEnv hooks -----

        @property
        def _default_human_render_camera_configs(self) -> Any:  # noqa: ANN401 - ManiSkill CameraConfig has no project type
            """Third-person render camera for rollout visualisation (ADR-026 §Decision 1)."""
            from mani_skill.sensors.camera import CameraConfig
            from mani_skill.utils import sapien_utils

            pose = sapien_utils.look_at(eye=[0.0, 1.0, 0.7], target=[0.0, 0.0, 0.25])
            return CameraConfig(
                "render_camera", pose=pose, width=640, height=480, fov=1.0, near=0.01, far=100.0
            )

        def _load_agent(  # type: ignore[override]
            self,
            options: dict[str, Any],
            initial_agent_poses: object = None,
            build_separate: bool = False,
        ) -> None:
            """Two-Panda ``_load_agent`` with deliberate mirrored base poses (R-2026-06-B)."""
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
            """Build table + rigid bar + goal site, then weld the dual-hold drives.

            Drives must be created before the GPU sim is initialised
            (SAPIEN ``@before_gpu_init`` guard on ``set_drive_property_*``
            / ``set_limit_*``), so the weld is created here in
            ``_load_scene`` (run during reconfigure). The weld is
            specified in each body's *local* frame — ``panda_hand`` grip
            frame to bar end — so it holds across arm configurations.
            """
            del options
            self.table_scene = TableSceneBuilder(self)
            self.table_scene.build()

            half = [
                COCARRY_BAR_LENGTH_M / 2.0,
                COCARRY_BAR_CROSS_SECTION_M / 2.0,
                COCARRY_BAR_CROSS_SECTION_M / 2.0,
            ]
            volume = (
                COCARRY_BAR_LENGTH_M * COCARRY_BAR_CROSS_SECTION_M * COCARRY_BAR_CROSS_SECTION_M
            )
            builder = self.scene.create_actor_builder()
            builder.add_box_collision(
                half_size=half,  # type: ignore[arg-type]
                density=COCARRY_BAR_MASS_KG / volume,
            )
            builder.add_box_visual(
                half_size=half,  # type: ignore[arg-type]
                material=sapien.render.RenderMaterial(base_color=[0.62, 0.42, 0.12, 1.0]),
            )
            builder.initial_pose = sapien.Pose(p=[0.0, 0.0, 0.35])  # type: ignore[assignment]
            self.bar = builder.build(name="cocarry_bar")

            self.goal_site = actors.build_sphere(
                self.scene,
                radius=COCARRY_GOAL_THRESH_M,
                color=[0, 1, 0, 1],
                name="goal_site",
                body_type="kinematic",
                add_collision=False,
                initial_pose=sapien.Pose(),
            )
            self._hidden_objects.append(self.goal_site)  # type: ignore[attr-defined]

            # Cache the per-uid stress-link index from each articulation's
            # ordered link list. The Panda reads ``panda_hand``; the xArm6
            # reads its Robotiq gripper base (the analogue wrist). Per-uid
            # because the link orderings differ across embodiments (Rung 4).
            for uid in self.robot_uids:  # type: ignore[union-attr]
                links = self.agent.agents_dict[uid].robot.links  # type: ignore[attr-defined]
                stress_link = _STRESS_LINK_BY_UID[uid]
                self._hand_link_index_by_uid[uid] = [ln.name for ln in links].index(stress_link)

            # Dual-hold weld(s). Ego always holds; the partner holds only
            # in the matched / embodiment-shift conditions (single-arm
            # positive-control omits the partner drive so a single arm
            # attempts the task). At the ready pose the two grippers reach
            # across the table centre: the ego TCP sits at world +x, the
            # partner TCP/eef at world -x (measured: +-0.115 m). So the ego
            # holds the bar's +x end and the partner the -x end. The grip
            # offset + weld link are per-uid (the xArm6 ``eef`` is its TCP,
            # offset 0; the Panda welds ``panda_hand`` offset to its TCP).
            self._drives = []
            self._add_weld(_EGO_UID, sapien.Pose(p=[COCARRY_BAR_LENGTH_M / 2.0, 0.0, 0.0]))
            if not self._single_arm:
                self._add_weld(
                    self._partner_uid, sapien.Pose(p=[-COCARRY_BAR_LENGTH_M / 2.0, 0.0, 0.0])
                )

        def _add_weld(self, uid: str, bar_end_local: Any) -> None:  # noqa: ANN401 - sapien.Pose
            """Create one high-stiffness linear pin between a gripper and a bar end.

            The attach locks the three *linear* DOF (the bar end tracks
            the gripper grip frame with high stiffness/damping) but leaves
            rotation free — a ball-and-socket pin. The bar is thus a rigid
            link suspended between the two grip pins; its tilt is the angle
            of the line connecting them, which the two arms govern jointly
            through their relative TCP height. A single pin leaves the bar
            a free pendulum (it cannot be held level), which is the
            physical mechanism of the coupling positive-control
            (ADR-026 §Decision 2). Leaving rotation free also keeps the
            transmitted load a pure linear force, which is exactly what
            the wrist constraint-solver stress proxy reads. The weld link +
            grip-in-link offset are per-uid (Panda ``panda_hand`` offset to
            TCP; xArm6 ``eef`` offset 0 — Rung 4 embodiment shift).
            """
            grip_in_hand = sapien.Pose(p=[0.0, 0.0, _GRIP_OFFSET_Z_BY_UID[uid]])
            weld_link = _WELD_LINK_BY_UID[uid]
            hand = self.agent.agents_dict[uid].robot.links_map[weld_link]  # type: ignore[attr-defined]
            drive = self.scene.create_drive(hand, grip_in_hand, self.bar, bar_end_local)
            for axis in ("x", "y", "z"):
                if weld_compliant:
                    # Rung-4b COMPLIANT coupling (ADR-026 §D4 Rung 4b): a passive
                    # linear spring-damper. The axis is freed (a wide ±1 m limit,
                    # far beyond any real deflection) so the drive — toward its
                    # default zero target (the bar end rests at the grip frame) —
                    # acts as a restoring spring of stiffness ``weld_stiffness``.
                    # A hard set_limit(0,0) would LOCK the axis (rigid) and make
                    # the drive stiffness inert, so it is deliberately NOT used.
                    getattr(drive, f"set_limit_{axis}")(-1.0, 1.0)
                    # force_limit caps the drive force (Variant B's knee): the
                    # spring is stiff below the cap (near-rigid for the matched
                    # pair) but saturates at weld_force_limit for a large
                    # mismatch, so the cross-embodiment fight cannot exceed it.
                    getattr(drive, f"set_drive_property_{axis}")(
                        weld_stiffness, weld_damping, weld_force_limit
                    )
                else:
                    # Rigid weld (the committed Rung-0..4 ladder): the axis is
                    # hard-locked to zero displacement; the high drive stiffness
                    # is belt-and-braces. Byte-identical to the shipped behaviour.
                    getattr(drive, f"set_drive_property_{axis}")(weld_stiffness, weld_damping)
                    getattr(drive, f"set_limit_{axis}")(0.0, 0.0)
            self._drives.append(drive)

        def _initialize_episode(self, env_idx: torch.Tensor, options: dict[str, Any]) -> None:
            """Reset ready poses, place the bar level between the grippers, sample the goal.

            All randomisation routes through the P6 RNG (never
            ``torch.rand``) so two ``reset(seed=K)`` calls reproduce
            byte-identical state (P6 / ADR-002).
            """
            del options
            with torch.device(self.device):  # type: ignore[attr-defined]
                b = len(env_idx)
                self.table_scene.initialize(env_idx)
                # Ready pose per uid (the xArm6 partner is 12-D, the Pandas
                # 9-D); ready (matched / embodiment shift) or retracted
                # (single-arm) for the partner.
                for uid in self.robot_uids:  # type: ignore[union-attr]
                    if uid == self._partner_uid and self._single_arm:
                        ready = _PANDA_RETRACTED_QPOS
                    else:
                        ready = _READY_QPOS_BY_UID[uid]
                    qpos = torch.from_numpy(np.tile(ready, (b, 1)).astype(np.float32)).to(
                        self.device
                    )
                    self.agent.agents_dict[uid].reset(qpos)  # type: ignore[attr-defined]

                # Place the bar level, oriented along world x, so its
                # ends coincide with the gripper grip frames at the start
                # (zero initial weld stress). Matched: bar centre is the
                # midpoint of the two TCPs. Single-arm: the partner is
                # retracted, so anchor on the ego TCP (the +x end) and
                # extend the bar toward where the partner would hold
                # (world -x); the free end then droops under gravity.
                ego_tcp = self.agent.agents_dict[_EGO_UID].tcp_pose.p  # type: ignore[attr-defined]
                ego_tcp_np = np.asarray(ego_tcp.detach().cpu(), dtype=np.float64).reshape(b, 3)
                if self._single_arm:
                    bar_center = ego_tcp_np.copy()
                    bar_center[:, 0] -= COCARRY_BAR_LENGTH_M / 2.0
                else:
                    partner_tcp = self.agent.agents_dict[self._partner_uid].tcp_pose.p  # type: ignore[attr-defined]
                    partner_tcp_np = np.asarray(
                        partner_tcp.detach().cpu(), dtype=np.float64
                    ).reshape(b, 3)
                    bar_center = (ego_tcp_np + partner_tcp_np) / 2.0
                self.bar.set_pose(
                    Pose.create_from_pq(
                        torch.from_numpy(bar_center.astype(np.float32)).to(self.device)
                    )
                )

                # Sample the per-episode goal centroid (P6 RNG jitter).
                jitter = self._rng.uniform(
                    -COCARRY_GOAL_JITTER_HALF_M, COCARRY_GOAL_JITTER_HALF_M, size=(b, 3)
                )
                goal = np.asarray(self._goal_centroid_base, dtype=np.float64)[None, :] + jitter
                self._goal_centroid = goal
                self.goal_site.set_pose(
                    Pose.create_from_pq(torch.from_numpy(goal.astype(np.float32)).to(self.device))
                )

                # Reset per-episode running maxima (tilt + stress proxy).
                self._max_tilt_deg = torch.zeros(b, device=self.device)
                self._max_stress = torch.zeros(b, device=self.device)

        def step(self, action: Any) -> tuple[Any, Any, Any, Any, dict[str, Any]]:  # type: ignore[override]  # noqa: ANN401
            """Enforce ``episode_length`` time-limit truncation (ADR-026 §Decision 1)."""
            obs, reward, terminated, truncated, info = super().step(action)
            truncated = truncated | (self.elapsed_steps >= self._episode_length)
            return obs, reward, terminated, truncated, info

        # ----- Telemetry / instrumentation (ADR-026 §Decision 1) -----

        def _bar_tilt_deg_now(self) -> torch.Tensor:
            """Current bar tilt from level, degrees, per env (torch path)."""
            q = self.bar.pose.q  # (b, 4) wxyz
            w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
            long_axis_z = 2.0 * (x * z - w * y)
            sin_tilt = torch.clamp(torch.abs(long_axis_z), 0.0, 1.0)
            return torch.rad2deg(torch.arcsin(sin_tilt))

        def _wrist_stress_proxy_now(self) -> torch.Tensor:
            """Current wrist constraint-solver force proxy, Newtons, per env.

            The stress proxy is the linear-force norm of each holding arm's
            wrist-link incoming joint force (``panda_hand`` for the Panda,
            the Robotiq gripper base for the xArm6 — the per-uid analogue),
            taken over the **holding** arms and reduced by max. For the
            matched / embodiment-shift pair both arms hold; for the
            single-arm positive-control only the ego holds. See the module
            docstring for why this articulation-solver force is the
            faithful proxy (SAPIEN exposes no free-drive force).
            """
            holding = [_EGO_UID] if self._single_arm else [_EGO_UID, self._partner_uid]
            per_arm: list[torch.Tensor] = []
            for uid in holding:
                robot = self.agent.agents_dict[uid].robot  # type: ignore[attr-defined]
                forces = robot.get_link_incoming_joint_forces()  # (b, n_links, 6)
                wrist = forces[:, self._hand_link_index_by_uid[uid], :3]  # linear force
                per_arm.append(torch.linalg.norm(wrist, axis=1))
            return torch.stack(per_arm, dim=0).amax(dim=0)

        def _centroid_to_goal_dist_now(self) -> torch.Tensor:
            """Current bar-centroid-to-goal distance, metres, per env."""
            import torch as _torch

            goal = _torch.from_numpy(self._goal_centroid.astype(np.float32)).to(self.device)
            return _torch.linalg.norm(self.bar.pose.p - goal, axis=1)

        def get_telemetry(self) -> dict[str, Any]:
            """Per-step telemetry dict (ADR-026 §Decision 1; R-2026-06-B instrumentation).

            Returns the current bar tilt (deg), wrist stress proxy (N),
            and bar-centroid-to-goal distance (m), plus the per-episode
            running maxima the success predicate reads. Consumed by the
            Rung-0 stability smoke and the Rung-1 constraint-binding
            characterisation (these are live handles, not obs-derived, so
            they are privileged by construction).
            """
            return {
                "tilt_deg": self._bar_tilt_deg_now(),
                "stress_proxy": self._wrist_stress_proxy_now(),
                "centroid_to_goal": self._centroid_to_goal_dist_now(),
                "max_tilt_deg": self._max_tilt_deg,
                "max_stress_proxy": self._max_stress,
            }

        def privileged_transport_distance(self) -> torch.Tensor:
            """Per-env bar-centroid-to-goal distance from live state (ADR-026 §Decision 1).

            The privileged accessor the Rung-2 transport-PBRS wrapper
            (:mod:`chamber.envs.cocarry_shaping`) reads at step entry/exit,
            so the potential is a function of privileged env state, never
            the (synthesised) observation — mirrors Stage-1's
            ``privileged_settle_state`` (issue #232; ADR-007 §Stage 1b Rev 18).
            Training-time reward computation is privileged by construction.
            """
            return self._centroid_to_goal_dist_now()

        # ----- Observation / reward / success (ADR-026 §Decision 1) -----

        def _get_obs_extra(self, info: dict[str, Any]) -> dict[str, Any]:
            """Task obs: goal pos, bar pose, bar tilt, stress proxy (ADR-026 §Decision 1).

            The matched impedance controller reads ``goal_pos`` to compute
            its bar-end target; ``bar_pose`` + telemetry are exposed for
            logging and for any later learned incumbent.
            """
            del info
            return {
                "goal_pos": self.goal_site.pose.p,
                "bar_pose": self.bar.pose.raw_pose,
                "bar_tilt_deg": self._bar_tilt_deg_now().reshape(-1, 1),
                "wrist_stress_proxy": self._wrist_stress_proxy_now().reshape(-1, 1),
            }

        def evaluate(self) -> dict[str, Any]:
            """Joint co-carry success predicate (ADR-026 §Decision 1; byte-frozen).

            Updates the per-episode running maxima (tilt + stress) then
            applies :func:`evaluate_cocarry_success`'s conjunction. The
            tilt + stress conjuncts make success a **joint** outcome — the
            construct fix the AS task lacked.
            """
            import torch as _torch

            tilt = self._bar_tilt_deg_now()
            stress = self._wrist_stress_proxy_now()
            # Settle window: ignore the placement-transient ring in the
            # episode maxima + success eligibility (R-2026-06-B settle
            # coefficient). elapsed_steps is a per-env tensor.
            past_settle = _torch.as_tensor(
                self.elapsed_steps >= COCARRY_SETTLE_WINDOW_STEPS, device=self.device
            ).reshape(-1)
            self._max_tilt_deg = _torch.where(
                past_settle, _torch.maximum(self._max_tilt_deg, tilt), self._max_tilt_deg
            )
            self._max_stress = _torch.where(
                past_settle, _torch.maximum(self._max_stress, stress), self._max_stress
            )

            dist = self._centroid_to_goal_dist_now()
            is_placed = dist <= COCARRY_GOAL_THRESH_M
            is_level = self._max_tilt_deg < COCARRY_TILT_MAX_DEG
            is_unstressed = self._max_stress < COCARRY_STRESS_MAX_PROXY_N
            is_settled = past_settle
            ego_static = self.agent.agents_dict[_EGO_UID].is_static(COCARRY_STATIC_QVEL_THRESH)  # type: ignore[attr-defined]
            if self._single_arm:
                both_static = ego_static
            else:
                partner_static = self.agent.agents_dict[self._partner_uid].is_static(  # type: ignore[attr-defined]
                    COCARRY_STATIC_QVEL_THRESH
                )
                both_static = ego_static & partner_static
            success = is_placed & is_level & is_unstressed & both_static & is_settled
            return {
                "success": success,
                "is_placed": is_placed,
                "is_level": is_level,
                "is_unstressed": is_unstressed,
                "both_static": both_static,
                "is_settled": is_settled,
                "centroid_to_goal": dist,
                "max_tilt_deg": self._max_tilt_deg,
                "max_stress_proxy": self._max_stress,
            }

        def compute_normalized_dense_reward(
            self,
            obs: Any,  # noqa: ANN401
            action: Any,  # noqa: ANN401
            info: dict[str, Any],
        ) -> Any:  # noqa: ANN401
            """Dense co-carry reward: transport + keep-level + settle (ADR-026 §Decision 1).

            Every shaping coefficient is a named module constant (the
            pre-registration-grade discipline that closes the AS gap). The
            reward is symmetric across the pair — it grades the *joint*
            bar state (centroid distance, tilt, both-arms-static), not one
            ego's competence.
            """
            del obs, action
            import torch as _torch

            dist = info["centroid_to_goal"]
            transport = COCARRY_REWARD_TRANSPORT_COEFF * (
                1.0 - _torch.tanh(COCARRY_REWARD_TANH_SCALE * dist)
            )
            tilt_rad = _torch.deg2rad(self._bar_tilt_deg_now())
            level = COCARRY_REWARD_LEVEL_COEFF * (
                1.0 - _torch.tanh(COCARRY_REWARD_LEVEL_TANH_SCALE * tilt_rad)
            )
            # Settle: both arms slow once the bar is placed. (Reward is a
            # training-time signal; the Rung-4 measurement evaluates the
            # FROZEN incumbent, so the partner-embodiment qvel slice here is
            # not on the eval path. The arm-DOF slice is per-uid.)
            qvel_norms: list[Any] = []
            holding = [_EGO_UID] if self._single_arm else [_EGO_UID, self._partner_uid]
            for uid in holding:
                arm_dof = _XARM6_ARM_DOF if uid == _XARM6_PARTNER_UID else _PANDA_ARM_DOF
                qvel = self.agent.agents_dict[uid].robot.get_qvel()[..., :arm_dof]  # type: ignore[attr-defined]
                qvel_norms.append(_torch.linalg.norm(qvel, axis=1))
            qvel_max = _torch.stack(qvel_norms, dim=0).amax(dim=0)
            settle = COCARRY_REWARD_SETTLE_COEFF * (
                1.0 - _torch.tanh(COCARRY_REWARD_TANH_SCALE * qvel_max)
            )
            # Rung-2 remediation (COCARRY_RUNG2_REMEDIATION_2026-06-16):
            # penalise EXCESS internal stress — the antagonistic fight the
            # success predicate gates on but the original reward gave no
            # gradient against. Gated past the placement-settle window
            # (consistent with evaluate()'s settle handling) so the startup
            # placement ring is not punished. ~0 on the matched cooperative
            # band (stress << f_max); bites the fight. Partner-agnostic: a
            # physical constraint force, never partner identity.
            stress_now = self._wrist_stress_proxy_now()
            past_settle = _torch.as_tensor(
                self.elapsed_steps >= COCARRY_SETTLE_WINDOW_STEPS, device=self.device
            ).reshape(-1)
            stress_penalty = cocarry_excess_stress_penalty(stress_now) * past_settle
            reward = transport + level + settle * info["is_placed"] - stress_penalty
            reward = _torch.where(
                info["success"],
                _torch.full_like(reward, COCARRY_REWARD_SUCCESS_BONUS),
                reward,
            )
            return reward / COCARRY_REWARD_NORMALIZER

        # ----- Public read-only API -----

        @property
        def condition_id(self) -> str:
            """The condition string this env was built for (ADR-026 §Decision 1)."""
            return self._config.condition_id

        @property
        def condition_config(self) -> CoCarryCondition:
            """The resolved :class:`CoCarryCondition` (read-only; ADR-026 §Decision 1)."""
            return self._config

        @property
        def single_arm(self) -> bool:
            """Whether this is the single-arm coupling positive-control (ADR-026 §Decision 2)."""
            return self._single_arm

        @property
        def ego_uid(self) -> str:
            """The ego uid (ADR-026 §Decision 1)."""
            return _EGO_UID

        @property
        def partner_uid(self) -> str:
            """The partner uid — Panda or xArm6 per the condition (ADR-026 §Decision 1; §D4)."""
            return self._partner_uid

        @property
        def goal_centroid(self) -> NDArray[np.float64]:
            """The per-episode goal centroid, shape ``(num_envs, 3)`` (ADR-026 §Decision 1)."""
            return self._goal_centroid

    try:
        return CoCarryEnv()
    except RuntimeError as exc:  # pragma: no cover - host-dependent SAPIEN failure
        raise ChamberEnvCompatibilityError(
            f"CoCarryEnv construction failed: {exc}; see ADR-001 §Risks."
        ) from exc


__all__ = [
    "COCARRY_BAR_LENGTH_M",
    "COCARRY_BAR_MASS_KG",
    "COCARRY_DRIVE_DAMPING_RATIO",
    "COCARRY_DRIVE_LINEAR_DAMPING",
    "COCARRY_DRIVE_LINEAR_STIFFNESS",
    "COCARRY_GOAL_THRESH_M",
    "COCARRY_REWARD_NORMALIZER",
    "COCARRY_REWARD_STRESS_COEFF",
    "COCARRY_REWARD_STRESS_SOFT_THRESHOLD_N",
    "COCARRY_REWARD_STRESS_TANH_SCALE_N",
    "COCARRY_REWARD_TRANSPORT_PBRS_COEFF",
    "COCARRY_STRESS_MAX_PROXY_N",
    "COCARRY_TILT_MAX_DEG",
    "CoCarryCondition",
    "cocarry_coupling",
    "cocarry_excess_stress_penalty",
    "evaluate_cocarry_success",
    "make_cocarry_env",
    "resolve_cocarry_condition",
    "tilt_deg_from_quaternion",
]
