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

#: Linear drive stiffness for the dual-hold attach, in N/m.
COCARRY_DRIVE_LINEAR_STIFFNESS: float = 2.0e4

#: Linear drive damping for the dual-hold attach, in N*s/m.
COCARRY_DRIVE_LINEAR_DAMPING: float = 2.0e3

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
#: penalised. Set to the success ceiling :data:`COCARRY_STRESS_MAX_PROXY_N`
#: (= f_max): the penalty bites exactly the stress that would VIOLATE the
#: success constraint. The matched cooperative band (committed Step-1
#: 12-seed: success-stress p99 = 104.5 N, max = 105 N) sits >= 25 N below
#: this, so the penalty is ~0 on the matched pair (the 100% reference is
#: unchanged) and bites only the fight (incumbent p90 = 854 N). Principled.
COCARRY_REWARD_STRESS_SOFT_THRESHOLD_N: float = COCARRY_STRESS_MAX_PROXY_N

#: ``tanh`` saturation scale (N) for the excess-stress penalty:
#: ``penalty = COEFF * tanh(relu(stress - threshold) / scale)``. 100 N is
#: the order of the f_max force scale, so the penalty saturates over a
#: force range comparable to the constraint itself (the ~724 N fight excess
#: saturates it). Bounded => safe under the reward normaliser.
COCARRY_REWARD_STRESS_TANH_SCALE_N: float = 100.0

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
    below the soft threshold (the success ceiling :data:`COCARRY_STRESS_MAX_PROXY_N`),
    rising monotonically and saturating at ``coeff`` for large excess. By
    construction the matched cooperative band (Step-1 success-stress p99 ≈
    104.5 N << threshold 130 N) incurs ≈0 penalty (so the 100% reference is
    unchanged); it bites only the antagonistic fight (incumbent p90 ≈ 854 N).
    Partner-agnostic — penalises a physical constraint force, never partner
    identity (spec §11 anti-leakage).

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

    class CoCarryEnv(BaseEnv):  # type: ignore[misc, valid-type]
        """Two-arm rigid-bar co-carry env (ADR-026 §Decision 1-2; R-2026-06-B Rungs 0-1).

        Subclasses :class:`mani_skill.envs.sapien_env.BaseEnv` directly
        (matching :class:`chamber.envs.stage1_pickplace.Stage1PickPlaceEnv`).
        See the module docstring for the coupling rationale, the
        dual-hold attach, and the wrist constraint-solver stress proxy.
        """

        SUPPORTED_ROBOTS: ClassVar[list[tuple[str, str]]] = [  # type: ignore[assignment]
            (_EGO_UID, _PARTNER_UID),
        ]

        def __init__(self) -> None:
            self._config: CoCarryCondition = config
            self._episode_length: int = int(episode_length)
            self._root_seed: int = int(root_seed)
            self._single_arm: bool = config.single_arm
            self._goal_centroid_base: tuple[float, float, float] = goal_xyz
            self._rng: np.random.Generator = derive_substream(
                _SUBSTREAM_NAME, root_seed=self._root_seed
            ).default_rng()
            # Per-episode goal centroid (num_envs, 3), running maxima, and
            # the cached hand-link index are populated below / at reset.
            # Assigned before super().__init__ because BaseEnv.__init__
            # calls reset() -> _initialize_episode during super-init.
            self._hand_link_index: int = -1
            self._drives: list[Any] = []
            try:
                super().__init__(
                    robot_uids=(_EGO_UID, _PARTNER_UID),  # type: ignore[arg-type]
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
                    xyz, quat = _AGENT_BASE_POSE[uid]
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

            # Cache the hand-link index from the ego articulation's ordered
            # link list (consistent ordering across the matched pair).
            ego_links = self.agent.agents_dict[_EGO_UID].robot.links  # type: ignore[attr-defined]
            self._hand_link_index = [link.name for link in ego_links].index(_HAND_LINK_NAME)

            # Dual-hold weld(s). Ego always holds; the partner holds only
            # in the matched condition (single-arm positive-control omits
            # the partner drive so a single arm attempts the task).
            # At the ready pose the two grippers reach across the table
            # centre: the ego TCP sits at world +x, the partner TCP at
            # world -x (measured: +-0.115 m). So the ego holds the bar's
            # +x end and the partner the -x end.
            grip_in_hand = sapien.Pose(p=[0.0, 0.0, COCARRY_GRIP_OFFSET_Z_M])
            self._drives = []
            self._add_weld(
                _EGO_UID, sapien.Pose(p=[COCARRY_BAR_LENGTH_M / 2.0, 0.0, 0.0]), grip_in_hand
            )
            if not self._single_arm:
                self._add_weld(
                    _PARTNER_UID,
                    sapien.Pose(p=[-COCARRY_BAR_LENGTH_M / 2.0, 0.0, 0.0]),
                    grip_in_hand,
                )

        def _add_weld(self, uid: str, bar_end_local: Any, grip_in_hand: Any) -> None:  # noqa: ANN401 - sapien.Pose
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
            the wrist constraint-solver stress proxy reads.
            """
            hand = self.agent.agents_dict[uid].robot.links_map[_HAND_LINK_NAME]  # type: ignore[attr-defined]
            drive = self.scene.create_drive(hand, grip_in_hand, self.bar, bar_end_local)
            for axis in ("x", "y", "z"):
                getattr(drive, f"set_drive_property_{axis}")(
                    COCARRY_DRIVE_LINEAR_STIFFNESS, COCARRY_DRIVE_LINEAR_DAMPING
                )
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
                # Ready pose for the ego; ready (matched) or retracted
                # (single-arm) for the partner.
                for uid in self.robot_uids:  # type: ignore[union-attr]
                    if uid == _PARTNER_UID and self._single_arm:
                        ready = _PANDA_RETRACTED_QPOS
                    else:
                        ready = _PANDA_READY_QPOS
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
                    partner_tcp = self.agent.agents_dict[_PARTNER_UID].tcp_pose.p  # type: ignore[attr-defined]
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

            The stress proxy is the linear-force norm of the ``panda_hand``
            incoming joint force, taken over the **holding** arms and
            reduced by max. For the matched pair both arms hold; for the
            single-arm positive-control only the ego holds. See the module
            docstring for why this articulation-solver force is the
            faithful proxy (SAPIEN exposes no free-drive force).
            """
            holding = [_EGO_UID] if self._single_arm else [_EGO_UID, _PARTNER_UID]
            per_arm: list[torch.Tensor] = []
            for uid in holding:
                robot = self.agent.agents_dict[uid].robot  # type: ignore[attr-defined]
                forces = robot.get_link_incoming_joint_forces()  # (b, n_links, 6)
                wrist = forces[:, self._hand_link_index, :3]  # linear force
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
                partner_static = self.agent.agents_dict[_PARTNER_UID].is_static(  # type: ignore[attr-defined]
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
            # Settle: both arms slow once the bar is placed.
            qvel_norms: list[Any] = []
            holding = [_EGO_UID] if self._single_arm else [_EGO_UID, _PARTNER_UID]
            for uid in holding:
                qvel = self.agent.agents_dict[uid].robot.get_qvel()[..., :_PANDA_ARM_DOF]  # type: ignore[attr-defined]
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
            """The partner uid (ADR-026 §Decision 1)."""
            return _PARTNER_UID

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
    "COCARRY_GOAL_THRESH_M",
    "COCARRY_REWARD_NORMALIZER",
    "COCARRY_REWARD_STRESS_COEFF",
    "COCARRY_REWARD_STRESS_SOFT_THRESHOLD_N",
    "COCARRY_REWARD_STRESS_TANH_SCALE_N",
    "COCARRY_REWARD_TRANSPORT_PBRS_COEFF",
    "COCARRY_STRESS_MAX_PROXY_N",
    "COCARRY_TILT_MAX_DEG",
    "CoCarryCondition",
    "cocarry_excess_stress_penalty",
    "evaluate_cocarry_success",
    "make_cocarry_env",
    "resolve_cocarry_condition",
    "tilt_deg_from_quaternion",
]
