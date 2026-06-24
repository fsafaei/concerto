# SPDX-License-Identifier: Apache-2.0
# pyright: reportPrivateImportUsage=false
#
# Same suppression rationale as :mod:`chamber.envs.cocarry` /
# :mod:`chamber.envs.stage1_pickplace`: ``torch.zeros`` / ``torch.from_numpy``
# are exported but not advertised in the torch stub's ``__all__``. Suppressed
# file-locally so the scene / spaces / telemetry logic stays free of per-line
# ``type: ignore`` noise.
r"""Co-insert env — contact-rich (hold-and-insert) heterogeneity setup (S0 skeleton).

Phase-2, **non-gating** research bet (invariant I1; ADR-026 §Decision 4):
the contact-rich successor to the co-carry ladder. Two manipulators perform a
two-robot hold-and-insert: the **ego = inserter** holds a peg rigidly in its
gripper and drives it into a blind socket; the **partner = holder** grips a
**FREE** receptacle (not fixed to the table) that carries the socket. Because
the receptacle is held only by the holder, a single inserter has nothing to
stabilise the socket against the insertion reaction force — so the task is
two-robot-necessary by construction and the ADR-026 positive control (single
inserter ⇒ success ≈ 0) is clean by construction.

The coupling-validity criterion (ADR-026 §Decision 1): *a heterogeneity axis
is informative about cooperation only if the manipulated heterogeneity is
coupled to the task outcome through the cooperation the task demands.* Here the
difficulty knob is **clearance** (hole - peg), a physically meaningful, monotone
parameter — deliberately **not** an inter-robot spring stiffness, which is the
structural escape from the co-carry over-coupling wall. Coupling-validity is
*contingent* on the holder's reaction being outcome-determining and is proven by
the Gate-0 base-failure criterion (S3) **before** any heterogeneity is measured.

Scope of THIS module (S0): stand up the **skeleton** only — the scene, the
spaces, reset, a stubbed :meth:`evaluate`, the frozen-able parameters as named
module constants, the pure-Tier-1 success predicate + condition resolver, and
the realism-compliance log-line emitter. There is **no reward and no contact
logic yet** — the peg-socket contact, the friction-inclusive force readout, the
workpiece-frame interaction-wrench instrument, and the structured base inserter
land in later, separately pre-registered slices (S1 contact-fidelity spike, S2
base inserter). The success-predicate force limits (``f_insert_max`` /
``f_couple_max``) are **derived from measured matched-pair distributions at S2**
and frozen before S3 — they are intentionally ``None`` placeholders here, never
asserted from intuition (the co-carry stress-derivation discipline).

Governance (ADR-026 §Decision 4; invariants I1/I6/I8/I9): this module adds
nothing to the Phase-1 gate, does not touch any Stage-1 env / condition /
pre-registration, does not edit any immutable archive under
``spikes/results/**``, triggers no ``chamber.comm.SCHEMA_VERSION`` bump (I9),
and reuses the existing wrapper-only / determinism / partner-interface
contracts. The realism shield is the proposed-ADR-018 §1 operationalisation
(the co-insert design) seeded by :func:`coinsert_realism_compliance_line`.

Tier-1 / Tier-2 split (mirrors :mod:`chamber.envs.cocarry`):
module-top-level imports are Tier-1-safe (``numpy``, ``concerto.training.
seeding``). ManiSkill / SAPIEN imports happen inside :func:`make_coinsert_env`
so ``python -c "import chamber.envs.coinsert"`` succeeds on a Vulkan-less host
(ADR-001 §Risks / P2). The pure-function Tier-1 surface
(:func:`resolve_coinsert_condition`, :func:`evaluate_coinsert_success`,
:func:`coinsert_realism_compliance_line`, the geometry / decision-rule
constants) is what the Tier-1 tests exercise; the SAPIEN-gated env construction,
the S1 contact instrument, and the S2 base inserter are Tier-2.

References:
- ADR-026 §Decision 1-4 (coupling-validity criterion; the falsifiable
  positive-control + pre-committed null rule; the Phase-2 non-gating scope;
  the ego-robustness admission gate stub committed at S7).
- ADR-007 §Discipline (pre-registration freeze + git-tag lock mirrored by
  ``spikes/preregistration/coinsert/coinsert_s0.yaml``).
- ADR-005 §Decision (ManiSkill v3 / SAPIEN 3; the hybrid/dual-sim posture).
- ADR-009 §Consequences (the frozen black-box partner contract every holder
  enters through; :data:`chamber.partners.interface._FORBIDDEN_ATTRS`).
- ADR-001 §Risks (wrapper-only; lazy SAPIEN import; ``ChamberEnvCompatibilityError``).
- proposed ADR-018 §1 (the realism-compliance checklist — seeded here, folded
  into the ADR when written).
- :mod:`chamber.envs.cocarry` (the factory / ``_load_agent`` /
  ``_initialize_episode`` / determinism / dual-hold-weld template).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple

import numpy as np

from chamber.envs.errors import ChamberEnvCompatibilityError
from chamber.partners.interface import _FORBIDDEN_ATTRS
from concerto.training.seeding import derive_substream

if TYPE_CHECKING:  # pragma: no cover
    import gymnasium as gym
    import torch
    from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Frozen-able task parameters (the nominal co-insert parameters). EVERY
# physical parameter is a named module constant NOW; they are frozen at the S3
# pre-registration before any shifted holder is seen (the co-carry "freeze
# coefficients first" discipline that closed the AS contamination gap). They
# are NOT yet pre-registration-locked — S0 only declares them; S3 tags them.
# Do not tune these against anything but the matched reference here.
# ---------------------------------------------------------------------------

#: Peg diameter, metres (16 mm). Mid-scale; within Franka/xArm payload and
#: gripper span (the co-insert design).
COINSERT_PEG_DIAMETER_M: float = 0.016

#: Graded clearance set (hole - peg diameter), metres: {1.0, 0.5, 0.2} mm. The
#: difficulty knob is **clearance** — a physically meaningful, monotone
#: parameter — NOT an inter-robot spring stiffness (the structural escape from
#: the co-carry over-coupling wall). The S1 contact-fidelity
#: spike validates that contact force rises monotonically with misalignment
#: across this set before any number is trusted.
COINSERT_CLEARANCE_SET_M: tuple[float, float, float] = (1.0e-3, 0.5e-3, 0.2e-3)

#: Insertion depth target, metres (40 mm). Deep enough that mid-insertion
#: jamming is possible (the co-insert design).
COINSERT_DEPTH_TARGET_M: float = 0.040

#: Socket lead-in chamfer, metres (0.5 mm x 45°). A realistic lead-in that
#: ensures graded (not binary) contact onset (the co-insert design). It also bounds the
#: structured base inserter's physical capture region (chamfer radius + the
#: declared spiral-search amplitude) — the pre-registered, non-gameable
#: "insertion envelope" boundary (the co-insert design).
COINSERT_CHAMFER_M: float = 0.0005

#: Receptacle (held workpiece) mass, kilograms (0.5 kg). Light enough that the
#: holder reaction matters; matches the co-carry bar scale (the co-insert design).
COINSERT_RECEPTACLE_MASS_KG: float = 0.5

#: Declared friction coefficient at the peg-socket pair. Jamming is
#: friction-mediated, so this is a **frozen, load-bearing** parameter (the
#: Stage-2 friction caveat makes it load-bearing). **Declared at S1** as a
#: single, frozen Coulomb coefficient (0.5 — a representative dry steel-on-steel
#: assembly value; the same number is applied to BOTH the SAPIEN physical
#: material and the MuJoCo oracle so the two sims share one declared contact
#: model). The trustworthy force readout is the friction-inclusive
#: ``get_link_incoming_joint_forces`` path, NOT the SAPIEN GPU contact-pair
#: impulse (which excludes friction); the S1 spike validates that the SAPIEN
#: signal is monotone in misalignment and agrees with the MuJoCo oracle.
COINSERT_PEG_SOCKET_FRICTION: float = 0.5

#: Socket outer half-width, metres — the receptacle is a square blind socket
#: (four walls + a floor around a central cavity), built from convex boxes (a
#: cavity is non-convex, so it is composed; ADR-001 §Risks wrapper-only — no
#: mesh asset). Held only by the holder (NOT table-fixed).
COINSERT_SOCKET_OUTER_HALF_M: float = 0.030

#: Socket wall height (depth of the blind cavity), metres — deeper than
#: :data:`COINSERT_DEPTH_TARGET_M` so the peg can seat with margin.
COINSERT_SOCKET_DEPTH_M: float = 0.050

#: Fixed world pose (xyz) of the kinematic socket in the S1 fidelity-probe rig.
#: The socket opening points world +z (identity orientation), so the sweep
#: drives the peg straight down in world axes; floats above the table (it is
#: kinematic — "the holder scripted to a fixed pose").
COINSERT_PROBE_SOCKET_XYZ: tuple[float, float, float] = (0.0, 0.0, 0.30)


def coinsert_socket_inner_half_width(
    clearance_m: float,
    *,
    peg_diameter_m: float = COINSERT_PEG_DIAMETER_M,
) -> float:
    """Square-socket inner half-width for a given diametral clearance (S1; ADR-026 §Decision 1).

    The socket is a square cavity around a cylindrical peg; the per-side gap is
    half the diametral ``clearance_m``, so the inner half-width is
    ``peg_radius + clearance_m / 2``. Lateral wall contact therefore onsets at a
    peg lateral offset of ``clearance_m / 2`` — the physically meaningful,
    monotone difficulty knob the S1 fidelity sweep exercises. Pure Tier-1
    function (no SAPIEN); the SAPIEN socket builder and the MuJoCo oracle both
    derive their geometry from it so the two sims are dimensionally identical.

    Args:
        clearance_m: Diametral clearance (hole - peg), metres.
        peg_diameter_m: Peg diameter (default :data:`COINSERT_PEG_DIAMETER_M`).

    Returns:
        The square-socket inner half-width, metres.
    """
    return float(peg_diameter_m) / 2.0 + float(clearance_m) / 2.0


#: Ego control rate, Hz (the co-insert design). The control mode is
#: :data:`COINSERT_CONTROL_MODE`.
COINSERT_CONTROL_HZ: int = 20

#: Ego control mode — matches the existing CHAMBER / co-carry control surface
#: (the co-insert design; the quasi-static-insertion precedent).
COINSERT_CONTROL_MODE: str = "pd_joint_delta_pos"

#: Episode horizon bounds, env ticks (the co-insert design: 250-320; quasi-static
#: insertion, pilot-tuned). :data:`COINSERT_DEFAULT_EPISODE_LENGTH` sits inside
#: this band; the exact horizon is frozen at S3 from the S2 pilot.
COINSERT_EPISODE_LENGTH_MIN: int = 250
COINSERT_EPISODE_LENGTH_MAX: int = 320

#: Default episode horizon, env ticks. Inside the quasi-static [250, 320] band.
COINSERT_DEFAULT_EPISODE_LENGTH: int = 320

# ---------------------------------------------------------------------------
# Success predicate thresholds (the co-insert design; the co-carry pattern extended to
# insertion: success = seated ∧ within_force ∧ static ∧ settled). The force
# limits are DERIVED FROM MEASURED MATCHED-PAIR DISTRIBUTIONS at S2 (the p99 of
# the cooperative-reference success distribution, scaled —), never
# from intuition, and frozen before any shifted-holder measurement. They are
# ``None`` placeholders here so S0 asserts no unmeasured number.
# ---------------------------------------------------------------------------

#: Seating depth tolerance ε_depth, metres: ``seated`` requires peg-tip depth
#: ≥ ``depth_target - ε_depth`` (the co-insert design). Geometric; frozen at S3.
COINSERT_DEPTH_EPS_M: float = 0.002

#: Seating axis-alignment tolerance, degrees: ``seated`` also requires the peg
#: axis aligned to the socket axis within this angle (the co-insert design). Geometric;
#: frozen at S3.
COINSERT_AXIS_ALIGN_TOL_DEG: float = 5.0

#: Peak peg-socket contact-force budget ``f_insert_max``, Newtons (the success
#: predicate ``within_force`` conjunct; no over-stress / no jam-through). **Derived at S2** from
#: the matched-pair success distribution (p99-scaled); ``None`` until then.
COINSERT_INSERT_FORCE_MAX_N: float | None = None

#: Peak workpiece interaction-wrench budget ``f_couple_max``, Newtons (the success
#: predicate ``within_force`` conjunct; the embodiment-invariant cooperation-cost limit).
#: **Derived at S2** from the matched-pair success distribution; ``None``
#: until then.
COINSERT_COUPLE_FORCE_MAX_N: float | None = None

#: Joint-velocity threshold (rad/s) for the per-arm ``is_static`` test in the
#: ``static`` conjunct. Mirrors the upstream / co-carry ``is_static(0.2)``
#: convention; success requires **both** robots static at termination (the co-insert design).
COINSERT_STATIC_QVEL_THRESH: float = 0.2

#: Settle window, env ticks, excluded from the episode force maxima and from
#: success eligibility — the placement/insertion-onset transient is an artifact
#: of instantaneous reset, not cooperative load (the co-carry settle
#: coefficient; the ``settled`` conjunct). Frozen at S3.
COINSERT_SETTLE_WINDOW_STEPS: int = 15

# ---------------------------------------------------------------------------
# Pre-committed decision-rule constants (the co-insert decision rule). Recorded
# here as named constants so the prereg YAML
# and the S3/S6 measurement harness read one source of truth. The seed-cluster
# count n is NOT a constant — it is sized by the S0 power simulation and frozen
# into the prereg, never inherited from co-carry.
# ---------------------------------------------------------------------------

#: Minimum coupling-valid drop Δ_min on insertion success (the co-insert Gate 2;
#:). The thesis lands iff pooled Δ ≥ this AND a one-sided 95% CI excludes 0
#: AND the positive control passes AND the drop is wrench-mediated AND it
#: survives the base-robustness control. A NULL is reported via the
#: equivalence / CI bound, NOT a power claim.
COINSERT_DELTA_MIN: float = 0.20

#: Capability-gate floor (the co-insert design): ``C_min = max(C_MIN_FLOOR, M -
#: C_MIN_MARGIN)``, RELATIVE to the cooperative-reference holder's achieved
#: hold-sub-task score ``M`` — not a bare 0.75. The reference must clear the
#: hold sub-task **comfortably** (well above the floor) for the gate to
#: discriminate (verified at S2).
COINSERT_C_MIN_FLOOR: float = 0.75
COINSERT_C_MIN_MARGIN: float = 0.25

#: Gate-0 base-failure ceiling (the co-insert Gate 0): the structured base
#: inserter must FAIL (≤ this) with the competent-but-non-accommodating,
#: non-destructive holder while succeeding (≥ :data:`COINSERT_REFERENCE_SUCCESS_MIN`)
#: with the cooperative reference. This gate is allowed to KILL the task.
COINSERT_GATE0_BASE_FAILURE_MAX: float = 0.5

#: Reference-success precondition (the co-insert Gate 0): the cooperative
#: reference insertion success must reach ~this for Δ_min to have dynamic range
#: (verified at S2). If the reference cannot reach it, the instrument lacks
#: headroom and the task is re-tuned before any measurement.
COINSERT_REFERENCE_SUCCESS_MIN: float = 0.9

#: Bootstrap resample count for the pooled paired cluster-bootstrap (the
#: co-insert decision rule). The cluster count n comes from the S0 power simulation.
COINSERT_N_BOOT: int = 10000

#: Bounded learned-residual cap ``r_max`` as a fraction of the action envelope
#: (the co-insert design; SPARR-style). Enforced at the command summer (not in the loss).
#: Used by the S5 residual; recorded here for the decomposition contract.
COINSERT_RESIDUAL_RMAX_FRAC: float = 0.05

# ---------------------------------------------------------------------------
# Robot / link wiring. The ego seat is the Panda inserter; the PARTNER (holder)
# seat is embodiment-configurable (Panda reference at S0; the Rung-B
# different-embodiment xArm6/UR holder is wired at S4). Only the Panda
# reference holder is built in this S0 skeleton.
# ---------------------------------------------------------------------------

#: Ego (inserter) uid — the Panda incumbent (AS-homo convention from
#: :mod:`chamber.envs.stage1_pickplace` / :mod:`chamber.envs.cocarry`).
_EGO_UID: str = "panda_wristcam"

#: Reference holder uid — the no-camera ``panda_v2.urdf`` partner body
#: (:class:`chamber.agents.panda_partner.PandaPartner`), mounted facing the ego.
_PARTNER_UID: str = "panda_partner"

#: Panda ``panda_hand`` link name (the grip frame the peg / receptacle welds
#: hang off, and whose incoming joint force is the S1 stress readout anchor).
_HAND_LINK_NAME: str = "panda_hand"

#: ``panda_hand_tcp`` offset down the hand z-axis, metres (measured exactly for
#: ``mani-skill==3.0.1`` panda; mirrors :data:`chamber.envs.cocarry.
#: COCARRY_GRIP_OFFSET_Z_M`). The peg / receptacle welds are specified in this
#: local frame so they hold across arm configurations.
_GRIP_OFFSET_Z_M: float = 0.1034

#: Half the base-to-base separation along world x, metres. The ego inserter
#: mounts at ``(-_BASE_X_M, 0, 0)`` facing +x; the holder at ``(+_BASE_X_M, 0,
#: 0)`` facing -x (yaw π about z), so peg and socket meet near the centre.
#: Pilot-refined at S2; declared here for the skeleton scene.
_BASE_X_M: float = 0.50

#: Substream name routed through :func:`derive_substream` for the env's
#: deterministic RNG (P6 / ADR-002 determinism rule).
_SUBSTREAM_NAME: str = "env.coinsert"

#: Canonical ready qpos for each Panda (7 arm joints + 2 fingers at 0.04).
#: Mirrors :data:`chamber.envs.cocarry._PANDA_READY_QPOS`; deterministic, no
#: init noise (P6 / ADR-002). Refined at S2 so peg and socket meet in a
#: well-conditioned configuration.
_PANDA_READY_QPOS: NDArray[np.float64] = np.array(
    [0.0, np.pi / 8, 0.0, -np.pi * 5 / 8, 0.0, np.pi * 3 / 4, -np.pi / 4, 0.04, 0.04]
)

#: Retracted holder qpos for the single-inserter positive control: the holder
#: is disabled (no receptacle weld) and folded back out of the workspace so it
#: cannot incidentally stabilise the free receptacle. Only used when
#: ``single_inserter=True``.
_PANDA_RETRACTED_QPOS: NDArray[np.float64] = np.array(
    [0.0, -np.pi / 4, 0.0, -np.pi * 7 / 8, 0.0, np.pi * 5 / 8, -np.pi / 4, 0.04, 0.04]
)

#: Per-uid initial base pose ``(xyz, quat_wxyz)`` — the holder is yawed π about
#: z to face the ego across the table (mirrored, deliberate poses; the co-insert design).
_BASE_POSE_BY_UID: dict[
    str, tuple[tuple[float, float, float], tuple[float, float, float, float]]
] = {
    _EGO_UID: ((-_BASE_X_M, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
    _PARTNER_UID: ((+_BASE_X_M, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
}


class CoInsertCondition(NamedTuple):
    """Resolved co-insert condition configuration (ADR-026 §Decision 1-2).

    Pure-Python NamedTuple so Tier-1 tests can construct / assert without any
    SAPIEN dependency. The single manipulated variable across the measurement
    is the **holder's co-design status** (the co-insert design); the holder identity
    enters through the :class:`chamber.partners.api.FrozenPartner` interface
    (ADR-009), not through the env condition. The env conditions select only
    the matched reference rig vs the two-robot-necessity positive control.

    Attributes:
        condition_id: Verbatim condition string (LOCKED once pre-registered —
            never rename in code; re-issue under a new tag, the ADR-007 §P4
            discipline).
        agent_uids: Ordered ``(ego_inserter_uid, holder_uid)`` tuple. The
            holder seat is embodiment-configurable (Panda reference at S0; the
            Rung-B body is wired at S4).
        single_inserter: ``True`` only for the two-robot-necessity positive
            control — the holder is disabled (no receptacle weld, retracted out
            of the workspace) and the receptacle is left unheld, so a single
            inserter attempts the task. Expected success ≈ 0 (positive
            control 1; ADR-026 §Decision 2).
    """

    condition_id: str
    agent_uids: tuple[str, str]
    single_inserter: bool


#: condition_id -> resolved configuration. Two conditions in the S0 skeleton:
#: the matched reference rig (the honest high reference / calibration anchor)
#: and the single-inserter positive control (expected ≈ 0). Both share the
#: identical Panda bodies and the byte-frozen success predicate; the positive
#: control differs only in whether the holder participates (ADR-026 §Decision 2
#: — the matched-vs-single contrast that makes the criterion falsifiable). The
#: non-accommodating Gate-0 holder (S3) and the own-objective / different-
#: embodiment holders (S4) are PARTNERS routed through FrozenPartner, not new
#: env conditions.
_CONDITION_TABLE: dict[str, CoInsertCondition] = {
    "coinsert_matched_reference": CoInsertCondition(
        condition_id="coinsert_matched_reference",
        agent_uids=(_EGO_UID, _PARTNER_UID),
        single_inserter=False,
    ),
    "coinsert_single_inserter_positive_control": CoInsertCondition(
        condition_id="coinsert_single_inserter_positive_control",
        agent_uids=(_EGO_UID, _PARTNER_UID),
        single_inserter=True,
    ),
}


def resolve_coinsert_condition(condition_id: str) -> CoInsertCondition:
    """Resolve a co-insert ``condition_id`` to its config (ADR-026 §Decision 1-2).

    Pure Tier-1 function — no SAPIEN dependency.

    Args:
        condition_id: ``"coinsert_matched_reference"`` (the matched reference
            rig) or ``"coinsert_single_inserter_positive_control"`` (the
            two-robot-necessity positive control).

    Returns:
        The resolved :class:`CoInsertCondition`.

    Raises:
        ValueError: If ``condition_id`` is unknown. The message lists the valid
            options and cites ADR-026 §Decision.
    """
    try:
        return _CONDITION_TABLE[condition_id]
    except KeyError as exc:
        msg = (
            f"CoInsertEnv: condition_id {condition_id!r} is not one of the co-insert "
            f"conditions {sorted(_CONDITION_TABLE)!r}. This S0 skeleton ships the matched "
            "reference rig + the single-inserter two-robot-necessity positive control only "
            "(ADR-026 §Decision 1-2; the co-insert design"
        )
        raise ValueError(msg) from exc


def evaluate_coinsert_success(
    *,
    seated_depth_m: NDArray[np.floating] | float,
    axis_align_deg: NDArray[np.floating] | float,
    peak_insert_force_n: NDArray[np.floating] | float,
    peak_couple_wrench_n: NDArray[np.floating] | float,
    both_static: NDArray[np.bool_] | bool,
    settled: NDArray[np.bool_] | bool,
    f_insert_max: float,
    f_couple_max: float,
    depth_target_m: float = COINSERT_DEPTH_TARGET_M,
    depth_eps_m: float = COINSERT_DEPTH_EPS_M,
    axis_align_tol_deg: float = COINSERT_AXIS_ALIGN_TOL_DEG,
) -> NDArray[np.bool_]:
    """Joint co-insert success predicate (the co-insert design; ADR-026 §Decision 1).

    Success iff **all** of (the co-carry ``seated ∧ within_force ∧ static ∧
    settled`` pattern extended to insertion):

    - **seated**: peg-tip depth ≥ ``depth_target_m - depth_eps_m`` AND the peg
      axis is aligned to the socket axis within ``axis_align_tol_deg``;
    - **within_force**: peak peg-socket contact force ≤ ``f_insert_max`` AND
      peak workpiece interaction wrench ≤ ``f_couple_max`` (no over-stress, no
      jam-through);
    - **static**: both robots' joint velocities below the static threshold at
      termination;
    - **settled**: the receptacle pose is stable over the final settle window.

    Like the co-carry predicate (and unlike the AS pick-place one, which read
    only the ego), this is a **joint** outcome — ``within_force`` binds only
    when the holder reacts to the insertion through the shared workpiece, so the
    score measures cooperation (the ADR-026 coupling criterion). Pure Tier-1
    function — the Tier-1 tests pin this logic on synthetic inputs.

    ``f_insert_max`` / ``f_couple_max`` are **required** (no default): they are
    derived from the measured matched-pair distribution at S2 and frozen before
    S3 (the co-insert design). The module-level :data:`COINSERT_INSERT_FORCE_MAX_N` /
    :data:`COINSERT_COUPLE_FORCE_MAX_N` are ``None`` until then, so the caller
    must supply the measured values explicitly — S0 asserts no unmeasured number.

    Args:
        seated_depth_m: Peg-tip insertion depth, metres.
        axis_align_deg: Peg-to-socket axis misalignment, degrees.
        peak_insert_force_n: Episode-peak peg-socket contact force, Newtons.
        peak_couple_wrench_n: Episode-peak workpiece interaction wrench, Newtons.
        both_static: Whether both robots are below the static qvel threshold.
        settled: Whether the receptacle pose is stable over the settle window.
        f_insert_max: Peak contact-force budget (S2-derived; required).
        f_couple_max: Peak interaction-wrench budget (S2-derived; required).
        depth_target_m: Seating depth target (default :data:`COINSERT_DEPTH_TARGET_M`).
        depth_eps_m: Seating depth tolerance (default :data:`COINSERT_DEPTH_EPS_M`).
        axis_align_tol_deg: Seating alignment tolerance (default
            :data:`COINSERT_AXIS_ALIGN_TOL_DEG`).

    Returns:
        Boolean array (broadcast of the inputs) — the per-env success.
    """
    seated = (np.asarray(seated_depth_m) >= (depth_target_m - depth_eps_m)) & (
        np.asarray(axis_align_deg) <= axis_align_tol_deg
    )
    within_force = (np.asarray(peak_insert_force_n) <= f_insert_max) & (
        np.asarray(peak_couple_wrench_n) <= f_couple_max
    )
    static = np.asarray(both_static, dtype=bool)
    is_settled = np.asarray(settled, dtype=bool)
    return np.asarray(seated & within_force & static & is_settled, dtype=bool)


def coinsert_capability_gate_floor(
    reference_hold_score: float,
    *,
    floor: float = COINSERT_C_MIN_FLOOR,
    margin: float = COINSERT_C_MIN_MARGIN,
) -> float:
    """Relative capability-gate floor ``C_min`` (the co-insert design; ADR-026 §Risks).

    ``C_min = max(floor, reference_hold_score - margin)`` — RELATIVE to the
    cooperative-reference holder's achieved hold-sub-task score ``M``, not a
    bare 0.75. Defeats the "just incompetent" confound: a candidate holder must
    clear ``C_min`` at the hold sub-task in isolation (paired with a cooperative
    reference *inserter*, never the measuring incumbent — non-circular) before
    entering the test, so any paired-test failure is a cooperation mismatch, not
    inability to hold. The gate only discriminates if the reference itself
    clears the hold sub-task **comfortably** above the floor (verified at S2).

    Excluding holders that fail the gate truncates heterogeneity from above and
    biases toward null: the directional-bias caveat (ADR-026 §Risks) is
    pre-registered, the excluded roster archived, and "Δ ≈ 0 ⇒ axis not
    coupling-valid" is forbidden when exclusions occurred — a null is reported
    "among capability-matched holders," never "heterogeneity is inert."

    Pure Tier-1 function.

    Args:
        reference_hold_score: ``M`` — the cooperative reference holder's
            achieved hold-sub-task score, in ``[0, 1]``.
        floor: Absolute floor (default :data:`COINSERT_C_MIN_FLOOR`).
        margin: Relative margin below ``M`` (default :data:`COINSERT_C_MIN_MARGIN`).

    Returns:
        The capability-gate floor ``C_min``.
    """
    return max(float(floor), float(reference_hold_score) - float(margin))


def coinsert_realism_compliance_line(
    *,
    partner_uid: str,
    partner_class: str,
    frozen_assert_passed: bool,
    forbidden_imports_absent: bool,
    condition_id: str,
) -> str:
    """Realism-compliance log line for the SpikeRun archive (proposed ADR-018 §1).

    Emits the one-line audit record the realism shield requires (proposed ADR-018 §1;
    proposed ADR-018 §1 — the realism-constraint operationalisation seeded by
    this bet): **which** partner ran, that the
    :meth:`chamber.benchmarks.ego_ppo_trainer.EgoPPOTrainer._assert_partner_is_frozen`
    black-box assert passed, and that no forbidden partner-policy-access symbol
    (:data:`chamber.partners.interface._FORBIDDEN_ATTRS`) was reached. The
    frozen contract is about **policy** access (weights, gradients, optimiser),
    not pose visibility — observing the receptacle the holder grips is permitted
    (ADR-009 §Consequences); reading the holder's weights is not.

    Pure Tier-1 function — the env/runner computes the booleans (it owns the
    live partner handle); this formats the immutable line (I8). CI-checked at
    the measurement slices.

    Args:
        partner_uid: The holder partner uid that ran.
        partner_class: The holder partner's concrete class name.
        frozen_assert_passed: Whether ``_assert_partner_is_frozen`` passed
            (no ``requires_grad=True`` partner parameter).
        forbidden_imports_absent: Whether the run reached no symbol in
            :data:`chamber.partners.interface._FORBIDDEN_ATTRS`.
        condition_id: The co-insert ``condition_id`` the run used.

    Returns:
        A single-line, parseable realism-compliance record.

    Raises:
        ValueError: If the compliance booleans are not both ``True`` — a
            realism violation must fail loudly, never be silently logged
            (proposed ADR-018 §1; the co-insert design).
    """
    if not (frozen_assert_passed and forbidden_imports_absent):
        msg = (
            "co-insert realism-compliance VIOLATION "
            f"(partner_uid={partner_uid!r}, frozen_assert_passed={frozen_assert_passed}, "
            f"forbidden_imports_absent={forbidden_imports_absent}): the black-box AHT "
            "contract was breached (proposed ADR-018 §1; ADR-009 §Consequences). The run "
            "is invalid — fix the breach, do not log past it."
        )
        raise ValueError(msg)
    shielded = ",".join(sorted(_FORBIDDEN_ATTRS))
    return (
        "realism_compliance "
        f"condition_id={condition_id} partner_uid={partner_uid} "
        f"partner_class={partner_class} frozen_assert_passed=true "
        f"forbidden_imports_absent=true shielded_attrs=[{shielded}] "
        "adr=ADR-009§Consequences,proposed-ADR-018§1"
    )


def make_coinsert_env(
    *,
    condition_id: str = "coinsert_matched_reference",
    episode_length: int = COINSERT_DEFAULT_EPISODE_LENGTH,
    root_seed: int = 0,
    num_envs: int = 1,
    render_mode: str | None = None,
    render_backend: str | None = None,
    peg_clearance_m: float = COINSERT_CLEARANCE_SET_M[0],
    peg_socket_friction: float = COINSERT_PEG_SOCKET_FRICTION,
    fidelity_probe: bool = False,
) -> gym.Env[Any, Any]:
    """Build a :class:`CoInsertEnv` instance (ADR-026 §Decision 1-4).

    Factory entry point. SAPIEN / ManiSkill imports are deferred to the body so
    ``python -c "import chamber.envs.coinsert"`` works on a Vulkan-less host
    (ADR-001 §Risks / P2). The :class:`CoInsertEnv` class is defined inside the
    factory body — the same pattern as
    :func:`chamber.envs.cocarry.make_cocarry_env`.

    **S1 scope:** the receptacle is now a real **blind square socket** (four
    walls + a floor around a cavity sized by ``peg_clearance_m``, composed from
    convex boxes — ADR-001 §Risks, no mesh asset) with a declared, frozen
    **friction** material on the peg-socket pair (jamming is friction-mediated).
    The trustworthy cooperation-cost readout is the friction-inclusive
    workpiece-frame interaction wrench via the holder articulation's
    ``get_link_incoming_joint_forces`` (NOT the SAPIEN GPU contact-pair impulse,
    which excludes friction). Reward / success contact logic + the structured
    base inserter remain S2 work; ``evaluate`` stays stubbed.

    Args:
        condition_id: ``"coinsert_matched_reference"`` (the matched reference
            rig) or ``"coinsert_single_inserter_positive_control"`` (the holder
            is disabled + retracted, receptacle unheld; expected success ≈ 0).
        episode_length: Truncation horizon, env ticks (the quasi-static band
            [250, 320]; default :data:`COINSERT_DEFAULT_EPISODE_LENGTH`).
        root_seed: Project root seed routed to the env's
            :func:`derive_substream` substream (P6 / ADR-002).
        num_envs: ManiSkill vectorisation count; default 1.
        render_mode: ManiSkill ``render_mode`` (``None`` disables).
        render_backend: ManiSkill ``render_backend``; ``"none"`` enables the
            headless URDF-material strip.
        peg_clearance_m: Diametral socket clearance (hole - peg), metres; one of
            :data:`COINSERT_CLEARANCE_SET_M` (default the loosest, 1.0 mm).
        peg_socket_friction: Coulomb friction on the peg-socket pair (default
            :data:`COINSERT_PEG_SOCKET_FRICTION`).
        fidelity_probe: When ``True``, build the **S1 contact-fidelity probe**
            rig: the ego is retracted and the peg is a **kinematic** body the
            caller poses directly (``set_peg_pose``) while the holder holds the
            socket at a fixed pose — a controlled lateral-misalignment sweep that
            isolates the peg-socket contact for the SAPIEN-vs-oracle check.
            ``False`` (default) keeps the peg welded to the ego inserter.

    Returns:
        A :class:`CoInsertEnv` ready to ``reset(seed=K)``.

    Raises:
        ValueError: If ``condition_id`` is unknown, or ``episode_length`` is
            outside the quasi-static [250, 320] band.
        ChamberEnvCompatibilityError: If ManiSkill / SAPIEN / Vulkan
            initialisation fails.
    """
    config = resolve_coinsert_condition(condition_id)
    if not (COINSERT_EPISODE_LENGTH_MIN <= int(episode_length) <= COINSERT_EPISODE_LENGTH_MAX):
        msg = (
            f"CoInsertEnv: episode_length {episode_length} is outside the "
            f"quasi-static band [{COINSERT_EPISODE_LENGTH_MIN}, {COINSERT_EPISODE_LENGTH_MAX}]."
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
            "CoInsertEnv requires mani_skill / sapien in the active venv. "
            "Install per pyproject.toml; see ADR-001 §Risks and ADR-005 §Decision."
        ) from exc

    if render_backend == "none":
        patch_sapien_urdf_no_visual_material()

    class CoInsertEnv(BaseEnv):  # type: ignore[misc, valid-type]
        """Two-robot hold-and-insert env — S0 skeleton (ADR-026 §Decision 1-4).

        Subclasses :class:`mani_skill.envs.sapien_env.BaseEnv` directly
        (matching :class:`chamber.envs.cocarry.CoCarryEnv`). Ego = inserter
        holding a peg; partner = holder gripping a FREE receptacle. See the
        module docstring for the coupling rationale and the S0/S1/S2 staging.
        The contact / force / reward logic is deliberately **stubbed** at S0.
        """

        SUPPORTED_ROBOTS: ClassVar[list[tuple[str, str]]] = [  # type: ignore[assignment]
            (_EGO_UID, config.agent_uids[1]),
        ]

        def __init__(self) -> None:
            self._config: CoInsertCondition = config
            self._episode_length: int = int(episode_length)
            self._root_seed: int = int(root_seed)
            self._single_inserter: bool = config.single_inserter
            self._partner_uid: str = config.agent_uids[1]
            self._clearance_m: float = float(peg_clearance_m)
            self._socket_inner_half: float = coinsert_socket_inner_half_width(self._clearance_m)
            self._friction: float = float(peg_socket_friction)
            self._fidelity_probe: bool = bool(fidelity_probe)
            self._rng: np.random.Generator = derive_substream(
                _SUBSTREAM_NAME, root_seed=self._root_seed
            ).default_rng()
            self._drives: list[Any] = []
            self._hand_link_index_by_uid: dict[str, int] = {}
            self._goal_xyz: NDArray[np.float64] = np.zeros((num_envs, 3), dtype=np.float64)
            try:
                super().__init__(
                    robot_uids=(_EGO_UID, self._partner_uid),  # type: ignore[arg-type]
                    num_envs=num_envs,
                    obs_mode="state_dict",
                    control_mode=COINSERT_CONTROL_MODE,
                    render_mode=render_mode,
                    render_backend=render_backend if render_backend is not None else "gpu",
                )
            except RuntimeError as exc:
                raise ChamberEnvCompatibilityError(
                    "SAPIEN/Vulkan initialisation failed during "
                    f"CoInsertEnv(condition_id={condition_id!r}) build: {exc}\n"
                    'Set render_backend="none" on CUDA-only hosts; see ADR-001 §Risks.'
                ) from exc

        # ----- ManiSkill v3 BaseEnv hooks -----

        @property
        def _default_human_render_camera_configs(self) -> Any:  # noqa: ANN401 - ManiSkill CameraConfig has no project type
            """Third-person render camera for rollout visualisation (ADR-026 §Decision 1)."""
            from mani_skill.sensors.camera import CameraConfig
            from mani_skill.utils import sapien_utils

            pose = sapien_utils.look_at(eye=[0.0, 0.9, 0.7], target=[0.0, 0.0, 0.2])
            return CameraConfig(
                "render_camera", pose=pose, width=640, height=480, fov=1.0, near=0.01, far=100.0
            )

        def _load_agent(  # type: ignore[override]
            self,
            options: dict[str, Any],
            initial_agent_poses: object = None,
            build_separate: bool = False,
        ) -> None:
            """Ego-inserter + holder ``_load_agent`` with mirrored base poses (ADR-026)."""
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
            """Build table + peg + blind-socket receptacle + goal, with frozen friction (S1).

            **S1 contact scene.** The receptacle is a real **blind square
            socket** — four walls + a floor around a central cavity of inner
            half-width :func:`coinsert_socket_inner_half_width` (sized by the
            diametral clearance), composed from convex boxes because a cavity is
            non-convex (ADR-001 §Risks: wrapper-only, no mesh asset). A declared,
            frozen Coulomb friction material is applied to the peg and the socket
            (jamming is friction-mediated). The peg is a cylinder; it is welded
            to the ego inserter in the normal rig, or built **kinematic** in the
            :data:`fidelity_probe` rig so the S1 sweep poses it directly while
            the holder holds the socket at a fixed pose.
            """
            del options
            self.table_scene = TableSceneBuilder(self)
            self.table_scene.build()

            # Frozen Coulomb friction material shared by the peg + socket pair —
            # the same coefficient handed to the MuJoCo oracle (one declared
            # contact model across both sims). Restitution 0 (quasi-static).
            contact_mat = sapien.physx.PhysxMaterial(self._friction, self._friction, 0.0)

            # Peg: a square-section box whose long axis is local +z (so it drops
            # straight into the +z socket opening). A box peg in a square socket
            # gives flat-face wall contact — numerically stable in both sims and
            # free of the corner-wedging a cylinder-in-square cavity suffers
            # (SAPIEN cylinder primitives also lie along local x, not z). The
            # probe rig uses a SHORT peg (half-length ~0.4x the cavity depth) so a
            # shallow lateral press engages the WALLS without bottoming on the
            # floor; the welded rig uses the full peg.
            peg_radius = COINSERT_PEG_DIAMETER_M / 2.0
            peg_half_len = (
                0.4 * COINSERT_SOCKET_DEPTH_M if self._fidelity_probe else COINSERT_DEPTH_TARGET_M
            )
            self._peg_half_len: float = peg_half_len
            peg_half = (peg_radius, peg_radius, peg_half_len)
            peg_builder = self.scene.create_actor_builder()
            # The probe peg gets a high density (real inertia) so a contact
            # impulse does not fling the near-massless body out of the socket
            # (numerical ejection); the welded rig uses the default density.
            peg_density = 8000.0 if self._fidelity_probe else 1000.0
            peg_builder.add_box_collision(
                half_size=peg_half,  # type: ignore[arg-type]
                material=contact_mat,
                density=peg_density,
            )
            peg_builder.add_box_visual(
                half_size=peg_half,  # type: ignore[arg-type]
                material=sapien.render.RenderMaterial(base_color=[0.30, 0.30, 0.35, 1.0]),
            )
            peg_builder.initial_pose = sapien.Pose(p=[-_BASE_X_M + 0.2, 0.0, 0.3])  # type: ignore[assignment]
            if self._fidelity_probe:
                # Kinematic peg: the sweep teleports it to a controlled lateral
                # penetration and reads the resulting peg-socket contact force.
                peg_builder.set_physx_body_type("kinematic")
            self.peg = peg_builder.build(name="coinsert_peg")

            # Receptacle: a blind square socket (4 walls + floor). In the normal
            # rig it is FREE (held only by the holder). In the fidelity-probe rig
            # it is KINEMATIC at a fixed pose ("holder scripted to a fixed pose"
            # — the socket is held immovable so the peg-socket contact is read
            # cleanly off the peg drive, free of holder-arm dynamics). The cavity
            # opens at the actor +z; the peg descends into it.
            # Normal rig: free dynamic socket (held by the holder). Probe rig:
            # a DYNAMIC socket rigidly welded to a fixed kinematic anchor — it
            # stays put ("holder scripted to a fixed pose") yet, being dynamic,
            # generates contacts against the kinematic peg (SAPIEN computes NO
            # contact between two kinematic bodies, so the socket cannot also be
            # kinematic).
            self.receptacle = self._build_socket_receptacle(contact_mat, kinematic=False)
            self._socket_anchor = None
            if self._fidelity_probe:
                anchor_builder = self.scene.create_actor_builder()
                anchor_builder.initial_pose = sapien.Pose(p=list(COINSERT_PROBE_SOCKET_XYZ))  # type: ignore[assignment]
                anchor_builder.set_physx_body_type("kinematic")
                self._socket_anchor = anchor_builder.build(name="coinsert_socket_anchor")

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

            # Cache the holder's wrist-link index for the friction-inclusive
            # interaction-wrench instrument (get_link_incoming_joint_forces).
            self._hand_link_index_by_uid = {}
            for uid in self.robot_uids:  # type: ignore[union-attr]
                links = self.agent.agents_dict[uid].robot.links  # type: ignore[attr-defined]
                self._hand_link_index_by_uid[uid] = [ln.name for ln in links].index(_HAND_LINK_NAME)

            # Welds. Probe rig welds nothing (kinematic peg + kinematic socket,
            # both posed directly by the sweep). Normal rig: peg → ego inserter;
            # socket → holder (unless the single-inserter positive control, which
            # leaves the socket unheld so a lone inserter cannot stabilise it).
            self._drives = []
            if self._fidelity_probe:
                # Rigid-weld the dynamic socket to the fixed kinematic anchor so
                # it holds its pose; the contact-pair impulse against the
                # kinematic peg is then the clean fidelity signal.
                assert self._socket_anchor is not None  # noqa: S101 - built above in probe mode
                drive = self.scene.create_drive(
                    self._socket_anchor, sapien.Pose(), self.receptacle, sapien.Pose()
                )
                for axis in ("x", "y", "z"):
                    getattr(drive, f"set_drive_property_{axis}")(2.0e5, 2.0e4)
                    getattr(drive, f"set_limit_{axis}")(0.0, 0.0)
                self._drives.append(drive)
            else:
                self._add_weld(_EGO_UID, self.peg)
                if not self._single_inserter:
                    self._add_weld(self._partner_uid, self.receptacle)

        def _build_socket_receptacle(
            self,
            contact_mat: Any,  # noqa: ANN401 - sapien material
            *,
            kinematic: bool = False,
        ) -> Any:  # noqa: ANN401 - sapien actor
            """Compose a blind square socket from convex boxes (S1; ADR-001 §Risks).

            Four walls + a floor enclose a central cavity of inner half-width
            :attr:`_socket_inner_half` (a cavity is non-convex, so it is built
            from convex primitives — no mesh asset). The cavity opens at +z and
            descends ``COINSERT_SOCKET_DEPTH_M``; the floor caps it (a *blind*
            socket). Friction is applied to every wall + floor collision.
            ``kinematic=True`` (the fidelity-probe rig) fixes the socket immovable
            so the contact reads cleanly off the peg drive.
            """
            w_in = self._socket_inner_half
            w_out = COINSERT_SOCKET_OUTER_HALF_M
            depth = COINSERT_SOCKET_DEPTH_M
            t_floor = 0.010
            wall = (w_out - w_in) / 2.0  # half-thickness of each wall slab
            # (half_size, center) for floor + 4 walls in the actor's local frame.
            boxes: list[tuple[list[float], list[float]]] = [
                ([w_out, w_out, t_floor / 2.0], [0.0, 0.0, -depth - t_floor / 2.0]),
                ([wall, w_out, depth / 2.0], [w_in + wall, 0.0, -depth / 2.0]),
                ([wall, w_out, depth / 2.0], [-(w_in + wall), 0.0, -depth / 2.0]),
                ([w_in, wall, depth / 2.0], [0.0, w_in + wall, -depth / 2.0]),
                ([w_in, wall, depth / 2.0], [0.0, -(w_in + wall), -depth / 2.0]),
            ]
            volume = sum(8.0 * h[0] * h[1] * h[2] for h, _ in boxes)
            density = COINSERT_RECEPTACLE_MASS_KG / volume
            builder = self.scene.create_actor_builder()
            colour = sapien.render.RenderMaterial(base_color=[0.55, 0.42, 0.20, 1.0])
            for half, centre in boxes:
                pose = sapien.Pose(p=centre)
                builder.add_box_collision(
                    pose=pose,
                    half_size=tuple(half),  # type: ignore[arg-type]
                    material=contact_mat,
                    density=density,
                )
                builder.add_box_visual(pose=pose, half_size=tuple(half), material=colour)  # type: ignore[arg-type]
            builder.initial_pose = sapien.Pose(p=[_BASE_X_M - 0.2, 0.0, 0.3])  # type: ignore[assignment]
            if kinematic:
                builder.set_physx_body_type("kinematic")
            return builder.build(name="coinsert_receptacle")

        def _add_weld(self, uid: str, body: Any) -> None:  # noqa: ANN401 - sapien actor
            """Rigidly weld a held body into a gripper grip frame (ADR-026; co-carry pattern).

            The held body (peg for the inserter, receptacle for the holder) is
            pinned to the ``panda_hand`` grip frame with a high-stiffness,
            hard-locked drive — an *attach*, not a free grasp, so failures
            reflect coordination, not fingertip friction (the deliberate
            co-carry simplification; real grasping is a later relaxation). The
            weld is specified in the hand link's **local** frame so it holds
            across arm configurations.
            """
            grip_in_hand = sapien.Pose(p=[0.0, 0.0, _GRIP_OFFSET_Z_M])
            hand = self.agent.agents_dict[uid].robot.links_map[_HAND_LINK_NAME]  # type: ignore[attr-defined]
            drive = self.scene.create_drive(hand, grip_in_hand, body, sapien.Pose())
            for axis in ("x", "y", "z"):
                getattr(drive, f"set_drive_property_{axis}")(2.0e4, 2.0e3)
                getattr(drive, f"set_limit_{axis}")(0.0, 0.0)
            self._drives.append(drive)

        def _initialize_episode(self, env_idx: torch.Tensor, options: dict[str, Any]) -> None:
            """Reset ready poses and sample the goal (P6 / ADR-002 determinism).

            All randomisation routes through the P6 RNG (never ``torch.rand``)
            so two ``reset(seed=K)`` calls reproduce byte-identical state. **S0
            skeleton:** sets the ready (or retracted, single-inserter) qpos and
            a placeholder goal; the peg/socket pre-contact staging is an S1/S2
            refinement.
            """
            del options
            with torch.device(self.device):  # type: ignore[attr-defined]
                b = len(env_idx)
                self.table_scene.initialize(env_idx)
                for uid in self.robot_uids:  # type: ignore[union-attr]
                    # Retract the holder for the single-inserter positive
                    # control, and retract the EGO in the fidelity-probe rig (the
                    # peg is kinematic there, posed by the sweep — the ego is not
                    # involved). Everyone else holds the ready pose.
                    retract_holder = uid == self._partner_uid and self._single_inserter
                    retract_ego = uid == _EGO_UID and self._fidelity_probe
                    ready = (
                        _PANDA_RETRACTED_QPOS
                        if (retract_holder or retract_ego)
                        else _PANDA_READY_QPOS
                    )
                    qpos = torch.from_numpy(np.tile(ready, (b, 1)).astype(np.float32)).to(
                        self.device
                    )
                    self.agent.agents_dict[uid].reset(qpos)  # type: ignore[attr-defined]

                if self._fidelity_probe:
                    # Probe rig: fix the KINEMATIC socket at a known world pose
                    # with the opening pointing world +z (identity orientation),
                    # so the sweep teleports the kinematic peg straight down in
                    # world axes — no holder-arm dynamics in the contact reading.
                    # Park the peg above the opening.
                    socket = torch.tensor(
                        COINSERT_PROBE_SOCKET_XYZ, dtype=torch.float32, device=self.device
                    ).reshape(1, 3)
                    self.receptacle.set_pose(Pose.create_from_pq(socket.expand(b, 3)))
                    above = socket + torch.tensor([0.0, 0.0, 0.10], device=self.device)
                    self.peg.set_pose(Pose.create_from_pq(above.expand(b, 3)))
                elif not self._single_inserter:
                    # Normal rig: warm-start the socket AT the holder grip frame so
                    # the weld starts at ~zero stress (the co-carry zero-initial-
                    # stress discipline — a socket initialised away from the grip
                    # frame would make the weld yank it in with a large preload).
                    hand = self.agent.agents_dict[self._partner_uid].robot.links_map[  # type: ignore[attr-defined]
                        _HAND_LINK_NAME
                    ]
                    grip = hand.pose * Pose.create_from_pq(
                        p=torch.tensor([0.0, 0.0, _GRIP_OFFSET_Z_M], device=self.device)
                    )
                    self.receptacle.set_pose(grip)

                # Placeholder goal (the seated socket pose); refined at S2 once
                # the success contact logic exists. Tiny P6 jitter for a
                # non-degenerate distribution.
                jitter = self._rng.uniform(-0.01, 0.01, size=(b, 3))
                goal = np.array([0.0, 0.0, 0.2], dtype=np.float64)[None, :] + jitter
                self._goal_xyz = goal
                self.goal_site.set_pose(
                    Pose.create_from_pq(torch.from_numpy(goal.astype(np.float32)).to(self.device))
                )

        def step(self, action: Any) -> tuple[Any, Any, Any, Any, dict[str, Any]]:  # type: ignore[override]  # noqa: ANN401
            """Enforce ``episode_length`` time-limit truncation (ADR-026 §Decision 1)."""
            obs, reward, terminated, truncated, info = super().step(action)
            truncated = truncated | (self.elapsed_steps >= self._episode_length)
            return obs, reward, terminated, truncated, info

        # ----- Observation / success / reward (S0: stubbed) -----

        def _get_obs_extra(self, info: dict[str, Any]) -> dict[str, Any]:
            """Task obs: goal pose + peg pose + receptacle pose (ADR-026 §Decision 1).

            S0 skeleton: exposes the physical poses the structured base inserter
            (S2) and any later learned residual will read. The contact-force /
            interaction-wrench channels are added with the S1 instrument.
            """
            del info
            return {
                "goal_pos": self.goal_site.pose.p,
                "peg_pose": self.peg.pose.raw_pose,
                "receptacle_pose": self.receptacle.pose.raw_pose,
            }

        def evaluate(self) -> dict[str, Any]:
            """Stubbed success predicate (S0 skeleton; the co-insert design; ADR-026 §Decision 1).

            The real predicate is ``seated ∧ within_force ∧ static ∧ settled``
            (:func:`evaluate_coinsert_success`), but its force conjuncts need
            the S1 contact instrument and the S2-derived ``f_insert_max`` /
            ``f_couple_max``. Until then this returns an all-``False`` success
            with a ``"stub": True`` marker so callers cannot mistake the S0
            skeleton for a measuring env. **No contact logic at S0.**
            """
            import torch as _torch

            n = self.peg.pose.p.shape[0]
            false = _torch.zeros(n, dtype=_torch.bool, device=self.device)
            return {"success": false, "stub": True}

        def compute_normalized_dense_reward(
            self,
            obs: Any,  # noqa: ANN401
            action: Any,  # noqa: ANN401
            info: dict[str, Any],
        ) -> Any:  # noqa: ANN401
            """Stubbed reward (S0 skeleton; ADR-026 §Decision 1).

            The dense reward (impedance + lead-in-search shaping) is an S2
            deliverable — **no reward logic at S0**. Returns zeros so the
            skeleton env steps without a reward signal.
            """
            del obs, action, info
            import torch as _torch

            return _torch.zeros(self.peg.pose.p.shape[0], device=self.device)

        # ----- S1 contact instrument + fidelity-probe rig -----

        def workpiece_interaction_wrench(self) -> Any:  # noqa: ANN401 - torch.Tensor
            """Friction-inclusive workpiece-frame interaction-wrench magnitude, N (S1; ADR-026).

            The embodiment-invariant cooperation-cost signal: the holder
            articulation's wrist (``panda_hand``) **incoming joint force** read
            via ``get_link_incoming_joint_forces`` — the full reduced-coordinate
            solver force, which IS friction-inclusive (unlike the SAPIEN GPU
            contact-pair impulse, which excludes friction; the Stage-2 friction
            caveat). This is the proven co-carry instrument generalised to the
            held workpiece: the force the peg transmits through the socket to the
            holder. Returns the linear-force norm per env (frame-invariant
            magnitude). The S1 fidelity sweep subtracts the no-contact baseline
            to isolate the contact-attributable interaction force, and validates
            it is monotone in misalignment and agrees with the MuJoCo oracle.
            """
            import torch as _torch

            robot = self.agent.agents_dict[self._partner_uid].robot  # type: ignore[attr-defined]
            forces = robot.get_link_incoming_joint_forces()  # (b, n_links, 6)
            idx = self._hand_link_index_by_uid[self._partner_uid]
            return _torch.linalg.norm(forces[:, idx, :3], axis=1)

        def peg_socket_contact_force(self) -> float:
            """Peg-socket contact-force magnitude, N (S1 fidelity signal; ADR-026 §Decision 1).

            Sums the SAPIEN contact-pair impulses between the peg and the socket
            and divides by the PhysX timestep to get a force. For the S1
            lateral-misalignment sweep the contact is a peg face pressing a socket
            **wall** — a NORMAL contact, so the friction-excluded contact-pair
            impulse (SAPIEN issue #281) captures the lateral force faithfully
            (friction here is tangential / vertical, not lateral). The friction
            caveat bites the axial jam force, which the experiment's
            cooperation-cost instrument (:meth:`workpiece_interaction_wrench`,
            the friction-inclusive joint force) covers; the S1 spike compares
            this contact force against the MuJoCo oracle's native contact sensor.
            Returns 0.0 when there is no peg-socket contact.
            """
            import torch as _torch

            impulses = self.scene.get_pairwise_contact_impulses(self.peg, self.receptacle)  # type: ignore[attr-defined]
            imp = _torch.as_tensor(impulses)
            return float(_torch.linalg.norm(imp).item()) / float(self.sim_timestep)

        def set_peg_pose(self, xyz: Any, quat_wxyz: Any) -> None:  # noqa: ANN401 - array-like
            """Teleport the kinematic probe peg (S1 fidelity-probe rig only; ADR-026 §Decision 1).

            Only valid when the env was built with ``fidelity_probe=True`` (the
            peg is a kinematic body). The S1 sweep teleports the peg to a
            controlled lateral penetration into the fixed socket wall and reads
            the resulting :meth:`peg_socket_contact_force`. A kinematic peg holds
            its commanded pose exactly, so the lateral offset (and hence the wall
            penetration) is set precisely — the cleanest controllable contact for
            the SAPIEN-vs-oracle fidelity comparison.
            """
            import torch as _torch
            from mani_skill.utils.structs.pose import Pose as _Pose

            if not self._fidelity_probe:
                msg = "set_peg_pose requires fidelity_probe=True (the peg is welded otherwise)."
                raise RuntimeError(msg)
            p = _torch.from_numpy(np.asarray(xyz, dtype=np.float32).reshape(-1, 3)).to(self.device)
            q = _torch.from_numpy(np.asarray(quat_wxyz, dtype=np.float32).reshape(-1, 4)).to(
                self.device
            )
            self.peg.set_pose(_Pose.create_from_pq(p, q))

        @property
        def socket_inner_half_width(self) -> float:
            """Square-socket inner half-width, metres (S1; ADR-026 §Decision 1)."""
            return self._socket_inner_half

        @property
        def peg_clearance_m(self) -> float:
            """The diametral socket clearance this env was built with (S1; ADR-026 §Decision 1)."""
            return self._clearance_m

        # ----- Public read-only API -----

        @property
        def condition_id(self) -> str:
            """The condition string this env was built for (ADR-026 §Decision 1)."""
            return self._config.condition_id

        @property
        def condition_config(self) -> CoInsertCondition:
            """The resolved :class:`CoInsertCondition` (read-only; ADR-026 §Decision 1)."""
            return self._config

        @property
        def single_inserter(self) -> bool:
            """Whether this is the single-inserter positive control (ADR-026 §Decision 2)."""
            return self._single_inserter

        @property
        def ego_uid(self) -> str:
            """The ego (inserter) uid (ADR-026 §Decision 1)."""
            return _EGO_UID

        @property
        def partner_uid(self) -> str:
            """The holder partner uid (ADR-026 §Decision 1)."""
            return self._partner_uid

    try:
        return CoInsertEnv()
    except RuntimeError as exc:  # pragma: no cover - host-dependent SAPIEN failure
        raise ChamberEnvCompatibilityError(
            f"CoInsertEnv construction failed: {exc}; see ADR-001 §Risks."
        ) from exc


__all__ = [
    "COINSERT_CLEARANCE_SET_M",
    "COINSERT_C_MIN_FLOOR",
    "COINSERT_C_MIN_MARGIN",
    "COINSERT_DELTA_MIN",
    "COINSERT_DEPTH_TARGET_M",
    "COINSERT_GATE0_BASE_FAILURE_MAX",
    "COINSERT_N_BOOT",
    "COINSERT_PEG_DIAMETER_M",
    "COINSERT_PEG_SOCKET_FRICTION",
    "COINSERT_RECEPTACLE_MASS_KG",
    "COINSERT_REFERENCE_SUCCESS_MIN",
    "COINSERT_SOCKET_DEPTH_M",
    "COINSERT_SOCKET_OUTER_HALF_M",
    "CoInsertCondition",
    "coinsert_capability_gate_floor",
    "coinsert_realism_compliance_line",
    "coinsert_socket_inner_half_width",
    "evaluate_coinsert_success",
    "make_coinsert_env",
    "resolve_coinsert_condition",
]
