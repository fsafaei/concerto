# SPDX-License-Identifier: Apache-2.0
"""Minimal *kinematic* handover-and-place env for the Gate-0 coupling-validity spike.

Phase-2, **non-gating** research spike on the open ADR-026 coupling-validity line
(invariant I1), of the same class as the co-carry and co-insert spikes. It is NOT
a Phase-1 gate, NOT a trained moat residual, and NOT a commercial commitment.

What the task models (ADR-026 §Decision). A black-box *presenter* partner hands a
part into a shared workspace; a legacy six-axis *ego* arm receives it and must
place/seat it within a downstream tolerance, at cycle time, **without a full
re-grasp** it cannot afford. The scientific question is whether a mismatched partner
measurably degrades the place outcome a matched partner achieves, for a real reason,
and not because the ego was hobbled.

The binding channel (executor prompt Rev 2, B1). Once the ego has the part the
partner lets go: it binds only through the *initial condition* it hands over, and
that condition has two sub-channels which behave differently:

* **Lateral position offset** — a competent six-axis ego *observes* this and corrects
  it by **translating** its place motion. Cheap, no re-grasp. Lateral therefore does
  NOT drive budget binding; it is the downstream **success tolerance** (does the
  placed part land in the next station's window), not the mismatch.
* **Grasp-pose / orientation error** — how the part is oriented *in the ego's gripper*.
  To seat it the ego must bring it to the seating orientation; if the required
  reorientation exceeds the wrist-correction range it cannot fix it while holding the
  part and must **set it down and re-grasp** (costs the re-grasp duration). If the
  presented grasp is beyond the re-acquire range, even a re-grasp leaves a residual —
  **kinematically intrinsic** binding. This is the channel that drives the budget and
  the intrinsic boundary.

So the **mismatch is a grasp-pose/orientation error**, and the **intrinsic boundary
is the wrist-correction / re-acquire range** (angular), NOT a lateral envelope and NOT
contact-impedance (the partner has let go — impedance is the co-hold mechanism, which
here would exercise the wrong channel and yield an uninformative null). Which channel
binds, and where its two boundaries lie, is *derived* by the Stage-0 pre-check
(``chamber.spikes.handover_place_gate0``) from the ego six-axis kinematics before the
prereg is frozen, not asserted here.

Because the binding is a *tolerance* phenomenon it is faithfully captured by a
pure-Python kinematic resolver with no SAPIEN contact solver. This module has **no
Tier-2 surface**: it imports no ManiSkill / SAPIEN and runs anywhere.

Determinism (P6 / ADR-002). Every reset draw routes through
:func:`concerto.training.seeding.derive_substream`; the module never calls
``np.random`` or ``torch.rand`` ad hoc.

Anti-artificiality. The binding physical numbers — the place tolerance windows, the
takt, the *derived* place-cycle / re-grasp-duration band, and the ego arm's kinematic
ranges — are NOT baked in as tuned magic constants. The ``HANDOVER_DEFAULT_*`` values
are explicit, NON-BINDING placeholders that only make the env runnable for unit tests;
the **binding** values are the externally-sourced / Stage-0-derived numbers committed
in the tagged Gate-0 pre-registration and injected by the spike runner via
:func:`make_handover_place_env`.

References:
- ADR-026 §Decision (coupling-validity criterion; non-gating Phase-2 line).
- ADR-007 §Discipline (the prereg is locked by a git-tag blob-SHA before measurement).
- ADR-009 §Decision (black-box frozen partner; the ego reads the presented part pose +
  grasp-pose but no presenter-policy information).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from concerto.training.seeding import derive_substream

if TYPE_CHECKING:
    from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Episode structure: phase 0 the presenter hands over (sets the initial
# condition), phase 1 the ego places. A 2-step horizon is the faithful minimum.
# ---------------------------------------------------------------------------
#: Number of env steps per episode: phase 0 (presentation) + phase 1 (placement).
HANDOVER_HORIZON: int = 2

#: Phase index after :meth:`HandoverPlaceEnv.reset`, awaiting the presentation action.
HANDOVER_PHASE_PRESENT: int = 0
#: Phase index after the presentation step, awaiting the ego placement action.
HANDOVER_PHASE_PLACE: int = 1

#: Substream label for env reset draws (P6 / ADR-002).
_SUBSTREAM_NAME: str = "env.handover_place"

# ---------------------------------------------------------------------------
# Decision-rule constants — the SINGLE SOURCE OF TRUTH for the Gate-0 rule.
# The power simulation and the spike runner import these so they cannot drift.
# Method constants (the test design), not the externally-sourced physical numbers.
# ---------------------------------------------------------------------------
#: Minimum coupling-validity degradation, in percentage points (Limb 2 bar).
HANDOVER_DELTA_MIN_PP: float = 20.0
#: Limb-1 solvability bar: matched success CI lower bound must clear this.
HANDOVER_TAU_SOLV: float = 0.90
#: Bootstrap resamples for the cluster / paired-cluster CIs.
HANDOVER_N_BOOT: int = 10000
#: Episodes per (seed-cluster, condition). Chamber convention is 20.
HANDOVER_K_EPISODES: int = 20

# ---------------------------------------------------------------------------
# NON-BINDING placeholder physical parameters (anti-artificiality guard). These
# exist ONLY so the env is constructible and unit-testable; the runner overrides
# every one with the externally-sourced / Stage-0-derived prereg values.
# ---------------------------------------------------------------------------
#: Lateral place-tolerance SUCCESS window (m). Placeholder only.
HANDOVER_DEFAULT_LATERAL_WINDOW_M: float = 1.0e-3
#: Angular place-tolerance SUCCESS window (deg). Load-bearing in Rev 2. Placeholder.
HANDOVER_DEFAULT_ANGULAR_WINDOW_DEG: float = 3.0
#: Seating-force proxy limit (N). Placeholder only.
HANDOVER_DEFAULT_SEATING_FORCE_LIMIT_N: float = 75.0

#: Ego lateral translation range (m) — the ego corrects a lateral offset by moving.
#: Generous so the lateral channel does NOT drive budget binding (Rev 2 B1).
HANDOVER_DEFAULT_TRANSLATION_RANGE_M: float = 0.10
#: Wrist-correction range (deg): grasp-pose error the ego nulls IN-GRASP (no re-grasp).
#: The (a)/(b) boundary; derived in Stage 0. Placeholder only.
HANDOVER_DEFAULT_WRIST_CORRECTION_DEG: float = 15.0
#: Re-acquire range (deg): grasp-pose error a *re-grasp* can null. Beyond it a residual
#: remains even under a FREE re-grasp (the intrinsic boundary; (b)/(c)). Derived in
#: Stage 0. Placeholder only.
HANDOVER_DEFAULT_REACQUIRE_RANGE_DEG: float = 90.0

#: Re-grasp budget (s) — the time the ego has. Placeholder; the binding value is
#: DERIVED in the prereg as ``takt - nominal_place_cycle``, never asserted.
HANDOVER_DEFAULT_REGRASP_BUDGET_S: float = 1.0
#: Full re-grasp duration (s): set-down + open + reorient + close + re-acquire +
#: return. Placeholder; the binding value is a six-axis-arm + gripper datasheet band.
HANDOVER_DEFAULT_REGRASP_DURATION_S: float = 2.0

#: Kinematic seating-force proxy stiffnesses (backstop inside the windows; pose is
#: primary). Tied to the sourced numbers, not free knobs (see force_proxy docs).
HANDOVER_DEFAULT_CONTACT_STIFFNESS_N_PER_M: float = 3.75e4
HANDOVER_DEFAULT_ANGULAR_STIFFNESS_N_PER_DEG: float = 12.5

#: Nominal place-target pose [x, y, seat_angle_deg]; small per-episode jitter is added.
HANDOVER_DEFAULT_TARGET_POSE: tuple[float, float, float] = (0.0, 0.0, 0.0)
#: Half-width of the uniform per-episode lateral target jitter (m), routed through P6.
HANDOVER_GOAL_JITTER_HALF_M: float = 5.0e-3

# Action layout.
#: Presenter action (phase 0): [lat_offset_x, lat_offset_y, grasp_pose_error_deg, skew_s].
HANDOVER_PRESENTATION_DIM: int = 4
#: Ego action (phase 1): [translate_x, translate_y, reorient_deg, regrasp_flag].
HANDOVER_EGO_DIM: int = 4

#: Threshold above which the ego's continuous re-grasp flag counts as a request.
_REGRASP_FLAG_THRESHOLD: float = 0.5

# ---------------------------------------------------------------------------
# Condition table — LOCKED names (ADR-007 §P4). Must match the Gate-0 prereg
# verbatim; never rename in code (re-issue under a new tag).
# ---------------------------------------------------------------------------


class HandoverCondition(NamedTuple):
    """One pre-registered handover-place condition (ADR-026; ADR-007 §P4).

    The ``condition_id`` is the locked name shared by the prereg YAML, the runner,
    and the emitted :class:`chamber.evaluation.results.SpikeRun`; ``presenter``
    selects the matched or mismatched presenter variant.
    """

    condition_id: str
    presenter: str
    description: str


#: Matched reference: a presenter whose grasp-pose is in the ego's correction
#: envelope. Limb 1 (solvability) runs against this condition.
HANDOVER_MATCHED_REFERENCE: str = "handover_matched_reference"
#: Grasp-pose mismatch: a presenter biased in the *grasp-pose/orientation channel*
#: only. Limb 2 (coupling validity) runs here.
HANDOVER_PRESENTATION_MISMATCH: str = "handover_presentation_mismatch"

_CONDITION_TABLE: dict[str, HandoverCondition] = {
    HANDOVER_MATCHED_REFERENCE: HandoverCondition(
        condition_id=HANDOVER_MATCHED_REFERENCE,
        presenter="matched",
        description="matched presenter + scripted ego (Limb 1 solvability)",
    ),
    HANDOVER_PRESENTATION_MISMATCH: HandoverCondition(
        condition_id=HANDOVER_PRESENTATION_MISMATCH,
        presenter="mismatched",
        description="grasp-pose-mismatched presenter + scripted ego (Limb 2)",
    ),
}


def handover_conditions() -> dict[str, HandoverCondition]:
    """Return a copy of the locked Gate-0 condition table (ADR-026; ADR-007 §P4).

    The names are the single source of truth shared with the prereg YAML; callers
    must not mutate the returned mapping.
    """
    return dict(_CONDITION_TABLE)


# ---------------------------------------------------------------------------
# Kinematic placement resolver — the testable scientific core.
# ---------------------------------------------------------------------------


class PlacementOutcome(NamedTuple):
    """Resolved outcome of one place attempt (ADR-026 §Decision; Gate-0 diagnostic).

    Carries the residual decomposition, the budget/envelope provenance, and the
    binding conjunct so the verdict report can (a) split coupling into
    budget-mediated vs kinematically intrinsic, and (b) verify that lateral is
    success-side while grasp-pose is coupling-side (executor prompt Rev 2, T3).
    ``failure_mode``: ``"none"`` (success), ``"budget_mediated"`` (would succeed
    under a free re-grasp), or ``"intrinsic"`` (fails even under a free re-grasp).
    ``binding_conjunct``: which window(s) the failure broke ("lateral"/"angular"/
    "force", "|"-joined; "none" on success).
    """

    success: bool
    residual_lateral_m: float
    residual_angular_deg: float
    seating_force_proxy_n: float
    regrasp_requested: bool
    regrasp_executed: bool
    regrasp_budget_blocked: bool
    beyond_reacquire: bool
    failure_mode: str
    binding_conjunct: str


def _angular_residual_for(
    *,
    regrasp_executed: bool,
    grasp_pose_error_deg: float,
    wrist_correction_deg: float,
    reacquire_range_deg: float,
) -> float:
    """Grasp-pose orientation residual for a re-grasp decision (ADR-026).

    In-grasp the ego nulls error up to ``wrist_correction_deg``; if that leaves a
    remainder a re-grasp (when executed) nulls error up to the larger
    ``reacquire_range_deg``; the beyond-re-acquire remainder is intrinsic.
    """
    err = abs(grasp_pose_error_deg)
    if regrasp_executed:
        return max(0.0, err - reacquire_range_deg)
    return max(0.0, err - wrist_correction_deg)


def evaluate_handover_place_success(
    residual_lateral_m: float,
    residual_angular_deg: float,
    seating_force_proxy_n: float,
    *,
    lateral_window_m: float,
    angular_window_deg: float,
    seating_force_limit_n: float,
) -> bool:
    """Downstream place/seat gate predicate (ADR-026 §Decision).

    A placed part passes iff its residual misalignment is inside BOTH the lateral
    and angular success windows AND the kinematic seating-force proxy is under the
    force limit — a real downstream tolerance, never a reward-threshold rubber-stamp.
    """
    return (
        residual_lateral_m < lateral_window_m
        and residual_angular_deg < angular_window_deg
        and seating_force_proxy_n < seating_force_limit_n
    )


def _binding_conjunct(
    *,
    residual_lateral_m: float,
    residual_angular_deg: float,
    seating_force_proxy_n: float,
    lateral_window_m: float,
    angular_window_deg: float,
    seating_force_limit_n: float,
) -> str:
    broken: list[str] = []
    if residual_lateral_m >= lateral_window_m:
        broken.append("lateral")
    if residual_angular_deg >= angular_window_deg:
        broken.append("angular")
    if seating_force_proxy_n >= seating_force_limit_n:
        broken.append("force")
    return "|".join(broken) if broken else "none"


def resolve_placement(
    *,
    lateral_offset_m: tuple[float, float],
    grasp_pose_error_deg: float,
    timing_skew_s: float,
    ego_translation_m: tuple[float, float],
    ego_reorient_deg: float,
    ego_regrasp_requested: bool,
    regrasp_budget_s: float,
    regrasp_duration_s: float,
    translation_range_m: float,
    wrist_correction_deg: float,
    reacquire_range_deg: float,
    contact_stiffness_n_per_m: float,
    angular_stiffness_n_per_deg: float,
    lateral_window_m: float,
    angular_window_deg: float,
    seating_force_limit_n: float,
) -> PlacementOutcome:
    """Resolve a kinematic place attempt into a :class:`PlacementOutcome` (ADR-026).

    The handover binds through the initial condition only. The lateral offset is
    corrected by ego *translation* (clipped to ``translation_range_m``; a competent
    six-axis ego, so it rarely binds — Rev 2 B1). The grasp-pose orientation error is
    the coupling channel: corrected in-grasp up to ``wrist_correction_deg``, else by a
    re-grasp that executes only if it fits the remaining budget (the budget-mediated
    channel) and nulls error only up to ``reacquire_range_deg`` (beyond it is the
    intrinsic channel). A late arrival (``timing_skew_s``) eats the budget.
    ``failure_mode`` compares the actual outcome with the free-re-grasp counterfactual.
    """
    lateral_offset = np.asarray(lateral_offset_m, dtype=np.float64)
    translation = np.asarray(ego_translation_m, dtype=np.float64)

    # Lateral: the ego cannot translate more than its range allows.
    trans_mag = float(np.linalg.norm(translation))
    if trans_mag > translation_range_m and trans_mag > 0.0:
        translation = translation * (translation_range_m / trans_mag)
    residual_lateral_m = float(np.linalg.norm(lateral_offset + translation))

    # Budget accounting: a late arrival eats into the re-grasp budget.
    effective_budget_s = regrasp_budget_s - max(0.0, timing_skew_s)
    regrasp_executed = ego_regrasp_requested and regrasp_duration_s <= effective_budget_s
    regrasp_budget_blocked = ego_regrasp_requested and not regrasp_executed
    beyond_reacquire = abs(grasp_pose_error_deg) > reacquire_range_deg
    del ego_reorient_deg  # the env clips correction analytically; the ego's value is advisory

    residual_angular_deg = _angular_residual_for(
        regrasp_executed=regrasp_executed,
        grasp_pose_error_deg=grasp_pose_error_deg,
        wrist_correction_deg=wrist_correction_deg,
        reacquire_range_deg=reacquire_range_deg,
    )
    force = (
        contact_stiffness_n_per_m * residual_lateral_m
        + angular_stiffness_n_per_deg * residual_angular_deg
    )
    success = evaluate_handover_place_success(
        residual_lateral_m,
        residual_angular_deg,
        force,
        lateral_window_m=lateral_window_m,
        angular_window_deg=angular_window_deg,
        seating_force_limit_n=seating_force_limit_n,
    )

    if success:
        failure_mode = "none"
        binding_conjunct = "none"
    else:
        free_angular = _angular_residual_for(
            regrasp_executed=True,
            grasp_pose_error_deg=grasp_pose_error_deg,
            wrist_correction_deg=wrist_correction_deg,
            reacquire_range_deg=reacquire_range_deg,
        )
        free_force = (
            contact_stiffness_n_per_m * residual_lateral_m
            + angular_stiffness_n_per_deg * free_angular
        )
        free_success = evaluate_handover_place_success(
            residual_lateral_m,
            free_angular,
            free_force,
            lateral_window_m=lateral_window_m,
            angular_window_deg=angular_window_deg,
            seating_force_limit_n=seating_force_limit_n,
        )
        failure_mode = "budget_mediated" if free_success else "intrinsic"
        binding_conjunct = _binding_conjunct(
            residual_lateral_m=residual_lateral_m,
            residual_angular_deg=residual_angular_deg,
            seating_force_proxy_n=force,
            lateral_window_m=lateral_window_m,
            angular_window_deg=angular_window_deg,
            seating_force_limit_n=seating_force_limit_n,
        )

    return PlacementOutcome(
        success=success,
        residual_lateral_m=residual_lateral_m,
        residual_angular_deg=residual_angular_deg,
        seating_force_proxy_n=force,
        regrasp_requested=ego_regrasp_requested,
        regrasp_executed=regrasp_executed,
        regrasp_budget_blocked=regrasp_budget_blocked,
        beyond_reacquire=beyond_reacquire,
        failure_mode=failure_mode,
        binding_conjunct=binding_conjunct,
    )


# ---------------------------------------------------------------------------
# The env.
# ---------------------------------------------------------------------------


class HandoverPlaceEnv:
    """Minimal kinematic handover-and-place env (ADR-026 §Decision; non-gating I1).

    A pure-Python, SAPIEN-free env with a 2-step horizon: phase 0 consumes the
    presenter's presentation action (lateral offset + grasp-pose error + timing), and
    phase 1 consumes the ego's placement action and resolves the downstream place/seat
    gate via :func:`resolve_placement`. The ego observes the presented lateral offset
    and grasp-pose error (allowed under ADR-009 §Decision); it never reads
    presenter-policy information. Determinism (P6 / ADR-002) routes every reset draw
    through :func:`concerto.training.seeding.derive_substream`. All binding physical
    numbers are constructor parameters injected by :func:`make_handover_place_env`;
    the defaults are NON-BINDING placeholders for unit tests only.
    """

    def __init__(
        self,
        *,
        condition_id: str = HANDOVER_MATCHED_REFERENCE,
        root_seed: int = 0,
        lateral_window_m: float = HANDOVER_DEFAULT_LATERAL_WINDOW_M,
        angular_window_deg: float = HANDOVER_DEFAULT_ANGULAR_WINDOW_DEG,
        seating_force_limit_n: float = HANDOVER_DEFAULT_SEATING_FORCE_LIMIT_N,
        regrasp_budget_s: float = HANDOVER_DEFAULT_REGRASP_BUDGET_S,
        regrasp_duration_s: float = HANDOVER_DEFAULT_REGRASP_DURATION_S,
        translation_range_m: float = HANDOVER_DEFAULT_TRANSLATION_RANGE_M,
        wrist_correction_deg: float = HANDOVER_DEFAULT_WRIST_CORRECTION_DEG,
        reacquire_range_deg: float = HANDOVER_DEFAULT_REACQUIRE_RANGE_DEG,
        contact_stiffness_n_per_m: float = HANDOVER_DEFAULT_CONTACT_STIFFNESS_N_PER_M,
        angular_stiffness_n_per_deg: float = HANDOVER_DEFAULT_ANGULAR_STIFFNESS_N_PER_DEG,
        target_pose: tuple[float, float, float] = HANDOVER_DEFAULT_TARGET_POSE,
    ) -> None:
        """Construct the env with injected physical params (ADR-026; ADR-009).

        Every binding number (success windows, takt-derived budget, ego kinematic
        ranges) is a keyword here; the defaults are NON-BINDING placeholders for unit
        tests. Raises ``KeyError`` for an unknown ``condition_id``.
        """
        if condition_id not in _CONDITION_TABLE:
            raise KeyError(
                f"unknown handover-place condition {condition_id!r}; "
                f"valid: {sorted(_CONDITION_TABLE)}"
            )
        self.condition_id = condition_id
        self.condition = _CONDITION_TABLE[condition_id]
        self.root_seed = int(root_seed)
        self.lateral_window_m = float(lateral_window_m)
        self.angular_window_deg = float(angular_window_deg)
        self.seating_force_limit_n = float(seating_force_limit_n)
        self.regrasp_budget_s = float(regrasp_budget_s)
        self.regrasp_duration_s = float(regrasp_duration_s)
        self.translation_range_m = float(translation_range_m)
        self.wrist_correction_deg = float(wrist_correction_deg)
        self.reacquire_range_deg = float(reacquire_range_deg)
        self.contact_stiffness_n_per_m = float(contact_stiffness_n_per_m)
        self.angular_stiffness_n_per_deg = float(angular_stiffness_n_per_deg)
        self._target_pose_base = np.asarray(target_pose, dtype=np.float64)

        self._phase: int = HANDOVER_PHASE_PRESENT
        self._initial_state_seed: int = 0
        self._target_pose: NDArray[np.float64] = self._target_pose_base.copy()
        self._lateral_offset: NDArray[np.float64] = np.zeros(2, dtype=np.float64)
        self._grasp_pose_error_deg: float = 0.0
        self._timing_skew_s: float = 0.0

    def _spec(self) -> dict[str, float]:
        """Task spec exposed in observations (windows + budget; ADR-026).

        Task facts the ego is allowed to know; they carry NO presenter-policy
        information (ADR-009 §Decision).
        """
        return {
            "lateral_window_m": self.lateral_window_m,
            "angular_window_deg": self.angular_window_deg,
            "seating_force_limit_n": self.seating_force_limit_n,
            "regrasp_budget_s": self.regrasp_budget_s,
        }

    def _obs(self) -> dict[str, Any]:
        return {
            "phase": self._phase,
            "target_pose": self._target_pose.copy(),
            "spec": self._spec(),
            "lateral_offset": (
                None if self._phase == HANDOVER_PHASE_PRESENT else self._lateral_offset.copy()
            ),
            "grasp_pose_error_deg": (
                None if self._phase == HANDOVER_PHASE_PRESENT else self._grasp_pose_error_deg
            ),
        }

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset to phase 0 and draw the per-episode target jitter (P6 / ADR-002).

        ``seed`` is the per-episode initial-state seed; two resets with the same seed
        produce byte-identical state. Returns ``(obs, info)`` with
        ``info["initial_state_seed"]`` for paired-bootstrap matching.
        """
        del options
        episode_seed = self.root_seed if seed is None else int(seed)
        self._initial_state_seed = episode_seed
        rng = derive_substream(_SUBSTREAM_NAME, root_seed=episode_seed).default_rng()
        jitter = rng.uniform(-HANDOVER_GOAL_JITTER_HALF_M, HANDOVER_GOAL_JITTER_HALF_M, size=2)
        target = self._target_pose_base.copy()
        target[:2] = target[:2] + jitter
        self._target_pose = target
        self._phase = HANDOVER_PHASE_PRESENT
        self._lateral_offset = np.zeros(2, dtype=np.float64)
        self._grasp_pose_error_deg = 0.0
        self._timing_skew_s = 0.0
        return self._obs(), {"initial_state_seed": self._initial_state_seed}

    def step(
        self, action: NDArray[np.float64]
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Advance one phase (ADR-026 §Decision).

        Phase 0 consumes the 4-vector presenter action (lateral offset, grasp-pose
        error, timing skew). Phase 1 consumes the 4-vector ego action and resolves the
        place/seat gate. Returns the Gymnasium 5-tuple; success is ``terminated and not
        truncated`` with the placed pose inside both windows.
        """
        act = np.asarray(action, dtype=np.float64).reshape(-1)
        if self._phase == HANDOVER_PHASE_PRESENT:
            return self._step_presentation(act)
        return self._step_placement(act)

    def _step_presentation(
        self, act: NDArray[np.float64]
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if act.shape[0] != HANDOVER_PRESENTATION_DIM:
            raise ValueError(
                f"presentation action must have dim {HANDOVER_PRESENTATION_DIM}, "
                f"got {act.shape[0]}"
            )
        self._lateral_offset = act[0:2].copy()
        self._grasp_pose_error_deg = float(act[2])
        self._timing_skew_s = float(act[3])
        self._phase = HANDOVER_PHASE_PLACE
        info = {"phase": HANDOVER_PHASE_PLACE, "initial_state_seed": self._initial_state_seed}
        return self._obs(), 0.0, False, False, info

    def _step_placement(
        self, act: NDArray[np.float64]
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if act.shape[0] != HANDOVER_EGO_DIM:
            raise ValueError(f"ego action must have dim {HANDOVER_EGO_DIM}, got {act.shape[0]}")
        outcome = resolve_placement(
            lateral_offset_m=(float(self._lateral_offset[0]), float(self._lateral_offset[1])),
            grasp_pose_error_deg=self._grasp_pose_error_deg,
            timing_skew_s=self._timing_skew_s,
            ego_translation_m=(float(act[0]), float(act[1])),
            ego_reorient_deg=float(act[2]),
            ego_regrasp_requested=bool(act[3] > _REGRASP_FLAG_THRESHOLD),
            regrasp_budget_s=self.regrasp_budget_s,
            regrasp_duration_s=self.regrasp_duration_s,
            translation_range_m=self.translation_range_m,
            wrist_correction_deg=self.wrist_correction_deg,
            reacquire_range_deg=self.reacquire_range_deg,
            contact_stiffness_n_per_m=self.contact_stiffness_n_per_m,
            angular_stiffness_n_per_deg=self.angular_stiffness_n_per_deg,
            lateral_window_m=self.lateral_window_m,
            angular_window_deg=self.angular_window_deg,
            seating_force_limit_n=self.seating_force_limit_n,
        )
        reward = 1.0 if outcome.success else 0.0
        info = {
            "phase": HANDOVER_PHASE_PLACE,
            "initial_state_seed": self._initial_state_seed,
            "success": outcome.success,
            "residual_lateral_m": outcome.residual_lateral_m,
            "residual_angular_deg": outcome.residual_angular_deg,
            "seating_force_proxy_n": outcome.seating_force_proxy_n,
            "regrasp_requested": outcome.regrasp_requested,
            "regrasp_executed": outcome.regrasp_executed,
            "regrasp_budget_blocked": outcome.regrasp_budget_blocked,
            "beyond_reacquire": outcome.beyond_reacquire,
            "failure_mode": outcome.failure_mode,
            "binding_conjunct": outcome.binding_conjunct,
        }
        return self._obs(), reward, True, False, info


def make_handover_place_env(
    *,
    condition_id: str = HANDOVER_MATCHED_REFERENCE,
    root_seed: int = 0,
    free_regrasp: bool = False,
    lateral_window_m: float = HANDOVER_DEFAULT_LATERAL_WINDOW_M,
    angular_window_deg: float = HANDOVER_DEFAULT_ANGULAR_WINDOW_DEG,
    seating_force_limit_n: float = HANDOVER_DEFAULT_SEATING_FORCE_LIMIT_N,
    regrasp_budget_s: float = HANDOVER_DEFAULT_REGRASP_BUDGET_S,
    regrasp_duration_s: float = HANDOVER_DEFAULT_REGRASP_DURATION_S,
    translation_range_m: float = HANDOVER_DEFAULT_TRANSLATION_RANGE_M,
    wrist_correction_deg: float = HANDOVER_DEFAULT_WRIST_CORRECTION_DEG,
    reacquire_range_deg: float = HANDOVER_DEFAULT_REACQUIRE_RANGE_DEG,
    contact_stiffness_n_per_m: float = HANDOVER_DEFAULT_CONTACT_STIFFNESS_N_PER_M,
    angular_stiffness_n_per_deg: float = HANDOVER_DEFAULT_ANGULAR_STIFFNESS_N_PER_DEG,
    target_pose: tuple[float, float, float] = HANDOVER_DEFAULT_TARGET_POSE,
) -> HandoverPlaceEnv:
    """Build a :class:`HandoverPlaceEnv` for the Gate-0 spike (ADR-026; ADR-007 §P4).

    The runner injects the externally-sourced / Stage-0-derived prereg numbers via the
    explicit keywords. ``free_regrasp=True`` removes the re-grasp budget (sets it to
    infinity) for the Gate-0 diagnostic that splits budget-mediated vs intrinsic
    binding; it must NEVER be used for the Limb-2 coupling-validity measurement itself.
    Mirrors the factory-returns-env pattern of
    :func:`chamber.envs.cocarry.make_cocarry_env`, minus the SAPIEN body (this env is
    purely kinematic and has no Tier-2 surface).
    """
    if free_regrasp:
        regrasp_budget_s = float("inf")
    return HandoverPlaceEnv(
        condition_id=condition_id,
        root_seed=root_seed,
        lateral_window_m=lateral_window_m,
        angular_window_deg=angular_window_deg,
        seating_force_limit_n=seating_force_limit_n,
        regrasp_budget_s=regrasp_budget_s,
        regrasp_duration_s=regrasp_duration_s,
        translation_range_m=translation_range_m,
        wrist_correction_deg=wrist_correction_deg,
        reacquire_range_deg=reacquire_range_deg,
        contact_stiffness_n_per_m=contact_stiffness_n_per_m,
        angular_stiffness_n_per_deg=angular_stiffness_n_per_deg,
        target_pose=target_pose,
    )
