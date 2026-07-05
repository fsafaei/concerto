# SPDX-License-Identifier: Apache-2.0
"""The CHAMBER-Bench v1.0 tier ladder (ADR-027 §Tier ladder).

Registers every task version the v1.0 suite pins. Statuses, axis
cells, and evidence paths transcribe the committed record: the tier
assignments and admission states are ADR-027 §Tier ladder verbatim;
axis cells follow ADR-027 §Validity matrix over the immutable archives
under ``spikes/results/`` (invariant I8); canonical stress instruments
follow ADR-027 §Versioning.

Import side effect: importing this module populates the registry.
:mod:`chamber.tasks` does so on package import, so ``chamber.tasks.get``
/ ``make`` / ``manifest`` work after a plain ``import chamber.tasks``.
"""

from __future__ import annotations

from chamber.tasks.registry import register_task
from chamber.tasks.spec import HETEROGENEITY_AXES, AxisValidity, TaskSpec


def _axes(**overrides: AxisValidity) -> dict[str, AxisValidity]:
    """All six cells ``untested`` except the given overrides (ADR-027 §Validity matrix)."""
    cells: dict[str, AxisValidity] = dict.fromkeys(HETEROGENEITY_AXES, "untested")
    cells.update(overrides)
    return cells


@register_task
def stage0_smoke_v1() -> TaskSpec:
    """Tier-0 rig diagnostic: the Stage-0 tri-embodiment smoke rig (ADR-027 §Tier ladder)."""
    return TaskSpec(
        task_id="stage0_smoke",
        version=1,
        tier=0,
        title="Stage-0 tri-embodiment smoke rig",
        env_factory="chamber.benchmarks.stage0_smoke.make_stage0_env",
        sim_backend="maniskill3",
        n_agents=3,
        action_space_summary=(
            "Dict over three uids (panda_wristcam, fetch, allegro_hand_right) "
            "stepping at heterogeneous control rates (20/10/50 Hz)"
        ),
        observation_summary="Per-agent ManiSkill state observations",
        stress_channel=None,
        axes=_axes(),
        admission_status="DIAGNOSTIC",
        evidence=[
            "adr/ADR-001-fork-vs-build.md",
            "adr/ADR-027-chamber-bench-v1-protocol.md",
        ],
        notes=(
            "Tier-0 rig diagnostic: proves the multi-agent harness (env "
            "construction, per-agent action routing, deterministic seeding) runs "
            "end to end; carries no cooperation claim (ADR-027 §Tier ladder). No "
            "dedicated smoke archive exists under spikes/results/ — the "
            "executable evidence is the Stage-0 acceptance suite (`make smoke`) "
            "plus ADR-001."
        ),
    )


@register_task
def mpe_cooperative_push_v1() -> TaskSpec:
    """Tier-0 rig diagnostic: the CPU-only MPE cooperative-push substrate (ADR-027 §Tier ladder)."""
    return TaskSpec(
        task_id="mpe_cooperative_push",
        version=1,
        tier=0,
        title="MPE cooperative push (CPU diagnostic)",
        env_factory="chamber.envs.mpe_cooperative_push.MPECooperativePushEnv",
        sim_backend="pure_python",
        n_agents=2,
        action_space_summary=(
            "Dict over two uids (ego, partner); continuous 2-D velocity actions "
            "clipped to [-1, 1] per axis"
        ),
        observation_summary=(
            "Per-agent numpy state vectors (agent + landmark positions) under the "
            "shared cooperative-coverage reward"
        ),
        stress_channel=None,
        axes=_axes(),
        admission_status="DIAGNOSTIC",
        evidence=[
            "tests/integration/test_empirical_guarantee.py",
            "adr/ADR-027-chamber-bench-v1-protocol.md",
        ],
        notes=(
            "CPU rig / CI smoke substrate for the ego-AHT empirical-guarantee "
            "experiment (`make empirical-guarantee`); carries no cooperation "
            "claim (ADR-027 §Tier ladder). No run archive exists under "
            "spikes/results/ — the executable empirical-guarantee suite is the "
            "evidence."
        ),
    )


@register_task
def stage1_pickplace_as_v1() -> TaskSpec:
    """Tier-1 control: Stage-1 pick-and-place, action-space conditions (ADR-027 §Tier ladder)."""
    return TaskSpec(
        task_id="stage1_pickplace_as",
        version=1,
        tier=1,
        title="Stage-1 pick-and-place — action-space (AS) control",
        env_factory="chamber.envs.stage1_pickplace.make_stage1_pickplace_env",
        sim_backend="maniskill3",
        n_agents=2,
        action_space_summary=(
            "Dict over ego + partner; homogeneous panda/panda pair vs "
            "heterogeneous panda/fetch pair (condition_id selects within the "
            "preregistered AS pair)"
        ),
        observation_summary=(
            "state_dict observations (qpos/qvel + task state); "
            "Stage1ASStateSynthesizer presents a synthesised flat state channel"
        ),
        stress_channel=None,
        axes=_axes(AS="invalid"),
        admission_status="CONTROL",
        evidence=[
            "adr/ADR-026-coupling-validity-criterion.md",
            "spikes/results/stage1-AS-2026-06-11/GATE_VERDICT_REPORT_2026-06-11.md",
        ],
        notes=(
            "Construct-invalid for cooperation as operationalized (ego-solvable; "
            "the AS cell is `invalid` per ADR-026's reinterpretation of the "
            "immutable 2026-06-11 archive) — retained precisely as a Tier-1 "
            "control (ADR-027 §Tier ladder): a method claiming cooperation gains "
            "here is measuring something else. Default condition is the "
            "heterogeneous panda+fetch cell; pass "
            "condition_id='stage1_pickplace_panda_only_mappo_shared_param' for "
            "the homogeneous baseline."
        ),
        factory_defaults={
            "condition_id": "stage1_pickplace_panda_plus_fetch_ego_aht_happo_per_agent",
        },
    )


@register_task
def stage1_pickplace_om_v1() -> TaskSpec:
    """Tier-1 control: Stage-1 pick-and-place, OM conditions (ADR-027 §Tier ladder)."""
    return TaskSpec(
        task_id="stage1_pickplace_om",
        version=1,
        tier=1,
        title="Stage-1 pick-and-place — observation-modality (OM) control",
        env_factory="chamber.envs.stage1_pickplace.make_stage1_pickplace_env",
        sim_backend="maniskill3",
        n_agents=2,
        action_space_summary="Dict over ego (panda_wristcam) + partner (fetch)",
        observation_summary=(
            "rgb+depth+state_dict; the vision-only condition masks proprio + "
            "force-torque via Stage1OMChannelFilter, the vision+FT+proprio "
            "condition enables synthesised force-torque (condition_id selects)"
        ),
        stress_channel=None,
        axes=_axes(),
        admission_status="CONTROL",
        evidence=[
            "adr/ADR-026-coupling-validity-criterion.md",
        ],
        notes=(
            "The OM axis was never fired at the gate — ADR-026 closed its "
            "promotion, so the OM cell stays `untested` and the task is retained "
            "as a Tier-1 control (ADR-027 §Tier ladder). Default condition is "
            "'stage1_pickplace_vision_only'; pass "
            "condition_id='stage1_pickplace_vision_plus_force_torque_plus_proprio' "
            "for the FT-augmented cell."
        ),
        factory_defaults={"condition_id": "stage1_pickplace_vision_only"},
    )


@register_task
def cocarry_v1() -> TaskSpec:
    """Tier-2 candidate: rigid dual-arm co-carry (ADR-027 §Tier ladder; ADR-026 §Decision 4)."""
    return TaskSpec(
        task_id="cocarry",
        version=1,
        tier=2,
        title="Co-carry — rigid dual-arm bar transport",
        env_factory="chamber.envs.cocarry.make_cocarry_env",
        sim_backend="maniskill3",
        n_agents=2,
        action_space_summary=(
            "Dict over ego (panda_wristcam) + partner (panda_partner, or "
            "xarm6_robotiq under the embodiment-shift condition); per-agent arm "
            "control with a rigid dual-hold bar attach"
        ),
        observation_summary=(
            "State observations including bar pose, tilt, and wrist "
            "constraint-solver force telemetry"
        ),
        stress_channel=(
            "Wrist constraint-solver force — the 6-axis spatial force through "
            "each holding arm's hand incoming joint — as it gates success on "
            "main (canonical instrument; f_max = 130.5697 N per "
            "spikes/results/cocarry/rung2/cocarry_rung2_freeze_manifest.json). "
            "The rung-4c embodiment-invariant bar coupling-stress instrument "
            "(365.6 N ceiling, post-hoc only) is secondary telemetry per "
            "ADR-027 §Versioning's canonical-instrument rule."
        ),
        axes=_axes(AS="null"),
        admission_status="CANDIDATE",
        evidence=[
            "spikes/results/cocarry/rung2/cocarry_rung2_freeze_manifest.json",
            "spikes/results/cocarry/rung2/cocarry_rung2_freeze_selection_prereg.json",
            "spikes/results/cocarry/rung2/cocarry_rung2_freeze_selection_validation.json",
            "spikes/results/cocarry/rung3",
            "spikes/results/cocarry/rung4",
            "spikes/results/cocarry/rung4b",
            "spikes/results/cocarry/rung4c",
            "spikes/results/cocarry/rung4d",
            "spikes/results/cocarry/rung4e",
            "spikes/results/cocarry/base_probe",
            "adr/ADR-026-coupling-validity-criterion.md",
        ],
        notes=(
            "CANDIDATE flips to ADMITTED only when the retroactive admission "
            "report (A1: matched reference 1.00, held-out validation 24/24 per "
            "the freeze-selection archives; A2: the rung positive controls; A3: "
            "measured against B-BLIND) is committed under ADR-027 §Admission "
            "protocol. The `null` AS cell records the embodiment-heterogeneity "
            "robust null established by the rung-4 archives (rung-4d "
            "pose-artifact retraction + rung-4e task-fair pose/controller "
            "search); the rung-3 policy-heterogeneity pooled null (with its "
            "stiff-impedance caveat) is recorded in the rung3 archive. Changing "
            "the canonical success predicate or stress instrument creates "
            "cocarry@2 (ADR-027 §Versioning)."
        ),
    )


@register_task
def handover_place_v1() -> TaskSpec:
    """Tier-2 candidate: handover-and-place under takt pressure (ADR-027 §Tier ladder)."""
    return TaskSpec(
        task_id="handover_place",
        version=1,
        tier=2,
        title="Handover-and-place under takt pressure",
        env_factory="chamber.envs.handover_place.make_handover_place_env",
        sim_backend="pure_python",
        n_agents=2,
        action_space_summary=(
            "Kinematic decision surface: the ego corrects the presented "
            "grasp-pose error by wrist correction or a budgeted "
            "set-down-and-regrasp; the presenter binds only through the hand-off "
            "initial condition"
        ),
        observation_summary=(
            "Kinematic pose/timing state (presented part pose + grasp-pose "
            "error, takt and regrasp budgets)"
        ),
        stress_channel=(
            "Seating contact force from the kinematic resolver, gated by seating_force_limit_n"
        ),
        axes=_axes(),
        admission_status="CANDIDATE",
        evidence=[
            "spikes/results/handover-place-gate0-2026-06-26/GATE_VERDICT_REPORT_2026-06-26.md",
            "spikes/preregistration/handover_place/gate0.yaml",
        ],
        notes=(
            "Admission-eligible (ADR-027 §Tier ladder): the measured Gate-0 "
            "verdict is COUPLING_VALID + SOLVABLE (PR #263) under the tagged "
            "Rev-2 prereg (prereg-handover-place-gate0-rev2-2026-06-26, blob "
            "b26cfba74a2f1bb9ac295810e4846fe23742ff1b); CANDIDATE flips to "
            "ADMITTED when the committed admission report wraps that verdict. "
            "Canonical benchmark cells are pinned to the measured coupling-valid "
            "region: the clearance-0.2 family with 30°/45° grasp-pose mismatch "
            "across the realistic takt band — 22 of 216 measured cells, valid "
            "takt windows at clearance 0.2 of [0.3, 1.5] s (fast profile) and "
            "[0.3, 2.0] s (slow profile) per GATE_VERDICT_REPORT_2026-06-26.md. "
            "Cells outside that region are not leaderboard cells."
        ),
    )


@register_task
def coinsert_v1() -> TaskSpec:
    """Tier-3 closure: co-insert hold-and-insert, runnable challenge (ADR-027 §Tier ladder)."""
    return TaskSpec(
        task_id="coinsert",
        version=1,
        tier=3,
        title="Co-insert — hold-and-insert (closed; open challenge)",
        env_factory="chamber.envs.coinsert.make_coinsert_env",
        sim_backend="maniskill3",
        n_agents=2,
        action_space_summary=(
            "Dict over ego inserter (panda_wristcam) + holder (panda_partner "
            "gripping a free receptacle); per-agent arm control"
        ),
        observation_summary=(
            "State observations including peg seating depth, relative tilt, and "
            "interaction-wrench telemetry"
        ),
        stress_channel=(
            "Insertion and coupling force limits (f_insert_max / f_couple_max), "
            "derived from measured matched-pair distributions"
        ),
        axes=_axes(),
        admission_status="CLOSED",
        evidence=[
            "spikes/results/coinsert/COINSERT_CLOSURE_2026-06-24.md",
        ],
        notes=(
            "Unsolved: no reference pair has met A1 at the committed tolerances "
            "— see the closure archive. The cause is understood (geometric "
            "tilt-wedge: the seatable region, tilt < 0.7°, and the "
            "achievable-control region under contact, ~0.9-2.8°, do not "
            "overlap); the simulator-fidelity suspicion was tested against a "
            "MuJoCo oracle and disproven (#257-#258). Solving the matched-pair "
            "problem is the open challenge; the env factory stays wired so the "
            "task is runnable as a frontier/challenge task. No leaderboard "
            "cells exist for this task."
        ),
    )


@register_task
def co_hold_secure_v0() -> TaskSpec:
    """Tier-3 candidate placeholder: co-hold-and-secure (ADR-027 §Tier ladder)."""
    return TaskSpec(
        task_id="co_hold_secure",
        version=0,
        tier=3,
        title="Co-hold-and-secure (spec-only candidate)",
        env_factory=None,
        sim_backend="unspecified (spec-only)",
        n_agents=2,
        action_space_summary=(
            "To be specified: one robot holds a part under continuous contact "
            "while the other performs a securing operation"
        ),
        observation_summary="To be specified at Gate-0 pre-registration",
        stress_channel=(
            "Holding-contact force under continuous contact (to be pinned at "
            "Gate-0 pre-registration)"
        ),
        axes=_axes(),
        admission_status="CANDIDATE",
        evidence=[
            "spikes/preregistration/handover_place/gate0.yaml",
            "adr/ADR-027-chamber-bench-v1-protocol.md",
        ],
        notes=(
            "The pre-committed escalation target if handover-and-place washes "
            "out (the handover-place Gate-0 prereg's escalation clause names "
            "cross-modal co-hold-and-secure), and the v1.1 flagship candidate. "
            "Fixation-tooling structure: one robot holds a part under "
            "continuous contact, the other performs a securing operation, at a "
            "moderate, sourced operate tolerance — a fastening or connector "
            "seat, never a zero-clearance peg (the co-insert correction). "
            "Industrial anchors (public sources): fixtureless welding and "
            "robotic machining/finishing, where one robot holds while another "
            "operates and high process force makes the coupling strongest. "
            "Committed falsifier (the A2 gate applied to fixation tooling): if "
            "a passive fixture — single robot plus static tooling — matches the "
            "two-robot team under the same tolerances, the task is "
            "fixture-design-plus-single-arm and is not admitted."
        ),
    )


@register_task
def amr_handover_dynamic_v0() -> TaskSpec:
    """Tier-3 candidate placeholder: timing-coupled AMR dynamic handover (ADR-027 §Tier ladder)."""
    return TaskSpec(
        task_id="amr_handover_dynamic",
        version=0,
        tier=3,
        title="AMR dynamic handover (spec-only candidate)",
        env_factory=None,
        sim_backend="unspecified (spec-only)",
        n_agents=2,
        action_space_summary=(
            "To be specified: a mobile robot hands a part to a fixed arm "
            "without stopping (kinematic, no contact solver)"
        ),
        observation_summary="To be specified at Gate-0 pre-registration",
        stress_channel=None,
        axes=_axes(),
        admission_status="CANDIDATE",
        evidence=[
            "adr/ADR-027-chamber-bench-v1-protocol.md",
        ],
        notes=(
            "Timing-coupled cooperation candidate: the coupling channel is "
            "arrival-timing/motion, not contact force, so the task would extend "
            "the benchmark's coupling vocabulary beyond force coupling "
            "(stress_channel is None by design). Committed falsifier: if a "
            "stop-and-pick fallback matches non-stop performance under "
            "realistic cycle time, the task is deconfliction-plus-pick and is "
            "not admitted. CPU-friendly to build (kinematic, no contact solver) "
            "— a low-build-risk v1.x candidate."
        ),
    )


__all__: list[str] = []
