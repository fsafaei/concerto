# SPDX-License-Identifier: Apache-2.0
"""Task-card prose — the human-language half of the task cards (ADR-027 §Consequences).

The generated ``docs/reference/tasks/`` cards render from the registry
so they can never drift from code. :class:`TaskSpec` carries the
machine-readable record; this module carries the card-only prose
(plain-language description, industrial analogue, run snippet) so the
manifest schema stays exactly the ADR-027 §Versioning record. Keeping
the prose inside :mod:`chamber.tasks` — not inside the render script —
keeps the registry the single source of truth; a registered task
without card prose fails the unit tests.

Industrial analogues cite public knowledge only (invariant: no
planning-document references in repo surfaces).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class TaskCardProse(BaseModel):
    """Card-only prose for one task id (ADR-027 §Consequences).

    Fields:
        description: What the task is — three sentences, plain language.
        industrial_analogue: The real production operation the task
            abstracts (public sources only).
        run_snippet: How to run one episode (Python, copy-pasteable).
        falsifier: The committed falsifier, for Tier-3 candidate cards
            (ADR-027 §Admission protocol applied forward); ``None``
            otherwise.
        open_challenge: The open-challenge statement, for closed-but-
            runnable frontier tasks; ``None`` otherwise.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    description: str
    industrial_analogue: str
    run_snippet: str
    falsifier: str | None = None
    open_challenge: str | None = None


_MAKE_SNIPPET = """\
import chamber.tasks

env = chamber.tasks.make("{task_id}"{extra})
obs, info = env.reset(seed=0)
"""


def _snippet(task_id: str, extra: str = "") -> str:
    return _MAKE_SNIPPET.format(task_id=task_id, extra=extra)


#: Card prose keyed by ``task_id`` (ADR-027 §Consequences). Every
#: registered task id must have an entry — pinned by the unit tests.
CARD_PROSE: dict[str, TaskCardProse] = {
    "stage0_smoke": TaskCardProse(
        description=(
            "Three heterogeneous ManiSkill agents — a Panda arm, a Fetch mobile "
            "manipulator, and an Allegro hand — share one scene and step at "
            "different control rates. The task proves the multi-agent rig (env "
            "construction, per-agent action routing, deterministic seeding) "
            "runs end to end. It carries no cooperation claim; it exists so "
            "harness failures are caught before any benchmark task runs."
        ),
        industrial_analogue=(
            "The commissioning dry-run of a mixed-vendor robot work cell: "
            "before production starts, integrators step every controller "
            "through its motion envelope to prove the cell's plumbing — power, "
            "signals, cycle timing — rather than its productivity."
        ),
        run_snippet=_snippet("stage0_smoke") + "# Requires a SAPIEN-capable host (Tier-2).\n",
    ),
    "mpe_cooperative_push": TaskCardProse(
        description=(
            "Two point agents on a unit-square plane cover two landmarks under "
            "a shared cooperative-coverage reward, using continuous 2-D "
            "velocity actions. The env is a deterministic, dependency-free "
            "equivalent of the classic MPE simple-spread task and runs on any "
            "CPU. It is the substrate for the ego-AHT empirical-guarantee "
            "experiment and for CI smoke coverage of the training loop."
        ),
        industrial_analogue=(
            "Deliberately thin by design: it abstracts two mobile units "
            "jointly spreading over drop-off points on a floor plan, and "
            "exists as a CPU-friendly diagnostic of the learning harness "
            "rather than a model of any production operation."
        ),
        run_snippet=_snippet("mpe_cooperative_push") + "# Pure Python — runs on any host.\n",
    ),
    "stage1_pickplace_as": TaskCardProse(
        description=(
            "A tabletop pick-and-place where the ego arm moves a cube to a "
            "goal while a second robot shares the cell, in a homogeneous "
            "panda/panda and a heterogeneous panda/fetch pairing. Per ADR-026 "
            "the task is ego-solvable, so its action-space cell is "
            "construct-invalid for cooperation claims. It is retained "
            "precisely as a control: a method claiming cooperation gains here "
            "is measuring something else."
        ),
        industrial_analogue=(
            "A single-robot bin-to-conveyor pick-and-place station with a "
            "second robot present in the same cell — the everyday case where "
            "co-presence does not imply cooperation, which is exactly what "
            "the control checks."
        ),
        run_snippet=_snippet("stage1_pickplace_as")
        + "# Requires a SAPIEN-capable host (Tier-2).\n",
    ),
    "stage1_pickplace_om": TaskCardProse(
        description=(
            "The same tabletop pick-and-place viewed through heterogeneous "
            "observation modalities: a vision-only condition masks "
            "proprioception and force-torque, while the augmented condition "
            "adds synthesised force-torque and proprioception. The OM axis "
            "was never fired at the gate and ADR-026 closed its promotion. "
            "The task is retained as a Tier-1 control for "
            "observation-modality claims."
        ),
        industrial_analogue=(
            "A camera-guided picking station versus one with joint encoders "
            "and a wrist force-torque sensor — the standard sensing-tier "
            "trade-off in industrial picking, here kept as a control rather "
            "than a cooperation measurement."
        ),
        run_snippet=_snippet("stage1_pickplace_om")
        + "# Requires a SAPIEN-capable host (Tier-2).\n",
    ),
    "cocarry": TaskCardProse(
        description=(
            "Two manipulators rigidly carry one shared bar to a goal while "
            "keeping it level and keeping internal stress within a frozen "
            "force ceiling. The task is infeasible for one robot, so success "
            "measures cooperation rather than a single ego's competence. Its "
            "rung archives (frozen incumbent, policy-shift and "
            "embodiment-shift measurements) are the committed evidence for "
            "Tier-2 admission."
        ),
        industrial_analogue=(
            "Dual-robot handling of long or heavy payloads — two industrial "
            "arms sharing one rigid part, as in dual-arm transport of "
            "sheet-metal or chassis components in automotive body shops, "
            "where tilt and internal stress limits bind exactly because two "
            "holds share the load."
        ),
        run_snippet=_snippet("cocarry") + "# Requires a SAPIEN-capable host (Tier-2).\n",
    ),
    "handover_place": TaskCardProse(
        description=(
            "A black-box presenter hands a part into a shared workspace; the "
            "ego arm must place and seat it within a downstream tolerance at "
            "cycle time, without a full re-grasp it cannot afford. The "
            "partner binds only through the hand-off initial condition — the "
            "grasp-pose error the ego must correct within its wrist-correction "
            "and re-grasp budgets. The binding is a tolerance phenomenon, so "
            "a pure-Python kinematic resolver captures it and the task runs "
            "on any host."
        ),
        industrial_analogue=(
            "Asynchronous part transfer between a feeder robot and a placing "
            "robot in machine tending: the upstream robot presents a part at "
            "takt, and the downstream robot must seat it in a fixture window "
            "without breaking cycle time for a re-grasp."
        ),
        run_snippet=_snippet("handover_place") + "# Pure Python — runs on any host.\n",
    ),
    "coinsert": TaskCardProse(
        description=(
            "A two-robot hold-and-insert: the ego drives a peg into a blind "
            "socket carried by a free receptacle that only the partner "
            "stabilises, so a single inserter has nothing to push against. "
            "The task closed at a pre-committed HARD_STOP: no reference pair "
            "seats the peg at the committed tolerances because insertion "
            "contact cocks the peg past the seatable tilt region (a geometric "
            "tilt-wedge). The env stays wired and runnable as a frontier "
            "challenge task."
        ),
        industrial_analogue=(
            "Two-robot assembly of close-tolerance connectors where one robot "
            "holds an unfixtured housing and the other inserts — the "
            "hardest-case cousin of standard robotic peg-in-hole insertion, "
            "here at a clearance regime that remains unsolved for the "
            "matched pair."
        ),
        run_snippet=_snippet("coinsert") + "# Requires a SAPIEN-capable host (Tier-2).\n",
        open_challenge=(
            "No reference pair has met the A1 solvability bar at the "
            "committed tolerances; the cause is understood (geometric "
            "tilt-wedge; the simulator-fidelity suspicion was tested against "
            "a MuJoCo oracle and disproven, #257-#258). Solving the "
            "matched-pair problem is the open challenge. No leaderboard cells "
            "exist for this task."
        ),
    ),
    "co_hold_secure": TaskCardProse(
        description=(
            "A fixation-tooling task: one robot holds a part under continuous "
            "contact while the other performs a securing operation at a "
            "moderate, sourced operate tolerance — a fastening or connector "
            "seat, never a zero-clearance peg (the co-insert correction). It "
            "is the pre-committed escalation target if handover-and-place "
            "washes out, and the v1.1 flagship candidate. This is a spec-only "
            "placeholder: no env factory is wired yet."
        ),
        industrial_analogue=(
            "Fixtureless welding and robotic machining or finishing: one "
            "robot holds the workpiece in place of a dedicated fixture while "
            "another welds, drills, or fastens — the settings where high "
            "process force makes the inter-robot coupling strongest."
        ),
        run_snippet=(
            "import chamber.tasks\n\n"
            "# Spec-only placeholder — no env factory is wired yet:\n"
            'chamber.tasks.make("co_hold_secure")  # raises NotImplementedError\n'
        ),
        falsifier=(
            "If a passive fixture — a single robot plus static tooling — "
            "matches the two-robot team under the same tolerances, the task "
            "is fixture-design-plus-single-arm and is not admitted (the A2 "
            "two-robot-infeasibility gate applied to fixation tooling)."
        ),
    ),
    "amr_handover_dynamic": TaskCardProse(
        description=(
            "A mobile robot hands a part to a fixed arm without stopping, so "
            "the cooperation channel is arrival-timing and motion rather than "
            "contact force. The task would extend the benchmark's coupling "
            "vocabulary beyond force coupling. This is a spec-only "
            "placeholder: kinematic and contact-solver-free by design, it is "
            "a low-build-risk v1.x candidate."
        ),
        industrial_analogue=(
            "Non-stop transfer between an autonomous mobile robot and a "
            "fixed picking arm in intralogistics: the AMR streams past the "
            "station and the arm picks from it on the move, buying cycle "
            "time that a stop-and-transfer layout gives up."
        ),
        run_snippet=(
            "import chamber.tasks\n\n"
            "# Spec-only placeholder — no env factory is wired yet:\n"
            'chamber.tasks.make("amr_handover_dynamic")  # raises NotImplementedError\n'
        ),
        falsifier=(
            "If a stop-and-pick fallback matches non-stop performance under "
            "realistic cycle time, the task is deconfliction-plus-pick and "
            "is not admitted."
        ),
    ),
}


def card_prose(task_id: str) -> TaskCardProse:
    """Return the card prose for ``task_id`` (ADR-027 §Consequences).

    Raises ``KeyError`` listing known ids, matching the registry error
    style (ADR-009 §Decision).
    """
    try:
        return CARD_PROSE[task_id]
    except KeyError:
        known = ", ".join(sorted(CARD_PROSE)) or "<none>"
        msg = f"no card prose for task id {task_id!r}; known ids: {known}"
        raise KeyError(msg) from None


__all__ = ["CARD_PROSE", "TaskCardProse", "card_prose"]
