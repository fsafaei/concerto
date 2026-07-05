# SPDX-License-Identifier: Apache-2.0
"""``TaskSpec`` — the per-task-version record of CHAMBER-Bench (ADR-027 §Versioning).

One frozen, ``extra="forbid"`` pydantic instance per *task version*
(``task_id@vN``). The registry in :mod:`chamber.tasks.registry` holds
these instances; the generated manifest (ADR-027 §Versioning: "suite
composition is pinned in a generated manifest") is a deterministic dump
of them. Any change to a task — env behaviour, canonical success
predicate, canonical stress instrument — bumps ``version`` rather than
mutating an existing spec (ADR-027 §Versioning; §Reversibility Type-1
for committed admission verdicts).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

#: The six heterogeneity sub-axes tracked per task/axis cell
#: (ADR-027 §Validity matrix; ADR-007 §Decision taxonomy).
HETEROGENEITY_AXES: tuple[str, ...] = ("AS", "OM", "CR", "CM", "PF", "SA")

#: Per-cell axis-validity status vocabulary (ADR-027 §Validity matrix).
AxisValidity = Literal["validated", "null", "invalid", "untested"]

#: Task admission-status vocabulary (ADR-027 §Tier ladder / §Admission
#: protocol). ``DIAGNOSTIC`` = Tier-0 rig diagnostics; ``CONTROL`` =
#: Tier-1 controls; ``CANDIDATE`` / ``ADMITTED`` = Tier-2 pipeline
#: states (ADMITTED requires a committed admission report); ``CLOSED``
#: = documented closure (a closure is a result, ADR-027 §Tier ladder).
AdmissionStatus = Literal["DIAGNOSTIC", "CONTROL", "CANDIDATE", "ADMITTED", "CLOSED"]


class TaskSpec(BaseModel):
    """Frozen record of one CHAMBER-Bench task version (ADR-027 §Versioning).

    Fields:
        task_id: Stable task identifier (the ``task_id`` half of
            ``task_id@vN``).
        version: Task version; **any** change to the task bumps it
            (ADR-027 §Versioning). ``0`` marks a spec-only placeholder
            with no runnable env factory.
        tier: ADR-027 §Tier ladder position (0 rig diagnostics,
            1 controls, 2 admitted cooperation tasks, 3 documented
            candidates and closures).
        title: One-line human-readable task name.
        env_factory: Dotted path to the env factory (resolved lazily by
            :func:`chamber.tasks.make`, preserving the Tier-1 import
            contract — ADR-001 §Risks / P2). ``None`` for spec-only
            placeholders; :func:`chamber.tasks.make` then raises.
        sim_backend: Simulator substrate (``maniskill3`` /
            ``pure_python`` / ``unspecified (spec-only)``).
        n_agents: Number of agents the task steps.
        action_space_summary: Plain-language action-space description.
        observation_summary: Plain-language observation description.
        stress_channel: How internal coupling force is measured — the
            canonical stress instrument pinned per task version
            (ADR-027 §Versioning). ``None`` when the task has no force
            coupling channel.
        axes: Per-axis validity status over the six heterogeneity
            sub-axes (ADR-027 §Validity matrix). All six keys are
            required.
        admission_status: ADR-027 §Tier ladder / §Admission protocol
            status.
        evidence: Repo-relative paths (files or directories) backing
            the tier and axis statuses — immutable archives, ADRs,
            preregistrations (ADR-027 §Evidence basis; invariant I8).
        notes: Free-text caveats: canonical-instrument rulings, pinned
            leaderboard cell families, open challenges, falsifiers.
        factory_defaults: Keyword defaults :func:`chamber.tasks.make`
            applies before caller overrides — pins the task's canonical
            condition without changing env behaviour (ADR-027
            §Versioning; delegation only, per the wrapper-only rule).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: str
    version: int = Field(ge=0)
    tier: Literal[0, 1, 2, 3]
    title: str
    env_factory: str | None
    sim_backend: str
    n_agents: int = Field(ge=1)
    action_space_summary: str
    observation_summary: str
    stress_channel: str | None
    axes: dict[str, AxisValidity]
    admission_status: AdmissionStatus
    evidence: list[str]
    notes: str
    factory_defaults: dict[str, object] = Field(default_factory=dict)

    @field_validator("axes")
    @classmethod
    def _axes_cover_all_six(cls, value: dict[str, AxisValidity]) -> dict[str, AxisValidity]:
        """Require exactly the six ADR-007 sub-axes as keys (ADR-027 §Validity matrix)."""
        expected = set(HETEROGENEITY_AXES)
        got = set(value)
        if got != expected:
            missing = sorted(expected - got)
            extra = sorted(got - expected)
            msg = (
                f"axes must carry exactly the six heterogeneity sub-axes "
                f"{sorted(expected)}; missing={missing} extra={extra}"
            )
            raise ValueError(msg)
        return value

    @field_validator("env_factory")
    @classmethod
    def _env_factory_is_dotted(cls, value: str | None) -> str | None:
        """A non-``None`` factory must be a module-qualified dotted path (ADR-027 §Versioning)."""
        if value is not None and "." not in value:
            msg = f"env_factory must be a dotted 'module.attr' path, got {value!r}"
            raise ValueError(msg)
        return value

    @property
    def slug(self) -> str:
        """``task_id@vN`` display form (ADR-027 §Versioning)."""
        return f"{self.task_id}@v{self.version}"


__all__ = [
    "HETEROGENEITY_AXES",
    "AdmissionStatus",
    "AxisValidity",
    "TaskSpec",
]
