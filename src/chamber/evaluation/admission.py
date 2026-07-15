# SPDX-License-Identifier: Apache-2.0
r"""The Tier-2 admission protocol, executable (ADR-027 §Admission protocol; ADR-026 §Decision 2).

ADR-027 admits a task to Tier 2 only by a **committed admission
report** showing, under preregistered thresholds:

- **A1 — solvability.** A reference policy pair (scripted allowed)
  reaches ``success_ci_low >= tau_solv`` with the task's stress channel
  within ``stress_limit``.
- **A2 — two-robot infeasibility.** The preregistered partner-ablated
  variant reaches ``success_ci_high <= tau_infeasible``, *and* the
  coupling constraint measurably binds on matched successful episodes —
  the committed stress distribution is reported as the binding evidence
  (ADR-026 §Decision 2's coupling positive-control, verbatim).
- **A3 — partner-relevance.** A partner-blind ego (B-BLIND, ADR-011 as
  amended) underperforms the coupling-aware reference by
  ``delta_ci_low >= delta_min``; if instead the gap's upper CI falls
  below ``delta_min`` the task is ego-solvable and is **demoted to a
  Tier-1 CONTROL** (ADR-027 §Admission protocol).
- **A4 — instrument (ego) robustness** (ADR-027 §Admission A4, added
  2026-07-15; runs only for instrument-based contrasts, i.e. when the
  spec commits ``c_min_ego``). The policy used as the fixed measuring
  instrument of a contrast must clear a preregistered ego-robustness
  floor — per-partner ``success_ci_low >= c_min_ego``, at the task's
  own success+stress bar — across the **admitted partner set**, not
  only its training/matched partner. A brittle instrument makes the
  contrast **inadmissible / un-instrumentable as built** (not a null,
  not a heterogeneity finding — ADR-026 §Decision 2: a construct
  problem is never re-described as a partner/axis result); the
  instrument's per-partner robustness profile is a mandatory reported
  artifact.

This module makes that protocol something a reviewer executes rather
than prose: :func:`run_admission` drives every measured check cell
through the ADR-028 result-bundle machinery (each cell is itself a
``chamber-eval verify``-passing bundle), applies the pre-committed
threshold rules, and writes an immutable admission archive —
``admission_report.json`` + ``ADMISSION_REPORT.md`` + the cell bundles
(invariant I8). Verdict vocabulary: ``ADMITTED`` (all checks pass),
``CONTROL`` (A2/A3 failure — the ego-solvable direction),
``NOT_SOLVABLE`` (A1 failure), ``UNINSTRUMENTABLE`` (A4 failure — a
brittle instrument), ``INDETERMINATE`` (a CI straddles its threshold;
resolved by the single pre-committed seed extension, then final).

Where a task carries a committed, tag-locked measurement already (the
handover-place Gate-0 archive), a check may **wrap** that immutable
evidence instead of re-running it (I8: the archive is the evidence;
numbers are re-extracted from SHA-verified files, never copied by
hand). Thresholds live in the pre-registration document
(ADR-007 §Discipline; document-form ``PreregDocument`` per ADR-028
§Decision 2) and are committed before any measured run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib import import_module
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator

import chamber
from chamber.evaluation.bootstrap import PairedEpisode, pacluster_bootstrap
from chamber.evaluation.bundles import (
    compute_summary,
    git_provenance,
    platform_fingerprint,
    sha256_file,
    write_bundle_dir,
)
from chamber.evaluation.prereg import (
    PreregDocument,
    load_prereg_document,
    verify_git_tag,
)
from chamber.evaluation.results import (
    BundleSummary,
    EpisodeResult,
    ResultBundle,
    SeedSchedule,
)
from concerto.training.seeding import derive_substream

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Mapping
    from pathlib import Path

#: Admission-report schema version (ADR-027 §Admission protocol).
#: Independent of the result-archive ``SCHEMA_VERSION`` (ADR-028) and
#: the prereg ``PREREG_SCHEMA_VERSION``; bumping requires a new ADR
#: (invariant I9).
ADMISSION_REPORT_SCHEMA_VERSION: int = 1

#: Canonical file names of an admission archive (ADR-027 §Admission protocol).
ADMISSION_REPORT_JSON: str = "admission_report.json"
ADMISSION_REPORT_MD: str = "ADMISSION_REPORT.md"

#: ``derive_substream`` label for the A3 paired-gap bootstrap — pinned so
#: the reported CI is byte-recomputable from the committed bundles
#: (ADR-002 P6; ADR-028 §Decision 3 discipline).
A3_BOOTSTRAP_SUBSTREAM: str = "evaluation.admission.a3_paired_bootstrap"

#: The admission checks, in evaluation order (ADR-027 §Admission
#: protocol; A4 per ADR-027 §Admission A4 runs only for
#: instrument-based contrasts).
AdmissionCheckId = Literal["A1", "A2", "A3", "A4"]

#: Per-check outcome vocabulary (ADR-027 §Admission protocol).
CheckOutcome = Literal["PASS", "FAIL", "INDETERMINATE"]

#: Overall verdict vocabulary (ADR-027 §Admission protocol; the CONTROL
#: demotion rule, the NOT_SOLVABLE short-circuit, and the A4
#: UNINSTRUMENTABLE brittle-instrument verdict — ADR-027 §Admission A4).
AdmissionVerdict = Literal[
    "ADMITTED", "CONTROL", "NOT_SOLVABLE", "UNINSTRUMENTABLE", "INDETERMINATE"
]


class AdmissionError(RuntimeError):
    """Raised when an admission run cannot proceed (ADR-027 §Admission protocol)."""


class AdmissionCellSpec(BaseModel):
    """One measured admission cell — a policy/partner configuration (ADR-027 §Admission protocol).

    Attributes:
        cell_id: Archive-stable cell name (becomes the bundle directory
            name, e.g. ``a1_reference``).
        runner: Key into the cell-runner registry
            (:data:`chamber.benchmarks.admission_cells.CELL_RUNNERS`);
            resolved lazily so this module stays Tier-1-importable
            (ADR-001 §Risks / P2).
        policy_id: Ego policy identity recorded in the cell's bundle
            (ADR-011 baseline ids; e.g. ``ref_script``, ``b_blind``).
        partner_name: Partner registry name for the partner seat
            (ADR-009 §Decision; recorded with its identity hash,
            ADR-028 §Decision 1).
        params: Runner-specific committed parameters (condition ids,
            env overrides). Frozen by the prereg tag blob.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    cell_id: str
    runner: str
    policy_id: str
    partner_name: str
    params: dict[str, Any] = Field(default_factory=dict)


class WrappedEvidenceSpec(BaseModel):
    """A check backed by an existing immutable archive (ADR-027 §Admission protocol; I8).

    Wrapping never re-runs and never hand-copies numbers: the archive
    files are SHA-256-verified against the hashes committed in the
    prereg, and the statistics are re-extracted from the verified bytes
    by a named extractor at report-build time.

    Attributes:
        archive: Repo-relative archive directory the evidence lives in.
        files: Repo-relative file paths → expected SHA-256 hex digests
            (committed in the prereg; the wrap fails loudly on any
            mismatch).
        extractor: Key into the wrap-extractor registry
            (:data:`chamber.benchmarks.admission_cells.WRAP_EXTRACTORS`).
        params: Extractor-specific committed parameters (e.g. the
            pinned coupling-valid cell family).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    archive: str
    files: dict[str, str]
    extractor: str
    params: dict[str, Any] = Field(default_factory=dict)


class AdmissionSpec(BaseModel):
    """The preregistered admission-protocol specification (ADR-027 §Admission protocol).

    Thresholds are committed before results exist and are never tuned
    afterward (ADR-007 §Discipline); the spec is carried inside a
    document-form pre-registration (``PreregDocument.parameters
    ["admission"]``) locked to ``git_tag``.

    Attributes:
        task_id: ADR-027 task under admission.
        task_version: ADR-027 task version (``task_id@vN``).
        tau_solv: A1 success floor (on the CI lower bound).
        stress_limit: A1 stress ceiling on successful episodes'
            ``force_peak`` (the task's canonical stress instrument,
            ADR-027 §Versioning); ``None`` for tasks with no stress
            channel.
        tau_infeasible: A2 success ceiling (on the CI upper bound) for
            the partner-ablated variant.
        delta_min: A3 margin the coupling-aware reference must hold
            over the partner-blind ego (CI lower bound of the gap).
        seeds: Committed cluster-seed list per measured cell.
        episodes_per_seed: Episode budget per ``(cell, seed)``.
        extension_seeds: The single pre-committed seed extension used
            to resolve an INDETERMINATE check, then final (ADR-027
            §Admission protocol).
        n_resamples: Bootstrap resample count (ADR-008 default 2000).
        root_seed: Run-level root seed for episode + bootstrap
            substreams (ADR-002 P6).
        a1: The A1 reference cell (measured, or wrapped committed
            evidence).
        a2: The A2 partner-ablated cell (always measured — it is the
            genuinely new, cheap cell even for wrapped tasks).
        a3: The A3 partner-blind cell (measured; compared against the
            A1 reference cell on matched initial states), or wrapped
            committed gap evidence.
        c_min_ego: A4 ego-robustness floor (ADR-027 §Admission A4) —
            the measuring instrument's per-partner success CI lower
            bound must clear it across the admitted partner set, at the
            task's own success+stress bar. ``None`` (the default) for
            plain task admissions: A4 is skipped and the A1/A2/A3
            semantics are unchanged.
        a4: The A4 instrument cell (always measured): ``policy_id``
            names the instrument under test and ``partner_name`` the
            admitted partner set it sweeps (an ADR-009 set slug, e.g.
            ``cocarry_partners@v1``); the runner stamps each episode's
            ``metadata["member"]`` with the set-member name (the
            leaderboard per-partner convention). Committed iff
            ``c_min_ego`` is.
        git_tag: Pre-registration tag the spec is locked to
            (ADR-007 §Discipline).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: str
    task_version: int
    tau_solv: float
    stress_limit: float | None
    tau_infeasible: float
    delta_min: float
    seeds: list[int]
    episodes_per_seed: PositiveInt
    extension_seeds: list[int]
    n_resamples: PositiveInt = 2000
    root_seed: int = 0
    a1: AdmissionCellSpec | WrappedEvidenceSpec
    a2: AdmissionCellSpec
    a3: AdmissionCellSpec | WrappedEvidenceSpec
    c_min_ego: float | None = None
    a4: AdmissionCellSpec | None = None
    git_tag: str

    @model_validator(mode="after")
    def _a4_fields_travel_together(self) -> AdmissionSpec:
        """A4 needs both the floor and the cell (ADR-027 §Admission A4); loud otherwise."""
        if (self.c_min_ego is None) != (self.a4 is None):
            msg = (
                "c_min_ego and a4 commit the A4 ego-robustness gate together "
                "(ADR-027 §Admission A4): set both for an instrument-based "
                "contrast, neither for a plain task admission"
            )
            raise ValueError(msg)
        return self


def admission_spec_from_prereg(doc: PreregDocument) -> AdmissionSpec:
    """Build the :class:`AdmissionSpec` from a document-form prereg (ADR-028 §Decision 2).

    The spec body lives in ``doc.parameters["admission"]``; ``task_id``
    and ``git_tag`` are injected from the document envelope so the two
    can never disagree with the tag-locked YAML.

    Raises:
        AdmissionError: When the ``admission`` parameter block is
            missing.
        pydantic.ValidationError: When the block does not match
            :class:`AdmissionSpec` (unknown keys fail loudly).
    """
    block = doc.parameters.get("admission")
    if not isinstance(block, dict):
        msg = (
            f"prereg for task {doc.task_id!r} carries no 'admission' parameter "
            "block (ADR-027 §Admission protocol)"
        )
        raise AdmissionError(msg)
    payload = {**block, "task_id": doc.task_id, "git_tag": doc.git_tag}
    return AdmissionSpec.model_validate(payload)


class CheckReport(BaseModel):
    """One admission check's result (ADR-027 §Admission protocol).

    Attributes:
        check: ``A1`` / ``A2`` / ``A3`` / ``A4``.
        outcome: Final outcome after any pre-committed extension.
        criterion: The pre-committed rule, verbatim, as applied.
        statistics: The numbers the outcome was computed from (IQM,
            CI bounds, stress percentiles, paired-gap CI).
        bundles: Bundle directory names (relative to the admission
            archive) contributing to this check; empty for wrapped
            checks.
        evidence: Evidence file → SHA-256 (bundle ``bundle.json``
            hashes, or the wrapped archive files).
        extended: Whether the pre-committed seed extension was used.
        notes: Free-text caveats.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    check: AdmissionCheckId
    outcome: CheckOutcome
    criterion: str
    statistics: dict[str, float]
    bundles: list[str]
    evidence: dict[str, str]
    extended: bool = False
    notes: str = ""


class AdmissionReport(BaseModel):
    """The committed admission report — the Tier-2 entry evidence (ADR-027 §Admission protocol).

    Attributes:
        schema_version: :data:`ADMISSION_REPORT_SCHEMA_VERSION`.
        task_id: Task under admission.
        task_version: Task version.
        prereg_git_tag: Verified pre-registration tag (ADR-007
            §Discipline).
        prereg_blob_sha: Verified prereg blob SHA at the tag.
        git_sha: Launch commit (ADR-028 §Decision 1 provenance).
        dirty: Whether the working tree was dirty at run time (a dirty
            report is never registry-flip evidence — mirrors ADR-028
            §Rev 2026-07-05).
        date_stamp: Caller-supplied date label of the archive
            (``<task_id>-<date>``); passed in, never sampled, so the
            report stays deterministic (ADR-002 P6).
        checks: Per-check reports in A1/A2/A3/A4 order (A4 present only
            for instrument-based contrasts); a NOT_SOLVABLE
            short-circuit records A1 only.
        verdict: Overall verdict (ADR-027 §Admission protocol).
        seed_extension_used: Whether any check consumed the single
            pre-committed extension.
        binding_evidence: The A2 coupling-binding evidence — the stress
            distribution on matched successful episodes (ADR-026
            §Decision 2 positive-control).
        ego_robustness_profile: The A4 mandatory reported artifact
            (ADR-027 §Admission A4): admitted-set member name → the
            instrument's success point + CI + stress percentiles with
            that partner. ``None`` for plain A1/A2/A3 admissions —
            additive and optional, so schema-1 reports written before
            A4 existed still validate unchanged.
        notes: Free-text caveats.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = ADMISSION_REPORT_SCHEMA_VERSION
    task_id: str
    task_version: int
    prereg_git_tag: str
    prereg_blob_sha: str
    git_sha: str
    dirty: bool
    date_stamp: str
    checks: list[CheckReport]
    verdict: AdmissionVerdict
    seed_extension_used: bool
    binding_evidence: dict[str, float]
    ego_robustness_profile: dict[str, dict[str, float]] | None = None
    notes: str = ""


@dataclass(frozen=True)
class CellRun:
    """Raw output of one measured admission cell (ADR-027 §Admission protocol).

    Produced by a cell runner
    (:mod:`chamber.benchmarks.admission_cells`); consumed by
    :func:`run_admission`, which wraps it into an ADR-028 v3 bundle.

    Attributes:
        episodes_by_seed: Episode records keyed by cluster seed
            (``force_peak`` carries the canonical stress channel's
            episode peak where the task has one).
        partner_material: Serialized partner-spec material for
            ``partners.json`` (ADR-028 §Decision 1).
        partner_hashes: Partner identity hashes keyed by member name
            (ADR-018 custody).
        substream_labels: ``derive_substream`` patterns the cell used
            (ADR-002 P6).
    """

    episodes_by_seed: dict[int, list[EpisodeResult]]
    partner_material: list[dict[str, object]]
    partner_hashes: dict[str, str]
    substream_labels: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pre-committed threshold rules (pure functions; unit-tested exhaustively).
# ---------------------------------------------------------------------------


def a1_outcome(ci_low: float, ci_high: float, tau_solv: float) -> CheckOutcome:
    """A1 solvability rule (ADR-027 §Admission protocol).

    PASS iff the success CI lower bound clears ``tau_solv``; FAIL iff
    even the upper bound falls short; INDETERMINATE on a straddle.
    """
    if ci_low >= tau_solv:
        return "PASS"
    if ci_high < tau_solv:
        return "FAIL"
    return "INDETERMINATE"


def a2_outcome(ci_low: float, ci_high: float, tau_infeasible: float) -> CheckOutcome:
    """A2 two-robot-infeasibility rule (ADR-027 §Admission protocol; ADR-026 §Decision 2).

    PASS iff the ablated variant's success CI upper bound stays at or
    below ``tau_infeasible``; FAIL iff even the lower bound exceeds it;
    INDETERMINATE on a straddle.
    """
    if ci_high <= tau_infeasible:
        return "PASS"
    if ci_low > tau_infeasible:
        return "FAIL"
    return "INDETERMINATE"


def a3_outcome(delta_ci_low: float, delta_ci_high: float, delta_min: float) -> CheckOutcome:
    """A3 partner-relevance rule with the demotion side (ADR-027 §Admission protocol).

    PASS iff the reference-minus-blind gap's CI lower bound clears
    ``delta_min``; FAIL (→ Tier-1 CONTROL demotion) iff the gap's upper
    CI falls below ``delta_min``; INDETERMINATE on a straddle.
    """
    if delta_ci_low >= delta_min:
        return "PASS"
    if delta_ci_high < delta_min:
        return "FAIL"
    return "INDETERMINATE"


def a4_outcome(
    success_ci_by_partner: Mapping[str, tuple[float, float]],
    c_min_ego: float,
) -> CheckOutcome:
    """A4 instrument (ego) robustness rule (ADR-027 §Admission A4; ADR-026 §Decision 2).

    The instrument is measured against every member of the admitted
    partner set; per partner the value is its success
    ``(ci_low, ci_high)``. PASS iff even the weakest partner clears the
    preregistered ego-robustness floor (every ``ci_low >= c_min_ego``);
    FAIL iff any partner's ``ci_high`` falls below the floor — the
    instrument is brittle and the contrast is un-instrumentable as
    built (a construct problem, never a partner/axis result);
    INDETERMINATE when the weakest partner straddles (resolvable by the
    single pre-committed seed extension, then final). The mapping form
    keeps the per-partner aggregation inside the pre-committed rule so
    callers cannot mispair one partner's lower bound with another's
    upper bound.

    Raises:
        AdmissionError: On an empty profile — robustness across an
            empty admitted set is unmeasurable, never vacuously true.
    """
    if not success_ci_by_partner:
        msg = "A4 requires a non-empty admitted-partner profile (ADR-027 §Admission A4)"
        raise AdmissionError(msg)
    if all(ci_low >= c_min_ego for ci_low, _ in success_ci_by_partner.values()):
        return "PASS"
    if any(ci_high < c_min_ego for _, ci_high in success_ci_by_partner.values()):
        return "FAIL"
    return "INDETERMINATE"


def overall_verdict(
    a1: CheckOutcome,
    a2: CheckOutcome | None,
    a3: CheckOutcome | None,
    a4: CheckOutcome | None = None,
) -> AdmissionVerdict:
    """Combine check outcomes into the admission verdict (ADR-027 §Admission protocol).

    ``A1 FAIL`` → ``NOT_SOLVABLE`` (short-circuit; A2/A3/A4 may be
    ``None`` = not run). Any remaining ``INDETERMINATE`` (after the
    pre-committed extension) → ``INDETERMINATE``. ``A2`` or ``A3``
    FAIL → ``CONTROL`` (the ego-solvable direction: the task is
    single-robot-solvable or partner-irrelevant — a Tier-1 control
    either way, ADR-027 §Tier ladder); the task-level demotion outranks
    the instrument-level verdict because a demoted task has no Tier-2
    contrast left to instrument. ``A4`` FAIL → ``UNINSTRUMENTABLE``
    (ADR-027 §Admission A4: the contrast is inadmissible as built — not
    a null, not a heterogeneity finding). All run checks PASS →
    ``ADMITTED``; ``a4 is None`` (not an instrument contrast) leaves
    the three-check verdict table unchanged.
    """
    if a1 == "FAIL":
        return "NOT_SOLVABLE"
    if a1 == "INDETERMINATE" or "INDETERMINATE" in (a2, a3, a4):
        return "INDETERMINATE"
    if a2 == "FAIL" or a3 == "FAIL":
        return "CONTROL"
    if a4 == "FAIL":
        return "UNINSTRUMENTABLE"
    return "ADMITTED"


def stress_statistics(
    episodes: list[EpisodeResult], *, successes_only: bool = True
) -> dict[str, float]:
    """Stress-channel distribution over episodes (ADR-026 §Decision 2 binding evidence).

    Percentiles of ``force_peak`` (the canonical stress instrument's
    episode peak, ADR-027 §Versioning) over the selected episodes;
    empty when no episode carries a stress reading.
    """
    import numpy as np  # noqa: PLC0415 - local to keep module import feather-light

    values = [
        float(ep.force_peak)
        for ep in episodes
        if ep.force_peak is not None and (ep.success or not successes_only)
    ]
    if not values:
        return {}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "stress_n": float(arr.size),
        "stress_p50": float(np.percentile(arr, 50)),
        "stress_p90": float(np.percentile(arr, 90)),
        "stress_p99": float(np.percentile(arr, 99)),
        "stress_max": float(np.max(arr)),
    }


def _flatten(episodes_by_seed: dict[int, list[EpisodeResult]]) -> list[EpisodeResult]:
    return [ep for seed in sorted(episodes_by_seed) for ep in episodes_by_seed[seed]]


def _paired_gap_statistics(
    reference: list[EpisodeResult],
    blind: list[EpisodeResult],
    *,
    n_resamples: int,
    root_seed: int,
) -> dict[str, float]:
    """Paired-cluster bootstrap on the reference-minus-blind success gap (ADR-027 reporting rules).

    Pairs on ``(seed, episode_idx, initial_state_seed)`` — the two
    cells run identical initial-state schedules by construction — and
    routes the bootstrap rng through :data:`A3_BOOTSTRAP_SUBSTREAM`
    so the CI is byte-recomputable (ADR-002 P6).
    """
    ref_by_key = {
        (ep.seed, int(ep.episode_idx), ep.initial_state_seed): 1.0 if ep.success else 0.0
        for ep in reference
    }
    pairs: list[PairedEpisode] = []
    for ep in blind:
        key = (ep.seed, int(ep.episode_idx), ep.initial_state_seed)
        if key in ref_by_key:
            pairs.append(
                PairedEpisode(
                    seed=key[0],
                    episode_idx=key[1],
                    initial_state_seed=key[2],
                    homogeneous=ref_by_key[key],
                    heterogeneous=1.0 if ep.success else 0.0,
                )
            )
    if not pairs:
        msg = "A3 pairing produced no matched (seed, episode, initial-state) pairs"
        raise AdmissionError(msg)
    rng = derive_substream(A3_BOOTSTRAP_SUBSTREAM, root_seed=root_seed).default_rng()
    ci = pacluster_bootstrap(pairs, n_resamples=n_resamples, rng=rng)
    return {
        "n_pairs": float(len(pairs)),
        "delta_iqm": ci.iqm,
        "delta_mean": ci.mean,
        "delta_ci_low": ci.ci_low,
        "delta_ci_high": ci.ci_high,
    }


def _summary_statistics(summary: BundleSummary) -> dict[str, float]:
    return {
        "n_episodes": float(summary.n_episodes),
        "success_mean": summary.success_mean,
        "success_iqm": summary.success_iqm,
        "success_ci_low": summary.success_ci_low,
        "success_ci_high": summary.success_ci_high,
    }


# ---------------------------------------------------------------------------
# The runner.
# ---------------------------------------------------------------------------


def _default_cell_runner_resolver(name: str) -> Callable[..., CellRun]:
    """Resolve a cell runner lazily from :mod:`chamber.benchmarks.admission_cells` (P2)."""
    module = import_module("chamber.benchmarks.admission_cells")
    resolver: Callable[[str], Callable[..., CellRun]] = module.resolve_cell_runner
    return resolver(name)


def _default_wrap_extractor_resolver(name: str) -> Callable[..., dict[str, float]]:
    """Resolve a wrap extractor lazily from :mod:`chamber.benchmarks.admission_cells` (P2)."""
    module = import_module("chamber.benchmarks.admission_cells")
    resolver: Callable[[str], Callable[..., dict[str, float]]] = module.resolve_wrap_extractor
    return resolver(name)


@dataclass
class _RunContext:
    """Internal bookkeeping shared across the checks."""

    spec: AdmissionSpec
    out_dir: Path
    repo_path: Path
    prereg_path: Path
    git_sha: str
    dirty: bool
    repro_command: str
    resolver: Callable[[str], Callable[..., CellRun]]
    render_backend: str | None
    extension_used: bool = False


def _run_cell(
    ctx: _RunContext,
    cell: AdmissionCellSpec,
    *,
    seeds: list[int],
    suffix: str = "",
    per_member_schedule: bool = False,
) -> tuple[str, list[EpisodeResult]]:
    """Run one measured cell and write its v3 bundle (ADR-028 §Decision 1).

    ``per_member_schedule`` (the A4 instrument cell, ADR-027 §Admission
    A4): the cell sweeps the admitted partner set, so its bundle
    records ``episodes_per_seed * n_members`` per seed — the
    ``bundle_runner`` partner-set convention (ADR-028 §Decision 3) —
    and a non-uniform or non-multiple per-seed count fails loudly.
    """
    runner = ctx.resolver(cell.runner)
    run: CellRun = runner(
        cell=cell,
        seeds=seeds,
        episodes_per_seed=int(ctx.spec.episodes_per_seed),
        root_seed=ctx.spec.root_seed,
        render_backend=ctx.render_backend,
    )
    name = f"{cell.cell_id}{suffix}"
    episodes = _flatten(run.episodes_by_seed)
    schedule_episodes_per_seed = int(ctx.spec.episodes_per_seed)
    if per_member_schedule:
        counts = {len(records) for records in run.episodes_by_seed.values()}
        if len(counts) != 1 or next(iter(counts)) % schedule_episodes_per_seed != 0:
            msg = (
                f"A4 instrument cell {name!r} must run the committed "
                f"episodes_per_seed ({schedule_episodes_per_seed}) for every "
                f"admitted member on every seed; got per-seed counts {sorted(counts)}"
            )
            raise AdmissionError(msg)
        schedule_episodes_per_seed = counts.pop()
    bundle = ResultBundle(
        task_id=ctx.spec.task_id,
        task_version=ctx.spec.task_version,
        policy_id=cell.policy_id,
        partner_set_id=f"adhoc:{cell.partner_name}",
        partner_hashes=run.partner_hashes,
        git_sha=ctx.git_sha,
        dirty=ctx.dirty,
        package_version=chamber.__version__,
        seed_schedule=SeedSchedule(
            root_seed=ctx.spec.root_seed,
            seeds=list(seeds),
            episodes_per_seed=schedule_episodes_per_seed,
            substream_labels=list(run.substream_labels),
        ),
        repro_command=ctx.repro_command,
        platform=platform_fingerprint(),
        manifest={},
        summary=compute_summary(
            episodes,
            n_resamples=int(ctx.spec.n_resamples),
            bootstrap_root_seed=ctx.spec.root_seed,
        ),
        prereg_git_tag=ctx.spec.git_tag,
        prereg_blob_sha=None,  # stamped below — the copy travels with every bundle
    )
    # The bundle's prereg linkage re-verifies at chamber-eval verify time
    # against the repo tag; stamp the verified blob SHA.
    blob = verify_git_tag(
        load_prereg_document(ctx.prereg_path), ctx.prereg_path, repo_path=ctx.repo_path
    )
    bundle = bundle.model_copy(update={"prereg_blob_sha": blob})
    write_bundle_dir(
        ctx.out_dir / name,
        bundle_without_manifest=bundle,
        episodes_by_seed=run.episodes_by_seed,
        partner_specs=run.partner_material,
        repro_command=ctx.repro_command,
        prereg_source=ctx.prereg_path,
    )
    return name, episodes


def _wrap_check(
    ctx: _RunContext,
    wrapped: WrappedEvidenceSpec,
) -> tuple[dict[str, float], dict[str, str]]:
    """Verify + extract wrapped committed evidence (I8: wrap, never re-run)."""
    evidence: dict[str, str] = {}
    for rel, expected in sorted(wrapped.files.items()):
        target = ctx.repo_path / rel
        if not target.is_file():
            msg = f"wrapped evidence file missing: {rel}"
            raise AdmissionError(msg)
        actual = sha256_file(target)
        if actual != expected:
            msg = (
                f"wrapped evidence SHA-256 mismatch for {rel}: "
                f"on-disk {actual} != committed {expected} (I8)"
            )
            raise AdmissionError(msg)
        evidence[rel] = actual
    extractor = _default_wrap_extractor_resolver(wrapped.extractor)
    stats = extractor(repo_path=ctx.repo_path, spec=wrapped)
    return stats, evidence


def _bundle_evidence(ctx: _RunContext, bundle_names: list[str]) -> dict[str, str]:
    return {
        f"{name}/bundle.json": sha256_file(ctx.out_dir / name / "bundle.json")
        for name in bundle_names
    }


def _recompute_summary_stats(ctx: _RunContext, episodes: list[EpisodeResult]) -> dict[str, float]:
    summary = compute_summary(
        episodes,
        n_resamples=int(ctx.spec.n_resamples),
        bootstrap_root_seed=ctx.spec.root_seed,
    )
    return _summary_statistics(summary)


def run_admission(
    spec: AdmissionSpec,
    *,
    out_dir: Path,
    repo_path: Path,
    prereg_path: Path,
    date_stamp: str,
    repro_command: str,
    allow_dirty: bool = False,
    render_backend: str | None = None,
    cell_runner_resolver: Callable[[str], Callable[..., CellRun]] | None = None,
) -> AdmissionReport:
    """Execute the admission checks and write the archive (ADR-027 §Admission protocol).

    Order of operations: the pre-registration gate runs first (tag +
    blob verification, ADR-007 §Discipline — nothing is measured on a
    failed gate); the working tree must be clean unless ``allow_dirty``
    (a dirty report is never flip evidence); then A1 → A2 → A3 — plus
    A4 iff the spec commits ``c_min_ego`` (an instrument-based
    contrast, ADR-027 §Admission A4) — with the NOT_SOLVABLE
    short-circuit on A1 failure ("stop before any threshold
    discussion") and the single pre-committed seed extension on a
    straddled CI. Every measured cell is written as an ADR-028 v3
    bundle under ``out_dir`` so ``chamber-eval verify`` admits each
    check independently; the archive is completed by
    ``admission_report.json``, ``ADMISSION_REPORT.md``, and
    ``SHA256SUMS.txt`` (invariant I8 — the directory must not already
    contain files).

    Args:
        spec: The preregistered admission spec.
        out_dir: Admission archive directory to create
            (``spikes/results/admission/<task_id>-<date>/``).
        repo_path: Git working tree (prereg tag + provenance source).
        prereg_path: The tag-locked prereg YAML on disk.
        date_stamp: Archive date label (caller-supplied; P6 — never
            sampled here).
        repro_command: Exact reproduction invocation for the bundles.
        allow_dirty: Tolerate a dirty tree (development only; the
            report records ``dirty: true``).
        render_backend: Optional render backend forwarded to SAPIEN
            cell runners (``"none"`` on headless hosts).
        cell_runner_resolver: Test seam; defaults to the
            :mod:`chamber.benchmarks.admission_cells` registry.

    Returns:
        The :class:`AdmissionReport` as written to the archive.

    Raises:
        AdmissionError: Dirty tree without ``allow_dirty``, non-empty
            ``out_dir``, wrapped-evidence hash mismatch, or a cell
            pairing failure.
        chamber.evaluation.prereg.PreregistrationError: Failed prereg
            gate (callers map this to exit code 4).
    """
    doc = load_prereg_document(prereg_path)
    blob = verify_git_tag(doc, prereg_path, repo_path=repo_path)
    if doc.task_id != spec.task_id or doc.git_tag != spec.git_tag:
        msg = (
            f"prereg document ({doc.task_id!r}, {doc.git_tag!r}) does not match "
            f"the admission spec ({spec.task_id!r}, {spec.git_tag!r})"
        )
        raise AdmissionError(msg)

    provenance = git_provenance(repo_path)
    if provenance.dirty and not allow_dirty:
        msg = (
            f"working tree at {repo_path} is dirty (or git state is unprovable); "
            "an admission report must be traceable to a commit (ADR-028 §Decision 1). "
            "Commit first, or pass allow_dirty for a development run."
        )
        raise AdmissionError(msg)

    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.iterdir()):
        msg = f"refusing to write admission archive into non-empty {out_dir} (I8)"
        raise AdmissionError(msg)

    ctx = _RunContext(
        spec=spec,
        out_dir=out_dir,
        repo_path=repo_path,
        prereg_path=prereg_path,
        git_sha=provenance.sha,
        dirty=provenance.dirty,
        repro_command=repro_command,
        resolver=cell_runner_resolver or _default_cell_runner_resolver,
        render_backend=render_backend,
    )

    checks: list[CheckReport] = []
    binding_evidence: dict[str, float] = {}

    # ----- A1 — solvability -------------------------------------------------
    a1_report, a1_episodes = _check_a1(ctx)
    checks.append(a1_report)
    if a1_report.outcome == "FAIL":
        return _finalize(
            ctx,
            doc_blob=blob,
            date_stamp=date_stamp,
            checks=checks,
            verdict="NOT_SOLVABLE",
            binding_evidence={},
            notes=(
                "A1 failed at the committed thresholds; A2/A3 were not run "
                "(the ADR-027 short-circuit — thresholds do not move after the fact)."
            ),
        )

    # ----- A2 — two-robot infeasibility + binding evidence ------------------
    binding_evidence = stress_statistics(a1_episodes) if a1_episodes else {}
    a2_report = _check_a2(ctx)
    checks.append(a2_report)

    # ----- A3 — partner-relevance -------------------------------------------
    a3_report = _check_a3(ctx, a1_episodes=a1_episodes)
    checks.append(a3_report)

    # ----- A4 — instrument (ego) robustness (instrument contrasts only) ------
    a4_check: CheckOutcome | None = None
    ego_profile: dict[str, dict[str, float]] | None = None
    if spec.c_min_ego is not None:
        a4_report, ego_profile = _check_a4(ctx)
        checks.append(a4_report)
        a4_check = a4_report.outcome

    verdict = overall_verdict(a1_report.outcome, a2_report.outcome, a3_report.outcome, a4_check)
    return _finalize(
        ctx,
        doc_blob=blob,
        date_stamp=date_stamp,
        checks=checks,
        verdict=verdict,
        binding_evidence=binding_evidence,
        ego_robustness_profile=ego_profile,
        notes="",
    )


def _check_a1(ctx: _RunContext) -> tuple[CheckReport, list[EpisodeResult]]:
    """A1 solvability (ADR-027 §Admission protocol): measured or wrapped."""
    spec = ctx.spec
    criterion = (
        f"success_ci_low >= tau_solv ({spec.tau_solv}) with successful-episode "
        f"stress peak <= stress_limit ({spec.stress_limit})"
    )
    if isinstance(spec.a1, WrappedEvidenceSpec):
        stats, evidence = _wrap_check(ctx, spec.a1)
        outcome = a1_outcome(stats["success_ci_low"], stats["success_ci_high"], spec.tau_solv)
        if outcome == "INDETERMINATE":
            outcome = "FAIL"  # wrapped evidence cannot be extended; a straddle does not admit
        return (
            CheckReport(
                check="A1",
                outcome=outcome,
                criterion=criterion,
                statistics=stats,
                bundles=[],
                evidence=evidence,
                notes=f"wrapped committed evidence: {spec.a1.archive} (I8)",
            ),
            [],
        )

    name, episodes = _run_cell(ctx, spec.a1, seeds=spec.seeds)
    bundles = [name]
    stats = _recompute_summary_stats(ctx, episodes)
    outcome = a1_outcome(stats["success_ci_low"], stats["success_ci_high"], spec.tau_solv)
    extended = False
    if outcome == "INDETERMINATE" and spec.extension_seeds:
        ext_name, ext_eps = _run_cell(ctx, spec.a1, seeds=spec.extension_seeds, suffix="-ext")
        bundles.append(ext_name)
        episodes = episodes + ext_eps
        stats = _recompute_summary_stats(ctx, episodes)
        outcome = (
            "PASS" if stats["success_ci_low"] >= spec.tau_solv else "FAIL"
        )  # final — no second extension
        extended = True
        ctx.extension_used = True
    has_stress_channel = any(ep.force_peak is not None for ep in episodes)
    if spec.stress_limit is not None and not has_stress_channel:
        msg = (
            f"A1 stress_limit is committed ({spec.stress_limit}) but the cell "
            "recorded no force_peak stress channel"
        )
        raise AdmissionError(msg)
    stress_ok = True
    stress_stats = stress_statistics(episodes)
    stats = {**stats, **stress_stats}
    if spec.stress_limit is not None and stress_stats:
        stress_ok = stress_stats["stress_max"] <= spec.stress_limit
    if outcome == "PASS" and not stress_ok:
        outcome = "FAIL"
    return (
        CheckReport(
            check="A1",
            outcome=outcome,
            criterion=criterion,
            statistics=stats,
            bundles=bundles,
            evidence=_bundle_evidence(ctx, bundles),
            extended=extended,
        ),
        episodes,
    )


def _check_a2(ctx: _RunContext) -> CheckReport:
    """A2 two-robot infeasibility (ADR-027 §Admission protocol; ADR-026 §Decision 2)."""
    spec = ctx.spec
    criterion = f"success_ci_high <= tau_infeasible ({spec.tau_infeasible})"
    name, episodes = _run_cell(ctx, spec.a2, seeds=spec.seeds)
    bundles = [name]
    stats = _recompute_summary_stats(ctx, episodes)
    outcome = a2_outcome(stats["success_ci_low"], stats["success_ci_high"], spec.tau_infeasible)
    extended = False
    if outcome == "INDETERMINATE" and spec.extension_seeds:
        ext_name, ext_eps = _run_cell(ctx, spec.a2, seeds=spec.extension_seeds, suffix="-ext")
        bundles.append(ext_name)
        episodes = episodes + ext_eps
        stats = _recompute_summary_stats(ctx, episodes)
        outcome = "PASS" if stats["success_ci_high"] <= spec.tau_infeasible else "FAIL"
        extended = True
        ctx.extension_used = True
    stats = {**stats, **stress_statistics(episodes, successes_only=False)}
    return CheckReport(
        check="A2",
        outcome=outcome,
        criterion=criterion,
        statistics=stats,
        bundles=bundles,
        evidence=_bundle_evidence(ctx, bundles),
        extended=extended,
    )


def _check_a3(ctx: _RunContext, *, a1_episodes: list[EpisodeResult]) -> CheckReport:
    """A3 partner-relevance with the demotion rule (ADR-027 §Admission protocol)."""
    spec = ctx.spec
    criterion = (
        f"paired reference-minus-blind gap: delta_ci_low >= delta_min ({spec.delta_min}) "
        f"passes; delta_ci_high < delta_min demotes to Tier-1 CONTROL"
    )
    if isinstance(spec.a3, WrappedEvidenceSpec):
        stats, evidence = _wrap_check(ctx, spec.a3)
        outcome = a3_outcome(stats["delta_ci_low"], stats["delta_ci_high"], spec.delta_min)
        if outcome == "INDETERMINATE":
            outcome = "FAIL"  # wrapped evidence cannot be extended; a straddle does not admit
        return CheckReport(
            check="A3",
            outcome=outcome,
            criterion=criterion,
            statistics=stats,
            bundles=[],
            evidence=evidence,
            notes=f"wrapped committed evidence: {spec.a3.archive} (I8)",
        )

    if not a1_episodes:
        msg = "A3 in measured form requires a measured A1 reference cell to pair against"
        raise AdmissionError(msg)
    name, blind_episodes = _run_cell(ctx, spec.a3, seeds=spec.seeds)
    bundles = [name]
    stats = _paired_gap_statistics(
        a1_episodes,
        blind_episodes,
        n_resamples=int(ctx.spec.n_resamples),
        root_seed=ctx.spec.root_seed,
    )
    blind_summary = _recompute_summary_stats(ctx, blind_episodes)
    stats = {**stats, **{f"blind_{k}": v for k, v in blind_summary.items()}}
    outcome = a3_outcome(stats["delta_ci_low"], stats["delta_ci_high"], spec.delta_min)
    extended = False
    if outcome == "INDETERMINATE" and spec.extension_seeds:
        # The extension extends BOTH sides of the pair on the committed
        # extension seeds so pairing stays complete.
        if isinstance(spec.a1, WrappedEvidenceSpec):  # pragma: no cover - spec shape guard
            msg = "A3 extension requires a measured A1 cell"
            raise AdmissionError(msg)
        ref_ext_name, ref_ext = _run_cell(ctx, spec.a1, seeds=spec.extension_seeds, suffix="-a3ext")
        blind_ext_name, blind_ext = _run_cell(
            ctx, spec.a3, seeds=spec.extension_seeds, suffix="-ext"
        )
        bundles.extend([ref_ext_name, blind_ext_name])
        stats = _paired_gap_statistics(
            a1_episodes + ref_ext,
            blind_episodes + blind_ext,
            n_resamples=int(ctx.spec.n_resamples),
            root_seed=ctx.spec.root_seed,
        )
        outcome = "PASS" if stats["delta_ci_low"] >= spec.delta_min else "FAIL"
        extended = True
        ctx.extension_used = True
    return CheckReport(
        check="A3",
        outcome=outcome,
        criterion=criterion,
        statistics=stats,
        bundles=bundles,
        evidence=_bundle_evidence(ctx, bundles),
        extended=extended,
    )


def _check_a4(ctx: _RunContext) -> tuple[CheckReport, dict[str, dict[str, float]]]:
    """A4 instrument (ego) robustness across the admitted set (ADR-027 §Admission A4)."""
    spec = ctx.spec
    if spec.a4 is None or spec.c_min_ego is None:  # pragma: no cover - spec validator guard
        msg = "A4 requires both c_min_ego and the a4 instrument cell"
        raise AdmissionError(msg)
    a4_cell = spec.a4
    c_min_ego = spec.c_min_ego
    criterion = (
        f"per admitted partner: success_ci_low >= c_min_ego ({c_min_ego}) with "
        f"successful-episode stress peak <= stress_limit ({spec.stress_limit}); any "
        f"partner with success_ci_high < c_min_ego is brittle -> UNINSTRUMENTABLE "
        f"(ADR-026 §Decision 2: a construct problem, not a partner/axis result)"
    )
    name, episodes = _run_cell(ctx, a4_cell, seeds=spec.seeds, per_member_schedule=True)
    bundles = [name]
    profile = _ego_robustness_profile(ctx, episodes)
    outcome = a4_outcome(_profile_cis(profile), c_min_ego)
    extended = False
    if outcome == "INDETERMINATE" and spec.extension_seeds:
        ext_name, ext_eps = _run_cell(
            ctx, a4_cell, seeds=spec.extension_seeds, suffix="-ext", per_member_schedule=True
        )
        ext_members = set(_episodes_by_member(ext_eps))
        if ext_members != set(profile):
            msg = (
                "A4 extension swept a different admitted member set "
                f"({sorted(ext_members)} vs base {sorted(profile)}); the "
                "pre-committed extension must re-run the same partners"
            )
            raise AdmissionError(msg)
        bundles.append(ext_name)
        episodes = episodes + ext_eps
        profile = _ego_robustness_profile(ctx, episodes)
        outcome = (
            "PASS"
            if all(stats["success_ci_low"] >= c_min_ego for stats in profile.values())
            else "FAIL"
        )  # final — no second extension
        extended = True
        ctx.extension_used = True
    has_stress_channel = any(ep.force_peak is not None for ep in episodes)
    if spec.stress_limit is not None and not has_stress_channel:
        msg = (
            f"A4 stress_limit is committed ({spec.stress_limit}) but the instrument "
            "cell recorded no force_peak stress channel"
        )
        raise AdmissionError(msg)
    if (
        spec.stress_limit is not None
        and outcome == "PASS"
        and any(stats.get("stress_max", 0.0) > spec.stress_limit for stats in profile.values())
    ):
        # The same success+stress bar the task uses, held per partner.
        outcome = "FAIL"
    weakest = min(profile, key=lambda member: (profile[member]["success_ci_low"], member))
    brittle = sorted(m for m, s in profile.items() if s["success_ci_high"] < c_min_ego)
    stats = {
        "n_partners": float(len(profile)),
        "min_success_ci_low": profile[weakest]["success_ci_low"],
        "min_success_ci_high": min(s["success_ci_high"] for s in profile.values()),
    }
    notes = f"weakest admitted partner: {weakest}"
    if brittle:
        notes += f"; brittle partners (success_ci_high < c_min_ego): {', '.join(brittle)}"
    return (
        CheckReport(
            check="A4",
            outcome=outcome,
            criterion=criterion,
            statistics=stats,
            bundles=bundles,
            evidence=_bundle_evidence(ctx, bundles),
            extended=extended,
            notes=notes,
        ),
        profile,
    )


def _episodes_by_member(episodes: list[EpisodeResult]) -> dict[str, list[EpisodeResult]]:
    """Group instrument-cell episodes by admitted-set member (ADR-027 §Admission A4)."""
    # Lazy: leaderboard pulls numpy at module import (keep this module feather-light).
    from chamber.evaluation.leaderboard import _member_of  # noqa: PLC0415

    by_member: dict[str, list[EpisodeResult]] = {}
    for ep in episodes:
        by_member.setdefault(_member_of(ep), []).append(ep)
    if "<unknown>" in by_member:
        msg = (
            "A4 instrument-cell episodes must stamp metadata['member'] (or "
            "'partner') with the admitted-set member name — the leaderboard "
            "per-partner convention (ADR-027 §Reporting rules)"
        )
        raise AdmissionError(msg)
    return by_member


def _ego_robustness_profile(
    ctx: _RunContext, episodes: list[EpisodeResult]
) -> dict[str, dict[str, float]]:
    """The A4 per-partner robustness profile (ADR-027 §Admission A4 mandatory artifact)."""
    by_member = _episodes_by_member(episodes)
    return {
        member: {
            **_recompute_summary_stats(ctx, by_member[member]),
            **stress_statistics(by_member[member]),
        }
        for member in sorted(by_member)
    }


def _profile_cis(profile: dict[str, dict[str, float]]) -> dict[str, tuple[float, float]]:
    return {
        member: (stats["success_ci_low"], stats["success_ci_high"])
        for member, stats in profile.items()
    }


def _finalize(
    ctx: _RunContext,
    *,
    doc_blob: str,
    date_stamp: str,
    checks: list[CheckReport],
    verdict: AdmissionVerdict,
    binding_evidence: dict[str, float],
    notes: str,
    ego_robustness_profile: dict[str, dict[str, float]] | None = None,
) -> AdmissionReport:
    """Write ``admission_report.json`` + ``ADMISSION_REPORT.md`` + sums (I8)."""
    report = AdmissionReport(
        task_id=ctx.spec.task_id,
        task_version=ctx.spec.task_version,
        prereg_git_tag=ctx.spec.git_tag,
        prereg_blob_sha=doc_blob,
        git_sha=ctx.git_sha,
        dirty=ctx.dirty,
        date_stamp=date_stamp,
        checks=checks,
        verdict=verdict,
        seed_extension_used=ctx.extension_used,
        binding_evidence=binding_evidence,
        ego_robustness_profile=ego_robustness_profile,
        notes=notes,
    )
    (ctx.out_dir / ADMISSION_REPORT_JSON).write_text(
        report.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    (ctx.out_dir / ADMISSION_REPORT_MD).write_text(
        render_admission_report_md(report, ctx.spec), encoding="utf-8"
    )
    sums = "".join(
        f"{sha256_file(p)}  {p.name}\n"
        for p in sorted(ctx.out_dir.iterdir())
        if p.is_file() and p.name != "SHA256SUMS.txt"
    )
    (ctx.out_dir / "SHA256SUMS.txt").write_text(sums, encoding="utf-8")
    return report


def load_admission_report(path: Path) -> AdmissionReport:
    """Load a committed ``admission_report.json`` (ADR-027 §Admission protocol).

    Raises ``ValueError`` when the payload carries a different
    ``schema_version`` (the ADR-016 loud-fail convention).
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    version = payload.get("schema_version") if isinstance(payload, dict) else None
    if version != ADMISSION_REPORT_SCHEMA_VERSION:
        msg = (
            f"{path}: admission-report schema_version {version!r} != "
            f"{ADMISSION_REPORT_SCHEMA_VERSION}"
        )
        raise ValueError(msg)
    return AdmissionReport.model_validate(payload)


def render_admission_report_md(report: AdmissionReport, spec: AdmissionSpec) -> str:
    """Render the human-readable admission report (ADR-027 §Admission protocol)."""
    lines = [
        f"# Admission report — {report.task_id}@v{report.task_version}",
        "",
        f"**Verdict.** `{report.verdict}`",
        "",
        f"- Pre-registration: tag `{report.prereg_git_tag}`, blob `{report.prereg_blob_sha}`",
        f"- Launch commit: `{report.git_sha}`"
        + ("  **[DIRTY — not flip evidence]**" if report.dirty else ""),
        f"- Archive date: {report.date_stamp}",
        f"- Seed extension used: {'yes' if report.seed_extension_used else 'no'}",
        "",
        "## Committed thresholds (ADR-007 §Discipline — fixed before any run)",
        "",
        f"- `tau_solv` = {spec.tau_solv} (A1 success floor, CI lower bound)",
        f"- `stress_limit` = {spec.stress_limit} (A1 stress ceiling on successes)",
        f"- `tau_infeasible` = {spec.tau_infeasible} (A2 success ceiling, CI upper bound)",
        f"- `delta_min` = {spec.delta_min} (A3 reference-minus-blind margin, CI lower bound)",
    ]
    if spec.c_min_ego is not None:
        lines.append(
            f"- `c_min_ego` = {spec.c_min_ego} "
            "(A4 ego-robustness floor, per-partner CI lower bound)"
        )
    lines += [
        f"- seeds = {spec.seeds}, episodes/seed = {spec.episodes_per_seed}, "
        f"extension = {spec.extension_seeds}",
        "",
        "## Checks",
        "",
    ]
    for check in report.checks:
        lines.append(f"### {check.check} — {check.outcome}")
        lines.append("")
        lines.append(f"Criterion: {check.criterion}")
        lines.append("")
        lines.extend(f"- `{key}` = {check.statistics[key]:.6g}" for key in sorted(check.statistics))
        if check.bundles:
            lines.append(f"- bundles: {', '.join(check.bundles)} (each `chamber-eval verify`-able)")
        if check.extended:
            lines.append("- pre-committed seed extension consumed")
        if check.notes:
            lines.append(f"- notes: {check.notes}")
        lines.append("")
    if report.binding_evidence:
        lines.append("## A2 binding evidence — stress on matched successes (ADR-026 §Decision 2)")
        lines.append("")
        lines.extend(
            f"- `{key}` = {report.binding_evidence[key]:.6g}"
            for key in sorted(report.binding_evidence)
        )
        lines.append("")
    if report.ego_robustness_profile:
        lines.append("## A4 instrument robustness — per admitted partner (ADR-027 §Admission A4)")
        lines.append("")
        columns = [
            "n_episodes",
            "success_iqm",
            "success_ci_low",
            "success_ci_high",
            "stress_p90",
            "stress_max",
        ]
        lines.append("| partner | " + " | ".join(f"`{col}`" for col in columns) + " |")
        lines.append("|---" * (len(columns) + 1) + "|")
        for member in sorted(report.ego_robustness_profile):
            stats = report.ego_robustness_profile[member]
            cells = [f"{stats[col]:.6g}" if col in stats else "—" for col in columns]
            lines.append(f"| {member} | " + " | ".join(cells) + " |")
        lines.append("")
    if report.notes:
        lines.append(f"_{report.notes}_")
        lines.append("")
    return "\n".join(lines)


__all__ = [
    "A3_BOOTSTRAP_SUBSTREAM",
    "ADMISSION_REPORT_JSON",
    "ADMISSION_REPORT_MD",
    "ADMISSION_REPORT_SCHEMA_VERSION",
    "AdmissionCellSpec",
    "AdmissionCheckId",
    "AdmissionError",
    "AdmissionReport",
    "AdmissionSpec",
    "AdmissionVerdict",
    "CellRun",
    "CheckOutcome",
    "CheckReport",
    "WrappedEvidenceSpec",
    "a1_outcome",
    "a2_outcome",
    "a3_outcome",
    "a4_outcome",
    "admission_spec_from_prereg",
    "load_admission_report",
    "overall_verdict",
    "render_admission_report_md",
    "run_admission",
    "stress_statistics",
]
