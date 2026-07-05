# SPDX-License-Identifier: Apache-2.0
"""Result schema for CHAMBER runs, bundles, and leaderboard entries (ADR-007/008/016/028).

ADR-007 §Discipline requires every spike run to be traceable to a
pre-registration YAML pinned to a git tag whose tree-SHA matches the
YAML's blob-SHA; ADR-008 §Decision requires the HRS bundle (vector +
scalar) to be emitted alongside every leaderboard entry. This module
defines the Pydantic v2 schema that carries those facts from the
spike-run side to the renderers in
:mod:`chamber.evaluation.render`.

The schema is versioned independently of the comm wire format. The
:data:`SCHEMA_VERSION` constant in this module is **not** the same as
:data:`chamber.comm.SCHEMA_VERSION` — the former pins the leaderboard
result-archive format (ADR-008 §Decision; ADR-016 §Decision); the
latter pins the fixed-format packet shape (ADR-003 §Decision).
Bumping either is a breaking change requiring a new ADR.

ADR-016 §Decision bumped this constant 1 → 2 to introduce the typed
``SpikeRun.sub_stage`` field that distinguishes Stage-1a (rig
validation, MPE stand-in, ≥20 pp gate NOT measured) from Stage-1b
(science evaluation, real ManiSkill env, ≥20 pp gate measured). The
prior v1 informal contract (``EpisodeResult.metadata["stage"]`` with
a silent default-by-axis fallback in the summarizer) is retired —
the metadata-dict slot is type-unenforced and the silent default was
the 2026-05-17 incident's root cause.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, NonNegativeFloat, NonNegativeInt

if TYPE_CHECKING:
    import os

#: Schema version for the CHAMBER evaluation result archive
#: (ADR-008 §Decision; ADR-016 §Decision; ADR-028 §Decision).
#:
#: Distinct from :data:`chamber.comm.SCHEMA_VERSION`. Bumping is a
#: breaking change to the serialised result-archive shape and requires
#: a new ADR. The current value is 3 per ADR-028 (introduces the
#: :class:`ResultBundle` provenance wrapper); ADR-016 bumped 1 → 2 to
#: introduce :attr:`SpikeRun.sub_stage`. Per ADR-028 §Decision 4, v2
#: archives are readable forever and are never migrated in place
#: (invariant I8) — readers dispatch on ``schema_version`` (see
#: :func:`load_run_archive`); anything writing results writes v3 only.
#:
#: :class:`SpikeRun`, :class:`LeaderboardEntry`, and
#: :class:`ResultBundle` alias this constant for their
#: ``schema_version`` defaults; bumping it co-bumps all three (the
#: ADR-016 co-bump precedent: the stamp marks the archive era, the
#: wire shape of the untouched models is unchanged).
SCHEMA_VERSION: int = 3

#: ADR-007 §Implementation-staging sub-stage labels (ADR-016 §Decision).
#:
#: Re-exported as a type alias so adapters and tests share a single
#: source of truth for the set of valid sub-stage strings.
#: ``"1a"`` = rig validation; ``"1b"`` = science evaluation; ``"2"``
#: and ``"3"`` are reserved for the Stage-2 / Stage-3 adapters that
#: ADR-007 §Decision pre-registers but have not yet been written
#: (Phase-1 work).
SubStage = Literal["1a", "1b", "2", "3"]


class EpisodeResult(BaseModel):
    """A single evaluation episode of a CHAMBER spike (ADR-007 §Validation criteria).

    The fields cover the three observable quantities the project
    reports per ADR-014 §Decision Table 2: episode success, peak
    constraint-violation magnitude, and fallback-fire count. The
    ``force_peak`` slot is the SA-axis specific signal used in
    ADR-014's Table 1 / Table 2 vendor-compliance rows when ADR-007
    Stage-3 SA decomposes the safety axis.

    Attributes:
        seed: Root seed for this episode (ADR-002 P6 determinism).
        episode_idx: Zero-based episode index within the ``(seed,
            condition)`` cell.
        initial_state_seed: Sub-stream seed for the initial state
            reset, used to pair homogeneous-vs-heterogeneous episodes
            in :func:`chamber.evaluation.bootstrap.pacluster_bootstrap`
            (ADR-007 §Validation criteria; reviewer P1-9).
        success: ``True`` iff the task succeeded per the env's
            terminal reward.
        constraint_violation_peak: Worst-case CBF-constraint gap over
            the episode (ADR-014 §Decision Table 2 "constraint
            violation" column).
        fallback_fired: Number of braking-fallback fires during the
            episode (ADR-014 §Decision Table 2 "fallback fired"
            column; separate from violations per reviewer P0).
        force_peak: Peak contact force in newtons; ``None`` outside
            the SA spike (ADR-007 Stage 3).
        metadata: Free-form per-episode metadata.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    seed: int
    episode_idx: NonNegativeInt
    initial_state_seed: int
    success: bool
    constraint_violation_peak: float = 0.0
    fallback_fired: NonNegativeInt = 0
    force_peak: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConditionPair(BaseModel):
    """Pre-registered homogeneous-vs-heterogeneous baseline pair (ADR-007 §Validation criteria).

    Each Phase-0 spike runs at least one homogeneous baseline (the
    "control" side of the ≥20pp test) and at least one heterogeneous
    condition (the "treatment" side). Identifiers are free-form but
    must match the labels in the pre-registration YAML.

    Attributes:
        homogeneous_id: Identifier of the homogeneous baseline.
        heterogeneous_id: Identifier of the heterogeneous condition.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    homogeneous_id: str
    heterogeneous_id: str


class SpikeRun(BaseModel):
    """Full record of one ADR-007 staged spike (ADR-007 §Discipline; ADR-016 §Decision).

    Carries the pre-registration provenance (``prereg_sha`` /
    ``git_tag``) so the leaderboard renderer can refuse entries that
    do not match the locked YAML. The ``episode_results`` list is the
    raw observation feed for
    :mod:`chamber.evaluation.bootstrap` and
    :mod:`chamber.evaluation.hrs`.

    Attributes:
        spike_id: Free-form identifier (e.g. ``"stage2_cm_urllc"``).
        prereg_sha: Git-blob SHA of the locked pre-registration YAML
            (ADR-007 §Discipline; verified by
            :func:`chamber.evaluation.prereg.verify_git_tag`).
        git_tag: Git tag the pre-registration was locked to
            (ADR-007 §Discipline).
        axis: Name of the ADR-007 heterogeneity axis under test
            (one of ``"CR"``, ``"AS"``, ``"OM"``, ``"CM"``,
            ``"PF"``, ``"SA"``).
        sub_stage: ADR-007 §Implementation-staging sub-stage label
            (one of ``"1a"`` / ``"1b"`` / ``"2"`` / ``"3"``).
            Required, no default — see ADR-016 §Decision and
            §Rationale for why the silent default-by-axis fallback
            the v1 schema relied on was retired.
        condition_pair: Homogeneous-vs-heterogeneous pair.
        seeds: List of root seeds used in this spike (ADR-002 P6).
        episode_results: Flat list of per-episode results across all
            seeds and both pair sides.
        schema_version: Result-archive schema version
            (default :data:`SCHEMA_VERSION`; currently 2 per
            ADR-016 §Decision).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    spike_id: str
    prereg_sha: str
    git_tag: str
    axis: str
    sub_stage: SubStage
    condition_pair: ConditionPair
    seeds: list[int]
    episode_results: list[EpisodeResult]
    schema_version: int = SCHEMA_VERSION


class ConditionResult(BaseModel):
    """Aggregated result for one ADR-007 axis (ADR-008 §Decision).

    Carries the homogeneous-vs-heterogeneous success-rate gap and the
    safety signals that feed the HRS-vector entry for this axis.
    ``gap_pp`` is the ≥20pp gap statistic from ADR-007 §Validation
    criteria; ``ci_low_pp`` / ``ci_high_pp`` are the cluster-bootstrap
    95% bounds from :mod:`chamber.evaluation.bootstrap`.

    Attributes:
        axis: ADR-007 axis name.
        n_episodes: Total episodes contributing to this row.
        homogeneous_success: IQM success rate on the homogeneous side.
        heterogeneous_success: IQM success rate on the heterogeneous
            side.
        gap_pp: ``(homogeneous_success - heterogeneous_success) *
            100`` — the percentage-point gap (ADR-007 §Validation
            criteria).
        ci_low_pp: Lower bound of the 95% cluster-bootstrap CI on
            ``gap_pp``.
        ci_high_pp: Upper bound of the 95% cluster-bootstrap CI on
            ``gap_pp``.
        violation_rate: Per-step CBF-constraint violation rate across
            episodes (ADR-014 §Decision Table 2).
        fallback_rate: Fraction of episodes in which the braking
            fallback fired at least once (separate from
            ``violation_rate`` per reviewer P0).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    axis: str
    n_episodes: NonNegativeInt
    homogeneous_success: NonNegativeFloat
    heterogeneous_success: NonNegativeFloat
    gap_pp: float
    ci_low_pp: float
    ci_high_pp: float
    violation_rate: NonNegativeFloat
    fallback_rate: NonNegativeFloat


class HRSVectorEntry(BaseModel):
    """One axis row of the HRS vector (ADR-008 §Decision).

    ADR-008 binds the HRS bundle to the surviving ADR-007 axes and
    fixes the default ordering CM > PF > CR > SA > OM > AS. Each
    entry carries the per-axis score plus the gap-PP that feeds
    :func:`chamber.evaluation.hrs.compute_hrs_scalar`.

    Attributes:
        axis: ADR-007 axis name.
        score: Normalised per-axis score in ``[0, 1]`` (success rate
            on the heterogeneous condition, by default).
        gap_pp: Percentage-point gap from the matching
            :class:`ConditionResult` (kept for traceability).
        weight: Weight assigned to this axis in the scalar
            aggregation (defaults to ADR-008 Option D ordering).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    axis: str
    score: NonNegativeFloat
    gap_pp: float
    weight: NonNegativeFloat


class HRSVector(BaseModel):
    """Per-axis HRS vector emitted alongside the scalar (ADR-008 §Decision).

    Reviewer P1-8 requires the vector to be emitted *unconditionally*
    so consumers can recompute the scalar under a different
    weighting; the leaderboard renderer enforces this by refusing
    entries that carry only the scalar.

    Attributes:
        entries: Per-axis entries in the ADR-008 default order
            (CM > PF > CR > SA > OM > AS), subject to axis survival.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    entries: list[HRSVectorEntry]


class LeaderboardEntry(BaseModel):
    """A single method/result pair as it appears on the CHAMBER leaderboard (ADR-008 §Decision).

    Ranks by ``hrs_scalar``. The accompanying :class:`HRSVector` is
    emitted unconditionally per reviewer P1-8; the renderer in
    :mod:`chamber.evaluation.render` displays the vector as a sub-row
    below the scalar.

    Attributes:
        method_id: Identifier of the method being evaluated (e.g.
            ``"concerto-ego-aht-v0.1"``).
        spike_runs: Spike-run IDs that contributed to this entry.
        hrs_vector: Per-axis HRS values (always emitted).
        hrs_scalar: Aggregated HRS scalar (ADR-008 §Decision; the
            ranking metric).
        violation_rate: Aggregate per-step violation rate across the
            contributing spike runs.
        fallback_rate: Aggregate fraction of episodes triggering the
            braking fallback (ADR-014 §Decision).
        schema_version: Result-archive schema version
            (default :data:`SCHEMA_VERSION`).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    method_id: str
    spike_runs: list[str]
    hrs_vector: HRSVector
    hrs_scalar: NonNegativeFloat
    violation_rate: NonNegativeFloat
    fallback_rate: NonNegativeFloat
    schema_version: int = SCHEMA_VERSION


class SeedSchedule(BaseModel):
    """Explicit seed schedule of a result bundle (ADR-028 §Decision 1).

    ADR-028 requires "the explicit seed list/derivation, not just a
    root seed": ``root_seed`` is the run-level pin, ``seeds`` the
    exact per-episode-cluster seed list, and ``substream_labels`` the
    :func:`concerto.training.seeding.derive_substream` name patterns
    the run used (ADR-002 P6 — the schedule is re-derivable from these
    three facts alone).

    Attributes:
        root_seed: Run-level root seed (ADR-002 P6).
        seeds: Exact per-cluster seed list, in run order.
        episodes_per_seed: Episodes run per seed cluster.
        substream_labels: ``derive_substream`` name patterns used
            (``{seed}`` / ``{episode}`` placeholders literal).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    root_seed: int
    seeds: list[int]
    episodes_per_seed: NonNegativeInt
    substream_labels: list[str]


class PlatformFingerprint(BaseModel):
    """Platform fingerprint of a result bundle (ADR-028 §Decision 1).

    ADR-028: "OS, Python, key dependency versions, device". CPU-only
    verification (ADR-028 §Decision 3) never *compares* this block —
    it is provenance for the reader, not a reproducibility gate.

    Attributes:
        os: ``platform.platform()`` string.
        python: ``platform.python_version()`` string.
        numpy: Installed numpy version.
        torch: Installed torch version, or ``None`` if absent.
        device: Compute device the run used (GPU name or ``"cpu"``).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    os: str
    python: str
    numpy: str
    torch: str | None
    device: str


class BundleSummary(BaseModel):
    """Recomputable summary statistics of a result bundle (ADR-028 §Decision 3).

    ``chamber-eval verify`` recomputes these from the bundle's raw
    per-seed episode records and compares within tolerance — so every
    field here must be a pure function of the episode records plus the
    pinned bootstrap parameters (``n_resamples`` +
    ``bootstrap_root_seed`` routed through ``derive_substream``, so
    the recomputation is byte-reproducible per ADR-002 P6).

    Attributes:
        n_episodes: Total episode count across all seeds.
        success_mean: Plain mean success over all episodes.
        success_iqm: Interquartile-mean success from the seed-cluster
            bootstrap (ADR-008 reporting discipline).
        success_ci_low: 2.5th percentile of the cluster bootstrap.
        success_ci_high: 97.5th percentile of the cluster bootstrap.
        n_resamples: Bootstrap resample count used.
        bootstrap_root_seed: Root seed of the bootstrap substream.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    n_episodes: NonNegativeInt
    success_mean: float
    success_iqm: float
    success_ci_low: float
    success_ci_high: float
    n_resamples: NonNegativeInt
    bootstrap_root_seed: int


class ResultBundle(BaseModel):
    """The v3 result-bundle record — ``bundle.json`` (ADR-028 §Decision 1).

    Wraps the existing :class:`EpisodeResult` machinery: the raw
    episode records live in the bundle directory's per-seed JSONL
    files (one :class:`EpisodeResult` JSON object per line), listed in
    ``manifest``; ``bundle.json`` carries the provenance fields ADR-028
    authorizes plus the recomputable :class:`BundleSummary`.

    Attributes:
        schema_version: Result-archive schema version (v3;
            :data:`SCHEMA_VERSION`).
        task_id: ADR-027 task identity (with ``task_version``:
            ``task_id@vN``).
        task_version: ADR-027 task version.
        policy_id: Ego policy reference the bundle evaluates (ADR-011
            baseline IDs; e.g. ``"random"`` = B-RND).
        partner_set_id: Partner-set identity (ADR-009 as amended).
            Ad-hoc single-partner runs use ``"adhoc:<registry-name>"``
            until partner sets ship.
        partner_hashes: Per-member partner identity hashes —
            ``PartnerSpec.partner_id`` (SHA-256 custody hash, ADR-018)
            keyed by member name.
        git_sha: Launch commit, captured once per run (ADR-002
            §Rev 2026-06-12 process-cache discipline).
        dirty: ``True`` iff the bundle was produced from a dirty
            working tree under ``--allow-dirty``. Dirty bundles are
            ineligible for leaderboard use (ADR-028 §Rev 2026-07-05).
        package_version: Installed ``concerto-multirobot``
            distribution version.
        seed_schedule: Explicit seed schedule (ADR-028 §Decision 1).
        repro_command: Exact reproduction invocation (the
            ``repro_command.txt`` convention promoted into the
            bundle).
        platform: Platform fingerprint (ADR-028 §Decision 1).
        manifest: SHA-256 hex digest of every file in the bundle
            directory except ``bundle.json`` itself (which cannot
            contain its own hash), keyed by file name.
        summary: Recomputable summary statistics (ADR-028 §Decision 3).
        prereg_git_tag: Pre-registration tag the run was gated on, or
            ``None`` for unpreregistered (diagnostic) runs.
        prereg_blob_sha: Verified prereg blob SHA at the tag, or
            ``None``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = SCHEMA_VERSION
    task_id: str
    task_version: int
    policy_id: str
    partner_set_id: str
    partner_hashes: dict[str, str]
    git_sha: str
    dirty: bool
    package_version: str
    seed_schedule: SeedSchedule
    repro_command: str
    platform: PlatformFingerprint
    manifest: dict[str, str]
    summary: BundleSummary
    prereg_git_tag: str | None = None
    prereg_blob_sha: str | None = None


#: First schema version carried by :class:`ResultBundle` archives
#: (ADR-028 §Decision 1) — the :func:`load_run_archive` dispatch pivot.
_RESULT_BUNDLE_MIN_SCHEMA_VERSION: int = 3


def load_run_archive(path: str | os.PathLike[str]) -> SpikeRun | ResultBundle:
    """Load a result archive, dispatching on ``schema_version`` (ADR-028 §Decision 4).

    v3 archives (``bundle.json``) validate as :class:`ResultBundle`;
    v1/v2 archives validate as :class:`SpikeRun` — the read-only
    compatibility path: v2 archives are readable forever, never
    migrated in place (invariant I8), and treated as historical inputs
    with explicitly-absent provenance, never as errors.

    Args:
        path: Path to the archive JSON file (``os.PathLike``/``str``).

    Returns:
        A :class:`ResultBundle` (v3) or :class:`SpikeRun` (v1/v2).

    Raises:
        ValueError: If the file carries no integer ``schema_version``.
        pydantic.ValidationError: If the payload does not match the
            model its version claims (the ADR-016 loud-fail mode).
    """
    raw = Path(path).read_text(encoding="utf-8")
    payload = json.loads(raw)
    version = payload.get("schema_version") if isinstance(payload, dict) else None
    if not isinstance(version, int):
        msg = f"{path}: no integer schema_version field; not a CHAMBER result archive"
        raise ValueError(msg)
    if version >= _RESULT_BUNDLE_MIN_SCHEMA_VERSION:
        return ResultBundle.model_validate(payload)
    return SpikeRun.model_validate(payload)


__all__ = [
    "SCHEMA_VERSION",
    "BundleSummary",
    "ConditionPair",
    "ConditionResult",
    "EpisodeResult",
    "HRSVector",
    "HRSVectorEntry",
    "LeaderboardEntry",
    "PlatformFingerprint",
    "ResultBundle",
    "SeedSchedule",
    "SpikeRun",
    "SubStage",
    "load_run_archive",
]
