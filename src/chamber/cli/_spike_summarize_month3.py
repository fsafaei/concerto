# SPDX-License-Identifier: Apache-2.0
"""``chamber-spike summarize-month3`` subcommand (plan/07 §6 #7; §T5b.10).

Reads the six per-axis SpikeRun JSON archives that landed under the
results directory (default ``spikes/results/``), bootstraps the
paired-cluster gap CI per axis via
:func:`chamber.evaluation.bootstrap.pacluster_bootstrap` (mirroring
:mod:`chamber.cli._spike_next_stage`'s convention — see that module's
docstring for why we read SpikeRun JSONs and not
``leaderboard.json``: :class:`LeaderboardEntry` does not carry CI
bounds), applies the ADR-007 §Implementation staging + ADR-008
§Decision rules, and emits a Markdown "Month-3 lock-priority" report
the maintainer presents to the senior-advisor review.

Four top-line recommendations are possible (parse-stable, exactly one
line in the report):

- ``**Recommendation: Accept-Validated**`` — every staged gate
  cleared and ADR-008's default headline bundle (CM x PF x CR) is
  composable. Locks ADR-007 / ADR-008 / ADR-011.
- ``**Recommendation: Accept-Partial-Defer**`` — Stage 1b cleared
  but the default headline bundle is not composable (one or more of
  CM / PF / CR failed or is missing). ADR-008 falls back per
  §Decision (Option A if CR failed; Option B if PF failed) or holds
  the bundle lock. Surfaces the ADR-007 §Stage 2 gate trigger if
  applicable.
- ``**Recommendation: Stop — ADR re-review**`` — Stage 1b was
  measured but neither AS nor OM cleared. Surfaces the ADR-007
  §Stage 1 gate trigger; recommends an ADR-007 §Decision re-review.
- ``**Recommendation: Defer — Stage 1b not yet measured (Phase-1
  milestone; guardrail ≤4 weeks)**`` — Stage 1 was run on the
  Phase-0 MPE stand-in (Stage 1a) but the Phase-1 real-env Stage 1b
  has not yet produced evidence. ADR-conformant per §Stage 1b's
  trigger guardrail (PR #121); distinct from Stop.

The decision rule is in :func:`decide_recommendation`; the renderer
is :func:`render_report`. Both are decoupled from the SpikeRun loader
so the test suite (``tests/unit/test_summarize_month3.py``) can pin
the report contract without re-running the bootstrap.

Stage metadata: the per-axis stage label (``"1a"`` / ``"1b"`` / ``"2"``
/ ``"3"``) is read from the SpikeRun's first
:attr:`EpisodeResult.metadata["stage"]` value; absent → default to
the axis's canonical stage per ADR-007 §Implementation staging
(AS/OM → 1b, CR/CM → 2, PF/SA → 3). The optional ``"stage"`` key is
a load-bearing affordance for Stage 1a runs (rig validation only;
no ≥20 pp measurement) so they render as ``gate_pass = n/a`` in the
report and do not trigger Stop / Accept-Validated transitions.
"""

from __future__ import annotations

import datetime as _dt
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import chamber
from chamber.evaluation.bootstrap import build_paired_episodes, pacluster_bootstrap
from chamber.evaluation.results import SpikeRun
from concerto.training.seeding import derive_substream

if TYPE_CHECKING:
    import argparse


#: ADR-007 §Validation criteria default gate threshold (percentage
#: points on the paired-cluster bootstrap CI lower bound of the
#: ``homo - hetero`` gap). Matches :mod:`chamber.cli._spike_next_stage`.
_DEFAULT_GATE_PP: float = 20.0

#: Default bootstrap resamples + RNG seed (ADR-002 P6). Matches
#: :mod:`chamber.cli._spike_next_stage`'s tuning.
_DEFAULT_N_RESAMPLES: int = 2000
_DEFAULT_SEED: int = 0

#: Canonical per-axis Phase-0 stage assignments (ADR-007
#: §Implementation staging). Used as the fallback when an
#: :class:`EpisodeResult` does not carry a ``"stage"`` metadata key.
_DEFAULT_STAGE_BY_AXIS: dict[str, str] = {
    "AS": "1b",
    "OM": "1b",
    "CR": "2",
    "CM": "2",
    "PF": "3",
    "SA": "3",
}

#: ADR-007 staging order; the per-axis evidence table renders in this
#: order so the report is parse-stable.
_AXIS_ORDER: tuple[str, ...] = ("AS", "OM", "CR", "CM", "PF", "SA")

#: Stage-1a axes don't produce a ≥20 pp measurement (rig validation
#: only); the gate column renders as ``n/a`` and the decision rule
#: ignores them. Defined explicitly so a future Stage refactor that
#: introduces e.g. ``"2a"`` doesn't accidentally turn 1a into a
#: gate-eligible stage.
_GATELESS_STAGES: frozenset[str] = frozenset({"1a"})

#: ADR-008 §Decision default headline HRS bundle. The fallbacks
#: (Option A / Option B) are named in the same §Decision and surfaced
#: by :func:`_adr_008_action` when the default is not composable.
_ADR_008_HEADLINE: tuple[str, str, str] = ("CM", "PF", "CR")


RECOMMENDATION_ACCEPT_VALIDATED: str = "Accept-Validated"
RECOMMENDATION_ACCEPT_PARTIAL_DEFER: str = "Accept-Partial-Defer"
RECOMMENDATION_STOP: str = "Stop"
RECOMMENDATION_DEFER: str = "Defer"


@dataclass(frozen=True)
class _AxisResult:
    """Per-axis bootstrap output + stage metadata (ADR-007 §Validation criteria).

    The fields are everything the renderer needs to fill one row of
    the per-axis evidence table and to drive
    :func:`decide_recommendation`. Built from a :class:`SpikeRun` via
    :func:`_bootstrap_axis_result`.
    """

    axis: str
    stage: str
    n_seeds: int
    n_episodes: int
    gap_iqm_pp: float
    ci_low_pp: float
    ci_high_pp: float
    spike_run_path: Path | None = None
    prereg_sha: str = ""
    git_tag: str = ""
    gate_pp: float = _DEFAULT_GATE_PP

    @property
    def gate_pass(self) -> bool:
        """``True`` iff this axis cleared the ≥``gate_pp`` gate AND is not Stage 1a."""
        if self.stage in _GATELESS_STAGES:
            return False
        return self.ci_low_pp >= self.gate_pp

    @property
    def gate_cell(self) -> str:
        """Rendered cell for the ``gate_pass`` column (``PASS`` / ``fail`` / ``n/a``)."""
        if self.stage in _GATELESS_STAGES:
            return "n/a"
        return "PASS" if self.gate_pass else "fail"


def decide_recommendation(results: list[_AxisResult]) -> str:
    """Apply ADR-007 + ADR-008 rules → one of four top-line states (plan/07 §6 #7).

    Decision rule (ADR-007 §Implementation staging; ADR-008 §Decision):

    1. **Defer** when Stage 1 was not measured at Stage 1b yet —
       either no Stage-1 axis is present in ``results`` or every
       Stage-1 row is ``stage == "1a"``.
    2. **Stop** when Stage 1b *was* measured (at least one Stage-1 row
       has ``stage != "1a"``) but neither AS nor OM cleared the gate.
    3. **Accept-Validated** when Stage 1b cleared *and* the default
       headline bundle ``{CM, PF, CR}`` is a subset of the passing
       set.
    4. **Accept-Partial-Defer** otherwise — Stage 1b cleared but the
       default bundle is not composable; ADR-008 falls back per
       §Decision (Option A / Option B) or holds the bundle lock.

    Args:
        results: Per-axis bootstrap outcomes; order does not matter.

    Returns:
        One of :data:`RECOMMENDATION_ACCEPT_VALIDATED`,
        :data:`RECOMMENDATION_ACCEPT_PARTIAL_DEFER`,
        :data:`RECOMMENDATION_STOP`, :data:`RECOMMENDATION_DEFER`.
    """
    by_axis = {r.axis: r for r in results}
    stage1_axes = ("AS", "OM")
    stage1_rows = [by_axis[a] for a in stage1_axes if a in by_axis]
    stage1_measured = any(r.stage not in _GATELESS_STAGES for r in stage1_rows)
    if not stage1_measured:
        return RECOMMENDATION_DEFER
    stage1_passed = any(r.gate_pass for r in stage1_rows)
    if not stage1_passed:
        return RECOMMENDATION_STOP
    passed_axes = {r.axis for r in results if r.gate_pass}
    if set(_ADR_008_HEADLINE).issubset(passed_axes):
        return RECOMMENDATION_ACCEPT_VALIDATED
    return RECOMMENDATION_ACCEPT_PARTIAL_DEFER


@dataclass(frozen=True)
class _ProvenanceFooter:
    """Provenance footer fields (Section 8 of the report)."""

    chamber_version: str
    git_sha: str
    timestamp_utc: str
    gate_pp: float
    n_resamples: int
    seed: int
    axis_archives: list[tuple[str, str, str, str]] = field(default_factory=list)
    # Per-axis (axis, spike_run_path, prereg_sha, git_tag) tuples.


def render_report(
    results: list[_AxisResult],
    *,
    provenance: _ProvenanceFooter | None = None,
) -> str:
    """Render the Month-3 lock-priority Markdown report (plan/07 §6 #7).

    Section list (matches the Plan-subagent design pass):

    1. ``# CONCERTO Month-3 lock-priority report`` (H1).
    2. ``## TL;DR — Recommendation`` — one of the four parse-stable
       top-line strings, plus the surviving-axis set and the ADR-008
       bundle composition the recommendation implies.
    3. ``## Per-axis evidence (≥20 pp gap test)`` — table with one
       row per axis in ADR-007 staging order
       (AS → OM → CR → CM → PF → SA).
    4. ``## Per-stage gate enumeration`` — H3 subsections per stage
       with verdict line + axis bullet list + next-stage launch
       verdict.
    5. ``## ADR-by-ADR action`` — table with rows for ADR-007,
       ADR-008, ADR-011.
    6. ``## ADR-014 safety-table integration`` — linked-by-path SA
       three-table summary (or deferred stub).
    7. ``## Open-work flags (ADR-INDEX)`` — verbatim restatements of
       footnotes (b), (c), (f).
    8. ``## Provenance`` — chamber version, git SHA, per-axis
       archives, bootstrap params, UTC timestamp.

    Args:
        results: Per-axis bootstrap outcomes. May be incomplete
            (missing axes render as ``Missing`` rows).
        provenance: Optional provenance footer. When ``None``, fields
            are filled with conservative defaults
            (``"unknown"`` for SHA, current UTC time, the constants
            in this module for gate / bootstrap params).

    Returns:
        UTF-8 Markdown text ending with a trailing newline.
    """
    by_axis = {r.axis: r for r in results}
    recommendation = decide_recommendation(results)
    surviving = {r.axis for r in results if r.gate_pass}
    lines: list[str] = []
    lines.append("# CONCERTO Month-3 lock-priority report")
    lines.append("")
    lines.extend(_render_tldr_section(recommendation, surviving))
    lines.extend(_render_per_axis_table(by_axis))
    lines.extend(_render_per_stage_gate_section(by_axis))
    lines.extend(_render_adr_action_table(by_axis, surviving, recommendation))
    lines.extend(_render_safety_section(by_axis))
    lines.extend(_render_open_work_section(recommendation))
    lines.extend(_render_provenance_section(provenance))
    return "\n".join(lines) + "\n"


def _render_tldr_section(recommendation: str, surviving: set[str]) -> list[str]:
    """Section 2: top-line recommendation + bundle composition (parse-stable)."""
    if recommendation == RECOMMENDATION_ACCEPT_VALIDATED:
        banner = "**Recommendation: Accept-Validated**"
    elif recommendation == RECOMMENDATION_ACCEPT_PARTIAL_DEFER:
        banner = "**Recommendation: Accept-Partial-Defer**"
    elif recommendation == RECOMMENDATION_STOP:
        banner = "**Recommendation: Stop — ADR re-review**"
    else:  # Defer
        banner = (
            "**Recommendation: Defer — Stage 1b not yet measured "
            "(Phase-1 milestone; guardrail ≤4 weeks)**"
        )
    surviving_str = ", ".join(a for a in _AXIS_ORDER if a in surviving) or "(none yet)"
    bundle = _bundle_composition_phrase(surviving)
    return [
        "## TL;DR — Recommendation",
        "",
        banner,
        "",
        f"Surviving axes (ADR-007 staging order): **{surviving_str}**.",
        f"ADR-008 HRS bundle: {bundle}.",
        "",
    ]


def _bundle_composition_phrase(surviving: set[str]) -> str:
    """Phrase for the ADR-008 bundle composition row (ADR-008 §Decision)."""
    if set(_ADR_008_HEADLINE).issubset(surviving):
        return "**CM x PF x CR** (default)"
    if "CM" in surviving and "PF" in surviving and "CR" not in surviving:
        return "**Option A — latency x drop x partner-familiarity** (CR failed)"
    if "CM" in surviving and "CR" in surviving and "PF" not in surviving:
        return "**Option B — latency x drop x degraded-partner** (PF failed)"
    return "**hold** — surviving headline-axis set < 3 (re-check ADR-008 §Decision)"


def _render_per_axis_table(by_axis: dict[str, _AxisResult]) -> list[str]:
    """Section 3: per-axis evidence table (parse-stable column order)."""
    lines: list[str] = [
        "## Per-axis evidence (≥20 pp gap test)",
        "",
        (
            "| Axis | Stage | Status | n_seeds | n_episodes | gap_iqm_pp | "
            "ci_low_pp | ci_high_pp | gate_pass |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for axis in _AXIS_ORDER:
        result = by_axis.get(axis)
        if result is None:
            lines.append(
                f"| {axis} | {_DEFAULT_STAGE_BY_AXIS[axis]} | Missing | "
                "n/a | n/a | n/a | n/a | n/a | n/a |"
            )
            continue
        status = _stage_status_label(result.stage)
        gap_iqm = f"{result.gap_iqm_pp:.2f}" if result.stage not in _GATELESS_STAGES else "n/a"
        ci_low = f"{result.ci_low_pp:.2f}" if result.stage not in _GATELESS_STAGES else "n/a"
        ci_high = f"{result.ci_high_pp:.2f}" if result.stage not in _GATELESS_STAGES else "n/a"
        lines.append(
            f"| {axis} | {result.stage} | {status} | "
            f"{result.n_seeds} | {result.n_episodes} | "
            f"{gap_iqm} | {ci_low} | {ci_high} | {result.gate_cell} |"
        )
    lines.append("")
    return lines


def _stage_status_label(stage: str) -> str:
    """Stage-name → human-readable status cell (Section 3)."""
    return {
        "1a": "Stage 1a (rig-only)",
        "1b": "Stage 1b (measured)",
        "2": "Stage 2 (measured)",
        "3": "Stage 3 (measured)",
    }.get(stage, f"Stage {stage} (measured)")


def _render_per_stage_gate_section(by_axis: dict[str, _AxisResult]) -> list[str]:
    """Section 4: per-stage gate enumeration with launch verdicts."""
    lines = ["## Per-stage gate enumeration", ""]
    lines.extend(
        _render_stage_subsection(by_axis, stage_label="1b", axes=("AS", "OM"), next_label="Stage 2")
    )
    lines.extend(
        _render_stage_subsection(by_axis, stage_label="2", axes=("CR", "CM"), next_label="Stage 3")
    )
    lines.extend(
        _render_stage_subsection(
            by_axis, stage_label="3", axes=("PF", "SA"), next_label="Month-3 lock"
        )
    )
    return lines


def _render_stage_subsection(
    by_axis: dict[str, _AxisResult],
    *,
    stage_label: str,
    axes: tuple[str, ...],
    next_label: str,
) -> list[str]:
    """H3 subsection for one stage (Section 4)."""
    rows = [by_axis.get(a) for a in axes]
    measured = [r for r in rows if r is not None and r.stage not in _GATELESS_STAGES]
    passing = [r for r in measured if r.gate_pass]
    # Inline verdicts are plain-text (no Markdown bold) so a maintainer
    # can grep the report for ``Stage 2 launch: PERMITTED`` without
    # having to know whether the token was emphasised.
    stage_pp = (
        f"Stage {stage_label} gate: PASS — {len(passing)}/{len(axes)} "
        "axes cleared (ci_low_pp ≥ 20.0)."
        if passing
        else f"Stage {stage_label} gate: fail — 0/{len(axes)} axes cleared."
    )
    if not measured:
        stage_pp = f"Stage {stage_label} gate: deferred — no Stage-{stage_label} SpikeRun present."
    lines = [f"### Stage {stage_label}", "", stage_pp, ""]
    for axis in axes:
        result = by_axis.get(axis)
        if result is None:
            lines.append(f"- {axis} — Missing")
            continue
        if result.stage in _GATELESS_STAGES:
            lines.append(f"- {axis} — Stage 1a (rig validation only; no ≥20 pp measurement)")
            continue
        marker = "PASS" if result.gate_pass else "fail"
        lines.append(
            f"- {axis} — ci_low_pp={result.ci_low_pp:.2f}  "
            f"ci_high_pp={result.ci_high_pp:.2f}  ({marker})"
        )
    lines.append("")
    # Next-stage launch verdict.
    if not measured:
        if stage_label == "1b":
            verdict = (
                f"{next_label} launch: BLOCKED — Stage 1b not measured "
                "(ADR-007 §Stage 1b guardrail: launch ≤4 weeks after Month-3 lock)."
            )
        else:
            verdict = f"{next_label} launch: deferred — Stage {stage_label} not measured."
    elif passing:
        verdict = f"{next_label} launch: PERMITTED — gate cleared (ADR-007 §Stage {stage_label})."
    else:
        verdict = (
            f"{next_label} launch: BLOCKED — Stage {stage_label} gate failed "
            f"(ADR-007 §Stage {stage_label} gate trigger fired; methodology / rig "
            "review required)."
        )
    lines.append(verdict)
    lines.append("")
    return lines


def _render_adr_action_table(
    by_axis: dict[str, _AxisResult],
    surviving: set[str],
    recommendation: str,
) -> list[str]:
    """Section 5: ADR-by-ADR action table (ADR-007 / ADR-008 / ADR-011)."""
    return [
        "## ADR-by-ADR action",
        "",
        "| ADR | Action | Rationale |",
        "| --- | --- | --- |",
        f"| ADR-007 | {_adr_007_action(by_axis, surviving, recommendation)} | "
        "Implementation staging gates; §Validation criteria threshold (≥20 pp). |",
        f"| ADR-008 | {_adr_008_action(surviving, recommendation)} | "
        "§Decision default headline bundle CM x PF x CR; Option A / Option B fallbacks. |",
        f"| ADR-011 | {_adr_011_action(recommendation)} | "
        "Baseline-set lock contingent on per-axis evidence for the surviving axes. |",
        "",
    ]


def _adr_007_action(
    by_axis: dict[str, _AxisResult],
    surviving: set[str],
    recommendation: str,
) -> str:
    """ADR-007 action cell (Section 5)."""
    if recommendation == RECOMMENDATION_STOP:
        return "Re-review §Decision (ADR-007 §Stage 1 gate trigger)"
    if recommendation == RECOMMENDATION_DEFER:
        return "Hold at Accepted; Stage 1b launch is Phase-1 milestone (§Stage 1b guardrail)"
    promoted = ", ".join(a for a in _AXIS_ORDER if a in surviving) or "(none)"
    held = ", ".join(a for a in _AXIS_ORDER if a in by_axis and a not in surviving) or "(none)"
    if recommendation == RECOMMENDATION_ACCEPT_VALIDATED:
        return f"Promote to Validated for: {promoted}"
    # Accept-Partial-Defer.
    return f"Promote to Validated for: {promoted}; hold at Accepted for: {held}"


def _adr_008_action(surviving: set[str], recommendation: str) -> str:
    """ADR-008 action cell (Section 5)."""
    if recommendation in {RECOMMENDATION_DEFER, RECOMMENDATION_STOP}:
        return "Hold bundle lock — insufficient evidence"
    if set(_ADR_008_HEADLINE).issubset(surviving):
        return "Lock CM x PF x CR (default)"
    if "CM" in surviving and "PF" in surviving and "CR" not in surviving:
        return "Fall back to Option A: latency x drop x partner-familiarity (CR failed)"
    if "CM" in surviving and "CR" in surviving and "PF" not in surviving:
        return "Fall back to Option B: latency x drop x degraded-partner (PF failed)"
    return "Block bundle lock — surviving headline-axis set < 3 (re-check §Decision)"


def _adr_011_action(recommendation: str) -> str:
    """ADR-011 action cell (Section 5)."""
    if recommendation == RECOMMENDATION_ACCEPT_VALIDATED:
        return "Lock baseline set per §Decision"
    return "Hold baseline-set lock — §Validation coverage check not satisfied"


def _render_safety_section(by_axis: dict[str, _AxisResult]) -> list[str]:
    """Section 6: ADR-014 safety-table integration (linked by path)."""
    lines = ["## ADR-014 safety-table integration", ""]
    sa = by_axis.get("SA")
    if sa is None or sa.spike_run_path is None or sa.stage in _GATELESS_STAGES:
        lines.append(
            "SA spike not present — three-table safety report deferred to Stage 3 "
            "launch (ADR-014 §Decision unchanged)."
        )
        lines.append("")
        return lines
    marker = "PASS" if sa.gate_pass else "fail"
    summary = (
        f"SA spike present (ci_low_pp={sa.ci_low_pp:.2f}, {marker}). Three-table "
        f"safety report rendered separately; archive at `{sa.spike_run_path}`."
    )
    lines.append(summary)
    lines.append("")
    return lines


def _render_open_work_section(recommendation: str) -> list[str]:
    """Section 7: ADR-INDEX open-work flags (b), (c), (f)."""
    lines = ["## Open-work flags (ADR-INDEX)", ""]
    flag_b = "- **(b) ADR-007 §Stage 1a / §Stage 1b split.** " + (
        "Stage 1b cleared; flag resolves once §Validation criteria is met for Stage 2 and Stage 3."
        if recommendation in {RECOMMENDATION_ACCEPT_VALIDATED, RECOMMENDATION_ACCEPT_PARTIAL_DEFER}
        else "Stage 1b pending; flag remains active until Phase-1 launch "
        "(ADR-007 §Stage 1b guardrail: ≤4 weeks after Month-3 lock review)."
    )
    flag_c = "- **(c) ADR-008 HRS bundle composition.** " + (
        "Bundle composition lockable per §Decision (see ADR-by-ADR action above)."
        if recommendation == RECOMMENDATION_ACCEPT_VALIDATED
        else "Bundle composition pending the surviving-axis set (§Decision footnote c)."
    )
    flag_f = (
        "- **(f) ADR-014 safety reporting.** Reporting contract qualified by "
        "ADR-008 bundle composition (flag c above) and the PR-A2 conformal-loss "
        "instrumentation; per-axis row counts firm up once Stage 3 SA lands."
    )
    lines.extend([flag_b, flag_c, flag_f, ""])
    return lines


def _render_provenance_section(provenance: _ProvenanceFooter | None) -> list[str]:
    """Section 8: provenance footer (chamber version, git SHA, archives, timestamp)."""
    if provenance is None:
        provenance = _ProvenanceFooter(
            chamber_version=chamber.__version__,
            git_sha="unknown",
            timestamp_utc=_dt.datetime.now(tz=_dt.UTC).isoformat(),
            gate_pp=_DEFAULT_GATE_PP,
            n_resamples=_DEFAULT_N_RESAMPLES,
            seed=_DEFAULT_SEED,
        )
    lines = ["## Provenance", ""]
    lines.append(f"- chamber version: `{provenance.chamber_version}`")
    lines.append(f"- git SHA: `{provenance.git_sha}`")
    lines.append(f"- timestamp (UTC): `{provenance.timestamp_utc}`")
    lines.append(f"- gate threshold (pp): `{provenance.gate_pp:.2f}`")
    lines.append(f"- bootstrap n_resamples: `{provenance.n_resamples}`")
    lines.append(f"- bootstrap root seed: `{provenance.seed}`")
    if provenance.axis_archives:
        lines.append("- per-axis archives:")
        for axis, path, sha, tag in provenance.axis_archives:
            lines.append(f"  - `{axis}` → `{path}` (prereg_sha=`{sha}`, git_tag=`{tag}`)")
    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


def add_parser(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register the ``summarize-month3`` subparser (T5b.10; plan/07 §6 #7)."""
    parser = sub.add_parser(
        "summarize-month3",
        help=(
            "Produce the Month-3 lock-priority Markdown report from per-axis "
            "SpikeRun archives (plan/07 §T5b.10)."
        ),
        description=(
            "Walks --results-dir for SpikeRun JSON archives (one per ADR-007 axis), "
            "bootstraps the paired-cluster ≥20 pp gate per axis, applies ADR-007 "
            "+ ADR-008 + ADR-011 rules, and emits the Markdown lock-priority "
            "report. Writes to --output when given, else to stdout."
        ),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("spikes/results"),
        help=(
            "Root of the per-axis SpikeRun archive tree. The subcommand "
            "recursively globs ``*.json`` and loads each as a SpikeRun "
            "(invalid files are skipped with a stderr note)."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to write the Markdown report. Default: stdout.",
    )
    parser.add_argument(
        "--gate-pp",
        type=float,
        default=_DEFAULT_GATE_PP,
        help=f"Gate threshold in percentage points (default {_DEFAULT_GATE_PP}).",
    )
    parser.add_argument(
        "--n-resamples",
        type=int,
        default=_DEFAULT_N_RESAMPLES,
        help=f"Bootstrap resamples per axis (default {_DEFAULT_N_RESAMPLES}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=_DEFAULT_SEED,
        help=f"Bootstrap root seed (ADR-002 P6; default {_DEFAULT_SEED}).",
    )


def run(args: argparse.Namespace) -> int:
    """Implementation of ``chamber-spike summarize-month3`` (T5b.10; plan/07 §6 #7)."""
    results_dir: Path = args.results_dir
    if not results_dir.exists():
        print(
            f"summarize-month3: results directory not found at {results_dir}",
            file=sys.stderr,
        )
        return 2
    spike_run_paths = sorted(results_dir.rglob("*.json"))
    results: list[_AxisResult] = []
    rng = derive_substream(
        "chamber.cli.summarize_month3.bootstrap", root_seed=args.seed
    ).default_rng()
    for path in spike_run_paths:
        try:
            run_ = SpikeRun.model_validate_json(path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            print(
                f"summarize-month3: skipping {path} (not a valid SpikeRun: {exc})",
                file=sys.stderr,
            )
            continue
        result = _bootstrap_axis_result(
            run_,
            spike_run_path=path,
            n_resamples=args.n_resamples,
            rng=rng,
            gate_pp=args.gate_pp,
        )
        if result is not None:
            results.append(result)
    report = render_report(results)
    if args.output is None:
        print(report, end="")
    else:
        args.output.write_text(report, encoding="utf-8")
        print(f"summarize-month3: wrote {args.output}", file=sys.stderr)
    return 0


def _bootstrap_axis_result(
    run_: SpikeRun,
    *,
    spike_run_path: Path,
    n_resamples: int,
    rng: object,
    gate_pp: float,
) -> _AxisResult | None:
    """Build one :class:`_AxisResult` from a SpikeRun + the pacluster bootstrap."""
    pairs = build_paired_episodes(run_)
    stage = _stage_from_spike_run(run_)
    if not pairs:
        # No paired episodes — render as a measurement-absent row.
        return _AxisResult(
            axis=run_.axis,
            stage=stage,
            n_seeds=len(run_.seeds),
            n_episodes=len(run_.episode_results),
            gap_iqm_pp=0.0,
            ci_low_pp=0.0,
            ci_high_pp=0.0,
            spike_run_path=spike_run_path,
            prereg_sha=run_.prereg_sha,
            git_tag=run_.git_tag,
            gate_pp=gate_pp,
        )
    ci = pacluster_bootstrap(pairs, n_resamples=n_resamples, rng=rng)  # type: ignore[arg-type]
    return _AxisResult(
        axis=run_.axis,
        stage=stage,
        n_seeds=len(run_.seeds),
        n_episodes=len(run_.episode_results),
        gap_iqm_pp=ci.iqm * 100.0,
        ci_low_pp=ci.ci_low * 100.0,
        ci_high_pp=ci.ci_high * 100.0,
        spike_run_path=spike_run_path,
        prereg_sha=run_.prereg_sha,
        git_tag=run_.git_tag,
        gate_pp=gate_pp,
    )


def _stage_from_spike_run(run_: SpikeRun) -> str:
    """Read the per-axis stage label from the first episode's metadata.

    Falls back to the canonical per-axis stage
    (:data:`_DEFAULT_STAGE_BY_AXIS`) when no episode carries a
    ``"stage"`` key — that's the path for legacy archives that
    pre-date the Stage 1a / Stage 1b split.
    """
    for ep in run_.episode_results:
        stage = ep.metadata.get("stage")
        if isinstance(stage, str) and stage:
            return stage
    return _DEFAULT_STAGE_BY_AXIS.get(run_.axis, "1b")


__all__ = [
    "RECOMMENDATION_ACCEPT_PARTIAL_DEFER",
    "RECOMMENDATION_ACCEPT_VALIDATED",
    "RECOMMENDATION_DEFER",
    "RECOMMENDATION_STOP",
    "add_parser",
    "decide_recommendation",
    "render_report",
    "run",
]
