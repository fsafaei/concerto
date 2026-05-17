# ADR-016: SpikeRun schema lifecycle (Stage 1a / 1b disambiguation)

**Status.** Accepted (2026-05-17)
**Open work.** Stage-2 and Stage-3 adapters do not yet exist; when they ship, they MUST populate `sub_stage` to one of `"2"` or `"3"` at construction. The `SubStage` Literal is forward-compatible with that work. See [ADR-INDEX footnote b](ADR-INDEX.md#open-work-flags).
**Authors.** Farhad Safaei
**Reviewers.** _to fill_
**Tags.** v0.2 §3.4 §3.7

## Context

`chamber.evaluation.results.SpikeRun` is the wire-format archive every spike run serialises to disk and every downstream consumer (the summarizer, the leaderboard, the reproduction harness) reads back. `SCHEMA_VERSION` (this module's, distinct from `chamber.comm.SCHEMA_VERSION`) governs that wire format.

ADR-007 revision 4 (2026-05-15) split Stage 1 of the ADR-007 §Implementation-staging protocol into two sub-stages:

- **Stage 1a** — rig validation. Runs against `chamber.envs.mpe_cooperative_push.MPECooperativePushEnv`, the CPU-friendly stand-in. Production ego is `_zero_ego_action_factory` by design (Stage 1a's purpose is to exercise the rig, not to evaluate the AS / OM axes as §Validation criteria defines them). **The ≥20 pp gate is NOT measured under Stage 1a.**
- **Stage 1b** — real-env science evaluation. Runs against the canonical ManiSkill v3 pick-place task with a trained ego per seed. **The ≥20 pp gate IS measured under Stage 1b** and is the §Validation-criteria gate that promotes ADR-007 to Validated for the AS / OM axes.

The two sub-stages share the pre-registration YAML, the seed list, and the condition-pair contract; only the env implementation and the ego policy differ. From the SpikeRun archive's perspective they produce identical-shaped output — same `axis`, same `condition_pair`, same `seeds`, same per-episode count.

The `chamber.cli._spike_summarize_month3` Month-3 lock-priority report routes on this distinction. A Stage-1a SpikeRun MUST short-circuit to `Defer — Stage 1b not yet measured (Phase-1 milestone; guardrail ≤4 weeks)` regardless of the synthesised gap; a Stage-1b SpikeRun runs through the four-state decision rule and can route to Stop / Accept-Partial-Defer / Accept-Validated. Mis-routing a Stage-1a run as Stage-1b reports a structural project failure when in fact the rig is fine and Stage 1b is queued exactly as planned (the 2026-05-17 incident).

The v1 SpikeRun schema (this module's `SCHEMA_VERSION: int = 1`) had no field that distinguished 1a from 1b. An informal contract had been wired into the summarizer module docstring (`chamber.cli._spike_summarize_month3` paragraph 6) — an optional `EpisodeResult.metadata["stage"]` key with a silent default-by-axis fallback (AS / OM → "1b") in `_stage_from_spike_run`. The Stage-1a adapters (`chamber.benchmarks.stage1_{as,om}`) never populated that metadata key; the silent fallback then misrouted the produced archives as Stage 1b, which produced the `Stop — ADR re-review (no axes survived)` output at the 2026-05-17 maintainer triage.

## Considered alternatives

| # | Option | Pros | Cons |
|---|--------|------|------|
| A | Add a direct field on `SpikeRun`: `sub_stage: Literal["1a", "1b", "2", "3"]`. Bump `SCHEMA_VERSION` 1 → 2. Adapter populates at construction. | Type-safe; one source of truth at the wire-format layer; no silent default-fallback. Forward-compatible for Stage 2 / Stage 3 adapters (which will need the field too). | Breaking change to the on-disk archive shape; existing v1 archives must be regenerated (one-shot migration); requires this ADR. |
| B | Wrap `SpikeRun` in a new `RunContext` dataclass that adds `sub_stage` + future metadata (run host, wall-clock, hardware profile). `SpikeRun` stays at v1. | More extensible; new metadata accretes on `RunContext` without further `SpikeRun` bumps. | Every consumer must be updated (leaderboard renderer, summarizer, bootstrap, reproduction harness); serialised SpikeRuns under `spikes/results/` become awkward; only one field is currently needed — no other `RunContext` field is imminent in any ADR. |
| C | Stamp `sub_stage` into the existing `SpikeRun.metadata` dict (`metadata: dict[str, Any]` on `EpisodeResult`). No schema bump. | No breaking change; matches the informal contract the summarizer was already shaped against. | Loses type safety; the summarizer must read with `.get("sub_stage")` and handle the absent / wrong-type case; the silent-fallback foot-gun stays around; Stage 2 adapters will need to remember to populate the dict for their canonical stage by convention rather than by type contract. |

## Decision

Adopt **Option A**. Add `sub_stage: Literal["1a", "1b", "2", "3"]` to `chamber.evaluation.results.SpikeRun` as a required (no-default) field. Bump `chamber.evaluation.results.SCHEMA_VERSION` from 1 to 2. The Stage-1a adapters (`chamber.benchmarks.stage1_{as,om}.run_axis`) stamp `sub_stage="1a"` at the `SpikeRun(...)` construction site. The Stage-1b adapters will stamp `sub_stage="1b"` when they ship in Phase 1. Stage-2 / Stage-3 adapters will stamp `"2"` / `"3"` respectively.

The summarizer at `chamber.cli._spike_summarize_month3._stage_from_spike_run` reads `run.sub_stage` directly with no fallback. The previous `EpisodeResult.metadata["stage"]` affordance is retired (the dict-key reader and the `_DEFAULT_STAGE_BY_AXIS` routing-fallback are removed; the per-axis-table renderer's `_DEFAULT_STAGE_BY_AXIS` UI affordance is retained for the *Missing*-row stage-cell only).

**v1 archives are not auto-migrated.** Per the Pydantic v2 `extra="forbid"` config on `SpikeRun`, v1 archives missing the `sub_stage` field fail to load against the v2 schema with a `ValidationError`. The two v1 archives previously committed at `spikes/results/stage1-{AS,OM}-20260517/` are removed in this same PR (PR 2) and regenerated under v2 in the follow-up PR (PR 3) using the same pre-registration YAMLs and the same git tags — the regeneration produces byte-identical content modulo the new `sub_stage` field and the bumped `schema_version`.

**`LeaderboardEntry.schema_version` co-bumps** because both `SpikeRun` and `LeaderboardEntry` alias the same module-level `SCHEMA_VERSION` constant. The `LeaderboardEntry` wire shape is unchanged at v2 (it gains no fields); the bump is driven entirely by `SpikeRun.sub_stage`. Future consumers that need to distinguish them per-class without changing this ADR can split the constant into `SPIKE_RUN_SCHEMA_VERSION` and `LEADERBOARD_SCHEMA_VERSION` in a follow-up ADR.

## Rationale

**(i) `SCHEMA_VERSION` discipline.** The module docstring on `chamber.evaluation.results` (lines 13–17 and 28–30) pins the rule: "Bumping either [comm or evaluation `SCHEMA_VERSION`] is a breaking change to `SpikeRun` / `LeaderboardEntry` serialised shape and requires a new ADR." This ADR is that ADR. The discipline is honoured by writing the schema bump and the ADR together rather than by avoiding the bump — Option C's "honour the discipline trivially by never bumping" trades type safety for an invariant the project explicitly wants the type system to enforce (the load-bearing sub-stage signal is exactly the class of fact a Pydantic v2 `Literal` is designed to lock down).

**(ii) ADR-007 §Discipline audit-trail requirement.** ADR-007 §Discipline already pins `prereg_sha` and `git_tag` as first-class `SpikeRun` fields because their absence would silently invalidate the run. The Stage-1a-vs-1b distinction is in the same load-bearing category: it determines whether the ≥20 pp gate is binding, and a missing-or-misread value flips the Month-3 lock recommendation between `Defer` and `Stop`. ADR-007 revision 4 elevated the sub-stage split to §Implementation-staging-level structure (the staging summary table now has separate 1a / 1b rows; the Stage-1b trigger guardrail is a 4-week clock keyed off this distinction). A field this load-bearing belongs at the SpikeRun level alongside `axis`/`prereg_sha`, not buried in `EpisodeResult.metadata` behind a `.get()`-and-fallback.

**(iii) Future-proofing for Stage 2 / Stage 3.** Stage 2 / Stage 3 adapters do not yet exist (only `chamber.benchmarks.stage1_{as,om}` ship at the time of this ADR). Locking `sub_stage` as a first-class typed field now means the Stage-2 / Stage-3 PRs inherit a single populate-at-construction discipline rather than rediscovering the metadata-dict contract. The `Literal["1a", "1b", "2", "3"]` type also closes the typo class (`"2a"`, `"1B"`, `"stage1"`) that a stringly-typed `metadata["stage"]` cannot reject.

## Evidence basis

- 2026-05-17 maintainer-triage incident: production `chamber-spike summarize-month3 --results-dir spikes/results/` against the four committed Stage-1a archives produced `**Recommendation: Stop — ADR re-review (no axes survived)**` instead of the expected `**Recommendation: Defer — Stage 1b not yet measured**`. Root cause: Stage-1a adapter never populated `EpisodeResult.metadata["stage"]`, the summarizer fell back to `_DEFAULT_STAGE_BY_AXIS["AS"] == "1b"`, and the always-zero policy produced by `_zero_ego_action_factory` produced gap_iqm_pp = 0.00 which routed to Stop.
- [ADR-007 §Stage 1a](ADR-007-heterogeneity-axis-selection.md#stage-1a--rig-validation-phase-0-mpe-stand-in) — names the sub-stage split and the load-bearing role of the gate-NOT-measured affordance.
- [ADR-008 §Decision + 2026-05-13 amendment](ADR-008-hrs-bundle.md#decision) — pins `chamber.evaluation.results.SCHEMA_VERSION` lifecycle for `LeaderboardEntry` (the LE co-bump rationale).

## Consequences

- **`chamber.evaluation.results`:** `SpikeRun.sub_stage` is required; `SCHEMA_VERSION = 2`; `SubStage` Literal is exported alongside the other public symbols.
- **`chamber.benchmarks.stage1_{as,om}`:** adapters populate `sub_stage="1a"` at the `SpikeRun(...)` construction site. The construction-site comment cites ADR-007 §Stage 1a and this ADR.
- **`chamber.cli._spike_run` (dry-run path):** the synthetic SpikeRun built by `--dry-run` populates `sub_stage` from a canonical per-axis mapping (AS / OM → "1b"; CR / CM → "2"; PF / SA → "3"). The dry-run defaults to the science-evaluation stage so the synthesised gap exercises the four-state recommendation logic; this is a UI affordance, not a routing fallback.
- **`chamber.cli._spike_summarize_month3`:** `_stage_from_spike_run` reads `run.sub_stage` directly; the prior `EpisodeResult.metadata["stage"]` reader is removed; the `_DEFAULT_STAGE_BY_AXIS` constant is retained but its semantic role narrows to a UI affordance for the per-axis-table renderer's *Missing*-row stage cell.
- **`spikes/results/` archives:** the four v1 blobs at `spikes/results/stage1-{AS,OM}-20260517/{spike,leaderboard}.json` are removed in PR 2 (this ADR) and regenerated under v2 in PR 3. Between PR 2 and PR 3, `chamber-spike summarize-month3` on a clean checkout returns the empty-dir Defer path (no archives present → no Pydantic error).
- **`LeaderboardEntry`:** co-bumps to `schema_version: 2`. Wire shape is unchanged at v2.
- **Tests:** every `SpikeRun(...)` construction in `tests/` is updated to set `sub_stage` explicitly (defaulting synthetic fixtures to `"1b"` so the four-state recommendation logic stays exercised).

## Risks and mitigations

- **Risk:** v1 archives that have leaked outside the repo (e.g. a teammate's local clone, a CI cache) fail to load against the v2 schema. — **Mitigation:** the Pydantic `ValidationError` is the load-bearing loud-fail mode; the failure message names the missing `sub_stage` field. The migration is one-shot: regenerate via `scripts/repro/stage1_{as,om}.sh`. The pre-registration tags do not rotate (ADR-007 §Discipline), so the regenerated archive's `prereg_sha` matches the previous v1's exactly.
- **Risk:** a Stage-2 / Stage-3 adapter author forgets to populate `sub_stage`. — **Mitigation:** the field is type-required at construction; `Pydantic.ValidationError` fires at `run_axis` exit, not after the spike has run for hours. The construction-site comment in `stage1_{as,om}.py` documents the contract.
- **Risk:** mixed-version archives (some v1, some v2) under `spikes/results/`. — **Mitigation:** the summarizer's load loop catches `ValidationError` per-file and skips with a stderr note; the surrounding report renders correctly for the valid archives. The post-PR-3 state is v2-only.

## Reversibility

Type-2 within Phase 0 — the schema can be re-bumped (or `sub_stage` renamed / re-typed) via a superseding ADR before any external consumer relies on the on-disk archive shape. Type-1 once the v2 schema lands in a tagged release (the Phase-0 0.x line ships with v2 as the audit-trail-compliant contract); past that, any further bump requires a deprecation cycle for the on-disk format.

## Validation criteria

- `chamber-spike summarize-month3 --results-dir <dir-with-stage-1a-archives>` produces `**Recommendation: Defer — Stage 1b not yet measured (Phase-1 milestone; guardrail ≤4 weeks)**` regardless of the per-episode success pattern. Pinned by `tests/unit/test_summarize_month3.py::TestSummarizeMonth3SubStageRouting::test_stage_1a_archive_routes_to_defer_regardless_of_gap`.
- `SpikeRun(...)` without `sub_stage` raises `pydantic.ValidationError`. Pinned by `tests/unit/test_spike_run_sub_stage_field.py::TestSubStageFieldShape::test_sub_stage_is_required`.
- `SCHEMA_VERSION == 2`. Pinned by `tests/unit/test_spike_run_sub_stage_field.py::TestSchemaVersionPin`.
- Production Stage-1a adapters stamp `sub_stage="1a"`. Pinned by `tests/integration/test_stage1_{as,om}_real.py::TestStage1{AS,OM}PreregDiscipline::test_run_axis_records_sub_stage_1a`.

## Open questions deferred to a later ADR

- Should `SPIKE_RUN_SCHEMA_VERSION` and `LEADERBOARD_SCHEMA_VERSION` split into separate module constants? This ADR co-bumps them via the shared `SCHEMA_VERSION` alias. The split is cleaner for downstream consumers that version each class independently but widens any single PR's scope; defer the split until a `LeaderboardEntry`-only bump is actually needed.
- Should `sub_stage` decompose further once Stage 2 / Stage 3 ship (e.g. `"2a"` for rig-validation analogues at Stage 2)? Defer until the Stage-2 / Stage-3 adapter design lands.
- **`chamber.cli._spike_next_stage` sub_stage-awareness.** The `next-stage` subcommand's `_load_spike_runs` filter at `src/chamber/cli/_spike_next_stage.py` selects Stage-1 axes by `r.axis in {"AS", "OM"}` only, with no `sub_stage` predicate. After PR 3 regenerates Stage-1a archives, invoking `chamber-spike next-stage --prior-stage 1 --spike-runs <stage-1a-archive>` would bootstrap a synthetic gap against rig-validation data and could route Phase-1-launch decisions off it — the same class of bug PR 2 closes for `summarize-month3`. Resolution: either (a) skip Stage-1a archives with a stderr note in `_load_spike_runs`, or (b) reject `--prior-stage 1` against a Stage-1a-only archive set with the same Defer semantics as the Month-3 summarizer. Tracked as a follow-up to PR 2; non-blocking for the schema bump itself.

## Revision history

- 2026-05-17 (initial draft + lock): introduces `SpikeRun.sub_stage`; bumps `SCHEMA_VERSION` 1 → 2; retires the `EpisodeResult.metadata["stage"]` informal contract. Motivated by the 2026-05-17 maintainer-triage incident (mis-routed Stage-1a archives reporting `Stop` instead of `Defer`).
