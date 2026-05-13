# ADR-008: HRS bundle composition

**Status.** Accepted (2026-05-13) (dependency update 2026-05-08 in light of ADR-007 rev 3 staged rollout)
**Open work.** Final axis lineup is data-dependent: the Decision below names a default bundle (CM × PF × CR) and fallback rules keyed to which axes survive the ADR-007 staged spikes; see [ADR-INDEX footnote c](ADR-INDEX.md#open-work-flags).
**Authors.** Farhad Safaei
**Reviewers.** _to fill_
**Tags.** v0.2 §3.7

## Context
v0.2 §3.7 fixes HRS = G × (1 − v) over a 3-axis bundle (latency × drop × partner-familiarity). ADR confirms this composition or revises based on customer-discovery and Phase-0 axis spikes. **Updated 2026-05-08:** ADR-007 rev 3 sets the candidate-axis universe at 6 axes (control rate, action space, observation modality, communication, partner familiarity, safety) and stages the spikes as Stage 1 (AS + OM), Stage 2 (CR + CM), Stage 3 (PF + SA). The HRS bundle now selects 3 axes from whichever subset survives the ≥20pp gate at each stage; Options A / B below remain the leading candidates but are joined by a new Option D anchored on the staged outcomes.

## Considered alternatives

| # | Option | Source / advocate | Pros | Cons |
|---|--------|-------------------|------|------|
| A | Latency × drop × partner-familiarity (v0.2 §3.7 default) | [plan] | Defensible against communication-pain hypothesis; aligns with ADR-007 Stage 2 CM + Stage 3 PF if both pass | Drops jitter / OOO / degraded-partner from headline; if Stage 3 PF fails the ≥20pp gate, Option A loses one of its three axes |
| B | Latency × drop × degraded-partner | [discovery if integrators name degraded as top-3] | Better matches industrial reality if discovery validates; has formal F < N/4 precedent in literature (note 39) | Less aligned with academic partner-familiarity framing; customer-discovery synthesis still pending |
| C | All five comm-related axes | [ambitious] | Most comprehensive | Headline number too dense for leaderboard |
| D | Bundle = 3 highest-information surviving axes from ADR-007 rev 3 staged spikes (default ordering: CM > PF > CR > SA > OM > AS by axis-distinctiveness for headline reporting) | [ADR-007 rev 3 + adrs/international_axis_evidence.md] | Decision-rule binds the bundle to whatever the empirical Phase-0 spikes validate, removing dependence on a fixed pre-spike list; aligns the HRS with the exact axis set the project has demonstrated ≥20pp gaps on | Requires ADR-007 staged spikes to complete before HRS is locked; if Stage 2 fails (CR + CM both <20pp) the bundle has only AS + OM + (PF or SA) and the headline is biased toward "free" axes; introduces axis-distinctiveness ranking as a new sub-decision |

## Decision
Adopt **Option D** — bind the HRS bundle to the 3 highest-information surviving axes from ADR-007's staged spikes, with default ordering CM > PF > CR > SA > OM > AS for headline reporting. The default bundle is **CM × PF × CR** (latency / drop within CM; partner-familiarity; control-rate ratio); if Stage 3 PF fails the ≥20 pp gate, fall back to Option B (latency × drop × degraded-partner) using the F < N/4 formula from note 39; if Stage 2 CR fails, fall back to Option A (latency × drop × partner-familiarity). The fallback rules are part of this Decision; no further ADR is required to apply them. Customer-discovery synthesis, if it adds a new top-3 pain point, may motivate a superseding ADR but does not block this Decision.

**HRS-vector emission is unconditional.** Every leaderboard entry MUST carry both the per-axis HRS vector and the aggregated HRS scalar; the scalar is the ranking metric, and the vector is the auditable per-axis breakdown that lets downstream consumers recompute the scalar under a different weighting without re-running the spikes. The leaderboard renderer (`chamber.evaluation.render.render_leaderboard`) refuses entries that carry only the scalar. The pre-registered scalar ordering is the §Rationale ordering: `CM > PF > CR > SA > OM > AS`. The implementation lives in `src/chamber/evaluation/hrs.py` (`compute_hrs_vector`, `compute_hrs_scalar`, `DEFAULT_AXIS_WEIGHTS`); per-axis weights are exposed as a parameter so a post-spike re-weighting can be applied without a code change, but any post-hoc reweighting that ships in the leaderboard requires a new ADR (see §Open questions).

## Rationale
The two available tier-2 notes confirm that communication latency and partner degradation are formally tractable axes in multi-robot safety literature, but neither directly addresses HRS bundle composition for §3.7. Note 41 (Ballotta & Talak 2024, `notes/tier2/41_ballotta_talak_2024.md`) operationalizes latency via Age-of-Information (AoI), establishing a concrete measurable proxy for the latency axis; however, the paper assumes homogeneous robots, so direct translation to heterogeneous settings requires care. Note 39 (Cavorsi 2022, `notes/tier2/39_cavorsi_2022.md`) formalises a degraded-partner budget (F < N/4 from the (2F+1)-robustness condition) that provides formal precedent for promoting "degraded-partner" to an explicit bundle axis, lending support to Option B over Option A — but the result comes from a navigation/consensus context, not manipulation. The critical disambiguation between Options A and B — whether integrators name degraded-partner rather than partner-familiarity as a top-3 pain point — depends on customer-discovery evidence that is still pending. Option C (all five axes) would produce a headline metric too dense for leaderboard use, making it unlikely to be adopted absent strong user demand.

## Evidence basis (links to reading notes)
- [customer-discovery synthesis] — _pending_
- [notes/tier2/41_ballotta_talak_2024.md] — AoI-conditioned predictor operationalises the latency axis; Theorem 11 locality result supports per-agent axis measurement decomposition in §3.7.
- [notes/tier2/39_cavorsi_2022.md] — F < N/4 degraded-partner budget formalises the third-axis candidate (Option B); nested-CBF relaxation motivates soft constraint handling when degraded-partner count exceeds budget.

## Consequences
- Project scope: The axis choice directly determines what the leaderboard reports and which robustness scenarios must be instrumented in Phase 1; Option B requires defining a formal degraded-partner threshold in §3.4 before Phase 1 data collection begins.
- v0.2 plan sections affected: §3.7 (HRS formula definition must be updated if Option B is chosen), §3.2 (benchmark task lattice must include comm-degradation scenarios), §3.10 (baseline evaluation must log all candidate axes for post-hoc comparison).
- Other ADRs: ADR-006 (partner-policy assumption set) must bound degraded-partner behaviour for the F-budget formula to apply; ADR-007 (heterogeneity-axis selection) constrains how partner-familiarity is quantified without joint-training access. **ADR-007 rev 3 staging interaction**: the HRS bundle cannot lock until ADR-007 Stage 2 has cleared its gate (Stage 2 produces the CR / CM data that Option D's default bundle relies on); if Stage 3 has not run by the Month-3 review, partner-familiarity remains a *prior* axis in Option A rather than a *validated* axis in Option D. ADR-014 (safety reporting) determines whether SA can enter the bundle as a 3rd or 4th axis once Stage 3 SA spike completes.
- Compute / time: Instrumenting latency + drop + one more axis is baseline; five axes (Option C) adds ≈2× metric-logging overhead and complicates leaderboard presentation.

## Risks and mitigations
- [risk] Customer discovery validates degraded-partner but Option A (partner-familiarity) is already hardcoded in preliminary baselines → mitigation: tag all §3.7 references with `#design-decision` so a batch update can propagate Option B; keep Option A as internal default until synthesis is complete.
- [risk] AoI-based latency measurement (note 41) assumes synchronous communication timestamps unavailable in heterogeneous setups → mitigation: define a fallback latency proxy (round-trip-time per message) in the §3.7 instrumentation spec before Phase 1 data collection; verify AoI availability in the chosen simulator (ADR-005 choice).
- [risk] Leaderboard axis framing may not match community expectations, reducing adoption → mitigation: align with COHERENT and RoCo benchmark framing before Phase 1 workshop submission.

## Reversibility
Type-2 (two-way door): the HRS formula is a post-hoc metric applied to independently logged axis signals; axes can be reweighted or substituted before Phase 2 paper freeze without invalidating logged data, provided the raw axis logs are retained. Escalates to Type-1 if any public disclosure (workshop paper, demo, preprint) has already cited the specific three-axis formula with named axes — at that point axis renaming incurs a citation-correction burden.

## Validation criteria
By Phase-2 paper: (1) HRS score separates B0 from each baseline by ≥10 HRS points on ≥3 of 4 baselines; (2) each bundle axis measurement (latency, drop, third axis) achieves inter-run reliability of ≥ 0.80 (intraclass correlation over two independent experiment runs) confirming the axis is stably instrumentable in the Phase 1 setup.

## Open questions deferred to a later ADR
- Does "partner-familiarity" have a quantitative definition compatible with the black-box constraint (no joint-training access), or does it require a learned proxy? `#design-decision`
- Does "degraded-partner" require a formal operational threshold (e.g., > 50% action-error rate relative to nominal policy) to be distinguishable from natural performance variation? `#verify`
- Is AoI (note 41) a better latency proxy than end-to-end round-trip time for heterogeneous systems where agents have different sensing rates? `#design-decision`
- **Post-spike reweighting of HRS-scalar weights** — the §Decision Option-D ordering `CM > PF > CR > SA > OM > AS` is pre-registered, and the implementation in `src/chamber/evaluation/hrs.py` exposes weights as a parameter so a downstream consumer can reweight the vector locally. But any reweighting that ships in the *headline* leaderboard scalar (i.e., changes the ordering CONCERTO publishes against) requires a new ADR superseding this one. Triggers for opening that ADR: (i) the ADR-007 staged spikes return a surviving-axis set that violates the default ordering's information-content assumption (e.g., AS clears ≥20 pp with a wider margin than CM); (ii) the DACH discovery synthesis returns PROCEED-WITH-ADJUSTMENTS naming a different headline axis; (iii) a peer reviewer specifically requests reweighting against a published metric (e.g., a venue-specific HRS variant). `#design-decision`

## Revision history

- 2026-05-13 status re-classification: status changed from Proposed to **RFC** under the new ADR status taxonomy (see [ADR-INDEX §Status taxonomy](ADR-INDEX.md#status-taxonomy)); HRS composition remains deferred to the surviving-axis outcome of the ADR-007 staged spike protocol; no Decision content is altered.
- 2026-05-13 lock: promoted to **Accepted (2026-05-13)** under the solo-developer working policy in ADR-INDEX. The Decision is rewritten to commit to Option D (default bundle CM × PF × CR; fallback rules to Option A / Option B keyed to which ADR-007 stages clear the ≥20 pp gate). Rationale, Evidence basis, Consequences, Risks, Reversibility, Validation criteria, and Open questions are unchanged.
- 2026-05-13 amendment (HRS-vector emission contract): the Decision is extended to make HRS-vector emission unconditional alongside the scalar and to pin the implementation in `src/chamber/evaluation/hrs.py` (per-axis weights are a parameter, default ordering = §Rationale ordering `CM > PF > CR > SA > OM > AS`). A new Open question is added covering when a post-spike reweighting of the headline scalar requires a superseding ADR; the fallback rules in §Decision are unchanged.
