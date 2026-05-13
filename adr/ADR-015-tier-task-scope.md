# ADR-015: Tier-task scope freeze

**Status.** RFC
**Authors.** Farhad Safaei
**Reviewers.** _to fill_
**Tags.** v0.2 §3.3 + §7.3

## Context
v0.2 §7.3 specifies that customer-discovery synthesis locks Tier 1/2/3 task specs against the dominant pain mapping. The three plan-default tiers are: Tier 1 pick-place (single-robot primitive), Tier 2 handover (sequential dual-robot handoff), and Tier 3 long-object co-manipulation (sustained physical coupling). This ADR records the freeze decision and its consequences once the synthesis is complete. Lock gate: Phase-1 Month-7 review.

## Considered alternatives

| # | Option | Source / advocate | Pros | Cons |
|---|--------|-------------------|------|------|
| A | Plan default (Tier 1 pick-place, Tier 2 handover, Tier 3 long-object co-manipulation) | [plan + Tier-3 #46 smith2012_dualarm, #48 ortenzi2020_handover] | Maps to canonical manipulation taxonomy; each tier has literature support | Customer-discovery may invalidate pain mapping |
| B | Reduced (Tier 1 + Tier 2 only; defer Tier 3) | [budget constraint] | Tighter Phase-1 scope; lower real-robot hardware cost | Weakens cooperation argument; long-horizon physical coupling is the strongest differentiation signal |
| C | Discovery-driven — replace/swap a tier with a customer-validated task | [discovery synthesis] | Best market fit | Risks losing canonical-task literature lineage; cascades into ADR-001, ADR-009, ADR-011 |

## Decision
_pending Phase-0 customer-discovery synthesis_

The alternatives table cannot disambiguate without the synthesis: Option A is the plan default but unvalidated against customer pain; Options B and C require discovery data. Decision will be filled at the Month-3/7 review gate.

## Rationale
The plan-default task ladder is grounded in converging evidence from the Phase-0 reading corpus, even though the primary evidence input (customer-discovery synthesis) is still pending. COHERENT (notes/tier1/16_coherent.md) establishes that task complexity scales monotonically with embodiment diversity — its Mono/Duo/Trio structure is a direct analogue of the Tier 1 / Tier 2 / Tier 3 progression, validating the principle of staged cooperation complexity. RoCoBench (notes/tier2/19_roco.md) anchors §3.2 with a parallel/sequential × workspace-overlap task lattice in which pick-place and handover occupy distinct canonical quadrants; long-object co-manipulation extends the lattice to tasks requiring sustained simultaneous contact, the hardest cooperation regime. BiGym (notes/tier2/57_bigym.md) calibrates expected success rates per difficulty tier — the five-category ladder confirms that each plan-default tier is tractable but non-trivial for current policies, providing the empirical justification for placing exactly three tiers. Tier-3 refs.bib entries #46 (smith2012_dualarm — canonical dual-arm survey) and #48 (ortenzi2020_handover — handover taxonomy) exist as cite_only bibtex anchors and supply taxonomic grounding for Tier-2 handover and Tier-3 co-manipulation respectively, but no deep reading note has been produced for either; both Evidence Basis links remain _pending_. Customer-discovery synthesis (v0.2 §7.3, primary input per ADR-INDEX locking rule) is required before this rationale can commit to any single option.

## Evidence basis (links to reading notes)
- [customer-discovery one-page synthesis] — _pending_ (primary input; required to select among A/B/C)
- [notes/tier3/refs.bib #48 ortenzi2020_handover] — _pending_ (cite_only bibtex anchor; no deep reading note; taxonomic support for Tier-2 handover)
- [notes/tier3/refs.bib #46 smith2012_dualarm] — _pending_ (cite_only bibtex anchor; no deep reading note; taxonomic support for Tier-3 co-manipulation)
- notes/tier1/16_coherent.md — Mono/Duo/Trio tier structure confirms task-complexity scaling with embodiment diversity (§3.2)
- notes/tier2/19_roco.md — parallel/sequential × workspace-overlap task lattice anchors §3.2 task set; canonical cells for pick-place and handover
- notes/tier2/57_bigym.md — five-category task difficulty ladder with ACT-calibrated success ranges; empirical support for three-tier tractability gradient

## Consequences
- Project scope: If Option A is confirmed, Phase-1 benchmark spec commits to all three manipulation tiers with no substitution; all downstream implementation (ADR-001 benchmark, ADR-009 zoo, ADR-011 baselines) targets this task set. If Option B is chosen, Tier-3 is deferred to Phase-2 and the cooperation argument at Phase-1 demo weakens. If Option C replaces a tier, scope cascades are significant.
- v0.2 plan sections affected: §3.2 (task set is directly constrained by the freeze); §3.3 (task-allocation policy must be calibrated to the confirmed tier set); §3.5 (partner-zoo policies must cover each tier's action requirements); §7.3 (freeze gate closes once this ADR is Accepted).
- Other ADRs: ADR-001 (benchmark fork must expose task-definition hooks for the confirmed tier set); ADR-009 (partner-zoo construction calibrated to task complexity gradient); ADR-011 (baseline set must include at least one entry per tier).
- Compute / time / hiring: Tier-3 long-object co-manipulation requires ≥ 2 robots with coordinated force sensing at real-robot validation; deferring Tier-3 (Option B) reduces Phase-1 hardware cost and timeline risk. No new hiring implied by Option A; Option C may require task-domain expertise depending on replacement task.

## Risks and mitigations
- [Customer-discovery synthesis invalidates Option A after Phase-1 infrastructure investment] → mitigation: design benchmark as modular per ADR-001 (per-component controller composition from ManiSkill2, notes/tier2/54_maniskill2.md) so task swap requires only task-definition files, not physics-stack rewrite; defer demo collection campaign until ADR-015 Decision is Accepted.
- [Tier-3 long-object co-manipulation imposes force/contact requirements that simulation cannot faithfully model until ADR-013 selects a real-robot platform] → mitigation: implement Tier-3 as a rigid-body co-transport proxy in simulation; mark all Tier-3 benchmark results as "sim-only" until physical validation is confirmed; use SafeBimanual (notes/tier2/53_safebimanual.md) cost-guided diffusion as an alternative safety-filter path for the sim proxy.

## Reversibility
Type-1 once task-specific demonstration data is collected (threshold: > 50 teleoperated demonstrations per tier per embodiment pairing). The irreversible step is the demo-collection campaign: after collection, swapping tiers requires full recollection. Exit ramp: preserve the BiGym VR teleoperation pipeline (notes/tier2/57_bigym.md — Valve Index → per-arm IK → 500 Hz capture) as a fallback collection pathway for any replacement task, and defer collection to the last responsible moment after the freeze is Accepted.

## Validation criteria
By Phase-1 end (Month-7 gate): (1) each confirmed tier has a documented integrator-workflow mapping in the benchmark spec (binary pass/fail); (2) at least one B1–B7 baseline achieves ≥ 20% success on Tier-1, ≥ 5% on Tier-2, and > 0% on Tier-3 in simulation; (3) the freeze is recorded as Accepted in ADR-INDEX with a linked customer-discovery synthesis document.

## Open questions deferred to a later ADR
- Does customer-discovery synthesis confirm, reduce, or replace the plan-default task tiers? (primary blocker for Decision; resolves at Month-3/7 review gate — no ADR split needed if Option A is confirmed)
- Is Tier-3 long-object co-manipulation achievable in ManiSkill2 without custom contact-simulation extensions? (#design-decision; defers to ADR-001 benchmark-fork decision and ADR-005 simulator-base selection)
- What partner-capability gap is required between tiers to ensure cooperation patterns are pedagogically distinct and not collinear in behavior space? (#verify against BiGym difficulty-ladder ACT calibration, notes/tier2/57_bigym.md, at Phase-1 benchmark-spec review)

## Revision history

- 2026-05-13 status re-classification: status changed from Proposed to **RFC** under the new ADR status taxonomy (see [ADR-INDEX §Status taxonomy](ADR-INDEX.md#status-taxonomy)); Phase-1 lock pending DACH customer-discovery synthesis; no Decision content is altered.
