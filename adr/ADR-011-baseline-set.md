# ADR-011: Baseline set — which of B1–B7 ship in Phase 1

**Status.** Proposed (dependency update 2026-05-08 in light of ADR-007 rev 3 staged rollout)
**Authors.** Farhad Safaei
**Reviewers.** _to fill_
**Tags.** v0.2 §3.10 + Phase 1

## Context
v0.2 Phase 1 deliverables list B1, B2, B3, B6, B7. B5 may slip to early Phase 2. ADR confirms scope or revises based on porting cost.

## Considered alternatives

| # | Option | Source / advocate | Pros | Cons |
|---|--------|-------------------|------|------|
| A | Plan default (B1, B2, B3, B6, B7) | [plan] | Phase-1 deliverable as specified | B5 misses the early-Phase-2 deadline |
| B | Reduced (B1, B2, B7) | [budget] | Tight Phase-1 scope | Weaker baseline coverage |
| C | Plan default + B5 on time | [Tier-1 #25 HARL] | Strongest comparative story | HARL port cost may not fit Phase 1 |

## Decision
**Option A tentatively selected — B1, B2, B3, B6, B7 ship in Phase 1; B5 (HAPPO + HATD3) slips to early Phase 2.**

Rationale for selecting A over C: `notes/tier1/25_harl.md` confirms that HARL's monotonic-improvement guarantee (Theorem 7) requires all agents to update simultaneously — it is lost when the partner policy is frozen, as required by B0's AHT setting. A custom AHT wrapper must be built and validated before B5 is runnable; this engineering effort is not accounted for in the Phase 1 budget and its duration is unknown without a porting spike. Selecting C now would commit Phase 1 to an unestimated task. Once a porting spike is completed, this ADR may be revised to Option C if the effort fits Phase 1.

Rationale for selecting A over B: `notes/tier1/23_hetgppo.md` confirms B3 (HetGPPO-style heterogeneous MARL) is feasible with a frozen-partner AHT fork of the VMAS+RLlib stack; this verification is a known, bounded effort. Dropping B3 (Option B) would remove the only heterogeneous-MARL Phase 1 baseline, weakening differentiation from homogeneous methods.

## Rationale

The four evidence notes collectively justify the plan-default scope with B5 deferred. `notes/tier1/25_harl.md` (HARL) establishes HAPPO as the B5 on-policy baseline and HATD3 as the off-policy variant, and provides the clearest justification for why port cost is non-trivial: the sequential update scheme that makes HARL theoretically sound requires access to all agents' policy parameters, which is incompatible with the frozen, opaque partner required in AHT; additionally, the Bi-DexHands evaluation (two identical Shadow Hands) may not represent the mobile-base × fixed-arm heterogeneity of B0, so replication is non-trivial. `notes/tier1/23_hetgppo.md` (HetGPPO) positions B3 in the H × black-box design space, warns that the VMAS+RLlib AHT fork needs a feasibility check before Phase 1 commits, and empirically motivates the per-agent parameter design (behavioral-typing brittleness under deployment noise) that B3 must embody. `notes/tier1/15_liu_2024_rss.md` (Liu 2024 RSS) anchors B1 (LLM-based AHT) with a concrete reference number — 26.3 % task success on ProcTHOR under IRoT-LLM — and confirms that the generate-rank-filter pattern (IRoT) is the appropriate implementation template for B1 in our §3.3 task-allocation loop. `notes/tier1/16_coherent.md` (COHERENT) supplies the PEFA loop structure and the Mono/Dual/Trio tier ladder that B1 must be evaluated against; it also confirms that a centralized white-box planner achieves 97.5 % on its own benchmark — a number that should not be used as a B0 comparison target without controlling for benchmark inflation.

## Evidence basis (links to reading notes)
- [notes/tier1/25_harl.md] — **resolved** (HAPPO/HATD3 as B5; AHT wrapper required; monotonic guarantee lost with frozen partner)
- [notes/tier1/23_hetgppo.md] — **resolved** (HetGPPO as B3 candidate; VMAS+RLlib AHT fork feasibility check needed)
- [notes/tier1/15_liu_2024_rss.md] — **resolved** (IRoT template for B1; 26.3 % SR reference on ProcTHOR)
- [notes/tier1/16_coherent.md] — **resolved** (PEFA loop for B1; Mono/Dual/Trio tier structure; 97.5 % SR on own benchmark)

## Consequences
- **Project scope**: Phase 1 ships 5 baselines — B1 (LLM-based AHT, IRoT/PEFA template), B2 (independent learning), B3 (HetGPPO-style HMARL, per-agent parameters), B6 (CBF safety baseline), B7 (conformal prediction baseline). B4 and B5 deferred. B5 (HAPPO + HATD3) targets early Phase 2 after a porting spike.
- **v0.2 plan sections affected**: §3.8 (baseline targets), §3.10 (Phase 1 deliverables list), §3.5 (partner zoo — HARL-trained policies cannot be confirmed as zoo entries until B5 porting is proven).
- **Other ADRs**: ADR-009 (partner-zoo construction): frozen HARL-trained policies are natural B5 zoo entries but their cross-partner generalization is unverified — ADR-009 must handle this risk. ADR-006 (partner-policy assumption set): B6 and B7 baselines are directly coupled to §6 safety architecture scope. **ADR-007 rev 3 dependency**: the Phase-1 baseline set is contingent on the surviving axes from ADR-007's staged spikes — if Stage 3 PF fails the ≥20pp gate, B5 (HAPPO + HATD3) loses its primary comparative justification (the frozen-partner AHT setting becomes weaker as a heterogeneity claim) and the B5 porting spike priority drops; if Stage 3 SA fails, B6 (CBF safety baseline) and B7 (conformal prediction baseline) keep their relevance regardless because they are property-of-the-stack baselines, not axis baselines. **B5 porting spike sequencing**: schedule the B5 AHT-wrapper porting spike *after* ADR-007 Stage 2 has cleared (so CR + CM are validated as load-bearing axes that the HARL comparison must cover) and *before* ADR-009's full zoo lock (so HAPPO/HATD3 frozen checkpoints can land in the zoo as Phase-1 RL-stratum entries).
- **Compute / time / hiring**: B5 porting spike estimated at 2–4 weeks of engineering (AHT wrapper + validation on one manipulation task); not in Phase 1 budget. HAPPO's 8 % wall-time overhead vs. MAPPO (HARL §5.7, Table 3) is acceptable for Phase 2 comparison runs.

## Risks and mitigations
- **B3 AHT fork infeasible in VMAS+RLlib** (`notes/tier1/23_hetgppo.md` §3.8 open question): HetGPPO's DTDE GNN training may not support freezing only the partner's parameters while updating the ego. → Mitigation: run the AHT-fork feasibility spike in the first four weeks of Phase 1; if blocked, fall back to MAPPO with per-agent parameters as B3 proxy and document the deviation.
- **B5 slip undermines comparative story against HARL** (`notes/tier1/25_harl.md` §5.2): without B5, the Phase 1 baseline table lacks a formal HMARL comparison on the bimanual manipulation task. → Mitigation: include HAPPO's published Bi-DexHands numbers from the HARL paper as a reference-only row in Phase 1 results, explicitly noting no in-house replication; flag as deferred to Phase 2. If B5 porting spike completes early, promote to Option C by amending this ADR.
- **B1 LLM latency incompatible with §6 safety loop** (`notes/tier1/15_liu_2024_rss.md` §IV): GPT-4 latency (seconds per call) is incompatible with the sub-100 ms CBF safety filter. → Mitigation: B1 runs in an offline planning mode with a pre-computed safety envelope; the CBF filter operates on the B1 trajectory, not inside the LLM call loop. Document this split clearly in the evaluation protocol.

## Reversibility
**Type-2** — this scope decision is easily reversible. If the B5 porting spike completes ahead of schedule, this ADR is amended to Option C by updating the Decision section and re-reviewing. No infrastructure commitment locks B5 out of Phase 1 before the spike finishes.

## Validation criteria
By Phase 1 end: (1) all five shipped baselines (B1, B2, B3, B6, B7) reproduce task success within **±5 percentage points** of any published reference number on at least one Tier-1 benchmark task; (2) the HetGPPO AHT-fork feasibility spike is completed, documented, and its outcome recorded in ADR-009 before the Phase 1 milestone lock; (3) the B5 porting spike has a written estimate and start date before Phase 1 closes — start date scheduled *after* ADR-007 Stage 2 gate clears and *before* ADR-009 full-zoo lock, per the staging dependency. (4) **Coverage check against ADR-007 surviving axes**: each shipped baseline must be evaluated under at least one experimental condition that exercises a surviving §3.4 axis from ADR-007's staged spikes — i.e., the baseline table must not silently drop the heterogeneity dimensions the project has just validated as ≥20pp.

## Open questions deferred to a later ADR
- #verify (from `notes/tier1/25_harl.md`): Can a frozen-partner HAPPO setup retain any useful property of the monotonic improvement guarantee — e.g., via a single-agent PPO objective on ego rollouts against a frozen partner? Theorem 7 requires all agents to update; with a frozen partner the guarantee is lost, but empirical convergence may still hold.
- #design-decision (from `notes/tier1/25_harl.md`): Is HATD3 or HAPPO preferred as the B5 implementation? HATD3 has better sample efficiency on continuous tasks (§5.2) but a replay buffer complicates AHT adaptation; HAPPO is simpler to freeze-partner.
- #verify (from `notes/tier1/25_harl.md`): Do the Bi-DexHands tasks (two identical Shadow Hands) constitute heterogeneous-embodiment cooperation for our purposes? If both embodiments are identical, HARL's Bi-DexHands result does not represent the mobile-base × fixed-arm setting and the comparison must use a different HARL benchmark.
- #verify (from `notes/tier1/23_hetgppo.md`): Is the DTDE GNN architecture compatible with a frozen-partner AHT setup — can only the ego agent's parameters be updated while the partner's GNN encoder/decoder are frozen and its communication outputs treated as observations?
