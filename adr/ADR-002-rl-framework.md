# ADR-002: RL framework — JAX (Mava) vs PyTorch (HARL) vs other

**Status.** Provisional
**Authors.** Farhad Safaei
**Reviewers.** _to fill_
**Tags.** v0.2 §13, HMARL

## Context
v0.2 §13 names this as an open question. The team's existing strength (JAX vs PyTorch) is the dominant input; HARL has a maintained PyTorch reference at PKU-MARL/HARL. The decision interacts directly with the B5 baseline selection (§3.8), the partner-zoo construction methodology (§3.5), and the benchmark physics stack already anchored to ManiSkill2's SAPIEN/PyTorch layer (ADR-001).

## Considered alternatives

| # | Option | Source / advocate | Pros | Cons |
|---|--------|-------------------|------|------|
| A | PyTorch + HARL fork | [Tier-1 #25] | Direct B5 baseline; community; matched to HARL's monotonic-improvement guarantees | Tooling overlap with VLA partners (most VLAs are PyTorch) |
| B | JAX + Mava | [external] | Performance; differentiable physics integration | Smaller MARL community; HARL re-implementation cost |
| C | TorchRL + BenchMARL | [Tier-3 #24] | Standardised benchmarking infra; HetGPPO compatibility | Less mature than HARL for heterogeneous monotonic-improvement |

## Decision
Adopt **Option A — PyTorch + HARL fork**, contingent on a team-skill audit confirming no blocking JAX expertise that would change the calculus for Option B.

## Rationale
HARL's sequential-update scheme (Lemma 4, Algorithm 1 in `notes/tier1/25_harl.md`) is the only published MARL algorithm with a provable monotonic joint-return improvement guarantee for heterogeneous-agent settings; its PyTorch implementation is actively maintained at PKU-MARL/HARL and runs directly on Bi-DexterousHands, the closest analogue to our bimanual manipulation target. Adopting the HARL fork delivers HAPPO (on-policy B5) and HATD3 (off-policy B5) as near-drop-in baselines (`stolen_ideas.md` row 13), eliminating re-implementation risk at a point where baseline credibility is more important than training throughput. The benchmark fork anchored to ManiSkill2's SAPIEN/PyTorch physics stack (ADR-001, `notes/tier2/54_maniskill2.md`) and the VLA partner checkpoints used in Phase 1 (OpenVLA `notes/tier2/32_openvla.md`, CrossFormer `notes/tier2/33_crossformer.md`) are all PyTorch-native; a single-stack environment minimises interface complexity at the action-interception boundary (v0.2 §6). Option B (JAX + Mava) is deferred unless the team-skill audit reveals a strong JAX majority, because HARL re-implementation cost is uncompensated by differentiable-physics gains at Phase-1 scale. Option C (TorchRL + BenchMARL) is complementary, not competitive: BenchMARL's standardised logging (`notes/tier3/refs.bib #24`) can be adopted as a reporting layer over the HARL training loop without replacing it.

## Evidence basis (links to reading notes)
- [notes/tier1/25_harl.md] — HAPPO and HATD3 as B5 on/off-policy baselines; multi-agent advantage decomposition lemma (Lemma 4) as theoretical foundation; empirical proof that MAPPO fails completely on 17-agent Humanoid while HAPPO succeeds (§5.2, Fig. 8); AHT adaptation requires frozen-partner ego-only wrapper (open question, §45 note)
- [notes/tier1/23_hetgppo.md] — per-agent separate-parameter design; behavioral-typing brittleness under deployment noise (§6, Figs. 4 & 6); DTDE GNN pattern as architecture candidate; HetGPPO as B3 candidate requiring AHT fork validation
- [notes/tier3/refs.bib #24 BenchMARL] — community standard for reproducible MARL benchmarking; complementary reporting layer candidate

## Consequences
- **Project scope**: ego-agent training loop is PyTorch throughout Phase 1; HAPPO is the B5 on-policy baseline and HATD3 is the B5 off-policy baseline; full JAX/Mava migration is out of Phase-1 scope.
- **v0.2 plan sections affected**: §3.8 (HAPPO/HATD3 confirmed as B5 baseline pair), §13 (RL framework resolved), §3.4 (per-agent separate parameters locked in — supported by HetGPPO behavioral-typing evidence), §3.5 (HARL-trained frozen policies as learning-based zoo candidates require AHT wrapper validation before admission).
- **Other ADRs**: depends on ADR-001 (ManiSkill2 PyTorch anchor); informs ADR-009 (partner-zoo construction — HARL-trained policies as zoo candidates); BenchMARL logging is a non-ADR implementation decision.
- **Compute / time / hiring**: IsaacGym GPU simulation required for Bi-DexHands B5 validation; HARL frozen-partner AHT wrapper estimated 1–2 weeks of engineering before B5 runs can begin.

## Risks and mitigations
- [**AHT frozen-partner HAPPO breaks the monotonic-improvement guarantee**] Theorem 7 in `notes/tier1/25_harl.md` requires all agents to update jointly; freezing the partner's policy eliminates the formal guarantee. Mitigation: prototype ego-only HAPPO with a frozen partner on a 2-agent MPE task and verify empirically that per-epoch reward is non-decreasing; document whether the guarantee is merely informal or quantifiably degraded before scheduling B5 runs.
- [**PyTorch version conflicts between HARL and VLA partner inference**] OpenVLA and CrossFormer may pin different PyTorch/CUDA versions than the HARL training loop, causing silent incompatibilities at the action-interception boundary. Mitigation: containerise the HARL trainer and each VLA inference server in separate Docker images communicating over a thin action-bus interface; the conformal CBF layer (v0.2 §6) is the natural inter-process boundary and isolates version drift.

## Reversibility
**Type-2** (two-way door). Porting HAPPO/HATD3 to JAX/Mava is non-trivial (estimated 2–4 person-weeks) but technically feasible at any point before Phase-2 scale-up; no data artifacts or trained checkpoints are framework-locked. The exit ramp is available as long as HARL customisations have not diverged substantially; the practical reversal deadline is Phase-1 mid before team-specific JAX expertise erodes.

## Validation criteria
By Phase-1 mid: ego-only HAPPO (frozen partner) reproduces published HARL Bi-DexterousHands ShadowHandOver score within 5 percentage points of Table 2 in `notes/tier1/25_harl.md`; HATD3 matches within 5pp on ShadowHandCatchOver2Underarm. Both runs must use per-agent separate parameters and the AHT frozen-partner wrapper.

## Open questions deferred to a later ADR
- (#verify, from `notes/tier1/25_harl.md`) Can the ego-only HAPPO frozen-partner adaptation preserve a weaker convergence guarantee, or does it reduce to an empirical baseline only? If the guarantee is entirely lost, document the implication for §3.8 B5 claim language.
- (#design-decision, from `notes/tier1/25_harl.md`) Should HATD3 or HAPPO be prioritised for the bimanual manipulation target? HATD3 is more sample-efficient but replay-buffer complicates AHT adaptation; defer to prototype benchmark results.
- (#verify, from `notes/tier1/23_hetgppo.md`) Is HetGPPO (B3) feasible as a frozen-partner AHT baseline with the VMAS + RLlib stack? Confirm before committing to the §3.8 baseline set.

## Revision history

- 2026-05-13 status re-classification: status changed from Proposed to **Provisional** under the new ADR status taxonomy (see [ADR-INDEX §Status taxonomy](ADR-INDEX.md#status-taxonomy)); no Decision content is altered.
