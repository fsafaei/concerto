# ADR-009: Partner-zoo construction

**Status.** Proposed (dependency update 2026-05-08 in light of ADR-007 rev 3 staged rollout)
**Authors.** _to fill_
**Reviewers.** _to fill_
**Tags.** v0.2 §3.5

## Context
v0.2 §3.5 specifies ≥4 algorithm classes, ≥3 seeds, ≥2 architecture variants, ≥2 obs-conditioning modes, public/private 70/30 split, FM partners from Phase 1.

## Considered alternatives

| # | Option | Source / advocate | Pros | Cons |
|---|--------|-------------------|------|------|
| A | Plan v0.2 §3.5 default | [plan] | Comprehensive | High implementation cost |
| B | Reduced — 3 algos × 3 seeds; FM from Phase 2 | [budget-driven] | Cheaper | Weaker zero-shot claim |
| C | Plan default + FCP/MEP-style population diversity | [Tier-2 #7, #10, #11] | Strongest diversity argument | Highest cost |

## Decision
**Option C — Plan v0.2 §3.5 default enhanced with FCP/MEP-style population diversity.**

Construct the partner zoo along three axes (policy class: heuristic / RL / VLA; random seed; skill-level checkpoint: init / 50%-reward / converged), with MEP Population Entropy as the admission criterion per policy-class stratum and frozen partner parameters during ego-agent training. The VLA stratum (OpenVLA LoRA-adapted) is integrated in Phase 1, not deferred to Phase 2.

## Rationale

FCP (notes/tier2/07_fcp.md) establishes the zoo-construction recipe: two-dimensional diversity via seed variation and temporal checkpoints at three skill levels, with N=32×3 partners shown sufficient and checkpoint diversity strictly dominating architectural diversity in Table 1 ablation. MEP (notes/tier2/10_mep.md) supplies the admission criterion: the two-level Population Entropy bonus (pairwise across zoo members + per-member individual entropy) prevents population collapse and ranks candidates for inclusion; the prioritized-sampling schedule ensures the ego agent trains proportionally across all diversity modes, applied per policy-class stratum because partners have incompatible action spaces. HARL (notes/tier1/25_harl.md) fills the learning-based stratum: frozen HAPPO and HATD3 checkpoints at three skill levels are the natural RL-tier entries, and §5.2's empirical result — MAPPO fails on 17-agent Humanoid while HAPPO succeeds — confirms that per-agent separate parameters are required throughout the zoo, ruling out index-conditioned shared-parameter variants. OpenVLA (notes/tier2/32_openvla.md) quantifies the VLA stratum: LoRA fine-tuning (10–150 demos, single consumer GPU at 7 GB VRAM) sets the per-partner data and compute ceiling, and the 256-bin-per-DoF action-discretization interface defines the natural boundary at which the CBF safety filter (v0.2 §6) can inspect and clip partner commands without accessing internal weights. Together, the four notes make Option C — plan defaults plus FCP/MEP construction discipline — clearly superior to Option B (weaker zero-shot claim, no FM until Phase 2) and to bare Option A (no formal diversity criterion for zoo admission).

## Evidence basis (links to reading notes)
- [notes/tier2/07_fcp.md] — **resolved** (Strouse 2021 NeurIPS FCP)
- [notes/tier2/10_mep.md] — **resolved** (Zhao 2023 AAAI MEP)
- [notes/tier1/25_harl.md] — **resolved** (Zhong 2024 JMLR HARL)
- [notes/tier2/32_openvla.md] — **resolved** (Kim 2024 CoRL OpenVLA)

## Consequences
- Project scope: Zoo construction becomes a formal three-axis pipeline (policy class × seed × checkpoint) with an entropy-based admission filter; partner count is ≥ 3 strata × 16 seeds × 3 checkpoints = 144 total, more principled and reproducible than ad-hoc selection.
- v0.2 plan sections affected: §3.5 (partner zoo composition and training procedure), §3.4 (separate-parameter design confirmed), §3.8 (HAPPO as B5 on-policy, HATD3 as B5 off-policy), §6 (CBF interception at 256-bin OpenVLA action-token boundary).
- Other ADRs: touches ADR-005 (safety-layer architecture — CBF interception point at action-token boundary); touches ADR-007 (baseline design — HAPPO/HATD3 as B5); should be read alongside ADR-010 (ego-agent architecture) to confirm the frozen-partner assumption is satisfied end-to-end. **Precondition for ADR-007 rev 3 Stage 3 PF spike**: a 3-partner *draft* zoo (one heuristic + one frozen MAPPO checkpoint + one frozen RL checkpoint, e.g. HAPPO at 50%-reward) must be available at Stage 3 entry — i.e., approximately 4–6 weeks into Phase 0 once Stage 2 has cleared its gate. The draft zoo is *not* the full Option C zoo (3 strata × 16 seeds × 3 checkpoints = 144 partners targeted for Phase 1) — it is a minimum viable zoo that lets the PF spike measure trained-with vs frozen-novel ≥20pp gap and instrument the partner-swap λ-reset transient for ADR-006's open question. Full zoo construction begins in Phase 1 after ADR-007 Stage 3 has cleared and partner familiarity is confirmed as a surviving axis; if Stage 3 PF fails the ≥20pp gate, full zoo size and entropy threshold are revisited downward.
- Compute / time / hiring: One LoRA fine-tuning run per VLA partner (10–150 demos, single GPU, ~hours); HAPPO/HATD3 training adds ≈8% wall-time overhead vs. MAPPO (HARL §5.7 Table 3); MEP entropy computation requires training-time access to all zoo members — consistent with build-time constraint. **Draft-zoo scoping for ADR-007 Stage 3**: the 3-partner Stage-3 draft does not require LoRA fine-tuning, MEP entropy filtering, or the public/private 70/30 split — those are full-zoo Phase-1 properties. The draft's only requirement is that all three partners expose the ADR-003 fixed-format channel and accept frozen-parameter inference; estimated 1–2 person-weeks to assemble (heuristic is scripted; the frozen MAPPO and frozen HAPPO/HATD3 checkpoints are by-products of ADR-002 / ADR-011 baseline training that is already happening).

## Risks and mitigations
- **Generalization of HARL-trained checkpoints**: Frozen HAPPO/HATD3 partners were jointly trained with a specific partner; zero-shot generalization to the B0 ego agent is not guaranteed. → Mitigation: Run a generalization probe — evaluate each frozen HARL checkpoint as a zoo partner against a novel ego agent before admission; flag as #verify (from notes/tier1/25_harl.md open questions).
- **Population Entropy inapplicable globally across incompatible action spaces**: MEP's entropy bonus is not directly computable across heuristic/RL/VLA partners with incompatible action spaces. → Mitigation: Apply entropy criterion per stratum using normalized trajectory-embedding distances (e.g., state-visitation histograms projected to a shared observation space); document the per-stratum metric explicitly in the zoo-construction spec.
- **OpenVLA throughput mismatch with CBF control rate**: OpenVLA runs at 5–15 Hz; manipulator CBF operates at 500 Hz; the gap could cause safety-layer saturation or stale-action collisions. → Mitigation: Stress-test the VLA stratum at minimum throughput in an isolated CBF harness before full zoo integration; link to note 41 (Ballotta & Talak AoI predictor) for comm-delay compensation in degraded-throughput conditions.
- **Partner-count sufficiency for heterogeneous setting**: FCP's N=32×3 is validated for a homogeneous two-agent gridworld; our heterogeneous zoo may need more partners per stratum before the ego agent plateaus. → Mitigation: Start with N=16 per stratum and monitor ego-agent generalization on a held-out validation partner set; increase N if success rate has not plateaued after 5M ego-training steps.

## Reversibility
**Type-2** — zoo construction parameters (N per stratum, entropy threshold, checkpoint tiers) are tunable hyperparameters that can be adjusted without discarding trained artifacts. The three-stratum (heuristic / RL / VLA) policy-class structure is partially Type-1: abandoning the VLA stratum after Phase 1 LoRA investments would waste fine-tuning runs. Exit ramp: if LoRA fine-tuning budget cannot be met in Phase 1, defer VLA stratum to Phase 2 and substitute a scripted-behavior heuristic as the third stratum placeholder.

## Validation criteria
By **Phase-0 Stage 3 entry** (ADR-007 rev 3 PF spike precondition):
- A 3-partner draft zoo exists with one scripted-heuristic, one frozen MAPPO checkpoint, and one frozen HAPPO or HATD3 checkpoint, all exposing the ADR-003 fixed-format channel and accepting frozen-parameter inference.

By Phase-1 end:
- Zoo contains ≥4 algorithm classes (at minimum: scripted heuristic, HAPPO, HATD3, OpenVLA-LoRA).
- Each learning-based class has ≥3 random seeds and ≥3 checkpoint tiers (init / 50%-reward / converged).
- Per-stratum Population Entropy of the submitted zoo is ≥ that of an equivalent FCP-only (seed + checkpoint, no entropy filter) population of the same size.
- At least one VLA partner (OpenVLA LoRA-adapted, target task) passes the isolated CBF-harness stress-test at minimum throughput (5 Hz).
- Public/private split is 70/30 as specified in §3.5.

## Open questions deferred to a later ADR
- #verify (notes/tier1/25_harl.md): Can a frozen-partner HAPPO setup preserve the monotonic-improvement guarantee for the ego agent when partner updates are suppressed? Theorem 7 requires all agents to update.
- #design-decision (notes/tier1/25_harl.md): Is HATD3 or HAPPO preferred as the primary B5 baseline for the bimanual manipulation target? HATD3 has better sample efficiency on continuous tasks but complicates AHT adaptation due to replay buffer.
- #verify (notes/tier1/25_harl.md): Confirm whether Bi-DexterousHands evaluation uses two identical Shadow Hands (homogeneous) or distinct embodiments; if homogeneous, HARL's result does not directly represent the mobile-base + fixed-arm setting.
- #verify (notes/tier2/32_openvla.md): Confirm whether the 256-bin action-token boundary is accessible without partner-internal instrumentation in a frozen LoRA checkpoint, or whether a wrapper inference harness is required for CBF interception.
