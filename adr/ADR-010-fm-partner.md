# ADR-010: Foundation-model partner selection

**Status.** Accepted (2026-05-13)
**Authors.** Farhad Safaei
**Reviewers.** _to fill_
**Tags.** v0.2 §3.5

## Context
Which VLA/FM is partner #1 in the zoo. OpenVLA is the default per plan; CrossFormer and Octo are alternatives.

## Considered alternatives

| # | Option | Source / advocate | Pros | Cons |
|---|--------|-------------------|------|------|
| A | OpenVLA (7B) only in Phase 1 | [Tier-2 #32] | Open weights; best documented; LoRA gives rapid per-task adaptation (10–150 demos, 7.0 GB VRAM) | Single-policy; FM stratum is not heterogeneous-via-cooperation; no cross-embodiment coverage |
| B | OpenVLA + CrossFormer | [Tier-2 #32, #33] | Two-stratum zoo: LoRA-adapted specialist (OpenVLA) + frozen cross-embodiment generalist (CrossFormer); complementary CBF interception interfaces (256-bin token vs. action-chunk) | Higher partner-zoo maintenance; CrossFormer shows no positive transfer from co-training; zero-shot reliability unconfirmed on target tasks |
| C | OpenVLA + Octo + CrossFormer | [Tier-2 #32, #33, Tier-3 #31] | Most diverse FM coverage across three architectures | Highest cost; Octo (note 31) still pending — cannot evaluate third stratum |

## Decision

**Option B — OpenVLA + CrossFormer for Phase 1.** Option C deferred until note 31 (Octo) is complete; note 31 is still _pending_ and cannot be evaluated from the available evidence. Option A is ruled out because a single-FM stratum provides no architectural diversity within the VLA partner class, undermining the §3.5 zoo-diversity rationale.

## Rationale

Notes 32 and 33 establish two architecturally complementary FM roles that together satisfy the §3.5 two-stratum FM requirement. `notes/tier2/32_openvla.md` shows that OpenVLA's LoRA protocol (10–150 demos, inference at 7.0 GB VRAM) is a concrete data/compute budget for the VLA-based specialist stratum, and its 256-bin-per-DoF action tokenization is the natural CBF interception boundary for §6: the safety filter can inspect and clip discretised action tokens without accessing the partner's internal weights, preserving the black-box constraint. `notes/tier2/33_crossformer.md` shows that CrossFormer's masked-modality + embodiment-specific readout token pattern provides a frozen generalist stratum — a single checkpoint operating across 20 embodiment classes at zero-shot without per-task fine-tuning — and its action-chunking cadence (100-action at 20 Hz bimanual; 4-action at 5–15 Hz single-arm) sets the worst-case latency budget for mid-chunk CBF interception. The critical negative result from note 33 — no positive transfer from cross-embodiment co-training — further justifies the black-box partner framing: the ego agent cannot assume the FM partner's policy is cooperation-aware or will respond to cooperative signals. Option A is therefore too narrow (no cross-embodiment partner) and Option C is premature (note 31 pending).

## Evidence basis (links to reading notes)

- [notes/tier2/32_openvla.md] — **resolved** (LoRA protocol, 256-bin action tokenization, throughput limits)
- [notes/tier2/33_crossformer.md] — **resolved** (masked-modality interface, no positive transfer, action-chunking cadence)
- [notes/tier3/refs.bib #31 Octo] — _pending_ (cannot evaluate Option C until this note is complete)

## Consequences

- **Project scope:** Phase 1 partner zoo has two FM strata: OpenVLA LoRA-adapted specialists (one checkpoint per task) and CrossFormer frozen generalist (single zero-shot checkpoint). Heuristic and RL strata (FCP / HARL baselines) are unchanged per §3.5.
- **v0.2 plan sections affected:** §3.5 (partner-zoo construction — FM strata are now specified); §6 (CBF safety filter — must support 256-bin action-token interception for OpenVLA and chunk-level interception for CrossFormer's 100-action / 4-action cadences at the respective control rates); §3.4 (heterogeneity-axis vocabulary must cover VLA embodiment classes with control-rate axis ≤ 15 Hz for OpenVLA).
- **Other ADRs:** ADR governing §6 CBF safety filter must specify interception latency budget ≤ one 4-action chunk at 5–15 Hz (CrossFormer worst case). Any §3.3 online-inference ADR must note that neither OpenVLA nor CrossFormer has a teammate-modelling loop — partner capability must be inferred from behavior observation only.
- **Compute / time / hiring:** LoRA fine-tuning cost per OpenVLA partner: 10–150 target-task demos, single GPU ≤ 7.0 GB VRAM. CrossFormer frozen — zero fine-tuning cost. If Option C is later adopted, Octo infrastructure cost is additive.

## Risks and mitigations

- **OpenVLA throughput (5–15 Hz) may be insufficient for high-frequency manipulation tasks.** → Stress-test each OpenVLA partner checkpoint at degraded-throughput conditions per note 41 (AoI-predictor axis); impose a 15 Hz ceiling on VLA partner control rate in the §3.4 zoo spec; record per-task failure modes systematically before using in safety-critical trials.
- **CrossFormer's zero-shot deployment may produce out-of-distribution behavior on target tasks** (note 33 acknowledges embodiment-misidentification risk; no positive transfer from co-training). → Confine CrossFormer to zero-shot baseline evaluation in Phase-1 validation set; do not use as primary partner in safety-critical manipulation trials without per-task ablation confirming >0% baseline success.
- **Adopting Octo later (Option C upgrade) may require changes to the CBF interception layer** if Octo's action interface (action-chunk size, bin count, control rate) differs from OpenVLA and CrossFormer. → Design the CBF interception contract around a minimal interface (action vector length, DoF count, control rate, max chunk size) rather than model-specific tokenization, so FM partners are drop-in substitutable within the contract.

## Reversibility

**Type-2.** Switching FM partners within the VLA stratum (e.g., swapping CrossFormer for Octo, or adding Octo as a third stratum) requires re-running the Phase-1 validation set against the new partner but does not require redesigning the ego agent's observation/action module, provided the CBF interception contract (action vector + DoF count + control rate + chunk size) is upheld. The Option B → Option C upgrade path is therefore low-cost. Replacing the entire FM stratum (reverting to Option A or adding a non-transformer FM architecture) would require additional §6 interface engineering and is a more expensive reversal but still within Phase-1 scope.

## Validation criteria

By Phase-1 end: the OpenVLA LoRA-adapted partner achieves ≥ 50% task success on ≥ 1 Tier-1 benchmark task across ≥ 3 evaluation runs; the CrossFormer zero-shot partner achieves ≥ 20% task success on the same task across ≥ 3 evaluation runs. Both partners must run through the §6 CBF safety filter without triggering interception-interface errors on ≥ 95% of episodes.

## Open questions deferred to a later ADR

1. **Note 31 (Octo) pending.** Does Octo offer a structurally distinct third FM stratum or overlap with CrossFormer's co-trained generalist role? Option B → C upgrade decision deferred until note 31 is complete.
2. **#verify:** Can CrossFormer's masked-modality interface be applied to bimanual tasks (two separate readout tokens, one per arm) without modifying the frozen checkpoint? No evidence in either manuscript (note 32 or 33).
3. **#design-decision:** CBF interception for CrossFormer's 100-action chunks at 20 Hz — intercept per-chunk before execution (cheaper but coarser granularity) or per-action within the chunk (safer but requires chunk decomposition)? Not resolvable from current notes alone; requires §6 design spike.

## Revision history

- 2026-05-13 status re-classification: status changed from Proposed to **RFC** under the new ADR status taxonomy (see [ADR-INDEX §Status taxonomy](ADR-INDEX.md#status-taxonomy)); Phase-1 lock pending; no Decision content is altered.
- 2026-05-13 lock: promoted to **Accepted (2026-05-13)** under the solo-developer working policy in ADR-INDEX (Option B — OpenVLA + CrossFormer — is the committed Phase-1 FM stratum); no Decision content is altered.
