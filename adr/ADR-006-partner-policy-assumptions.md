# ADR-006: Partner-policy assumption set

**Status.** Provisional (dependency update 2026-05-08 in light of ADR-007 rev 3 staged rollout; claims remain qualified — see [ADR-004 §Open Questions](ADR-004-safety-filter.md#open-questions-deferred-to-a-later-adr))
**Authors.** Farhad Safaei
**Reviewers.** _to fill_
**Tags.** v0.2 §6.1, SAFE

## Context
v0.2 §6.1 specifies bounded action norm, bounded action rate, and bounded comm latency as stated assumptions. ADR locks the specific bounds and the empirical-violation reporting requirement.

## Considered alternatives

| # | Option | Source / advocate | Pros | Cons |
|---|--------|-------------------|------|------|
| A | Singh-style conformal abstraction | [Tier-1 #42] | Adapts online to actual partner behaviour | Lacks explicit named bounds |
| B | Explicit bounded-norm + bounded-rate + bounded-latency (v0.2 §6.1) | [Tier-1 #42 + plan-explicit] | Reviewable; reportable; partner-zoo can be profiled against | Must specify numeric bounds per task |
| C | Hybrid: explicit bounds for collision/force, conformal for prediction | [Tier-1 #42 + plan] | Best of both | Implementation complexity |

## Decision
Option C. Numeric bounds in benchmark spec per task; conformal layer handles partner trajectory prediction.

## Rationale
Option C is supported by complementary evidence from all four cited notes. Note 42 (Huriot & Sibai 2025, ICRA) demonstrates that a conformal slack variable λ in CBF constraints, updated via λ_{k+1} = λ_k + η(ε − l_k), provides distribution-free online adaptation to black-box partner trajectory prediction errors — the core adaptive capability Option A alone cannot supply without explicit bound hooks. However, note 42's Table I (gt/noLearn row) shows that even with perfect trajectory predictions, an actuator-tracking gap produced 12 collision frames, establishing that explicit numeric bounds at the actuator command level (Option B's contribution) are a hard prerequisite: the conformal layer adapts within those bounds but cannot compensate for a missing bound specification at the hardware interface. Note 45 (Garg et al. 2024, §7.3) identifies CBF-QP infeasibility under actuator limits as the central open problem in multi-robot safe control, corroborating the necessity of explicit bounds that gate QP feasibility. Notes 39 and 41 supply the constraint-decomposition scaffolding that makes the hybrid viable at scale: Cavorsi et al.'s nested-CBF stacking (eq. 15–16) provides a graceful relaxation template when partner-induced infeasibility occurs, and Ballotta & Talak's Theorem 11 locality result justifies per-agent QP decomposition in §6.2 even under communication delay, without requiring homogeneous robot state spaces. Taken together, explicit numeric bounds govern feasibility and reviewability while the conformal prediction layer governs online adaptation, and neither approach alone satisfies §6.1 under heterogeneous embodiments.

## Evidence basis (links to reading notes)
- [notes/tier1/42_singh_2024.md] — Conformal slack variable λ in CBF constraint (eq. 5) and Theorem 3 ε + o(1) long-term risk bound; Table I gt/noLearn row establishes that the CBF must operate at the actuator command level; open questions on per-step guarantee (#verify) and λ re-init on partner swap (#design-decision) propagate to Open questions below.
- [notes/tier2/45_lindemann_safety_survey.md] — Table 1 establishes no existing method achieves formal safety + dynamics-free + distributed simultaneously; §7.3 names CBF-QP infeasibility under actuator limits as the feasibility challenge §6.2 must solve; distributed CBF constraint-decomposition (§3.1.4–3.1.5) provides the per-agent QP architecture for §6.2.
- [notes/tier2/41_ballotta_talak_2024.md] — Theorem 11 locality result: safety certification requires only within-neighborhood constraint checks, providing theoretical justification for per-agent CBF-QP decomposition in §6.2; AoI-conditioned state predictor pattern applicable when partner communication is delayed or degraded.
- [notes/tier2/39_cavorsi_2022.md] — Degraded-partner budget formula F < N/4 as formal precedent for the tolerable-degraded-partner ceiling in §3.4; nested-CBF hard/soft relaxation pattern (eq. 15–16) as §6 template for CBF stacking under partner-induced infeasibility.

## Consequences
- Project scope: Must enumerate numeric bound values (action norm, action rate, comm latency) per benchmark task in the §6.1 spec; requires partner-zoo profiling runs to measure empirical violation rates against those bounds before benchmark results are frozen; conformal prediction layer (λ update rule, ε and η parameters) must be implemented in §6 alongside the CBF-QP safety filter. **Numeric bounds for the communication axis are now derivable from public 5G-TSN industrial-trial data** surfaced by adrs/international_axis_evidence.md §2.4 — URLLC 1 ms latency at 99.9999% reliability, jitter μs–10 ms, drop 10⁻⁶ to 10⁻² per arXiv 2501.12792 and 3GPP Release 17 — so the long-standing "specific numeric bound values per benchmark task" gap can be closed for comm-related bounds before Phase-1 starts; action-norm and action-rate bounds remain task-specific and must still be enumerated per §6.1. These comm-axis bounds are normatively anchored in the IEEE 802.1Qbv scheduled-traffic guarantees and the 3GPP Release 17 URLLC service class (clause-level standards summary at [`docs/reference/standards.md` §Deterministic networking and 5G-TSN](../docs/reference/standards.md#2-deterministic-networking-and-5g-tsn)), with the 5G-TSN integration model in 3GPP TS 23.501 §5.27 establishing the system-architecture contract under which the URLLC service-class targets apply.
- v0.2 plan sections affected: §6.1 (locked by the numeric bounds this ADR commits to), §6.2 (per-pair actuator-limit budget split depends on the bound format established here), §3.5 (partner-zoo profiling runs and per-partner violation-rate reporting are new Phase-1 deliverables).
- Other ADRs: Depends on ADR-004 (safety filter formulation — QP structure and solver choice must be consistent with the nested-CBF stacking pattern from note 39). The conformal ε parameter may need to be re-derived if ADR-005's simulator choice changes the actuator tracking model. **Precondition for ADR-007 rev 3 Stage 3 PF spike**: per-task numeric bounds must be enumerated for at least one contact-rich benchmark task before the Stage 3 PF spike runs the partner-swap protocol — otherwise λ re-init on partner swap (Risk #3 below) cannot be tested against a defined bound envelope. **Provides input to ADR-007 rev 3 Stage 2 CM spike**: the comm-axis numeric bounds enumerated here become the sweep ranges for the CM spike, so the enumeration must be complete (or at least bracketed at URLLC anchors) by Stage 2 entry.
- Compute / time / hiring: λ update rule is O(1) per timestep — negligible online overhead; partner-zoo profiling adds approximately N_zoo × N_tasks evaluation episodes per Phase-1 results freeze.

## Risks and mitigations
- [risk] Assumption A2 (bounded prediction error E_v, E_d from note 42) may be violated by multi-modal VLA partners (e.g., OpenVLA, note 32) whose trajectory distributions are heavy-tailed or mode-switching, making the conformal bound ε loose and Theorem 3 vacuous in practice.  →  mitigation: Set ε < 0 in the conservative manipulation regime as flagged in note 42 §6 implications; validate per-stratum in the partner zoo before each Phase-1 results freeze and exclude partners whose empirical violation rate exceeds ε until bounds are tightened.
- [risk] Numeric bounds specified per-task may be too coarse for contact-rich sub-phases within a single episode (e.g., free-space approach vs. squeeze/push), causing the conformal layer to over-tighten during free motion or under-tighten during contact.  →  mitigation: Allow per-phase bound switching in the benchmark spec and track violation rates at sub-task granularity in the §6.1 empirical-violation reporting pipeline.
- [risk] λ re-initialization on mid-episode partner swap: the update rule assumes a stationary prediction-error sequence; a sudden partner change violates stationarity and may cause a burst of constraint violations before λ re-adapts (open question from note 42).  →  mitigation: Implement λ reset with a warm-start from the partner-class prior whenever a partner change is detected in the partner-zoo episode protocol; treat as a testable #design-decision in Phase-1 ablation experiments.

## Reversibility
Type-2. The numeric bounds live in the benchmark spec and can be revised without touching safety-filter code; the conformal layer is parameterized by ε and η, which are per-task hyperparameters. If Phase-1 results are published citing specific bound values, revising those bounds requires rerunning partner-zoo profiling (moderate effort) but no architectural changes to the CBF-QP or conformal update pipeline.

## Validation criteria
By Phase-1 end: (a) every partner in the partner-zoo has its empirical CBF-constraint violation rates reported alongside benchmark results; (b) average violation rate ≤ ε per 100-episode block for each partner–task pair (target ε ≤ 0.05 for manipulation tasks); (c) zero catastrophic violations (inter-robot distance < 0.02 m or contact force > 1.5× the stated force limit) in any 100-episode block.

## Open questions deferred to a later ADR
- Can Theorem 3's average-loss bound (note 42) be sharpened to a per-step or high-probability safety guarantee for manipulation tasks where a single constraint violation may cause irreversible contact damage? #verify — defer to safety-layer ADR post Phase-1.
- What modifications are needed when the ego is a manipulator with joint-space dynamics and the CBF encodes joint-torque limits rather than Euclidean distance? #verify — see Morton 2025 (note 44); defer to ADR-004 refinement.
- Does λ require re-initialization on partner swap mid-episode, and what warm-start strategy minimizes the transient violation burst? #design-decision — Phase-1 ablation, **also instrumented by ADR-007 rev 3 Stage 3 PF spike** (the partner-swap protocol provides early empirical data on the burst magnitude).
- Is Assumption A2 (bounded prediction error) satisfiable for a black-box VLA partner whose trajectory may be multi-modal and non-Lipschitz (e.g., OpenVLA, note 32)? #verify — requires per-stratum conformal calibration experiments in Phase-1.

## Revision history
- 2026-05-13 revision: anchors the comm-axis numeric bounds (latency, jitter, drop) in IEEE 802.1Qbv scheduled-traffic guarantees and the 3GPP Release 17 URLLC service class, with 3GPP TS 23.501 §5.27 as the 5G-TSN integration model. No change to the Decision. Ties to the public standards reference at docs/reference/standards.md.
- 2026-05-13 status re-classification: status changed from Proposed to **Provisional** under the new ADR status taxonomy (see [ADR-INDEX §Status taxonomy](ADR-INDEX.md#status-taxonomy)); claims remain qualified by [ADR-004 §Open Questions](ADR-004-safety-filter.md#open-questions-deferred-to-a-later-adr); no Decision content is altered.
