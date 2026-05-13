# ADR-014: Safety violation reporting protocol

**Status.** Accepted (2026-05-13) (dependency update 2026-05-08 in light of ADR-007 rev 3 staged rollout)
**Open work.** Per-axis row counts firm up once ADR-008 lands its surviving-axis set, and the prediction-gap / constraint-violation split is tracked under PR-A2; see [ADR-INDEX footnote f](ADR-INDEX.md#open-work-flags).
**Authors.** Farhad Safaei
**Reviewers.** _to fill_
**Tags.** v0.2 §6.3

## Context
v0.2 §6.3 requires per-assumption empirical violation rates plus per-condition safety violation rates plus conservativeness gap vs. oracle CBF. ADR locks the table format and reporting cadence. The conformal CBF architecture (v0.2 §6, seeded by note 42) generates three structurally distinct safety metrics—assumption-level, condition-level, and conservativeness-level—that do not collapse into a single table without information loss.

Adjacent to ISO 10218-2:2025, the machine functional-safety stack comprises ISO 13849-1:2023 (safety-related parts of control systems: performance levels PL a..e and the categorical structure), IEC 62061:2021 (machinery functional safety: SIL claims for machinery control systems), and IEC 61508-1:2010 (the cross-sector E/E/PE functional safety standard that defines SIL 1..4 and is the conceptual anchor for both ISO 13849 PL and IEC 62061 SIL). When ADR-007 Open Q #4 resolves toward decomposing the safety axis into per-vendor force-limit compliance vs SIL/PL, these three standards become load-bearing for the per-vendor variance reporting (Table 2 per-condition row, Table 3 conservativeness-gap row).

## Considered alternatives

| # | Option | Source / advocate | Pros | Cons |
|---|--------|-------------------|------|------|
| A | Plan §6.3 default (3-table report per condition) | [plan; note 42 Table I structure; note 45 Table 1 structure] | Preserves per-assumption / per-condition / gap breakdown that §6.3 requires; matches Huriot & Sibai's own evaluation decomposition | Verbose in the main paper |
| B | Single combined table + appendix | [paper-friendly] | Cleaner main-paper presentation | Merges assumption-violation rates with condition-violation rates, hiding the theory/practice distinction that note 45 §7.3 identifies as central; reduces reviewable safety surface |
| C | Streamed per-episode dashboard + paper summary | [infrastructure] | Maximally reviewable; enables online monitoring during experiments | Substantial implementation cost; neither note 42 nor note 45 requires or demonstrates this format; adds infrastructure not budgeted in Phase 1 |

## Decision
**Option A — Plan §6.3 default: 3-table safety report per condition, submitted with every Phase-1 result.**

Table 1: per-assumption empirical violation rates (one row per CDT assumption A1–A3 from note 42 §III).
Table 2: per-condition safety violation rates (one row per experimental condition — predictor type × conformal mode, mirroring note 42 Table I).
Table 3: conservativeness gap vs. oracle CBF (conformal λ mean and variance vs. gt/noLearn baseline from note 42 Table I).

Aggregate metrics across seeds use rliable-style robust statistics (Agarwal et al. 2021): interquartile mean, optimality gap, and bootstrap performance profiles, in addition to mean ± 95% bootstrap CI. The minimum-seed count per cell is the figure committed in `docs/reference/evaluation.md` §3.1. This is the explicit avoidance of the reporting anti-patterns catalogued by Henderson et al. 2018.

## Rationale
The conformal CBF framework adopted from note 42 (Huriot & Sibai 2025) requires per-step tracking of the loss l_k — defined as the worst-case CBF-constraint gap between the conformal and ground-truth constraints across all agent pairs and time steps (§IV.A) — to verify Theorem 3's ε + o(1) average-loss bound. This directly mandates the per-condition (predictor × conformal-mode) decomposition of Option A: note 42's own Table I already structures results exactly this way, with rows gt/noLearn, gt/Learn, pred/noLearn, pred/Learn. Note 45 (Garg et al. 2024) demonstrates that multi-robot safety reporting requires orthogonal columns for Safety Theory (formal guarantee satisfaction) and Safety Practice (empirical violation rates), with the Known-Dynamics axis as a third dimension (Table 1, §2.4); the three-table structure in §6.3 provides exactly this orthogonal decomposition without conflation. Option B's single combined table would merge per-assumption violation rates with per-condition violation rates, erasing the theory/practice distinction that note 45 §7.3 identifies as the central unresolved gap in the field. Option C's streaming dashboard introduces implementation overhead that neither paper requires and that Phase 1's timeline cannot absorb.

## Evidence basis (links to reading notes)
- [notes/tier1/42_singh_2024.md] — Theorem 3 risk bound (ε + o(1) on average loss l_k) and Table I per-condition decomposition (gt/noLearn, gt/Learn, pred/noLearn, pred/Learn) directly motivate the per-condition structure of Tables 2 and 3; Table I gt/noLearn row provides the oracle CBF baseline for the conservativeness-gap metric in Table 3.
- [notes/tier2/45_lindemann_safety_survey.md] — Table 1 (Safety Theory × Safety Practice × Known Dynamics × Distributed Policy) and §7.3 open problem (CBF-QP infeasibility under actuator limits) establish that three orthogonal safety dimensions must be reported separately; adopted as the scaffold for the three-table structure.
- ISO 13849-1:2023 — PL structure (Performance Levels `a`..`e`) and the categorical architecture of safety-related parts of control systems; load-bearing for the per-vendor SIL/PL pair reported in Table 2 if ADR-007 Open Q #4 decomposes the safety axis.
- IEC 62061:2021 — machinery SIL framework for safety-related electrical/electronic/programmable-electronic control systems; the parallel claim path to ISO 13849-1's PL grading and the source of the SIL labels in Table 2 vendor rows.
- IEC 61508-1:2010 — SIL 1..4 cross-sector functional-safety anchor for E/E/PE systems; the conceptual baseline that both ISO 13849 PL and IEC 62061 SIL grade against and that fixes the semantics of the Table 1 "ISO 10218-2:2025 SIL/PL precondition satisfied" assumption row.
- Agarwal et al. (2021 NeurIPS) — `rliable` library and the robust aggregate-metrics protocol (interquartile mean, optimality gap, performance profiles) adopted verbatim for Table 2 cross-seed aggregation.
- Henderson et al. (2018 AAAI) — RL reproducibility anti-pattern catalogue (single-seed bar charts, undisclosed best-of-N, mean returns without CI, re-implemented baselines) that the rliable contract on Table 2 is explicitly written to refuse.

## Consequences
- **Project scope:** Every Phase-1 experiment must produce all three safety tables before a result is reportable. Conditions that cannot generate an oracle CBF baseline (e.g., VLA partners with unknown internal dynamics) must substitute the gt/noLearn configuration from note 42 as the oracle proxy.
- **v0.2 plan sections affected:** §6.3 locked to 3-table format; §6.2's per-pair CBF budget split must generate per-pair l_k data that populates Table 2 rows; §3.8 baseline experiments must run under matching conditions to produce the oracle CBF comparison column in Table 3.
- **Other ADRs:** ADR-006 (partner-policy assumption set) determines which assumptions populate Table 1 rows — the two ADRs must use consistent assumption labels. ADR-007 (heterogeneity-axis selection) determines the condition rows in Table 2 — condition labels must be shared. **ADR-007 rev 3 dependency**: the Stage 3 SA spike outcome determines whether the safety axis stays as a single §3.4 axis or decomposes into (a) per-vendor force-limit compliance (ISO/TS 15066 force-pressure tables) and (b) per-vendor functional-safety SIL/PL ratings (ISO 10218-2:2025 certification-level variance) per ADR-007 Open Q #4. **If safety decomposes, Table 2 rows must split by vendor-compliance level** — i.e., one row per (predictor type × conformal mode × vendor-compliance pair) — and Table 1 must add an A4 row for "ISO 10218-2:2025 SIL/PL precondition satisfied" alongside the existing A1–A3 from CDT. **Regulatory note**: ISO/TS 15066 has been absorbed into ISO 10218-2:2025 since v0.2 was drafted (per adrs/international_axis_evidence.md §2.6); the assumption labels in Table 1 should reference ISO 10218-2:2025 directly when locking, not the superseded ISO/TS 15066:2016.
- **Compute / time / hiring:** Each condition requires an oracle CBF run (gt/noLearn variant) roughly doubling per-condition wall-clock time. No new hiring implied; budgeted within Phase-1 experiment allocation.

## Risks and mitigations
- **[Oracle CBF infeasibility for contact-rich conditions]** The conservativeness-gap metric (Table 3) requires an oracle CBF that uses ground-truth partner trajectories; for manipulation tasks with contact dynamics, the QP may be infeasible under torque limits (note 45 §7.3). → Mitigation: adopt the gt/noLearn configuration from note 42 Table I as the oracle proxy (pre-trained predictor with λ = η = 0) — this is experimentally demonstrated in note 42's own evaluation and avoids the analytical oracle requirement.
- **[Condition explosion as task space grows]** With 6 tasks × 2+ partner types × 3+ CBF configurations, the 3-table format could generate dozens of sub-tables, making the paper unreadable. → Mitigation: cap reported conditions at the §3.2 task-lattice primary cells (6 tasks × 2 partner archetypes = 12 conditions maximum); aggregate across seeds within each condition; full per-condition tables go to supplementary material.

## Reversibility
**Type-2** (two-way door). The 3-table format is a reporting convention, not a code or architectural commitment. Switching to Option B (single combined table) or adding Option C (streaming dashboard) requires only experiment re-analysis, template revision, and supplementary reorganization — no system redesign. The sole Type-1 element is if the venue paper locks the table format at camera-ready submission, after which reverting requires editorial negotiation; that risk is resolved at submission time, not Phase-1 design time.

## Validation criteria
By Phase-1 end: (1) all reported results include all three safety tables with no _to fill_ cells; (2) per-assumption empirical violation rates in Table 1 are at or below the conformal ε threshold for ≥ 90% of experimental conditions; (3) conservativeness gap in Table 3 does not exceed 2× the mean λ slack across conditions, confirming the conformal filter is not excessively conservative.

## Open questions deferred to a later ADR
- Can Theorem 3's average-loss bound (note 42) be sharpened to a per-step or high-probability safety guarantee? If yes, Table 2's per-condition violation rates become a stronger safety claim — carry to ADR-006 for assumption-set locking. #verify
- Does λ re-initialization protocol on partner swap (mid-episode) affect per-episode violation rates in Table 2, and should per-swap boundary periods be excluded from reporting? #design-decision — carry to ADR-006 — *also instrumented by ADR-007 rev 3 Stage 3 PF spike's partner-swap protocol*.
- (#design-decision, from ADR-007 rev 3 Open Q #4) If the safety axis decomposes into force-limit compliance and SIL/PL ratings, must Table 2 add a vendor-compliance dimension (one row per vendor-pair × predictor × conformal mode) or is reporting at the worst-case per-pair level sufficient? Resolution requires ADR-007 Stage 3 SA spike outcome plus a determination of whether the venue paper's safety claim is per-vendor or worst-case.

## Revision history
- 2026-05-13 revision: backfills the machine functional-safety stack (ISO 13849-1:2023, IEC 62061:2021, IEC 61508-1:2010) in Context and Evidence basis alongside the existing ISO 10218-2:2025 anchor, and pins Table 2 aggregate metrics to rliable-style robust statistics (Agarwal et al. 2021) with Henderson et al. 2018 as the anti-pattern catalogue. The Decision is unchanged at the three-table structure but is now explicitly typed in terms of rliable metrics for Table 2. Mirrors the rliable contract into `docs/reference/evaluation.md` §3.2 and the SA row of `docs/reference/standards.md` §2.3.
- 2026-05-13 status re-classification: status changed from Proposed to **Provisional** under the new ADR status taxonomy (see [ADR-INDEX §Status taxonomy](ADR-INDEX.md#status-taxonomy)); the reporting contract remains qualified by ADR-008 HRS bundle composition and by PR-A2 conformal-loss instrumentation that separates prediction-gap loss from constraint-violation signal; no Decision content is altered.
- 2026-05-13 lock: promoted to **Accepted (2026-05-13)** under the solo-developer working policy in ADR-INDEX; the per-axis row counts and the prediction-gap / constraint-violation split remain flagged as Open work; no Decision content is altered.
