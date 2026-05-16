# ADR Index — Heterogeneous Multi-Robot Ad-Hoc Teamwork

## Status taxonomy

| Status | Meaning |
|---|---|
| RFC | Drafted; not yet implementation-binding. |
| Provisional | Implementation may proceed; claims must remain qualified. |
| Accepted | Binding; change only via a superseding ADR. |
| Validated | Accepted and empirically supported by spike or test evidence. |
| Superseded by ADR-NNN | Historical only. |

While the project is solo, the working policy is to **treat each ADR
that has a written Decision as Accepted as of its lock date** and to
flag any open follow-up work on a per-ADR basis using the footnote
column below. ADRs that genuinely have no Decision yet (no platform
selected, no scope locked) remain RFC. Promotion from Accepted to
Validated is reserved for ADRs that subsequently accrue spike or test
evidence per their own §Validation criteria.

| #   | Title                                                              | v0.2 ref           | Lock by    | Status                       |
|-----|--------------------------------------------------------------------|--------------------|------------|------------------------------|
| 001 | Fork ManiSkill2 vs. build standalone benchmark                     | §3.1               | Phase 0    | Accepted (2026-05-13)        |
| 002 | RL framework — JAX (Mava) vs PyTorch (HARL)                        | §13                | Phase 0    | Accepted (2026-05-13)<sup>g</sup> |
| 003 | Communication interface — fixed-format, learned, or both           | §13                | Phase 0    | Accepted (2026-05-13)        |
| 004 | Safety filter formulation — exp CBF, HO-CBF, MPC, learned          | §6.2 + §13         | Phase 0    | Accepted (2026-05-13)<sup>a</sup> |
| 005 | Simulator base — Isaac Lab, MuJoCo, PyBullet, ManiSkill            | §3.1, §3.9         | Phase 0    | Accepted (2026-05-13)        |
| 006 | Partner-policy assumption set — bounded action norm/rate/latency   | §6.1               | Phase 0    | Accepted (2026-05-13)<sup>a</sup> |
| 007 | Heterogeneity-axis selection (≥20pp gap rule)                      | §3.4               | Phase 0    | Accepted (2026-05-13)<sup>b</sup> |
| 008 | HRS bundle composition                                             | §3.7               | Phase 0–1  | Accepted (2026-05-13)<sup>c</sup> |
| 009 | Partner-zoo construction (algos / seeds / public-private split)    | §3.5               | Phase 1    | Accepted (2026-05-13)        |
| 010 | Foundation-model partner selection                                 | §3.5               | Phase 1    | Accepted (2026-05-13)        |
| 011 | Baseline set — which of B1–B7 ship in Phase 1                      | §3.10 + Phase 1    | Phase 0–1  | Accepted (2026-05-13)        |
| 012 | License & CLA                                                      | §10                | Phase 0    | Accepted (2026-05-13)<sup>d</sup> |
| 013 | Real-robot demo platform & vendor                                  | §5.1               | Phase 0    | RFC<sup>e</sup>              |
| 014 | Safety violation reporting protocol                                | §6.3               | Phase 1    | Accepted (2026-05-13)<sup>f</sup> |
| 015 | Tier-task scope freeze                                             | §3.3 + §7.3        | Phase 1    | RFC<sup>e</sup>              |

### Open work flags

<sup>a</sup> Per-step safety bound under heterogeneous action spaces
is not yet established — the conformal layer gives an average-loss
bound (Huriot & Sibai 2025 Theorem 3); sharpening to per-step is
gated by the Stage-1 AS spike and the follow-up safety-stack refactor.
The partner-swap reset value
`concerto.safety.conformal.reset_on_partner_swap(..., lambda_safe=0.0, ...)`
ships with a Phase-0 placeholder default; the derived form
`lambda_safe(bounds, predictor_error_bound, dt, pair_geometry)` that
would preserve QP feasibility under the worst-case bounded prediction
error is deferred alongside the per-step bound (external-review P0-4,
2026-05-16). See
[ADR-004 §Open questions](ADR-004-safety-filter.md#open-questions-deferred-to-a-later-adr)
and [ADR-006 §Open questions](ADR-006-partner-policy-assumptions.md#open-questions-deferred-to-a-later-adr).

<sup>b</sup> The staged Phase-0 spike protocol (Stage 1: AS + OM;
Stage 2: CR + CM; Stage 3: PF + SA) is committed but the spikes have
not yet been run; the ≥20 pp gate has not yet been measured for any
axis. Promotion to **Validated** requires the per-axis spike evidence
named in [ADR-007 §Validation criteria](ADR-007-heterogeneity-axis-selection.md#validation-criteria).
Revision 4 (2026-05-15) splits Stage 1 into Stage 1a (Phase-0
rig-validation on the MPE stand-in; no ≥20 pp measurement) and
Stage 1b (Phase-1 real-env science evaluation; the ≥20 pp gate);
Stage 1b's trigger guardrail pins it to the first Phase-1 milestone,
before any Stage-2 spike, no later than 4 weeks after the Month-3
lock review (see
[ADR-007 §Stage 1a / §Stage 1b](ADR-007-heterogeneity-axis-selection.md#stage-1a--rig-validation-phase-0-mpe-stand-in)).

<sup>c</sup> The HRS bundle's third axis (CM × PF × CR vs the
fallback latency × drop × degraded-partner formula) depends on which
axes survive the ADR-007 staged spikes. Default bundle is locked at
**CM × PF × CR**; fallback rules are written into the Decision so the
ADR self-resolves once Stage 2 / Stage 3 evidence lands.

<sup>d</sup> Apache 2.0 license is locked; CLA bot wiring is not yet
in place. See [ADR-012 §Open questions](ADR-012-license.md#open-questions-deferred-to-a-later-adr).

<sup>e</sup> No Decision yet — ADR is genuinely pending. ADR-013
waits on hardware-partner outreach; ADR-015 waits on Phase-0
customer-discovery synthesis. These remain RFC until their Decision
sections can be filled.

<sup>f</sup> Reporting contract is qualified by ADR-008 HRS bundle
composition (footnote c) and by the PR-A2 conformal-loss
instrumentation that separates prediction-gap loss from
constraint-violation signal. Table content is stable; per-axis row
counts firm up once ADR-008 lands its surviving-axis set.

<sup>g</sup> §Risks #1 was amended on 2026-05-15: the canonical
empirical-guarantee statistic is now a one-sided slope test at
α = 0.05 (`concerto.training.empirical_guarantee.assert_positive_learning_slope`),
replacing the legacy moving-window-of-10 non-decreasing-fraction
wording. The amendment also surfaces the trainer-side partner-freeze
gate (`chamber.benchmarks.ego_ppo_trainer._assert_partner_is_frozen`,
ADR-009 §Consequences; plan/05 §6 #3). See [ADR-002 §Revision history
(2026-05-15)](ADR-002-rl-framework.md#revision-history).

## Lock-by phases (per v0.2 §8 gates)
- **Phase 0** (Month 3): ADRs 001, 002, 003, 004, 005, 006, 007, 012, 013.
- **Phase 0–1** (Month 3–7): ADRs 008, 011.
- **Phase 1** (Month 7): ADRs 009, 010, 014.
- **Phase 1** (Month 12 gate): ADR 015.

## Locking rule

An ADR cannot be promoted to Accepted unless its **Evidence basis**
cites at least one Tier-1 or Tier-2 reading note (or the
customer-discovery synthesis for ADRs 008, 013, 015 where reading is
not the primary input). Promotion from Accepted to Validated requires
the corresponding spike or test evidence cited in the ADR §Validation
criteria. While the project is solo, the lock decision rests with the
project lead; once external reviewers / advisors are onboarded, the
gate becomes a review checkpoint.
