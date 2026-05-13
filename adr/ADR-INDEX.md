# ADR Index — Heterogeneous Multi-Robot Ad-Hoc Teamwork

## Status taxonomy

| Status | Meaning |
|---|---|
| RFC | Drafted; not yet implementation-binding. |
| Provisional | Implementation may proceed; claims must remain qualified. |
| Accepted | Binding; change only via a superseding ADR. |
| Validated | Accepted and empirically supported by spike or test evidence. |
| Superseded by ADR-NNN | Historical only. |

This replaces the prior binary "Proposed / Accepted" model. All 15
existing ADRs are re-classified as **Provisional** or **RFC** below
based on whether implementation is in progress for them. No ADR
Decision is changed by this re-classification; only the Status label
and §Revision history of each ADR are updated.

| #   | Title                                                              | v0.2 ref           | Lock by    | Status                  |
|-----|--------------------------------------------------------------------|--------------------|------------|-------------------------|
| 001 | Fork ManiSkill2 vs. build standalone benchmark                     | §3.1               | Phase 0    | Provisional             |
| 002 | RL framework — JAX (Mava) vs PyTorch (HARL)                        | §13                | Phase 0    | Provisional             |
| 003 | Communication interface — fixed-format, learned, or both           | §13                | Phase 0    | Provisional             |
| 004 | Safety filter formulation — exp CBF, HO-CBF, MPC, learned          | §6.2 + §13         | Phase 0    | Provisional<sup>†</sup> |
| 005 | Simulator base — Isaac Lab, MuJoCo, PyBullet, ManiSkill            | §3.1, §3.9         | Phase 0    | Provisional             |
| 006 | Partner-policy assumption set — bounded action norm/rate/latency   | §6.1               | Phase 0    | Provisional<sup>†</sup> |
| 007 | Heterogeneity-axis selection (≥20pp gap rule)                      | §3.4               | Phase 0    | RFC                     |
| 008 | HRS bundle composition                                             | §3.7               | Phase 0–1  | RFC                     |
| 009 | Partner-zoo construction (algos / seeds / public-private split)    | §3.5               | Phase 1    | RFC                     |
| 010 | Foundation-model partner selection                                 | §3.5               | Phase 1    | RFC                     |
| 011 | Baseline set — which of B1–B7 ship in Phase 1                      | §3.10 + Phase 1    | Phase 0–1  | RFC                     |
| 012 | License & CLA                                                      | §10                | Phase 0    | Provisional             |
| 013 | Real-robot demo platform & vendor                                  | §5.1               | Phase 0    | RFC                     |
| 014 | Safety violation reporting protocol                                | §6.3               | Phase 1    | Provisional<sup>‡</sup> |
| 015 | Tier-task scope freeze                                             | §3.3 + §7.3        | Phase 1    | RFC                     |

<sup>†</sup> Claims qualified — see [ADR-004 §Open Questions](ADR-004-safety-filter.md#open-questions-deferred-to-a-later-adr)
for the average-loss → per-step bound gap and the related partner
assumption-set qualifications that propagate to ADR-006.

<sup>‡</sup> Reporting contract qualified — depends on the ADR-008
HRS bundle composition and on the PR-A2 conformal-loss instrumentation
that separates prediction-gap loss from constraint-violation signal.

## Lock-by phases (per v0.2 §8 gates)
- **Phase 0** (Month 3): ADRs 001, 002, 003, 004, 005, 006, 007, 012, 013.
- **Phase 0–1** (Month 3–7): ADRs 008, 011.
- **Phase 1** (Month 7): ADRs 009, 010, 014.
- **Phase 1** (Month 12 gate): ADR 015.

## Locking rule
An ADR cannot be promoted to Accepted unless its **Evidence basis**
cites at least one Tier-1 or Tier-2 reading note (or the
customer-discovery synthesis for ADRs 008, 013, 015 where reading is
not the primary input). ADRs without note citations remain RFC until
reading catches up. Promotion from Provisional to Accepted requires
the senior-advisor lock review; promotion from Accepted to Validated
requires the corresponding spike or test evidence cited in the ADR
§Validation criteria.
