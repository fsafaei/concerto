# ADR Index — Heterogeneous Multi-Robot Ad-Hoc Teamwork

| #   | Title                                                              | v0.2 ref           | Lock by    | Status   |
|-----|--------------------------------------------------------------------|--------------------|------------|----------|
| 001 | Fork ManiSkill2 vs. build standalone benchmark                     | §3.1               | Phase 0    | Proposed |
| 002 | RL framework — JAX (Mava) vs PyTorch (HARL)                        | §13                | Phase 0    | Proposed |
| 003 | Communication interface — fixed-format, learned, or both           | §13                | Phase 0    | Proposed |
| 004 | Safety filter formulation — exp CBF, HO-CBF, MPC, learned          | §6.2 + §13         | Phase 0    | Proposed |
| 005 | Simulator base — Isaac Lab, MuJoCo, PyBullet, ManiSkill            | §3.1, §3.9         | Phase 0    | Proposed |
| 006 | Partner-policy assumption set — bounded action norm/rate/latency   | §6.1               | Phase 0    | Proposed |
| 007 | Heterogeneity-axis selection (≥20pp gap rule)                      | §3.4               | Phase 0    | Proposed |
| 008 | HRS bundle composition                                             | §3.7               | Phase 0–1  | Proposed |
| 009 | Partner-zoo construction (algos / seeds / public-private split)    | §3.5               | Phase 1    | Proposed |
| 010 | Foundation-model partner selection                                 | §3.5               | Phase 1    | Proposed |
| 011 | Baseline set — which of B1–B7 ship in Phase 1                      | §3.10 + Phase 1    | Phase 0–1  | Proposed |
| 012 | License & CLA                                                      | §10                | Phase 0    | Proposed |
| 013 | Real-robot demo platform & vendor                                  | §5.1               | Phase 0    | Proposed |
| 014 | Safety violation reporting protocol                                | §6.3               | Phase 1    | Proposed |
| 015 | Tier-task scope freeze                                             | §3.3 + §7.3        | Phase 1    | Proposed |

## Status legend
- **Proposed** — drafted; not yet reviewed.
- **Accepted (YYYY-MM-DD)** — locked; immutable; supersede only by a new ADR.
- **Deferred** — needs more reading or discovery before decision.
- **Superseded by ADR-MMM** — the named ADR replaces this one.

## Lock-by phases (per v0.2 §8 gates)
- **Phase 0** (Month 3): ADRs 001, 002, 003, 004, 005, 006, 007, 012, 013.
- **Phase 0–1** (Month 3–7): ADRs 008, 011.
- **Phase 1** (Month 7): ADRs 009, 010, 014.
- **Phase 1** (Month 12 gate): ADR 015.

## Locking rule
An ADR cannot be Accepted unless its **Evidence basis** cites at least
one Tier-1 or Tier-2 reading note (or the customer-discovery synthesis
for ADRs 008, 013, 015 where reading is not the primary input). ADRs
without note citations are Deferred until reading catches up.
