# Why conformal CBFs?

The exponential CBF-QP backbone (Wang–Ames–Egerstedt 2017) provides
instantaneous safety guarantees but requires tight a-priori bounds on
partner behaviour. In heterogeneous ad-hoc teamwork those bounds are
unknown at deployment time.

The conformal overlay (Huriot–Sibai 2025) relaxes this requirement: it
maintains a running average-loss guarantee (Theorem 3) with a
self-tuning slack variable λ that widens when the partner's actions
are worse than expected and tightens when they are better. The OSCBF
(Morton–Pavone 2025) resolves the multi-arm coupling into tractable
per-arm QPs.

The key open theoretical question is whether the Huriot–Sibai
average-loss bound can be sharpened to a per-step bound. This is not
required for Phase 0 correctness but would strengthen the safety claim.
See ADR-004 §Open questions.
