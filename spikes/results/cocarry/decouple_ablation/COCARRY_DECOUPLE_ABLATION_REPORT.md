# Co-carry decouple ablation — measurement report

**What this is.** An eval-only, **no-training**, pre-registered Phase-2 ablation (board convening R-2026-06-C, condition 5): is the co-carry base ego's near-perfect success **cooperation-contingent** (it needs the coupled partner actually coordinating) or **coupling-trivial** (it would succeed anyway at K=8000 with the coupling removed or the partner not acting)? It is the second independent line under the singly-sourced `base_robustness_control`, and the downward sibling of the base-probe's upward stiffness sweep. Phase-2, **non-gating**; touches no Phase-1 / M10 work.

**Pre-registration.** `spikes/preregistration/cocarry_decouple_ablation_prereg.json`, locked at the SSH-signed annotated tag `prereg-cocarry-decouple-ablation-2026-07-11` (blob `118a8d3a6366824fe260f8ad8a6012dcc9514804`), cut **before any episode ran**. The driver verified the tag/blob gate, the clean tree, and seed disjointness (block `[71000..71019]`, n = 20, vs 13 prior co-carry seed sets — no overlap) before the first rollout. Fixed and unchanged: predicate `evaluate_cocarry_success` (placed ∧ level ∧ unstressed ∧ static ∧ settled), `fmax_coupling_n = 365.6080997467041`, `stress_measure='coupling'`, `episode_length=320`, goal 0.10 m, tilt 15°.

## Results (recomputed from the committed per-episode JSONL — never in-memory values)

| Cell | Coupling K (N/m) | Partner | success_rate [95% CI] | Classification |
|---|---|---|---|---|
| **C0** coupled anchor | 8000 | matched, acting | **1.000** [1.000, 1.000] | FEASIBLE_TRIVIAL |
| C3 | 4000 | matched, acting | 1.000 [1.000, 1.000] | FEASIBLE_TRIVIAL |
| C3 | 2000 | matched, acting | 1.000 [1.000, 1.000] | FEASIBLE_TRIVIAL |
| C3 | 1000 | matched, acting | 1.000 [1.000, 1.000] | FEASIBLE_TRIVIAL |
| C3 | 500 | matched, acting | 1.000 [1.000, 1.000] | FEASIBLE_TRIVIAL |
| C3 | 100 | matched, acting | 1.000 [1.000, 1.000] | FEASIBLE_TRIVIAL |
| **C1** spring-off (= C3 endpoint) | 0 | matched, acting | **0.000** [0.000, 0.000] | STABLE_INFEASIBLE |
| **C2** limp-but-coupled | 8000 | `partner_ablated_zero` (zero actions, coupling intact) | **0.000** [0.000, 0.000] | STABLE_INFEASIBLE |

**Primary contrast (C0 − C2):** gap = **1.000**, one-sided 95% lower bound = **1.000** (seed-cluster bootstrap, n_boot 10000, 20 seed clusters). C0 anchor = 1.000 ≥ the 0.90 substrate-drift kill bound (20/20; consistent with every prior identical-config observation).

**Verdict, strictly per the pre-committed rule (prereg §decision_rule):**

> **COOPERATION_CONTINGENT** — C0 = 1.000 ≥ 0.90 AND C2 = 0.000 ≤ 0.10 AND gap CI_lower = 1.000 ≥ 0.50; no trivial-solve fired (C1 = 0.000 < 0.90, C2 = 0.000 < 0.90).

Per the rule's meaning: the base ego needs the coupled partner actually coordinating; the `base_robustness_control` is a **valid** cooperation demonstration, and the co-carry `CONFOUNDED_BY_INCUMBENT_BRITTLENESS` verdict **holds**, now hardened by a second independent line. This report makes no call beyond that rule.

## Corroboration and mechanism (reported, not gated)

- **C2 (the primary instrument, limp-but-coupled — distinct from the admission A2 "retracted" anchor)** fails 20/20 with the bar mechanically supported and physics stable: all 20 episodes fail `placed` **and** `level` (tilt p90 59.5° vs the 15° limit; centroid p50 0.11 m vs the 0.10 m radius; coupling stress p90 315 N stays *under* f_max 365.6, and below the 3×f_max artifact bound). The ego alone drags the shared bar to the goal's edge but cannot keep it level without the partner's coordinated lift — exactly the cooperation channel, ablated.
- **C1/C3 dose-response:** success is flat at 1.000 from K=8000 all the way down to K=100 (coupling stress at K=100: p90 8.5 N) and collapses only at K=0 (stress ≡ 0, tilt ≡ 0, centroid p50 0.80 m — the bar is never transported). The coordination channel works through even a very weak spring; only its complete removal breaks the task. Per the prereg, the C1 collapse is the **weaker** evidence (at K=0 the bar is unsupported, so failure is mechanically expected); C2 carries the verdict.
- All 160 episodes finite; no tripwire fired; wall clock 663 s (cap 14400 s); 160 episodes (cap 160).

## Artifacts (immutable; new directory — I8)

| File | SHA-256 |
|---|---|
| `cocarry_decouple_ablation_measurement.json` | `eaf065a2de7e1ed1b99acd0cc46bc7540948231d2c55a38e7725db9ca5341783` |
| `cocarry_decouple_ablation_episodes.jsonl` | `efb82d8ea7c9f76f4d99eacf91d94e1d59c8308e81ddbf663a73ac27051827fd` |
| `cocarry_decouple_ablation_trajectories.json` | `86eb2391425adaa758430be940833bd8c7702b1000d8389656292dad85ea6e96` |
| `REPRO.txt` | `eb1d604682f35bbf68c25e9e043e3476958b278d66a404521ebe63f0b451e97c` |

Repro: `uv run --no-sync python scripts/repro/_cocarry_decouple_ablation.py` (refuses a dirty tree, exit 7, and a prereg blob-SHA mismatch vs the tag, exit 4).

**Governance.** Eval-only; no training / residual / heterogeneity; predicate and f_max frozen; only the coupling varied. Prior `spikes/results/**` immutable (I8); no `SCHEMA_VERSION` bump (I9); no Phase-1 / M10 code touched (I1). Do not merge — founder review.

ADR-026 §Decision 1-2; ADR-026 §Open-questions (coupling stiffness is a task parameter).
