# AS-homo path smoke at the gate regime — SMOKE, not evidence

**Date.** 2026-06-11.
**Author.** Farhad Safaei.
**Status.** Gate-spec precondition (founder addition 1, 2026-06-11). **A rig-health smoke —
explicitly not gate or campaign evidence.** New dated directory (I8).
**Why.** The AS-homo cell (`stage1_pickplace_panda_only_mappo_shared_param`, the prereg's
shared-param baseline condition) had never been exercised at the vectorised regime — every
post-fix run was the hetero cell. The gate spec is not launchable until the homo path is
shown to train, evaluate, and archive end-to-end at the gate regime.

## Protocol

One pass through the **production factory dispatch** (the #215-fixed ordering: factory
constructed first → real homo eval env → factory invoked, training inside) at the gate
regime: N=1024, γ=0.8, **`shaping.settle_alpha=0.5` ON**, filter-off, 1M frames (smoke
budget), then a 10-episode cold eval and archive-integrity checks. Code vintage:
`feat/pbrs-settle-shaping` (PR #219).

## Results — PASS on every smoke criterion

| check | result |
|---|---|
| factory dispatch (homo eval env after GPU enable; training env N=1024 two-panda build) | ✅ end-to-end, no errors |
| partner action_dim resolved per-cell from the homo eval env | ✅ 8 (panda_partner) |
| shaped training health at 1M | ✅ value 0.77, adv_std 0.17, entropy 3.32 — no pathology markers |
| learning signal on the homo path | ✅ grasp_rate 0 → 0.036 and mean_reward 0.064 → 0.161, monotone over 30 update windows — the same early-trajectory shape every healthy hetero cell showed at this budget fraction |
| archive integrity | ✅ config-fingerprinted run_id `893b2374481bab7f`; 30 train + 30 rollout windows in the JSONL; results JSON written |
| wall-clock | 1.4 min / 1M frames (consistent with the measured 11k steps/s) |
| 10-episode cold eval | 0 grasped / 0 placed / 0 success — **expected mid-learning state at 1M** (hetero cells consolidate between ~5M and 20M); recorded for completeness, carries no evidential weight |

Artifacts: `homo_smoke_results.json`, `homo_smoke_893b2374481bab7f.jsonl` (this directory).

## Cross-references

`GATE_REGIME_SPEC_2026-06-11.md` (the precondition this discharges); PR #219 (shaping);
PR #217 (the dispatch-order fix this smoke exercises in production form);
ADR-007 §Stage 1b Rev 17/18; `spikes/preregistration/AS.yaml`.
