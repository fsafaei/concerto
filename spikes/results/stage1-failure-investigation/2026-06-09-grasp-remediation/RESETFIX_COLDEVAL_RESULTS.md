# Reset-fix cold-eval — seed 1 grasps cold with the init fix alone (unconfounded)

**Date.** 2026-06-09.
**Author.** Farhad Safaei.
**Status.** New file in this directory (the snapshot/I8 rule holds — `REMEDIATION_LOG.md` and
`ADDENDUM_reset-state-confound_2026-06-09.md` are untouched). Records the decisive post-#210
single-seed cold-eval pre-committed in `REMEDIATION_LOG.md` §0/§C and the addendum.
**What ran.** AS-hetero, **seed 1**, 1M frames, **fix only** — reset-state init (ADR-007 Rev 16) +
env truncation (Rev 15), with **no reward bridge, no entropy schedule, no curriculum**. Cold eval
(prereg'd init). Run id `add44268f15c0674`; production `TrainedPolicyFactory` path; gitignored
`.local/` driver. The only variable vs the §1 broken-init baseline is the reset-state fix.

## Result — before/after (seed 1)

| metric (seed 1, 1M) | broken init (truncation only) | **fix only (reset-state + truncation)** |
|---|---|---|
| `grasp_rate` max (nonzero windows) | 0.000 (**0**) | **0.040 (301)** |
| `place_rate` / `success_rate` max | ~0.002 (floor) | ~0.003 (floor) |
| `value_mean` (final) | 5.2 | 17.0 |
| `advantage_std` (final) | 0.031 | 0.145 |
| `dist_entropy` (final) | 4.01 | 3.76 |
| **cold eval `ever_grasped`** | **0 / 10** | **1 / 10** |
| episodes in run | 10090 | 10086 |

## Conclusion

The **reset-state fix alone** — no bridge, no entropy, no curriculum — **unblocks grasp emergence on
a seed that was flat 0/10 under the broken init.** Seed 1 went from zero grasp windows (0/301k) and
0/10 cold eval to a peak `grasp_rate` of ~4 % across **301 windows** and **1/10 cold-eval grasps**,
with healthy advantages (0.031 → 0.145). This is a **clean, single-variable, unconfounded** result
(same seed, same RNG-harnessed cube/goal stream; only the reset state differs) and meets the
addendum's pre-committed criterion ("if it grasps cold …"). It directly confirms the reset-state
init was a real, upstream blocker for grasping.

## Caveats (progress, not a solved task)

- **Modest.** ~4 % `grasp_rate`, 1/10 cold eval — comparable to seed-0's ~10 % *under the broken
  init*. Grasping now *emerges* on a previously-flat seed but is not *reliable*. Whether it
  replicates across seeds is the open question (the §B characterization spike).
- **Place/success still at the floor** (~0.003). The reset-state fix does not touch the
  grasp→place→static credit chain; the grasp→place conversion (PB1 territory) remains the open second
  problem and is gated behind reliable grasping. A *success*-gap gate cannot clear while place is
  floored.
- **Confounded arms.** Per `ADDENDUM_reset-state-confound_2026-06-09.md`, the PA1/PA2/PA3 lever
  conclusions were measured under the broken init and stay suspended; this result does not re-rank
  them.

## Archived artifacts (this directory)

- `seed1_coldeval_results.json` — the run's results/signature JSON (seed 1, fix only).
- `add44268f15c0674.jsonl` — the per-step / per-window JSONL (PPO health + grasp/place/success ladder).
- `launch_seed1.log` — the run's launch log (SAPIEN URDF-parse warnings stripped for size).

## Cross-references

- Fix: PR #210 (ADR-007 §Stage-1b Rev 16; ADR-002 2026-06-09 determinism amendment); truncation fix
  Rev 15 (#206).
- `ADDENDUM_reset-state-confound_2026-06-09.md` (the confound this result tests); `REMEDIATION_LOG.md`
  §1 (broken-init baseline) / §C (the pre-committed cold-eval).
