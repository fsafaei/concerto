# Reset-fix characterization spike — 3-seed before/after under the corrected init

**Date.** 2026-06-10.
**Author.** Farhad Safaei.
**Status.** New file in this directory (I8: `REMEDIATION_LOG.md`, the addendum, and
`RESETFIX_COLDEVAL_RESULTS.md` are untouched). Completes the §B characterization pre-committed in
the reset-fix cold-eval plan: seeds 0 and 2 at 1M frames, **fix only** (reset-state init Rev 16 +
truncation Rev 15; no bridge, no entropy schedule, no curriculum), cold eval — joining the seed-1
run already recorded in `RESETFIX_COLDEVAL_RESULTS.md`. Single-variable vs the §1 broken-init
triplet: only the reset state differs; same seeds, same production `TrainedPolicyFactory` path,
same config.
**One question.** Does grasp emergence *replicate* across seeds under the corrected init, or was
seed 1 a fluke?

## The 3-seed before/after

| seed | broken init: grasp max (nonzero win) / cold eval | **fix only: grasp max (nonzero win) / cold eval** | fix only: place / success max | value_mean | adv_std |
|---|---|---|---|---|---|
| 0 | 0.098 (57) / 1/10 | **0.061 (324) / 0/10** | 0.003 / 0.003 | 15.1 | 0.234 |
| 1 | 0.000 (0) / 0/10 | **0.040 (301) / 1/10** | 0.003 / 0.003 | 17.0 | 0.145 |
| 2 | 0.002 (3 ≈ noise) / 0/10 | **0.034 (149) / 0/10** | 0.003 / 0.003 | 14.3 | 0.168 |

Run ids (fix-only): seed 0 `4193162f5e313d7f`, seed 1 `add44268f15c0674` (prior file), seed 2
`93d0b85c9ceeb804`. PPO health is sane on all three (advantages alive 0.145–0.234; entropy
3.8–3.9; ~10.1k episodes per run).

## Reading — split verdict, stated precisely

1. **Training-time grasp emergence replicates 3/3.** Every seed now grasps during training, and
   far more *persistently* than anything under the broken init: 324 / 301 / 149 nonzero
   grasp-rate windows vs 57 / 0 / 3. Under the broken init, grasping was a one-seed fluke; under
   the corrected init it is a reproducible training behaviour on every seed. The reset-state fix
   is confirmed as the upstream blocker for grasp *emergence*.
2. **Cold deterministic-eval grasping does NOT meet the pre-stated replication bar.** The §C
   criterion was "≥2 of 3 seeds grasp cold"; the result is **1/3** (seed 1's 1/10; seeds 0 and 2
   0/10; 1/30 episodes pooled). Strictly, the **does-not-replicate branch fired**.
3. **The gap between (1) and (2) is itself the finding.** Grasping is learned as a *stochastic*
   training behaviour (peak window rates 3–6 %) but is not consolidated into the deterministic
   policy at 1M frames — and at these rates a 10-episode deterministic eval has an expected hit
   count of ~0–1, so the instrument is also thin. The live problem is no longer *emergence* (solved
   by the fix) but **consolidation/reliability**: converting a 3–6 % stochastic behaviour into a
   deterministic skill.
4. **Place/success floored on all three seeds** (~0.003 ≈ spawn-coincidence): the grasp→place
   conversion remains untouched, as expected.

## §C adjudication and recommendation (recorded, NOT executed)

Per the pre-stated rule, the **does-not-replicate** branch governs: grasp *reliability* is the live
problem, and the consultation ladder is the next question. The evidence sharpens what that ladder
should weigh:

- The confounded-arms suspension stands, but the corrected-init evidence now *favours* re-running
  the **PA3 reset-state curriculum on the corrected init** as the first ladder rung: it is the
  consolidation mechanism (dense on-policy grasp experience), it was early-positive even under the
  broken init, and its prior confound (it bypassed the broken reset) is exactly what the fix
  removed.
- **Demonstrations / BC seeding** is the next rung if curriculum-on-corrected-init fails.
- The **gate spike stays parked**: a *success*-gap gate cannot clear while place is floored, and
  cold grasping is not yet reliable. No PB1, no gate spike, no new lever is launched by this
  record — founder cut.

## Archived artifacts (this directory)

- `seed0_char_results.json`, `seed2_char_results.json` — per-run results/signature JSONs.
- `4193162f5e313d7f.jsonl`, `93d0b85c9ceeb804.jsonl` — per-window JSONLs (PPO health + ladder).
- `launch_seed0_char.log`, `launch_seed2_char.log` — trimmed launch logs (summary + structural
  events; per-window streams live in the JSONLs).
- Seed 1's artifacts are in this directory under the prior record (`RESETFIX_COLDEVAL_RESULTS.md`).

## Cross-references

- `RESETFIX_COLDEVAL_RESULTS.md` (seed 1; the §A record), `ADDENDUM_reset-state-confound_2026-06-09.md`
  (the confound), `REMEDIATION_LOG.md` §1 (broken-init triplet baseline).
- Fixes: PR #210 (ADR-007 Rev 16; ADR-002 2026-06-09), PR #206 (Rev 15).
- GPU non-determinism caveat per ADR-007 §Stage-1b (95 % CI overlap, not byte-identity).
