# Admission report — cocarry@v1

**Verdict.** `ADMITTED`

- Pre-registration: tag `prereg-admission-cocarry-2026-07-05`, blob `7525a066d7d55a2f98312344d7272634844617c1`
- Launch commit: `d68dee1d33277726fd18f6eaae153425261b96f0`
- Archive date: 2026-07-05
- Seed extension used: no

## Committed thresholds (ADR-007 §Discipline — fixed before any run)

- `tau_solv` = 0.95 (A1 success floor, CI lower bound)
- `stress_limit` = 130.56970529556276 (A1 stress ceiling on successes)
- `tau_infeasible` = 0.05 (A2 success ceiling, CI upper bound)
- `delta_min` = 0.2 (A3 reference-minus-blind margin, CI lower bound)
- seeds = [90000, 90001, 90002, 90003, 90004], episodes/seed = 12, extension = [90005, 90006]

## Checks

### A1 — PASS

Criterion: success_ci_low >= tau_solv (0.95) with successful-episode stress peak <= stress_limit (130.56970529556276)

- `n_episodes` = 60
- `stress_max` = 107.912
- `stress_n` = 60
- `stress_p50` = 88.5999
- `stress_p90` = 104.079
- `stress_p99` = 106.422
- `success_ci_high` = 1
- `success_ci_low` = 1
- `success_iqm` = 1
- `success_mean` = 1
- bundles: a1_reference (each `chamber-eval verify`-able)

### A2 — PASS

Criterion: success_ci_high <= tau_infeasible (0.05)

- `n_episodes` = 60
- `stress_max` = 17.9965
- `stress_n` = 60
- `stress_p50` = 8.16705
- `stress_p90` = 9.45765
- `stress_p99` = 15.1805
- `success_ci_high` = 0
- `success_ci_low` = 0
- `success_iqm` = 0
- `success_mean` = 0
- bundles: a2_single_arm (each `chamber-eval verify`-able)

### A3 — PASS

Criterion: paired reference-minus-blind gap: delta_ci_low >= delta_min (0.2) passes; delta_ci_high < delta_min demotes to Tier-1 CONTROL

- `blind_n_episodes` = 60
- `blind_success_ci_high` = 0
- `blind_success_ci_low` = 0
- `blind_success_iqm` = 0
- `blind_success_mean` = 0
- `delta_ci_high` = 1
- `delta_ci_low` = 1
- `delta_iqm` = 1
- `delta_mean` = 1
- `n_pairs` = 60
- bundles: a3_blind (each `chamber-eval verify`-able)

## A2 binding evidence — stress on matched successes (ADR-026 §Decision 2)

- `stress_max` = 107.912
- `stress_n` = 60
- `stress_p50` = 88.5999
- `stress_p90` = 104.079
- `stress_p99` = 106.422
