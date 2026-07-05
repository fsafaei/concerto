# Admission report — handover_place@v1

**Verdict.** `ADMITTED`

- Pre-registration: tag `prereg-admission-handover-place-2026-07-05`, blob `0b158e4bfae51eb40a30f033936292ca9d6427fd`
- Launch commit: `d68dee1d33277726fd18f6eaae153425261b96f0`
- Archive date: 2026-07-05
- Seed extension used: no

## Committed thresholds (ADR-007 §Discipline — fixed before any run)

- `tau_solv` = 0.9 (A1 success floor, CI lower bound)
- `stress_limit` = None (A1 stress ceiling on successes)
- `tau_infeasible` = 0.05 (A2 success ceiling, CI upper bound)
- `delta_min` = 0.2 (A3 reference-minus-blind margin, CI lower bound)
- seeds = [0, 1, 2, 3, 4], episodes/seed = 20, extension = [5, 6]

## Checks

### A1 — PASS

Criterion: success_ci_low >= tau_solv (0.9) with successful-episode stress peak <= stress_limit (None)

- `n_cells` = 120
- `success_ci_high` = 1
- `success_ci_low` = 1
- `success_iqm` = 1
- notes: wrapped committed evidence: spikes/results/handover-place-gate0-2026-06-26 (I8)

### A2 — PASS

Criterion: success_ci_high <= tau_infeasible (0.05)

- `n_episodes` = 100
- `stress_max` = 9375
- `stress_n` = 100
- `stress_p50` = 9375
- `stress_p90` = 9375
- `stress_p99` = 9375
- `success_ci_high` = 0
- `success_ci_low` = 0
- `success_iqm` = 0
- `success_mean` = 0
- bundles: a2_presenter_ablated (each `chamber-eval verify`-able)

### A3 — PASS

Criterion: paired reference-minus-blind gap: delta_ci_low >= delta_min (0.2) passes; delta_ci_high < delta_min demotes to Tier-1 CONTROL

- `delta_ci_high` = 1
- `delta_ci_low` = 0.325
- `delta_iqm` = 0.455
- `delta_max` = 1
- `n_cells` = 10
- notes: wrapped committed evidence: spikes/results/handover-place-gate0-2026-06-26 (I8)
