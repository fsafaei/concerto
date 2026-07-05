# Admission report — stage1_pickplace_as@v1

**Verdict.** `CONTROL`

- Pre-registration: tag `prereg-admission-stage1-pickplace-as-2026-07-05`, blob `a03067490eae2917706f858fb921422358fadb6f`
- Launch commit: `d68dee1d33277726fd18f6eaae153425261b96f0`
- Archive date: 2026-07-05
- Seed extension used: no

## Committed thresholds (ADR-007 §Discipline — fixed before any run)

- `tau_solv` = 0.95 (A1 success floor, CI lower bound)
- `stress_limit` = None (A1 stress ceiling on successes)
- `tau_infeasible` = 0.05 (A2 success ceiling, CI upper bound)
- `delta_min` = 0.2 (A3 reference-minus-blind margin, CI lower bound)
- seeds = [91000, 91001, 91002, 91003, 91004], episodes/seed = 12, extension = [91005, 91006]

## Checks

### A1 — PASS

Criterion: success_ci_low >= tau_solv (0.95) with successful-episode stress peak <= stress_limit (None)

- `n_episodes` = 60
- `success_ci_high` = 1
- `success_ci_low` = 1
- `success_iqm` = 1
- `success_mean` = 1
- bundles: a1_reference (each `chamber-eval verify`-able)

### A2 — FAIL

Criterion: success_ci_high <= tau_infeasible (0.05)

- `n_episodes` = 60
- `success_ci_high` = 1
- `success_ci_low` = 1
- `success_iqm` = 1
- `success_mean` = 1
- bundles: a2_partner_ablated (each `chamber-eval verify`-able)

### A3 — FAIL

Criterion: paired reference-minus-blind gap: delta_ci_low >= delta_min (0.2) passes; delta_ci_high < delta_min demotes to Tier-1 CONTROL

- `blind_n_episodes` = 60
- `blind_success_ci_high` = 1
- `blind_success_ci_low` = 1
- `blind_success_iqm` = 1
- `blind_success_mean` = 1
- `delta_ci_high` = 0
- `delta_ci_low` = 0
- `delta_iqm` = 0
- `delta_mean` = 0
- `n_pairs` = 60
- bundles: a3_partner_blind (each `chamber-eval verify`-able)
