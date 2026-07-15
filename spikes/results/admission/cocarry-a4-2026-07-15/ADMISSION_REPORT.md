# Admission report — cocarry@v1

**Verdict.** `UNINSTRUMENTABLE`

- Pre-registration: tag `prereg-admission-cocarry-a4-2026-07-15`, blob `ffc1610049d156bd4f8cddea415f9e10f9a3feae`
- Launch commit: `111203ef29b4efcb1b9262f58aac5edcadfe235f`
- Archive date: 2026-07-15
- Seed extension used: no

## Committed thresholds (ADR-007 §Discipline — fixed before any run)

- `tau_solv` = 0.95 (A1 success floor, CI lower bound)
- `stress_limit` = 130.56970529556276 (A1 stress ceiling on successes)
- `tau_infeasible` = 0.05 (A2 success ceiling, CI upper bound)
- `delta_min` = 0.2 (A3 reference-minus-blind margin, CI lower bound)
- `c_min_ego` = 0.95 (A4 ego-robustness floor, per-partner CI lower bound)
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
- notes: wrapped committed evidence: spikes/results/admission/cocarry-2026-07-05 (I8)

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
- notes: wrapped committed evidence: spikes/results/admission/cocarry-2026-07-05 (I8)

### A3 — PASS

Criterion: paired reference-minus-blind gap: delta_ci_low >= delta_min (0.2) passes; delta_ci_high < delta_min demotes to Tier-1 CONTROL

- `delta_ci_high` = 1
- `delta_ci_low` = 1
- `delta_iqm` = 1
- `delta_mean` = 1
- `n_pairs` = 60
- notes: wrapped committed evidence: spikes/results/admission/cocarry-2026-07-05 (I8)

### A4 — FAIL

Criterion: per admitted partner: success_ci_low >= c_min_ego (0.95) with successful-episode stress peak <= stress_limit (130.56970529556276); any partner with success_ci_high < c_min_ego is brittle -> UNINSTRUMENTABLE (ADR-026 §Decision 2: a construct problem, not a partner/axis result)

- `min_success_ci_high` = 1
- `min_success_ci_low` = 0.833135
- `n_partners` = 7
- notes: wrapped committed evidence: spikes/results/benchmark/cocarry-v1/b-aht-2026-07-06 (I8); weakest admitted partner: imp_blend_c

## A4 instrument robustness — per admitted partner (ADR-027 §Admission A4)

| partner | `n_episodes` | `success_iqm` | `success_ci_low` | `success_ci_high` | `stress_p90` | `stress_max` |
|---|---|---|---|---|---|---|
| imp_blend_b | 250 | 1 | 1 | 1 | 115.112 | 129.517 |
| imp_blend_c | 250 | 1 | 0.833135 | 1 | 107.684 | 121.484 |
| imp_damp_high | 250 | 1 | 1 | 1 | 109.665 | 127.174 |
| imp_damp_low | 250 | 1 | 1 | 1 | 115.89 | 126.012 |
| imp_lag_bounded | 250 | 1 | 0.936508 | 1 | 101.641 | 124.381 |
| imp_stiff_high | 250 | 1 | 1 | 1 | 96.7806 | 123.223 |
| imp_stiff_low | 250 | 1 | 1 | 1 | 96.3132 | 113.152 |
