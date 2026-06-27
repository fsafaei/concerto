# Gate-0 handover-and-place — VERDICT REPORT (2026-06-26)

Non-gating Phase-2 ADR-026 coupling-validity spike (invariant I1). Measured two-limb
slice against the FROZEN, tagged pre-registration.

## Provenance
- prereg tag: `prereg-handover-place-gate0-rev2-2026-06-26`
- prereg blob-SHA (verified == tag blob): `b26cfba74a2f1bb9ac295810e4846fe23742ff1b`
- code SHA (launch, captured once): `1b6f38f386e775166cad355cfb206089c97a2750`
- decision rule: `decision_rule.canonical_cell_rule` — two-sided 95% CI; COUPLING_VALID
  iff gap CI_lower >= delta_min; WASHOUT iff gap CI_upper < delta_min; INDETERMINATE else.
- every number below cites its archive JSON.

## HEADLINE
- **Verdict: `COUPLING_VALID`** (verdict_space; spike_handover_place_gate0_2026-06-26.json +
  crossover_curves.json)
- **Branch:** handover-and-place confirmed in principle; proceed to discovery convergence + cert-evidence spec.
- **Variance check:** sigma_u_hat = 0.137 vs high
  bracket 1.0 -> within bracket (n=20 holds); seed extension not invoked (spike JSON variance_check).

## Limb 1 — solvability (T1)
- matched grasp_pose_sigma = 2.0 deg (reported in degrees, T2).
- realistic takt band [1.0, 5.0] s: min matched IQM = 1.000,
  min matched CI_lower = 1.000 (crossover_curves.json).
- tau_solv = 0.9; **SOLVABLE = True**; Limb-1 headroom =
  +10.0 pp.

## Limb 2 — coupling validity (canonical rule; crossover_curves.json)
- coupling-valid cells (gap CI_lower >= delta_min): 22 of 216.
- coupling-valid cells (clearance / bias / arm / takt : gap_pp [CI_lo, CI_hi]):
    - clr 0.2/30d/fast/0.3s : 58.5 [46.5, 70.5]
    - clr 0.2/30d/fast/0.5s : 58.5 [46.5, 70.5]
    - clr 0.2/30d/fast/0.75s : 58.5 [46.5, 70.5]
    - clr 0.2/30d/fast/1.0s : 58.5 [46.5, 70.5]
    - clr 0.2/30d/fast/1.5s : 45.5 [32.5, 59.0]
    - clr 0.2/30d/slow/0.3s : 58.5 [46.5, 70.5]
    - clr 0.2/30d/slow/0.5s : 58.5 [46.5, 70.5]
    - clr 0.2/30d/slow/0.75s : 58.5 [46.5, 70.0]
    - clr 0.2/30d/slow/1.0s : 58.5 [46.5, 70.5]
    - clr 0.2/30d/slow/1.5s : 58.5 [46.5, 70.0]
    - clr 0.2/30d/slow/2.0s : 58.5 [46.5, 70.5]
    - clr 0.2/45d/fast/0.3s : 100.0 [100.0, 100.0]
    - clr 0.2/45d/fast/0.5s : 100.0 [100.0, 100.0]
    - clr 0.2/45d/fast/0.75s : 100.0 [100.0, 100.0]
    - clr 0.2/45d/fast/1.0s : 100.0 [100.0, 100.0]
    - clr 0.2/45d/fast/1.5s : 100.0 [100.0, 100.0]
    - clr 0.2/45d/slow/0.3s : 100.0 [100.0, 100.0]
    - clr 0.2/45d/slow/0.5s : 100.0 [100.0, 100.0]
    - clr 0.2/45d/slow/0.75s : 100.0 [100.0, 100.0]
    - clr 0.2/45d/slow/1.0s : 100.0 [100.0, 100.0]
    - clr 0.2/45d/slow/1.5s : 100.0 [100.0, 100.0]
    - clr 0.2/45d/slow/2.0s : 100.0 [100.0, 100.0]

## Crossover band mapped to takt (both arms; crossover_curves.json)
- 15deg / clr 0.2 / fast: window_takts_s=[] (floor=0.3, ceiling=None)
- 15deg / clr 0.35 / fast: window_takts_s=[] (floor=0.3, ceiling=None)
- 15deg / clr 0.5 / fast: window_takts_s=[] (floor=0.3, ceiling=None)
- 15deg / clr 0.7 / fast: window_takts_s=[] (floor=0.3, ceiling=None)
- 15deg / clr 0.2 / slow: window_takts_s=[] (floor=0.3, ceiling=None)
- 15deg / clr 0.35 / slow: window_takts_s=[] (floor=0.3, ceiling=None)
- 15deg / clr 0.5 / slow: window_takts_s=[] (floor=0.3, ceiling=None)
- 15deg / clr 0.7 / slow: window_takts_s=[] (floor=0.3, ceiling=None)
- 30deg / clr 0.2 / fast: window_takts_s=[0.3, 0.5, 0.75, 1.0, 1.5] (floor=0.3, ceiling=1.5)
- 30deg / clr 0.35 / fast: window_takts_s=[] (floor=0.3, ceiling=None)
- 30deg / clr 0.5 / fast: window_takts_s=[] (floor=0.3, ceiling=None)
- 30deg / clr 0.7 / fast: window_takts_s=[] (floor=0.3, ceiling=None)
- 30deg / clr 0.2 / slow: window_takts_s=[0.3, 0.5, 0.75, 1.0, 1.5, 2.0] (floor=0.3, ceiling=2.0)
- 30deg / clr 0.35 / slow: window_takts_s=[] (floor=0.3, ceiling=None)
- 30deg / clr 0.5 / slow: window_takts_s=[] (floor=0.3, ceiling=None)
- 30deg / clr 0.7 / slow: window_takts_s=[] (floor=0.3, ceiling=None)
- 45deg / clr 0.2 / fast: window_takts_s=[0.3, 0.5, 0.75, 1.0, 1.5] (floor=0.3, ceiling=1.5)
- 45deg / clr 0.35 / fast: window_takts_s=[] (floor=0.3, ceiling=None)
- 45deg / clr 0.5 / fast: window_takts_s=[] (floor=0.3, ceiling=None)
- 45deg / clr 0.7 / fast: window_takts_s=[] (floor=0.3, ceiling=None)
- 45deg / clr 0.2 / slow: window_takts_s=[0.3, 0.5, 0.75, 1.0, 1.5, 2.0] (floor=0.3, ceiling=2.0)
- 45deg / clr 0.35 / slow: window_takts_s=[] (floor=0.3, ceiling=None)
- 45deg / clr 0.5 / slow: window_takts_s=[] (floor=0.3, ceiling=None)
- 45deg / clr 0.7 / slow: window_takts_s=[] (floor=0.3, ceiling=None)
- realistic takt band = [1.0, 5.0] s; COUPLING_VALID fires iff a window overlaps it.

## Intrinsic vs budget-mediated split (free-re-grasp endpoint)
- free-re-grasp gap CI_lower = 0.0 pp -> **BUDGET-MEDIATED (vanishes at free re-grasp)** (spike JSON free episodes).

## (clearance x mismatch) MEASURED coupling region
  (clearance_mismatch_region_measured.json)
- clearance / bias / coupling_valid_any_takt / max_gap_CI_lower_pp:
    - clr 0.2 / 15deg : coupling=False / max_gap_CI_lower=0.0 pp
    - clr 0.2 / 30deg : coupling=True / max_gap_CI_lower=46.5 pp
    - clr 0.2 / 45deg : coupling=True / max_gap_CI_lower=100.0 pp
    - clr 0.35 / 15deg : coupling=False / max_gap_CI_lower=0.0 pp
    - clr 0.35 / 30deg : coupling=False / max_gap_CI_lower=0.0 pp
    - clr 0.35 / 45deg : coupling=False / max_gap_CI_lower=17.5 pp
    - clr 0.5 / 15deg : coupling=False / max_gap_CI_lower=0.0 pp
    - clr 0.5 / 30deg : coupling=False / max_gap_CI_lower=0.0 pp
    - clr 0.5 / 45deg : coupling=False / max_gap_CI_lower=0.0 pp
    - clr 0.7 / 15deg : coupling=False / max_gap_CI_lower=0.0 pp
    - clr 0.7 / 30deg : coupling=False / max_gap_CI_lower=0.0 pp
    - clr 0.7 / 45deg : coupling=False / max_gap_CI_lower=0.0 pp

## Binding-conjunct log (T3 — lateral success-side, grasp-pose coupling-side)
- matched failures by conjunct: {}
- mismatched failures by conjunct: {'angular': 2546, 'angular|force': 6407}
  (expected: mismatched failures dominated by 'angular' (grasp-pose); 'lateral' absent.)

## Selected branch
- **COUPLING_VALID** -> handover-and-place confirmed in principle; proceed to discovery convergence + cert-evidence spec.

_No prose-only numbers: every figure traces to a committed JSON in this archive._
