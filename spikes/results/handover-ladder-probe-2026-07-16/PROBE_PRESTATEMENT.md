# Pre-statement — handover ladder Slice 0: oracle-headroom probe (2026-07-16)

**Status: committed BEFORE any probe episode runs.** The commit carrying this file
precedes the commit carrying `probe_results.json` / `PROBE_REPORT.md`; the branch
history is the rule-before-result record (ADR-007 §Discipline, applied in its
pre-statement form — this is NOT a gate claim and no prereg tag is rotated). The
pre-statement commit additionally carries the signed tag
`probe-handover-ladder-slice0-2026-07-16`.

## What this probe is

Eval-only, CPU-only, **non-gating** (invariant I1: nothing here gates Phase-1/M10).
It informs exactly one named decision: whether learned-ego `handover_place` rows move
from "v1.1 roadmap" into v1.1 scope (the Slice 1–4 ladder program). No leaderboard
row changes; no `src/` behaviour changes; no schema constants change.

## Cells and draws (replication of the committed schedule)

- Task: `handover_place@1` at the committed coupling-valid anchor
  (`chamber.benchmarks.partner_probe.HANDOVER_PROBE_ENV_PARAMS`, verbatim: clearance
  0.2 → wrist correction 25°, fast basis, takt 1.5 s → re-grasp budget 1.13 s,
  re-grasp duration 1.06 s, windows 1 mm / 5° / 75 N).
- Cells: `presenter_mismatch_30`, `presenter_mismatch_45` from
  `handover_place_partners@v1` (the two coupling-valid leaderboard cells of prereg
  `prereg-handover-baselines-v1-2026-07-06`).
- Draws: the exact committed schedule — seeds 0–4 × 50 episodes/partner,
  `initial_state_seed = seed*1000 + episode`; presenter, env reset, and step order
  identical to `chamber.benchmarks.handover_eval.run_handover_episodes_for_set`.
- The committed reference bundle
  `spikes/results/benchmark/handover-v1/ref-script-2026-07-06/` is read
  SHA-verified and never modified (I8).

## Policies evaluated on the same draws

1. **REF** — the actual `ScriptedHandoverEgo` driven live (never copied numbers).
2. **Fixed policies** (each with the standard translate/wrist corrections):
   *always-regrasp*, *never-regrasp*, and *the scripted rule* (= REF's rule).
3. **Oracle** — per-state search of the ego phase-1 action space
   `[translate_x, translate_y, reorient_deg, regrasp_flag]` under the exact success
   predicate: translation analytic (clamp toward zero lateral residual; §1 verified
   the residual-norm objective is monotone for both the lateral window and the force
   proxy, so exact cancellation within the 0.10 m range is optimal), plus a dense
   grid over `reorient_deg × regrasp_flag` driven through the real env step.

Per-episode record: best-found outcome and a classification —
*feasible-and-found* / *infeasible-by-budget* (re-grasp budget-blocked and the
angular window unreachable by wrist alone) / *infeasible-by-window* (unreachable
even under a free re-grasp).

## Decision rule (verbatim; bounds founder-confirmed at kickoff 2026-07-16)

- **Headroom condition:** oracle mean-success − REF mean-success ≥ **+0.15**
  absolute on at least one of the two canonical cells (same draws, same success
  predicate).
- **Learnable-structure condition:** oracle − best *fixed* policy ≥ **+0.05** on at
  least one canonical cell, where the fixed policies are {always-regrasp,
  never-regrasp, the scripted rule}, each with the standard translate/wrist
  corrections. This tests that the optimal action is state-dependent — that a
  policy has something to learn beyond a constant rule.
- **GO** iff both conditions hold; **NO-GO** otherwise.
- Secondary (reported, not gating): seating-force distributions at oracle vs REF;
  the per-episode infeasibility decomposition.

## Reconciliation gate (hard)

The recomputed REF over the probe draws must match the committed row
(pooled mean 0.338 / IQM 0.176, `n_resamples=2000`, bootstrap root seed 0) within
resampling noise — the driver checks per-episode equality against the committed
`episodes_seed*.jsonl` records and recomputes the bundle summary with
`chamber.evaluation.bundles.compute_summary`. A mismatch means the protocol
diverged: **stop and report; no verdict is computed.**

## Determinism

Fixed seeds throughout; every draw routes through
`concerto.training.seeding.derive_substream` (P6 / ADR-002); no wall-clock, no
unseeded RNG anywhere in the driver.
