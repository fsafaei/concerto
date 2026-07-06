# cocarry@v1 baseline campaign report (prereg rev2, 2026-07-05/06)

Prereg: `spikes/preregistration/benchmark/cocarry_baselines_v1.yaml`,
tags `prereg-cocarry-baselines-v1-2026-07-05` (rev1, blob 8dbf903e) /
`prereg-cocarry-baselines-v1-rev2-2026-07-05` (rev2, blob 31914599 —
eval n locked at 50 by the rev1 power pilot; pilot bundles committed
under `power-pilot-*`, excluded from every manifest). All measured
episodes ran under the tag gate; nothing outside the preregistered
cell list appears under `spikes/results/benchmark/`.

## Leaderboard rows (as rendered in README; every bundle verify-PASS)

| Row | Success IQM [95% CI] | Stress p90 (N) | Per-partner min…max | n episodes |
|---|---|---|---|---|
| REF-SCRIPT *(oracle reference)* | 1.000 [1.000, 1.000] | 102.0 | 1.00 … 1.00 | 1750 |
| B-RND | 0.000 [0.000, 0.000] | 277.8 | 0.00 … 0.00 | 1750 |
| B-STAT | 0.492 [0.396, 0.591] | 101.4 | 0.16 (imp_lag_bounded) … 0.66 (imp_blend_b) | 1750 |
| B-BLIND | 1.000 [1.000, 1.000] | 107.6 | 0.86 (imp_lag_bounded) … 1.00 | 1750 |
| B-AHT | 1.000 [1.000, 1.000] | 108.4 | 0.88 (imp_lag_bounded) … 1.00 | 1750 |
| B-JOINT *(non-AHT upper anchor, as-a-pair)* | 1.000 [1.000, 1.000] | 41.2 | per-seed: 4×1.000 + seed4 0.923 [0.692, 1.000] | 250 |

Headline reading, stated plainly: **the co-carry success metric
saturates at the row level** — with ADR-027 checkpoint selection, the
partner-blind ego (B-BLIND) matches the coupling-aware B-AHT at 1.000
across the public eval set. The learned B-BLIND retains own-proprioception
load feedback (the documented rev2 asymmetry vs the scripted A3 blind,
which scored 0/60 at admission) and the partners carry ~49% alone
(B-STAT), which together suffice. Discrimination lives in: the
per-partner floor (imp_lag_bounded 0.86/0.88), stress (B-JOINT pairs
run at 41 N vs ~102-108 N for all cross-play dyads — joint training
finds gentler coordination), and the zoo cross-play gate (below).

## Checkpoint selection + instability events (decision rule 4)

Selection evidence: `selection/` (15 artifacts + per-row manifests,
committed before any eval cell). Selected steps — B-BLIND: 50k/200k/
100k/50k/150k; B-AHT: 100k/150k/150k/50k/50k; B-JOINT: 150k/200k/50k/
150k/150k (seeds 0-4). **Every (row, seed) tripped the preregistered
>40-point instability rule** on the validation partner (max
consecutive-checkpoint drops 50-100 points; B-JOINT worst with
repeated 100-point swings). Per the rule: seeds completed, the
selection rule was applied, events reported here. Budgets were never
extended; no row hit its cap (all trained the full 300k frames/seed).

## Compute

Training 15 runs × 300k frames, sequential on one RTX 2080 SUPER:
8.8 h wall-clock total (B-BLIND ≈43 min/seed, B-AHT ≈41, B-JOINT ≈25).
Eval: eval-only rows ≈5 h (B-RND 1.6 h, B-STAT 3 h, REF-SCRIPT 0.5 h);
learned rows ≈50 min total (success terminates episodes early).
Selection ≈3 h. No committed wall-clock cap existed (prereg: frames
are the cap denomination); reported here per the prereg.

## Zoo enrichment (version 2)

Five jointly-trained partner-side candidates preregistered; the
committed 0.75 cross-play floor (fingerprint probe vs the reference
ego) **dropped four** (0.000/0.000/0.050/0.650) — evidence at
`spikes/results/partner-fingerprints/cocarry_partners-v2-dropped-candidates/`.
`joint_s4` (0.800) is the one admitted learned member;
`cocarry_partners@v2` = 12 members (9 public / 3 private; every v1
split label reproduced), full archive + cards committed.

## Recompute (one command per row)

    uv run chamber-eval verify spikes/results/benchmark/cocarry-v1/<row-dir>

Row dirs: `ref-script-2026-07-05`, `b-rnd-2026-07-05`,
`b-stat-2026-07-05`, `b-blind-2026-07-06`, `b-aht-2026-07-06`,
`b-joint-seed{0..4}-2026-07-06`. Full re-runs: each bundle's
`REPRO.txt` (learned rows consume the committed `selection/*.json`
manifests; B-JOINT re-pairs via `--partner-weights`).
