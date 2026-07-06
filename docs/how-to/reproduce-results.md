# How-to: Reproduce the leaderboard results

Every leaderboard row is backed by a committed result bundle, and
every bundle records the exact command that produced it (its
`REPRO.txt`). This page lists, for each row: the reproduction
command, the committed bundle it must match, and the verification
command. The command blocks below are checked against the committed
`REPRO.txt` files by CI (`scripts/check_repro_docs.py`), so they
cannot drift from the evidence.

Two levels of reproduction, from cheap to thorough:

1. **Verify** (seconds per row, no simulation): re-check every file
   hash in the committed bundle, recompute the summary statistics
   from the raw episode records, and re-derive the preregistration
   tag. This is what CI does on every push.
2. **Re-run** (minutes to hours per row, CPU): execute the row's
   command with a fresh output directory. CPU evaluation is
   byte-identical under the committed `uv.lock` and the recorded
   seeds, so the fresh bundle's summary must equal the committed
   one's exactly. The scheduled `repro-rows` workflow
   (`.github/workflows/repro-rows.yml`) re-runs every scripted row
   this way weekly and on demand.

Setup, once per checkout (the preregistration gate resolves git
tags, so fetch them):

```bash
git fetch --tags
uv sync --group dev
```

Notes that apply to every block below:

- Re-run commands write to a fresh directory under `out/repro/`;
  point `--out` anywhere empty. The committed bundle path is listed
  as *expected bundle* — do not write into it (committed archives
  are immutable).
- On a locally-modified checkout, add `--allow-dirty` to `run`
  commands (the resulting bundle is marked dirty and is
  leaderboard-ineligible).
- Approximate CPU runtimes are for one modern core-count laptop/CI
  runner; scripted cocarry rows are ~80 minutes (1750 episodes),
  scripted handover rows are lighter.

## cocarry@v1 (campaign tag `prereg-cocarry-baselines-v1-rev2-2026-07-05`)

### REF-SCRIPT — oracle reference

```bash
uv run chamber-eval run --task cocarry --policy ref_script_cocarry_impedance --partner-set cocarry_partners@v1 --exclude-member imp_nominal --seeds 5 --episodes 50 --out out/repro/cocarry-ref-script --prereg spikes/preregistration/benchmark/cocarry_baselines_v1.yaml
```

Expected bundle: `spikes/results/benchmark/cocarry-v1/ref-script-2026-07-05`

```bash
uv run chamber-eval verify spikes/results/benchmark/cocarry-v1/ref-script-2026-07-05
```

### B-RND — random ego

```bash
uv run chamber-eval run --task cocarry --policy random --partner-set cocarry_partners@v1 --exclude-member imp_nominal --seeds 5 --episodes 50 --out out/repro/cocarry-b-rnd --prereg spikes/preregistration/benchmark/cocarry_baselines_v1.yaml
```

Expected bundle: `spikes/results/benchmark/cocarry-v1/b-rnd-2026-07-05`

```bash
uv run chamber-eval verify spikes/results/benchmark/cocarry-v1/b-rnd-2026-07-05
```

### B-STAT — static (hold-position) ego

```bash
uv run chamber-eval run --task cocarry --policy static --partner-set cocarry_partners@v1 --exclude-member imp_nominal --seeds 5 --episodes 50 --out out/repro/cocarry-b-stat --prereg spikes/preregistration/benchmark/cocarry_baselines_v1.yaml
```

Expected bundle: `spikes/results/benchmark/cocarry-v1/b-stat-2026-07-05`

```bash
uv run chamber-eval verify spikes/results/benchmark/cocarry-v1/b-stat-2026-07-05
```

### B-BLIND — learned ego, partner observations masked

Requires the selected training checkpoints referenced by the
committed selection manifest (see the note on checkpoints below).

```bash
uv run chamber-eval run --task cocarry --policy happo_blind_manifest:spikes/results/benchmark/cocarry-v1/selection/b-blind_selected_manifest.json --partner-set cocarry_partners@v1 --exclude-member imp_nominal --seeds 5 --episodes 50 --out out/repro/cocarry-b-blind --prereg spikes/preregistration/benchmark/cocarry_baselines_v1.yaml
```

Expected bundle: `spikes/results/benchmark/cocarry-v1/b-blind-2026-07-06`

```bash
uv run chamber-eval verify spikes/results/benchmark/cocarry-v1/b-blind-2026-07-06
```

### B-AHT — learned ad-hoc-teamwork ego

Requires the selected training checkpoints referenced by the
committed selection manifest (see the note on checkpoints below).

```bash
uv run chamber-eval run --task cocarry --policy happo_manifest:spikes/results/benchmark/cocarry-v1/selection/b-aht_selected_manifest.json --partner-set cocarry_partners@v1 --exclude-member imp_nominal --seeds 5 --episodes 50 --out out/repro/cocarry-b-aht --prereg spikes/preregistration/benchmark/cocarry_baselines_v1.yaml
```

Expected bundle: `spikes/results/benchmark/cocarry-v1/b-aht-2026-07-06`

```bash
uv run chamber-eval verify spikes/results/benchmark/cocarry-v1/b-aht-2026-07-06
```

### B-JOINT — non-AHT upper anchor (one bundle per training seed)

Each jointly-trained pair is evaluated as the pair it trained as, so
the row ships five single-seed bundles. Requires the pair checkpoints
(see the note on checkpoints below).

```bash
uv run chamber-eval run --task cocarry --policy joint_ego:local://artifacts/4ace772a2efe7dd3_step150000.pt --partner frozen_cocarry_joint --partner-weights local://artifacts/4ace772a2efe7dd3_step150000.pt --seeds 0, --episodes 50 --out out/repro/cocarry-b-joint-seed0 --prereg spikes/preregistration/benchmark/cocarry_baselines_v1.yaml
uv run chamber-eval run --task cocarry --policy joint_ego:local://artifacts/24e5f7483ad7b86e_step200000.pt --partner frozen_cocarry_joint --partner-weights local://artifacts/24e5f7483ad7b86e_step200000.pt --seeds 1, --episodes 50 --out out/repro/cocarry-b-joint-seed1 --prereg spikes/preregistration/benchmark/cocarry_baselines_v1.yaml
uv run chamber-eval run --task cocarry --policy joint_ego:local://artifacts/e2f99cc34a4c5356_step50000.pt --partner frozen_cocarry_joint --partner-weights local://artifacts/e2f99cc34a4c5356_step50000.pt --seeds 2, --episodes 50 --out out/repro/cocarry-b-joint-seed2 --prereg spikes/preregistration/benchmark/cocarry_baselines_v1.yaml
uv run chamber-eval run --task cocarry --policy joint_ego:local://artifacts/461dbbcae360f85e_step150000.pt --partner frozen_cocarry_joint --partner-weights local://artifacts/461dbbcae360f85e_step150000.pt --seeds 3, --episodes 50 --out out/repro/cocarry-b-joint-seed3 --prereg spikes/preregistration/benchmark/cocarry_baselines_v1.yaml
uv run chamber-eval run --task cocarry --policy joint_ego:local://artifacts/e9040cbc3c3b2456_step150000.pt --partner frozen_cocarry_joint --partner-weights local://artifacts/e9040cbc3c3b2456_step150000.pt --seeds 4, --episodes 50 --out out/repro/cocarry-b-joint-seed4 --prereg spikes/preregistration/benchmark/cocarry_baselines_v1.yaml
```

Expected bundles:
`spikes/results/benchmark/cocarry-v1/b-joint-seed0-2026-07-06`
`spikes/results/benchmark/cocarry-v1/b-joint-seed1-2026-07-06`
`spikes/results/benchmark/cocarry-v1/b-joint-seed2-2026-07-06`
`spikes/results/benchmark/cocarry-v1/b-joint-seed3-2026-07-06`
`spikes/results/benchmark/cocarry-v1/b-joint-seed4-2026-07-06`

```bash
uv run chamber-eval verify spikes/results/benchmark/cocarry-v1/b-joint-seed0-2026-07-06
uv run chamber-eval verify spikes/results/benchmark/cocarry-v1/b-joint-seed1-2026-07-06
uv run chamber-eval verify spikes/results/benchmark/cocarry-v1/b-joint-seed2-2026-07-06
uv run chamber-eval verify spikes/results/benchmark/cocarry-v1/b-joint-seed3-2026-07-06
uv run chamber-eval verify spikes/results/benchmark/cocarry-v1/b-joint-seed4-2026-07-06
```

## handover_place@v1 (campaign tag `prereg-handover-baselines-v1-2026-07-06`)

### REF-SCRIPT — oracle reference

```bash
uv run chamber-eval run --task handover_place --policy ref_script_handover_ego --partner-set handover_place_partners@v1 --exclude-member presenter_mismatch_15 --exclude-member presenter_offgrid_a --exclude-member presenter_offgrid_b --seeds 5 --episodes 50 --out out/repro/handover-ref-script --prereg spikes/preregistration/benchmark/handover_baselines_v1.yaml
```

Expected bundle: `spikes/results/benchmark/handover-v1/ref-script-2026-07-06`

```bash
uv run chamber-eval verify spikes/results/benchmark/handover-v1/ref-script-2026-07-06
```

### B-RND — random ego

```bash
uv run chamber-eval run --task handover_place --policy random --partner-set handover_place_partners@v1 --exclude-member presenter_mismatch_15 --exclude-member presenter_offgrid_a --exclude-member presenter_offgrid_b --seeds 5 --episodes 50 --out out/repro/handover-b-rnd --prereg spikes/preregistration/benchmark/handover_baselines_v1.yaml
```

Expected bundle: `spikes/results/benchmark/handover-v1/b-rnd-2026-07-06`

```bash
uv run chamber-eval verify spikes/results/benchmark/handover-v1/b-rnd-2026-07-06
```

### B-STAT — static (hold-position) ego

```bash
uv run chamber-eval run --task handover_place --policy static --partner-set handover_place_partners@v1 --exclude-member presenter_mismatch_15 --exclude-member presenter_offgrid_a --exclude-member presenter_offgrid_b --seeds 5 --episodes 50 --out out/repro/handover-b-stat --prereg spikes/preregistration/benchmark/handover_baselines_v1.yaml
```

Expected bundle: `spikes/results/benchmark/handover-v1/b-stat-2026-07-06`

```bash
uv run chamber-eval verify spikes/results/benchmark/handover-v1/b-stat-2026-07-06
```

## Hosted artifacts

The data artifacts behind the leaderboard are hosted as three
Hugging Face datasets (Croissant metadata included; see the
[datasheet](../reference/datasheet.md)):

| Dataset | Location |
|---|---|
| `chamber-bench-partner-sets` | `<pending first upload>` |
| `chamber-bench-leaderboard-bundles` | `<pending first upload>` |
| `chamber-bench-reference-trajectories` | `<pending first upload>` |

The placeholders are replaced with the live URLs at the first upload
(CB-08).

## Note on checkpoints for the learned rows

The learned rows (B-BLIND, B-AHT, B-JOINT) load training checkpoints
via `local://artifacts/...` URIs. The checkpoint files (~3 MB each)
are content-addressed and are **not committed to the git repository**;
they are part of the hosted leaderboard-bundles dataset artifact (see
the [datasheet](../reference/datasheet.md)). Place them under
`artifacts/artifacts/` before re-running a learned row; `chamber-eval
verify` on the *committed* bundles works without them. To regenerate
the checkpoints from scratch, the training configuration and the
checkpoint-selection rule are preregistered in the campaign YAML and
documented in
`spikes/results/benchmark/cocarry-v1/CAMPAIGN_REPORT.md`
(15 runs × 300k frames; roughly 9 hours on one consumer GPU).

## What "match" means

A successful re-run reproduces the committed bundle's summary block
(`success_iqm`, `success_mean`, and both confidence-interval
endpoints) exactly on CPU — byte-identical determinism under
`uv.lock` + recorded seeds is a project invariant (principle P6), and
the scheduled `repro-rows` workflow fails if any value differs.
Fields that record provenance (git SHA, platform fingerprint,
timestamps in file metadata) legitimately differ between your machine
and the original run.
