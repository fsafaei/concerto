# Observability cookbook (P1.05.11; ADR-017)

This cookbook covers the workflows P1.05.11 enables — live W&B dashboards
for the founder + JSON-first co-analysis for the agent. The design lives
in [`ADR-017`](../adr/ADR-017-observability-and-experiment-tracking.md);
this document is the operator-side recipe collection.

---

## TL;DR

```bash
# 1. Install observability extras (imageio[ffmpeg] for rollout MP4 encoding).
uv sync --extra observability

# 2. Authenticate W&B (one-time, founder-side).
wandb login

# 3. Enable the W&B sink for a Stage-1b cell.
uv run chamber-spike train \
    --config configs/training/ego_aht_happo/stage1_pickplace.yaml \
    --override wandb.enabled=true

# 4. Browse the dashboard at https://wandb.ai/<entity>/concerto-chamber.

# 5. Programmatic co-analysis (agent-side; works offline, no W&B account).
uv run chamber-analyze list-runs --archive-root spikes/results/ --json | jq .
```

---

## How to enable W&B for a spike run

The W&B sink is opt-in via `cfg.wandb.enabled`. Two ways to flip it on:

**Per-run override (no config edit):**

```bash
uv run chamber-spike train \
    --config configs/training/ego_aht_happo/stage1_pickplace.yaml \
    --override wandb.enabled=true
```

**Persistent (edit the YAML):**

```yaml
# configs/training/ego_aht_happo/stage1_pickplace.yaml
wandb:
  enabled: true
  project: concerto-chamber   # ADR-017 §Decisions D2 — single project
```

Degrade-to-no-op paths (each emits one `UserWarning`, never raises):

| Condition | Result |
|-----------|--------|
| `cfg.wandb.enabled=false` | Silent — JSONL-only path, no warning. |
| `WANDB_API_KEY` missing AND `WANDB_MODE != "offline"` | UserWarning; JSONL-only run. |
| `import wandb` fails | UserWarning; JSONL-only run. |
| `wandb.init` raises | UserWarning; JSONL-only run. |

The canonical record is **always** `<cfg.log_dir>/<run_id>.jsonl`. W&B is downstream of that. If your run completed and the JSONL exists, the run is valid.

---

## How to back-fill historical archives

```bash
# Dry-run against committed archives — show what would replay, no W&B calls.
uv run python scripts/wandb_backfill.py \
    --archive-root spikes/results/ \
    --project concerto-chamber --mode offline --dry-run

# Replay the three committed Stage-1 archives to W&B.
uv run python scripts/wandb_backfill.py \
    --archive-root spikes/results/ \
    --project concerto-chamber --mode online

# Also replay the four gitignored .local/ post-widening snapshots.
uv run python scripts/wandb_backfill.py \
    --archive-root .local/ \
    --include-local \
    --project concerto-chamber --mode online
```

Idempotent: re-running against the same archive skips existing W&B runs (`wandb.init(id=<run_id>, resume="never")`).

**What backfill does deliver.** Replays per-cell JSONL events (training_start, safety_telemetry, rollout_update, safety_telemetry_final, training_end) + envelope-level terminal metrics (per-episode `success`, `mean_reward`, `constraint_violation_peak`, `fallback_fired`) into W&B.

**What backfill does NOT deliver.** Per-step PPO health curves (policy_loss, value_loss, dist_entropy, approx_kl, clip_fraction, grad_norm) for historical archives — those scalars were never written to disk before P1.05.11. They are only available for runs going forward.

---

## How to compare runs with `chamber-analyze`

```bash
# Enumerate runs under an archive root.
uv run chamber-analyze list-runs \
    --archive-root spikes/results/stage1-failure-investigation/2026-05-20 \
    --json | jq .

# Summary of one run.
uv run chamber-analyze summary 43a0d043cb9f54ac \
    --archive-root spikes/results/stage1-failure-investigation/2026-05-20 \
    --json

# Time-series for one metric on one run.
uv run chamber-analyze metrics 43a0d043cb9f54ac \
    --archive-root spikes/results/stage1-failure-investigation/2026-05-20 \
    --metric lambda_mean --namespace safety --json

# Side-by-side comparison across N runs.
uv run chamber-analyze compare 43a0d043cb9f54ac e3ca669746e8f4bb ef23e3b576f38c91 \
    --archive-root spikes/results/stage1-failure-investigation/2026-05-20 \
    --metrics mean_reward_final,terminal_success_rate,lambda_steady_state \
    --json
```

---

## How to inspect a rollout at per-step granularity

```bash
# Dump every step of the recorded eval episode at step_025000.
uv run chamber-analyze rollout-frames <run-id> \
    --archive-root <archive_root> \
    --episode 25000 | jq .

# Filter to a single obs/info field across all recorded steps.
uv run chamber-analyze rollout-frames <run-id> \
    --archive-root <archive_root> \
    --field is_grasped | jq .
```

Per ADR-017 §Decisions D6 the per-step records live in sidecar JSONLs at `<archive_root>/rollouts/<condition>/step_<NNNNNN>.jsonl`, paired 1:1 with the MP4 at the same path. The MP4 is for the founder; the JSONL is for the agent.

The `obs_summary` is curated (cube_pose, goal_pos, tcp_pose, ego_qpos/qvel, partner_qpos/qvel, is_grasped, gripper_width) — never the full 65-D flattened ego state. Full-obs logging is deferred (ADR-017 §Open questions).

---

## How to embed a comparison plot in a PR description

```bash
uv run chamber-analyze plot 43a0d043cb9f54ac e3ca669746e8f4bb \
    --archive-root spikes/results/stage1-failure-investigation/2026-05-20 \
    --metric lambda_mean --namespace safety \
    --out /tmp/lambda_comparison.png \
    --title "lambda_mean: seed 0 vs seed 1"
```

The PNG is matplotlib-rendered (Agg backend; headless-safe). Embed it in a markdown PR description via `![title](attachment-url)`.

---

## JSON output schemas

Every `chamber-analyze` subcommand supports `--json`. Schemas are stable across patch versions; adding fields is allowed, removing or renaming requires a major version bump.

### `chamber-analyze list-runs --json`

```json
[
  {
    "run_id": "<16-hex>",
    "jsonl_path": "<path>",
    "stage": "1" | "2" | "3" | null,
    "sub_stage": "1a" | "1b" | "2" | "3" | null,
    "condition": "<condition_id>" | null,
    "seed": <int> | null,
    "task": "stage1_pickplace" | ...,
    "run_kind": "ego_aht_happo" | ...,
    "git_sha": "<40-hex>",
    "n_steps": <int> | null,
    "terminal_success_rate": <float in [0, 1]> | null,
    "mean_reward_final": <float> | null,
    "lambda_steady_state": <float> | null
  }, ...
]
```

### `chamber-analyze metrics --json`

```json
[
  {"step": <int>, "wall_time": "<iso8601>", "value": <float>},
  ...
]
```

### `chamber-analyze compare --json`

```json
[
  {
    "run_id": "<16-hex>",
    "<metric_1>": <value> | null,
    "<metric_2>": <value> | null,
    ...
  }, ...
]
```

### `chamber-analyze rollout-frames --json`

When `--field` is unset, emits the full per-step JSONL line shape:

```json
[
  {
    "event": "rollout_step",
    "metric_namespace": "rollout",
    "step_global": <int>,
    "step_episode": <int>,
    "condition": "<condition>",
    "seed": <int>,
    "obs_summary": {"cube_pose": [...], "goal_pos": [...], ...},
    "action": [<float>, ...],
    "reward": <float>,
    "terminated": <bool>,
    "truncated": <bool>,
    "info": {...},
    "frame_index": <int>
  }, ...
]
```

With `--field <key>`, the rows are projected to `{"step_global": <int>, "<key>": <value>}`.

---

## What lives where on disk

```
<spike_archive>/
├── <run_id>.jsonl                # per-cell canonical log (one per seed × condition cell)
├── spike_*.json                  # SpikeRun envelope (ADR-016)
├── leaderboard.json              # ADR-008 leaderboard entry
└── rollouts/                     # per-rollout MP4 + sidecar JSONL (ADR-017 §Decisions D6)
    └── <condition>/
        ├── step_000000.mp4
        ├── step_000000.jsonl
        ├── step_025000.mp4
        ├── step_025000.jsonl
        └── ...
```

`<run_id>` is the 16-hex SHA from `concerto.training.logging.compute_run_metadata(seed, run_kind, repo_root)`. Two reruns with identical `(seed, git_sha, pyproject_hash, run_kind)` produce the same `run_id` — the W&B run name matches the JSONL filename byte-for-byte.

---

## Cross-references

- [ADR-017 — Observability and experiment tracking](../adr/ADR-017-observability-and-experiment-tracking.md) — design & rationale.
- [ADR-002 — RL framework](../adr/ADR-002-rl-framework.md) §Decisions — the structured-logging contract this slice extends.
- [ADR-016 — SpikeRun envelope schema](../adr/ADR-016-spike-run-schema.md) — the wire-format archive shape (NOT changed by P1.05.11).
- `src/concerto/training/logging.py` — `RunContext`, `bind_run_logger`, `log_scalars`, `log_eval`, `make_wandb_run_sink`.
- `src/chamber/observability/rollout_recorder.py` — `RolloutRecorder`.
- `src/chamber/cli/analyze.py` — `chamber-analyze`.
- `scripts/wandb_backfill.py` — backfill script.
