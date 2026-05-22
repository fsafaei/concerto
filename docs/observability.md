# Observability cookbook (P1.05.11; ADR-017)

This cookbook covers the workflows P1.05.11 enables — live W&B dashboards
for the founder + JSON-first co-analysis for the agent. The design lives
in [`ADR-017`](https://github.com/fsafaei/concerto/blob/main/adr/ADR-017-observability-and-experiment-tracking.md);
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

## Cookbook recipes

Worked examples with real run IDs from
`spikes/results/stage1-failure-investigation/2026-05-20/` — the five
post-launch Stage-1b cells (seeds 0–4) committed before P1.05.11
landed. Pre-P1.05.11 archives carry the envelope-level events
(`training_start`, `safety_telemetry`, `rollout_update`,
`safety_telemetry_final`, `training_end`) but **not** per-step PPO
scalars (`policy_loss`, `dist_entropy`, `approx_kl`, …) or per-step
rollout sidecar JSONLs — those are emitted only by cells launched
after this slice merged with `observability.wandb.enabled=true`. The
recipes note when an example is forward-looking vs grounded in the
committed archive.

### Cookbook recipe 1 — First-instrumented-cell walkthrough

End-to-end: edit the config, launch a Stage-1b cell with the W&B sink
on, view results in the browser, find the artefacts on disk.

**Step 1 — Edit the config (or use a per-run override).**

Persistent (edit
`configs/training/ego_aht_happo/stage1_pickplace.yaml`):

```yaml
wandb:
  enabled: true
  project: concerto-chamber   # ADR-017 §Decisions D2 — single project
rollout_recorder:
  enabled: true
  interval_frames: 25000      # record one eval episode every 25k frames
  fps: 20
```

Or per-run (no config edit; useful for one-off experiments):

```bash
uv run chamber-spike train \
    --config configs/training/ego_aht_happo/stage1_pickplace.yaml \
    --override wandb.enabled=true \
    --override rollout_recorder.enabled=true
```

**Step 2 — Launch the cell.**

```bash
make stage1-as       # or: uv run chamber-spike train ...
```

During training the canonical record is the per-cell
`logs/<run_id>.jsonl`. W&B receives the same events live; you can
open the dashboard before the cell completes and watch the curves
update.

**Step 3 — View in W&B (founder-side).**

Browse `https://wandb.ai/<entity>/concerto-chamber`. The cell appears
as a run named after its 16-hex `run_id` (derived deterministically
from `(seed, git_sha, pyproject_hash, run_kind)`; see
`concerto.training.logging.compute_run_metadata`). Default chart
group is `train/` (PPO health scalars); `safety/` carries
`lambda_mean/var/min/max`; `eval/` carries `success_rate` and
`mean_episode_*` per eval-cell completion. The run's Overview panel
shows the `wandb.config` fields: `run_id`, `seed`, `git_sha`,
`pyproject_hash`, `run_kind`, `prereg_sha` (full SHA; the short-form
also appears as a tag — see §Decisions D10).

**Step 4 — Find artefacts on disk (agent-side).**

```
<cfg.log_dir>/                     # default ./logs/
└── <run_id>.jsonl                 # canonical per-cell log

<spike_archive>/                   # default spikes/results/<spike-id>/
├── spike_as.json                  # SpikeRun envelope (ADR-016)
├── leaderboard.json               # ADR-008 leaderboard entry
└── rollouts/                      # paired MP4 + sidecar JSONL artefacts
    └── as-hetero/                 # one sub-dir per condition_id
        ├── step_025000.mp4        # founder-side: drag into W&B Reports / PR
        ├── step_025000.jsonl      # agent-side: parse with chamber-analyze
        ├── step_050000.mp4
        ├── step_050000.jsonl
        └── …
```

The W&B run name (16-hex `run_id`) matches the JSONL filename
byte-for-byte. The agent's `chamber-analyze` subcommands work
against the JSONLs alone — no W&B account needed.

### Cookbook recipe 2 — Comparing two runs (worked: 2026-05-20 archive)

Side-by-side comparison across N runs. Worked example uses the two
lowest-seed cells from the committed 2026-05-20 failure-investigation
archive (`43a0d043cb9f54ac` = seed 0; `e3ca669746e8f4bb` = seed 1).
The actual numbers reflect the pre-P1.05.8 Surface 1 rubber-stamp
bug (every episode trivially passed under the old success rule); the
recipe is structural — same commands work against post-merge cells
that produce honest metrics.

**Step 1 — List candidates.**

```bash
uv run chamber-analyze list-runs \
    --archive-root spikes/results/stage1-failure-investigation/2026-05-20 \
    --json | jq '.[] | {run_id, seed, terminal_success_rate, lambda_steady_state}'
```

Expected:

```json
{"run_id": "43a0d043cb9f54ac", "seed": 0, "terminal_success_rate": 1.0, "lambda_steady_state": -7.0}
{"run_id": "e3ca669746e8f4bb", "seed": 1, "terminal_success_rate": 1.0, "lambda_steady_state": -7.0}
{"run_id": "ef23e3b576f38c91", "seed": 2, "terminal_success_rate": 1.0, "lambda_steady_state": -7.0}
{"run_id": "a7b549a31fbd8222", "seed": 3, "terminal_success_rate": 1.0, "lambda_steady_state": -7.0}
{"run_id": "da40f205c78355ac", "seed": 4, "terminal_success_rate": 1.0, "lambda_steady_state": -7.0}
```

**Step 2 — Compare two cells head-to-head.**

```bash
uv run chamber-analyze compare 43a0d043cb9f54ac e3ca669746e8f4bb \
    --archive-root spikes/results/stage1-failure-investigation/2026-05-20 \
    --metrics mean_reward_final,terminal_success_rate,lambda_steady_state \
    --json
```

Expected:

```json
[
  {"run_id": "43a0d043cb9f54ac",
   "mean_reward_final": null,
   "terminal_success_rate": 1.0,
   "lambda_steady_state": -7.0},
  {"run_id": "e3ca669746e8f4bb",
   "mean_reward_final": null,
   "terminal_success_rate": 1.0,
   "lambda_steady_state": -7.0}
]
```

`mean_reward_final` is `null` here because the 2026-05-20 archive's
`rollout_update` events serialised the per-step reward as the string
repr `"tensor([3.18e-05])"` rather than a numeric field — pre-P1.05.11
type-bug, not slice-induced. Cells launched post-merge emit
`mean_episode_reward` numerically via the new `event="eval"` lines
and surface here as a float.

**Step 3 — Drill into a single scalar.**

```bash
uv run chamber-analyze metrics 43a0d043cb9f54ac \
    --archive-root spikes/results/stage1-failure-investigation/2026-05-20 \
    --metric lambda_mean --namespace safety --json | jq '.[0:5]'
```

Expected first 5 of ~100 rollout-window stats:

```json
[
  {"step": 1024, "wall_time": null, "value": -0.2679},
  {"step": 2048, "wall_time": null, "value": -0.7802},
  {"step": 3072, "wall_time": null, "value": -1.4115},
  {"step": 4096, "wall_time": null, "value": -2.0428},
  {"step": 5120, "wall_time": null, "value": -2.6741}
]
```

`wall_time` is `null` for pre-P1.05.11 archives; post-merge cells
carry an ISO-8601 stamp on every emitted line.

### Cookbook recipe 3 — Per-step rollout analysis

Find the step in a recorded eval episode where a key `info` flag
flips (e.g. `is_grasped` goes False → True), and read the
`obs_summary` fields at that step. This recipe is **forward-looking**:
the 2026-05-20 committed archive pre-dates the recorder, so the
sidecar JSONLs at `<archive>/rollouts/<condition>/step_*.jsonl` do
not exist there. Any cell launched post-P1.05.11 with
`rollout_recorder.enabled=true` produces them.

**Step 1 — Find a recorded rollout.**

```bash
# List the rollout sidecar JSONLs for one run's archive.
find <archive_root>/rollouts -name "step_*.jsonl" | head
```

Expected layout:

```
<archive_root>/rollouts/as-hetero/step_025000.jsonl
<archive_root>/rollouts/as-hetero/step_050000.jsonl
<archive_root>/rollouts/as-hetero/step_075000.jsonl
…
```

**Step 2 — Dump per-step records for one rollout via `chamber-analyze`.**

```bash
uv run chamber-analyze rollout-frames <run_id> \
    --archive-root <archive_root> \
    --episode 50000 | jq '.[0]'
```

Expected first record (illustrative; one line per env step):

```json
{
  "event": "rollout_step",
  "metric_namespace": "rollout",
  "step_global": 50000,
  "step_episode": 0,
  "condition": "as-hetero",
  "seed": 0,
  "wall_time": "2026-05-22T15:54:48.752+00:00",
  "obs_summary": {
    "cube_pose": [0.073, 0.000, 0.020, 1.000, 0.000, 0.000, 0.000],
    "goal_pos": [0.000, 0.100, 0.250],
    "tcp_pose": [0.250, 0.000, 0.500, 1.000, 0.000, 0.000, 0.000],
    "ego_qpos": [0.000, -0.785, 0.000, -2.356, 0.000, 1.571, 0.785, 0.040],
    "ego_qvel": [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
    "partner_qpos": [0.350, 0.200, 0.000, 0.000, 0.500, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
    "partner_qvel": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "is_grasped": false,
    "gripper_width": 0.080
  },
  "action": [0.10, 0.00, -0.05, 0.00, 0.00, 0.00, 0.00, 0.00],
  "reward": 0.0,
  "terminated": false,
  "truncated": false,
  "info": {"is_grasped": false, "elapsed_steps": 0},
  "frame_index": 0
}
```

**Step 3 — Find the step where `is_grasped` flips.**

```bash
uv run chamber-analyze rollout-frames <run_id> \
    --archive-root <archive_root> \
    --episode 50000 \
    --field is_grasped | jq '[.[] | select(.is_grasped == true)] | .[0]'
```

Expected (first step where the gripper closed on the cube):

```json
{"step_global": 50018, "is_grasped": true}
```

**Step 4 — Read the surrounding `obs_summary` at that step.**

```bash
uv run chamber-analyze rollout-frames <run_id> \
    --archive-root <archive_root> \
    --episode 50000 | jq '.[] | select(.step_global == 50018) | .obs_summary'
```

Expected:

```json
{
  "cube_pose": [0.075, 0.001, 0.038, 1.000, 0.000, 0.000, 0.000],
  "goal_pos": [0.000, 0.100, 0.250],
  "tcp_pose": [0.075, 0.001, 0.038, 1.000, 0.000, 0.000, 0.000],
  "ego_qpos": [0.103, -0.612, -0.005, -2.182, 0.012, 1.520, 0.802, 0.000],
  "ego_qvel": [0.012, 0.041, 0.003, 0.022, 0.001, -0.005, 0.001, -2.310],
  "partner_qpos": [0.350, 0.200, 0.000, 0.000, 0.500, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
  "partner_qvel": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  "is_grasped": true,
  "gripper_width": 0.000
}
```

Note the TCP and cube poses are co-located (the gripper closed on
the cube); `gripper_width` dropped to 0; the rapid `ego_qvel[7]`
negative spike is the gripper-close velocity. The agent now has a
mechanistic picture of the grasp moment without watching the MP4.

The `obs_summary` carries a curated subset of fields per ADR-017
§Decisions D6 — never the full 65-D flattened ego state. Extending
the field set requires editing
`chamber.observability.rollout_recorder._{OBS_EXTRA,AGENT_SUB,INFO}_KEYS`
and rebuilding any archived rollouts.

### Cookbook recipe 4 — Embedding a comparison plot in a PR description

Render a multi-run scalar comparison to PNG and attach it to a GitHub
PR description.

**Step 1 — Generate the PNG via `chamber-analyze plot`.**

```bash
uv run chamber-analyze plot 43a0d043cb9f54ac e3ca669746e8f4bb \
    --archive-root spikes/results/stage1-failure-investigation/2026-05-20 \
    --metric lambda_mean --namespace safety \
    --out /tmp/lambda_seed0_vs_seed1.png \
    --title "lambda_mean: seed 0 vs seed 1 (Stage-1b post-launch, 2026-05-20)"
```

`plot` uses matplotlib's Agg backend (headless-safe; works in CI and
on remote hosts without an X display). Output PNG is suitable for
direct embed.

**Step 2 — Attach to GitHub via drag-and-drop OR via `gh`.**

*Browser path.* Drag the PNG into a GitHub PR description editor; the
editor uploads to `user-images.githubusercontent.com` and inserts
`![filename.png](https://user-images.githubusercontent.com/.../filename.png)`
at the cursor.

*CLI path.* Edit the PR body via `gh pr edit --body-file`:

```bash
# Compose the body (which references the eventual attachment URL).
cat <<'EOF' > /tmp/pr_body.md
## Lambda divergence — seed 0 vs seed 1

![lambda comparison](https://user-images.githubusercontent.com/<UID>/<HASH>-lambda_seed0_vs_seed1.png)

The two seeds' λ trajectories diverge at ~25k frames; seed 0 saturates the
clamp by 50k while seed 1's overshoot is bounded ~6% lower.
EOF

# Upload the image via GitHub's web UI (manual step), then paste the
# user-images URL into the body above before:
gh pr edit <pr-number> --body-file /tmp/pr_body.md
```

**Step 3 — Inline the chart in a Markdown README or ADR.**

For permanently-embedded charts (vs ephemeral PR-description images),
commit the PNG under `docs/assets/` and reference via a relative
path:

```markdown
![](../assets/lambda_seed0_vs_seed1.png)
```

mkdocs renders this against the docs/ build root; GitHub's source-view
renders it against the file's own location. Both work.

---

## How-to guides

### How to enable W&B for a spike run

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

Degrade-to-no-op paths (each emits one `UserWarning`, never raises) —
the canonical full list is documented in §Environment variables below.

### How to back-fill historical archives

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

Idempotent: re-running against the same archive skips existing W&B
runs (`wandb.init(id=<run_id>, resume="never")`).

**What backfill does deliver.** Replays per-cell JSONL events
(`training_start`, `safety_telemetry`, `rollout_update`,
`safety_telemetry_final`, `training_end`) + envelope-level terminal
metrics (per-episode `success`, `mean_reward`,
`constraint_violation_peak`, `fallback_fired`) into W&B.

**What backfill does NOT deliver.** Per-step PPO health curves
(`policy_loss`, `value_loss`, `dist_entropy`, `approx_kl`,
`clip_fraction`, `grad_norm`) for historical archives — those
scalars were never written to disk before P1.05.11. They are only
available for runs going forward.

---

## Reference

### Config schema reference

The two Pydantic blocks P1.05.11 adds, with every field, its default,
and a one-line semantic. Both blocks live in
`concerto.training.config` and round-trip through Hydra YAML.

**`WandbConfig`** (`src/concerto/training/config.py:46-69`):

```yaml
wandb:
  # Master switch. When False (default), bind_run_logger is built
  # without a W&B sink and only the JSONL fallback fires. CI runs
  # and unit tests should leave this off.
  enabled: false

  # W&B project name. Single project across all stages per
  # ADR-017 §Decisions D2; tag-based filtering separates runs
  # in the UI by stage:/sub_stage:/condition:/prereg:/backfill:.
  # Pre-ADR-017 default was "concerto-m4b" (migrated 2026-05-22).
  project: concerto-chamber
```

**`RolloutRecorderConfig`** (`src/concerto/training/config.py:295-339`):

```yaml
rollout_recorder:
  # Master switch. When False (default), the eval-cell loop skips
  # frame capture, the env factory does not request
  # render_mode="rgb_array", and finalise() short-circuits.
  enabled: false

  # Record one eval episode every K training frames. Default
  # 25_000 frames ≈ 4 recordings over a 100k Stage-1b cell.
  interval_frames: 25000

  # When True (default), emit the paired per-step sidecar JSONL
  # alongside the MP4. When False, MP4-only. The agent-side
  # co-analysis workflow assumes per_step_jsonl=true.
  per_step_jsonl: true

  # How many eval episodes to record at each interval trigger.
  episodes_per_record: 1

  # Video frame rate for the MP4 encode. Default 20 matches
  # ManiSkill's Stage-1b control_freq.
  fps: 20
```

Both blocks are `_FrozenModel` subclasses (frozen + `extra="forbid"`)
so YAML typos fail loud with `pydantic.ValidationError` at
`load_config()` time.

### Environment variables

P1.05.11 reads three W&B environment variables. None are mandatory;
each missing or misconfigured value resolves to a documented
degrade-to-no-op path per ADR-017 §Decisions D4.

| Env var | Default | Effect when set | Effect when missing |
|---------|---------|------------------|----------------------|
| `WANDB_API_KEY` | (none) | Authenticates the W&B online client. Required for `WANDB_MODE=online`. | If `cfg.wandb.enabled=true` AND `WANDB_MODE != "offline"`: `make_wandb_run_sink` emits a `UserWarning` and returns `None`; the run continues with JSONL only. |
| `WANDB_MODE` | `online` | `online` uploads to W&B servers; `offline` writes to local `./wandb/` directory only; `disabled` skips both. `offline` is what the equivalence smoke + the integration test use. | Defaults to `online`. |
| `WANDB_ENTITY` | (login default) | Overrides the W&B entity (user or team) the run lands under. | Uses the entity tied to the active `wandb login` session. |

The complete degrade-to-no-op decision tree
(`concerto.training.logging.make_wandb_run_sink`):

| Condition | Result |
|-----------|--------|
| `cfg.wandb.enabled=false` | Silent — JSONL-only path, no warning. |
| `cfg.wandb.enabled=true` AND `WANDB_API_KEY` missing AND `WANDB_MODE != "offline"` | `UserWarning("WANDB_API_KEY missing …")`; JSONL-only. |
| `cfg.wandb.enabled=true` AND `WANDB_MODE=offline` | Constructs the sink; writes to `./wandb/` locally; no upload. |
| `import wandb` fails (ImportError) | `UserWarning("`import wandb` failed …")`; JSONL-only. |
| `wandb.init(...)` raises | `UserWarning("wandb.init(...) raised …")`; JSONL-only. |
| All checks pass | Live W&B run; tags + config payload from `make_wandb_run_sink`'s arguments. |

The canonical record is **always** `<cfg.log_dir>/<run_id>.jsonl`.
W&B is downstream of that. If your run completed and the JSONL
exists, the run is valid (ADR-017 §Decisions D1).

### JSON output schemas

Every `chamber-analyze` subcommand supports `--json`. This is the
**agent-co-analysis contract** (ADR-017 §Decisions D5); schemas are
public-API-stable across patch versions. Adding fields is allowed;
removing or renaming a field requires a major-version bump on the
CLI. Treat the field names listed here as load-bearing.

#### `chamber-analyze list-runs --json`

Array of per-cell `_RunSummary` records (one per `<run_id>.jsonl`
under `--archive-root`). Fields without source data in the archive
resolve to `null` rather than being omitted so the JSON shape is
stable across archive vintages.

```json
[
  {
    "run_id": "<16-hex>",
    "jsonl_path": "<absolute-path>",
    "stage": "1" | "2" | "3" | null,
    "sub_stage": "1a" | "1b" | "2" | "3" | null,
    "condition": "<condition_id-or-pair>" | null,
    "seed": <int> | null,
    "task": "stage1_pickplace" | "mpe_cooperative_push" | "stage0_smoke" | null,
    "run_kind": "ego_aht_happo" | null,
    "git_sha": "<40-hex>" | null,
    "n_steps": <int> | null,
    "terminal_success_rate": <float in [0, 1]> | null,
    "mean_reward_final": <float> | null,
    "lambda_steady_state": <float> | null
  },
  ...
]
```

Sort order: by `seed` ascending, then `run_id` ascending. Stable
across invocations against the same archive.

#### `chamber-analyze summary <run_id> --json`

One `_RunSummary` record (same shape as a single element of
`list-runs`). Exit code `2` if the `run_id` is not found.

#### `chamber-analyze metrics <run_id> --metric <name> --json`

Array of time-series points for one metric on one run. Filtered by
`--namespace` (one of `train`, `eval`, `safety`, `hardware`, `rollout`).

```json
[
  {
    "step": <int>,
    "wall_time": "<iso8601-with-tz>" | null,
    "value": <float>
  },
  ...
]
```

`step` is always present and is the global training-frame counter
at emission. `wall_time` is `null` for pre-P1.05.11 archives.

#### `chamber-analyze compare <run_id_a> <run_id_b> [...] --metrics M1,M2 --json`

Array of one record per requested `run_id` (in input order). Each
record carries `run_id` plus one field per metric. Unknown
`run_id`s are kept in the array with an `"error": "not_found"`
field so the response shape matches the request length.

```json
[
  {
    "run_id": "<16-hex>",
    "<metric_1>": <number> | null,
    "<metric_2>": <number> | null,
    ...
  },
  ...
]
```

Default metric set when `--metrics` is omitted:
`mean_reward_final`, `terminal_success_rate`,
`lambda_steady_state`.

#### `chamber-analyze plot ... --out <path.png>`

Side effect only (writes the PNG to `--out`). Exits 0 on success,
2 if matplotlib isn't installed (`uv sync --extra viz` resolves).
No JSON output.

#### `chamber-analyze rollout-frames <run_id> --json`

Two output shapes depending on whether `--field <key>` is passed:

*Without `--field`:* the full per-step JSONL line:

```json
[
  {
    "event": "rollout_step",
    "metric_namespace": "rollout",
    "step_global": <int>,
    "step_episode": <int>,
    "condition": "<condition>",
    "seed": <int>,
    "wall_time": "<iso8601>",
    "obs_summary": {"cube_pose": [...], "goal_pos": [...], ...},
    "action": [<float>, ...],
    "reward": <float>,
    "terminated": <bool>,
    "truncated": <bool>,
    "info": {...},
    "frame_index": <int>
  },
  ...
]
```

*With `--field <key>`:* each row projects to two fields —
`step_global` and the requested key (looked up in
`obs_summary` first, then `info`):

```json
[
  {"step_global": <int>, "<key>": <value> | null},
  ...
]
```

The lookup order (`obs_summary` then `info`) means `is_grasped`
resolves from `obs_summary.is_grasped` when the env's per-step info
exposes it there; if the recorder's `obs_summary` selector misses
the key, the `info` mirror catches it.

### Filesystem layout

```
<cfg.log_dir>/
└── <run_id>.jsonl                # per-cell canonical log

<spike_archive>/                  # default spikes/results/<spike-id>/
├── <run_id>.jsonl                # one per (seed, condition) cell
├── spike_*.json                  # SpikeRun envelope (ADR-016)
├── leaderboard.json              # ADR-008 leaderboard entry
└── rollouts/                     # ADR-017 §Decisions D6 sidecars
    └── <condition>/              # one sub-dir per condition_id
        ├── step_000000.mp4
        ├── step_000000.jsonl
        ├── step_025000.mp4
        ├── step_025000.jsonl
        └── ...
```

`<run_id>` is the 16-hex SHA from
`concerto.training.logging.compute_run_metadata(seed, run_kind, repo_root)`.
Two reruns with identical `(seed, git_sha, pyproject_hash, run_kind)`
produce the same `run_id` — the W&B run name matches the JSONL
filename byte-for-byte.

---

## Troubleshooting

### `env.render()` returns `None`; rollout MP4s not generated

**Symptom.** `chamber.observability.RolloutRecorder.finalise()` returns
`(None, jsonl_path)` instead of `(mp4_path, jsonl_path)`. ManiSkill
logs the warning *"Requested to use render device 'sapien_cuda', but
CUDA device was not found. Falling back to 'cpu' device. Rendering
might be disabled."* The per-step sidecar JSONL still emits cleanly;
the agent-side co-analysis path is unaffected.

**Root cause.** NVIDIA driver/library version mismatch. `nvidia-smi`
reports *"Failed to initialize NVML: Driver/library version
mismatch (NVML library version: 535.x)."* The userspace
`libnvidia-ml.so` upgraded but the kernel's `nvidia.ko` module did
not — a classic post-kernel-upgrade-without-reboot state.
`torch.cuda.is_available()` returns `False`. The SAPIEN renderer
falls back from `sapien_cuda` to CPU and returns `None` from
`render(mode="rgb_array")`.

**Diagnostic steps.**

```bash
nvidia-smi              # "Failed to initialize NVML" → driver mismatch
uv run python -c "import torch; print(torch.cuda.is_available())"   # False
```

**Recovery.**

1. **Reboot the host.** This re-syncs the kernel module against the
   updated userspace library. `nvidia-smi` should report a healthy
   driver/CUDA version table after restart.
2. **Re-verify rendering.**

   ```bash
   uv run python -c "
   import numpy as np
   from chamber.envs.stage1_pickplace import make_stage1_pickplace_env
   env = make_stage1_pickplace_env(
       condition_id='stage1_pickplace_panda_plus_fetch_ego_aht_happo_per_agent',
       episode_length=20,
       root_seed=0,
       render_mode='rgb_array',
   )
   env.reset(seed=0)
   frame = env.render()
   print(type(frame).__name__, getattr(frame, 'shape', None))
   "
   # Expect: ndarray (H, W, 3)
   ```

3. **Flip the env-render xfail-strict test.** The integration test at
   `tests/integration/test_rollout_recorder_env_render.py` is marked
   `@pytest.mark.xfail(strict=True)` for exactly this host state.
   Run it post-reboot:

   ```bash
   uv run pytest tests/integration/test_rollout_recorder_env_render.py --no-cov -v
   # XPASSED → strict=True fails the run → forces marker removal.
   ```

   When the test xpasses, **remove the `@pytest.mark.xfail` marker in
   the same commit** that re-runs it (lifecycle hygiene precedents:
   ADR-004 §Revision history 2026-05-17 / P1.02 `Bounds.action_norm`;
   ADR-002 §Revision history 2026-05-21 / P1.05.8 trainer obs
   reader). The follow-up commit subject:
   `test(observability): remove env-render xfail post-host-reboot`.

**Related tracking issue.**
[#188](https://github.com/fsafaei/concerto/issues/188) — the
`sapien_gpu_available()` predicate is looser than what
`test_draft_zoo`'s `skipif` actually needs (a tighter
`sapien_cuda_renderer_available()` check would prevent the
downstream `KeyError: 'state'` on this host state). Not slice-induced;
filed during P1.05.11's push-gate failure review.

---

## Cross-references

- [ADR-017 — Observability and experiment tracking](https://github.com/fsafaei/concerto/blob/main/adr/ADR-017-observability-and-experiment-tracking.md) — design & rationale.
- [ADR-002 — RL framework](https://github.com/fsafaei/concerto/blob/main/adr/ADR-002-rl-framework.md) §Decisions — the structured-logging contract this slice extends.
- [ADR-016 — SpikeRun envelope schema](https://github.com/fsafaei/concerto/blob/main/adr/ADR-016-spike-run-schema.md) — the wire-format archive shape (NOT changed by P1.05.11).
- `src/concerto/training/logging.py` — `RunContext`, `bind_run_logger`, `log_scalars`, `log_eval`, `make_wandb_run_sink`.
- `src/concerto/training/config.py` — `WandbConfig`, `RolloutRecorderConfig`.
- `src/chamber/observability/rollout_recorder.py` — `RolloutRecorder`.
- `src/chamber/cli/analyze.py` — `chamber-analyze`.
- `scripts/wandb_backfill.py` — backfill script.
