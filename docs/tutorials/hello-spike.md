# Tutorial: hello-spike — 1k-frame ego-AHT training demo

This tutorial walks through running a tiny ego-AHT training loop end-to-end
on a CPU in roughly five minutes. You will see the env step, the partner
act, the trainer collect rollouts, and a checkpoint emit — every moving
part of the M4b training stack against a small but real task
(see [ADR-002 §Decisions][adr-002]).

## What you will build

A 1000-frame training run with:

- The 2-agent MPE Cooperative-Push env from
  [`chamber.envs.mpe_cooperative_push`][mpe-env].
- A scripted-heuristic frozen partner from M4a's draft zoo.
- The reference [`RandomEgoTrainer`][random-trainer] in place of the real
  HAPPO trainer (the HARL fork's `EgoAHTHAPPO` swaps in via the trainer-
  factory seam in PR M4b-7).
- A JSONL log carrying the per-step provenance bundle (run id, seed,
  git SHA, pyproject hash) and `.pt` checkpoints at frames 500 and 1000.

## Prerequisites

- A CONCERTO checkout with `make install` run (this installs the HARL
  fork via the SHA pin in `pyproject.toml` from PR M4b-6).
- No GPU required. Total wall-time: roughly five minutes on a modern
  laptop CPU.

## Run it

Save this as `hello_spike.py` in a clean directory and run it. It uses
the existing Hydra config under `configs/training/ego_aht_happo/` with
a few CLI-style overrides — `total_frames`, `checkpoint_every`, and
`happo.rollout_length` are scaled down so the demo fits a five-minute
laptop CPU budget; the production defaults from
`configs/training/ego_aht_happo/mpe_cooperative_push.yaml` are 100k /
10k / 1024 respectively.

```python
from pathlib import Path

from chamber.benchmarks.training_runner import run_training
from concerto.training.config import load_config

repo_root = Path(__file__).resolve().parent
configs_dir = repo_root / "configs"

cfg = load_config(
    config_path=configs_dir
    / "training"
    / "ego_aht_happo"
    / "mpe_cooperative_push.yaml",
    overrides=[
        "total_frames=1000",
        "checkpoint_every=500",
        "happo.rollout_length=250",
        "artifacts_root=./hello_spike_artifacts",
        "log_dir=./hello_spike_logs",
    ],
)
curve = run_training(cfg, repo_root=repo_root)

print(f"run_id={curve.run_id}")
print(f"steps={len(curve.per_step_ego_rewards)}")
print(f"episodes={len(curve.per_episode_ego_rewards)}")
print(f"checkpoints={len(curve.checkpoint_paths)}")
```

You should see output similar to:

```
run_id=4f1e8a2c0a3d62e5
steps=1000
episodes=20
checkpoints=2
```

The exact `run_id` will differ from machine to machine because the
provenance bundle includes the current git SHA + the hash of
`pyproject.toml` (see [`RunContext`][run-context]).

## Inspect the run

The JSONL log is a self-contained record: every line is a valid JSON
object carrying the run-level provenance plus the per-event `step`,
event name, and any payload fields. Substitute the `run_id` printed by
the script for `<run-id>` below.

```shell
head -2 hello_spike_logs/<run-id>.jsonl
```

Each checkpoint has a `.pt` payload plus a `.pt.json` sidecar with the
SHA-256 integrity digest and the producing run's metadata
(see [`save_checkpoint`][save-checkpoint] / [`load_checkpoint`][load-checkpoint]).
The `local://artifacts/...` URI scheme resolves under
`<artifacts_root>/artifacts/`, so the on-disk path doubles the
`artifacts/` segment — that is intentional and grep-able:

```shell
ls hello_spike_artifacts/artifacts
```

## What's next

The `RandomEgoTrainer` reference impl exercises the loop's plumbing but
does not learn — `update()` is a no-op and the per-episode reward is just
the partner's heuristic-guided trajectory. To demonstrate that the
algorithm is non-decreasing on per-epoch reward, swap in the HARL fork's
`EgoAHTHAPPO` via the trainer-factory seam: pass
`trainer_factory=EgoAHTHAPPO.from_config` to `run_training`. PR M4b-7
wires this default once the fork's `v0.1.0-aht` tag is pushed; the
empirical-guarantee experiment lands in PR M4b-8 and produces the plot
embedded in `docs/explanation/why-aht.md`.

[adr-002]: ../reference/adrs.md
[mpe-env]: ../reference/api.md
[random-trainer]: ../reference/api.md
[run-context]: ../reference/api.md
[save-checkpoint]: ../reference/api.md
[load-checkpoint]: ../reference/api.md
