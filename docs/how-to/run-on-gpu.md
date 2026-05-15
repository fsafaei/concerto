# How to run CONCERTO on a GPU host

This page is the recipe for reproducing the M4a draft-zoo seed
checkpoint on a Linux + CUDA box. The Mac-CPU tutorial in
[`tutorials/hello-spike.md`](../tutorials/hello-spike.md) covers
day-to-day development; this page is what an outside contributor with
a fresh CUDA-capable Linux box and Docker installed reads to
reproduce the canonical Phase-0 artefact.

Reproduction takes ~2 hours of GPU wall-time + ~10 minutes of build
time. It targets [ADR-001 §Validation criteria][adr-001] (the
rig-validated `panda_wristcam` + `fetch` + `allegro_hand_right` env)
and writes the Phase-0 M4a draft-zoo seed checkpoint
(`local://artifacts/happo_seed7_step50k.pt`, per
[ADR-009 §Decision][adr-009]).

## Prerequisites

- **Linux** (Ubuntu 22.04 or equivalent) — native, OR Windows 11 +
  WSL2 (Ubuntu) + Docker Engine. The WSL2 path is CUDA-only (no
  Vulkan); pass `-e CHAMBER_RENDER_BACKEND=none` on the `docker run`
  line as described in "Run the zoo-seed training" below.
- **NVIDIA GPU** with CUDA 12.x driver. The image uses the
  `nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04` base; minimum
  driver version per NVIDIA's CUDA compatibility table is **535.54**.
- **Docker** ≥ 24, with [**`nvidia-container-toolkit`**][nvidia-container]
  installed and configured so `--gpus all` works:

    ```shell
    docker run --rm --gpus all nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04 nvidia-smi
    ```

    If this prints `nvidia-smi` output, the GPU is wired correctly.
- **`make`**, **`git`**, **~12 GB of disk** for the image + caches.

If your box doesn't satisfy the above, the
[`tutorials/hello-spike.md`](../tutorials/hello-spike.md) CPU recipe
runs the same training stack against the `mpe_cooperative_push` env
on Mac / Linux laptop CPU in ~5 minutes.

## 1. Clone

```shell
git clone https://github.com/fsafaei/concerto.git
cd concerto
```

## 2. Build the GPU image

The repository ships two Dockerfiles:

- `Dockerfile.cpu` — Ubuntu 26.04 base; no CUDA. Used by the
  `.devcontainer/` Mac dev box. **Not** what you want here.
- `Dockerfile.gpu` — `nvidia/cuda` base with three stages
  (`dev` / `prod` / `gpu-host`). The `gpu-host` stage installs
  `libvulkan1` + `mesa-vulkan-drivers` + `libegl1` + `libgles2`,
  which SAPIEN / ManiSkill v3 need to initialise the rig
  ([ADR-001 §Risks][adr-001-risks]).

Build the `gpu-host` stage:

```shell
docker build -f Dockerfile.gpu --target gpu-host -t concerto:gpu .
```

First build takes ~10 minutes (driven by `uv sync --frozen
--no-dev`); subsequent builds reuse the apt + uv layer caches.

## 3. Smoke-test the rig

Inside a container against your GPU, run the ADR-001 Stage-0 smoke
test. This verifies the rig is talking to SAPIEN correctly before you
spend GPU wall-time on a 100k-frame training run.

```shell
docker run --rm --gpus all -v "$PWD:/workspace" -w /workspace \
    concerto:gpu make smoke
```

Expected: `Tier-2 real-SAPIEN` tests pass (the Mac CI run skips them
via `sapien_gpu_available()`; here they execute).

## 4. Run the zoo-seed training

This is the load-bearing step. It runs the 100k-frame ego-AHT HAPPO
training on the Stage-0 env at `seed=7` and publishes the step-50000
checkpoint under the canonical name [ADR-009 §Decision][adr-009]
specifies:

```shell
docker run --rm --gpus all -v "$PWD:/workspace" -w /workspace \
    concerto:gpu make zoo-seed-gpu
```

On Windows 11 + WSL2 + Docker Engine, add `-e
CHAMBER_RENDER_BACKEND=none` to the same command:

```shell
docker run --rm --gpus all \
    -e CHAMBER_RENDER_BACKEND=none \
    -v "$PWD:/workspace" -w /workspace \
    concerto:gpu make zoo-seed-gpu
```

`CHAMBER_RENDER_BACKEND` defaults to `"gpu"` so the original command
above still works unchanged on native Linux hosts with a real NVIDIA
Vulkan driver. Setting it to `"none"` disables SAPIEN's Vulkan render
path; CUDA still drives PyTorch (HAPPO training) and the Stage-0 env
uses `obs_mode="state_dict"` so no rendered frames are ever requested.

What the target does (`scripts/repro/zoo_seed.sh`):

1. Probes the device. Loud-fails (exit 2) if `torch_device() !=
   "cuda"`.
2. Invokes
   `chamber-spike train --config configs/training/ego_aht_happo/stage0_smoke.yaml`.
3. On training success, copies
   `./artifacts/artifacts/<run_id>_step50000.pt` (+ sidecar) to the
   M4a-contract path
   `./artifacts/artifacts/happo_seed7_step50k.pt`.
4. Computes the SHA-256 of the published `.pt` and writes it to
   `scripts/repro/artifacts/happo_seed7_step50k.pt.sha256`.
5. Prints a closing line directing you to commit the SHA file via
   the Branch-5 manifest PR.

Exit codes:

| Code | Meaning |
|---|---|
| 0 | Training completed, artefact published, SHA written. |
| 1 | Training failed or the published artefact wasn't found. |
| 2 | CPU-only host (`torch_device() != "cuda"`). |

Total GPU wall-time on a single consumer GPU: ~2 hours, plus
~1 minute of artefact handling.

> **Why is `--check-guarantee` not used here?**
>
> The Stage-0 smoke env (`make_stage0_training_env` /
> `_Stage0SmokeEnv`) loads an empty scene with no task and no
> reward function — it exists to validate the rig (wrapper
> dispatch, action routing, observation namespacing, GPU + CUDA
> + SAPIEN + Vulkan integration), per [ADR-001 §Validation
> criteria][adr-001]. Reward stays at 0.0, so the slope test in
> `concerto.training.empirical_guarantee` would always trip with
> `slope = 0`, `passed = False`.
>
> The empirical-guarantee trip-wire is the [ADR-002 risk-mitigation
> #1][adr-002] gate for *learnable* tasks; running it on a zero-
> reward rig-validation env is a category error. The Stage-1 spike
> protocol ([ADR-007 §Stage 1][adr-007]) is where the gap test
> against a learnable task happens, and the Stage-1 runner wires
> `--check-guarantee` into its own script.

## 5. Publish + verify

After the run, you should have on disk:

- `./artifacts/artifacts/happo_seed7_step50k.pt` (the .pt payload)
- `./artifacts/artifacts/happo_seed7_step50k.pt.json` (the SHA-256
  + provenance sidecar, per
  [`save_checkpoint`](../reference/api.md))
- `scripts/repro/artifacts/happo_seed7_step50k.pt.sha256` (the
  hex digest, ready for `git commit`)

Commit the `.sha256` file and the `.pt` + sidecar (the M4a Phase-0
draft-zoo artefacts the project hosts out-of-tree per
[plan/04 §3.8](../reference/api.md)) via the Branch-5 manifest PR
(T4b.14). After that PR merges, future contributors verify the
artefact integrity with:

```shell
make zoo-seed-verify
```

which loads the `.pt` through
[`load_checkpoint`](../reference/api.md) and asserts the recomputed
SHA-256 matches the committed manifest.

## Troubleshooting

- **`Cannot connect to the Docker daemon`** — Docker is not running.
  Start it with `sudo systemctl start docker` (Linux) or via Docker
  Desktop.
- **`could not select device driver "" with capabilities: [gpu]`** —
  `nvidia-container-toolkit` is not installed or not configured. See
  [NVIDIA's install guide][nvidia-container].
- **`ChamberEnvCompatibilityError: SAPIEN/Vulkan initialisation failed`** —
  the container started but couldn't reach the GPU's Vulkan stack.
  On native Linux hosts with NVIDIA Vulkan drivers, check the host
  driver version (must be ≥535.54) and that `--gpus all` is on the
  `docker run` line. On hosts where CUDA is available but Vulkan is
  not (e.g. Windows 11 + WSL2 + Docker Engine), set
  `CHAMBER_RENDER_BACKEND=none` via `docker run -e
  CHAMBER_RENDER_BACKEND=none …`; the simulator's render path is
  skipped (the env uses `obs_mode="state_dict"` so no rendered frames
  are needed) while CUDA continues to drive HAPPO training.

[adr-001]: ../reference/adrs.md
[adr-001-risks]: ../reference/adrs.md
[adr-002]: ../reference/adrs.md
[adr-007]: ../reference/adrs.md
[adr-009]: ../reference/adrs.md
[nvidia-container]: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
