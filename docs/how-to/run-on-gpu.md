# How to run CONCERTO on a GPU host

!!! info "Coming in M4b-9b"

    The full reproduction recipe (Dockerfile, prerequisites, the
    zoo-seed run that publishes the M4a partner checkpoint) lands in
    M4b-9b. This page is a deliberate placeholder linked from
    [`docs/tutorials/hello-spike.md`](../tutorials/hello-spike.md)'s
    "On a GPU host" footer so the cross-reference doesn't break the
    `mkdocs --strict` build.

In the meantime, the Stage-0 training adapter
([`make_stage0_training_env`](../reference/api.md)) and the
[`stage0_smoke.yaml`](https://github.com/fsafaei/concerto/blob/main/configs/training/ego_aht_happo/stage0_smoke.yaml)
Hydra config are already wired up. On a Vulkan-capable Linux box with
the M4b-9b dependencies installed, the entry point will be:

```shell
make zoo-seed-gpu
```

That target is added in M4b-9b. Until then, this page documents the
contract:

- ADR-001 §Risks: SAPIEN / ManiSkill v3 require a Vulkan-capable GPU.
- CPU-only hosts will receive a clear
  `ChamberEnvCompatibilityError` from
  [`make_stage0_training_env`](../reference/api.md) — the same error
  shape [`make_stage0_env`](../reference/api.md) already emits.
- The training-side glue lives in
  [`chamber.benchmarks.stage0_smoke_adapter`](../reference/api.md);
  see ADR-001 §Validation criteria for the role split.
