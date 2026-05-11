# HARL fork — creation recipe (T4b.1, T4b.4–T4b.7)

This directory ships the recipe + scaffolding to create
[`concerto-org/harl-fork`](https://github.com/concerto-org/harl-fork) from
[`PKU-MARL/HARL`](https://github.com/PKU-MARL/HARL) per ADR-002 §Decisions
and plan/05 §2 ("Hard fork at `concerto-org/harl-fork`, pinned by SHA in
`pyproject.toml`. Upstream changes pulled manually with PR-tagged sync
commits").

The recipe is mechanical: copy three files into a fresh fork, commit, tag,
push. The CONCERTO-side wiring (`pyproject.toml` SHA pin + integration
test) lands in two follow-up PRs (M4b-6 and M4b-7).

## Step-by-step

1. **Fork upstream.** On GitHub, create the
   [`concerto-org`](https://github.com/concerto-org) organisation if it does
   not exist yet, then fork
   [`PKU-MARL/HARL`](https://github.com/PKU-MARL/HARL) into it as
   `concerto-org/harl-fork`.
2. **Clone the fresh fork.**
   ```
   git clone git@github.com:concerto-org/harl-fork.git
   cd harl-fork
   ```
3. **Tag the upstream HEAD as `v0.0.0-vendored`.** Plan/05 §2: the first
   fork commit applies *zero* patches; this tag exists so we have a
   bisectable baseline if we ever need to rebase onto upstream.
   ```
   git tag v0.0.0-vendored
   git push origin v0.0.0-vendored
   ```
   Capture the SHA of `v0.0.0-vendored` (`git rev-parse v0.0.0-vendored`)
   and report it back to CONCERTO. PR M4b-6 will pin it in
   `pyproject.toml`.
4. **Apply the v0.1.0-aht patch.** Copy every file under
   `v0.1.0-aht/` into the fork's tree at the same relative path:
   ```
   v0.1.0-aht/harl/algorithms/actors/ego_aht_happo.py
       -> harl/algorithms/actors/ego_aht_happo.py
   v0.1.0-aht/harl/runners/ego_aht_runner.py
       -> harl/runners/ego_aht_runner.py
   v0.1.0-aht/harl/envs/concerto_env_adapter.py
       -> harl/envs/concerto_env_adapter.py
   v0.1.0-aht/tests/test_ego_aht_happo.py
       -> tests/test_ego_aht_happo.py
   ```
5. **Verify the `# UPSTREAM-VERIFY:` markers.** The new files reference
   upstream HARL classes and methods (`harl.algorithms.actors.happo.HAPPO`,
   the runner's expected env API, etc.). Each spot where the CONCERTO patch
   guesses at the upstream signature is tagged with a
   `# UPSTREAM-VERIFY:` comment. Skim the upstream HARL HEAD and either
   confirm the guess is correct or adjust the patch in place.
6. **Commit + tag + push.** Use the message in `v0.1.0-aht/COMMIT_MSG.txt`:
   ```
   git add harl/algorithms/actors/ego_aht_happo.py \
           harl/runners/ego_aht_runner.py \
           harl/envs/concerto_env_adapter.py \
           tests/test_ego_aht_happo.py
   git commit -F <path-to-CONCERTO>/scripts/harl-fork-patches/v0.1.0-aht/COMMIT_MSG.txt
   git tag v0.1.0-aht
   git push origin main v0.1.0-aht
   ```
   Capture the SHA of `v0.1.0-aht` (`git rev-parse v0.1.0-aht`) and report
   it back to CONCERTO. PR M4b-7 will bump `pyproject.toml` to it and add
   the integration test.

## What ships in `v0.1.0-aht/`

| File | Purpose |
|------|---------|
| `harl/algorithms/actors/ego_aht_happo.py` | `EgoAHTHAPPO(HAPPO)`. Validates the partner is frozen at construction; ego-only advantage decomposition; reduces to single-agent PPO update. ADR-002 §Decisions; ADR-009 §Consequences. |
| `harl/runners/ego_aht_runner.py` | Hydra-driven runner. Bridges to `chamber.benchmarks.training_runner.run_training` via the trainer-factory seam from M4b-5. |
| `harl/envs/concerto_env_adapter.py` | Thin adapter from CONCERTO's Gymnasium-multi-agent env to the env API the runner expects. |
| `tests/test_ego_aht_happo.py` | Subclassing + frozen-partner-validation smoke tests; runnable via the fork's CI. |
| `COMMIT_MSG.txt` | Conventional Commits message body for the v0.1.0-aht commit. |

## Inheritance contract (CONCERTO ↔ harl-fork)

The fork's `EgoAHTHAPPO` is **reference scaffolding** for external
researchers who want a HARL-fork-internal HAPPO subclass with a frozen-
partner guard. The bodies of `EgoAHTHAPPO.from_config`, `collect_rollout`,
and `update` are intentionally `NotImplementedError` — the architectural
rule (plan/05 §3.5) is that *rollout / update / training logic lives on
the CONCERTO side*, not in the fork.

CONCERTO's training run does **not** depend on the fork's
`EgoAHTHAPPO.from_config` body. Instead, M4b-8a ships
`chamber.benchmarks.ego_ppo_trainer.EgoPPOTrainer`, which:

- Imports `harl.algorithms.actors.happo.HAPPO` directly (the upstream
  actor — no fork-specific class required).
- Wraps it with a hand-rolled MLP critic and a hand-rolled rollout buffer.
- Exposes `from_config(cfg, *, env, ego_uid)` as the
  `concerto.training.ego_aht.TrainerFactory` callable that
  `chamber.benchmarks.training_runner.run_training` selects by default.

External researchers who *do* want a fork-internal trainer subclass can
fill in the `# UPSTREAM-VERIFY:` markers in the fork's `EgoAHTHAPPO`
following the same structure as the CONCERTO-side `EgoPPOTrainer`. The
property test `tests/property/test_advantage_decomposition.py` documents
the required GAE math; the scaffold's docstrings cite the specific
upstream HARL signatures to call.

## Reproducibility

The fork's history after this recipe is two commits:

1. Upstream HEAD (tagged `v0.0.0-vendored`).
2. CONCERTO AHT-wrapper patch (tagged `v0.1.0-aht`).

If we ever need to rebase onto a newer upstream HEAD: cherry-pick the
`v0.1.0-aht` commit onto the new base, re-tag, push. The patch surface
is intentionally tiny (three new files + one test) so this stays
mechanical.

## Licence hygiene

HARL is MIT-licensed; the fork inherits MIT for every file copied from
upstream. CONCERTO's added files (the four files under `v0.1.0-aht/`)
are dual-licensed Apache 2.0 / MIT-compatible per their SPDX header
(`# SPDX-License-Identifier: Apache-2.0`). The fork's `NOTICE` should
add a "Modifications by CONCERTO Contributors, licensed under
Apache 2.0" header at the top of the v0.1.0-aht commit. Plan/05 §2:
get this right at first commit; retroactive licence cleanups are
painful.
