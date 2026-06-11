# ADR-017: Observability and experiment tracking

**Status.** Accepted (2026-05-22)
**Authors.** Farhad Safaei
**Reviewers.** _to fill_
**Tags.** v0.2 §13 (RL framework + observability), P1.05.11, ADR-002 cross-reference

## Context

P1.05.8 (commit dd02b80, PR #185, 2026-05-22) closed Surface 1 (rubber-stamp success rule, Phase-0 MPE holdover) and Surface 6 (state-synthesizer width — ego trained blind to cube/goal/partner) of the Stage-1b post-launch failure investigation cataloged in [`spikes/results/stage1-failure-investigation/2026-05-20/INVESTIGATION.md`](../spikes/results/stage1-failure-investigation/2026-05-20/INVESTIGATION.md). It did not recover task performance: the post-widening smoke run and the A1 / A2 / A5 ablations (at `.local/{p1_05_8_as_hetero_smoke,a1_zero_action_partner,a2_safety_disabled,a5_partner_placement}.{json,log,py}`, 2026-05-21) all sit at ~0% terminal success.

The next slice — Surfaces 2 (training budget), 3 (partner placement / interference), 5 (safety-stack saturation), 7 (reward) ablations — produces evidence that is interpretable **only** with three artefacts the project does not yet emit:

1. **Per-step PPO health curves** (`policy_loss`, `value_loss`, `dist_entropy`, `approx_kl`, `clip_fraction`, `actor_grad_norm`, ratio min/max, advantage/value stats, learning rate). The trainer's `EgoPPOTrainer.update()` call to `HARL.HAPPO.update(sample)` at `src/chamber/benchmarks/ego_ppo_trainer.py:972` returned a four-tuple `(policy_loss, dist_entropy, actor_grad_norm, imp_weights)` per HARL `harl.algorithms.actors.happo.HAPPO.update` line 102; pre-P1.05.11 the trainer **discarded** it. Without these scalars the founder cannot diagnose PPO collapse vs PPO healthy-but-under-budgeted vs reward-signal pathology.
2. **Side-by-side comparison surface** across N runs at scalar-metric granularity. The 2026-05-21 ablations produced four ~100k-frame archives that should be compared head-to-head; doing so today requires manual JSONL grep + tabulation.
3. **Per-step rollout records** at obs/action/reward granularity. The founder watches MP4 videos; the agent cannot. Without a machine-readable rollout dump the agent cannot answer "at what step did the gripper close in A2?" or "what is the policy doing in A1 vs A5?" — questions the consultation flow has already needed to ask.

The slice prompt (P1.05.11) names the gap explicitly: *every future ablation or remediation slice will produce evidence that is interpretable only if the founder can see curves and rollouts and the agent can co-analyse them programmatically.* This ADR pins the design that closes the gap.

[`CONSULTATION_BRIEF.md`](../spikes/results/stage1-failure-investigation/2026-05-20/CONSULTATION_BRIEF.md) (2026-05-20) §6.5 notes the implicit follow-up: "*NOW run ablations A1 / A2 / A5 against the corrected baseline. Their results are interpretable post-Surface-6-fix.*" That sentence is true only with this ADR's tooling in place.

## Considered alternatives

| # | Option | Source / advocate | Pros | Cons |
|---|--------|-------------------|------|------|
| A | **Adopt W&B as a downstream visualization layer over JSONL canon, extend `concerto.training.logging` in place, ship a read-only `chamber-analyze` CLI + a rollout recorder with paired MP4 + per-step sidecar JSONL.** | The slice prompt; HARL canonical training pattern; the existing `WandbSink` Protocol already in `concerto.training.logging`. | (i) Reuses the existing structured-logging contract (ADR-002 §Decisions) instead of duplicating it. (ii) JSONL canon is unchanged; W&B is purely additive — runs without auth still complete with valid records. (iii) `chamber-analyze` reads local archives, so it works offline + in CI + reproducibly. (iv) Rollout sidecar JSONL gives the agent step-granularity reasoning the MP4 cannot. | (i) Adds the matplotlib/imageio surface to the agent's tooling. (ii) Per-rollout sidecar JSONLs are bulky; per-step storage at 25k-frame intervals can mount up. |
| B | Build a parallel `src/chamber/observability/{logger,wandb_logger}.py` module that duplicates the existing logger. | The slice prompt's initial framing. | (i) Cleaner chamber/concerto split if the logger were chamber-specific. | (i) Duplicates ~250 LOC of structlog/JSONL/W&B-sink plumbing already in `concerto.training.logging`. (ii) Splits the `RunContext` provenance contract across two modules. (iii) Forces two test suites for the same shape. (iv) The slice prompt was written before re-reading the existing module; the duplication is incidental. |
| C | Use TensorBoard + a custom `tensorboard.SummaryWriter` extension, skip W&B. | tch ecosystem standard. | (i) No external service dependency. (ii) Local logs already work. | (i) TensorBoard's per-run UI is weaker than W&B's at side-by-side comparison; the slice's central use case is comparison. (ii) No artefact tracking (rollout MP4 / sidecar JSONL upload story is awkward). (iii) The founder's stated workflow is browser-based; W&B fits that workflow directly. |
| D | Self-host W&B (or use mlflow). | Project autonomy preferences. | (i) No external SaaS lock-in. | (i) Project is solo + early-phase; the W&B free tier is sufficient through Phase 1. (ii) Self-hosting costs founder time the slice does not have. (iii) Easy to migrate later — the slice's W&B usage is shallow. |

## Decision

Adopt **Option A** with nine sub-decisions (D1–D9 from the slice prompt, two amended per the 2026-05-22 plan-mode reconciliation):

### D1. JSONL as the canonical record; W&B as a downstream visualization layer.

Per ADR-002 §Decisions, every line emitted during a training run is written to a per-cell `<run_id>.jsonl` under `cfg.log_dir`. W&B is opt-in via `cfg.wandb.enabled`. If W&B is unreachable or auth is missing, the run **must** still complete and the JSONL must still be valid — degrade-to-no-op is structural, never raised. Existing pre-P1.05.11 logger code already implements this (`bind_run_logger(ctx, jsonl_path=jsonl_path, wandb_sink=None)` is the silent-W&B path that production has always used).

### D2. Single W&B project `concerto-chamber`; tag-based filtering for stage / sub_stage / condition / prereg / backfill.

Cross-stage comparison is a primary use case (Stage-1a vs Stage-1b vs eventual Stage-2). W&B's UI handles single-project filtering well; multiple projects break the comparison view. The default `concerto.training.config.WandbConfig.project` is migrated `"concerto-m4b"` (legacy Month-4-Phase-B) → `"concerto-chamber"` (2026-05-22). All three Hydra YAML configs (`stage0_smoke.yaml`, `mpe_cooperative_push.yaml`, `stage1_pickplace.yaml`) are migrated in the same PR.

### D3. Run identity uses the existing per-cell `run_id`.

The 16-hex SHA derived by `concerto.training.logging.compute_run_metadata` from `(seed, git_sha, pyproject_hash, run_kind)` is the W&B `id` field. The slice prompt named "the spike envelope's run_id"; on reconciliation this is amended to **per-cell** run_id (the W&B-idiomatic granularity for side-by-side seed/condition charts is per cell, not per spike).

### D4. `wandb` stays a **runtime** dependency; the W&B *sink* is opt-in.

`wandb>=0.17,<1.0` is already in `[project.dependencies]` (`pyproject.toml:53`); demoting it to extras would break existing installs. The W&B sink is opt-in via `cfg.wandb.enabled` and degrades to no-op when `enabled=false` OR `WANDB_API_KEY` missing OR `import wandb` fails OR `wandb.init` raises (each path emits a single `UserWarning` and returns `None` from `make_wandb_run_sink`). The slice prompt's original D4 (optional dep) is amended here.

### D5. `chamber-analyze` CLI reads local JSONL archives, **never** the W&B API.

This is a first-class deliverable, not a stretch goal. Six subcommands ship with the slice: `list-runs`, `summary`, `metrics`, `compare`, `plot`, `rollout-frames`. Each supports `--json` for agent consumption. Implementing against the local archive (not W&B) gives the CLI three structural properties:

- Works offline (no auth, no network).
- Works in CI (no W&B account in CI).
- Reproducible from canonical record alone (matches ADR-002's provenance discipline).

The agent reads JSON output from this CLI; the founder reads W&B in the browser. Both consume the same underlying records.

### D6. Rollout-level co-analysis via per-step sidecar JSONLs paired with each MP4.

`chamber.observability.RolloutRecorder` captures one full eval episode every `cfg.rollout_recorder.interval_frames` training frames, emitting two paired artefacts at `<archive_dir>/rollouts/<condition>/step_<NNNNNN>.{mp4,jsonl}`. The agent reads the JSONL; the founder watches the MP4. Per the §Schema appendix below the per-step records land in **sidecar files**, not appended to the main per-cell `<run_id>.jsonl` stream — rollout data is bulky and tooling that streams the main JSONL must not be forced to parse rollout-frame payloads. The per-step JSONL carries a curated `obs_summary` (cube_pose, goal_pos, tcp_pose, ego_qpos/qvel, partner_qpos/qvel, is_grasped, gripper_width) — **never** the full 65-D flattened ego state (deferred to §Open questions).

### D7. Backfill committed archives in the same PR.

`scripts/wandb_backfill.py` replays the three committed Stage-1 archives (`spikes/results/stage1-AS-20260517/`, `spikes/results/stage1-OM-20260517/`, `spikes/results/stage1-failure-investigation/2026-05-20/`) into W&B as historical runs tagged `backfill:true`. The script is idempotent (`wandb.init(id=<run_id>, resume="never")`). `--include-local` opts into envelope-only replay of the four gitignored `.local/*.json` post-widening snapshots. Landing this in the same PR means the first time the founder opens the W&B dashboard, the full comparative history is already there — the highest-leverage moment for adopting the new tool.

The slice prompt's original D7 named a "2026-05-21 post-widening" archive that does not exist as a committed Stage-1b spike directory; the amended scope (committed + `--include-local`) matches what is actually on disk.

### D8. ADR-017 cross-references ADR-002; ADR-016 is untouched.

The slice prompt's original D8 named ADR-016 (SpikeRun Pydantic schema) as the cross-reference target. On reconciliation: ADR-016 governs the `chamber.evaluation.results.SpikeRun` schema (`sub_stage` Literal, `schema_version` 1 → 2). The training-time per-cell JSONL is an ADR-002 artefact — that is the right cross-reference target. ADR-002 gets a one-line §Revision history entry pointing forward to this ADR. **No ADR-016 edit is made.**

### D9. No prereg rotation. No science axis change.

This slice touches no env, no policy, no reward, no safety filter, no obs synthesizer. The Stage-1b prereg tag (`prereg-stage1-AS-2026-05-15`) does not rotate. The deterministic-seed equivalence smoke (§Validation criteria) is the gate that proves observability does not perturb the science.

### D10 (P1.05.11 §3.7 amendment). `prereg_sha` is written as BOTH a W&B tag AND a `wandb.config` field.

The short-form `prereg:<sha8>` tag enables cheap filter-bar matching in the W&B UI. The full SHA `wandb.config.prereg_sha` enables auditable per-run inspection (the founder clicks into the run; the value is right there in the Overview panel). The audit story requires both UI affordances. Implemented in both `concerto.training.logging.make_wandb_run_sink` (live runs) and `scripts/wandb_backfill.py` (historical replay).

## Rationale

The slice is pure infrastructure — it does not advance the science axis by a single step. Its value is leverage: every future science step is interpretable to the founder (curves + rollouts in browser) AND to the agent (JSON output + per-step JSONL reasoning). Options C (TensorBoard) and D (self-host) trade founder time and stability against the comparison-surface use case that the failure-investigation flow specifically demanded. Option B duplicates infrastructure that already exists.

Option A's reuse of `concerto.training.logging` is decisive: the existing module already implements (i) per-line `RunContext` provenance, (ii) the `WandbSink` Protocol with offline-fake support, (iii) the `_JSONLSink` terminal processor, (iv) the `_WandbProcessor` event forwarder. The slice extends that surface with `log_scalars` / `log_eval` helpers + `make_wandb_run_sink` factory rather than building a parallel module.

The xfail-strict discipline applied to the env-render integration test (`tests/integration/test_rollout_recorder_env_render.py`) mirrors two precedents in this project that both resolved cleanly: the P1.02 `Bounds.action_norm` regression flag (`tests/property/test_bounds_action_norm_inconsistency_documented.py`, removed in the same P1.02 commit that split the field per ADR-004 §Revision history 2026-05-17) and the P1.05.8 trainer-obs-reader contract (`tests/integration/test_trainer_obs_reader_contract.py:42-48`, removed in PR #185 per ADR-002 §Revision history 2026-05-21). Both forced the remediation PR to close the regression-pin lifecycle in a single commit; this slice's env-render xfail follows the same lifecycle (marker removed in the same commit that flips it to xpass post-host-reboot).

## Evidence basis

- [`spikes/results/stage1-failure-investigation/2026-05-20/CONSULTATION_BRIEF.md`](../spikes/results/stage1-failure-investigation/2026-05-20/CONSULTATION_BRIEF.md) — 2026-05-20 failure analysis naming the visibility gaps that motivated this ADR.
- [`spikes/results/stage1-failure-investigation/2026-05-20/INVESTIGATION.md`](../spikes/results/stage1-failure-investigation/2026-05-20/INVESTIGATION.md) — Surface 1–7 audit; the post-P1.05.8 closure context for the four `.local/` ablation archives.
- [ADR-002](ADR-002-rl-framework.md) §Decisions — the structured-logging contract this ADR extends.
- [ADR-016](ADR-016-spike-run-schema.md) — the SpikeRun envelope schema (NOT amended; cited for delineation).
- Stage 0 probe outputs at `.local/stage0_probes.md` (gitignored; embedded in the PR description for archaeology).

## Schema appendix

### Main per-cell JSONL line types (additive to ADR-002's existing shape)

The per-cell `<run_id>.jsonl` already carried (pre-P1.05.11) `event ∈ {training_start, safety_telemetry, safety_telemetry_final, rollout_update, checkpoint_saved, training_end}`. This ADR adds three line types (additive; existing readers ignore unknown `event` values):

**`event="scalar"`** (P1.05.11 / commit 3 — `EgoPPOTrainer.update` emission). One emission per PPO update (= one per rollout):

```json
{"event": "scalar", "metric_namespace": "train", "step": <int>,
 "wall_time": "<iso8601>", "run_id": "<16hex>", "seed": ...,
 "policy_loss": <float>, "value_loss": <float>, "dist_entropy": <float>,
 "approx_kl": <float>, "clip_fraction": <float>, "actor_grad_norm": <float>,
 "ratio_min": <float>, "ratio_max": <float>, "value_mean": <float>,
 "value_std": <float>, "advantage_mean": <float>, "advantage_std": <float>,
 "learning_rate": <float>, "level": "info", ...}
```

**`event="eval"`** (P1.05.11 / future commit — eval-cell wiring). One emission per eval-cell completion:

```json
{"event": "eval", "metric_namespace": "eval", "step": <int>,
 "wall_time": "<iso8601>", "condition": "<condition_id>",
 "success_rate": <float>, "mean_episode_length": <float>,
 "mean_episode_reward": <float>, "n_terminated": <int>, "n_truncated": <int>,
 "run_id": ..., "seed": ..., ...}
```

The `metric_namespace` allow-list (per `concerto.training.logging._LOG_NAMESPACES`) is `{"train", "eval", "safety", "hardware", "rollout"}`. Extending the allow-list requires editing both the constant and this appendix together. `log_scalars` raises `ValueError` on unrecognised namespaces (defensive backstop against typos that would silently make a metric unfindable by `chamber-analyze`).

### Per-rollout sidecar JSONL (sidecar, NOT main stream)

Per D6, rollout records land at `<archive_dir>/rollouts/<condition>/step_<NNNNNN>.jsonl` paired 1:1 with the MP4 at the same path:

```json
{"event": "rollout_step", "metric_namespace": "rollout",
 "step_global": <int>, "step_episode": <int>, "condition": "<condition>",
 "seed": <int>, "wall_time": "<iso8601>", "obs_summary": {...},
 "action": [<float>], "reward": <float>, "terminated": <bool>,
 "truncated": <bool>, "info": {...}, "frame_index": <int>}
```

`obs_summary` is the curated subset (cube_pose, goal_pos, tcp_pose, ego_qpos/qvel, partner_qpos/qvel, is_grasped, gripper_width). Full-obs logging is deferred to §Open questions.

**No `schema_version` field is added** to the JSONL stream — it didn't have one before (ADR-002 §Decisions delegates schema discipline to the `event` dispatch); readers should continue to dispatch on `event`. ADR-016's SpikeRun schema (`SCHEMA_VERSION = 2`) is unaffected.

## Consequences

### Positive

- Founder-side debugging unblocked: opening the W&B dashboard answers "what is the policy doing in A2 vs A1?" with curves + videos in under 5 minutes (slice §8 acceptance).
- Agent-side co-analysis unblocked: `chamber-analyze compare ... --json | jq` answers the same question from the agent side in under 1 minute (slice §8 acceptance).
- First-time-open W&B dashboard has full retrospective history (D7).
- Per-step PPO health surfaces for the next ablation slice (Surface 2 budget, Surface 3 partner placement, Surface 5 safety de-saturation).

### Negative

- New optional `imageio[ffmpeg]` dep (under `[observability]`). Lazy import + degrade-gracefully — callers who skip the extra still get the per-step JSONLs.
- ~1500 LOC of new module + CLI + backfill surface to maintain (~6 commits in this slice).
- Ongoing W&B free-tier dependency (acceptable for a solo / personal project; D drop later if needed).
- The env-render xfail-strict test must be removed when the host driver/library mismatch resolves; that's one extra one-line commit gated on a host reboot.

### Neutral

- ADR-002 §Decisions retains its logging contract verbatim. This ADR's additions are downstream of ADR-002, not in conflict with it.
- ADR-016 SpikeRun schema is **untouched**. The slice prompt's misread (ADR-016 → "per-step JSONL") is corrected here.
- `concerto.training.config.WandbConfig.project` default migrates from `"concerto-m4b"` → `"concerto-chamber"`. Existing YAMLs that override this field are migrated in the same PR; downstream users with hand-tuned configs see a one-line change.

## Risks and mitigations

- **Risk:** observability writes perturb training trajectories (silent determinism violation). — **Mitigation:** the deterministic-seed equivalence smoke at slice §9.1 runs the same Stage-1b cell with observability off vs on (offline W&B); diffs the resulting `spike_as.json` after stripping `wall_time` fields. Non-empty diff is a stop-and-surface trigger per slice §12. Logger-emission equivalence proved at slice-time; recorder equivalence proved post-reboot.
- **Risk:** W&B service outage or rate-limiting silently degrades runs. — **Mitigation:** every W&B sink failure path returns `None` from `make_wandb_run_sink` with a single `UserWarning`. Runs complete on JSONL alone; W&B uploads are not load-bearing.
- **Risk:** the env-render xfail-strict marker is forgotten after host reboot. — **Mitigation:** `strict=True` forces the maintainer who flips the xfail to xpass to remove the marker in the same commit. The two precedents (P1.02 `Bounds.action_norm`, P1.05.8 trainer obs reader) both resolved cleanly via this discipline.
- **Risk:** `imageio` ABI drift breaks the MP4 encode at runtime. — **Mitigation:** `RolloutRecorder.finalise()` catches encode exceptions, logs a structured warning, and emits the JSONL sidecar only. The JSONL is the agent's canonical view; the MP4 is the founder's affordance.
- **Risk:** W&B free-tier storage limits trip during a future high-cadence rollout sweep. — **Mitigation:** rollout recorder is opt-in; the `cfg.rollout_recorder.interval_frames` knob bounds the artefact volume. If a sweep ever exceeds the free tier, the founder receives a W&B email and can decide (downsample, paid tier, self-hosted).

## Reversibility

Type-2 within Phase 0–1. The W&B project name, the namespace allow-list, the rollout-recorder sidecar JSONL shape, and the `chamber-analyze` CLI surface are all reversible via superseding ADR before any external consumer (paper, downstream user, external collaborator) relies on them. The JSONL canon (ADR-002's contract) is the load-bearing piece; this ADR is purely additive to it.

Type-1 once a published paper cites a `chamber-analyze --json` shape — at that point the JSON output schema becomes the public contract and any further change needs a deprecation cycle. The slice prompt's §3.5 "stable across versions" discipline acknowledges this.

## Validation criteria

This ADR is **Accepted** at landing; promotion to **Validated** requires the following evidence within Phase 1:

1. The deterministic-seed equivalence smoke (slice §9.1) passes in CI: same `spike_as.json` (modulo `wall_time` fields) between `observability.wandb.enabled=false` and `observability.wandb.enabled=true` runs of the same Stage-1b cell. Pinned by `tests/integration/test_ppo_observability.py` (commit 9 of this slice).
2. `chamber-analyze list-runs --archive-root spikes/results/stage1-failure-investigation/2026-05-20 --json` returns 5 cells with non-null `terminal_success_rate`. Pinned by `tests/unit/cli/test_analyze.py::TestListRunsRealArchive::test_returns_five_cells_with_non_null_success`.
3. The env-render xfail-strict test at `tests/integration/test_rollout_recorder_env_render.py` flips from xfail to xpass when the host driver/library mismatch resolves; the marker is removed in the same commit (mirroring the P1.02 / P1.05.8 precedents).
4. The W&B online backfill of the three committed archives (slice §10) lands historical runs with the expected tag set (`sub_stage:1b`, `stage:1`, `axis:AS`, `prereg:29e397a4`, `backfill:true`); founder confirms via the W&B project URL inserted into the PR description.

## Open questions deferred to a later ADR

- **Rollout per-step JSONL compression at rest.** Sidecar files are small individually but a long-running sweep accumulates them. Measure first; revisit if storage becomes load-bearing.
- **`chamber-analyze policy-replay` subcommand** — load a checkpoint + run an interactive rollout. **Locked deferred** (per the slice's plan §14): this subcommand must not migrate into scope during the P1.05.11 implementation. If the temptation appears, stop and surface per the slice's §12.
- **Migration to a W&B *team* (vs personal account) once a co-author joins.** Defer until there's an actual co-author.
- **Full-obs (65-D) per-step logging vs the curated `obs_summary` subset.** The agent-side rollout reader currently sees only the curated fields. If a future agent-reasoning workflow needs the full state vector, revisit; for now the curated subset is what keeps the per-step JSONL tractable.
- **Stage-2 / Stage-3 connector for `chamber-analyze`.** Defer until Stage 2 / Stage 3 archives exist.

## Revision history

- 2026-05-22 (initial draft + lock): introduces the observability slice (P1.05.11). Cross-references ADR-002 §Decisions (logging contract extension) and ADR-007 §Stage 1b (per-cell `run_id` provenance). No edit to ADR-016. ADR-002 gets a §Revision history forward-pointer in the same PR.
- 2026-05-22 amendment (doc-only follow-up): `docs/observability.md` expanded with four labelled Cookbook-recipe subsections (first-instrumented-cell walkthrough; comparing two runs — worked example uses `43a0d043cb9f54ac` + `e3ca669746e8f4bb` from `spikes/results/stage1-failure-investigation/2026-05-20/`; per-step rollout analysis; embedding a comparison plot in a PR description) + Reference subsections (config-schema YAML for `WandbConfig` + `RolloutRecorderConfig`; environment-variable degrade-to-no-op decision table per D4; per-subcommand JSON-output schemas as the public agent-co-analysis contract per D5) + a Troubleshooting section covering the NVIDIA driver/library-mismatch recovery path. No change to §Decisions content; no source-tree edit.
