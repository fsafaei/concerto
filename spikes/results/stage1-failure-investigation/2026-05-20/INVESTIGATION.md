# Stage-1b AS launch — failure investigation

**Date:** 2026-05-20
**Slice:** P1.05 PR 5b (first attempt)
**Result:** `gap_pp = 0.0` (gate requires ≥20 pp). §4a runbook fired.
**Runbook reference:** [P1.05 prompt §6 if-the-gate-fails-entirely](../../../README.md) (paraphrased: failed archive + launch logs → `stage1-failure-investigation/<date>/`, senior-advisor consultation, no Phase-2).

## Headline

**The failure-detection chain worked as designed.** The archive carries every audit-discipline invariant per ADR-007 §Discipline — `schema_version=2`, `sub_stage="1b"`, `prereg_sha=29e397a4a012813c58fd4a0c3077ea8c754affc8` (matches the tagged blob at `prereg-stage1-AS-2026-05-15`), λ telemetry captured per cell, audit-gate predicate A + B both PASSed. The failure is the *science evidence*, not a corruption of it.

**Positive reframe.** Phase 0 closed methodologically with the §4a runbook pre-committed; Phase 1 fires it on the **first** Stage-1b attempt at ~30 GPU-h sunk cost — instead of 200+ GPU-h after Stage-2/3 had also tripped on the same underlying defect. The staged-rollout discipline working at the cheap-failure end of the gradient is the entire point.

## What ran

```
$ bash scripts/repro/stage1_as_stage1b.sh
==> Stage-1b AS spike (ADR-007 §Stage 1b; P1.05)
    Hardware — SAPIEN/Vulkan: available   | PyTorch device: cuda
    torch=2.11.0+cu128, cuda=12.8
==> Verifying pre-registration (ADR-007 §Discipline)
    PASS (prereg_sha=29e397a4a012813c58fd4a0c3077ea8c754affc8)
==> Launching spike — output: spikes/results/stage1-AS-stage1b-20260520/spike_as.json
[5 seeds × 2 conditions × (100k training frames + 20 eval episodes)]
chamber-spike run: PASS axis=AS output=...  episodes=200

==> Running ADR-008 evaluation pipeline
==> Auditing regenerated SpikeRun
    PASS schema_version=2  sub_stage=1b  prereg_sha=29e397a4a012813c58fd4a0c3077ea8c754affc8

==> Running audit-gate hook (predicates A + B; ADR-007 §Stage 1b)
stage1_as_stage1b_audit_gate: PASS audit gate (predicates A + B).
  λ_steady_state=-7.0 (threshold=9.000000)
  λ_mean=-6.511710312499984, λ_var=2.0363003614092605

==> PASS — Stage-1b AS spike complete.
```

Wallclock: ~8.7 hours on RTX 2080 SUPER (driver 535.288.01, torch 2.11.0+cu128).

## Why the science gate failed

```
$ uv run chamber-spike next-stage --prior-stage 1 --spike-runs ./spike_as.json
chamber-spike next-stage: FAIL -- Stage-1 gate (gate_pp=20.0); 0/1 axis(es) pass.
  axis=AS  ci_low_pp=0.00  ci_high_pp=0.00  (fail)
```

Per-condition success-rate breakdown:

| condition_id | n | n_success | success_rate |
|---|---|---|---|
| `stage1_pickplace_panda_only_mappo_shared_param` (AS-homo) | 100 | 100 | **100%** |
| `stage1_pickplace_panda_plus_fetch_ego_aht_happo_per_agent` (AS-hetero) | 100 | 100 | **100%** |

Bootstrap CI: `[0.00, 0.00]`. Both conditions trivially pass the success rule → no AS-axis signal.

## Five-surface audit

Each rig surface gets an audit paragraph, regardless of whether the audit finds the surface clean. The point is to catch the surfaces NOT pre-surfaced in the headline analysis.

### Surface 1 — Success rule (`chamber.benchmarks.stage1_as._run_one_episode`)

**Status: DEFECT (stale Phase-0 MPE holdover).**

The success rule is:
```python
success = mean_reward > _SUCCESS_THRESHOLD or terminated
# _SUCCESS_THRESHOLD = -0.30
```

`_SUCCESS_THRESHOLD = -0.30` was the MPE Cooperative-Push boundary (MPE has negative rewards centered around -0.5 to -0.2). The Stage-1b ManiSkill v3 env has small-magnitude positive rewards (per-episode `mean_reward` field in spike_as.json metadata: `0.0000` to `0.0001`), so `> -0.30` is a **rubber-stamp**: every episode trivially passes regardless of policy quality.

The code's own TODO comment from PR #175 acknowledged this:

> *Phase-1: the real Stage-1 pick-place env exposes `terminated=True` on success, at which point the `or terminated` clause becomes the dominant signal and the rule-based mean-reward fallback can be dropped (plan/07 §T5b.2 follow-up).*

The follow-up never landed.

### Surface 2 — Training hyperparameters (`configs/training/ego_aht_happo/stage1_pickplace.yaml`)

**Status: SUSPECT (likely under-budgeted for ManiSkill v3 pick-place).**

Current config:
- `total_frames = 100000`
- `rollout_length = 1024`, `batch_size = 256`, `hidden_dim = 64`
- `gamma = 0.99`, `gae_lambda = 0.95`, `clip_eps = 0.2`, `n_epochs = 4`, `lr = 3e-4`

ManiSkill v3's `PickCube` is a 4-stage sequential task (reach → grasp → place → static). Community baselines (ManiSkill v3 paper, ManiSkill-style HARL benchmarks) typically train PickCube for **1M+ frames** with hidden_dim ≥ 128. The Phase-1 budget of 100k frames is **~10× below the community baseline** for a single-robot PickCube and was inherited unchanged from the MPE-stand-in's empirical-guarantee experiment (where 100k was sufficient because MPE Cooperative-Push is a single-stage push task with dense reward, not a 4-stage manipulation task).

`hidden_dim = 64` is also notably small for a Stage-1b cell: the actor/critic must learn to discriminate 7-DOF arm joint positions + 2-DOF/13-DOF partner state into a 4-stage sequential policy. The MPE empirical-guarantee experiment used `hidden_dim = 64` and that was fine; ManiSkill PickCube probably wants 128 or 256.

### Surface 3 — Partner heuristic (`chamber.partners.heuristic.ScriptedHeuristicPartner`)

**Status: DEFECT (active interference with ego task).**

The scripted heuristic is a greedy planar-reach toward `spec.extra["target_xy"]`. The Stage-1b config sets `target_xy = "0.0,0.0"`. The ManiSkill `PickCube` env spawns the cube **at the table origin (0,0) ± 5 cm** (from `Stage1PickPlaceEnv` module docstring: *"Cube and goal spawn locations follow the upstream `PickCubeEnv` defaults for now (cube within ±5 cm of the table origin)"*).

So the partner robot's behavior in both conditions is: **drive directly toward where the cube spawns**. This produces:
- **AS-homo (panda_partner targets cube):** the second panda races toward the cube alongside the ego panda. Possible outcomes: blocking the ego's grasp, knocking the cube away, or two-panda collision near the cube spawn. None of these help the ego succeed.
- **AS-hetero (fetch base targets cube):** the fetch base drives toward the cube position. The fetch base is a wheeled mobile platform; it can't pick the cube but it can collide with the panda or block the table approach.

The Phase-0 `target_xy = "0.0,0.0"` was the MPE landmark position (a stationary target for both agents to converge on). For Stage-1b, this turns the "partner" into an active adversary on the pick task. No cooperative structure exists in the current setup.

### Surface 4 — Adapter env-factory (`chamber.benchmarks.stage1_as._stage1b_env_factory`)

**Status: CLEAN.**

The launcher ran mechanically end-to-end: prereg verification PASSed, all 10 cells produced 20 evaluation episodes each, `chamber-eval` rendered the leaderboard, post-regeneration audit PASSed (`schema_version=2`, `sub_stage=1b`, `prereg_sha` matches the tagged blob). The factory + cross-check between adapter-side `_CONDITION_UIDS` and env-side `_CONDITION_TABLE` worked as designed. No bug here.

### Surface 5 — Safety filter saturation (`concerto.safety.cbf_qp.ExpCBFQP` + λ clamp)

**Status: SUSPECT (compounds with surfaces 2 + 3).**

λ telemetry per seed (from archived JSONLs):

| run_id | seed | λ_steady | λ_mean | λ_var | n_qp_infeasible | n_braking_fires |
|---|---|---|---|---|---|---|
| `43a0d043cb9f54ac` | 0 | -7.0 | -6.5117 | 2.0363 | 0 | 0 |
| `e3ca669746e8f4bb` | 1 | -7.0 | -6.5117 | 2.0363 | 0 | 0 |
| `ef23e3b576f38c91` | 2 | -7.0 | -6.5117 | 2.0363 | 0 | 0 |
| `a7b549a31fbd8222` | 3 | -7.0 | -6.5117 | 2.0363 | 0 | 0 |
| `da40f205c78355ac` | 4 | -7.0 | -6.5117 | 2.0363 | 0 | 0 |

All five seeds **byte-identical** — confirming the [P1.05.7 / #180 finding](../../../adr/ADR-004-safety-filter.md): predictor loss is ≈ 0 → λ drifts purely on `-η × |ε|` → clamps at -7.0 at step ~14000 → stays there for the remaining 86k frames.

Per-cell consequence: the CBF rhs `drift + γ·h_ij + λ_ij` is offset by -7.0 m/s² *throughout the entire training*. The ego's effective control authority on the relative-acceleration channel is constrained to `≥ +7.0 m/s² − drift − γ·h_ij` for every pair. For two pandas (or panda + fetch) starting near the cube spawn (within ±5 cm), `h_ij` is small (close to collision distance) → `γ·h_ij` is small → the relative-acceleration constraint is dominated by the +7.0 m/s² λ contribution. The ego's actor must produce actions that consistently drive away from the partner.

This is the **operationally-correct safety behaviour** (the conformal layer is doing its bounded job; QP is feasible throughout). But it interacts with surface 3 (partner racing toward cube) to produce a regime where the ego is mostly *defending against* the partner rather than pursuing the pick objective. The safety filter doesn't *fail*; it dominates.

The Stage-2-review pointer from ADR-004 §Revision history (2026-05-20) was: *"if the clamp saturates persistently the ε=-0.05 design assumption may be empirically vacuous under the constant-velocity predictor stub"*. This launch confirms persistent saturation across all 5 seeds; the assumption is confirmed empirically vacuous for the Stage-1b regime.

## Four-ablation design

Each ablation tests ONE hypothesis cleanly. Total cost ~10 GPU-h (vs the 90+ GPU-h of a "fix everything + relaunch" approach). Pattern: buy information cheaply before commit.

### Ablation 1 — Partner pathology (Surface 3)

**Hypothesis:** the ScriptedHeuristicPartner targeting (0,0) actively interferes with the ego's pick task.

**Design:** replace partner with `_zero_ego_action_factory`-style partner (returns zero-action vector). Re-run 1 seed × 1 condition (AS-hetero) × 100k frames. Compare success rate.

**Falsification rule:** if `success_rate > 0` (vs current 100% rubber-stamp, but with terminated-only success rule), partner heuristic is confirmed-pathological → partner redesign required before re-launch.

**Cost:** ~30 min on RTX 2080.

### Ablation 2 — Safety filter saturation (Surface 5)

**Hypothesis:** the saturating λ clamp dominates the ego's effective control authority, preventing policy learning.

**Design:** set `cfg.safety.enabled = False`; re-run 1 seed × 1 condition × 100k frames. Compare success rate.

**Falsification rule:** if `success_rate > 0` (with terminated-only success), safety-filter saturation is confirmed as the dominant constraint → either (a) widen `clamp_floor_ratio` to give more headroom, (b) re-derive `lambda_safe` per ADR-004 §Open questions, or (c) drop ε=-0.05 in favor of the standard Theorem 3 positive-ε regime.

**Cost:** ~30 min on RTX 2080.

### Ablation 3 — Training budget (Surface 2)

**Hypothesis:** 100k frames is insufficient for the ego to learn a 4-stage sequential pick-place policy.

**Design:** extend to 500k frames (5× current); 1 seed × 1 condition × 500k. Use terminated-only success rule. Track success_rate vs frame budget.

**Falsification rule:** if success_rate climbs meaningfully past 500k (e.g., from 0% to ≥20%), budget is confirmed as the binding constraint → re-budget Stage-1b at 1M+ frames per cell (~10× wallclock).

**Cost:** ~6-7 GPU-h on RTX 2080.

### Ablation 4 — Env contract reachability (Surface 1)

**Hypothesis:** the env's `terminated=True` signal is reachable from the initial state distribution with a hand-coded oracle (sanity-check that the env contract isn't itself broken).

**Design:** static analysis of `Stage1PickPlaceEnv.evaluate()` (already done in Surface 1 above: success = `is_obj_placed & is_robot_static`, both panda-routed). Plus a 1-cell oracle test: hand-construct a trajectory (cube → grasp → goal → static) and verify the env returns `terminated=True`. No training involved.

**Falsification rule:** if oracle cannot trigger `terminated=True`, env-side bug in the success predicate → env redesign required.

**Cost:** ~10 min (5 min static analysis + 5 min oracle run).

## What this PR does NOT do

Per §4a runbook discipline: **the investigation PR does NOT propose fixes yet.** Fixes come AFTER root-cause analysis + senior-advisor consultation. The engineering instinct is "I see the bug, let me fix it" — resist. This PR's job:

1. ✅ Surface the failure with full audit-trail evidence (archive + launch log + per-seed JSONLs).
2. ✅ Five-surface audit (this document).
3. ✅ Four-ablation design (this document) — falsifiable hypotheses, not solutions.
4. ⏳ Run the four ablations (queued as follow-up work).
5. ⏳ Senior-advisor consultation with this PR as briefing material.
6. ⏳ Remediation slice(s) opened *after* consultation converges on a root cause.

## What this PR DOES

- Archive the failed run under `spikes/results/stage1-failure-investigation/2026-05-20/`:
  - `spike_as_FAILED.json` (the SpikeRun JSON; carries all 200 episode results)
  - `leaderboard_FAILED.json` (chamber-eval output; `hrs_scalar=1.0` is misleading by the rubber-stamp rule)
  - `launch.log` (full launcher output; 940 KB)
  - 5 per-seed JSONLs (`{run_id}.jsonl`; λ telemetry + rollout_update events)
  - `INVESTIGATION.md` (this document)
- ADR-007 §Revision history Revision 11 records the §4a firing.
- M7 milestone marked **BLOCKED** (Stage-2 work — P1.06+ — halted pending consultation).
- ADR-007 status holds at **Accepted** (NOT promoted to Validated for AS; no successful gate measurement).

## Next steps

1. Open this PR for review.
2. Run the four ablations on RTX 2080 (~10 GPU-h total).
3. Schedule senior-advisor consultation with this PR + ablation results as briefing.
4. Halt Stage-2 work (no Stage-2 prereg-tag cuts, no P1.06/P1.07 opens).
5. After consultation, open remediation slice(s) per agreed root cause(s). Remediation may involve: env reward redesign, hyperparameter retune, partner replacement, safety-filter de-saturation work, or some combination. Each remediation lands as its own slice with its own design pass.
