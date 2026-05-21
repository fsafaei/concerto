# Post-widening Stage-1b AS investigation — Surface 6 fix did not lift the gate

**Date.** 2026-05-21 (smoke launched ~17:13 UTC, completed ~18:01 UTC).
**Branch.** `chore/spikes-stage1b-postwidening-investigation` off `dd02b80` (P1.05.8 merged).
**Trigger.** Second §4a runbook firing — Surface 6 widening + Surface 1 success-rule fix landed via PR #185, but the post-merge single-cell smoke against AS-hetero still reports `success_rate = 0/20 = 0.0%`. The per-firing audit-trail discipline (ADR-007 §Discipline; PR #149 precedent) requires a new immutable evidence directory for the new firing; the 2026-05-20 directory carries the original failure and is not modified.

## Headline

**The Surface 6 widening is structurally correct but is not sufficient to lift the gate.** The trainer now sees the widened state vector (`obs["agent"]["panda_wristcam"]["state"].shape = (65,)` end-to-end — verified at env build and at trainer construction), the post-P1.05.8 terminated-only success rule reports honestly (no `terminated=True` ⇒ no `success=True`), and the eval loop ran cleanly against the production cfg. The policy did not converge on the 4-stage pick-place. λ telemetry is byte-identical to the 2026-05-20 failed launch (`λ_steady_state = -7.0`, `λ_mean = -6.5117`, `λ_var = 2.0363`), which is the strongest fingerprint we have for where the remaining defect lives.

## What ran

```
$ uv run python .local/p1_05_8_as_hetero_smoke.py
========================================================================
P1.05.8 AS-hetero smoke — one cell, full 100k training frames
========================================================================
Prereg: prereg-stage1-AS-2026-05-15; sha=29e397a4a012813c...
Hetero condition: stage1_pickplace_panda_plus_fetch_ego_aht_happo_per_agent
Homo  condition: stage1_pickplace_panda_only_mappo_shared_param (SKIPPED — single-cell smoke)
Cfg total_frames=100000; hidden_dim=64
Cfg rollout_length=1024; batch_size=256
Cfg device=cuda; deterministic_torch=False

Env built; observation_space['agent']['panda_wristcam']['state'].shape=(65,)
           observation_space['agent']['fetch']['state'].shape=(30,)

Training (100k frames; expect ~30-50 min on RTX 2080)...
Training done in 48.4 min
```

Wallclock: 48.6 min (1 cell × seed 0 × 100k training frames + 20 eval episodes) on RTX 2080 SUPER (driver 535.288.01, torch 2.11.0+cu128).

Cfg: `configs/training/ego_aht_happo/stage1_pickplace.yaml` (the production yaml the Stage-1b dispatch loads); no per-cell overrides apart from `seed=0` and `condition_id=AS-hetero` (the production factory's overrides).

Smoke driver: `.local/p1_05_8_as_hetero_smoke.py` (local-only, gitignored). Carries the run command, the prereg / cfg path resolution, and the post-run aggregation; routes through the production `TrainedPolicyFactory` end-to-end.

## Why the science gate failed

```
Condition: stage1_pickplace_panda_plus_fetch_ego_aht_happo_per_agent
Episodes: 20
Success rate: 0/20 = 0.0%
n_terminated: 0; n_truncated: 0
mean_reward: min=0.0000  max=0.0719  avg=0.0280
```

Per-episode breakdown is in `launch.log` (eval section). All 20 episodes ran to the natural step horizon (50 steps) without firing the env's `evaluate()` success predicate (`is_obj_placed & is_robot_static`). `mean_reward` per episode varied across the 20 evaluation initial states (cube spawned within ±5 cm of table origin per the env's defaults; goal sampled in the same envelope) — episodes 0-4 sat at 0.0000 (likely cube/goal configurations the policy could not approach productively), episodes 5-19 climbed to 0.072 max (configurations the shaped reward partially rewarded).

**Comparison to the 2026-05-20 failed launch (pre-widening):**

| metric | 2026-05-21 post-widening (this firing) | 2026-05-20 pre-widening (PR #182) |
|---|---|---|
| ego obs shape (per uid `state`) | **(65,)** widened | (18,) ego qpos+qvel only |
| success_rate (terminated-only) | 0/20 = 0.0% | 0/200 = 0.0% (under terminated-only; original archive's "100%" was rubber-stamp) |
| n_terminated | 0 | 0 |
| n_truncated | 0 | 0 |
| mean_reward avg | 0.0280 | 0.0269 (AS-hetero mean of 5 seeds × 20 eps) |
| mean_reward max | 0.0719 | 1.0000 (≈1 % of eps under pre-widening) |
| λ_steady_state | **−7.0** (byte-identical) | −7.0 |
| λ_mean | **−6.5117** (byte-identical) | −6.5117 |
| λ_var | **2.0363** (byte-identical) | 2.0363 |
| n_qp_infeasible / n_braking_fires | 0 / 0 | 0 / 0 |

The byte-identical λ telemetry across the two firings — under different obs contracts, different seeds (the failed launch used seeds 0-4; this smoke is seed 0 only), different success rules — is the load-bearing diagnostic. The conformal-overlay update path is the constant-velocity predictor's `−η × |ε|` drift to the clamp bound (per ADR-004 Rev 2026-05-20 / Revision 11); that drift does not depend on the policy's observation channel or the success rule.

## Surface attribution under the corrected baseline

The five-surface table from `INVESTIGATION.md` (2026-05-20) updates as follows under this firing's evidence:

| # | Surface | Pre-widening verdict | Post-widening update |
|---|---|---|---|
| 1 | Success rule | DEFECT | **CLOSED** (P1.05.8 / ADR-007 §Stage 1b Rev 12) |
| 2 | Training budget | SUSPECT | **STILL SUSPECT** (100k frames; community baselines are 1M+) |
| 3 | Partner heuristic targets cube spawn | DEFECT | **STILL UNDER INVESTIGATION** (untouched by P1.05.8) |
| 4 | Adapter env-factory | CLEAN | CLEAN (mechanical end-to-end here too) |
| 5 | Safety filter saturation | SUSPECT | **PROMOTED to PRIMARY suspect** (byte-identical λ to pre-widening firing) |
| 6 | Observation contract | DEFECT | **CLOSED** (P1.05.8 / ADR-007 §Stage 1b Rev 12; `state.shape=(65,)` verified end-to-end) |
| 7 | Reward function | SUSPECT | STILL SUSPECT (the climbing-then-plateauing signature persists; rollout `last_reward` climbed 3.8e-5 → 0.0203 across 100k frames — same range as pre-widening) |

Surfaces 1 and 6 are closed by the P1.05.8 merge. The remaining live surfaces are 2, 3, 5, and 7 (where 7's signature is "the reward function is fine, but the policy doesn't get the reward" — derived, not load-bearing). The byte-identical λ signal isolates Surface 5 as the strongest single-cause candidate; Surfaces 3 and 2 remain compound contributors.

## Ablation reordering — A2 → A1 → A5 (deferring A3)

The original `INVESTIGATION.md` priority order ran `A4 → A1 → A2 → A5 → A3`. Under the post-widening evidence, A4 (env-contract reachability via hand-coded oracle) is no longer load-bearing — the rollout `last_reward` data already establishes that the env's success terminal is reachable in principle (1 % of pre-widening eval episodes hit `mean_reward = 1.0` under fully untrained pre-widening data; the env contract is not broken). The remaining order changes as follows:

| New order | Ablation | Cost | Why this order |
|---|---|---|---|
| 1 | **A2 — safety filter disabled (`cfg.safety.enabled = False`)** | ~50 min | Byte-identical λ telemetry across two independent firings is the strongest single fingerprint the investigation has. If success_rate ≥ 10-20 % with safety disabled, Surface 5 is fingerprinted as the dominant remaining defect — fingerprint matches the ADR-004 2026-05-20 amendment's empirical-vacuous-conformal-overlay hypothesis. |
| 2 | **A1 — zero-action partner (replace ScriptedHeuristicPartner with a zero-action wrapper)** | ~50 min | Second-strongest single-surface hypothesis (Surface 3). Removes the (0,0)-targeting partner that races toward the cube spawn. Distinguishes "partner as presence" from "partner as targeting". |
| 3 | **A5 — partner placement (`target_xy = "0.5,0.5"`)** | ~50 min | Isolates the Surface 3 × Surface 5 interaction. Tests whether the partner-cube proximity (not the partner itself) is the load-bearing component of the failure mode. Run **only if** A1 lifts the gate, so A5's result has interpretive weight against A1's. |
| skipped | **A3 — 5× training budget (500k frames)** | ~6-7 GPU-h | Deferred (was the 5th-priority ablation in the pre-widening table; defers further under the post-widening evidence because the per-window `last_reward` trend across 100k frames is climb-then-plateau, not climb-then-still-climbing-at-end-of-training; a 5× budget would extend the plateau, not break it). Re-open only if A1/A2/A5 all fail to produce a single dominant signal. |
| skipped | **A4 — env-contract oracle** | ~10 min | Closed by the rollout `last_reward` evidence (env terminal reachable in principle); no need for the oracle pass under this firing's data. |

### Justification for promoting A2 to first

1. **Byte-identical λ telemetry** across two firings: pre-widening (5 seeds × 100k frames each) and post-widening (1 seed × 100k frames). The conformal-overlay update math (`λ_t+1 = clip(λ_t − η × |ε_t|, −clamp_bound, +clamp_bound)`) is deterministic from `ε_t` (the predictor error), and `ε_t` is small under the constant-velocity predictor at the relative-velocity scales the multi-arm rig produces, so `−η × |ε|` is small-negative-each-step, monotonically pushing λ to the lower clamp bound. The policy's observation channel does not enter this loop. Whether the ego sees cube/goal/TCP/partner (post-P1.05.8) or only its own qpos+qvel (pre-P1.05.8), λ converges to the same steady state because λ is policy-independent.
2. **Operational consequence of λ_steady_state = −7.0**: the CBF rhs `drift + γ·h_ij + λ_ij` is offset by −7.0 m/s² for every pair. The ego's effective control authority on the relative-acceleration channel is constrained to `≥ +7.0 m/s² − drift − γ·h_ij`. With two pandas (or panda + fetch) starting near the cube spawn (within ±5 cm), `h_ij` is small (close to safety radius), `γ·h_ij` is small, the relative-acceleration constraint is dominated by the +7.0 contribution from λ. The ego's actor must drive away from the partner to satisfy QP feasibility; this is the operationally-correct safety behaviour, but it dominates over the reward gradient toward the cube.
3. **The pre-P1.05.8 brief's own framing**: §2 named Surface 5 as a SUSPECT under the original five-surface audit, and ADR-004 Rev 2026-05-20 explicitly flagged the persistent-saturation hypothesis as "empirically vacuous under the constant-velocity predictor stub". This firing's evidence promotes that SUSPECT to a primary diagnostic target.
4. **Cheapest informative measurement**: A2 is a single config flip (`cfg.safety.enabled = False`); it does not require new code, new agents, or new partner adapters. A1 needs a new partner class (zero-action wrapper); A5 needs a partner-placement param change.

### Why A1 stays second, not displaced

The scripted-heuristic partner targeting `(0, 0)` is still a real defect under the pre-widening evidence (the cube spawns at the table origin ± 5 cm). Even if A2 lifts the gate, knowing whether A1 ALSO lifts the gate (with the original safety stack enabled) is decision-relevant for ADR-009 §Decision (partner-zoo construction): a partner that is actively-adversarial-by-spawn is a partner-zoo defect, not a safety-stack one. Running A1 second gives the rig the second-strongest single-cause datapoint.

### Why A5 is conditional on A1's outcome

A5 isolates the **interaction** between Surface 3 (partner targeting cube spawn) and Surface 5 (safety filter dominating near cube). If A1 (zero-action partner) lifts the gate, A5 (partner moved away from cube) tests whether the dominant defect was the targeting choice rather than the partner-as-presence. If A1 does NOT lift the gate, A5 is not informative — the partner-as-presence problem isn't a placement problem.

### What this PR does NOT do

- It does **not** propose remediation. The §4a runbook discipline (per the consultation brief at `spikes/results/stage1-failure-investigation/2026-05-20/CONSULTATION_BRIEF.md` §5) keeps engineering questions (what surface is dominant) separate from science-contract questions (what remediation lands). The founder owns the remediation decision; this PR's job is to surface the findings.
- It does **not** open a new slice (e.g. P1.05.9 / lambda_safe derivation / ε-positive regime / partner redesign). Surfacing findings only.
- It does **not** edit the 2026-05-20 directory. The §4a per-firing immutability rule binds.

## Sequenced next steps in this PR

1. ✅ Initial commit: framing + immutable smoke evidence (this document + `spike_as_hetero_widened_FAILED.json` + `launch.log` + `7b033577fce7de7b.jsonl` + `SHA256SUMS.txt`).
2. ⏳ Run A2 (safety disabled). Commit ablation driver + result JSON + log; append A2 §Findings paragraph here.
3. ⏳ Run A1 (zero-action partner). Same pattern.
4. ⏳ Run A5 IFF A1 lifts the gate. Same pattern.
5. ⏳ Final commit: §Findings summary section here, naming the convergent fingerprint (or noting that no ablation converged).
6. Open PR; surface to founder for remediation decision.

## Evidence inventory

| File | SHA-256 (truncated) | Description |
|---|---|---|
| `POSTWIDENING_INVESTIGATION.md` | (this document) | Investigation framing + ablation results + findings |
| `spike_as_hetero_widened_FAILED.json` | `7ed74b15...` | SpikeRun JSON; 1 seed × 1 condition × 20 eval episodes; post-P1.05.8 success rule applied |
| `launch.log` | `95a41538...` | Smoke driver stdout incl. training event stream + eval per-episode summary |
| `7b033577fce7de7b.jsonl` | `a6c705fb...` | Per-step λ telemetry + rollout_update events; run_id from `training_start` |
| `SHA256SUMS.txt` | (canonical) | sha256sum of every artefact above |

Run_id: `7b033577fce7de7b` (matches `training_start.run_id` in the JSONL and the checkpoint files under `artifacts/artifacts/`; checkpoints not committed per the project's `.gitignore` rule on `*.pt`).

Commit SHA at smoke time: `f7b34979eb9afce44ceca146b800dc9a2acc696e` (PR #185 head; squash-merged to `main` as `dd02b80`).

Pyproject hash: `dc5fe28f424967492706d2b9004c172600330975c74dda3091f0730e2bef83fc`.

Torch: `2.11.0+cu128` (per the `torch.cuda` log at smoke start); CUDA 12.8.
