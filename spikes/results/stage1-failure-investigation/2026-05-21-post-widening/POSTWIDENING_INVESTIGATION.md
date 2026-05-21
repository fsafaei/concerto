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
2. ✅ Run A2 (safety disabled). **Result: NEGATIVE — Surface 5 NOT dominant.** See §A2 Findings below.
3. ✅ Run A1 (zero-action partner). **Result: NEGATIVE on success_rate AND structurally identical eval-reward distribution to the smoke.** See §A1 Findings below.
4. ⏳ Run A5 — **promoted from conditional to RUN** to validate the no-op-partner hypothesis A1 surfaced. The original A5 conditional ("only if A1 lifts the gate") assumed A1's outcome would cleanly separate partner-as-targeting from partner-as-presence. A1's byte-identical-to-smoke eval rewards changed that interpretation; A5 with `target_xy="0.5,0.5"` is now the cheapest empirical test of whether the scripted partner is structurally a no-op against fetch (in which case Surface 3 is a phantom defect) or whether the policy is partner-insensitive at this seed/budget (a different defect).
5. ⏳ Final commit: §Findings summary section, naming the convergent fingerprint (or noting that no ablation converged).
6. Open PR; surface to founder for remediation decision.

## A2 Findings — safety filter disabled (NEGATIVE)

Driver: `.local/a2_safety_disabled.py` (committed to evidence dir as `a2_safety_disabled.log`; SpikeRun at `spike_as_hetero_a2_safety_disabled.json`; per-step events at `5e2f7085688bcd65.jsonl`).

```
Condition: stage1_pickplace_panda_plus_fetch_ego_aht_happo_per_agent
Episodes: 20
Success rate: 0/20 = 0.0%
n_terminated: 0; n_truncated: 0
mean_reward: min=0.0000  max=0.0739  avg=0.0285
Total wallclock: 8.3 min (vs 48.6 min smoke — CBF-QP overhead removed)
```

**Falsification rule from the driver:** `success_rate >= 10-20%` would fingerprint Surface 5 as dominant. Outcome: 0/20. **Surface 5 is NOT the dominant remaining defect.**

The byte-identical λ signal across the 2026-05-20 and 2026-05-21 firings was a real fingerprint about *how the safety stack behaves* (λ converges to the clamp bound under the constant-velocity predictor regardless of policy), but disabling the safety filter does not lift the gate — meaning the saturating λ wasn't the binding obstacle to convergence. The ego's actor learns a similar policy with or without the safety constraint; either way, the policy plateaus in the same reward range without ever satisfying `is_obj_placed & is_robot_static`.

Cross-comparison (deterministic post-training eval, 20 episodes each):

| metric | post-widening smoke (safety ON) | A2 (safety OFF) | delta |
|---|---|---|---|
| training wallclock | 48.4 min | 8.2 min | −83 % (CBF-QP per-step overhead removed) |
| eval success_rate | 0 / 20 | 0 / 20 | 0 |
| eval mean_reward avg | 0.0280 | 0.0285 | +0.0005 (noise) |
| eval mean_reward max | 0.0719 | 0.0739 | +0.0020 (noise) |
| training rollout `last_reward` endpoint | 0.0203 | 0.0374 | +84 % (policy unconstrained, but still no termination) |

The training-time `last_reward` is materially higher under A2 (the unconstrained actor can drive harder), but the eval-time `mean_reward` distribution is essentially identical to the smoke. This is the diagnostic shape of "the binding constraint is NOT in the action constraint — it is in the env's success predicate not being driven to true." The ego learns to climb the shaped reward (reach + grasp + place + static) but does not complete the 4-stage sequence within the 50-step eval horizon.

**Routing implication.** Per the driver's decision rule and the brief's logic, A1 (partner heuristic) is the next ablation. A2's negative result also has independent value for ADR-004 §Open questions: the λ-saturation hypothesis flagged in the 2026-05-20 ADR-004 amendment ("the conformal overlay's ε=−0.05 design may be empirically vacuous under the constant-velocity predictor stub") is empirically vacuous in the sense that it does not affect policy convergence in this regime — but the operational λ behaviour itself (drift to clamp, no QP infeasibility, no braking fires) is consistent across firings.

Cost: ~8 GPU-minutes (5× cheaper than expected because the safety-stack overhead is the bulk of the per-step cost; disabling it makes the rollout ~6× faster on RTX 2080). Total investigation compute budget so far: 48.4 + 8.2 = 56.6 GPU-minutes.

## A1 Findings — zero-action partner at eval (NEGATIVE on success; structurally identical to smoke)

Driver: `.local/a1_zero_action_partner.py` (committed as `a1_zero_action_partner.log`; SpikeRun at `spike_as_hetero_a1_zero_action_partner.json`; per-step events at `7dd889f55295b33f.jsonl`).

```
Condition: stage1_pickplace_panda_plus_fetch_ego_aht_happo_per_agent
Episodes: 20
Success rate: 0/20 = 0.0%
n_terminated: 0; n_truncated: 0
mean_reward: min=0.0000  max=0.0719  avg=0.0280
Total wallclock: 50.5 min
```

The training-time partner is the factory's own scripted partner; only the EVAL-time partner is swapped for a `_ZeroActionPartner` (subclass of `ScriptedHeuristicPartner` for the ADR-009 black-box-policy gate; `.act()` returns `np.zeros(action_dim)` every step). Same training cfg, same training seed; only the eval inner loop's partner changes.

**Falsification rule from the driver:** `success_rate >= 10-20%` would fingerprint Surface 3 as dominant. Outcome: 0/20.

### The byte-identical anomaly (load-bearing diagnostic)

Per-episode `mean_reward` (formatted to 4 decimal places) from the smoke (scripted partner at eval) vs A1 (zero-action partner at eval):

| episode | smoke `mean_reward` | A1 `mean_reward` | smoke==A1 |
|---|---|---|---|
| 00 | 0.0000 | 0.0000 | yes |
| 01 | 0.0000 | 0.0000 | yes |
| 02 | 0.0000 | 0.0000 | yes |
| 03 | 0.0000 | 0.0000 | yes |
| 04 | 0.0000 | 0.0000 | yes |
| 05 | 0.0016 | 0.0016 | yes |
| 06 | 0.0085 | 0.0085 | yes |
| 07 | 0.0325 | 0.0325 | yes |
| 08 | 0.0445 | 0.0445 | yes |
| 09 | 0.0600 | 0.0600 | yes |
| 10 | 0.0385 | 0.0385 | yes |
| 11 | 0.0516 | 0.0516 | yes |
| 12 | 0.0719 | 0.0719 | yes |
| 13 | 0.0421 | 0.0421 | yes |
| 14 | 0.0394 | 0.0394 | yes |
| 15 | 0.0434 | 0.0434 | yes |
| 16 | 0.0329 | 0.0329 | yes |
| 17 | 0.0239 | 0.0239 | yes |
| 18 | 0.0380 | 0.0380 | yes |
| 19 | 0.0320 | 0.0320 | yes |

**Smoke and A1 produce identical per-episode reward sequences to 4 decimal places across all 20 episodes.** A2 (safety disabled) does NOT — A2's rewards diverge from the smoke starting at episode 05 (smoke 0.0016 vs A2 0.0000), confirming the eval is *sensitive* to env-state changes when something materially differs.

Two competing explanations:

1. **The scripted partner is structurally a no-op against the Stage-1b ManiSkill fetch.** `ScriptedHeuristicPartner._read_agent_xy` reads `obs["agent"][partner_uid]["state"][:2]` — pre-P1.05.8 that mapped to fetch's first two joint positions (NOT a Cartesian xy; mobile-base joint coordinates). Even post-widening the partner's own `state` Box stays at `concat(qpos, qvel)` (only the ego's `state` was widened to include task fields — the partner side asymmetry is documented in `chamber.envs.stage1_obs_filter` and pinned by `test_partner_state_excludes_task_and_ego`). The partner's `_compute_action` does `dx = clip(target_x - x, -1, 1); action[0] = dx; action[1] = dy; rest = 0`. With fetch's wheel joints starting near 0 and `target_xy = (0, 0)`, the partner emits ≈ zero action vectors; physics keeps wheel positions at ≈ 0; the heuristic stays at the fixed point. Empirically identical to the zero-action wrapper.

2. **The trained policy is partner-insensitive at this seed/budget.** The widened ego state includes `partner_qpos + partner_qvel` (30-D for fetch). If the actor MLP's input weights for those slots are near-zero (orthogonal init + Adam under 100k frames + no informative gradient on those slots), the policy could ignore the partner channel entirely. The deterministic eval would then produce identical actions regardless of partner movement.

A2's comparison (rewards diverge) rules out a third explanation ("the env doesn't apply the partner's action at all").

The two explanations have **different remediation implications**:
- Under explanation 1 (no-op partner): Surface 3 is a phantom defect. The original brief's "partner races toward cube spawn" framing was empirically wrong for fetch. Fix is to either (a) replace the scripted partner with one that actually moves the fetch base (constant-velocity drive, or one that reads Cartesian pose from `comm.pose`), or (b) accept that Stage-1b AS has a no-op partner and verify whether the AS heterogeneity gate is still measurable under that.
- Under explanation 2 (partner-insensitive policy): the widening worked structurally but the policy didn't learn to use partner state. Fix is upstream of P1.05.8 — possibly hidden_dim too small (64 → 128/256), or training budget (100k → 500k+ to give SGD time to discover the partner channel as informative).

### Routing implication — A5 promoted from conditional to RUN

The brief's original conditional ("run A5 only if A1 lifts the gate") assumed A1's outcome would cleanly separate partner-as-targeting from partner-as-presence. The byte-identical-to-smoke finding inverts that assumption: A1 didn't lift the gate, but the reason might be that the partner is empirically irrelevant (explanation 1) — in which case A5 with `target_xy="0.5,0.5"` is the cheapest empirical test of the no-op hypothesis. If A5 produces byte-identical eval rewards, the scripted partner is confirmed no-op for fetch (explanation 1 wins). If A5 produces materially different eval rewards, the partner DOES move with a non-(0,0) target and the policy responds to the partner channel — pointing at explanation 2.

Cost: ~50 GPU-minutes per run. Cumulative investigation compute after A1: 48.4 + 8.2 + 50.5 = 107.1 GPU-min. Adding A5: ~157 GPU-min worst case — still cheap vs the 200+ GPU-h of a "fix-and-relaunch" approach.

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
