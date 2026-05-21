# Stage-1b AS failure — consultation brief

**Date.** 2026-05-21
**Author.** Farhad Safaei
**Trigger.** `PHASE1_IMPLEMENTATION_PLAN.md` §4a runbook fired on 2026-05-20: Stage-1b AS launch returned `gap_pp = 0` (`chamber-spike next-stage --prior-stage 1` exit 5).
**Status of this brief.** Decisions taken below. Re-read 24 hours after first draft before any code is shipped; if the rationale still holds, proceed to §6.
**Solo-founder substitute discipline.** The runbook names "senior-advisor consultation" as the gate. No advisor exists for this project. This brief substitutes by (a) writing the case in full before deciding, (b) imposing a 24-hour sleep on the conclusions, and (c) recording the rationale in an ADR amendment when the remediation slice opens — so a future reader can audit the call.

---

## 1. What happened

PR #175 + #179 + #180 + #181 closed the P1.05 engineering slice (Stage-1b dispatch on the AS / OM adapters). The first real Stage-1b AS science launch ran on the RTX 2080 box on 2026-05-20 (≈8.7 h wall-clock; 5 seeds × 2 conditions × 100k training frames + 20 eval episodes per cell). Every audit-discipline invariant held — `schema_version=2`, `sub_stage="1b"`, `prereg_sha` matches `prereg-stage1-AS-2026-05-15`'s tagged blob, λ telemetry captured per cell, audit-gate predicates A + B both PASS. **The failure is the science evidence, not a corruption of it.** Bootstrap CI: `[0.00, 0.00]`; both AS-homo and AS-hetero report 100% success at the rule-based threshold — i.e. the success rule is a rubber stamp, not a measurement.

The §4a runbook discipline fires for the first time at exactly the cheapest possible failure point: ~30 GPU-h sunk cost, before any Stage-2 / Stage-3 compute has been committed. The pre-committed plan-time discipline is working.

## 2. Dominant defect — Surface 6 (observation contract)

`Stage1PickPlaceEnv` emits the right fields: `obs["extra"]["cube_pose"]` (7-D), `obs["extra"]["goal_pos"]` (3-D), `obs["extra"]["tcp_pose"]` (7-D), and per-uid `qpos` / `qvel` under `obs["agent"][uid]` for both the ego and the partner. The wrapper chain (`chamber.envs.stage1_obs_filter.Stage1ASStateSynthesizer`) synthesises `obs["agent"][uid]["state"]` as `concat(uid's own qpos, uid's own qvel)` only — shape `(18,)` per agent. `chamber.benchmarks.ego_ppo_trainer.EgoPPOTrainer._flat_ego_obs` (line 466) reads exactly `obs["agent"][ego_uid]["state"]` and nothing else. The ego trained for 100k frames seeing only its own arm joints; cube, goal, and partner state were structurally invisible to the optimiser.

This is a Tier-2 acceptance gap, not a wire bug. The synthesizer was correct for Stage-1a (MPE Cooperative-Push, agent-relative landmarks, 18-D ego state sufficient). The Stage-1a → Stage-1b sub-stage swap moved the trainer onto a manipulation env that needs absolute-frame task observables and the obs contract was never widened. Tier-2 tests verified the trainer ran end-to-end against `Stage1PickPlaceEnv` but did not assert what fields the trainer actually read.

Mechanical consequence: A1 (zero-action partner), A2 (safety disabled), A3 (5× training budget) are all non-recovering remedies against Surface 6 — no policy whose only input is its own arm `qpos+qvel` can locate a randomly-spawned cube.

## 3. Confirming positive evidence — Surface 7 (reward function)

Static analysis (`chamber/envs/stage1_pickplace.py:918-949`): direct port of upstream `PickCubeEnv.compute_dense_reward` (reach → grasp → place·grasp → static·placed → 5·success, divided by 5). Panda-routed via `self._panda_agent`. No sign flip, no scale bug, no missing component. Per-rollout-window reward climbs in every seed (seed 3: 0.0001 → 0.0175, +143×). The env's success terminal IS reachable end-to-end (`mean_reward = 1.0` observed in ~1% of evaluation episodes). The reward function is structurally fine; the policy is climbing a shaped signal through impoverished input. **This narrows the remediation surface to Surface 6 cleanly.**

## 4. Three decisions

### Q1 — Widen the trainer's obs reader to include cube / goal / partner state? **YES.**

The defect is not a science choice; it is a wrapper that wasn't updated when the env class changed. Every published `PickCube` baseline trains against ManiSkill's native `obs_mode="state"`, which flattens cube + goal + tcp + per-agent state into one Box. Upstream `PickCube` SAC gets ≈ 0.5 mean episodic reward with that observation; this trainer gets ≈ 0.027 without it. The size of the defect matches the size of the missing fields.

### Q2 — Does direct partner-pose observation violate ADR-009's ad-hoc-partner contract? **NO — with one documented nuance.**

ADR-009's "frozen black-box partner" contract is about *policy* access (weights, reward, training data) and is enforced by the `_FORBIDDEN_ATTRS` shield on `PartnerBase`. It is not a contract that the partner's physical pose is invisible to the ego. A real robot's cameras / LIDAR / proximity sensors see other robots' poses; observing the partner's joint state in simulation is the same affordance, not policy access. The named AHT precedents (Liu 2024 RSS, COHERENT, Huriot–Sibai 2025) all let the ego observe partner state.

The interesting variant — *should partner pose flow through the comm channel (ADR-003) instead of direct observation?* — matters for **Stage-2 CM**, where bandwidth / latency / drop on the channel is the axis under test. For Stage-1b AS, the heterogeneity is the partner's embodiment, not the channel; routing partner pose through ADR-003 here would conflate two axes. Direct observation is correct for Stage-1b.

**Documented nuance for the ADR-009 amendment.** With partner pose directly observed, the ego can in principle learn partner-specific behaviour, weakening the ad-hoc claim slightly. The existing conformal-CBF λ re-init on partner swap (ADR-004 §Risks #2) is the project's safeguard. Surface the nuance in writing; do not change the architecture for it.

### Q3 — Add a Tier-2 acceptance gate for "trainer reads task-relevant fields"? **YES.**

The xfail-strict regression-pin prompt is already drafted in the planning kit. Three xfail-strict tests covering `cube_pose` / `goal_pose` / `partner_state` visibility. Fails today on Surface 6; flips to xpass when the remediation lands and the marker MUST be removed in the same PR (precedent: P1.02 `Bounds.action_norm` xfail-strict). ADR-002 §Revision history gets an entry naming the Tier-2 acceptance gap. ~1 hour of work; closes the foot-gun forever.

## 5. What the consultation discipline buys (and what it does not)

Honest accounting: the §4a discipline was written for a setting with a human advisor. Working solo, I preserve (a) the pause between failure and remediation, (b) the requirement to state assumptions in writing, (c) the separation of engineering questions from science-contract questions, and (d) the requirement to amend the relevant ADRs alongside any code change. I do not get (e) a second pair of eyes catching reasoning errors. I substitute (e) with the 24-hour sleep and with cross-checks against the named precedents (Liu 2024, COHERENT, Huriot–Sibai 2025) and the upstream ManiSkill conventions. This is weaker than a real advisor and stronger than no discipline at all. It should be revisited if and when an advisor relationship becomes available.

## 6. Sequenced next steps

1. **24-hour sleep on this brief.** If the reasoning above still reads clean tomorrow, proceed.
2. **xfail regression-pin commit on PR #182.** Use the existing planning-kit prompt for the regression-pin commit. ~1 hour. Closes Q3 in process terms before any engineering change.
3. **Merge PR #182.** Investigation evidence lands on `main` as canonical record (per ADR-007 §Discipline + PR #149 precedent). M7 is officially BLOCKED at this point; ADR-007 stays at Accepted (no AS promotion to Validated).
4. **Open slice P1.05.8 — Surface 6 remediation.** Scope: widen `Stage1ASStateSynthesizer` (or the trainer's obs reader, depending on the cleanest seam) so the ego sees `cube_pose` + `goal_pos` + partner state. Concurrently fix Surface 1 (`_SUCCESS_THRESHOLD = -0.30` rubber-stamp → `success = terminated` per the env's own predicate). Amend ADR-009 §Decision with the direct-observation-is-not-policy-access paragraph. Amend ADR-002 §Revision history with the Tier-2 acceptance-gap finding. The xfail marker from step 2 flips to xpass and is removed in this same PR.
5. **One smoke run.** Re-run a single AS-hetero cell on RTX 2080 (~25-30 min) with the terminated-only success rule. Two outcomes:
   - Success rate ≥ 10-20%: the fix is real. Re-launch the full PR 5b spike. M7 unblocks. Resume the master plan.
   - Success rate still ~0%: NOW run ablations A1 / A2 / A5 against the corrected baseline. Their results are interpretable post-Surface-6-fix.
6. **Resume the master plan.** Stage-1b OM (P1.05.6 vision-head, or its successor given what step 4 implies for the OM trainer), Stage 2 (P1.06-08), Stage 3 (P1.09-12), Phase-1 closure (P1.13).

## 7. What is NOT decided here

- The vision-head extension for the OM trainer (P1.05.6) — separate slice, separate design pass, gated on AS gate clearance.
- Whether `λ_safe` should be derived rather than the Phase-0 placeholder `0.0` (ADR-004 §Open questions; out of scope for Surface 6).
- Whether `AgentSnapshot.qpos` extension (P1.05.5) is mandatory — still gated on Stage-1b λ saturation telemetry, which only becomes interpretable after Surface 6 is fixed and a real Stage-1b run completes.
- The ADR-019 / certification-target work from the 2026-05-20 strategy recommendation — orthogonal to the Phase-1 engineering remediation.

The single most important property preserved by this brief: the pre-registration tags do not rotate, the failed archive on disk stays as honest evidence, and the next code change carries an ADR amendment naming what changed and why. The audit chain stays closed.
