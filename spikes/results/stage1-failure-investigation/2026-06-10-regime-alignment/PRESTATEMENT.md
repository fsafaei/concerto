# Regime-alignment characterization — pre-statement

**Date.** 2026-06-10.
**Author.** Farhad Safaei.
**Status.** APPROVED by founder decision 2026-06-10, with amendments applied (single-variable
chain framing recorded in §3; PB1 data-arming made explicit in §4; audit-gate / eval-posture
sentences added to the Rev 17 draft §3; `action_linf_component` recalibration queued in the
planning kit's decision queue). **FROZEN at launch: from the first run's start this document is
immutable (I8) and no mid-run changes of any kind are permitted — runs are chained with no
intervention between them.**
**Scope guard (I1).** Rig/regime work inside P1.05, same class as the Rev 15 horizon fix. Gate
criteria, prereg axes (no tag rotation), task, reward, action space: untouched. No curriculum, no
reward changes, no entropy schedule, no demonstrations, no PB1 in this slice regardless of outcome.

---

## §1 — Question and grounding

**One question.** Does grasp **consolidation** (stochastic training behaviour → reliable cold
deterministic skill) appear once the Stage-1b training regime is aligned with field practice for
this task class — without any learning-signal lever?

Grounding chain:

- **Fix-only triplet (the single-variable baseline).** Under Rev 15 (horizon) + Rev 16 (reset
  state), grasp emergence replicates 3/3 in training but the cold bar (≥2/3 seeds) failed at 1/3
  (`../2026-06-09-grasp-remediation/RESETFIX_CHARACTERIZATION_seeds_2026-06-10.md`). Consolidation
  is the live problem; no lever is authorised.
- **Field-practice review** (`STAGE1B_FIELD_PRACTICE_REVIEW_2026-06-10.md`, planning kit): the
  published recipe for this task class is 1,024–4,096 parallel envs × 10–50M frames × γ=0.8
  (ManiSkill PickCube; the two-robot variant uses 20M), per-update batches 16k–51k transitions.
  Our cell: 1 env × 1M × γ=0.99 × 1,024-transition updates — 10–50× below on every axis, plus a
  training-time safety stack no published baseline carries.
- **Safety-interference probe**
  (`../2026-06-09-grasp-remediation/SAFETY_INTERFERENCE_PROBE_2026-06-10.md`): zero braking /
  fallback / QP-infeasible events in 3M filtered steps, but **material training-time interference
  via the QP's action-box clamp** (`action_linf_component = 0.1` against a [−1, 1] action space;
  ~80 % of sampled action components saturated; PPO buffered unexecuted nominal actions) **and a
  train/eval dynamics mismatch** (the eval closure has never applied the filter). The probe gates
  the filter-off arm IN.
- **Vectorised rig (Phase 2, this slice).** Branch `feat/stage1b-vectorised-cell`, commit
  `a1820ce5b47987e1aea75b8b038542c0f4175087`: `num_envs > 1` cells on ManiSkill `physx_cuda`,
  per-env `derive_substream` cube/goal streams (P6/ADR-002; verified cross-process on GPU),
  chamber-side auto-reset with Pardo-correct truncation bootstraps, batched HAPPO rollout with
  per-env GAE. All Tier-1/Tier-2 tests green. **The training-time safety stack is not batched in
  this slice; `num_envs > 1` loud-fails unless `safety.enabled=false` (the config's designed
  operator override).** Measured throughput (RTX 2080 SUPER 8 GB, AS-hetero cell):

  | num_envs | steps/s | device memory over baseline |
  |---|---|---|
  | 64 | 1,489 | ~1.3 GB |
  | 256 | 4,185 | ~1.4 GB |
  | **1,024** | **11,094 (sustained over 1.02M frames)** | **~1.9 GB** |
  | 2,048 | 15,747 | ~2.4 GB |

  Chosen N = **1,024**: inside the field band, validated at production length, and with
  `rollout_length=32` gives **32,768-transition updates** — inside the field's 16k–51k band
  (N=2,048 would force per-env rollouts below the field band to stay under 51k).

## §2 — Regime settings (all arms; frozen)

| setting | value | provenance |
|---|---|---|
| num_envs | 1,024 | §1 benchmark; field band |
| total_frames | 20,000,000 per run | two-robot ManiSkill variant's published budget; ≈30 min/run at measured steps/s |
| rollout_length | 32 per env (32,768-transition updates) | field per-update band |
| batch_size (minibatch) | 4,096 | 8 minibatches × 4 epochs per update |
| n_epochs / lr / gae_lambda / clip_eps / hidden_dim | 4 / 3e-4 / 0.95 / 0.2 / 256 | unchanged from the fix-only cell (Rev 14) |
| episode_length | 100 | Rev 15 contract, unchanged |
| reset state | canonical ready pose + open gripper | Rev 16, unchanged |
| reward / action space / task / condition_id | unchanged | I1 |
| safety (training-time) | `enabled=false` (operator override) | required at N>1 (Phase 2); justified independently by the probe; eval/audit posture recorded as "safety disabled by operator" per the yaml contract; deploy/eval-time safety stack untouched (ADR-004) |
| partner | scripted_heuristic, `target_xy 0.0,0.0` | unchanged |
| condition | AS-hetero (`stage1_pickplace_panda_plus_fetch_ego_aht_happo_per_agent`) | the campaign's cell; condition symmetry note: any gate-facing follow-up runs BOTH AS conditions under these identical settings |

## §3 — Arms (draft; founder may amend; frozen at launch)

- **A1 (primary): γ = 0.8, 20M frames, seeds 0/1/2.** Fix-only at the aligned regime. The
  primary decision rule (§5) reads A1 alone.
- **A2 (discount isolation): γ = 0.99, 20M frames, seed 0.** Identical to A1 except the
  discount — separates the parallelism/budget variable from the per-task-discount variable.
  Diagnostic read only; not part of the primary rule.
- **A3 (filter isolation, re-purposed; founder-approved 2026-06-10).** The brief's A3
  ("`safety.enabled=false` at the new N, only if gated in") is **redundant as written**: the
  filter cannot run at N>1, so every N>1 arm is already filter-off, and the A1-vs-baseline delta
  confounds {regime} with {filter removal}. A3 is therefore a **single-env** run: 1 env × 1M
  frames × γ=0.99 × `safety.enabled=false`, seed 1 — single-variable against the fix-only
  triplet's seed 1 (the only seed that grasped cold, 1/10). The probe predicts material
  improvement from removing the ±0.1 box clamp and the train/eval mismatch. ≈8 h on the local
  GPU (the old regime's wall-clock). Diagnostic read only; informs the `action_linf_component`
  recalibration decision (ADR-004-adjacent, queued in the planning kit's decision queue, not
  decided in this slice).

**Single-variable chain (founder framing, recorded at approval).** The four cells form one
isolation ladder: **fix-only baseline → A3** isolates the training-time filter (same env count,
frames, γ, seed); **A3 → A2** isolates parallelism + budget (same γ, filter-off on both);
**A2 → A1** isolates the discount (same N, frames, filter posture). Each arrow changes exactly
one variable class.

Run order (chained, no intervention between runs): A1-seed0 → A1-seed1 → A1-seed2 → A2 → A3
(A3 overnight).

## §4 — Instrument

Per run, after training completes:

- **Cold deterministic eval, N = 30 episodes per seed** — prereg'd init (Rev 16 ready pose, cold
  start; the PA3 curriculum env-var is not set anywhere in this slice), `num_envs=1` eval env,
  deterministic policy via the production `TrainedPolicyFactory` closure path. Eval posture is
  unchanged from every prior record (the eval closure has never applied the training-time filter
  — probe §3.3).
- **Report per seed AND per arm:** `ever_grasped` (count/30), the full grasp/place/success
  ladder — `grasp_rate`, `place_rate`, `success_rate` as training max-window rates AND
  cold-eval episode rates — and PPO health (`advantage_std`, `dist_entropy`, `value_mean`)
  from the per-window JSONL. **The place/success columns are mandatory on every arm so the PB1
  (place-conversion) question is armed with data on either branch of the §5 rule** — if A1
  consolidates, the same table says whether the staged reward's place rung climbs on its own at
  scale; if it does not consolidate, the consultation brief carries the place evidence anyway.
- **Artifacts per run (I8, new files in this directory):** results/signature JSON, per-window
  JSONL, trimmed launch log; `SHA256SUMS.txt` regenerated to cover all files.

## §5 — Decision rule (FROZEN at launch; no mid-run changes)

Primary read, on A1 only:

- **Regime explains consolidation:** ≥2/3 A1 seeds with cold `ever_grasped` ≥ 3/30. → The
  remediation campaign **closes with zero levers**; the next questions are place conversion at
  scale and the gate-spike regime spec (per the ADR-007 revision note).
- **Regime does not explain it:** anything else. → **PA3-at-scale becomes rung 1** (the staged
  reset-curriculum prompt, upgraded to N=1,024), and the **senior-advisor consultation fires**
  with this slice + the field-practice review as material — three structural explanations
  (truncation, reset state, sampling regime) would then be exhausted and demonstrations /
  off-policy (rung 2) is on the table.

A2 and A3 are diagnostic only: they cannot flip the primary verdict; their reads are recorded in
the characterization with the same evidentiary status as the probe.

**Characterization instruction (founder decision 2026-06-10, recorded so it is not lost):** the
characterization file records, in one line and without re-verdicting, that the PA1 and PA2
verdicts now carry two independent confounds — the broken reset-state init (Rev 16) and the
±0.1 action-box clamp with unfiltered eval (probe) — and remain suspended.

## §6 — Budget

Measured 11.1k steps/s at N=1,024 (sustained, 1.02M-frame validation): A1 = 3 × 20M ≈ 90 min;
A2 = 20M ≈ 30 min; A3 = 1M at 1 env ≈ 8 h; cold evals ≈ minutes each. **Total ≈ 10–11 h
wall-clock** — well under the ≤3-day cap, chained on the local RTX 2080 SUPER (one process per
run; GPU PhysX is once-per-process). No A100 needed for this slice.

## §7 — Constraints restated

I1 (no Phase-1 scope expansion), I6 (no AI-tooling references; founder authors every document),
I7 (every document/commit cites its ADR/log grounding), I8 (per-firing immutability; this
directory receives only new files after launch). No prereg tag rotation; no `SCHEMA_VERSION`
bump (P10); ADR-007 stays Accepted; identical regime settings for both AS conditions in anything
gate-facing.

## Cross-references

PR #206 / #210 / #211; ADR-007 §Stage 1b Revs 14–16 + the draft revision note in this directory;
ADR-002 (determinism; GPU 95 %-CI caveat); ADR-004 (safety stack; deploy posture untouched);
`../2026-06-09-grasp-remediation/` (REMEDIATION_LOG, RESETFIX_CHARACTERIZATION,
SAFETY_INTERFERENCE_PROBE); `STAGE1B_FIELD_PRACTICE_REVIEW_2026-06-10.md` (planning kit).
