# Safety-stack training-time interference probe — fix-only triplet, zero-compute

**Date.** 2026-06-10.
**Author.** Farhad Safaei.
**Status.** New file in this directory (I8: all prior records untouched). Zero-compute analysis of
the archived fix-only characterization triplet — no new runs, no code or config changes.
**Question.** Does the training-time safety stack (CBF-QP outer filter + braking fallback, wired
per P1.04.5/P1.04.6, ADR-007 §Stage 1b Revs 7–8) materially interfere with grasp approaches during
Stage-1b AS-hetero training? This gates whether the regime-alignment pre-statement includes a
training-time filter-off arm.
**Evidence base.** Per-window `safety_telemetry` + rollout `scalar` JSONL events of the fix-only
triplet (seed 0 `4193162f5e313d7f`, seed 1 `add44268f15c0674`, seed 2 `93d0b85c9ceeb804`; 976
windows × 1,024 steps × 3 runs = 3,000,000 filtered steps), the `safety_telemetry_final` summaries,
and code inspection of the live filter path at the runs' pinned SHA (`dca03ad`).

---

## §1 — What the archive measures, and what it cannot

The trainer logs one `safety_telemetry` event per 1,024-step rollout window (aligned with the
rollout `scalar` events carrying `grasp_rate`), with four intervention channels: `n_braking_fires`
(`concerto.safety.braking.maybe_brake` per-step TTC backstop), `n_fallback_fires` (the CBF-QP's
internal 1-D projection fallback), `n_qp_infeasible` (QP raised), and the conformal-λ window stats.

**Granularity caveat.** The QP's per-step `FilterInfo["constraint_violation"]` and the executed
(post-filter) action are **not** logged — `SafetyAggregator.observe` takes only λ and the three
boolean flags (`src/concerto/training/safety_telemetry.py`). A QP that solves *feasibly* but
returns a modified action ("quiet clipping") is therefore invisible to the archived counters. §3
closes that gap by code inspection plus the logged *nominal*-action statistics; the per-window
gripper telemetry records the **nominal** (pre-filter) action
(`src/concerto/training/ego_aht.py:789`), not the executed one.

## §2 — Archived telemetry: every escalation channel is zero

| run | windows | n_braking_fires | n_fallback_fires | n_qp_infeasible | n_filter_calls | grasp-active windows | escalations within grasp-active windows |
|---|---|---|---|---|---|---|---|
| seed 0 `4193162f5e313d7f` | 976 | 0 | 0 | 0 | 1,000,000 | 324 | 0 |
| seed 1 `add44268f15c0674` | 976 | 0 | 0 | 0 | 1,000,000 | 301 | 0 |
| seed 2 `93d0b85c9ceeb804` | 976 | 0 | 0 | 0 | 1,000,000 | 149 | 0 |

- **Braking fallback: zero fires in 3,000,000 evaluations.** `maybe_brake` evaluates pairwise TTC
  at every step; zero fires means pairwise TTC never dropped below `tau_brake = 100 ms`
  (`configs/training/ego_aht_happo/stage1_pickplace.yaml`) at any training step of any seed —
  including every grasp-active window. The proximity concern (the scripted partner targets the cube
  spawn region `target_xy 0.0,0.0`, so pairwise TTC is smallest exactly where grasping happens) is
  conclusively answered for this layer: the tracked pair geometry (panda **TCP** vs fetch **base
  link**, radii 0.10 m + 0.35 m, bases at ∓0.615 m; `src/chamber/envs/stage1_pickplace.py:205,219,370`)
  never approached the braking threshold.
- **QP escalations: zero.** No infeasibility, no internal fallback, on any step.
- **Correlation with grasp activity is vacuously zero.** With all counters identically zero in all
  774 grasp-active and 2,154 grasp-inactive windows, there is no braking/CBF-escalation signal to
  correlate. At this channel's granularity the stack looks inert.
- **λ dynamics confirm a zero-violation regime.** On all three seeds `lambda_mean` drifts
  monotonically from 0 to the symmetric clamp floor −7.0 (= `clamp_floor_ratio 0.7 ×
  cartesian_accel_capacity 10`; P1.05.7 / #180) and pins there from step 15,360 (window 15, 1.5 %
  of the run); `lambda_max_observed = 0.0` on every run. The −η·|ε| floor-rate drift is the
  documented no-prediction-gap signature (#180): the conformal layer never once observed a
  predictor under-estimate of the barrier. Audit-gate predicates A/B: pass (clamp-saturated regime,
  operationally correct per the P1.05.7 contract).

Read in isolation, §2 says the safety stack never *escalated*. It does **not** say the stack never
*modified actions* — that channel is unlogged, and §3 shows it was active at almost every step.

## §3 — Code-level finding: the QP's action-box clamp saturates the policy's action space

The CBF-QP stacks box rows `|u_i| ≤ bounds.action_linf_component` over **every** component of the
ego action on **every** solve (`src/concerto/safety/cbf_qp.py:870-871`). The Stage-1b env pins
`action_linf_component = 0.1` (`src/chamber/envs/stage1_pickplace.py:1321`, a Phase-1 default in
`build_bounds()`), while the env's `pd_joint_delta_pos` action space is normalised to [−1, 1] and
the HAPPO Gaussian policy samples unsquashed. Since braking never fired (§2), the QP ran on all
1,000,000 steps of each run and every executed ego action was the minimal-norm projection of the
nominal into the ±0.1 box (the pairwise CBF row adds nothing — see below). Three consequences:

1. **~10× authority reduction at every training step.** Whenever the nominal action component
   exceeds ±0.1, the executed component is pinned at ±0.1. The archived nominal-action statistics
   show this was the dominant regime:

   | run | windows with nominal \|gripper_cmd_mean\| > 0.1 | overall nominal gripper min | grasp-active-window gripper_cmd_mean (avg) | final dist_entropy → per-comp σ | P(\|a_i\| > 0.1), zero-mean lower bound |
   |---|---|---|---|---|---|
   | seed 0 | 712 / 976 | −2.41 | −0.26 | 3.86 → 0.39 | ≈ 0.80 |
   | seed 1 | 443 / 976 | −2.38 | −0.07 | 3.75 → 0.39 | ≈ 0.80 |
   | seed 2 | 844 / 976 | −2.70 | −0.57 | 3.90 → 0.39 | ≈ 0.80 |

   (σ from the 8-D diagonal-Gaussian entropy identity; the bound is conservative — non-zero means
   raise the saturation fraction. Seed 2 commanded a *mean* gripper closure of −0.57 in its
   grasp-active windows; the executed closure command was −0.1.)

2. **PPO learns on actions that were not executed.** The rollout loop buffers the **nominal**
   action (`trainer.act` at `src/concerto/training/ego_aht.py:658`; the HARL trainer buffers its
   own `act()` output) while the env steps the **filtered** action. For the ~80 % of sampled
   components outside the box, variation in the nominal produces zero variation in behaviour, yet
   PPO assigns credit along the nominal — exploration variance is mostly spent in a saturated,
   gradient-flat region of action space. This is a structural candidate for the campaign's
   "fragile-commitment" phenomenology (REMEDIATION_LOG §2: both more reward density and more
   entropy destabilised the grasp).

3. **Train/eval dynamics mismatch.** The deterministic eval closure applies **no** filter
   (`src/chamber/benchmarks/stage1_common.py:394` — `trainer.act(obs, deterministic=True)`
   straight into the env). The cold eval therefore runs the policy under per-step action authority
   up to 10× larger than anything it experienced during training. This directly confounds the
   emergence-vs-cold-consolidation gap that
   `RESETFIX_CHARACTERIZATION_seeds_2026-06-10.md` isolated as the live problem: the stochastic
   grasp behaviour was learned under clipped dynamics, and the cold instrument measures it under
   unclipped dynamics.

**Why the pairwise CBF row itself never bound.** With λ pinned at −7, the row's allowance at
grasp-zone geometry (TCP↔fetch-base distance ≈ 0.62 m, excess ≈ 0.17 m over the 0.45 m combined
radius, `alpha_pair = 20`, `cbf_gamma = 5`) is still ≈ 3–6 m/s² of ego acceleration toward the
partner — but actions confined to the ±0.1 box cannot realise Cartesian accelerations anywhere near
that allowance, so the box dominates the QP everywhere. This is fully consistent with §2's zeros
and with λ never receiving a violation signal: **the box clamp is also why the rest of the stack
looks inert.**

## §4 — Verdict

**Material training-time interference is indicated — but through the filter's action-box clamp,
not through braking or CBF-row activity.** The archived intervention counters alone would say "no
interference"; that read is wrong because the interfering channel (per-step QP box projection) is
unlogged. The evidence chain is: code inspection of the live filter path (§3, exact lines cited) +
archived nominal-action statistics showing the box was exceeded in 44–86 % of windows on the
gripper channel alone + the entropy-derived ≈80 % per-component saturation lower bound.

**Gate decision input.** The Phase-3 pre-statement **should include the training-time filter-off
arm (A3)**: `safety.enabled=false` is the config's designed operator override
(`configs/training/ego_aht_happo/stage1_pickplace.yaml` §safety), eval/audit posture recorded as
"safety disabled by operator". Two additional consequences for the slice:

- **Condition symmetry is preserved either way** — the clamp applies identically to both AS
  conditions (the ego is the panda in both), so prior *comparative* reads are not invalidated;
  absolute rates on both sides were suppressed alike.
- **The `action_linf_component = 0.1` value itself needs a founder decision** (out of scope for
  this probe): it reads as an uncalibrated Phase-1 default rather than a deliberate envelope for a
  [−1, 1] normalised action space, and at N>1 the same value would propagate into any batched
  filter. Whether to recalibrate it (a `fix`, ADR-004-adjacent) or to characterise via A3 first is
  a pre-statement question, not a lever this probe may pull.

## Cross-references

- ADR-004 (safety stack; deploy-time posture untouched by anything here), ADR-007 §Stage 1b
  Revs 7–8 (telemetry wiring), Rev 15/16 (the fixes the triplet characterises), ADR-002
  (determinism contract).
- Issue #180 / P1.05.7 (λ symmetric clamp; the −7 pin is its documented operating point).
- `REMEDIATION_LOG.md` §1–§2, `RESETFIX_CHARACTERIZATION_seeds_2026-06-10.md` (the triplet under
  analysis), PR #206 / #210 / #211.
- Field-practice review (`STAGE1B_FIELD_PRACTICE_REVIEW_2026-06-10.md`, planning kit): no published
  baseline for this task class carries a training-time filter — this probe quantifies what ours
  does.
