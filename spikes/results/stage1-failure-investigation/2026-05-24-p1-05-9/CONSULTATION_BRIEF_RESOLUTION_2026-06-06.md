# Stage-1b AS — Third §4a Firing: Consultation Brief (RESOLUTION UPDATE)

**Date.** 2026-06-06.
**Author.** Farhad Safaei.
**Supersedes / updates.** `spikes/results/stage1-failure-investigation/2026-05-24-p1-05-9/THIRD_FIRING.md`
(the at-firing brief; its §3 enumerated six surviving candidates). This update reports the
investigation that ran between 2026-05-24 and 2026-06-06 and **resolves the surviving-candidates
question to a single, confirmed, validated root cause.**
**Status.** DECISION-INPUT for the third-firing consultation. Factual sections (§0–§6, §8) are the
mechanical input; §9 founder prior is held off-disk per the discipline THIRD_FIRING.md §4 argued.
Nothing is merged or shipped on the strength of this brief alone (see §7).
**Solo-founder substitute discipline (carried from THIRD_FIRING.md §preamble).** No advisor exists.
The substitute holds: (a) the case written in full before deciding; (b) a **seven-day sleep** on the
conclusions before any code lands; (c) the rationale recorded in the ADR amendment when the slice
opens; (d) named cross-checks; plus the third-firing extensions (e) literature cross-checks and (f) a
written devil's-advocate pass — the latter is discharged here in §5.

---

## §0 — Machine-readable summary

```yaml
firing: 2026-05-24-p1-05-9            # AS-hetero, success_rate 0/20 @ 1M frames + hidden_dim=256
status: ROOT-CAUSE FOUND, FIXED, VALIDATED-AT-PATHOLOGY-LEVEL (sufficiency pending gate-scale run)
root_cause:
  id: ENV-NO-TRUNCATION
  one_line: >
    Stage1PickPlaceEnv never enforced episode_length (ManiSkill BaseEnv.step hard-codes
    truncated=False); the training loop resets only on terminated|truncated, so the env was reset
    ONCE (single cube/goal config for the whole run) and GAE saw a non-terminating MDP whose critic
    converged to the infinite-horizon value r/(1-gamma) with collapsed advantages.
  verified: [empirical (0 boundaries/250 steps), code (sapien_env.py L1069), metric (value_mean=r/(1-gamma))]
surviving_candidates_resolution:        # vs THIRD_FIRING.md §3
  d_eval_horizon:   ROOT-CONFIRMED + DEEPENED (env never truncates at all; not merely eval 50-vs-100)
  a_exploration:    REFUTED (entropy healthy/rising; failure is advantage-collapse, not exploration)
  b_reward_discont: DOWNSTREAM (policy never reached the grasp transition; bridge @0.15 did not help)
  c_repositioning:  NOT INDICATED (task feasible; defect was an env bug, not task infeasibility)
  e_curriculum_bc:  SECOND-LINE (fix re-enabled learning; curriculum is a possible accelerant, not the cause)
  f_method_vs_scaling: RESOLVED -> METHOD (1M frames did not help because the MDP itself was broken)
fix:
  pr: 206 (merged then reverted via PR 207 to keep it consultation-gated; preserved on fix/p1-05-9-env-truncation)
  change: env step() truncates at episode_length (BaseEnv elapsed_steps); eval-cap sourced from DEFAULT_EPISODE_LENGTH
validation_200k_exploratory:            # .local; NOT a gate result
  value_mean: {baseline_broken: 14.74, fixed: 0.53}     # infinite-horizon runaway eliminated
  advantage_std: {baseline_broken: 0.0013, fixed: 0.024_and_rising}   # learning signal restored
  episodes_in_run: {baseline_broken: 1, fixed: 2011}    # MDP now episodic
  mean_reward: {trend: "2e-5 -> 0.038 (climbing past baseline plateau)"}
  grasp_rate: 0 ; ever_grasped: 0/5     # NOT yet emerged at 200k frames (under-trained, not stalled)
decision_asked:
  - Q1: greenlight landing the env-truncation fix (merge #206 / revert #207)?
  - Q2: ratify the horizon contract (ADR-007 rev) + reset-cadence determinism re-check (ADR-002 rev)?
  - Q3: authorise the decisive gate-scale validation (1M frames x >=3 seeds) as the sufficiency test?
recommendation: YES to Q1/Q2/Q3, gated on the 7-day sleep + the §5 devil's-advocate holding.
gates_unchanged: ADR-007 status stays Accepted; no AS->Validated promotion until a gate-scale PASS lands.
```

## §1 — Resolution since the firing

THIRD_FIRING.md left six surviving candidates and a consultation question of the form "which of
these, and is this method-vs-scaling?" The 2026-05-24 → 2026-06-06 investigation answered it. The
chain (all artifacts under `.local/experiment_analyses/` + the committed
`spikes/results/.../2026-06-06-missing-gripper/`):

1. **Embodiment cleared.** A rollout looked like the panda had no gripper. Forensics (committed
   investigation dir) proved the hand + fingers are present, actuated, and rendered — embodiment is
   sound; the under-reach is behavioural, not hardware. Removed "broken embodiment" from the set.
2. **Metric-health forensics** (`METRIC_HEALTH_value_vs_return_2026-06-06.md`). The firing's PPO
   scalars show `value_mean ≈ 14.74`, `value_loss → 2e-6`, `advantage_std → 0.001`, rising entropy,
   `grasp_rate 0/488`. `14.74 = r/(1-gamma)` exactly (per-step reward 0.147, γ=0.99) — the **correct
   value of a never-terminating MDP**, not a runaway. This *refuted* the value-runaway reading and
   pointed at a missing episode boundary.
3. **Root cause isolated** (`ROOT_CAUSE_no_env_truncation_2026-06-06.md`) — see §2.
4. **Fix + validation** — see §3.

## §2 — Root cause: ENV-NO-TRUNCATION

`Stage1PickPlaceEnv` stores `episode_length` and stubbed `_step_count` but **never wrote the
`step()` override that enforces them**. ManiSkill v3 `BaseEnv.step` hard-codes `truncated=False`
(`sapien_env.py` L1069 — truncation is the gym `TimeLimit` wrapper's job, which a directly-built
subclass lacks). Verified: the env emits **0** truncations in 250 steps; `max_episode_steps` absent.
The training loop resets only on `terminated | truncated`, so:

- the env was **reset once** → a **single cube/goal configuration for the entire 1M-frame run**
  (`n_episodes = 1` in the firing's `training_end`); and
- GAE saw a **non-terminating MDP** → the critic learned `r/(1-gamma)` and **advantages collapsed**
  (no gradient), with rising entropy — the exact reach-without-grasp / under-reach fingerprint.

This **subsumes THIRD_FIRING.md §3 candidate (d)** (it is deeper than "eval 50 vs 100" — the env
never bounded an episode in *training*), and reframes the others: (a) exploration is **refuted**
(entropy was healthy), (b) the grasp-reward discontinuity is **downstream** (the policy never
reached it; the reward bridge at weight 0.15 over 500k frames left `ever_grasped 0/5`), (c)
re-positioning away from PickCube is **not indicated** (the task is feasible; this was an env bug),
(f) **method, not scaling** (1M frames could not help a structurally broken MDP).

## §3 — The fix and its validation

**Fix** (PR #206, on `fix/p1-05-9-env-truncation`; merged then reverted via #207 to stay gated):
`Stage1PickPlaceEnv.step()` ORs in truncation from BaseEnv's public `elapsed_steps >= episode_length`
(the subclass analogue of the `_step_count` idiom in `mpe_cooperative_push` / `stage0_smoke_adapter`);
the eval cap (`_max_steps_for("1b")`) is sourced from `DEFAULT_EPISODE_LENGTH` so train and eval share
one horizon. Reward/obs/render-neutral.

**Exploratory validation** (200k frames, single seed, bridge OFF; `.local`, NOT a gate result;
`TRUNCATION_VALIDATION_RESULTS_2026-06-06.json`):

| metric | baseline (broken) | fixed (200k) | reading |
|---|---|---|---|
| episodes in run | 1 | **2011** | MDP now episodic ✅ |
| `value_mean` (final) | 14.74 (= r/(1−γ)) | **0.53** | infinite-horizon runaway eliminated ✅ |
| `advantage_std` | →0.0013 (collapsed) | **0.024, rising** | learning signal restored ✅ |
| `mean_reward` | ~0.02 then plateau | **2e-5 → 0.038, climbing** | improving past baseline ✅ |
| `grasp_rate` / `ever_grasped` | 0 / 0 | **0 / 0** | not yet emerged ⚠️ |

The three pathologies the root cause predicted are all resolved, and — unlike the broken run —
learning is *alive and improving*. Grasping has not emerged at 200k frames; the trajectory is
"under-trained," not "stalled" (200k ≪ the 1M+ PickCube needs).

## §4 — Consultation question (re-framed)

The original question ("which of six candidates?") is answered. The decision is now:

- **Q1.** Greenlight landing the env-truncation fix? (merge #206 / merge the revert-revert; the fix
  is validated at the pathology level.)
- **Q2.** Ratify the **horizon contract** as an ADR-007 §Stage-1b revision (the per-episode horizon
  is implementation-detail-of-the-cell — the prereg pins seeds + `episodes_per_seed`, NOT the step
  horizon — so **no prereg tag rotation**) and the paired **ADR-002 reset-cadence determinism
  re-check**? (Stubs drafted: `ADR-007_REV_horizon_contract_STUB`, `ADR-002_REV_..._STUB`.)
- **Q3.** Authorise the **decisive sufficiency test** — a gate-scale **1M-frame × ≥3-seed** run with
  the fix — as the experiment that determines whether the gate can clear?

Recommended: **YES to all three**, conditioned on the §5 devil's-advocate holding after the 7-day
sleep. Q3 is the real scientific test; Q1/Q2 are its preconditions.

## §5 — Devil's advocate (discharging the third-firing discipline extension)

The strongest case *against* "root cause found, proceed":

- **Sufficiency is unproven.** The 200k run did **not** produce a single grasp. The claim that this
  is "under-training, not a second blocker" rests on trajectory shape (advantages alive, reward
  climbing), not on an observed grasp. It is possible the bounded-MDP policy still plateaus
  pre-grasp for a *second* reason — the grasp-reward discontinuity (old candidate b) or the 7-DOF
  joint-space exploration manifold — that the truncation fix does not touch. If the 1M×3-seed run
  returns `ever_grasped ≈ 0`, candidates (b)/(e) and the action-space remedy (P-H2-a) re-activate,
  and we will have spent a gate-scale budget to learn it.
- **Mitigation already in hand.** The remediation proposal (`REMEDIATION_PROPOSAL_2026-05-24-p1-05-9.md`)
  pre-registers those second-line arms as *conditional-on-the-Q3-read*, so a null result is still
  informative and the next step is pre-decided, not improvised.
- **Verdict of the devil's advocate:** proceed to Q3, but **state the sufficiency claim as a
  hypothesis under test, not a conclusion**, and pre-commit the branch (grasp emerges → land; no
  grasp but signature healthy → P-H2-a action-space arm; signature regresses → re-open).

## §6 — ADR / prereg implications

- **No prereg edit, no tag rotation.** `spikes/preregistration/{AS,OM}.yaml` pin `seeds` +
  `episodes_per_seed` + estimator + bootstrap — **not** the step horizon. The horizon is
  implementation-detail-of-the-cell per ADR-007 §Discipline (same class as the Rev-14 budget/capacity
  bumps). Recorded as an ADR-007 §Revision entry, not a new prereg tag.
- **Comparability.** Prior firings ran under the broken (boundary-free / 50-step-eval) regime; their
  `gap_pp` numbers are not horizon-comparable to post-fix runs. Archives remain immutable; the
  revision establishes the contract going forward.
- **Determinism (P6 / ADR-002).** The reset-cadence change alters the per-episode seed stream;
  re-validate the CPU byte-identical contract under the new cadence and record it (ADR-002 stub).
  Verified locally: same `root_seed` → identical cube/goal; per-episode re-randomisation restored.
- **ADR-007 status stays Accepted.** No AS → Validated promotion until a gate-scale PASS lands; the
  §Reversibility exit-ramp is **not** triggered (this is a Stage-1b implementation-contract
  amendment, not an axis-selection reversal).

## §7 — What does NOT happen before the consultation closes

- #206 stays **off `main`** (revert PR **#207** restores the gated state; merge it to complete).
- No gate-scale spike, no new prereg tag, no ADR amendment landed, no AS promotion.
- The fix, the ADR revisions, and the gate-scale run wait on the 7-day sleep + the founder cut.

## §8 — Audit artifacts (all referenced; immutable archive untouched)

- Firing record: `spikes/results/stage1-failure-investigation/2026-05-24-p1-05-9/` (THIRD_FIRING.md, JSONL, FAILED.json).
- Embodiment: `spikes/results/stage1-failure-investigation/2026-06-06-missing-gripper/` (committed, #205).
- Decision-input (`.local/experiment_analyses/`): `FIRING_ANALYSIS_2026-05-24-p1-05-9.md`,
  `METRIC_HEALTH_value_vs_return_2026-06-06.md`, `ROOT_CAUSE_no_env_truncation_2026-06-06.md`,
  `REMEDIATION_PROPOSAL_2026-05-24-p1-05-9.md`, `ADR-007_REV_horizon_contract_STUB_2026-06-06.md`,
  `ADR-002_REV_reset_cadence_determinism_STUB_2026-06-06.md`,
  `TRUNCATION_VALIDATION_RESULTS_2026-06-06.json`.
- Code: PR #206 (fix), PR #207 (revert-to-keep-gated), branch `fix/p1-05-9-env-truncation`.

## §9 — Founder prior — RESERVED
<!-- held off-disk per THIRD_FIRING.md §4; record the in-conversation prior + the chosen branch in
     the Rev-15+ ADR amendment after the 7-day sleep, alongside the §5 devil's-advocate pass. -->
