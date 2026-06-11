# Success-static decomposition probe — why `is_robot_static` never consolidates

**Date.** 2026-06-10.
**Author.** Farhad Safaei.
**Status.** New dated directory (I8). Probes and replays only — **no training runs**; no edits
to the success predicate, static threshold, goal_thresh, reward terms, episode_length, or
partner heuristic.
**Question.** `success = is_obj_placed ∧ is_robot_static` never consolidates (A1 training
success ≤ 0.0016; cold 0/30 on all three seeds) despite 30/30 cold grasp+place. Which of five
named causes is binding?
**Builds on.** `../2026-06-10-regime-alignment/FOLLOWON_NOTE_DRAFT_2026-06-10.md` §(a) (reads 1
and 2 are executed here, extended); ADR-007 §Stage 1b through Rev 17; ADR-004 (bounds);
`evaluate()` at `src/chamber/envs/stage1_pickplace.py` (panda-routed predicate).

## Candidate causes (named for the record)

- **C1 — incentive/discount:** stillness pays too little at γ=0.8.
- **C2 — partner jostling:** the heuristic partner physically perturbs the hold (cooperation
  signal if true; quantify, never patch).
- **C3 — rig:** `is_static` mis-reads through the multi-agent shim.
- **C4 — place flicker:** `is_obj_placed` oscillates at the goal-sphere boundary.
- **C5 — settle time:** placement lands too late in the 100-step episode to decelerate.

## Step 1 — zero-compute archive decomposition (`window_decomposition.json`)

From the five archived per-window JSONLs (granularity caveat: window aggregates; run lengths /
margins / timing require the replays below):

| run | static_rate overall → last-50 | static_rate in place>0.5 windows vs elsewhere | last-100 P(success \| placed-step) |
|---|---|---|---|
| A1 s0 | 0.0125 → **0.0006** | 0.0011 vs 0.0218 | 0.00017 |
| A1 s1 | 0.0128 → **0.0004** | 0.0009 vs 0.0227 | 0.00012 |
| A1 s2 | 0.0144 → **0.0004** | 0.0009 vs 0.0234 | 0.00018 |
| A2 s0 (γ=0.99) | 0.0091 → 0.0008 | (place sparse) | **0.00106** |
| A3 s1 (γ=0.99, 1 env) | 0.0208 → 0.0043 | (place sparse) | **0.0063–0.018** (8 windows) |

Three archive-level facts: (i) stillness **declines with training** on every A1 seed — the
policy learns to move more, not less; (ii) stillness is *rarer* during well-placed windows than
elsewhere (the "elsewhere" static is early-training post-reset rest); (iii) the conditional
P(success | placed-step) is **monotonically increasing in γ** (≈0.00016 → ≈0.001 → ≈0.006+),
matching the cold-eval chain P(static∧placed | placed): A1 0/90, A2 2/16, A3 1/1. The C1
fingerprint, visible before any replay.

## Step 2 — rig unit probe (C3; `rig_probe_results.json`)

Source: upstream `Panda.is_static(0.2)` is `max |qvel[..., :-2]| ≤ 0.2 rad/s` — own
articulation, finger DOF excluded; `PandaWristCam` inherits it unchanged; the env's
`evaluate()` routes it through `agents_dict["panda_wristcam"]` (ego only). Live probe through
the production factory shim: ego articulation distinct from partner's (9-DOF); **static at
rest True**; **static under full partner motion True** (partner |qvel| 3.15 while the flag
holds); ego arm motion → False; gripper-only motion (finger |qvel| 0.18, arm |qvel| ~1e-7) →
True (finger exclusion works); `evaluate()["is_robot_static"]` ≡ the method. **C3 dead.**

## Steps 3+4+5 — checkpoint replay: settle time, flicker, proximity, frozen partner

A1 seed-0 policy reloaded from the final archived checkpoint (step **19,000,320** — the
20M-frame in-memory policy was cold-evaled by the run driver, but the vector loop's
frame-bucketed checkpointer last fires at 19M since 19,531×1,024 = 19,999,744 never reaches the
20M bucket; rig nit recorded in the readiness note). Eval env at `episode_length=200`
(eval-rig knob; H3-protocol precedent — H=100 and H=200 reads derive from the same
deterministic pass). Episode seeds replicate the production cold-eval stream. 30 episodes live
partner + 10 episodes frozen partner (zero action — eval-rig knob, stated as such).
Artifacts: `replay_summary.json`, `replay_{live,frozen}_partner_steps.jsonl` (per-step
`arm_qvel_max`, `cube_goal_dist`, `tcp_fetch_dist`, both flags).

| read | live partner (30 eps) | frozen partner (10 eps) |
|---|---|---|
| placed by H=100 / H=200 | 30/30 / 30/30 | 10/10 / 10/10 |
| **success by H=100 / H=200** | **0/30 / 0/30** | **0/10 / 0/10** |
| first-place step (median, max) | 11, 19 | 10, 16 |
| placed fraction after first place | 1.000 (single run, ~190 steps) | 1.000 |
| arm max\|qvel\| in placed steps: median (min) | **0.521 (0.317)** rad/s | 0.525 (0.341) |
| placed steps with qvel < 0.2 / < 0.3 | **0 / 0** of 5,637 | 0 / 0 of 1,893 |
| static∧placed steps | 0/5,637 | 0/1,893 |
| tcp–fetch distance, slow-ish vs moving placed steps | 0.646 vs 0.635 m (no separation) | 0.620 vs 0.626 m |

- **C5 dead.** Placement completes by ~step 11 and is held; ~190 further steps of available
  settle time at H=200 produce zero successes. Time is not the constraint.
- **C4 dead.** Zero flicker: after first placement, `is_obj_placed` holds every remaining step
  as one unbroken run (the follow-on note's flicker hypothesis is refuted; the hold is deep,
  not boundary-grazing).
- **C2 dead.** The frozen-partner set is statistically indistinguishable from live on every
  measure (hold qvel 0.525 vs 0.521; 0 successes both; no proximity–stillness correlation).
  The partner neither jostles the hold nor explains the motion.
- **C1 confirmed, with margins.** While holding the placed cube the policy sustains arm motion
  at median max|qvel| ≈ **0.52 rad/s — 2.6× the 0.2 threshold — and never once dips below the
  threshold in 7,530 placed steps** (minimum 0.317). This is not borderline jitter the
  predicate unluckily clips; it is a stable moving-hold regime. The incentive arithmetic
  matches: the forfeited static term `1 − tanh(5‖qvel‖)` at qvel 0.52 is ≈ 0.011 of a max 0.2
  reward-units/step (post-normalisation), and the discounted tail value of *full* stillness
  while placed is ≈ 1.0 at γ=0.8 vs ≈ 20 at γ=0.99 — and success has only ever fired on
  γ=0.99 arms (A2 2/16, A3 1/1, A1 0/90).

## Verdict

**C1 — incentive/discount — pure.** C3, C4, C5 are killed outright by direct measurement; C2 is
killed by the frozen-partner control and the absent proximity correlation (so there is no
partner-mediated cooperation signal hiding in the static conjunct either — relevant to the
gate framing). The policy holds the placed cube indefinitely in a moving equilibrium ~2.6×
above the static threshold because, at γ=0.8, the discounted incremental return of
transitioning to stillness (≈1.0) does not outweigh the learned hold pattern, while at γ=0.99
(where it is worth ≈20) stillness begins to appear exactly as often as the archives show. Not
C2, not mixed — per the brief's stop-gate, Phases 2–3 proceed.

## Artifacts (this directory; SHA256SUMS covers all)

`window_decomposition.json`, `rig_probe_results.json`, `replay_summary.json`,
`replay_live_partner_steps.jsonl`, `replay_frozen_partner_steps.jsonl`,
`factory_dispatch_verification.json` (readiness-note evidence, recorded here for I8 locality).

## Cross-references

`../2026-06-10-regime-alignment/` (characterization §2–§3; follow-on note §(a));
ADR-007 §Stage 1b Rev 17 (merged, PR #213); ADR-002 (provenance caveats per the RUNID-COLLISION
finding); ADR-004 §Decision (no bounds touched); PR #212 (evidence vintage under analysis).
