# ADR-030: Size the pickcube contract horizon to a reach-margin gate (v3)

**Status.** RFC (2026-09-24) — the founder flips this to Accepted (or amends) at PR review.
**Authors.** Farhad Safaei
**Reviewers.** _solo lock per ADR-INDEX working policy_
**Tags.** pickcube task contract; sim/real EE-delta tracking; amends the code-level 200-step horizon and the 50% per-step tracking gate (neither governed by a prior ADR).

## Context

The `pickcube` contract (v2) commands ±5 mm end-effector translation per step at
20 Hz over a 200-step horizon, realised in sim by `ScaleDeltaActions` in front of
ManiSkill's `pd_ee_delta_pose` controller (`use_target=False`). Empirical
characterization (2026-09-24) shows the sim controller realises a **first-order-lag
fraction** of each commanded delta: `fraction = 1 − e^(−T/τ)` with **τ ≈ 100 ms**
(τ equals the joint-PD damping/stiffness ratio), i.e. only **~39%** of the 5 mm at
20 Hz (T = 50 ms). Two consequences follow:

- **Reach per step ≈ (1 − e^(−T/τ)) · (commanded lead)**, and the lead is capped by
  the conservative safety envelope at 5 mm, so realised reach ≈ **1.9–2.3 mm/step
  regardless of controller mode**. An accumulating target (`use_target=True`)
  clamped to the 5 mm envelope is numerically identical to `use_target=False`
  (measured 38.9%); its headline ~92% only appears when the target is allowed to
  wind up 8–13 mm past the arm — a 1.6–2.5× envelope breach, i.e. a contact hazard
  (the gripper lunging when it is freed after being blocked against the cube or the
  table during the pick).
- PickCube's mean pick-then-place TCP travel is **~0.39 m**; at ~1.9 mm/step a
  200-step horizon yields ~0.39 m of budget — **zero margin**, and the harder
  resets (~0.46 m) are infeasible even for a perfect controller.

A decision is forced before any GPU training is spent on an unsolvable spec.

## Considered alternatives

| # | Option | Pros | Cons |
|---|--------|------|------|
| A | Shrink τ via higher joint-PD stiffness (smaller damping/stiffness ratio) | Restores ~87% per-step tracking at 20 Hz, no contract change | Under the assumption the real arm tracks like the sim (~39%), a faster sim breaks sim/real parity; raises peak contact forces ~1.6× |
| B | Accumulating-target controller (`use_target=True`) | ~92% in free space | Unbounded windup on contact (safety); clamped to the envelope it equals `use_target=False` (38.9%) — buys no reach and changes the observation (42→49) |
| C | Lower control rate to 10 Hz at 200 steps | More sim-time per step → 62% tracking | τ is **not** floored by velocity/torque limits, so 10 Hz is unnecessary; halves the decision rate for contact; equals option D in episode time and sim substeps |
| **D** | **Lengthen the horizon to ~400 steps at 20 Hz (chosen)** | Sizes the budget for ≥2× reach margin; keeps 20 Hz, `pd_ee_delta_pose`/`use_target=False`, the 42-dim obs and the 5 mm envelope untouched; same 20 s episode and same sim substeps/episode as 10 Hz@200 but keeps the faster decision rate for contact | Longer episodes; a fixed discount covers a smaller share of the episode (see Risks) |

## Decision

Introduce pickcube contract **version 3**, identical to v2 except
**`max_episode_steps = 400`** (was 200). Controller (`pd_ee_delta_pose`,
`use_target=False`), control rate (20 Hz), observation layout (42-dim) and per-step
action bounds (5 mm / 0.05 rad) are unchanged. Replace the per-step **"≥50%
tracking"** acceptance gate with a **reach-margin gate**: the sim must be able to
deliver **≥2× the task's required pick-place travel** within the horizon
(`horizon × realised-per-step-reach ≥ 2 × required-travel`). The ~39% tracking is
treated as a **rescaled action space, not a defect** — joint velocity is in the
observation, so the plant is Markovian and learnable.

**Margins recorded for v3** (400 steps × ~1.94 mm/step = ~776 mm budget):
- **2.00×** against the **mean** pick-place travel (~0.387 m) — **this is the value
  the gate asserts** (`REQUIRED_REACH_M = 0.387`, mean over resets).
- **1.70×** against the **worst-case** reset travel (~0.456 m).

The gate uses the mean; the worst-case is accepted at 1.70× for now. If, in a
training run, episodes time out near the goal with a high `target_clamp_fraction`,
that is the trigger to revisit with a ~460-step horizon (~2.0× worst-case) — as a
new deliberation, not a mid-run contract edit.

## Rationale

The reach shortfall, not per-step tracking, is the only thing that blocks learning,
and it is bounded by envelope × lag — which no controller trick escapes safely.
Lengthening the horizon is the single minimal change that restores solvability
without touching the controller, the observation, the action scale, or the safety
envelope, and without asserting a sim tracking rate the (uncharacterized) real arm
may not match. The 50% per-step gate is **self-inconsistent**: under the same
assumption used to reject gain-tuning (real arm ≈ 39%), the real arm would itself
fail a 50% gate, so per-step tracking cannot be the acceptance criterion — reach
margin is.

## Evidence basis (links to reading notes)

- Empirical control-lag characterization recorded 2026-09-24: first-order-lag fit
  `1 − e^(−T/τ)`, τ ≈ 100 ms constant across 10–200 ms control periods; τ = joint-PD
  damping/stiffness ratio (ratio sweep); clamped-target equivalence to
  `use_target=False`; pick-place reset-geometry travel (~0.39 m mean).
- _No Tier-1/Tier-2 reading note underpins this engineering decision; per the
  ADR-INDEX locking rule this ADR remains RFC until the founder promotes it at PR
  review._

## Consequences

- **concerto:** new gym id `concerto/pickcube-v3`; `get("pickcube")` now returns v3;
  v1/v2 stay registered and immutable. Unit and integration tests updated for the
  new latest version.
- **robolab:** `GYM_ID` → `concerto/pickcube-v3`; the sim parity gate moves from a
  50% per-step floor to a ≥2× reach-margin check; checkpoint and eval-report
  provenance now record v3.
- Amends the **code-level** 200-step horizon and the 50% tracking gate (neither was
  governed by a prior ADR).
- No change to the safety envelope, controller, observation layout, or action scale.

## Risks and mitigations

- **[Discount horizon]** γ stays at the reference 0.99. A 400-step episode is covered
  less fully by a fixed γ than a 200-step one (the effective horizon 1/(1−γ) = 100
  steps is now a quarter of the episode, was a half). Treated as a **known risk**;
  revisit γ only if training shows value-propagation failure over the longer episode.
- **[Real-arm parity assumption — tripwire]** The rejection of gain-tuning rests on
  the assumption that the real arm's EE-delta step response is comparably slow
  (τ on the order of ~80–100 ms), taken from the mock real-env model and **not
  measured on hardware**. _Tripwire:_ characterize the real arm's realised fraction
  of a 5 mm EE-delta command at 20 Hz. If it tracks substantially faster
  (≥ ~85% at 20 Hz, τ ≪ ~30 ms), matching the sim joint PD to the real τ becomes
  legitimate and would restore high per-step tracking with no contract change — and
  a shorter horizon might then suffice; revisit this ADR. If the real arm is
  comparably slow, the longer horizon is the correct, parity-preserving choice.
- **[Episode wall-clock]** 400-step episodes double per-episode sim compute vs 200;
  per-step throughput is unaffected but there are 2× more steps per episode. No
  mitigation needed — the total-timestep budget is counted in env steps.

## Reversibility

Type-2 (two-way door). v3 is additive; reverting is dropping the version and
pointing consumers back at v2.

## Validation criteria

1. The reach-margin gate reports **≥2×** for v3 in sim.
2. FlashSAC on `concerto/pickcube-v3` shows a positive learning slope in episodic
   return and a non-trivial eval success rate within an adequate budget. Failing (2)
   reopens the horizon / γ / task-geometry question.

## Open questions deferred to a later ADR

- Real-arm τ characterization (the tripwire above).
- Whether γ needs raising for the longer horizon.
- Whether the required-travel constant in the reach-margin gate should track the
  worst-case reset rather than the mean.
