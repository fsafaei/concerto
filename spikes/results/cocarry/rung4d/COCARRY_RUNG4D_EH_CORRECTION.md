# Co-carry Rung 4d — Stage-A1 correction: the Rung-4c EH headline is a pose artifact

> **RESOLVED by Rung-4e (2026-06-20).** The "honest residual" below — the
> stress↔tilt pose tradeoff flagged as a *candidate* embodiment difficulty — was
> itself a two-pose snapshot. The Rung-4e task-fair pose×controller search found
> base 0.40 (vcompliance 3.16) carries the bar in-band on **both** stress (291 N
> < 366) and tilt (1.6° < 15°), joint success 1.0 — comparable to the matched
> Panda. The candidate is **resolved**, the EH axis closes as a **robust null**.
> See [`../rung4e/COCARRY_RUNG4E_EH_REPORT.md`](../rung4e/COCARRY_RUNG4E_EH_REPORT.md).

**Bottom line.** The Rung-4c embodiment headline — *"a different body (xArm6)
genuinely over-loads the cooperative coupling ~5× f_max"* (PR #252) — **does
not survive a fair carry-pose optimisation.** When the xArm6 is given its best
fair configuration (the construct-validity audit's central ask), the coupling
over-load **disappears**. Per the pre-committed Stage-A1 stop rule, this is
reported as a correction and the heavy incumbent re-freeze is **not** run.

This is the cheap construct-closer doing its job: it caught a pose confound
*before* the expensive substrate build — and it changes the conclusion.

## What was wrong

Rung-4c measured the xArm6 at its **default** carry pose, whose vertical
endpoint compliance (J Jᵀ)⁻¹_zz ≈ 6.91 is 1.46× the Panda's (4.74) — and
attributed the coupling over-load to "poor carry-pose endpoint stiffness."
But endpoint stiffness is **configuration-dependent**, the xArm6's pose was
never optimised (the Panda's was), and the base sweep varied only *where the
arm is bolted*, not its *arm configuration*.

## The fair test (Stage A1)

A joint-limit-clamped kinematic search over base placement + the xArm6's 3-DOF
redundancy (the position weld leaves 3 redundant DOF) for the configuration
that **minimises** vertical endpoint compliance, then a committed env test
(invariant coupling measure, compliant K=8000) of default vs optimised pose,
static-hold and active-carry. Reproducible:
`scripts/repro/cocarry_rung4d_pose_falsification.sh` →
`cocarry_rung4d_pose_falsification.json` (uses the new committed
`make_cocarry_env(xarm6_base_x=, xarm6_ready_qpos=)` overrides — no monkeypatch).

| xArm6 pose | endpoint vcompliance | static coupling (p50/max) | active coupling p90 | active tilt p90 |
|---|---|---|---|---|
| **default** (Rung-4c) | 6.91 (1.46× Panda) | 1041 / 2632 N | **1842 N** (≫ f_max 366) | 9.1° (level OK) |
| **fairly-optimised** (feasible min-compliance) | **2.34 (0.49× Panda — *stiffer*)** | 152 / 159 N | **264 N** (< f_max 366; ≈ the matched 267 baseline) | 26.3° (**over-tilt**) |

(Panda endpoint vcompliance 4.74; f_max 366 N; matched baseline ~267 N.
Figures from the committed `cocarry_rung4d_pose_falsification.json`, seeds 80100–80105.)

## What this means

1. **A fairly-posed xArm6 is *stiffer* vertically than the Panda** (compliance
   2.34 vs 4.74), not more compliant — the Rung-4c mechanism claim was an
   artifact of the unoptimised default pose.
2. **The coupling over-load is a pose artifact.** Fairly posed, the xArm6 holds
   and carries the bar **in-band** on coupling (static max 159, active p90 264,
   both < f_max 366; active ≈ the matched-Panda 267 baseline). The "embodiment
   over-loads the coupling" headline does **not** hold.
3. **Per Stage-A1, STOP.** The construct gate failed, so the heavy re-freeze
   (Stage B) — which would have built the validated substrate around this
   headline — is **not** run. Failing cheap is the point.

## The honest residual (a candidate, not a finding)

The stiffness-optimal pose, while in-band on coupling, then **fails the *level*
conjunct** (active tilt p90 ≈ 26.3° > 15°). So across the poses tested the xArm6
shows a **stress↔tilt tradeoff** (default: level OK / over-stressed;
stiffness-optimal: in-band stress / over-tilted), whereas the matched Panda
pair does both. This *could* be a genuine, narrower embodiment difficulty — but
it is **not established**: neither pose was optimised for the *task* (carry
level at low stress), only for stiffness or by default. Establishing it
requires a **task-fair pose + controller optimisation** (minimise the joint
success deficit, not endpoint compliance) — the recommended next slice, *before*
any EH claim or re-freeze.

## Status updates

- **EH axis: NOT established.** The committed coupling-over-stress EH result
  (Rung-4c, PR #252) is **retracted as a pose artifact**. No qualifying EH drop
  stands; no qualifying null is claimed either (the task-fair search hasn't run).
- **Substrate: not built.** The re-freeze (Stage B) is deferred — there is no
  validated EH headline for it to support yet. The compliant coupling + the
  invariant coupling instrument (both validated, hardened) remain sound and
  reusable; only the EH *conclusion* is withdrawn.
- **Prior artifacts untouched (I8).** Rigid ladder + Rung-4b + the Rung-4c
  instrument/invariance artifacts stand as-is; this slice adds the correction
  and the `xarm6_base_x`/`xarm6_ready_qpos` override + the falsification
  generator. Rung-4c's report carries a correction banner pointing here.

## What the next slice must do

A **task-fair xArm6 pose + controller search**: optimise the xArm6 carry
configuration *and* its controller to minimise the joint-success deficit
(level + placed + unstressed + static together), not a single proxy. Outcome:
either a pose exists that carries cleanly → **EH null** (embodiment does not
degrade cooperation on this task); or no fair pose clears all conjuncts → a
genuine, precisely-characterised embodiment difficulty (likely the stress↔tilt
tradeoff) → **EH real**, established robustly. Only then re-freeze the substrate.

_ADR-026 §Decision 2 / §Decision 4 / §Validation criteria; ADR-005; ADR-009;
R-2026-06-B §15 Rung 4d. Generator: `cocarry_rung4d_pose_falsification.sh`;
seeds 80100–80105; f_max 366 N; compliant K=8000._
