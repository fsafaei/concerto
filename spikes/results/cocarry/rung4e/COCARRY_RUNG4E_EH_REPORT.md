# Co-carry Rung 4e — the task-fair embodiment search → a robust EH null

**Verdict: `EH_ROBUSTLY_NOT_ESTABLISHED`.** Under a bounded, task-fair search
over the xArm6's **pose × controller** against the **full** joint-success
criterion, a fairly posed+controlled different body (xArm6 + Robotiq) co-carries
the bar **as well as the matched Panda body** — joint success **1.0** on 12
held-out confirm seeds, every conjunct in-band together. The Rung-4d candidate
(a stress↔tilt pose tradeoff) is **resolved**, not confirmed: the search found a
configuration that is in-band on *both*. The pre-registered hypothesis — that the
6-DOF xArm6 lacks the redundancy to satisfy stiffness *and* leveling at once — is
**falsified**.

Pre-registered before any episode was scored:
`cocarry_rung4e_taskfair_prereg.json` (tag
`prereg-cocarry-rung4e-taskfair-2026-06-20`). Reproduce:
`bash scripts/repro/cocarry_rung4e_taskfair_search.sh` →
`cocarry_rung4e_taskfair_search.json`.

## The result

Fixed cooperative reference ego, compliant coupling K=8000, invariant coupling
stress measure, f_max 366 N, 15° tilt ceiling, 0.10 m radius — all reused
verbatim; **only** the xArm6 pose × controller varied.

| Condition | joint success | stress p90 (N) | tilt p90 (°) | centroid p90 (m) |
|---|---|---|---|---|
| **xArm6, fairly optimised** (base 0.40, level_gain 0.5, admit_bias 0.7) | **1.0** | 291.3 | **1.6** | 0.011 |
| matched Panda anchor | 1.0 | 279.3 | 6.4 | 0.006 |

(f_max 366 N; tilt ceiling 15°; radius 0.10 m; confirm seeds 81100–81111, n=12.)

The fairly-posed xArm6 is **in-band on every conjunct** (stress 291 < 366, tilt
1.6 < 15, placed, static) — and is actually *more level* than the matched Panda
pair (1.6° vs 6.4°). It co-carries as well as the matched body. There is no
qualifying embodiment drop, and the null is not an artifact of under-search (see
adequacy below).

## Why the Rung-4d "tradeoff" was a two-pose snapshot, not a law

The search (4 search seeds, 30 configs) shows the stress/tilt outcome is
**non-monotonic in carry pose** — neither the stiffest pose nor the default is
best:

| xArm6 pose | endpoint vcompliance | stress p90 (N) | tilt p90 (°) | in-band on both? |
|---|---|---|---|---|
| base 0.30 (min-vc) | 4.46 (0.94× Panda) | 444–728 | 39–42 | no (both over) |
| base 0.35 (min-vc) | 3.96 (0.84× Panda) | 334–346 | 31–33 | no (over-tilt) |
| **base 0.40 (min-vc)** | **3.16 (0.67× Panda)** | **290–299** | **1.4–2.5** | **YES (all controllers)** |
| base 0.45 (min-vc) | 2.34 (0.49× Panda) | 253–259 | 22–28 | no (over-tilt) — *the Rung-4d pose* |
| default base 0.35 | 6.91 (1.46× Panda) | 1734–1754 | 7–8 | no (over-stress) — *the Rung-4c pose* |

Rung-4c measured only the **default** pose (over-stress) → false "embodiment
over-loads" headline. Rung-4d measured the default and the **stiffness-optimal**
pose (base 0.45, over-tilt) → the apparent stress↔tilt tradeoff. Both were
**snapshots of a non-monotone landscape**. The fair search over the reach
geometry finds **base 0.40**, where the 6-DOF arm satisfies stiffness *and*
leveling simultaneously, across *every* controller variant tested. The tradeoff
is a property of two unlucky poses, not of the embodiment.

## Search adequacy (the null is not an under-search artifact)

The pre-registered fairness-both-ways criteria:

- **Budget ≥ the Panda's tuning effort:** 30 pose×controller configs × 4 search
  seeds = 120 scored search episodes — a documented search at least as thorough
  as the matched Panda controller's deliberate tuning.
- **Plateau:** the best joint-deficit reached **0.0 at config 13** (the first
  base-0.40 config) and stayed there; tail improvement over the final 40% of the
  budget = 0.0 ≤ 0.02. The search converged.
- **In-band point exists:** the stress↔tilt Pareto frontier (4 non-dominated
  placed+static configs) *contains* in-band-on-both points (base 0.40:
  stress 289.9, tilt 1.5, js 1.0). Closest normalised approach to the in-band
  box = **0.0** — the box is not merely approached, it is **reached**. So the
  "EH established via multi-objective infeasibility" branch is correctly **not**
  taken (one cannot claim infeasibility when a feasible point is in hand).

## What this means

1. **Embodiment heterogeneity does not degrade cooperation on this task** under
   fair pose+controller matching, against a fixed cooperative ego. A robust,
   honest EH null.
2. Combined with the Rung-3 PH null (policy heterogeneity Δ ≈ 0 under capability
   matching), co-carry shows **both** policy *and* embodiment heterogeneity are
   benign under fair matching on this task — a clean methods+result contribution
   (the rigid-weld over-load and the two "drops" were all instrument/pose
   artifacts, each caught and corrected by a cheaper check before any heavy
   spend).
3. **The incumbent re-freeze (Rung-4d Stage B/C) is not warranted.** It was
   conditional on EH being established; it is not. There is no embodiment effect
   for an incumbent-faithful Δ to measure on this task.

## Honest scope of the claim

- The contrast is against the **fixed cooperative reference ego** (the
  capability/feasibility question: *can a fair different body co-carry?* — yes).
  It is ego-independent in the same sense the Rung-4c/4d coupling results were.
- The null is **task-scoped**: this co-carry task, this bar/goal geometry, this
  compliant coupling, the matched-family compliant control style. A heavier or
  more dynamic task, or a stiffer control regime, could still surface an
  embodiment effect; this result does not claim universality.
- The xArm6 controller was held in the **matched compliant family** (kp not
  stiffened) so the null is not bought by making the partner artificially soft;
  only the leveling/yielding levers and the pose were searched.

## What's next (the mandatory decision)

Per the pre-registration's framing, this was the **last axis-measurement spend**
before the synthesis + scope decision. With both PH and EH closing as robust
nulls under fair matching, the strategic call — whether the thesis lives in a
**different (heavier/more dynamic) task regime**, what M10 claims, and the
Phase-1/2 scope + M7 cut — is now due. That is a founder decision; this report
is its last input. See `COCARRY_RUNG4E_SYNTHESIS.md`.

_ADR-026 §Decision 2 / §Decision 4 / §Validation criteria; ADR-005; ADR-009;
R-2026-06-B §15 Rung 4e. Generator
`scripts/repro/_cocarry_rung4e_taskfair_search.py`; search seeds 81000–81003,
confirm seeds 81100–81111; f_max 366 N; compliant K=8000; tilt ceiling 15°;
C_min 0.75 — all reused, none weakened._
