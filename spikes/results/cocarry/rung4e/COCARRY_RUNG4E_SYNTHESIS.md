# Co-carry heterogeneity ladder — synthesis + the scope decision (post-Rung-4e)

This closes the co-carry forward-design ladder (ADR-026 §Decision 4;
R-2026-06-B §15). With Rung-4e, every staged heterogeneity axis the ladder set
out to measure has a pre-registered verdict. This document synthesises them and
states the decision that is now due — explicitly a **founder call**, not a
mechanical next run.

## The ladder, end to end

| Rung | Axis | Verdict | One-line |
|---|---|---|---|
| 2 | (incumbent freeze) | incumbent FROZEN | step 100k, 24/24 held-out, SHA-verified — the fixed reference all measurements use. |
| 3 | **PH** (policy heterogeneity) | **robust null** | pooled Δ = +0.083 < 0.20; capability-matched policy-shift teammates (same Panda body) do not degrade cooperation. One stiff teammate dropped (control-style, not policy). |
| 4 | **EH** (embodiment), rigid weld | feasibility wall (artifact) | rigid weld over-couples; xArm6 excluded at calibration. Later shown to be a **rigid-coupling + instrument** artifact, not embodiment. |
| 4b | compliant coupling | task-physics blocker resolved | compliant weld removes the over-coupling; residual block isolated to the **embodiment-biased wrist stress proxy**. |
| 4c | invariant stress instrument | (claimed EH drop) → **retracted** | the bar-coupling-force instrument is sound; the EH "drop" drawn from it was a **default-pose artifact** (Rung-4d). |
| 4d | fair-pose falsification | pose artifact → STOP | a feasible xArm6 pose carries in-band; the 4c over-load was the unoptimised default pose. Left a *candidate* stress↔tilt tradeoff. |
| **4e** | **EH**, task-fair pose × controller | **robust null** | a fairly posed+controlled xArm6 co-carries as well as the matched Panda (js 1.0; stress 291 < 366; tilt 1.6 < 15). The candidate tradeoff was a two-pose snapshot of a non-monotone landscape; **EH does not bite** under fair matching. |

## The scientific bottom line

**Under fair matching, neither policy heterogeneity nor embodiment heterogeneity
degrades cooperation on this co-carry task.** Every apparent "drop" along the way
— the rigid-weld over-load, the wrist-proxy block, the Rung-4c coupling
over-stress, the Rung-4d tilt tradeoff — turned out to be an **instrument or pose
artifact**, and each was caught by a *cheaper* check before any heavy spend
(re-freeze, substrate build). The discipline (pre-register, isolate one factor,
let a cheap check undermine the headline, report the correction) is what produced
a trustworthy null instead of a fragile positive.

This is a real, publishable outcome: a **negative result + a methods
contribution** (the compliant coupling, the embodiment-invariant coupling
instrument, the capability gate, the task-fair pose×controller search with a
falsifiable-both-ways adequacy criterion). It is *not* a thesis-confirming
"heterogeneity is hard" headline.

## What is NOT warranted

- **The incumbent re-freeze (Rung-4d Stage B/C).** It was conditional on EH being
  established. EH is a null → there is no embodiment effect for an
  incumbent-faithful Δ to measure on this task. Do not spend it.
- **A new EH measurement axis on this task.** The axis is closed (robust null).
  Further pose/controller search would only reconfirm the null.

## The decision that is now due (founder)

The ladder has spent its last axis-measurement on this task regime. The honest
inputs to the M7 / Phase-1-2 scope / M10 call are:

1. **The thesis as originally framed (heterogeneity degrades cooperation, the
   safety stack mitigates) does not reproduce on this co-carry task** under fair
   matching — at the policy or the embodiment axis. The benign result is robust,
   not a measurement failure.
2. **Option A — change the task regime.** Move the thesis to a heavier / more
   dynamic / contact-richer task where heterogeneity plausibly *does* bite (the
   quasi-static co-carry may simply be too forgiving — the compliant coupling and
   matched-family control absorb the embodiment difference). The instruments and
   harness built here transfer.
3. **Option B — re-frame M10 around the null + methods.** Claim the negative
   result and the measurement methodology (artifact-catching discipline,
   invariant instrument, fair-search adequacy) as the contribution; narrow the
   safety-stack claim to where it is actually load-bearing.
4. **Option C — pursue the remaining weaker axes (OM/CM/CR) on this task.** Lower
   expected yield: if PH and EH (the strongest-prior axes) are null under fair
   matching, OM/CM are likely null too — but cheap to check.

These are strategic, not mechanical. **This is the founder's call** (it has been
flagged as overdue across the last rungs; the EH outcome was its last awaited
input). Recommendation for discussion, not decision: **A or B** — either relocate
the thesis to a regime where the effect exists, or commit to the null+methods
framing — rather than spending more on axes of this (forgiving) task.

_ADR-026 §Decision 4; R-2026-06-B §15. Evidence: `spikes/results/cocarry/rung3`
(PH), `rung4b`/`rung4c`/`rung4d`/`rung4e` (EH). Reports + committed generators +
pre-registration tags per rung._
