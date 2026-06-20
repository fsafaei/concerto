# Co-carry spec §4 amendment proposal — rigid weld → calibrated compliant drive

**Status:** proposal for the founder to fold into the co-carry spec
(R-2026-06-B §4) — the spec lives in the planning kit (not in this repo), so
this is recorded here as the committed amendment text. Grounded in the Rung-4
over-coupling finding (`spikes/results/cocarry/rung4`) and the Rung-4b
compliant-coupling result (`spikes/results/cocarry/rung4b`).

## What §4 currently says (rigid dual-hold attach)

§4 specifies the dual-hold attach as a **rigid weld**: each gripper link is
coupled to its bar end by a high-stiffness 6-DOF drive with the three linear
axes **hard-locked** (`set_limit_{x,y,z}(0,0)`), rotation free. Stiffness/
damping = 20 000 N/m / 2 000 N·s/m. This removes grasp slip so failures reflect
coordination, not fingertip friction.

## Why amend it

The hard-locked weld is what makes the attach rigid (the locked limit, not the
drive stiffness, carries the load). Rung-4 showed this **over-couples
embodiment**: a different-bodied partner (xArm6 + Robotiq) is forced to track
the bar end with zero compliance, so the two different arms fight through the
bar at ~518 N (vs the matched Panda pair's ~75 N) and cannot be
capability-matched — an artifact of coupling *stiffness*, not of cooperation.
Rung-4b confirmed this: a passive compliant drive at the stiffest valid setting
lets the xArm6 transport and level the bar (the over-coupling wall disappears),
while the matched pair stays clean and the coupling still binds.

## Proposed amended §4 text

> **§4 Dual-hold attach (calibrated compliant drive).** Each gripper link is
> coupled to its bar end by a passive 6-DOF spring-damper drive
> (`ManiSkillScene.create_drive`), rotation free. The three linear axes are
> driven toward the grip frame (zero-offset target) with a **calibrated
> stiffness `K_c`**; damping scales with `K_c` at the reference ratio
> (0.1 N·s/m per N/m, conservatively overdamped). `K_c` is a **frozen-able task
> parameter** selected per coupling-study by the C1–C4 calibration
> (matched-pair clean with small deflection; single-arm ≈ 0; coupling binds;
> the embodiment teammate admitted), choosing the stiffest setting that
> satisfies all four. The **rigid** weld (hard-locked axes, `K_c → ∞`) remains
> the default for the embodiment-symmetric Rungs 0–3 and is byte-identical to
> the committed ladder; embodiment-heterogeneous studies (Rung 4b+) use the
> calibrated compliant drive. The attach is still a weld, not a grasp (no
> gripper-force action, no friction) — only its compliance is calibrated.

## Companion predicate caveat (for §15 / the success predicate)

The Rung-4b gate also surfaced that the **stress proxy** (`is_unstressed`,
wrist incoming joint force vs f_max) is **embodiment-dependent**: the xArm6
wrist reads ~3–6× the Panda at an identical static hold, over the
Panda-calibrated f_max before any fight. A cross-embodiment success predicate
needs an embodiment-invariant stress measure (e.g. the weld constraint force)
or a per-embodiment f_max calibration. That is a separate, deliberate
predicate/f_max revision (ADR-026 + R-2026-06-B §15) — **not** part of this §4
compliance amendment, and required before a graded EH Δ can be measured.

_Refs: ADR-026 §Decision 4 + Open-questions (Rung-4/4b notes); ADR-005;
R-2026-06-B §15 Rung 4/4b; `spikes/results/cocarry/rung4`,
`spikes/results/cocarry/rung4b`._
