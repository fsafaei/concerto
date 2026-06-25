# Co-insert S2 — bounded competence-tuning on the fixed-link rig → BOUND_HIT (2026-06-25)

**Slice.** Founder-authorised bounded competence-tuning to reach the matched insertion seat on the validated
fixed-link rig (the `create_drive` "wall" disproven). **Firm pre-committed bound: 8 tuning configurations.**
Phase-2, non-gating (ADR-026 §Decision 1-4). Pre-S3-freeze; **DO NOT MERGE.**

**Verdict.** **BOUND_HIT** — the 8-config bound is reached without matched ≥ ~0.9 at 1.0 mm. The matched pair
reliably reaches **~30 mm with good alignment** (median depth 30.3 mm, align 0.9°) but cannot clear the last
~8 mm to the 38 mm seat: a **friction wedge at ~30 mm**, robust across the `create_drive` AND fixed-link
architectures and all 8 control configs. This is a **task-parameterisation** finding, **not** the (disproven)
sim/constraint-fidelity wall and **not** a control-competence failure → a founder call (adjust the still-unfrozen
task params or honest stop).

**Evidence.** `coinsert_s2_fixedlink_tuning_probe.json` (this dir). Reproduce (GPU+oracle):
`uv run python scripts/repro/_coinsert_s2_fixedlink_tuning_probe.py`. Seeds {0..4}; deterministic.

## What the tuning fixed (the rig works)

The fixed-link architecture is confirmed competent: the socket braces rock-steady (sockDrift 0 mm, opening-up 0°,
**no over-constraint**), alignment is excellent when the press is gentle (~0.3°), and the two-robot-necessity
positive control holds — **single-inserter success rate 0.0** (the matched holder-pose change to the long bracket
did not world-anchor the socket). Two control fixes were decisive:

- **Orientation-hold coupling** (config 1): the 6-DOF orientation hold at gain 6 coupled into lateral motion → the
  peg drifted ~10 mm and never centred. Reducing `_ORI_HOLD_GAIN` to 2.5 fixed it (pegLat ~1 mm).
- **Arm-arm interference** (configs 3-4): the reaching-under below-hold holder pose put the holder arm in the ego's
  descent corridor — with the holder active, it froze the ego (with it passive, the ego descended 96 mm). Swinging
  the holder arm clear via a **long-bracket fixed-link** pose (which, unlike `create_drive`, does NOT over-constrain)
  unblocked the ego and the socket still braces.

## The wall (the last ~8 mm)

With the arm clear and the socket braced, the peg inserts to ~30 mm with align ~0.3°, then friction-locks. The press
authority needed to overcome the friction at 30 mm either **cocks the peg** (align 0.3° → 2.4-4.7°, worsening the
wedge — the wedge needs align < ~0.75°) or, once the peg friction-locks, **drags the holder** (the joint impedance
cannot resist the locked-pair reaction; stiffening it destabilises — config 7 flung the socket). This press-authority-
vs-alignment / drag tension is the same ~30 mm wall seen under `create_drive`, so it is intrinsic to the task
parameterisation, not the attachment mechanism.

## The 8-config tuning log

| # | change | result (max depth / outcome) |
|---|---|---|
| 1 | baseline (ori_gain 6) | seat 0; peg drifts laterally (ori-coupling), never centres |
| 2 | ori_gain → 2.5 | seat 0; lateral fixed, peg hovers at standoff |
| 3 | approach standoff 15→4 mm | seat 0; peg stuck ~13 mm above (below-hold arm-arm interference) |
| 4 | holder → long-bracket fixed-link | seat 0; **inserts to 23 mm**, socket braces (sockDrift 0, align 0.3°); unjam walks it out |
| 5 | descend 0.002→0.005, stall 3→8 | seat 0; **reaches ~30 mm, holds** (the committed best regime) |
| 6 | descend→0.003, retract 0.009×5→0.004×3 | seat 0; friction-lock drags holder (socket sank 210 mm + 113 mm lateral) |
| 7 | stiff holder (kp 2→8, step→0.04) + firm press | seat 0; destabilises (socket flung, depth 86 mm, align 12.7°) |
| 8 | ori_gain→4 + moderate unjam | seat 0; reaches ~30 mm then drags 167 mm |

## Founder decision triggered

Per the bounded-escalation rule: the bound is hit, so this is informative — the task as parameterised (0.5 mm-per-side
clearance + the declared peg-socket friction + the 40 mm seat depth in the quasi-static control regime) appears
mis-conditioned even for a competent controller on a validated rig. **Founder call:** adjust the still-unfrozen task
params (the declared friction and/or the seat depth) and re-attempt, or honest stop. Do **not** tune unbounded.
The committed state is the best stable regime (config 5: ~30 mm). The locked `coinsert_s0.yaml` is untouched.
