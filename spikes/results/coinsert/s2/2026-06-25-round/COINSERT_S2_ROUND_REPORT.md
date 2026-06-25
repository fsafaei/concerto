# Co-insert S2 — round-geometry Gate-D sweep → HARD_STOP (2026-06-25)

**Slice.** S2, the founder (i)/(ii)/(iii) decision tree after the BOUND_HIT. Phase-2, non-gating (ADR-026 §Decision 1-4).
Pre-S3-freeze; STEP-0 tag cut; the locked `coinsert_s0.yaml` is untouched (the square→round cross-section is **not** in
the frozen prereg set, so it is a documented geometry change, recorded here). **DO NOT MERGE.**

**Verdict.** **HARD_STOP** — the bankable founder-defined stop: neither the SQUARE geometry at the loosest clearance
nor the canonical ROUND geometry at **any** clearance in the frozen set yields an operating point that is BOTH seatable
by a competent matched pair (≥~0.9) AND coupling-valid (a rigid hold fails). **Not** the (disproven) sim wall, **not** a
control-competence failure — a **task-parameterisation** finding. Per the bound: do NOT reduce the seat depth, loosen
clearance beyond the frozen set, or use active mating — each is weaken-to-pass / construct-invalid.

**Evidence.** All numbers from the committed artifact `coinsert_s2_round_sweep.json` (this directory). Reproduce
(GPU + oracle): `uv run python scripts/repro/_coinsert_s2_round_sweep.py` (seeds {0..4}, deterministic).

## The decision tree, executed

- **STEP (i) — square @ 1.0 mm:** walls (matched seat 0, ~30.3 mm; established by the committed
  `2026-06-25-fixedlink-tuning` artifact) ⇒ STEP (ii).
- **STEP (ii) — ONE round-geometry build + clearance sweep.** The peg/socket cross-section is swapped square → ROUND: a
  **cylinder peg** (SAPIEN primitive, axis rotated to the approach +z) + a **12-facet convex-box N-gon bore** (inscribed
  radius = peg_radius + clearance/2; the same no-mesh convex-decomposition technique as the square walls — ADR-001
  respected). Round is the canonical, sim2real-validated insertion geometry: the cylinder self-centres and **frees the
  yaw DOF**. Everything else frozen (depth 40 mm, chamfer, clearance set, fixed-link attach, force cut, the cooperative
  reference holder + base inserter with its pre-registered insertion envelope unchanged). The S1 fidelity-probe rig keeps
  its square geometry (its STAY verdict stands; round contact is gentler than square corners).

### Round clearance sweep (loosest-first, 5 seeds)

| clearance | matched seat | matched depth (median) | rigid-hold seat | single-inserter |
|---|---|---|---|---|
| 1.0 mm | **0/5** | ~29.7 mm | 0/5 | 0 |
| 0.5 mm | **0/5** | ~30.7 mm | 0/5 | 0 |
| 0.2 mm | **0/5** | ~4.6 mm | 0/5 | 0 |

Round walls at the **same ~30 mm** as square (and jams almost immediately at the tightest 0.2 mm). The single-inserter
positive control is 0 at every clearance (the free receptacle is preserved). No clearance seats, so the coupling check is
moot (recorded for completeness).

## Why — and the levers ruled out

The ~30 mm wall is a **geometric tilt-wedge**: a 40 mm seat at 0.5 mm/side clearance requires the relative peg–bore tilt
held **< 0.7°** through the full insertion, but the insertion contact itself cocks the peg to ~0.9–2.8°, which
two-point-wedges at ~30 mm. The **seatable** region (tilt < 0.7°) and the **achievable-control** region (~0.9–2.8° under
contact) do not overlap. This is robust to every lever:

- **architecture** — create_drive AND fixed-link both wall at ~30 mm;
- **friction** — depth pinned ~30 mm across μ 0.5 → 0.05;
- **cross-section** — square AND round both wall at ~30 mm (this sweep);
- **holder rotational compliance** — holder ori-hold gain 2.5 → 1.0 walls ~30 mm; ≤ 0.5 the socket *flops* (align
  55–71°, worse). No setting clears the wedge. (The cooperative holder's per-controller `ori_hold_gain` was added this
  slice; the base inserter keeps 2.5.)

## Bankable finding (founder call)

Contact-rich cooperative free-receptacle insertion at **40 mm depth × 0.5 mm-per-side clearance** has **no operating
point in this sim**, even with the canonical round geometry, where a competent matched pair seats AND holder
heterogeneity could be load-bearing. The remaining unfrozen task params (seat depth, clearance) cannot be adjusted to
clear it without weaken-to-pass. Founder decision: adjust the task parameterisation (a scientific re-scoping) or honest
close of the co-insert bet — banking the wall-disproof, the competent fixed-link rig, and this geometric finding.
