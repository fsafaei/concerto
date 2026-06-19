# Co-carry Rung 4 — Embodiment-Heterogeneity (EH) measurement (the capstone)

**The last rung, and the close of the ladder.** Does a different partner
*body* — a genuinely different robot arm — degrade cooperation against the
frozen, validated Rung-2 Panda incumbent, *separated from control-style* via
the Rung-3 PH floor?

- **Outcome:** a **pre-registered feasibility finding**. The only viable
  embodiment teammate (an **xArm6 + Robotiq 2F-85**; the `ur_e`/`ur_10e`
  backup is gripper-less and structurally unusable) **cannot be
  capability-matched** to the co-carry task with a hand-written controller:
  paired with the cooperative Panda ego it scored **0/12** (fails *level*
  12/12 and *unstressed* 12/12; stress p90 **518 N** vs the 130 N ceiling and
  the matched pair's ~75 N). Under the pre-committed rule (C_min = 0.75, never
  weakened) it is **EXCLUDED** and the EH measurement **does not run**.
- **Why — the load-bearing finding:** the rigid dual-hold weld (20 000 N/m)
  only tolerates embodiment-**symmetric** pairs. The matched Panda pair holds
  the bar at ~75 N internal stress *by symmetry* (two identical arms move in
  lockstep); **any** different-bodied partner breaks that symmetry and the two
  arms **fight through the rigid bar at 3–7× that force**, far past f_max, and
  tilt the bar past 15°. This is a property of the **rig**, surfaced honestly.
- **Directional-bias caveat (binding):** a failed gate may **not** be read as
  "embodiment does not couple to cooperation." The opposite holds here — the
  rig couples embodiment *so strongly* that no capability-matched
  cross-embodiment teammate exists. The measurement is precluded, not null.

Governance: incumbent **frozen**, never retrained; partner black-box (I3);
predicate / f_max (130.6 N) / 0.10 m radius **unchanged**; no schema bump (I9);
no immutable-archive edits (I8). **Do not merge — founder review.**

---

## 1. What was built (and works end-to-end)

The full embodiment-shift rig is implemented and verified — the feasibility
finding is a *capability* result, not a missing rig:

| Component | Status |
|---|---|
| `chamber.agents.xarm6_jacobian` — xArm6 6-DOF FK/Jacobian (pytorch_kinematics) | ✓ Tier-1 tested |
| `CoCarryEnv` partner-embodiment-configurable (`cocarry_xarm6_partner` condition) | ✓ builds; Panda matched path byte-identical |
| Partner-observation adapter (xArm6 12-DOF → Panda 18-D ego-state slot) | ✓ frozen actor **and critic load** on the xArm6 env (ego state 46-D) |
| `chamber.partners.cocarry_xarm6` — compliant + cooperative-leveling controller | ✓ Tier-1 tested |
| Mixed-embodiment rig (Panda ego + xArm6 on one welded bar) | ✓ **stable** (finite, bounded) |

The frozen incumbent loads and runs against the xArm6 (verified on a throwaway
seed: it executed end-to-end). The blocker is upstream — the xArm6 cannot be
*capability-matched*, so it never enters the measurement.

## 2. Gate 1 — mixed-embodiment stability (PASS)

Panda ego (cooperative) + xArm6 partner rigidly holding one bar, driven to the
goal, 3 seeds (`cocarry_rung4_calibration_roster.json`):

- **stable = True** — all telemetry finite, no solver blow-up; max constraint
  force 506 N (bounded), max tilt 30°.

The closed two-different-arm chain is physically stable. The issue downstream
is **capability** (the arms fight), not numerical instability.

## 3. Gate 2 — capability calibration (xArm6 EXCLUDED)

xArm6 teammate + the cooperative Panda ego (matched `cocarry_impedance`, **not**
the frozen incumbent), 12 seeds (60000–60011), C_min = 0.75:

| | xArm6 + cooperative ego | matched Panda pair (ref) |
|---|---|---|
| joint success | **0 / 12** | 12 / 12 |
| fails *placed* | 10 / 12 | 0 |
| fails *level* | **12 / 12** | 0 |
| fails *unstressed* | **12 / 12** | 0 |
| wrist stress p90 | **518 N** | ~75 N |
| f_max ceiling | 130 N | 130 N |

**EXCLUDED** (0 < C_min). The xArm6 reaches the goal on some seeds but the bar
tilts past 15° and the internal stress runs at ~7× the matched pair and ~4× the
130 N ceiling.

**Effort (no weakening of C_min):** a wide hand-written-controller sweep was
explored — compliant impedance (matched gains), an admittance/force-follower,
cooperative bar-leveling (track the other end's height), and gain / damping /
step / base-pose variations. The base was moved closer (0.35 m vs the Panda's
0.50 m) to keep the goal within the xArm6's reach (which *fixed* transport:
`placed` became reachable). **No configuration cleared `placed` AND `level` AND
`unstressed` jointly** — the internal fight (stress) is irreducible with a
position-controlled arm on the stiff rigid weld.

**Mechanism (validated, not assumed):** the fight force was read at three
xArm6 wrist links during a cooperative rollout — `link6` 366 N ≈
`robotiq_arg2f_base_link` 366 N, `eef` (closest to the weld) 228 N — all
≥ 3× the matched pair's 75 N and above f_max. So the stress is physical
cross-embodiment fighting, not a stress-link artifact.

## 4. The `ur_e` backup (structurally unusable)

The spec's `ur_e` does not exist in `mani-skill==3.0.1`; the only UR
(`ur_10e`) ships with **no gripper / TCP** (`urdf_path=None`,
`gripper_joint_names=None`), so the dual-hold bar weld has no grip frame to
attach to. It cannot be a co-carry teammate — a documented structural
exclusion, not a tuning gap. `xarm6_robotiq` is the only ManiSkill 3.0.1 robot
with a different body **and** a weldable Robotiq gripper.

## 5. Verdict (pre-committed) + the EH-vs-control-style reading

**Feasibility finding** (the pre-committed `feasibility_rule`): no embodiment
teammate clears the capability gate, so the reference/shifted measurement does
not run — itself the EH result for this rig.

**This is not "embodiment is inert."** The EH-vs-control-style separation
(against the Rung-3 PH floor) reads as follows:

- Rung 3 (PH, same Panda body): compliant / timing control-style shifts cost
  **Δ ≈ 0**; a *stiffer* control style cost Δ = 0.25. Control style mostly does
  not break cooperation on this rig.
- Rung 4 (EH, different body): a different body cannot even be brought to the
  cooperation ceiling with the cooperative reference — the rigid weld makes
  cross-embodiment pairs fight at 3–7× the cooperative stress.

So the honest, defensible headline is: **on this rigid-weld co-carry task,
*embodiment* is the heterogeneity that bites — far harder than control
style.** A different control style is mostly absorbed (Rung 3); a different
body cannot be made cooperative at all with a hand-written controller, because
the stiff rigid attach only tolerates embodiment symmetry. Whether that is
"embodiment breaks cooperation" or "the rig over-couples embodiment" is the
same physics from two sides — and it is **decision-relevant**: to measure EH as
a graded Δ (rather than a feasibility wall), the task needs a **more compliant
attach** (a real grasp or a compliant coupling) that admits embodiment-
heterogeneous teammates. That is the concrete next step the ladder surfaces.

## 6. The ladder, closed

| Rung | Axis | Result |
|---|---|---|
| 0–1 | rig + coupling | matched 100%, single-arm ≈ 0 (coupling real) |
| 2 | frozen incumbent | held-out-validated, frozen (step 100k, 24/24) |
| 3 | **policy** heterogeneity | pooled **null** (Δ=0.083<0.20); one stiff teammate Δ=0.25; compliant/timing Δ=0 |
| 4 | **embodiment** heterogeneity | **feasibility finding** — no capability-matched cross-embodiment teammate (rigid weld tolerates only symmetric pairs) |

The complete coupling-valid heterogeneity read: **control-style differences
are largely absorbed; embodiment differences are not** — on this rig the latter
cannot even be made cooperative, the sharpest possible statement that
embodiment, not control style, is the heterogeneity that matters here. Both
rungs are pre-registered, capability-controlled, and reported with their
caveats. That is the Phase-1-spirit question answered with evidence.

## 7. Reproduce

```bash
# Stability gate + capability calibration (xArm6 EXCLUDED, RC=2)
bash scripts/repro/cocarry_rung4_eh_calibrate.sh
# Pre-registration committed + tagged: prereg-cocarry-rung4-eh-2026-06-19
# Measurement does NOT run (gate failed). If a capability-matched embodiment
# teammate becomes available, the harness path is:
#   chamber.benchmarks.cocarry_ph.evaluate_incumbent_vs_partner(
#     condition_id="cocarry_xarm6_partner", partner_uid="xarm6_robotiq", ...)
#   over the reserved measurement seeds 50000-50011.
```

| Artifact | Path |
|---|---|
| Pre-registration (tagged) | `spikes/results/cocarry/rung4/cocarry_rung4_eh_prereg.json` |
| Calibration roster (stability + exclusion + ur backup + caveat) | `spikes/results/cocarry/rung4/cocarry_rung4_calibration_roster.json` |
| Rung-3 PH floor (compared against) | `spikes/results/cocarry/rung3/cocarry_rung3_ph_measurement.json` |
| Frozen incumbent | `spikes/results/cocarry/rung2/cocarry_rung2_freeze_manifest.json` (step 100000, SHA `6290e2ae…`) |

- **Seeds:** calibration 60000–60011; measurement 50000–50011 (reserved,
  unused). Disjoint from all prior sets (Rung-2 S/V 10000-/20000-, Rung-3
  30000-/40000-).
- Determinism: env + bootstrap RNG via `derive_substream` (P6); the frozen
  incumbent acts in policy-mode (no RNG).

_ADR-026 §Decision 4; ADR-026 §Validation criteria; ADR-005 §Decision;
ADR-009 §Decision; R-2026-06-B §15 Rung 4._
