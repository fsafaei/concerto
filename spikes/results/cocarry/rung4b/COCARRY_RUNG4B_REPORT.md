# Co-carry Rung 4b — compliant-coupling ladder (Stage-1 gate result)

**What this slice set out to do.** Rung 4 hit a feasibility wall: the **rigid**
dual-hold weld over-couples embodiment, so a different-bodied xArm6 partner
fought the bar at ~518 N (vs the matched pair's ~75 N) and could not be
capability-matched. Rung 4b implements a **passive compliant coupling** and
re-establishes the ladder on it (rig → incumbent → PH floor → EH), so the
embodiment result is measured on a coupling that admits the embodiment.

**Headline (Stage-1 gate): `STOP_PROXY_ARTIFACT` — a sharper finding than Rung 4.**
The compliant coupling **works** — it removes the over-coupling fight: the
xArm6 now **transports and levels the bar** (placed + level pass on all 8/8
seeds, at every swept setting; they failed on all seeds under the rigid weld).
But the embodiment measurement still cannot run, for a **new and different**
reason: the success predicate's stress conjunct uses a **wrist-incoming-force
proxy that is embodiment-dependent**, and the rules forbid changing f_max /
the predicate. So the gate STOPs at the honest fallback — but it has *moved the
blocker* from the task physics (rigid over-coupling) to the **measurement
instrument** (a Panda-calibrated proxy applied to a heavier-wristed body).

This is decision-relevant and publishable: it isolates exactly what a correct
embodiment-heterogeneity measurement needs next (an embodiment-invariant
stress proxy or a per-body f_max calibration), and it confirms the
compliant-coupling design hypothesis.

Governance: the **rigid ladder (Rungs 0–4 on `main`) is untouched** (I8); this
writes only under `rung4b/`. Predicate / f_max (130.6 N) / 0.10 m radius / C_min
(0.75) unchanged; no schema bump (I9); compliant coupling is the only task
change; passive (no grasp, no learning). **Do not merge — founder review.**

---

## 1. What was built (Stage 1)

The compliant coupling is a passive, frozen-able env parameter
(`chamber.envs.cocarry.cocarry_coupling`), threaded through
`make_cocarry_env` / `make_cocarry_training_env` / the `cocarry_ph` eval
functions. **Default `None` ⇒ the rigid hard-locked weld, byte-identical to the
committed ladder** (158 existing cocarry Tier-1 tests still pass; rigid
artifacts immutable).

The key mechanism: the rigid weld **hard-locks** each drive axis
(`set_limit(0,0)`), which makes the drive stiffness inert — that lock *is* the
over-coupling. The compliant variant **frees** the axis and uses the drive as a
passive spring toward its zero target:

- **Variant A (linear):** a lower stiffness `K_c`, damping = `0.1·K_c`
  (the rig's overdamped ratio), force unbounded.
- **Variant B (force-saturated / bilinear knee):** a stiff drive + a finite
  `force_limit` near f_max — near-rigid for the small-force matched pair,
  force-saturated for a large mismatch.

## 2. Stage-1 gate: the C1–C4 sweep

Pre-registered (`cocarry_rung4b_coupling_prereg.json`, tag
`prereg-cocarry-rung4b-coupling-2026-06-20`, blob-SHA verified). 8 seeds
(70010–70017). Full data: `cocarry_rung4b_coupling_sweep.json`.

| Setting | C1 matched-clean | C2 single≈0 | C3 binds | C4 xArm6 admitted | matched (cen p90) | xArm6 |
|---|---|---|---|---|---|---|
| A K=8000 | ✅ (100%, 0.049) | ✅ | ✅ | ❌ | 100% | 0% — **Pfail 0, Lfail 0**, Ufail 8/8 (str p90 478) |
| A K=4000 | ❌ (0.099, edge) | ✅ | ✅ | ❌ | 100% | 0% — Pfail 0, Lfail 0, Ufail 8/8 (312) |
| A K=2000 | ❌ (0.098, edge) | ✅ | ✅ | ❌ | 100% | 0% — Pfail 0, Lfail 0, Ufail 8/8 (252) |
| B K=8000, FL=110 | ✅ (100%, 0.049) | ✅ | ✅ | ❌ | 100% | 0% — Pfail 0, Lfail 0, Ufail 8/8 (478) |

**Reading:**
- **C1–C3 are satisfiable** at the stiffest setting (K=8000): matched pair 100%
  with comfortable deflection (centroid p90 0.049 ≪ 0.10 m), single-arm ≈0
  (coupling-validity preserved), coupling binds. Softening below K≈8000 pushes
  the matched deflection to the 0.10 m edge (C1 margin lost) — the Variant-A
  squeeze.
- **C4 fails at every setting — but only on the `unstressed` conjunct.** The
  xArm6 **places and levels on all 8 seeds** (Pfail 0, Lfail 0). Under the
  rigid weld it failed placed+level on every seed; the compliant coupling has
  **resolved the cooperation problem**. The sole remaining failure is stress.

## 3. Why C4 fails — the embodiment-biased stress proxy (the diagnostic)

The success predicate's `is_unstressed` reads the **wrist incoming joint force**
(`get_link_incoming_joint_forces` at the holding link), with f_max = 130.6 N
calibrated on **Panda–Panda** successes (p99 ≈ 104 N). The zero-action
static-hold baseline (both arms hold the bar, **no cooperative motion at all**):

| | wrist stress p50 | wrist stress max |
|---|---|---|
| Panda partner (K=8000) | 40 N | 82 N |
| **xArm6 partner (K=8000)** | **138 N** | **514 N** |

The xArm6 (Robotiq 2F-85) wrist reads **~3–6× the Panda for the identical rest
hold — already over f_max before any cooperative fight.** Confirmed
controller-independent: *gentler* xArm6 controllers raised the reading, not
lowered it (and lost placement), because the proxy is dominated by the heavier
wrist/gripper assembly's joint reaction, not by control aggressiveness.

So the `unstressed` conjunct rejects the xArm6 on a **body artifact**, not a
cooperative fight. The Panda-calibrated f_max is not embodiment-comparable.

## 4. Verdict + the honest fallback (pre-committed)

Per the pre-registered gate rule: no setting satisfies C1–C4, and the cause is
isolated to the proxy (C1–C3 pass; xArm6 places+levels; static wrist ≥ f_max),
so the verdict is **`STOP_PROXY_ARTIFACT`**. The rules forbid moving
f_max / the predicate / the radius and weakening C_min, so the EH measurement
(Stages 2–4) **does not run** on the unchanged predicate.

**This STOP is progress, not a dead end.** It cleanly separates two blockers
the single Rung-4 result had conflated:
1. **Task physics (rigid over-coupling)** — *resolved* here by the compliant
   coupling (xArm6 transports + levels).
2. **Measurement instrument (embodiment-biased stress proxy)** — *newly
   isolated*, now the sole blocker, with a precise fix.

**The fix (out of scope here; needs an ADR + predicate change):** an
embodiment-invariant stress measure — e.g. the **weld constraint force itself**
(what the cooperation actually transmits, body-independent) instead of the
arm's wrist incoming joint force, or a **per-embodiment f_max calibration**
(each body's static/cooperative baseline). Either is a deliberate predicate/ADR
revision that must not be slipped in under "no bar-moving"; it is the
recommended next slice.

## 5. The ladder, updated

| Rung | Coupling | Result |
|---|---|---|
| 0–3 | rigid | rig valid; incumbent frozen; PH pooled null (one stiff teammate Δ=0.25) |
| 4 | rigid | EH feasibility wall — rigid weld over-couples; xArm6 fights (518 N), fails placed+level+unstressed |
| **4b** | **compliant** | **over-coupling resolved (xArm6 places+levels); EH still blocked, now by an embodiment-biased stress proxy vs Panda-calibrated f_max** |

**Honest framing.** The rigid ladder stands as the rigid-coupling finding
(incl. the over-coupling wall). The compliant variant shows that wall was a
*coupling-stiffness* artifact — softening it lets a different body cooperate
(transport + level). The graded embodiment Δ the M10 narrative wants is
**one instrument-fix away**: an embodiment-invariant stress proxy. Until then,
the disciplined result is this gate STOP, not a Δ measured under a predicate
that rejects the body on contact.

## 6. Reproduce

```bash
# Pre-registration committed + tagged BEFORE the sweep:
#   prereg-cocarry-rung4b-coupling-2026-06-20
bash scripts/repro/cocarry_rung4b_coupling_sweep.sh   # C1-C4 + static baseline + verdict (RC=2)
```

| Artifact | Path |
|---|---|
| Pre-registration (tagged) | `spikes/results/cocarry/rung4b/cocarry_rung4b_coupling_prereg.json` |
| Sweep + static-baseline + verdict | `spikes/results/cocarry/rung4b/cocarry_rung4b_coupling_sweep.json` |
| Rigid-coupling baselines (read-only) | `spikes/results/cocarry/rung{2,3,4}/` |

- Seeds: 70010–70017 (disjoint from all prior sets). f_max 130.6 N, radius
  0.10 m, C_min 0.75 — all unchanged. Frozen incumbent SHA `6290e2ae…`
  (not reached — Stage 2 does not run).
- Determinism: env RNG via `derive_substream` (P6); coupling resolved by
  `cocarry_coupling`; rigid default byte-identical to the committed ladder.

_ADR-026 §Decision 4 / §Validation criteria; ADR-005 §Decision; ADR-009
§Decision; R-2026-06-B §15 Rung 4b._
