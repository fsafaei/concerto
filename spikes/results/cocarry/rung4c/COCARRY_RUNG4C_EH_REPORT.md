# Co-carry Rung 4c — embodiment-invariant stress instrument → the embodiment result

> **⚠ CORRECTION — RETRACTED AS A POSE ARTIFACT (2026-06-20, Rung-4d Stage-A1).**
> The headline below — *"a different body (xArm6) genuinely over-loads the
> cooperative coupling ~5× f_max, separated from control style"* — **does not
> survive a fair carry-pose optimisation and is retracted.** A joint-limit-clamped
> kinematic search found a *feasible* xArm6 carry configuration with vertical
> endpoint compliance **2.34 (0.49× the Panda's 4.74 — i.e. *stiffer*)**, and at
> that pose the xArm6 holds and carries the bar **in-band** on coupling (static
> max 159 N, active p90 264 N — both < f_max 366; ≈ the matched 267 baseline),
> versus 1842 N at the unoptimised default pose. The over-load was an artifact of
> the **default pose**, not embodiment. Per the pre-committed Stage-A1 stop rule
> the heavy incumbent re-freeze was **not** run. **The EH axis is NOT established
> by this report.** A residual stress↔tilt pose tradeoff (the stiffness-optimal
> pose then over-tilts, p90 26.3°) is a *candidate* difficulty, not a finding —
> it needs a task-fair pose+controller search. Full correction +
> committed-from-code reproduction:
> [`../rung4d/COCARRY_RUNG4D_EH_CORRECTION.md`](../rung4d/COCARRY_RUNG4D_EH_CORRECTION.md)
> (`scripts/repro/cocarry_rung4d_pose_falsification.sh`, seeds 80100–80105).
> Everything below is preserved as the superseded record.

> **Audit-hardening addendum (2026-06-20, PR-#252 follow-up).** The headline
> below is now reproducible from **committed code** and the construct crux is
> closed with **committed data** (replacing the original `/tmp` script + prose):
>
> - **Headline floor reproduced from committed code** —
>   `scripts/repro/cocarry_rung4c_sameego_floor.sh` regenerates
>   `cocarry_rung4c_sameego_floor.json` exactly: matched 267 / stiff 289 /
>   admittance 317 / slew 334 / **xArm6 1867 N** (control-style floor 267–334 N;
>   xArm6 ~5.6× the ceiling).
> - **1867-vs-1773 N discrepancy reconciled** — same condition, different
>   pre-registered seed sets (floor 80100– → 1867; EH 80200– → 1773); both now
>   reproduced from committed code; the ~5% gap is n=12 sampling, the over-load
>   is seed-robust.
> - **Controlled invariance — committed proof** (`cocarry_rung4c_robustness.json`):
>   with the bar teleported to the *same* world pose, Panda- vs xArm6-held
>   coupling agrees within **0.61 N** across a 0→0.05 m deflection sweep → the
>   instrument is embodiment-invariant *given the same state*; the high xArm6
>   reading is a real state difference, **not** a measurement bias.
> - **Static-hold committed** — zero-action, gravity only: Panda p50 163 / max
>   247 N vs xArm6 p50 1041 / max 2632 N (**6.4×**, previously prose-only).
> - **Pose/base robustness — committed** — the xArm6 coupling is over f_max at
>   **every** swept base (min 818–1910 N vs f_max 366 N at base x = 0.30/0.35/0.40);
>   it is **not** a single geometrically-bad spawn (the AS-axis failure mode).
>   Magnitude is pose-sensitive (872–2255 N) but over-loaded everywhere.
> - **Mechanism (partial)** — the xArm6 end-effector is **1.46×** more
>   vertically compliant than the Panda at the carry pose (Jacobian (JJᵀ)_zz
>   6.91 vs 4.74); a contributing factor, not the sole cause of the 6.4× static
>   sag (joint PD/force-limit + link inertia under load also contribute).
> - **Precise framing:** the cooperative-ego xArm6 fails **unstressed 12/12;
>   placed + level pass; static fails on 2/12** — not "only unstressed". The
>   claim is **"this xArm6 controller/pose on this task over-loads the
>   coupling"**, not a universal embodiment law.
>
> All Stage-1/2 hardening gates passed (reproduced + invariant + pose-robust),
> so the cooperative-ego headline is **audit-ready**; no committed check
> contradicted it. The incumbent-specific Δ (Stage 3) still needs a
> compliant-coupling re-freeze. Generators: `cocarry_rung4c_sameego_floor.sh`,
> `_cocarry_rung4c_robustness.py`. Seeds: floor 80100–80111, EH 80200–80211.

**The convergence slice, and the thesis-bearing payoff.** Rung-4b solved the
*physics* blocker (a compliant coupling lets a different body transport+level
the bar) but isolated an *instrument* one: the "unstressed" conjunct read the
**arm's wrist joint force**, which is embodiment-biased. Rung-4c replaces it
with an **embodiment-invariant** measure — the **bar's internal coupling
force** — re-derives f_max for it, and measures the embodiment effect.

**Headline: embodiment heterogeneity genuinely degrades cooperation — and it is
NOT a measurement artifact.** Under the embodiment-invariant coupling
instrument, against the *same* cooperative ego, a different-bodied (xArm6)
partner over-loads the cooperative coupling to **~1867 N — ~7× the matched
Panda pair, ~5× the re-derived f_max (366 N) — and fails the unstressed
conjunct on 12/12 seeds**, while the matched pair and every same-body
control-style (policy-shift) teammate stay **in-band (267–334 N)**. The
embodiment effect is **~6× beyond the entire control-style spread** — a clean,
qualifying embodiment drop, cleanly separated from control-style.

This **refutes the Rung-4b "it's just the biased ruler" hope**: fixing the
instrument (wrist → coupling force) *reduces* the bias (xArm6/Panda ratio
17×→7.8×) but does not remove the effect — the different body really does
over-tension the coupling, because its carry-pose endpoint stiffness is poor
(it sags under the bar load). Embodiment, not control style, is the
heterogeneity that bites — and now that is measured under a *correct*,
embodiment-invariant instrument.

**Honest scope caveat (→ the next slice).** The *frozen incumbent* was trained
on the **rigid** coupling and does **not transfer** to the compliant coupling
(the incumbent+matched reference fails the *level* conjunct, 0%), so the
pre-registered *incumbent-based* EH Δ is **inconclusive this run** (its
reference did not reconfirm; the script's auto "null" is therefore void by the
pre-registration's own STOP rule). The embodiment finding above is established
against the **cooperative reference ego** (a fixed, competent ego that *does*
transfer) — a valid fixed-ego EH contrast (only the partner body changes). The
incumbent-specific Δ requires **re-freezing an incumbent on the compliant
coupling** (the heaviest step), which is the next slice.

Governance: predicate-change governed via the ADR-026 open-question resolution
+ co-carry spec §4/§8 amendment; pre-registered + tagged
(`prereg-cocarry-rung4c-eh-2026-06-20`) before f_max re-derivation/measurement;
f_max re-derived only from the matched-Panda pair, then held (never inflated to
admit the xArm6); C_min unchanged; rigid + Rung-4b artifacts immutable (I8); no
schema bump (I9). **Do not merge — founder review.**

---

## 1. The embodiment-invariant instrument (Stage 1)

The old proxy read each arm's **wrist incoming joint force**, which folds in the
arm's own link weight/inertia (mass that never reaches the bar) → a
body-dependent offset (xArm6 reads 138/514 N just holding still). The new
measure is the **bar coupling spring force**: `F = K_c · ‖grip_frame_world −
bar_end_world‖`, the actual force the compliant dual-hold spring transmits — a
function of the bar/coupling **geometry only**, identical regardless of which
arm holds it (`chamber.envs.cocarry`: `_coupling_stress_now`; selectable via
`stress_measure="coupling"`; default `"wrist"` keeps the rigid ladder
byte-identical). SAPIEN exposes no free-drive constraint force, so it is
computed from the anchor geometry recorded at weld time.

**Invariance investigation** (`cocarry_rung4c_invariance.json`, active
cooperative state):

| measure | matched p90 | xArm6 p90 | xArm6/Panda ratio |
|---|---|---|---|
| wrist (old) | 27 | 461 | 17.1× |
| **coupling (new)** | 261 | 2036 | **7.8×** |

The coupling measure removes the arm-self-weight confound (17→7.8×). It does
**not** equalise the values — and that is correct, because the xArm6 genuinely
loads the coupling more. Confirmed not an actuation/tuning artifact: the xArm6
over-loads ~6.4× even at a **zero-action static hold**, and a joint
force-limit sweep (100–1000 N; its joint stiffness is already 10× the Panda's)
does not converge to Panda levels. The residual is a **real embodiment load**
(poor carry-pose endpoint stiffness → sag → coupling over-tension).

## 2. Re-derived f_max + task re-validation (Stage 2)

f_max re-derived from the matched-Panda-pair coupling distribution
(`stress_measure="coupling"`, 12 seeds): matched-success coupling p99 = 292 N →
**f_max = 1.25 × 292 = 366 N**, then held. The matched **controller** pair
scores **100%** at this f_max (the task + f_max are sound); single-arm ≈ 0
(coupling-validity preserved).

**Incumbent transfer check — FAILS.** The frozen (rigid-trained) incumbent +
matched partner on the compliant coupling fails the *level* conjunct on 12/12
seeds (0% success) — it does not transfer from the rigid to the compliant
coupling. So the incumbent-based EH measurement is not valid this run; the
embodiment effect is measured against the cooperative reference ego instead
(§3), and a compliant-coupling re-freeze is the next slice.

## 3. The embodiment measurement + EH-vs-control-style separation (Stage 3)

Against the **same cooperative reference ego** (the matched impedance ego,
which transfers to the compliant coupling), under the invariant coupling
instrument + f_max = 366 N, 12 seeds (`cocarry_rung4c_sameego_floor.json`):

| Partner (same ego; body/policy shift) | kind | coupling p90 (N) | success @ f_max |
|---|---|---|---|
| matched Panda `cocarry_impedance` | baseline | 267 | 100% |
| `cocarry_stiff_impedance` | control-style (PH) | 289 | 100% |
| `cocarry_admittance` | control-style (PH) | 317 | 100% |
| `cocarry_slew_impedance` | control-style (PH) | 334 | 0%* |
| **`cocarry_xarm6_impedance`** | **embodiment (EH)** | **1867** | **0%** (unstressed 12/12; placed+level pass; static 2/12) |

\* slew's coupling load is **in-band (334 N < f_max)**; its 0% is a *non-stress*
conjunct (placement/timing under the compliant coupling — a control-style
artifact, not coupling over-stress). On the **stress channel** that the EH
blocker lives on, all four Panda-bodied teammates are 267–334 N.

**Separation:** the **control-style floor on the coupling-stress channel is
267–334 N** (all within ~1.25× the matched baseline, ≤ / ~f_max). The
**embodiment shift is 1867 N — ~5.6× the control-style ceiling, ~7× the matched
baseline, ~5× f_max.** The embodiment effect is unambiguously beyond the
control-style spread → attributable to **embodiment**, not control style.

**Mechanism (per-conjunct):** the xArm6 fails **`unstressed` 12/12; placed +
level pass on all 12; `static` fails on 2/12** (the over-tensioned coupling
also disturbs settling on 2 seeds). Under the compliant coupling it transports
and levels, but this xArm6 controller/pose sags and over-tensions the coupling
far past any matched/Panda level — the dominant failure is the coupling
over-stress, with a secondary settle effect.

## 4. Verdict

**Qualifying embodiment drop, under an embodiment-invariant instrument,
separated from control-style.** A different robot *body* degrades cooperation
on this coupling-valid task — genuinely (real coupling over-load, not the
wrist-proxy artifact), and far beyond any control-style (policy) variation
(which the same instrument shows stays in-band). This is the thesis-bearing
heterogeneity result the ladder was built to produce: **embodiment is the axis
that matters.**

Two honest qualifications, both pre-committed:
1. **Fixed-ego, not incumbent-based.** Measured against the cooperative
   reference ego (valid: only the partner body changes), because the
   rigid-trained frozen incumbent does not transfer to the compliant coupling.
   The incumbent-specific Δ needs a compliant-coupling re-freeze (next slice);
   the embodiment effect is large and ego-independent (it shows against the
   cooperative ego and at static hold), so the re-freeze is expected to confirm,
   not overturn, it.
2. **The effect includes station-keeping load.** The xArm6 over-tensions the
   coupling partly via sag (poor endpoint stiffness), which is a genuine
   embodiment property and a genuine cooperative-safety failure (the coupling
   really bears 1867 N) — but it is "this xArm6 controller/posture on this
   task", not a universal claim about all embodiments. The convergence loop
   (re-freeze; alternative carry postures) can sharpen it further.

## 5. The full co-carry ladder

| Rung | Coupling / instrument | Result |
|---|---|---|
| 0–3 | rigid / wrist | rig valid; incumbent frozen; PH pooled null (one stiff teammate Δ=0.25) |
| 4 | rigid / wrist | EH feasibility wall — rigid weld over-couples; xArm6 fights, fails placed+level+unstressed |
| 4b | compliant / wrist | over-coupling resolved (xArm6 places+levels); EH blocked by embodiment-biased wrist proxy |
| **4c** | **compliant / coupling force** | **embodiment-invariant instrument; qualifying EH drop — xArm6 over-loads the coupling ~7× the matched pair / ~6× the control-style floor; embodiment ≫ control style. (Incumbent-based Δ pending a compliant re-freeze.)** |

## 6. Reproduce

```bash
# Pre-registration committed + tagged before measurement:
#   prereg-cocarry-rung4c-eh-2026-06-20
bash scripts/repro/cocarry_rung4c_eh_measure.sh   # instrument + f_max + measurement
uv run python /tmp/...  # same-ego floor (committed: cocarry_rung4c_sameego_floor.json)
```

| Artifact | Path |
|---|---|
| Pre-registration (tagged) | `spikes/results/cocarry/rung4c/cocarry_rung4c_eh_prereg.json` |
| Invariance investigation | `spikes/results/cocarry/rung4c/cocarry_rung4c_invariance.json` |
| Measurement (f_max, transfer check, calibration) | `spikes/results/cocarry/rung4c/cocarry_rung4c_eh_measurement.json` |
| Same-ego EH-vs-control-style floor | `spikes/results/cocarry/rung4c/cocarry_rung4c_sameego_floor.json` |

- Seeds: f_max 80000–80011, PH/floor 80100–80111, EH 80200–80211 (disjoint
  from all prior). f_max(coupling)=366 N; compliant K=8000 N/m; C_min 0.75,
  radius 0.10 m unchanged. Frozen incumbent SHA `6290e2ae…` (eval-only; does
  not transfer to compliant — see §2).

_ADR-026 §Decision 4 / §Validation criteria / §Open-questions (resolved);
ADR-005; ADR-009; R-2026-06-B §15 Rung 4c._
