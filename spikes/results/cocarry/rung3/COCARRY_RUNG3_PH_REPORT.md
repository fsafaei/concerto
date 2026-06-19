# Co-carry Rung 3 — Policy-Heterogeneity (PH) measurement

**The first real heterogeneity result on a coupling-valid task.** A
pre-registered, capability-controlled measurement of whether a *different
partner policy* (same Panda body) degrades cooperation against the frozen,
held-out-validated Rung-2 incumbent.

- **Axis:** PH — policy heterogeneity (control-style shift; embodiment held
  fixed — the xArm6 teammate is Rung 4).
- **Verdict (pre-committed pooled rule):** **thesis-disconfirming null** —
  pooled mean Δ = **+0.083** < Δ_min = 0.20; the two-sided 95% CI **[0.000,
  0.167]** excludes Δ_min.
- **But the pooled null is heterogeneous and must NOT be read as "PH is
  inert":** the measurement *positively demonstrates* that policy
  heterogeneity **can** reduce cooperation — one capability-matched teammate
  (a stiffer impedance partner) degrades it by **Δ = 0.25** (individually a
  qualifying drop), while the compliant and timing-shifted teammates degrade
  it by **exactly 0**. The pooled estimator dilutes one real effect with two
  zeros. The capability gate also **excluded** a candidate, which biases the
  pooled Δ toward zero (caveat below).

Governance: incumbent **frozen** and never retrained; partners black-box
(read task leaves only, I3/ADR-009); success predicate, f_max (130.6 N), and
the 0.10 m radius **unchanged**; no schema bump (I9); no immutable-archive
edits (I8). Phase-2 (scope question open). **Do not merge — founder review.**

---

## 1. What was held fixed (the frozen reference)

| | |
|---|---|
| Frozen incumbent checkpoint | `local://artifacts/f6dad85ec9df4d58_step100000.pt` |
| Checkpoint SHA-256 (sidecar) | `6290e2ae61f5f801bae51fbf602b0d0f456e516465309e93d3db80ffa014bccb` |
| Freeze manifest | `spikes/results/cocarry/rung2/cocarry_rung2_freeze_manifest.json` |
| Matched reference partner | `cocarry_impedance` (the incumbent's cooperation ceiling) |
| Success predicate / f_max / radius | UNCHANGED (`is_placed & is_level & is_unstressed & both_static & is_settled`; 130.6 N; 0.10 m) |

The matched reference **reconfirmed 100% (12/12)** on the fresh measurement
seeds (the Step-2 STOP guard passed — nothing changed). Reference centroid
sits at p50 **0.0986 m** vs the 0.10 m radius (max 0.0998) — the razor-thin
transport-precision margin the pre-registration flagged.

## 2. Pre-registration (locked before measuring)

- **Pre-registration:** `cocarry_rung3_ph_prereg.json`, git tag
  `prereg-cocarry-rung3-ph-2026-06-19` (blob-SHA verified: on-disk == tagged).
- **Honesty:** the measurement seeds (30000–30011) were untouched at
  pre-registration time; only calibration (40000–40011, cooperative
  reference — **not** the incumbent) and a synthetic power simulation had run.
- **Δ_min = 0.20**, fixed against the archived power simulation
  (`cocarry_rung3_power_sim.json`): type-I ≈ 0 under the null, MDE ≈ 0.30 at
  80 % power, power ≈ 0.45 exactly at the Δ_min boundary.
- **Decision rule (primary):** conclude *PH reduces cooperation* iff pooled
  mean Δ ≥ 0.20 **and** the one-sided 95 % CI on the pooled Δ excludes 0.
  IQM secondary (non-gate); cluster-robust binomial GEE confirmatory.
- **Null rule:** positive control holds (single-arm ≈ 0, Rung 1) **and** Δ ≈ 0
  (CI excludes Δ_min) ⇒ thesis-disconfirming null.

## 3. Capability calibration (the weaker-teammate confound, defused)

Each candidate was paired with a **cooperative reference ego** (the matched
impedance on the ego seat — *not* the frozen incumbent) and gated at
**C_min = 0.75** (= matched M − 0.25; M measured = 1.0). Roster:
`cocarry_rung3_calibration_roster.json`.

| Candidate | Policy shift | Calib. success | Gate |
|---|---|---|---|
| `cocarry_stiff_impedance` | stiffness (high-gain, low-damping, fast) | 100 % | **admitted** |
| `cocarry_admittance` | compliance (force-follower, lets partner lead) | 100 % | **admitted** |
| `cocarry_slew_impedance` | trajectory timing (rate-limited target ramp) | 100 % | **admitted** |
| `cocarry_nullspace_impedance` | redundancy resolution (null-space posture) | **0 %** | **EXCLUDED** |

The null-space variant could not be capability-matched (0/12 with the
cooperative reference: centroid p50 0.146 m > 0.10 radius, fails placed
12/12; stress p90 184 N, fails unstressed 12/12) — a genuinely weaker
teammate the gate is designed to catch. It is archived, **not** entered into
the test. **3 teammates cleared the gate** (the ≥ 3 floor).

## 4. The measurement (frozen incumbent + each teammate)

Paired within-seed over the 12 measurement seed-clusters (30000–30011); one
deterministic episode per seed. Full per-seed data + summaries:
`cocarry_rung3_ph_measurement.json`.

| Condition | Success | Δ (vs ref) | 1-sided 95% CI lower | Mechanism (conjunct fails / telemetry) |
|---|---|---|---|---|
| **reference** (`cocarry_impedance`) | **100 %** (12/12) | — | — | tilt p90 13.0°, stress p90 75 N, centroid p50 0.099 m |
| `cocarry_admittance` | **100 %** (12/12) | **0.000** | 0.000 | *gentler*: tilt p90 11.8°, stress p90 65 N, centroid p50 0.078 m |
| `cocarry_slew_impedance` | **100 %** (12/12) | **0.000** | 0.000 | tilt p90 8.7° (lowest), stress p90 82 N |
| `cocarry_stiff_impedance` | **75 %** (9/12) | **+0.250** | **+0.083** | **fail level ×3, fail placed ×2**; tilt p90 **31.2°**, stress p90 **122 N**, centroid max 0.109 m |

**Pooled (the pre-registered gate):** mean Δ = **+0.083**, one-sided 95 % CI
lower = **+0.028**, two-sided CI **[0.000, 0.167]**. Pooling = reference rate
− mean over teammates of shifted rate (each teammate weighted equally;
10 000-iteration seed-cluster bootstrap; P6-deterministic).

**IQM (secondary):** pooled IQM Δ = 0.0 (the 25 %-trim drops the single
stiff-impedance failures; non-gate-bearing, as pre-registered).

**Cluster-robust binomial GEE confirmatory:** *degenerate* — with the
reference at the 1.0 ceiling and two of three teammates also at 1.0, the
`shifted` coefficient is near-perfectly separated (coef ≈ −26, Wald p
undefined). Uninformative here; the bootstrap is the headline, exactly as the
pre-registration specified for this contingency.

## 5. The mechanism (which cooperative channel tips over)

The drop is **entirely the stiff-impedance partner**, and the channel is the
one the coupling is carried by: **bar tilt** (the *level* conjunct), with
placement secondary.

- The stiff partner drives its bar-end hard and fast (high gain, low
  damping). Through the rigidly-welded shared bar this transmits a larger
  internal force the incumbent did not train against: tilt p90 jumps from
  13.0° (reference) to **31.2°**, over the 15° limit on 3 seeds, and wrist
  stress p90 rises from 75 N to **122 N** (near the 130 N ceiling). On 2 more
  seeds the centroid is pushed just past the 0.10 m radius (max 0.109 m) —
  the predicted transport-precision channel.
- The **admittance** follower is *more* compliant than the matched partner,
  so it presents an even gentler load — the incumbent cooperates with margin
  to spare (centroid p50 0.078 m, stress p90 65 N).
- The **slew** (timing) partner reaches the same target on a delayed ramp;
  the incumbent simply meets the goal-ward motion later and settles (tilt p90
  8.7°, the lowest of all).

So the incumbent is **robust to compliance and timing shifts** but
**specifically brittle to higher-stiffness / higher-force partners**.

## 6. Verdict and how to read it

**Pre-committed pooled rule ⇒ thesis-disconfirming null.** Pooled mean Δ =
0.083 < Δ_min = 0.20, and the two-sided 95 % CI [0, 0.167] excludes Δ_min.
The positive control holds (single-arm ≈ 0, established at Rung 1; the bar is
unchanged). Under the rule fixed before the data, **no qualifying pooled
drop** was observed.

**This is decision-relevant, and it is NOT "PH is inert."** Three honest
qualifications, all pre-committed or mechanical:

1. **The pooled null is driven by dilution, not by absence of effect.** The
   per-teammate breakdown — pre-registered to be reported — shows
   `cocarry_stiff_impedance` is an *individually qualifying drop* (Δ = 0.25 ≥
   Δ_min; its own one-sided CI lower +0.083 > 0). The pooled estimator
   averages that real effect with two exact zeros.
2. **The one-sided pooled CI excludes zero (+0.028).** There *is* a small,
   statistically-resolvable nonzero pooled degradation — it is simply **below
   the pre-registered meaningful threshold**. "Sub-threshold," not "absent."
3. **The capability gate excluded a candidate (`cocarry_nullspace_impedance`),
   which truncates heterogeneity from above and biases Δ toward zero.** Per
   the pre-committed caveat, this null may **not** be read as "the PH axis
   does not couple to cooperation" — only as "*among capability-matched
   policies, no qualifying pooled drop was observed.*" And in fact the data
   show the axis *does* couple, policy-specifically (stiff impedance).

**Bottom line for the thesis:** policy heterogeneity on this coupling-valid
task produces a **policy-specific, asymmetric** effect — a stiffer partner
degrades cooperation through the bar-tilt/over-stress channel, while
compliant and differently-timed partners do not. The pre-registered *pooled*
gate does not clear Δ_min, so the axis does **not** earn a benchmark slot on
the pooled criterion under this task; but the result motivates the solution
phase precisely (robustify the incumbent against high-stiffness / high-force
partners) and is **not** a clean "no effect."

## 7. What this hands to Rung 4 (embodiment shift)

Rung 4 swaps the embodiment (xArm6-Robotiq). This PH result is the control —
it shows a **higher-stiffness control style alone** (same Panda body) already
costs ~25 % via bar tilt/over-stress. So when the xArm6 teammate degrades
cooperation, Rung 4 must separate a genuine *embodiment* effect from this
*control-style* effect — e.g. by matching the xArm6 teammate's effective
end-point stiffness to the admitted Panda controllers, or by reporting the
xArm6 Δ against the stiff-impedance Δ as the control-style floor.

## 8. Reproduce

```bash
# 1. Calibration (done; roster committed)
bash scripts/repro/cocarry_rung3_calibrate.sh
# 2. Power sim (done; pre-registration support)
uv run python scripts/repro/_cocarry_rung3_power_sim.py
# 3. Pre-registration: committed + tagged BEFORE the measurement
#    git tag prereg-cocarry-rung3-ph-2026-06-19
# 4. Measurement (frozen incumbent + reference + 3 admitted teammates)
bash scripts/repro/cocarry_rung3_ph_measure.sh
```

| Artifact | Path |
|---|---|
| Pre-registration (tagged) | `spikes/results/cocarry/rung3/cocarry_rung3_ph_prereg.json` |
| Calibration roster (incl. exclusion) | `spikes/results/cocarry/rung3/cocarry_rung3_calibration_roster.json` |
| Power simulation | `spikes/results/cocarry/rung3/cocarry_rung3_power_sim.json` |
| Measurement (per-seed / per-teammate / Δ / CI) | `spikes/results/cocarry/rung3/cocarry_rung3_ph_measurement.json` |

- **Seeds:** measurement 30000–30011; calibration 40000–40011 (both disjoint
  from Rung-2 selection S 10000–10011 and validation V 20000–20023).
- **Frozen incumbent SHA-256:** `6290e2ae…a014bccb` (step 100000).
- Determinism: env + bootstrap RNG route through
  `concerto.training.seeding.derive_substream` (P6); the frozen incumbent
  acts in policy-mode (no RNG); CPU/GPU byte-identical per the §3.4 reload
  contract.

_ADR-026 §Decision 4; ADR-026 §Validation criteria; ADR-009 §Decision;
R-2026-06-B §15 Rung 3._
