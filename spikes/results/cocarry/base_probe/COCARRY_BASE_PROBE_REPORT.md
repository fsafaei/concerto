# Co-carry base-difficulty probe — synthesis & verdict

**Date.** 2026-06-22 · **Author.** fsafaei · **Branch.** `feat/cocarry-base-difficulty-probe` (off `main`) · **Do not merge — founder review.**

**What this is.** A cheap, eval-only, **no-training** Phase-2 probe (board base-failure criterion, Convening R-2026-06-C): does a harder co-carry regime exist where the competent hand-built **matched pair** itself stops trivially solving the task — and degrades for **coordination** reasons (graceful coordination-strain) rather than **numerical over-coupling** (the Rung-4 wall)? It decides between **Option 2** (build the full heterogeneity test in a genuinely-hard regime) and **Option 3** (conclude heterogeneity is not a cooperation-breaker on co-carry-class tasks). Phase-2, **non-gating**; does not touch Phase-1 or the M10 gate.

**Pre-registration (LOCKED before running).** `cocarry_base_probe_prereg.json`, tag `prereg-cocarry-base-probe-2026-06-22` (signed). The grid, seeds, artifact-vs-real thresholds, and the falsifiable-both-ways decision rule were fixed and committed before any rollout; the driver reads the prereg as its single source of truth.

---

## One-line verdict

> **`HARD_BUT_FEASIBLE_BAND_EXISTS`** — but the only graceful band is a **precision-tolerance** band (tilt limit 5°, success 0.667); the coupling-**physics** knob (stiffness) produces **no** hard-but-feasible band at all — it goes trivial → over-coupling cliff. **The co-carry coupling physics does not have a graceful "hard regime"; only tightening the success tolerance does.**

This is a real answer for little spend (~10 GPU-min). It points to a **narrow Option-2 opening** (a precision-strained regime) *and* strongly corroborates the **Option-3 / reframe** reading for the coupling physics (a genuinely contact-hard regime needs a *different task class*, not more stiffness).

---

## Method (eval-only, matched pair, predicate never weakened)

- **The pair (fixed):** cooperative-impedance ego (`build_cooperative_ego`) + matched impedance partner (`cocarry_impedance`) — the "base" config of the Rung-5 `base_robustness_control`. No training, no residual, no heterogeneity. The **only** thing varied is task difficulty.
- **Instrument:** the embodiment-invariant bar **coupling force** (`stress_measure="coupling"`), success ceiling f_max = **365.6 N** (Rung-4c grounding: 1.25× matched-pair coupling success-stress p99; matched-pair rate at f_max = 1.0).
- **Predicate:** the frozen joint predicate `placed ∧ level ∧ unstressed ∧ static ∧ settled`, **never weakened**. On `main` the env's success flag gates at the 130 N *wrist* ceiling (a Rung-5 `stress_max` param is not on `main`), so success is computed **post-hoc** via the pure `evaluate_cocarry_success(…, stress_max=365.6)` on recorded raw metrics. The matched hand-built controllers ignore reward, so rollout physics are byte-identical to the Rung-5 base.
- **Seeds:** 12 (70000–70011), disjoint from all prior co-carry seed sets.
- **Artifact-vs-real test:** per-step instantaneous trajectories (tilt, coupling stress, centroid) recorded for every stiffness setting; a setting is `NUMERICAL_ARTIFACT` (over-coupling / blow-up) iff non-finite **or** median episode-max stress ≥ 3× f_max (1097 N) **or** median episode-max tilt ≥ 60°.

---

## The difficulty → base-performance curve

### Primary: coupling stiffness (real rollouts; pre-registered geometric grid)

| K (N/m) | success | coupling stress p50/p90/max (N) | stress p90 / f_max | tilt p90 (°) | dominant failure | class |
|--------:|:-------:|:-------------------------------:|:------------------:|:------------:|:----------------:|:------|
| **8000** (base) | **1.000** | 249 / 275 / 283 | 0.8× | 6.0 | — | `FEASIBLE_TRIVIAL` |
| 16000 | 0.000 | 584 / 619 / 619 | 1.7× | 9.8 | unstressed 12/12 | `STABLE_INFEASIBLE` |
| 32000 | 0.000 | 1206 / 1269 / 1283 | 3.5× | 18.5 | unstressed 12/12; level 9/12 | `NUMERICAL_ARTIFACT` |
| 64000 | 0.000 | 2303 / 2381 / 2399 | 6.5× | 17.0 | unstressed; level 6/12 | `NUMERICAL_ARTIFACT` |
| 128000 | 0.000 | 3335 / 3541 / 3612 | 9.7× | 14.9 | unstressed 12/12 | `NUMERICAL_ARTIFACT` |
| 256000 | 0.000 | 6401 / 6918 / 6970 | 18.9× | 12.7 | unstressed 12/12 | `NUMERICAL_ARTIFACT` |
| 512000 | 0.000 | 12655 / 14077 / 14362 | 38.5× | 15.7 | unstressed; level 12/12 | `NUMERICAL_ARTIFACT` |

The coupling force scales ~**linearly** with K (post-settle stress p90 ≈ 275 → 619 → 1269 → 2381 → 3541 → 6918 → 14077 N — each doubling of K roughly doubles the force). This is the `F = K·deflection` over-coupling signature: the matched pair has a small steady kinematic tracking mismatch; the calibrated K=8000 spring keeps the transmitted force below f_max (benign), and **any** stiffening over-transmits that same mismatch proportionally.

### Exploratory boundary (NON-pre-registered; fills the 8k–16k geometric gap)

`cocarry_base_probe_stiffness_boundary_exploratory.json` — does NOT change the locked verdict; characterises the cliff.

| K (N/m) | success | stress p50/p90 (N) | fail (unstr/level/place) | mechanism |
|--------:|:-------:|:------------------:|:------------------------:|:----------|
| 9000 | 1.000 | 259 / 270 | 0/0/0 | — |
| 10000 | 0.917 | 335 / 344 | 1/0/0 | over-coupling (unstressed) |
| 11000 | 0.000 | 497 / 521 | 12/0/0 | over-coupling (unstressed) |
| 12000–14000 | 0.000 | ~420–480 | 12/(2–3)/0 | over-coupling (unstressed) |

**Finding `NO_SUB1_BAND_IN_GAP`.** The transition is razor-thin: 100% (9k) → 91.7% marginal (10k) → 0% (11k). **No stiffness setting anywhere lands success in [0.10, 0.90]**, and every degraded setting fails on the `unstressed` conjunct (the bar stays placed and level — the coupling simply over-fights). This is over-coupling onset, **not** graceful coordination-strain.

### Secondary: success-predicate tightening (post-hoc on the K=8000 base rollouts; physics unchanged)

| goal radius (m) | 0.10 | 0.05 | 0.02 | | tilt limit (°) | 15 | 10 | **5** |
|:---:|:---:|:---:|:---:|---|:---:|:---:|:---:|:---:|
| **success** | 1.000 | 1.000 | 1.000 | | **success** | 1.000 | 1.000 | **0.667** |

The matched pair places the bar centroid at ~**0.005 m** (max 0.006) — 20× inside the tightest 0.02 m radius, so **radius never bites**. It runs at tilt p90 ~6° (8/12 seeds peak < 5°), so tightening the tilt limit to **5°** is the **only** knob that produces a graceful, physics-stable sub-1.0 band (success 0.667).

---

## Artifact-vs-real: the two failure modes are cleanly distinguishable

The per-step trajectories (`cocarry_base_probe_trajectories.json`) confirm the classification (values are post-settle; the step-0 placement transient is excluded by the 15-step settle window):

- **K=8000 (real, feasible):** post-settle stress decays 266 → 79 N, tilt → 0.6°. Clean convergence.
- **K=16000 (stable-infeasible):** stress rises and plateaus ~450–550 N (1.5× f_max), tilt held ~9° — bounded, bar held, but the coupling fights past f_max on every seed.
- **K≥32000 (over-coupling artifact):** finite and non-divergent (no NaN), but coupling forces are physically absurd (1.2–14 kN through a 0.4 kg bar) and scale with K. This is the Rung-4 over-coupling wall — **not** a coordination phenomenon.

No setting failed via numerical *divergence*; the "artifact" is over-coupling force magnitude, exactly the mechanism the prereg named.

---

## Verdict & honest distinction

**Formal (pre-committed rule):** `HARD_BUT_FEASIBLE_BAND_EXISTS` — one graceful, physics-stable setting with success in [0.10, 0.90]: **tilt limit = 5°** (success 0.667), a **precision-tolerance** band.

**The distinction the decision turns on (pre-registered `report_distinction`):**

1. **The coupling-PHYSICS knobs yield no graceful regime.** Stiffness goes trivial (≤9k, 100%) → cliff (10k 91.7% → 11k 0%) → over-coupling (≥16k), failing only on `unstressed`. Goal radius never bites. **There is no co-carry coupling regime, via these knobs, where the physics genuinely strains a competent controller without the coupling over-fighting.** This is the mechanistic confirmation of "the spring does the cooperation": the calibrated compliant coupling either absorbs partner mismatch (benign) or over-couples (artifact) — there is no graceful middle.

2. **The only graceful band is the success TOLERANCE.** Tightening the leveling limit to 5° makes the matched pair degrade gracefully (0.667), because it runs with only ~1° of tilt margin there. This is a *measurement-precision* hard regime, not a *coordination-physics* one.

---

## Implications & next step (for the founder + review board)

- **Narrow Option-2 opening.** A heterogeneity test in the **tilt-5° regime** is the only "harder regime" these knobs offer. There, a mismatched partner that adds even a few degrees of tilt would plausibly tip the matched controller — so the test would be meaningful, *but* it measures **tolerance to a tilt perturbation under a tightened predicate**, not coordination under genuinely hard contact physics. If pursued, instrument it per R-2026-06-C (diversity-trained / partner-robust incumbent; the coupling positive-control + pre-committed null rule), and state plainly that the "difficulty" is predicate-tolerance, not physics.

- **Strong corroboration of the reframe (Option B) for the physics.** No buildable co-carry **coupling** regime makes coordination strain a competent controller via stiffness/radius — the compliant spring absorbs or over-couples. Getting genuine coordination-physics hardness requires a **contact-rich, different task class** (e.g. tasks where the cooperation must shape contact, not just suspend a bar between two compliant pins). That is a **separate, bigger, gated decision** — explicitly **not** "more stiffness."

- **Recommendation (input only — the founder/board make the call).** Treat the coupling-stiffness axis as *settled*: it is a binary benign/over-coupling switch, not a difficulty dial. Before committing to the full Option-2 build on the tilt-5° precision band, weigh it against the reframe: the precision band is cheap to test but answers a narrower question than the board's "is heterogeneity a cooperation-breaker under hard coordination physics?" — which this task class, on this evidence, cannot pose.

---

## Reproduction (evidence)

```bash
# Locked pre-registration (tag prereg-cocarry-base-probe-2026-06-22)
scripts/repro/cocarry_base_probe.sh                 # = uv run python scripts/repro/_cocarry_base_probe.py
uv run python scripts/repro/_cocarry_base_probe_boundary.py   # exploratory 8k-16k boundary (non-prereg)
```

| artifact | sha256 (first 16) |
|:--|:--|
| `cocarry_base_probe_prereg.json` | `894f92370c6466d1` |
| `cocarry_base_probe_measurement.json` | `a4f8e875912b6b8e` |
| `cocarry_base_probe_trajectories.json` | `c9b15cb77e859677` |
| `cocarry_base_probe_stiffness_boundary_exploratory.json` | `6a0d4cdfb8fcddb6` |
| `src/chamber/envs/cocarry.py` (env module at run) | `60a0c44083db7c8d` |

- Seeds: 70000–70011 (12). f_max coupling = 365.608 N. Episode 320. Render backend `none`. Run on `main` @ `2e91d44` + this branch's driver.
- Governance: eval-only; no training/residual/heterogeneity; matched pair; predicate never weakened. Phase-2, NON-gating. Prior artifacts immutable (I8); no `chamber.comm.SCHEMA_VERSION` bump (I9). ADR-026 §Decision 1-2, §Open-questions (coupling stiffness is a task parameter), §Validation-criteria.
