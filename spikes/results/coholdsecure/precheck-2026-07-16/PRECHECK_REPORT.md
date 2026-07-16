# Co-hold-secure PR-A engineering precheck — report (2026-07-16)

**VERDICT: PROCEED** — computed by the pre-stated rule (P1 ∧ P2 ∧ P4) from
`precheck_results.json` via the committed `compute_verdict()`
(`scripts/repro/_coholdsecure_precheck.py --render` reproduces every table
and the verdict below from the JSON; nothing here is hand-copied).
Non-registered, non-gating (ADR-029 §Decision): this verdict's only effect is
to enable the founder gate — signing the Gate-0 prereg tag for PR-B.

The rule was committed before any measured episode
(`PRECHECK_PRESTATEMENT.md`, the preceding commit on this branch).

## Per-cell table (recomputed from `precheck_results.json`)

Values are seed-0 records; all three seeds are byte-identical per cell (the
rig and controllers are deterministic — the pre-stated expectation; the
per-seed records are in the JSON). Success counts pool seeds {0, 1, 2}.

| control | clearance (mm/side) | detent (N) | success | depth (mm) | align (°) | part excursion (mm) | part tilt (°) | peak wrench (N) |
|---|---|---|---|---|---|---|---|---|
| matched | 1.5 | 40 | **3/3** | 10.0 | 0.0 | 4.22 | 0.33 | 42.4 |
| limp | 1.5 | 40 | 0/3 | 2.54 | 29.8 | **176.0** | 14.8 | 76.3 |
| none | 1.5 | 40 | 0/3 | −186.5 | 114.0 | **382.8** | 136.7 | — |
| fixture | 1.5 | 40 | 3/3 | 10.0 | 0.0 | 0.0 | 0.0 | — |
| matched | 2.5 | 40 | **3/3** | 10.0 | 0.0 | 3.46 | 0.31 | 36.6 |
| limp | 2.5 | 40 | 0/3 | 40.7¹ | 36.9 | **382.0** | 36.7 | 57.2 |
| none | 2.5 | 40 | 0/3 | 348.1¹ | 130.6 | **421.1** | 175.9 | — |
| fixture | 2.5 | 40 | 0/3 | 0.5 | 1.3 | 0.0 | 0.0 | — |
| matched | 3.5 | 40 | **3/3** | 10.0 | 0.0 | 3.59 | 0.33 | 18.8 |
| limp | 3.5 | 40 | 0/3 | −2.3 | 5.5 | **56.7** | 5.3 | 22.4 |
| none | 3.5 | 40 | 0/3 | −192.1 | 114.1 | **388.1** | 113.3 | — |
| fixture | 3.5 | 40 | 0/3 | −3.0 | 0.2 | 0.0 | 0.0 | — |

¹ Once the part escapes, the depth/align channels are relative to a runaway
part pose and read as arbitrary large values; the pose-excursion channel is
the meaningful failure signature. "—" = no holder participates in the
`none` / `fixture` controls, so the holder-wrench channel reads
identically zero there (stored as `0.0` in the JSON; rendered as "—" here
to mark the channel as not physically meaningful for those controls).

## P1 — solvability: **PASS** (3/3 cells; the bar was ≥ 2/3 incl. middle)

C-matched seats through the 40 N detent with success = 1.0 (full predicate:
`seated ∧ pose_held ∧ within_force ∧ static ∧ settled`, force budgets `+inf`)
on **all three** clearance cells, all seeds: full 10.0 mm engagement,
alignment ≤ 0.05°, part excursion 3.5–4.2 mm (inside the 5 mm pose_held
tolerance), peak holder wrench 18.8–42.4 N.

## P2 — coupling-liveness: **PASS**

- **C-limp** (grasp maintained, zero corrective action) success = 0.0 on
  every matched-passing cell, and the pre-stated mechanism is observed on
  every episode: **part-pose escape under the detent load** — max part
  excursion 56.7 / 382.0 / 176.0 mm ≫ the 5 mm tolerance, with every
  telemetry channel finite (no physics artifact; peak wrench 22–76 N,
  far below the 573–1306 N S2 over-constraint signature).
- **C-none** (part on the passive stand, unrestrained) success = 0.0 on
  every cell — the securing load's lateral component walks the part off the
  stand (excursions 380–420 mm). Two-robot necessity holds by construction
  and by measurement.

## P3 — fixture pre-read (reported, not gating): **MIXED**

C-fixture (world-fixed part — the A2 rehearsal) seats only on the tightest
cell (1.5 mm: 3/3) and fails on the middle and loosest cells (0/3; the press
stalls at the mouth or the detent ejects the plug with no compliant reaction
path). Two honest caveats recorded for the ADR-029 A2-posture discussion:

1. The securer's press config was tuned on the **matched** rig (a tuning
   asymmetry); at a higher-authority press during bring-up the fixture cells
   did seat. This pre-read therefore does **not** establish that a
   single-robot-plus-fixture baseline fails the task — only that at the
   frozen instrument it does not trivially match the team.
2. The A2 question proper belongs to the ADR-027 admission instruments at
   PR-B, under the founder-signed prereg with its own reference-policy
   protocol. No admission claim is made here (the pre-stated P3 scope).

## P4 — stress-channel liveness: **PASS**

Creep-press probe on the middle cell, statistic = p90 of the holder
workpiece-wrench over detent-active steps:

| detent (N) | active steps | wrench p90 (N) | wrench max (N) |
|---|---|---|---|
| 20 | 6 | 16.06 | 18.80 |
| 40 | 101 | 29.97 | 34.74 |
| 60 | 85 | 33.75 | 36.32 |

Strictly monotone across the sweep; every sample finite; max 36.3 N < the
180 N artifact bound (3 × the 60 N working force limit). The stress channel
transmits the process load to the holder and shows no over-constraint
artifact (the S2 lesson does not recur on the fixed-link attach). At 20 N
the creep press clicks through (6 active steps then latch); at 40/60 N it
rides the ramp quasi-statically (85–101 active steps) — the intended
near-equilibrium sampling. (Every number in this table is a
``detent_active_steps`` / ``wrench_active_p90_n`` / ``wrench_active_max_n``
field of the committed JSON.)

## Determinism

All three seeds are byte-identical per cell, as pre-stated (fixed ready
poses, no init noise, substream-routed RNG the controllers do not read).
Re-running `scripts/repro/coholdsecure_precheck.sh` rewrites
`precheck_results.json` byte-identically — verified twice during this run's
finalisation, across two behaviour-identical env hardening edits (skipping
the rejected detent reaction on the kinematic fixture body; making same-tick
re-evaluation a no-op + a loud `num_envs != 1` guard from review).

## What PR-B needs (the PROCEED hand-off)

1. **Founder actions:** flip ADR-029 RFC → Accepted (or amend), and sign the
   Gate-0 prereg tag (ADR-028 document-form schema) with the frozen
   thresholds — including the sourced public industrial-tolerance citation
   the card defers to Gate-0.
2. **Instrument freeze:** the driver seat configs in this archive's JSON
   (`driver_ego_press`, `driver_ego_creep`, `driver_holder_reference`) are
   the working instrument; PR-B freezes them (or their re-derived
   successors) under the tag before any registered episode.
3. **Registered run:** the coupling-validity + solvability verdict under the
   signed tag (the ADR-027 A1–A4 instruments), including the A2 posture
   ruling with this P3 pre-read as input.
4. Only after Gate-0 passes: the separately-ratified learned-ladder plan
   (ADR-029 §Decision, stage 4) — including the own-channel/partner-channel
   definability check before any B-BLIND cell is preregistered.
