# Pre-statement — co_hold_secure PR-A engineering precheck (2026-07-16)

**Status: committed BEFORE any measured episode runs.** The commit carrying
this file precedes the commit carrying `precheck_results.json` /
`PRECHECK_REPORT.md`; the branch history is the rule-before-result record
(ADR-007 §Discipline applied in its pre-statement form — this is **NOT a gate
claim**, no pre-registration tag exists or is rotated here; the founder-signed
Gate-0 pre-registration is PR-B and happens only on a PROCEED verdict, per
ADR-029 §Decision).

## What this precheck is

Eval-only, **non-registered, non-gating** Phase-2 benchmark work (invariant
I1: nothing here touches the Phase-1 gate; no leaderboard, admission, or
Gate-0 claim is made). It informs exactly one named decision: whether the
founder signs the Gate-0 prereg tag for PR-B (ADR-029 §Decision, staged
gates). The four bounds below are the PR-A brief's pre-stated bounds,
founder-confirmed at kickoff (2026-07-16), together with the geometry anchors
(engagement depth 10 mm; per-side clearance band {1.5, 2.5, 3.5} mm; >= 2 mm
45° lead-in chamfer; 40 N detent over the final 2 mm) — which the ADR-029
wedge derivation independently supports (wedge-limit tilts 8.5° / 14.0° /
19.3° >= 2 x the measured 2.8° achievable-control ceiling).

## Cells and draws

- Env: `chamber.envs.co_hold_secure.make_co_hold_secure_env` at the ADR-029
  geometry; horizon 320 ticks at 20 Hz; success = the full committed predicate
  `seated ∧ pose_held ∧ within_force ∧ static ∧ settled` with the Gate-0 force
  budgets unset (`+inf` — raw distributions are collected, no unmeasured
  number is asserted).
- Controls (the four flags; ADR-029 §Decision): **C-matched** (structured base
  securer + cooperative reference holder), **C-limp** (identical rig, holder
  seat driven by the registered zero-action instrument
  `partner_ablated_zero` — the PR #298 C2-style coupling-liveness control),
  **C-none** (no holder; free part resting on the passive stand at the
  nominal pose, unrestrained), **C-fixture** (part world-fixed — the A2
  rehearsal).
- Cells: the three per-side clearance cells {1.5, 2.5, 3.5} mm at the 40 N
  detent anchor, every control; plus the P4 detent sweep {20, 40, 60} N on
  the **middle** cell (C-matched, creep-press probe).
- Seeds: {0, 1, 2} per cell. The rig and both controllers are deterministic
  (P6 / ADR-002: fixed ready poses, no init noise, substream-routed RNG that
  the controllers do not read), so the three seeds are expected to reproduce
  **byte-identical** outcomes — they are the determinism/replication check,
  not a variance sample, and are recorded as such.
- Driver: `scripts/repro/_coholdsecure_precheck.py` (committed beside this
  pre-statement). Two frozen ego-press configs: the seating press
  (`descend_step` 1 mm/tick, `step_max` 25 mm, press ratchet on, unjam
  detector off, seat threshold at the click margin) for P1–P3, and the
  low-authority **creep press** (`descend_step` 0.5 mm, `step_max` 8 mm) for
  P4, so the ramp traversal is quasi-static and the wrench samples
  near-equilibrium. Controller gains are instrument tuning and were fixed at
  the PR-A bring-up **before this pre-statement**; the task geometry and this
  rule are what they are frozen against.

## The pre-stated bounds (verbatim; founder-confirmed at kickoff)

- **P1 solvability:** C-matched seats with success = 1.0 (deterministic
  controllers, seeds {0, 1, 2}) on **>= 2 of the 3 clearance cells including
  the middle cell**.
- **P2 coupling-liveness:** C-limp success = 0.0 on **every cell where
  C-matched succeeds**, with the failure mechanism observed as **part-pose
  escape under the detent load** (operationalised: per-episode max part-pose
  excursion > the 5 mm pose_held tolerance, all telemetry channels finite —
  not a physics artifact); **and** C-none success = 0.0 **everywhere**
  (two-robot necessity — the A2 rehearsal's first half).
- **P3 fixture pre-read (reported, not gating):** C-fixture per-cell outcomes
  and wrench profile beside C-matched — this informs the ADR-029 A2-posture
  record; **no admission claim is made**.
- **P4 stress-channel liveness:** the holder workpiece-wrench responds
  **monotonically** to the detent-force sweep {20, 40, 60} N on the middle
  cell (operationalised: p90 of the wrench over detent-active steps, creep
  probe, strictly increasing across the sweep) and stays finite and **below
  3 x the working force limit** (working force limit = the largest sweep
  setting, 60 N; bound = 180 N — the S2 over-constraint artifact was
  573–1306 N, an order of magnitude over working forces).

## Verdict rule (verbatim)

**PROCEED iff P1 ∧ P2 ∧ P4. STOP otherwise.** P3 never gates. The verdict is
computed by `compute_verdict()` in the committed driver from
`precheck_results.json` — the report's tables and verdict are recomputed from
the committed records, never hand-copied. On STOP: no PR-B, fall back per
ADR-029 (option C posture; one closure paragraph in the ADR's revision
history).

## Archive layout (immutable once written; I8)

```
spikes/results/coholdsecure/precheck-2026-07-16/
  PRECHECK_PRESTATEMENT.md   (this file; the pre-statement commit)
  precheck_results.json      (results commit; per-cell, per-control records)
  PRECHECK_REPORT.md         (results commit; tables + the verdict as computed)
  REPRO.txt                  (results commit; exact reproduction command)
  SHA256SUMS.txt             (results commit)
```
