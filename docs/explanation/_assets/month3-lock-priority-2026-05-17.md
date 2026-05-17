# CONCERTO Month-3 lock-priority report

## TL;DR — Recommendation

**Recommendation: Defer — Stage 1b not yet measured (Phase-1 milestone; guardrail ≤4 weeks)**

Surviving axes (ADR-007 staging order): **(none yet)**.
ADR-008 HRS bundle: **hold** — surviving headline-axis set < 3 (re-check ADR-008 §Decision).

## Per-axis evidence (≥20 pp gap test)

| Axis | Stage | Status | n_seeds | n_episodes | gap_iqm_pp | ci_low_pp | ci_high_pp | gate_pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AS | 1a | Stage 1a (rig-only) | 5 | 200 | n/a | n/a | n/a | n/a |
| OM | 1a | Stage 1a (rig-only) | 5 | 200 | n/a | n/a | n/a | n/a |
| CR | 2 | Missing | n/a | n/a | n/a | n/a | n/a | n/a |
| CM | 2 | Missing | n/a | n/a | n/a | n/a | n/a | n/a |
| PF | 3 | Missing | n/a | n/a | n/a | n/a | n/a | n/a |
| SA | 3 | Missing | n/a | n/a | n/a | n/a | n/a | n/a |

## Per-stage gate enumeration

### Stage 1b

Stage 1b gate: deferred — no Stage-1b SpikeRun present.

- AS — Stage 1a (rig validation only; no ≥20 pp measurement)
- OM — Stage 1a (rig validation only; no ≥20 pp measurement)

Stage 2 launch: BLOCKED — Stage 1b not measured (ADR-007 §Stage 1b guardrail: launch ≤4 weeks after Month-3 lock).

### Stage 2

Stage 2 gate: deferred — no Stage-2 SpikeRun present.

- CR — Missing
- CM — Missing

Stage 3 launch: deferred — Stage 2 not measured.

### Stage 3

Stage 3 gate: deferred — no Stage-3 SpikeRun present.

- PF — Missing
- SA — Missing

Month-3 lock launch: deferred — Stage 3 not measured.

## ADR-by-ADR action

| ADR | Action | Rationale |
| --- | --- | --- |
| ADR-007 | Hold at Accepted; Stage 1b launch is Phase-1 milestone (§Stage 1b guardrail) | Implementation staging gates; §Validation criteria threshold (≥20 pp). |
| ADR-008 | Hold bundle lock — insufficient evidence | §Decision default headline bundle CM x PF x CR; Option A / Option B fallbacks. |
| ADR-011 | Hold baseline-set lock — §Validation coverage check not satisfied | Baseline-set lock contingent on per-axis evidence for the surviving axes. |

## ADR-014 safety-table integration

SA spike not present — three-table safety report deferred to Stage 3 launch (ADR-014 §Decision unchanged).

## Open-work flags (ADR-INDEX)

- **(b) ADR-007 §Stage 1a / §Stage 1b split.** Stage 1b pending; flag remains active until Phase-1 launch (ADR-007 §Stage 1b guardrail: ≤4 weeks after Month-3 lock review).
- **(c) ADR-008 HRS bundle composition.** Bundle composition pending the surviving-axis set (§Decision footnote c).
- **(f) ADR-014 safety reporting.** Reporting contract qualified by ADR-008 bundle composition (flag c above) and the PR-A2 conformal-loss instrumentation; per-axis row counts firm up once Stage 3 SA lands.

## Provenance

- chamber version: `0.6.0`
- git SHA: `b3a5cb2d10d686f60863583fccf85a297fafd928`
- timestamp (UTC): `2026-05-17T11:43:12.291751+00:00`
- gate threshold (pp): `20.00`
- bootstrap n_resamples: `2000`
- bootstrap root seed: `0`
- per-axis archives:
  - `AS` → `spikes/results/stage1-AS-20260517/spike_as.json` (prereg_sha=`29e397a4a012813c58fd4a0c3077ea8c754affc8`, git_tag=`prereg-stage1-AS-2026-05-15`)
  - `OM` → `spikes/results/stage1-OM-20260517/spike_om.json` (prereg_sha=`9fad74738b56d4e067ea8bb29c60ef9f652b5b68`, git_tag=`prereg-stage1-OM-2026-05-15`)
