# ADR-027: CHAMBER-Bench v1.0 protocol — tier ladder, admission, validity matrix, reporting, versioning

**Status.** Accepted (2026-07-05)
**Authors.** Farhad Safaei
**Reviewers.** _solo lock per ADR-INDEX working policy_
**Tags.** v0.2 §3.3, §3.7, §3.10; benchmark protocol; supersedes-in-part ADR-007 §Stage framing (public benchmark story); depends on ADR-026.

## Context

ADR-026 ended the six-axis gate framing as the public benchmark story:
AS/OM as operationalized are construct-invalid controls, the co-carry
PH/EH axes closed as pre-registered robust nulls
(`spikes/results/cocarry/rung3`, `rung4e`), co-insert closed at a
documented HARD_STOP
(`spikes/results/coinsert/COINSERT_CLOSURE_2026-06-24.md`), and
handover-and-place returned a measured `COUPLING_VALID` + `SOLVABLE`
Gate-0 verdict under a tagged pre-registration
(`spikes/results/handover-place-gate0-2026-06-26/`, PR #263). What the
project now possesses is not six validated axes but a set of tasks with
*known, individually-established validity properties* and the machinery
that established them. CHAMBER-Bench v1.0 is the benchmark protocol
that publishes exactly that: tasks admitted by executable evidence,
validity tracked per cell, nothing aggregated past what the evidence
licenses. This ADR commits the protocol; it is a decision, not a
proposal.

## Considered alternatives

| # | Option | Source / advocate | Pros | Cons |
|---|--------|-------------------|------|------|
| A | Scalar composite leaderboard (one CHAMBER-Bench score) | [common benchmark practice] | Legible; rankable; marketable | Weight-gaming across tasks; today it would aggregate nulls and controls into a number implying validated cooperation evidence that does not exist |
| B | Keep the six-axis gate framing as the public benchmark story | [ADR-007 as locked] | Continuity with the published README/plan | Rejected per ADR-026: two axes are construct-invalid as operationalized, none is Validated; the framing would misrepresent the evidence |
| C | Tiered task ladder + executable admission protocol + per-cell validity matrix + per-task reporting, all preregistered | [notes/tier1/construct_validity_in_cooperative_evaluation.md — Melting Pot's per-scenario reporting; Gorsane et al. protocol discipline; Raji et al. construct-validity critique] | Publishes exactly what the evidence licenses; admission is falsifiable both ways; nulls stay informative | More complex to communicate; no single headline number |

## Decision

**Option C.** CHAMBER-Bench v1.0 is a tiered, admission-gated,
per-task-reported benchmark. One line: **tasks earn Tier-2 status by a
committed admission report proving solvability, two-robot
infeasibility, and partner-relevance; validity is tracked per
task×axis cell; results are per-task IQM + bootstrap-CI tables with no
scalar composite; every task and partner set is versioned.**

### Tier ladder

Every task carries exactly one tier:

- **Tier 0 — rig diagnostics.** `stage0_smoke`, `mpe_cooperative_push`.
  Prove the harness runs; carry no cooperation claim.
- **Tier 1 — controls.** Ego-solvable by design-of-record:
  `stage1_pickplace` AS/OM conditions (per ADR-026 these are
  construct-invalid *for cooperation* and are retained precisely as
  controls — a method claiming cooperation gains here is measuring
  something else).
- **Tier 2 — admitted cooperation tasks.** Admission only by the
  protocol below. Co-carry enters now (its rung archives constitute the
  evidence). Handover-and-place is **admission-eligible** — the
  measured Gate-0 verdict (`COUPLING_VALID`, `SOLVABLE`, PR #263,
  archive `spikes/results/handover-place-gate0-2026-06-26/`) covers A1
  and the coupling limb of A2 — and is formalized by its own committed
  admission report as the follow-up implementation work under this
  protocol.
- **Tier 3 — documented candidates and closures.** Co-insert: CLOSED,
  with its closure archive as the record
  (`spikes/results/coinsert/COINSERT_CLOSURE_2026-06-24.md`).
  Co-hold-and-secure: CANDIDATE. VLA partner variants: deferred per the
  ADR-009 exit ramp. Tier 3 entries are part of the benchmark's honest
  surface — a documented closure is a result.

### Admission protocol (Tier-2 entry)

A task enters Tier 2 only by a **committed admission report** showing,
under preregistered thresholds:

- **A1 — solvability.** A reference policy pair meets a preregistered
  success threshold with the stress channel within limits.
- **A2 — two-robot infeasibility.** The best single-robot /
  partner-ablated variant achieves ≈ 0 success, *and* coupling
  constraints measurably bind on successful matched episodes — this is
  ADR-026's coupling positive-control, verbatim.
- **A3 — partner-relevance** *(amended 2026-07-06; see §Revision
  history)*. The admission instrument is the **preregistered
  scripted/ablated partner-relevance check**: a scripted ego stripped
  of exactly its coupling channel underperforms the coupling-aware
  scripted reference by a preregistered margin. **Failure of A3
  demotes the task to Tier 1** — a task whose outcome is insensitive
  to the partner is a control, whatever its physics. The learned
  B-AHT-vs-B-BLIND contrast (ADR-011 rows) is a **reported per-task
  finding, not the admission gate**: under rigid physical coupling,
  masking partner observations does not blind a learned ego — partner
  state re-enters through the ego's own proprioception — so the
  learned contrast measures the value of explicit partner
  observation, not partner-relevance.

Nulls measured **on admitted tasks** are reported as informative nulls,
never re-described as task failures (ADR-026 §Decision 2's rule,
carried over verbatim).

### Validity matrix

Axis validity is tracked per **task × axis** cell with status
`validated | null | invalid | untested`. The HRS aggregate (ADR-008)
computes only over `validated` cells and is **suspended while none
exist** — the aggregate resumes automatically when its first validated
cell does, with no protocol change.

### Reporting rules

- IQM plus 95 % paired-cluster-bootstrap CIs over seeds; mandatory
  per-partner breakdown; **no single-seed claims**.
- Leaderboard is per-task tables. **No scalar composite in v1.0.**
- Checkpoint selection for learned baselines is preregistered: per
  seed, the checkpoint with highest stress-compliant success on a
  held-out validation partner.

### Versioning

- Tasks and partner sets are versioned (`task_id@vN`); **any** change
  bumps the version; old results stay interpretable under their
  version; suite composition is pinned in a generated manifest
  ("CHAMBER-Bench v1.0").
- Each task version additionally pins **one canonical success
  predicate and one canonical stress instrument**; changing either is a
  version bump. Concretely for `cocarry@1`: the **wrist-force
  instrument as it gates success on `main` today is canonical**
  (f_max 130.57 N per the freeze manifest,
  `spikes/results/cocarry/rung2/cocarry_rung2_freeze_manifest.json`);
  the rung-4c coupling-stress instrument (365.6 N ceiling, currently
  post-hoc only) is reported as **secondary telemetry**, and promoting
  it to the gate would create `cocarry@2`.

## Rationale

The protocol publishes the project's actual epistemic state. The
admission protocol is the coupling-validity criterion made executable
(A2 is its positive control; A3 is the cooperation-relevance test the
Stage-1 rig lacked), so a task cannot enter the cooperation tier by
narrative. Per-cell validity with a suspended aggregate is the only
honest treatment of a benchmark whose current cells are controls,
nulls, and one admission-eligible verdict. The reporting rules are the
Agarwal/Gorsane discipline the rung archives already practice. The
alternatives are recorded honestly: a scalar composite (A) invites
weight-gaming and would today aggregate nulls; the six-axis story (B)
was rejected by ADR-026 on construct-validity grounds.

## Evidence basis (links to reading notes)

- [notes/tier1/construct_validity_in_cooperative_evaluation.md] —
  sources 4–7: Melting Pot's per-scenario, held-out-partner design;
  Raji et al. on benchmark construct validity (the standing reason to
  refuse a composite); Gorsane et al. on preregistered protocol
  discipline; Agarwal et al. on IQM + interval reporting.
- `spikes/results/cocarry/` rungs 2–4e; `spikes/results/coinsert/`;
  `spikes/results/handover-place-gate0-2026-06-26/` — the committed
  evidence the tier assignments cite.
- [ADR-026 §Decision] — the coupling-validity criterion and null rule
  this protocol operationalizes.

## Consequences

- **Project scope.** The v1.0 build-out becomes: task registry +
  manifest generation, the admission-report machinery, the right-sized
  partner sets (ADR-009 as amended), the baseline set (ADR-011 as
  amended), and the result-bundle/prereg schema work authorized by
  ADR-028. The handover-and-place admission report is the first new
  admission under the protocol.
- **ADR-008.** The HRS aggregate is suspended (not repealed); its
  bundle composition question becomes moot until validated cells
  exist.
- **ADR-007.** Superseded-in-part a second time: the staged six-axis
  spike protocol remains the historical record of Phase 0–1; the
  public benchmark story is this protocol.
- **Public surfaces.** README leaderboard section points at this ADR;
  the full README rewrite is a separate documentation work item.

## Risks and mitigations

- [risk] Tier 2 holds one task (co-carry) at v1.0 launch, inviting a
  "benchmark of one" critique. → Handover-and-place is
  admission-eligible with its Gate-0 verdict already measured; Tier 3
  documents the candidate pipeline; the honest count is itself the
  point.
- [risk] A3's preregistered margin is set opportunistically after
  looking at B-BLIND runs. → The margin is fixed in the admission
  report's pre-registration before the A3 measurement, under the
  ADR-028 prereg-document model with its tag discipline.
- [risk] Version bumps fragment results across `task_id@vN`. → The
  manifest pins suite composition per bench release; old versions stay
  queryable; a bump requires a one-line justification in the manifest.

## Reversibility

**Type-2 for tiers and reporting** (tables and statuses can be
recomputed); **Type-1 for admission verdicts already committed** — an
admission report, once merged, is immutable evidence (I8) and can only
be superseded by a new report under a new task version, never edited.

## Validation criteria

v1.0 ships when: (1) the manifest generator pins the suite; (2)
co-carry's admission report is committed retroactively from its rung
archives (A1: matched reference 1.00; A2: the rung positive controls;
A3: measured against B-BLIND); (3) every leaderboard table renders from
result bundles carrying the ADR-028 provenance fields; (4) at least one
baseline campaign (ADR-011 as amended) reports under these rules
end-to-end. The protocol is Validated when an external party reproduces
one admitted task's table from the manifest + bundles alone.

## Open questions deferred to a later ADR

- Whether Gate-0-style crossover curves (takt-parameterized) become a
  standing admission artefact for time-budgeted tasks, or remain
  task-specific.
- The private-partner verification workflow (the 70/30 split's private
  members, ADR-009 as amended) — submission handling is out of v1.0
  scope.
- Whether a Tier-2 task can hold `validated` cells on one axis and
  `invalid` on another without confusing the public table (expected:
  yes, the matrix is the interface; revisit after the first two-axis
  task).

## Revision history

- **2026-07-06 — A3 ruling (founder).** The A3 admission instrument is
  the preregistered scripted/ablated partner-relevance check, as
  executed in the co-carry admission (#269: `cocarry_blind_impedance`,
  0/60 vs the scripted reference, gap CI lower 1.0 ≥ δ_min 0.20, tag
  `prereg-admission-cocarry-2026-07-05`). The learned
  B-AHT-vs-B-BLIND contrast (ADR-011 rows) is a reported per-task
  finding, not the admission gate. Rationale: under rigid physical
  coupling, masking partner observations does not blind the ego —
  partner state re-enters through the ego's own proprioception (see
  `chamber/envs/cocarry_blind_mask.py`'s documented asymmetry) — so
  the learned contrast measures the value of explicit partner
  observation, not partner-relevance. Co-carry's measured learned
  contrast is a null (B-BLIND 1.000 ≈ B-AHT 1.000, campaign
  2026-07-06) and is recorded as an informative null per ADR-026
  §Decision 2; on success-saturated tasks, discrimination lives in the
  preregistered secondary channels — the stress channel (B-JOINT
  co-trained pairs at ~41 N internal force vs ~102–108 N for every
  cross-play dyad) and the per-partner breakdown (`imp_lag_bounded`
  floor 0.88). Decision content of the A3 clause is amended
  accordingly; the co-carry ADMITTED status stands on its committed
  report.
- **2026-07-06 — Tier-3 registry additions (retroactive record for
  #267).** The task registry added `amr_handover_dynamic@v0` (Tier 3
  CANDIDATE, timing-coupled) and the co-insert runnable
  frontier/challenge labeling. No Decision change — Tier 3 is defined
  as documented candidates and closures.
