# Phase 0 results

!!! success "Phase 0 closed — 2026-05-17"
    Stage 1a (rig validation) cleared. Stage 1b is the first
    Phase-1 milestone under [ADR-007 §Stage 1b][adr-007-stage-1b]'s
    ≤4-week trigger guardrail from the Month-3 lock review. See the
    [Month-3 lock-priority report][snapshot] for the full per-stage
    evidence.

The CONCERTO project gates Phase-0 close-out on the ADR-007
§Implementation staging protocol: Stage 0 rig validation, then the
six-axis staged spike sweep (Stage 1: AS + OM; Stage 2: CR + CM;
Stage 3: PF + SA). Today's state, captured in the
[2026-05-17 Month-3 lock-priority report][snapshot] from
`chamber-spike summarize-month3` against the regenerated v2
SpikeRun archives:

> **Recommendation: Defer — Stage 1b not yet measured
> (Phase-1 milestone; guardrail ≤4 weeks)**

The recommendation correctly attributes the Phase-0 null gap signal
to Stage 1a's rig-validation framing — under
[ADR-007 §Stage 1a][adr-007-stage-1a], the ≥20 pp gate is **not**
measured against the MPE stand-in; what is exercised is the
infrastructure that Stage 1b will rely on once Phase 1 begins.

## Stage 0 — Rig validation (CLOSED)

Stage 0 is the precondition under
[ADR-001 §Validation criteria][adr-001-validation]: instantiate a
three-robot heterogeneous environment in ManiSkill v3, apply the
per-agent action-repeat and comm-shaping wrappers, run 100 control
steps without error. The 50-line acceptance test ships at
[`tests/integration/test_stage0_smoke.py`][stage0-smoke-test] under
the `@pytest.mark.smoke` marker; `make smoke` runs it. With Stage 0
green, ADR-007's staged spike protocol can proceed.

## Stage 1a — Foundation axes rig validation (CLOSED)

[ADR-007 revision 4][adr-007-stage-1a] (2026-05-15) split Stage 1 of
the §Implementation staging protocol into two sub-stages: Stage 1a
runs against
[`chamber.envs.mpe_cooperative_push.MPECooperativePushEnv`][mpe-env]
— the CPU-friendly stand-in — with `_zero_ego_action_factory` as the
production ego, while Stage 1b runs against the real ManiSkill v3
pick-place env with a trained ego. The ≥20 pp gate is **not**
measured under Stage 1a by design; its job is to exercise the rig
that Stage 1b will inherit.

What Stage 1a exercised end-to-end:

- The `chamber-spike` CLI surface — `run`, `verify-prereg`,
  `summarize-month3`, `next-stage`.
- The SpikeRun v2 wire-format schema established by
  [ADR-016][adr-016] (typed `sub_stage` field, no silent
  default-fallback).
- The audit chain anchored at every layer: `prereg_sha` +
  `git_tag` + `sub_stage` + `schema_version`, enforced at adapter
  construction by
  [`chamber.evaluation.prereg.verify_git_tag`][verify-git-tag] and
  re-asserted as a `jq` post-regeneration check in
  [`scripts/repro/stage1_as.sh`][stage1-as-script] and
  [`scripts/repro/stage1_om.sh`][stage1-om-script].

All six axis YAMLs are locked under git tags
(`prereg-stage1-{AS,OM}-2026-05-15`,
`prereg-stage2-{CR,CM}-2026-05-15`,
`prereg-stage3-{PF,SA}-2026-05-15`) and pass
`chamber-spike verify-prereg --all` end-to-end — the launch-time
tamper-detect gate that refuses any spike whose YAML's on-disk
blob SHA does not match the blob SHA stored at its tag
([ADR-007 §Discipline][adr-007]). The two Stage-1a archives consumed
for the Defer recommendation cite their tags inline (AS:
`prereg-stage1-AS-2026-05-15`; OM:
`prereg-stage1-OM-2026-05-15`). The four Stage-2 / Stage-3 tags are
committed audit anchors with their corresponding spike runs queued
for Phase 1+.

Per-axis archives — 200 episodes each (5 seeds × 20 episodes ×
2 conditions per the [evaluation-contract sample-size
table][evaluation-contract]), both at `gap_iqm_pp = n/a` because
Stage-1a doesn't measure the gate:

- AS — [`spikes/results/stage1-AS-20260517/spike_as.json`][as-archive]
- OM — [`spikes/results/stage1-OM-20260517/spike_om.json`][om-archive]

The [Month-3 lock-priority report][snapshot] folds both archives
plus the per-stage gate enumeration into the canonical Defer
recommendation. ADR-by-ADR action (per the report's table): ADR-007
held at Accepted (Stage 1b launch is the Phase-1 milestone);
ADR-008 HRS bundle held (insufficient evidence yet); ADR-011
baseline-set lock held.

## Stage 1b — Foundation axes science evaluation (PHASE 1)

[ADR-007 §Stage 1b][adr-007-stage-1b] pins the launch as the first
Phase-1 milestone, before any Stage-2 spike is permitted, no later
than 4 weeks after the Month-3 lock review. The Phase-1
prerequisites are:

- A real `chamber.envs.stage1_pickplace` env — a thin ManiSkill v3
  wrapper over the canonical pick-place task with the
  panda + fetch agent tuple, replacing the MPE stand-in.
- A trained ego per seed via
  [`concerto.training.ego_aht.train`][ego-aht-train] plugged through
  the [`EgoActionFactory`][ego-action-factory] seam, swapping
  `_zero_ego_action_factory` for the trained policy.

The pre-registration YAMLs (`AS.yaml`, `OM.yaml`) and their git
tags carry forward unchanged into Stage 1b — the same condition
IDs, seed list, and episode budget apply; only the env
implementation behind the condition IDs changes. Re-using the
canonical pre-registrations is the entire point of the sub-stage
split per [ADR-007][adr-007]; tag rotation is not allowed.

## Stage 2 — Empirical-uncertainty axes (PHASE 1, queued behind Stage 1b)

[ADR-007 §Stage 2][adr-007-stage-2] covers control rate (CR) and
communication (CM). The CR spike runs a single embodiment with two
independently-clocked control loops (500 Hz vs 50 Hz), chunk size
held constant, to isolate frequency from action-space
dimensionality. The CM spike is a latency / jitter / drop sweep on
the fixed-format channel using URLLC and 5G-TSN industrial-trial
numbers as anchor points
(latency 1–100 ms; jitter μs–10 ms; drop 10⁻⁶ to 10⁻²) per
arXiv 2501.12792 and 3GPP Release 17.

Stage 2 launch is gated on Stage 1b clearance (≥1 axis ≥20 pp); the
`chamber-spike next-stage` subcommand enforces the gate against the
Stage-1b archives that Phase-1 will produce. Both Stage-2
pre-registrations (`prereg-stage2-{CR,CM}-2026-05-15`) are already
tag-cut and tamper-detect-clean per `verify-prereg --all`; rig-side
exercise is deferred to the Stage-2 launch itself.

## Stage 3 — Systems-dependent axes (PHASE 1, queued behind Stage 2)

[ADR-007 §Stage 3][adr-007-stage-3] covers partner familiarity (PF)
and safety (SA). PF uses a trained-with vs frozen-novel partner
swap mid-episode to test conformal-slack λ re-init
(ADR-004 / ADR-006 dependency). SA runs a heterogeneous force-limit
/ SIL-PL pair on a contact-rich scenario, with HIL escalation per
ADR-006 if simulation cannot capture per-vendor controller-stack
variance under ISO 10218-2:2025. Stage 3 is per-axis cut rather
than binary halt — PF or SA failing individually is not a
project-wide stop. Both Stage-3 pre-registrations
(`prereg-stage3-{PF,SA}-2026-05-15`) are tag-cut and
tamper-detect-clean per `verify-prereg --all`; rig-side exercise is
deferred to the Stage-3 launch itself.

## Phase-1 delta — what changes between Phase-0 close and Stage 1b launch

- Build `chamber.envs.stage1_pickplace` — a thin ManiSkill v3
  wrapper over the canonical pick-place task with the
  panda + fetch agent tuple.
- Wire [`concerto.training.ego_aht.train`][ego-aht-train] through
  the [`EgoActionFactory`][ego-action-factory] seam so the Stage-1
  adapter swaps `_zero_ego_action_factory` for a trained ego per
  `(seed, condition)`. The adapter's
  `EgoActionFactory`-parameterised entry point already lands at
  this seam; what's needed is the Phase-1 trained-policy factory
  that consumes a checkpoint produced by `ego_aht.train`.
- Rerun Stage 1 against the real env; the same pre-registered AS /
  OM YAMLs apply unchanged; produce Stage-1b archives under
  `spikes/results/stage1-{AS,OM}-<date>/` with `sub_stage="1b"`.
- Re-render `chamber-spike summarize-month3`; the new top-line
  depends on the measured gaps per
  [ADR-007 §Validation criteria][adr-007] (≥20 pp on at least one
  axis transitions the recommendation away from Defer).
- (Phase-1 second wave) Stage 2 CR + CM adapters and runs, then
  Stage 3 PF + SA adapters and runs, each gated by the prior
  stage per ADR-007 §Implementation staging.
- (Phase-1 correctness debt from the 2026-05-16 external review,
  flagged by ADR-INDEX footnote (a) and several follow-ups)
  `Bounds.action_norm` field split into `action_linf_component` +
  `cartesian_accel_capacity`; `JacobianEmergencyController` real
  implementation; structural dict-keyed `lambda_` per the Option-(a)
  pair-keying refactor; position-aware joint CBFs; force-limit /
  contact-force constraints in OSCBF.

[snapshot]: _assets/month3-lock-priority-2026-05-17.md
[adr-001-validation]: https://github.com/fsafaei/concerto/blob/main/adr/ADR-001-fork-vs-build.md#validation-criteria
[adr-007]: https://github.com/fsafaei/concerto/blob/main/adr/ADR-007-heterogeneity-axis-selection.md
[adr-007-stage-1a]: https://github.com/fsafaei/concerto/blob/main/adr/ADR-007-heterogeneity-axis-selection.md#stage-1a--rig-validation-phase-0-mpe-stand-in
[adr-007-stage-1b]: https://github.com/fsafaei/concerto/blob/main/adr/ADR-007-heterogeneity-axis-selection.md#stage-1b--real-env-science-evaluation-phase-1-trigger-guardrail-below
[adr-007-stage-2]: https://github.com/fsafaei/concerto/blob/main/adr/ADR-007-heterogeneity-axis-selection.md#stage-2--empirical-uncertainty-axes-control-rate--communication
[adr-007-stage-3]: https://github.com/fsafaei/concerto/blob/main/adr/ADR-007-heterogeneity-axis-selection.md#stage-3--systems-dependent-axes-partner-familiarity--safety
[adr-016]: https://github.com/fsafaei/concerto/blob/main/adr/ADR-016-spike-run-schema.md
[as-archive]: https://github.com/fsafaei/concerto/blob/main/spikes/results/stage1-AS-20260517/spike_as.json
[om-archive]: https://github.com/fsafaei/concerto/blob/main/spikes/results/stage1-OM-20260517/spike_om.json
[stage0-smoke-test]: https://github.com/fsafaei/concerto/blob/main/tests/integration/test_stage0_smoke.py
[stage1-as-script]: https://github.com/fsafaei/concerto/blob/main/scripts/repro/stage1_as.sh
[stage1-om-script]: https://github.com/fsafaei/concerto/blob/main/scripts/repro/stage1_om.sh
[verify-git-tag]: https://github.com/fsafaei/concerto/blob/main/src/chamber/evaluation/prereg.py
[mpe-env]: https://github.com/fsafaei/concerto/blob/main/src/chamber/envs/mpe_cooperative_push.py
[ego-action-factory]: https://github.com/fsafaei/concerto/blob/main/src/chamber/benchmarks/stage1_common.py
[ego-aht-train]: https://github.com/fsafaei/concerto/blob/main/src/concerto/training/ego_aht.py
[evaluation-contract]: ../reference/evaluation.md
