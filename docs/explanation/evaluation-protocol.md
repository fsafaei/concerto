# The evaluation protocol

This page explains how CHAMBER-Bench v1.0 produces a number you can
trust: which runs count, how a task earns the right to be scored, how
results are reported, and how anything can change without silently
invalidating what came before. The normative source is
[ADR-027](../reference/adrs.md) (protocol) with
[ADR-026](../reference/adrs.md) (coupling validity) and
[ADR-028](../reference/adrs.md) (result bundles); this page is the
plain-language walkthrough with the committed evidence linked in
place.

## Run classes: preregistered leaderboard runs vs exploratory runs

A **preregistered run** is one whose full specification — task,
method, partner set, seeds, episode counts, estimator, and decision
thresholds — was committed to the repository and locked with a signed
git tag *before* the first episode ran. The specification file is a
preregistration YAML (for the current campaigns:
`spikes/preregistration/benchmark/cocarry_baselines_v1.yaml`, tag
`prereg-cocarry-baselines-v1-rev2-2026-07-05`, and
`spikes/preregistration/benchmark/handover_baselines_v1.yaml`, tag
`prereg-handover-baselines-v1-2026-07-06`). `chamber-eval run
--prereg <yaml>` refuses to start if the on-disk file disagrees with
the blob recorded at the tag, and the produced bundle records both
the tag and the blob hash, so a reader can prove after the fact that
the plan predates the data.

Only preregistered runs may appear on the leaderboard, and only if
they are listed in the per-task manifest
(`spikes/results/benchmark/<task>/LEADERBOARD_BUNDLES.txt`) — bundles
are listed explicitly, never discovered by glob, so a stray run
cannot drift into the table.

Everything else is an **exploratory run**: power pilots that size the
campaign (committed as `power-pilot-*` bundles with
`run_purpose: power` and excluded from every manifest), debugging
runs, and unpreregistered experiments. Exploratory runs may be
committed as evidence but never rendered as leaderboard rows.

## The admission protocol: A1, A2, A3

CHAMBER-Bench scores cooperation, so a task must first prove that it
*requires* cooperation. A task enters Tier 2 (the scored tier) only
by a committed **admission report** produced by `chamber-eval
admission` under a tag-locked preregistration, showing three things
under preregistered thresholds:

- **A1 — solvability.** A scripted reference pair meets a
  preregistered success threshold, with the physical-stress channel
  within limits. This proves failures elsewhere in the table are the
  method's fault, not the task's.
- **A2 — two-robot infeasibility.** The best single-robot (partner-
  ablated) variant achieves approximately zero success. This proves
  the second robot is load-bearing, not decorative.
- **A3 — partner-relevance.** A scripted ego stripped of exactly its
  coupling channel underperforms the coupling-aware scripted
  reference by a preregistered margin. This proves the *interaction*
  matters, not just the second robot's presence. A task that fails
  A3 is demoted to Tier 1 (controls).

### Worked example: the co-carry admission

Evidence: `spikes/results/admission/cocarry-2026-07-05/`
(`ADMISSION_REPORT.md` + machine-readable `admission_report.json` +
one verifiable sub-bundle per check), preregistration tag
`prereg-admission-cocarry-2026-07-05`. Verdict: **ADMITTED**.

- **A1 PASS.** The scripted impedance reference pair succeeded in
  60/60 episodes (success confidence interval [1.00, 1.00] against
  threshold ≥ 0.95), with a worst-episode wrist-force peak of
  107.9 N against the task's canonical stress limit of 130.57 N.
- **A2 PASS.** The single-arm variant succeeded in 0/60 episodes
  (confidence-interval upper bound 0.00 against threshold ≤ 0.05) —
  one arm cannot carry the bar.
- **A3 PASS.** The scripted ego with its coupling channel removed
  dropped from 60/60 to 0/60; the paired reference-minus-blind gap
  is 1.00 (confidence interval [1.00, 1.00]) against the
  preregistered minimum of 0.20.

The handover-and-place admission
(`spikes/results/admission/handover_place-2026-07-05/`, verdict
**ADMITTED**) is the non-saturated version of the same story: its A3
gap is 0.455 with confidence-interval lower bound 0.325 — well clear
of the 0.20 margin but nowhere near the co-carry ceiling.

### Falsification example: the pick-and-place rejection

The protocol is only meaningful if it can reject a task, and it has:
`spikes/results/admission/stage1_pickplace_as-2026-07-05/` (verdict
**CONTROL**, tag `prereg-admission-stage1-pickplace-as-2026-07-05`).
Pick-and-place passed A1 (the reference pair succeeds) but **failed
A2** — the single-arm variant also succeeded in 60/60 episodes, so
the partner is not load-bearing — and **failed A3** — the measured
coupling gap was 0.00 (confidence interval [0.00, 0.00]), below the
0.20 minimum. The task is retained as a Tier-1 control: useful for
separating manipulation skill from cooperation, and never scored as
cooperation. This is the coupling-validity criterion of ADR-026
doing its job; the Stage-1 action-space result on this task is
thereby *uninterpretable for the cooperation thesis*, which is why
the axis-validity column of the task table marks it `invalid`.

One consequence, stated plainly: results measured on admitted tasks
that come out null are reported as **informative nulls**, never
re-described as task failures.

## Reporting rules

**The estimator is the interquartile mean (IQM).** Sort the
per-episode success indicators, drop the bottom 25% and the top 25%,
and average the middle half. The IQM is robust to outliers but
intentionally conservative away from saturation: on handover-v1 the
oracle-reference row has mean success 0.338 but IQM 0.176, because
successes are concentrated in a minority of partner cells (bundle:
`spikes/results/benchmark/handover-v1/ref-script-2026-07-06`). At
saturation it trims minority cells: cocarry B-AHT has mean 0.961 but
IQM 1.000 (bundle:
`spikes/results/benchmark/cocarry-v1/b-aht-2026-07-06`). The rendered
tables therefore always show mean and per-partner range next to the
IQM.

**Uncertainty is a 95% cluster bootstrap over seeds.** Episodes from
one training/evaluation seed are correlated, so resampling episodes
individually would understate uncertainty. The cluster bootstrap
resamples whole seeds (2000 resamples), and the resampling stream is
derived deterministically from a recorded root seed, so `chamber-eval
verify` recomputes the interval byte-identically.

**Per-partner breakdown is mandatory.** A row's headline can hide a
partner it fails with; the tables carry the per-partner minimum and
maximum (on cocarry-v1, B-AHT spans 0.88 with `imp_lag_bounded` up
to 1.00 — the discriminating cell at an otherwise saturated
headline).

**No single-seed claims.** Leaderboard rows pool five seeds; a
one-seed result is not a reportable claim under this protocol.

**No scalar composite.** The leaderboard is per-task tables. The
heterogeneity-robustness score (HRS, ADR-008) computes only over
task×axis cells whose heterogeneity effect is *validated*, and is
suspended while none exist; it resumes automatically with its first
validated cell.

**Honest labels.** REF-SCRIPT rows are the *oracle reference* — a
scripted controller with privileged access, shown as the solvability
ceiling, never a baseline. B-JOINT rows are the *non-AHT upper
anchor* — a jointly-trained pair evaluated with its own training
partner, which violates the unseen-partner condition by construction
and anchors the non-ad-hoc ceiling.

## The checkpoint-selection rule

Learned rows (B-AHT, B-BLIND, B-JOINT) must not cherry-pick
checkpoints after seeing evaluation results, so selection is
preregistered: **per seed, select the checkpoint with the highest
stress-compliant success on a held-out validation partner** — a
partner excluded from every evaluation cell (`imp_nominal` on
cocarry, hence `--exclude-member imp_nominal` in every cocarry
reproduction command). The selection artifacts are committed *before*
the evaluation cells run
(`spikes/results/benchmark/cocarry-v1/selection/`), and the
preregistration includes an instability rule: any
consecutive-checkpoint drop larger than 40 points on the validation
partner is reported, not silently absorbed. The cocarry campaign
tripped that rule on every learned (row, seed) pair; the events and
selected steps are recorded in
`spikes/results/benchmark/cocarry-v1/CAMPAIGN_REPORT.md`.

## Task and partner versioning

Tasks and partner sets are versioned as `name@vN`, and a version is
never mutated: any change to a task's physics, success predicate, or
stress instrument creates `@v(N+1)`, and old results stay
interpretable under their version. Each task version pins one
canonical success predicate and one canonical stress instrument —
for `cocarry@v1` the wrist-force instrument with its 130.57 N limit,
frozen in
`spikes/results/cocarry/rung2/cocarry_rung2_freeze_manifest.json`.
Partner sets version the same way (`cocarry_partners@v1` has 11
members; `@v2` added one admitted learned member), each member
carrying an identity hash so a bundle proves exactly which partners
it was evaluated against. Admission reports and result bundles are
immutable once merged: a verdict can be superseded by a new report
under a new task version, never edited.

## Where to go next

- Re-run any leaderboard row:
  [Reproduce the results](../how-to/reproduce-results.md).
- Put your method on the board:
  [Submit a leaderboard entry](../how-to/submit-leaderboard.md).
- What a result bundle contains and what `verify` checks:
  [Evaluation reference](../reference/evaluation.md) and ADR-028.
- The released data artifacts: [Datasheet](../reference/datasheet.md).
