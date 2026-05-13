# Evaluation

CHAMBER is committed to reproducibility and statistical discipline
from the first leaderboard entry. This page records the contract:
how many seeds, which aggregate metrics, which guard rails. The
rationale anchors are
[ADR-007](https://github.com/fsafaei/concerto/blob/main/adr/ADR-007-heterogeneity-axis-selection.md)
(≥20pp gap rule for axis admission),
[ADR-009](https://github.com/fsafaei/concerto/blob/main/adr/ADR-009-partner-zoo.md)
(partner-zoo stratum sizing), and
[ADR-014](https://github.com/fsafaei/concerto/blob/main/adr/ADR-014-safety-reporting.md)
(safety-reporting tables). The peer-reviewed evidence is on the
[literature page](literature.md) §5.

---

## 1. Seeds and reporting

CHAMBER leaderboard entries report multi-seed runs with 95% bootstrap
confidence intervals on every published metric. The minimum seed
counts are:

| Run class                                 | Minimum seeds | Source                                                                                       |
|-------------------------------------------|---------------|----------------------------------------------------------------------------------------------|
| Stage-1 / Stage-2 axis spike (Phase 0)    | 5             | [ADR-007 §Implementation staging](https://github.com/fsafaei/concerto/blob/main/adr/ADR-007-heterogeneity-axis-selection.md) |
| Stage-3 axis spike (Phase 0)              | 5             | ADR-007 §Implementation staging                                                              |
| Phase-1 leaderboard entry                 | 16            | [ADR-009 §partner-zoo stratum sizing](https://github.com/fsafaei/concerto/blob/main/adr/ADR-009-partner-zoo.md) |

Each metric — episode success rate, violation rate, conformal λ mean,
inter-robot-collision rate, force-limit violation rate — is reported
as point estimate with a 95% bootstrap CI computed across the seed
budget above. Submissions that report fewer seeds or omit the CI are
not admitted to the leaderboard.

This page exists, in part, because Henderson et al. (2018) catalogued
a set of evaluation anti-patterns the project explicitly refuses to
fall into:

- single-seed bar charts;
- mean returns without a confidence interval;
- cherry-picked checkpoints (best-of-N reporting without disclosing
  N);
- undocumented hyperparameter sweeps masked as a single "tuned"
  configuration;
- comparison against re-implementations of baselines instead of the
  baseline authors' released code.

See `henderson2018drl_matters` in [`literature.md §5`](literature.md)
for the citation.

---

## 2. Aggregate metrics

Beyond mean ± 95% CI, the CHAMBER leaderboard reports the
rliable-style robust aggregate metrics introduced by Agarwal et al.
(2021): interquartile mean (IQM), optimality gap, and performance
profiles. IQM is the median of the middle 50% of scores, robust to
outliers in either tail; optimality gap reports the proportion of the
score distribution below a target threshold; performance profiles
visualise the full score distribution as a CDF, surfacing dispersion
that point estimates hide.

The Phase-1 leaderboard renderer `chamber-render-tables` *must* emit
a per-axis IQM column when the `rliable` package is available as an
optional dependency, and *should* additionally emit optimality-gap
and performance-profile artefacts in the same run. The renderer is
the contractual surface — its CLI flags and output schema are the
implementation work that closes this contract, scheduled as a
Phase-1 follow-up (see
[`ADR-014`](https://github.com/fsafaei/concerto/blob/main/adr/ADR-014-safety-reporting.md)
for the three-table-format scaffold the renderer fills in).

See `agarwal2021rliable` in [`literature.md §5`](literature.md) for
the citation.

---

## 3. Pre-registration and statistical guard rails

Every Phase-0 axis spike runs against a pre-registration YAML
committed to `spikes/preregistration/` *before* launch. The YAML
fixes: the hypothesis, the homogeneous baseline pair, the
heterogeneous condition, the seed list, the metric, the analysis
formula, and the ≥20pp gap threshold that decides admission to the
v1 benchmark. Editing the YAML after a spike has launched is a
project anti-pattern (see ADR-007 §Validation criteria); the
corrective action is to re-launch with a new YAML and a new git tag.

The ≥20pp gap rule from
[ADR-007 §Validation criteria](https://github.com/fsafaei/concerto/blob/main/adr/ADR-007-heterogeneity-axis-selection.md)
is the project's binary admission criterion: an axis survives Phase
0 if it produces a ≥20 percentage-point gap in episode success rate
between homogeneous and heterogeneous agent pairs on at least one
benchmark scenario, measured at the seed budget above. The 20pp
threshold and the seed budget jointly determine the **minimum
detectable effect size (MDE)** the spike is powered to find. With a
binary success metric, 5 seeds across 100 evaluation episodes each
gives roughly 500 paired trials, sufficient to discriminate a 20pp
gap from null at standard significance levels; 16 seeds at the
Phase-1 sample size tightens the MDE substantially and is the level
required for leaderboard admission. The underlying
evaluation-comparison framework is Jordan et al. (2020); see
`jordan2020evaluating_rl` in [`literature.md §5`](literature.md).

The pre-registration YAML template — including the seed list, the
hypothesis, the analysis formula, and the threshold — lives in
[`spikes/preregistration/`](https://github.com/fsafaei/concerto/tree/main/spikes/preregistration);
see the
[run-spike how-to](../how-to/run-spike.md) for the operational flow.
