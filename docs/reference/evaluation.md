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

CHAMBER leaderboard entries report multi-seed runs with 95%
**cluster bootstrap** confidence intervals on every published metric.
Episodes within a seed are correlated (same partner roll-out, same
env-reset stream), so a pooled iid bootstrap understates the CI; the
implementation in
[`src/chamber/evaluation/bootstrap.py`](https://github.com/fsafaei/concerto/blob/main/src/chamber/evaluation/bootstrap.py)
resamples seeds (the cluster level) with replacement, then resamples
episodes within each resampled seed. The minimum seed counts are:

| Run class                                 | Minimum seeds | Source                                                                                       |
|-------------------------------------------|---------------|----------------------------------------------------------------------------------------------|
| Stage-1 / Stage-2 axis spike (Phase 0)    | 5             | [ADR-007 §Implementation staging](https://github.com/fsafaei/concerto/blob/main/adr/ADR-007-heterogeneity-axis-selection.md) |
| Stage-3 axis spike (Phase 0)              | 5             | ADR-007 §Implementation staging                                                              |
| Phase-1 leaderboard entry                 | 16            | [ADR-009 §partner-zoo stratum sizing](https://github.com/fsafaei/concerto/blob/main/adr/ADR-009-partner-zoo.md) |

Each metric — episode success rate, violation rate, conformal λ mean,
inter-robot-collision rate, force-limit violation rate — is reported
as point estimate with a 95% **cluster-bootstrap** CI computed across
the seed budget above. Submissions that report fewer seeds, omit the
CI, or use a pooled iid bootstrap on episode-level data are not
admitted to the leaderboard.

### 1.1 Homogeneous-vs-heterogeneous pairing

The ≥20pp gap test from
[ADR-007 §Validation criteria](https://github.com/fsafaei/concerto/blob/main/adr/ADR-007-heterogeneity-axis-selection.md)
is computed on **matched pairs**, not on pooled means.
[`chamber.evaluation.bootstrap.pacluster_bootstrap`](https://github.com/fsafaei/concerto/blob/main/src/chamber/evaluation/bootstrap.py)
takes an iterable of paired episodes — homogeneous and heterogeneous
rollouts sharing `(seed, episode_idx, initial_state_seed)` — and
resamples first at the seed (cluster) level, then within each
resampled seed at the paired-episode level. Pairing by initial state
seed plus partner seed removes the dominant source of cross-condition
variance (different initial configurations being rolled out for
homogeneous vs heterogeneous): the gap statistic is the within-pair
delta, not the cross-pool mean difference.

This page exists, in part, because Henderson et al. (2018)
[`henderson2018matters`] catalogued a set of evaluation anti-patterns
the project explicitly refuses to fall into:

- single-seed bar charts;
- mean returns without a confidence interval;
- cherry-picked checkpoints (best-of-N reporting without disclosing
  N);
- undocumented hyperparameter sweeps masked as a single "tuned"
  configuration;
- comparison against re-implementations of baselines instead of the
  baseline authors' released code.

See `henderson2018matters` in [`literature.md §5`](literature.md) for
the citation.

---

## 2. Aggregate metrics

Beyond mean ± 95% CI, the CHAMBER leaderboard reports the
rliable-style robust aggregate metrics introduced by Agarwal et al.
(2021) [`agarwal2021precipice`]: interquartile mean (IQM),
optimality gap, and performance profiles. IQM is the median of the middle 50% of scores, robust to
outliers in either tail; optimality gap reports the proportion of the
score distribution below a target threshold; performance profiles
visualise the full score distribution as a CDF, surfacing dispersion
that point estimates hide.

The rliable contract is pinned in
[`ADR-014`](https://github.com/fsafaei/concerto/blob/main/adr/ADR-014-safety-reporting.md)
§Decision and mirrored verbatim here so that the two documents stay
in sync (this page is §3 in the `docs/reference/` outline; §3.1 below
is the seed-count table in §1, and the present subsection is §3.2):

> Aggregate metrics across seeds use rliable-style robust statistics
> (Agarwal et al. 2021): interquartile mean, optimality gap, and
> bootstrap performance profiles, in addition to mean ± 95% bootstrap
> CI. The minimum-seed count per cell is the figure committed in
> `docs/reference/evaluation.md` §3.1. This is the explicit avoidance
> of the reporting anti-patterns catalogued by Henderson et al. 2018.

Henderson et al. (2018) is catalogued in §1's anti-pattern list above.

The Phase-1 leaderboard renderer `chamber-render-tables` *must* emit
a per-axis IQM column when the `rliable` package is available as an
optional dependency, and *should* additionally emit optimality-gap
and performance-profile artefacts in the same run. The renderer is
the contractual surface — its CLI flags and output schema are the
implementation work that closes this contract, scheduled as a
Phase-1 follow-up (see
[`ADR-014`](https://github.com/fsafaei/concerto/blob/main/adr/ADR-014-safety-reporting.md)
for the three-table-format scaffold the renderer fills in).

[`chamber.evaluation.bootstrap.aggregate_metrics`](https://github.com/fsafaei/concerto/blob/main/src/chamber/evaluation/bootstrap.py)
computes IQM and optimality gap natively (rliable-compatible
definitions) so the leaderboard remains renderable without the
optional extra; performance profiles delegate to `rliable` when the
extra is installed and return `None` with a `RuntimeWarning`
otherwise.

See `agarwal2021precipice` in [`literature.md §5`](literature.md)
for the citation.

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
evaluation-comparison framework is Jordan et al. (2020)
[`jordan2020evaluating`]; see also
[`literature.md §5`](literature.md).

The pre-registration YAML template — including the seed list, the
hypothesis, the analysis formula, and the threshold — lives in
[`spikes/preregistration/`](https://github.com/fsafaei/concerto/tree/main/spikes/preregistration);
see the
[run-spike how-to](../how-to/run-spike.md) for the operational flow.

---

## 3.4 Stewardship and machine-readability

> Beyond per-experiment statistics, CHAMBER artifacts (datasets,
> policies, leaderboard entries) follow the FAIR principles for
> scientific data management and stewardship (Wilkinson et al. 2016
> [`wilkinson2016fair`]): every artifact is Findable (Zenodo DOI on
> every release), Accessible (Apache-2.0 licence + public mirror),
> Interoperable (uv.lock-pinned dependencies + SCHEMA_VERSION-pinned
> wire format), and Reusable (CITATION.cff + SBOM + this reporting
> contract).

See `wilkinson2016fair` in [`literature.md §5`](literature.md) and
the canonical entry in [`refs.bib`](refs.bib) for the bibliographic
record.
