# SPDX-License-Identifier: Apache-2.0
"""CHAMBER evaluation harness — HRS bundle, leaderboard, safety reports.

Implements the HRS bundle composition (Option D, axis-survival rule) from
ADR-008 and consumes the three-table safety reporting format from ADR-014.
Statistical tooling (cluster + paired-cluster bootstrap) and the
pre-registration discipline (ADR-007 §Discipline) live in sub-modules of
this package.

Public modules:

- :mod:`chamber.evaluation.results` — Pydantic schema for spike runs and
  leaderboard entries (ADR-008 §Decision).
- :mod:`chamber.evaluation.prereg` — YAML loader + git-tag SHA
  verification (ADR-007 §Discipline).
- :mod:`chamber.evaluation.bootstrap` — cluster + paired-cluster bootstrap
  with rliable-compatible aggregate metrics (ADR-008 §Decision; reviewer
  P1-9).
- :mod:`chamber.evaluation.hrs` — HRS vector + scalar (ADR-008 §Decision;
  reviewer P1-8).
- :mod:`chamber.evaluation.render` — three-table safety report +
  leaderboard renderers (ADR-014 §Decision, ADR-008 §Decision).
"""

from chamber.evaluation.bootstrap import (
    BootstrapCI,
    PairedEpisode,
    aggregate_metrics,
    cluster_bootstrap,
    pacluster_bootstrap,
)
from chamber.evaluation.hrs import (
    DEFAULT_AXIS_WEIGHTS,
    compute_hrs_scalar,
    compute_hrs_vector,
)
from chamber.evaluation.oscbf_aggregate import (
    SlackAggregate,
    aggregate_oscbf_slack,
    make_condition_row,
)
from chamber.evaluation.prereg import (
    PREREG_SCHEMA_VERSION,
    PreregDocument,
    PreregistrationError,
    PreregistrationSpec,
    load_prereg,
    load_prereg_document,
    verify_git_tag,
)
from chamber.evaluation.render import (
    render_leaderboard,
    render_three_table_safety_report,
)
from chamber.evaluation.results import (
    SCHEMA_VERSION,
    BundleSummary,
    ConditionPair,
    ConditionResult,
    EpisodeResult,
    HRSVector,
    HRSVectorEntry,
    LeaderboardEntry,
    PlatformFingerprint,
    ResultBundle,
    SeedSchedule,
    SpikeRun,
    load_run_archive,
)

__all__ = [
    "DEFAULT_AXIS_WEIGHTS",
    "PREREG_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "BootstrapCI",
    "BundleSummary",
    "ConditionPair",
    "ConditionResult",
    "EpisodeResult",
    "HRSVector",
    "HRSVectorEntry",
    "LeaderboardEntry",
    "PairedEpisode",
    "PlatformFingerprint",
    "PreregDocument",
    "PreregistrationError",
    "PreregistrationSpec",
    "ResultBundle",
    "SeedSchedule",
    "SlackAggregate",
    "SpikeRun",
    "aggregate_metrics",
    "aggregate_oscbf_slack",
    "cluster_bootstrap",
    "compute_hrs_scalar",
    "compute_hrs_vector",
    "load_prereg",
    "load_prereg_document",
    "load_run_archive",
    "make_condition_row",
    "pacluster_bootstrap",
    "render_leaderboard",
    "render_three_table_safety_report",
    "verify_git_tag",
]
