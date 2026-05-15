# SPDX-License-Identifier: Apache-2.0
"""``chamber-eval`` console entry point (ADR-007 §Discipline, ADR-008 §Decision).

Pipeline: load one or more spike-run JSON archives, build per-axis
condition results from the episodes, bootstrap the gaps, compute the
HRS vector + scalar over the full set of axes, and emit the
leaderboard entry. The renderer is in :mod:`chamber.evaluation.render`;
``chamber-eval`` is the orchestration shim.

ADR-008 §Decision binds the HRS bundle to the *surviving ADR-007
axes* — i.e. a leaderboard row is only complete once the spikes that
cover the surviving-axis set have all been evaluated. Reviewer P1-3
flagged that the original single-spike CLI silently produced a one-
axis "leaderboard" that read as a headline number but wasn't. The
multi-spike invocation here is the fix: callers pass every spike-run
archive in one ``chamber-eval`` call, and the renderer prefixes
single-spike entries with ``[PARTIAL: <axis>]`` so they are never
mistaken for a complete row (see
:func:`chamber.evaluation.render.render_leaderboard`).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import chamber
from chamber.evaluation import (
    ConditionResult,
    LeaderboardEntry,
    SpikeRun,
    cluster_bootstrap,
    compute_hrs_scalar,
    compute_hrs_vector,
    pacluster_bootstrap,
    render_leaderboard,
)
from chamber.evaluation.bootstrap import build_paired_episodes
from chamber.evaluation.hrs import DEFAULT_AXIS_WEIGHTS
from concerto.training.seeding import derive_substream

if TYPE_CHECKING:
    import numpy as np


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="chamber-eval",
        description=(
            "Run the ADR-008 evaluation pipeline on one or more spike-run "
            "JSON archives and emit the leaderboard entry. Pass one path "
            "per surviving ADR-007 axis."
        ),
    )
    parser.add_argument(
        "spike_runs",
        type=Path,
        nargs="*",
        help=(
            "Paths to spike_run.json archives (one SpikeRun per file). "
            "Pass one path per surviving axis to compute the full HRS "
            "vector; passing a single path emits a PARTIAL row."
        ),
    )
    parser.add_argument(
        "--method-id",
        type=str,
        default="concerto",
        help="Method identifier for the resulting leaderboard entry.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Root seed for the bootstrap rng (ADR-002 P6 determinism).",
    )
    parser.add_argument(
        "--n-resamples",
        type=int,
        default=2000,
        help="Number of bootstrap resamples (default: 2000).",
    )
    parser.add_argument(
        "--allow-duplicate-axes",
        action="store_true",
        help=(
            "Allow multiple spike-run archives that target the same ADR-007 "
            "axis. When set, axis names are suffixed with the spike_id for "
            "disambiguation in the rendered output. Without this flag, "
            "duplicate axes cause the CLI to exit with status 2."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write the leaderboard entry JSON here; print Markdown otherwise.",
    )
    return parser.parse_args(argv)


def _condition_result_for(
    run: SpikeRun,
    *,
    rng: np.random.Generator,
    n_resamples: int,
    axis_key: str,
) -> ConditionResult:
    pairs = build_paired_episodes(run)
    gap_ci = pacluster_bootstrap(pairs, n_resamples=n_resamples, rng=rng)

    by_seed_homo: dict[int, list[float]] = {}
    by_seed_hetero: dict[int, list[float]] = {}
    for ep in run.episode_results:
        condition = ep.metadata.get("condition")
        score = 1.0 if ep.success else 0.0
        if condition == run.condition_pair.homogeneous_id:
            by_seed_homo.setdefault(ep.seed, []).append(score)
        elif condition == run.condition_pair.heterogeneous_id:
            by_seed_hetero.setdefault(ep.seed, []).append(score)

    homo_ci = cluster_bootstrap(by_seed_homo, n_resamples=n_resamples, rng=rng)
    hetero_ci = cluster_bootstrap(by_seed_hetero, n_resamples=n_resamples, rng=rng)

    n_total = len(run.episode_results)
    violations = sum(1 for ep in run.episode_results if ep.constraint_violation_peak > 0.0)
    fallbacks = sum(1 for ep in run.episode_results if ep.fallback_fired > 0)

    return ConditionResult(
        axis=axis_key,
        n_episodes=n_total,
        homogeneous_success=max(0.0, homo_ci.iqm),
        heterogeneous_success=max(0.0, hetero_ci.iqm),
        gap_pp=gap_ci.iqm * 100.0,
        ci_low_pp=gap_ci.ci_low * 100.0,
        ci_high_pp=gap_ci.ci_high * 100.0,
        violation_rate=violations / n_total if n_total else 0.0,
        fallback_rate=fallbacks / n_total if n_total else 0.0,
    )


def _build_axis_keys(
    runs: list[SpikeRun],
    *,
    allow_duplicate_axes: bool,
) -> list[tuple[str, str, SpikeRun]]:
    """Resolve per-run axis keys, honouring the duplicate-axis flag.

    Returns a list of ``(base_axis, axis_key, run)`` triples preserving
    the user-supplied run order. ``axis_key`` equals ``base_axis``
    unless the axis appears more than once across ``runs`` and
    ``allow_duplicate_axes`` is set — in which case the key is
    ``f"{base_axis}_{run.spike_id}"`` for disambiguation
    (reviewer P1-3).
    """
    counts: dict[str, int] = {}
    for run in runs:
        counts[run.axis] = counts.get(run.axis, 0) + 1
    duplicated = {axis for axis, n in counts.items() if n > 1}
    if duplicated and not allow_duplicate_axes:
        offenders = ", ".join(sorted(duplicated))
        msg = (
            f"error: duplicate axis IDs across spike runs ({offenders}); "
            "pass --allow-duplicate-axes to keep both and suffix the axis "
            "name with the spike_id."
        )
        raise ValueError(msg)
    triples: list[tuple[str, str, SpikeRun]] = []
    for run in runs:
        key = f"{run.axis}_{run.spike_id}" if run.axis in duplicated else run.axis
        triples.append((run.axis, key, run))
    return triples


def _weights_for(keys_in_order: list[tuple[str, str]]) -> dict[str, float]:
    """Build a per-key weights dict that preserves ADR-008 ordering.

    ``keys_in_order`` is a list of ``(base_axis, axis_key)`` tuples in
    the user-supplied order. The returned dict iterates in ADR-008
    Option D order (``CM > PF > CR > SA > OM > AS``), with any
    duplicated-axis keys placed consecutively at the position of
    their base axis. Unknown base axes (defensive only — the spike
    runner restricts to the ADR-007 shortlist) fall back to weight 0.5.
    """
    by_base: dict[str, list[str]] = {}
    for base, key in keys_in_order:
        by_base.setdefault(base, []).append(key)
    weights: dict[str, float] = {}
    for base in DEFAULT_AXIS_WEIGHTS:
        for key in by_base.get(base, []):
            weights[key] = DEFAULT_AXIS_WEIGHTS[base]
    for base, key_list in by_base.items():
        if base in DEFAULT_AXIS_WEIGHTS:
            continue
        for key in key_list:
            weights[key] = 0.5
    return weights


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``chamber-eval`` console script (ADR-008 §Decision).

    Args:
        argv: Optional list of CLI arguments (testing hook).

    Returns:
        ``0`` on success; ``2`` when a spike-run JSON is missing or
        duplicate axes are passed without ``--allow-duplicate-axes``.
    """
    args = _parse_args(argv)
    if not args.spike_runs:
        print(f"chamber-eval  (CHAMBER {chamber.__version__})")
        print(
            "Usage: chamber-eval SPIKE_RUN.json [SPIKE_RUN.json ...] "
            "[--method-id ID] [--allow-duplicate-axes] [--output OUT.json]"
        )
        return 0

    for path in args.spike_runs:
        if not path.exists():
            print(f"error: {path} not found", file=sys.stderr)
            return 2

    runs = [
        SpikeRun.model_validate_json(path.read_text(encoding="utf-8")) for path in args.spike_runs
    ]
    try:
        triples = _build_axis_keys(runs, allow_duplicate_axes=args.allow_duplicate_axes)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    rng = derive_substream("chamber.evaluation.bootstrap", root_seed=args.seed).default_rng()
    conditions: dict[str, ConditionResult] = {}
    for _, key, run in triples:
        conditions[key] = _condition_result_for(
            run, rng=rng, n_resamples=args.n_resamples, axis_key=key
        )

    weights = _weights_for([(base, key) for base, key, _ in triples])
    vector = compute_hrs_vector(conditions, weights=weights)
    scalar = compute_hrs_scalar(vector)

    total_eps = sum(c.n_episodes for c in conditions.values())
    if total_eps > 0:
        agg_violation = (
            sum(c.violation_rate * c.n_episodes for c in conditions.values()) / total_eps
        )
        agg_fallback = sum(c.fallback_rate * c.n_episodes for c in conditions.values()) / total_eps
    else:
        agg_violation = 0.0
        agg_fallback = 0.0

    entry = LeaderboardEntry(
        method_id=args.method_id,
        spike_runs=[run.spike_id for run in runs],
        hrs_vector=vector,
        hrs_scalar=max(0.0, scalar),
        violation_rate=agg_violation,
        fallback_rate=agg_fallback,
    )

    if args.output is not None:
        args.output.write_text(entry.model_dump_json(indent=2), encoding="utf-8")
        return 0
    print(render_leaderboard([entry]))
    return 0
