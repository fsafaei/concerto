# SPDX-License-Identifier: Apache-2.0
"""``chamber-eval`` console entry point (ADR-007 §Discipline, ADR-008 §Decision).

Pipeline: load a spike-run JSON archive, build per-axis condition
results from the episodes, bootstrap the gaps, compute the HRS
vector + scalar, and emit the leaderboard entry. The renderer is in
:mod:`chamber.evaluation.render`; ``chamber-eval`` is the
orchestration shim.
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
    PairedEpisode,
    SpikeRun,
    cluster_bootstrap,
    compute_hrs_scalar,
    compute_hrs_vector,
    pacluster_bootstrap,
    render_leaderboard,
)
from concerto.training.seeding import derive_substream

if TYPE_CHECKING:
    import numpy as np


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="chamber-eval",
        description=(
            "Run the ADR-008 evaluation pipeline on a spike-run JSON archive "
            "and emit the leaderboard entry."
        ),
    )
    parser.add_argument(
        "spike_run",
        type=Path,
        nargs="?",
        help="Path to a spike_run.json (one SpikeRun per file).",
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
        "--output",
        type=Path,
        default=None,
        help="Write the leaderboard entry JSON here; print Markdown otherwise.",
    )
    return parser.parse_args(argv)


def _build_paired_episodes(run: SpikeRun) -> list[PairedEpisode]:
    homo = run.condition_pair.homogeneous_id
    hetero = run.condition_pair.heterogeneous_id
    key_homo: dict[tuple[int, int, int], float] = {}
    key_hetero: dict[tuple[int, int, int], float] = {}
    for ep in run.episode_results:
        condition = ep.metadata.get("condition")
        key = (ep.seed, int(ep.episode_idx), ep.initial_state_seed)
        score = 1.0 if ep.success else 0.0
        if condition == homo:
            key_homo[key] = score
        elif condition == hetero:
            key_hetero[key] = score
    pairs: list[PairedEpisode] = []
    for key, h_score in key_homo.items():
        if key in key_hetero:
            pairs.append(
                PairedEpisode(
                    seed=key[0],
                    episode_idx=key[1],
                    initial_state_seed=key[2],
                    homogeneous=h_score,
                    heterogeneous=key_hetero[key],
                )
            )
    return pairs


def _condition_result_for(
    run: SpikeRun,
    *,
    rng: np.random.Generator,
    n_resamples: int,
) -> ConditionResult:
    pairs = _build_paired_episodes(run)
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
        axis=run.axis,
        n_episodes=n_total,
        homogeneous_success=max(0.0, homo_ci.iqm),
        heterogeneous_success=max(0.0, hetero_ci.iqm),
        gap_pp=gap_ci.iqm * 100.0,
        ci_low_pp=gap_ci.ci_low * 100.0,
        ci_high_pp=gap_ci.ci_high * 100.0,
        violation_rate=violations / n_total if n_total else 0.0,
        fallback_rate=fallbacks / n_total if n_total else 0.0,
    )


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``chamber-eval`` console script (ADR-008 §Decision).

    Args:
        argv: Optional list of CLI arguments (testing hook).

    Returns:
        ``0`` on success; ``2`` when the spike-run JSON is missing.
    """
    args = _parse_args(argv)
    if args.spike_run is None:
        print(f"chamber-eval  (CHAMBER {chamber.__version__})")
        print("Usage: chamber-eval SPIKE_RUN.json [--method-id ID] [--output OUT.json]")
        return 0

    if not args.spike_run.exists():
        print(f"error: {args.spike_run} not found", file=sys.stderr)
        return 2

    run = SpikeRun.model_validate_json(args.spike_run.read_text(encoding="utf-8"))
    rng = derive_substream("chamber.evaluation.bootstrap", root_seed=args.seed).default_rng()
    condition = _condition_result_for(run, rng=rng, n_resamples=args.n_resamples)

    vector = compute_hrs_vector({run.axis: condition})
    scalar = compute_hrs_scalar(vector)
    entry = LeaderboardEntry(
        method_id=args.method_id,
        spike_runs=[run.spike_id],
        hrs_vector=vector,
        hrs_scalar=max(0.0, scalar),
        violation_rate=condition.violation_rate,
        fallback_rate=condition.fallback_rate,
    )

    if args.output is not None:
        args.output.write_text(entry.model_dump_json(indent=2), encoding="utf-8")
        return 0
    print(render_leaderboard([entry]))
    return 0
