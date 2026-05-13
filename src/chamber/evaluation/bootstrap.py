# SPDX-License-Identifier: Apache-2.0
"""Cluster + paired-cluster bootstrap utilities (ADR-008 §Decision; reviewer P1-9).

ADR-008 §Decision pins rliable-style aggregate metrics (IQM,
optimality gap, performance profiles) on Table 2. Reviewer P1-9
flagged that the original iid bootstrap on episode results
understates the CI because episodes within a seed are correlated
(same partner roll-out, same env reset stream). This module ships
the cluster bootstrap (resample seeds with replacement, then
resample episodes within each seed) and the paired-cluster variant
(pair episodes across homogeneous-vs-heterogeneous conditions by
``(seed, episode_idx, initial_state_seed)`` so the ≥20pp gap test
is computed on matched pairs, not on pooled means).

The :func:`aggregate_metrics` entry point dispatches to the
optional ``rliable`` package if available, computing IQM and
optimality gap natively otherwise. Performance profiles require the
full ``rliable`` extra — when missing, a runtime warning explains
how to install it without failing the call.

The deterministic stream is sourced from
:mod:`concerto.training.seeding`: callers MUST pass an explicit
``np.random.Generator``. Library code never constructs ``default_rng``
of its own (P6 determinism; ADR-002 §"deterministic seeding harness").
"""

from __future__ import annotations

import importlib.util
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class BootstrapCI:
    """Bootstrap point estimate + 95% CI (ADR-008 §Decision).

    Attributes:
        iqm: Interquartile mean of the resampled point estimates.
        mean: Plain arithmetic mean of the resampled point estimates.
        ci_low: 2.5th percentile of the resampled estimates (lower
            bound of the 95% CI).
        ci_high: 97.5th percentile of the resampled estimates (upper
            bound of the 95% CI).
        n_resamples: Number of bootstrap resamples that produced the
            interval.
    """

    iqm: float
    mean: float
    ci_low: float
    ci_high: float
    n_resamples: int


def _interquartile_mean(values: np.ndarray) -> float:
    """IQM — mean of the middle 50% of ``values`` (ADR-014 §Decision).

    Matches the rliable implementation: drop the bottom 25% and the
    top 25% of values, then take the mean of the remainder. For a
    sample of size ``n``, drops ``floor(n/4)`` from each tail.
    """
    if values.size == 0:
        return float("nan")
    sorted_values = np.sort(values)
    lo = int(np.floor(sorted_values.size * 0.25))
    hi = sorted_values.size - lo
    middle = sorted_values[lo:hi] if hi > lo else sorted_values
    return float(np.mean(middle))


def _resample_cluster(
    values: Mapping[int, Sequence[float]],
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    """Resample seeds (clusters), then episodes within each resampled seed."""
    seeds = list(values.keys())
    if not seeds:
        return np.empty(0, dtype=np.float64)
    seed_idx = rng.integers(low=0, high=len(seeds), size=len(seeds))
    resampled: list[float] = []
    for i in seed_idx:
        episodes = np.asarray(values[seeds[i]], dtype=np.float64)
        if episodes.size == 0:
            continue
        ep_idx = rng.integers(low=0, high=episodes.size, size=episodes.size)
        resampled.extend(episodes[ep_idx].tolist())
    return np.asarray(resampled, dtype=np.float64)


def cluster_bootstrap(
    values: Mapping[int, Sequence[float]],
    *,
    n_resamples: int,
    rng: np.random.Generator,
) -> BootstrapCI:
    """Cluster bootstrap over ``{seed: [per-episode values]}`` (reviewer P1-9; ADR-008 §Decision).

    Resamples seeds with replacement (the cluster level), then
    resamples episodes with replacement within each resampled seed.
    Returns IQM, plain mean, and the 95% CI on the *IQM* estimator.
    This is the headline aggregator for ADR-014 §Decision Table 2
    when the bootstrap method is ``"cluster"``.

    Args:
        values: ``{seed: [per-episode metric]}`` mapping. Empty
            seeds are dropped from the resample but a fully empty
            input yields a degenerate ``nan`` interval.
        n_resamples: Number of bootstrap resamples (e.g. ``2000``).
        rng: Deterministic ``np.random.Generator`` (ADR-002 P6).

    Returns:
        :class:`BootstrapCI` with the IQM, plain mean, and 95% bounds.
    """
    if n_resamples <= 0:
        msg = f"n_resamples must be positive, got {n_resamples}"
        raise ValueError(msg)

    all_values = np.concatenate(
        [np.asarray(v, dtype=np.float64) for v in values.values() if len(v) > 0]
        or [np.array([], dtype=np.float64)]
    )
    point_iqm = _interquartile_mean(all_values)
    point_mean = float(np.mean(all_values)) if all_values.size else float("nan")

    iqm_samples = np.empty(n_resamples, dtype=np.float64)
    for k in range(n_resamples):
        resample = _resample_cluster(values, rng=rng)
        iqm_samples[k] = _interquartile_mean(resample) if resample.size else np.nan

    finite = iqm_samples[np.isfinite(iqm_samples)]
    if finite.size == 0:
        return BootstrapCI(
            iqm=point_iqm,
            mean=point_mean,
            ci_low=float("nan"),
            ci_high=float("nan"),
            n_resamples=n_resamples,
        )
    ci_low = float(np.percentile(finite, 2.5))
    ci_high = float(np.percentile(finite, 97.5))
    return BootstrapCI(
        iqm=point_iqm,
        mean=point_mean,
        ci_low=ci_low,
        ci_high=ci_high,
        n_resamples=n_resamples,
    )


@dataclass(frozen=True)
class PairedEpisode:
    """One paired homogeneous-vs-heterogeneous episode (ADR-007 §Validation criteria).

    Pairs are matched on ``(seed, episode_idx, initial_state_seed)``
    so the gap test in :func:`pacluster_bootstrap` is computed on
    matched pairs rather than on pooled means (reviewer P1-9).

    Attributes:
        seed: Shared root seed.
        episode_idx: Shared episode index within the seed.
        initial_state_seed: Shared env-reset sub-stream seed.
        homogeneous: Per-episode metric on the homogeneous side.
        heterogeneous: Per-episode metric on the heterogeneous side.
    """

    seed: int
    episode_idx: int
    initial_state_seed: int
    homogeneous: float
    heterogeneous: float


def _resample_paired_cluster(
    by_seed: Mapping[int, Sequence[PairedEpisode]],
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    """Resample seeds, then matched episodes within seed; return per-pair deltas."""
    seeds = list(by_seed.keys())
    if not seeds:
        return np.empty(0, dtype=np.float64)
    seed_idx = rng.integers(low=0, high=len(seeds), size=len(seeds))
    deltas: list[float] = []
    for i in seed_idx:
        episodes = list(by_seed[seeds[i]])
        if not episodes:
            continue
        ep_idx = rng.integers(low=0, high=len(episodes), size=len(episodes))
        deltas.extend(episodes[j].homogeneous - episodes[j].heterogeneous for j in ep_idx)
    return np.asarray(deltas, dtype=np.float64)


def pacluster_bootstrap(
    pairs: Iterable[PairedEpisode],
    *,
    n_resamples: int,
    rng: np.random.Generator,
) -> BootstrapCI:
    """Paired-cluster bootstrap for the ≥20pp gap test (ADR-007 §Validation criteria).

    Pairs are matched on ``(seed, episode_idx, initial_state_seed)``
    so the ``homogeneous - heterogeneous`` delta is computed on the
    *same* initial state, removing one of the dominant sources of
    cross-condition variance. The cluster level is the seed, exactly
    as in :func:`cluster_bootstrap`.

    Args:
        pairs: Iterable of :class:`PairedEpisode` records.
        n_resamples: Number of bootstrap resamples.
        rng: Deterministic ``np.random.Generator`` (ADR-002 P6).

    Returns:
        :class:`BootstrapCI` on the per-pair delta (in raw units,
        not percentage points — the renderer multiplies by 100 when
        formatting).
    """
    if n_resamples <= 0:
        msg = f"n_resamples must be positive, got {n_resamples}"
        raise ValueError(msg)

    by_seed: dict[int, list[PairedEpisode]] = {}
    for pair in pairs:
        by_seed.setdefault(pair.seed, []).append(pair)

    all_deltas = np.asarray(
        [p.homogeneous - p.heterogeneous for episodes in by_seed.values() for p in episodes],
        dtype=np.float64,
    )
    point_iqm = _interquartile_mean(all_deltas)
    point_mean = float(np.mean(all_deltas)) if all_deltas.size else float("nan")

    iqm_samples = np.empty(n_resamples, dtype=np.float64)
    for k in range(n_resamples):
        resample = _resample_paired_cluster(by_seed, rng=rng)
        iqm_samples[k] = _interquartile_mean(resample) if resample.size else np.nan

    finite = iqm_samples[np.isfinite(iqm_samples)]
    if finite.size == 0:
        return BootstrapCI(
            iqm=point_iqm,
            mean=point_mean,
            ci_low=float("nan"),
            ci_high=float("nan"),
            n_resamples=n_resamples,
        )
    return BootstrapCI(
        iqm=point_iqm,
        mean=point_mean,
        ci_low=float(np.percentile(finite, 2.5)),
        ci_high=float(np.percentile(finite, 97.5)),
        n_resamples=n_resamples,
    )


_RLIABLE_AVAILABLE = importlib.util.find_spec("rliable") is not None


def _optimality_gap(values: np.ndarray, *, threshold: float = 1.0) -> float:
    """Native optimality-gap estimator (ADR-014 §Decision; rliable-compatible).

    Defined as ``mean(max(threshold - x, 0))`` — the expected
    shortfall below ``threshold`` (typically 1.0 for normalised
    success rates). Matches the rliable definition when the
    threshold is the optimal score.
    """
    if values.size == 0:
        return float("nan")
    return float(np.mean(np.maximum(threshold - values, 0.0)))


def aggregate_metrics(
    values: Mapping[int, Sequence[float]],
    *,
    metrics: Sequence[str] = ("iqm", "optimality_gap", "performance_profile"),
    threshold: float = 1.0,
) -> dict[str, float | None]:
    """rliable-compatible aggregate metrics (ADR-008 §Decision; ADR-014 §Decision).

    If the optional ``rliable`` package is installed, delegates to
    its ``library.get_interval_estimates`` / ``aggregate_func``
    helpers. Otherwise, IQM and optimality gap are computed natively
    and a runtime warning explains that performance profiles require
    the optional extra. The native path is sufficient for unit tests
    and the Phase-0 golden-file integration test.

    Args:
        values: ``{seed: [per-episode metric]}`` mapping.
        metrics: Subset of ``{"iqm", "optimality_gap",
            "performance_profile"}`` to compute.
        threshold: Reference score for optimality gap (default
            ``1.0`` for normalised success rates).

    Returns:
        Dict keyed by metric name with float values; the
        ``"performance_profile"`` slot is ``None`` when the optional
        ``rliable`` extra is not installed.
    """
    flat = np.concatenate(
        [np.asarray(v, dtype=np.float64) for v in values.values() if len(v) > 0]
        or [np.array([], dtype=np.float64)]
    )
    out: dict[str, float | None] = {}
    if "iqm" in metrics:
        out["iqm"] = _interquartile_mean(flat)
    if "optimality_gap" in metrics:
        out["optimality_gap"] = _optimality_gap(flat, threshold=threshold)
    if "performance_profile" in metrics:
        if _RLIABLE_AVAILABLE:
            # The rliable performance-profile API returns a histogram of
            # threshold-vs-coverage; CHAMBER's renderer summarises it to
            # area-under-the-profile to fit a single leaderboard column.
            from rliable import library as rly  # type: ignore[import-not-found]  # noqa: PLC0415

            tau = np.linspace(0.0, 1.0, num=51)
            profile, _ = rly.create_performance_profile({"x": flat[None, :]}, tau)
            # Trapezoidal area under the performance profile; np.trapezoid
            # supersedes np.trapz from NumPy 2.0.
            trapezoid = getattr(np, "trapezoid", None) or np.trapz  # type: ignore[attr-defined]
            out["performance_profile"] = float(trapezoid(profile["x"], tau))
        else:
            warnings.warn(
                "performance_profile requires the optional `rliable` extra "
                "(`uv pip install rliable`); returning None for this column",
                RuntimeWarning,
                stacklevel=2,
            )
            out["performance_profile"] = None
    return out


__all__ = [
    "BootstrapCI",
    "PairedEpisode",
    "aggregate_metrics",
    "cluster_bootstrap",
    "pacluster_bootstrap",
]
