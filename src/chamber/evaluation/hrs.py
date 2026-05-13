# SPDX-License-Identifier: Apache-2.0
"""Heterogeneity-Robustness Score (HRS) vector + scalar (ADR-008 §Decision).

ADR-008 §Decision binds the HRS bundle to the surviving ADR-007
axes and fixes the default ordering ``CM > PF > CR > SA > OM > AS``
for headline reporting. Reviewer P1-8 added the requirement that the
*vector* be emitted alongside the scalar so consumers can recompute
the scalar under a different weighting without re-running the
spikes. The leaderboard renderer enforces this contract by refusing
entries that carry only the scalar (see
:func:`chamber.evaluation.render.render_leaderboard`).

The default weights below match ADR-008 Option D ordering: the
most-distinctive axis carries the largest weight, the least-
distinctive the smallest. The weights are exposed as a parameter so
a post-spike re-weighting can be applied without a code change; any
post-hoc reweighting that ships in the leaderboard requires a new
ADR per ADR-008 §Open questions.
"""

from __future__ import annotations

from chamber.evaluation.results import (
    ConditionResult,
    HRSVector,
    HRSVectorEntry,
)

#: ADR-008 §Decision Option D ordering (CM > PF > CR > SA > OM > AS).
#:
#: Weights linearly interpolate between 1.0 (most distinctive) and
#: 0.5 (least distinctive) across the six-axis shortlist.
DEFAULT_AXIS_WEIGHTS: dict[str, float] = {
    "CM": 1.0,
    "PF": 0.9,
    "CR": 0.8,
    "SA": 0.7,
    "OM": 0.6,
    "AS": 0.5,
}


def compute_hrs_vector(
    condition_results: dict[str, ConditionResult],
    *,
    weights: dict[str, float] | None = None,
) -> HRSVector:
    """Compute the per-axis HRS vector (ADR-008 §Decision; reviewer P1-8).

    The per-axis score is the heterogeneous-side success rate
    (normalised to ``[0, 1]``). Axes missing from
    ``condition_results`` are *not* emitted as zero-score entries —
    ADR-007's axis-survival rule means missing axes are cut from the
    headline, not silently scored as failure.

    Args:
        condition_results: ``{axis: ConditionResult}`` covering the
            surviving ADR-007 axes.
        weights: Override the per-axis weight map; defaults to
            :data:`DEFAULT_AXIS_WEIGHTS` (ADR-008 §Decision Option D).

    Returns:
        :class:`HRSVector` with one entry per axis in the canonical
        ADR-008 ordering (CM > PF > CR > SA > OM > AS), filtered to
        the axes present in ``condition_results``.
    """
    chosen_weights = weights if weights is not None else DEFAULT_AXIS_WEIGHTS
    ordered_axes = [a for a in chosen_weights if a in condition_results]
    entries = [
        HRSVectorEntry(
            axis=axis,
            score=float(condition_results[axis].heterogeneous_success),
            gap_pp=float(condition_results[axis].gap_pp),
            weight=float(chosen_weights[axis]),
        )
        for axis in ordered_axes
    ]
    return HRSVector(entries=entries)


def compute_hrs_scalar(
    vector: HRSVector,
    *,
    weights: dict[str, float] | None = None,
) -> float:
    """Aggregate the HRS vector into a single scalar (ADR-008 §Decision).

    Defined as ``sum(w_i * s_i) / sum(w_i)`` over the entries in
    ``vector`` — a weighted mean over the surviving axes, with the
    ADR-008 Option D ordering as default. Returns ``0.0`` for an
    empty vector (no surviving axes), which makes a method that
    failed every ≥20pp test sort to the bottom of the leaderboard.

    Args:
        vector: Per-axis HRS values from :func:`compute_hrs_vector`.
        weights: Override the per-axis weight map; if ``None``, the
            weights embedded in ``vector.entries`` are used (so the
            scalar matches the vector that ships with it).

    Returns:
        Scalar in ``[0, 1]``.
    """
    if not vector.entries:
        return 0.0
    if weights is None:
        weight_seq = [e.weight for e in vector.entries]
    else:
        weight_seq = [weights[e.axis] for e in vector.entries]
    score_seq = [e.score for e in vector.entries]
    weight_total = sum(weight_seq)
    if weight_total <= 0.0:
        return 0.0
    return sum(w * s for w, s in zip(weight_seq, score_seq, strict=True)) / weight_total


__all__ = [
    "DEFAULT_AXIS_WEIGHTS",
    "compute_hrs_scalar",
    "compute_hrs_vector",
]
