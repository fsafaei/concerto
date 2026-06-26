# SPDX-License-Identifier: Apache-2.0
"""Equivalence-bounded decision rule for the Gate-0 handover-and-place spike (ADR-026).

Pure decision logic — no env, no measurement — so it is testable on synthetic cell
results and imported by the runner. Implements (executor-prompt Rev 2):

* the **two-crossover** reading rule over the takt grid (per clearance level, per
  arm-basis endpoint): the *solvability floor* (below it even the matched pair fails
  ``tau_solv``) and the *coupling ceiling* (below it the mismatched partner cannot
  afford its re-grasp, so the gap clears ``delta_min``); the solvable-and-coupling-valid
  window is between them, mapped back to takt;
* the **clearance-sweep readout** (decision #1): the binding grasp-pose mismatch
  threshold in BOTH degrees and matched-sigma units as a function of the in-grasp
  correction limit, plus the realistic mismatched-presenter mass that clears it;
* the **verdict** over the verdict space, including ``WASHOUT_FOR_REAL_CELLS`` (coupling
  only at takts below any defensible station) and ``COUPLING_VALID_INTRINSIC``
  (degradation persists at the free-re-grasp endpoint -> kinematic, takt-independent).

A null is informative via the equivalence / CI bound (two-sided CI upper < delta_min),
never a power claim. The decision-rule magnitudes are imported from
:mod:`chamber.envs.handover_place` so the rule, the power sim, and the runner cannot
drift (single source of truth).

References:
- ADR-026 §Decision (coupling validity; non-gating Phase-2 line).
- ADR-008 (IQM + cluster-bootstrap estimator family).
"""

from __future__ import annotations

import math
from typing import NamedTuple

from chamber.envs.handover_place import HANDOVER_DELTA_MIN_PP, HANDOVER_TAU_SOLV

#: Defensible per-part station takt band (s) for the realistic-overlap check. A per-part
#: handover-and-seat station runs on the order of a few seconds; this band is the
#: committed comparison range against which the coupling window is read (it is NOT a
#: hard-sourced point — that is the whole reason takt is swept).
REALISTIC_TAKT_BAND_S: tuple[float, float] = (1.0, 5.0)

# Verdict labels (must match the prereg verdict_space).
VERDICT_COUPLING_VALID: str = "COUPLING_VALID"
VERDICT_COUPLING_VALID_INTRINSIC: str = "COUPLING_VALID_INTRINSIC"
VERDICT_WASHOUT: str = "WASHOUT"
VERDICT_WASHOUT_FOR_REAL_CELLS: str = "WASHOUT_FOR_REAL_CELLS"
VERDICT_LIMB1_FAIL: str = "LIMB1_FAIL"
VERDICT_INDETERMINATE: str = "INDETERMINATE"


class CellVerdict(NamedTuple):
    """Classification of one (clearance, takt, arm-basis) cell (ADR-026 §Decision).

    ``solvable`` iff the matched success CI lower bound clears ``tau_solv``;
    ``coupling_valid`` iff the gap CI lower bound clears ``delta_min_pp``;
    ``washout`` iff the gap CI upper bound is below ``delta_min_pp`` (informative null);
    otherwise ``indeterminate`` (the CI straddles the bar).
    """

    solvable: bool
    coupling_valid: bool
    washout: bool
    indeterminate: bool


def classify_cell(
    *,
    matched_ci_low: float,
    gap_ci_low_pp: float,
    gap_ci_high_pp: float,
    tau_solv: float = HANDOVER_TAU_SOLV,
    delta_min_pp: float = HANDOVER_DELTA_MIN_PP,
) -> CellVerdict:
    """Classify a cell from its matched/gap CIs (ADR-026; ADR-008 equivalence bound)."""
    solvable = matched_ci_low >= tau_solv
    coupling_valid = gap_ci_low_pp >= delta_min_pp
    washout = gap_ci_high_pp < delta_min_pp
    indeterminate = not coupling_valid and not washout
    return CellVerdict(
        solvable=solvable,
        coupling_valid=coupling_valid,
        washout=washout,
        indeterminate=indeterminate,
    )


def binding_threshold_deg(wrist_correction_deg: float, angular_window_deg: float) -> float:
    """Grasp-pose error above which a re-grasp is forced AND its absence fails (ADR-026).

    The ego nulls error up to ``wrist_correction_deg`` in-grasp; a remainder breaks the
    place iff it exceeds the angular window. So the binding threshold is their sum.
    """
    return wrist_correction_deg + angular_window_deg


def threshold_in_sigma(threshold_deg: float, matched_sigma_deg: float) -> float:
    """Express the binding threshold in matched-presenter-sigma units (ADR-026; Rev 2 T2)."""
    if matched_sigma_deg <= 0.0:
        return math.inf
    return threshold_deg / matched_sigma_deg


def mismatch_mass_clearing(
    threshold_deg: float, mismatch_bias_deg: float, mismatch_sigma_deg: float
) -> float:
    """Fraction of the mismatched-presenter grasp-pose mass with |error| > threshold.

    The realistic mismatch overlay (decision #1): does real mismatch mass clear the
    binding threshold? Models the grasp-pose error as ``Normal(bias, sigma)`` and
    returns ``P(|error| > threshold)`` (both tails), via the complementary error
    function (ADR-026 §Decision).
    """
    if mismatch_sigma_deg <= 0.0:
        return 1.0 if abs(mismatch_bias_deg) > threshold_deg else 0.0
    root2 = math.sqrt(2.0)
    upper = 0.5 * math.erfc((threshold_deg - mismatch_bias_deg) / (mismatch_sigma_deg * root2))
    lower = 0.5 * math.erfc((threshold_deg + mismatch_bias_deg) / (mismatch_sigma_deg * root2))
    return upper + lower


class CrossoverWindow(NamedTuple):
    """The solvable-and-coupling-valid takt window from one grid (ADR-026 §Decision)."""

    solvability_floor_takt_s: float | None
    coupling_ceiling_takt_s: float | None
    window_takts_s: list[float]
    non_empty: bool


def clearance_mismatch_overlay(
    *,
    clearance_factors: list[float],
    mismatch_biases_deg: list[float],
    angular_window_deg: float,
    matched_sigma_deg: float,
    mismatch_sigma_deg: float,
    j5_pitch_half_deg: float,
) -> list[dict[str, float | bool]]:
    """The (clearance x mismatch) design overlay (decision #2; ADR-026).

    Closes the symmetric knob: for each (clearance_factor, mismatch_bias) cell, the
    binding grasp-pose threshold (= wrist_correction + angular_window) in degrees and
    matched-sigma units, whether the mismatch bias crosses it, and the realistic
    mismatch mass clearing it. Analytical (design-time); the MEASURED coupling region
    is produced by the runner at PR C. ``j5_pitch_half_deg`` is the cited binding wrist
    axis (wrist_correction = j5_pitch_half * clearance_factor).
    """
    wrist_by_f = {f: j5_pitch_half_deg * f for f in clearance_factors}
    thr_by_f = {
        f: binding_threshold_deg(wrist_by_f[f], angular_window_deg) for f in clearance_factors
    }
    return [
        {
            "clearance_factor": f,
            "mismatch_bias_deg": bias,
            "wrist_correction_deg": wrist_by_f[f],
            "binding_threshold_deg": thr_by_f[f],
            "binding_threshold_sigma": threshold_in_sigma(thr_by_f[f], matched_sigma_deg),
            "threshold_crossed": bias > thr_by_f[f],
            "mismatch_mass_clearing": mismatch_mass_clearing(thr_by_f[f], bias, mismatch_sigma_deg),
        }
        for f in clearance_factors
        for bias in mismatch_biases_deg
    ]


def two_crossover_window(takt_sorted_cells: list[tuple[float, CellVerdict]]) -> CrossoverWindow:
    """Locate the solvable-and-coupling-valid takt window from a takt-sorted grid (ADR-026).

    ``takt_sorted_cells`` is ``[(takt_s, CellVerdict), ...]`` ascending in takt. Returns
    the solvability floor (smallest takt that is solvable), the coupling ceiling (largest
    takt that is coupling-valid — above it the partner can always afford its re-grasp),
    and the window of takts that are BOTH, with ``non_empty``.
    """
    solvable_takts = [t for t, v in takt_sorted_cells if v.solvable]
    coupling_takts = [t for t, v in takt_sorted_cells if v.coupling_valid]
    both_takts = [t for t, v in takt_sorted_cells if v.solvable and v.coupling_valid]
    return CrossoverWindow(
        solvability_floor_takt_s=min(solvable_takts) if solvable_takts else None,
        coupling_ceiling_takt_s=max(coupling_takts) if coupling_takts else None,
        window_takts_s=both_takts,
        non_empty=bool(both_takts),
    )


def _overlaps_band(takts: list[float], band: tuple[float, float]) -> bool:
    lo, hi = band
    return any(lo <= t <= hi for t in takts)


def classify_verdict(
    *,
    window: CrossoverWindow,
    free_regrasp_gap_ci_low_pp: float,
    matched_solvable_at_realistic: bool,
    any_indeterminate: bool,
    realistic_takt_band_s: tuple[float, float] = REALISTIC_TAKT_BAND_S,
    delta_min_pp: float = HANDOVER_DELTA_MIN_PP,
) -> str:
    """Map the window + free-regrasp diagnostic to a verdict (ADR-026 §Decision; Rev 2).

    Precedence: intrinsic coupling (persists at free re-grasp) -> coupling-valid (window
    overlaps a realistic takt) -> washout-for-real-cells (window only below the realistic
    band) -> Limb-1 fail -> washout -> indeterminate.
    """
    if free_regrasp_gap_ci_low_pp >= delta_min_pp:
        return VERDICT_COUPLING_VALID_INTRINSIC
    window_takts = window.window_takts_s
    if window_takts and _overlaps_band(window_takts, realistic_takt_band_s):
        return VERDICT_COUPLING_VALID
    if window_takts:  # non-empty but only below the realistic band
        return VERDICT_WASHOUT_FOR_REAL_CELLS
    if not matched_solvable_at_realistic:
        return VERDICT_LIMB1_FAIL
    if any_indeterminate:
        return VERDICT_INDETERMINATE
    return VERDICT_WASHOUT
