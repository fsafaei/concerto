# SPDX-License-Identifier: Apache-2.0
"""Tier-1 tests for the Gate-0 decision rule (ADR-026; Rev 2 §6).

Cell classification, the binding-threshold (deg + sigma) and mismatch-overlay machinery
(decision #1), the two-crossover window, and every verdict branch — exercised on
synthetic inputs (no env).
"""

from __future__ import annotations

import math

from chamber.spikes.handover_place_gate0.decision import (
    REALISTIC_TAKT_BAND_S,
    VERDICT_COUPLING_VALID,
    VERDICT_COUPLING_VALID_INTRINSIC,
    VERDICT_LIMB1_FAIL,
    VERDICT_WASHOUT,
    VERDICT_WASHOUT_FOR_REAL_CELLS,
    CellVerdict,
    binding_threshold_deg,
    classify_cell,
    classify_verdict,
    mismatch_mass_clearing,
    threshold_in_sigma,
    two_crossover_window,
)


class TestClassifyCell:
    def test_coupling_valid(self) -> None:
        v = classify_cell(matched_ci_low=0.95, gap_ci_low_pp=25.0, gap_ci_high_pp=40.0)
        assert v.solvable
        assert v.coupling_valid
        assert not v.washout

    def test_washout_equivalence_bound(self) -> None:
        v = classify_cell(matched_ci_low=0.95, gap_ci_low_pp=-5.0, gap_ci_high_pp=10.0)
        assert v.washout
        assert not v.coupling_valid
        assert not v.indeterminate

    def test_indeterminate_straddles_bar(self) -> None:
        v = classify_cell(matched_ci_low=0.95, gap_ci_low_pp=5.0, gap_ci_high_pp=30.0)
        assert v.indeterminate

    def test_not_solvable_below_tau(self) -> None:
        v = classify_cell(matched_ci_low=0.80, gap_ci_low_pp=25.0, gap_ci_high_pp=40.0)
        assert not v.solvable


class TestThresholdAndOverlay:
    def test_binding_threshold_is_wrist_plus_window(self) -> None:
        assert binding_threshold_deg(25.0, 3.0) == 28.0

    def test_threshold_in_sigma(self) -> None:
        assert threshold_in_sigma(28.0, 2.0) == 14.0

    def test_mismatch_mass_clearing_high_when_bias_above_threshold(self) -> None:
        # bias 30 well above threshold 28 -> most mass clears.
        assert mismatch_mass_clearing(28.0, 30.0, 10.0) > 0.5

    def test_mismatch_mass_clearing_low_when_threshold_far_above_bias(self) -> None:
        # threshold 90 far above bias 30 -> little mass clears.
        assert mismatch_mass_clearing(90.0, 30.0, 10.0) < 0.01

    def test_mismatch_mass_degenerate_sigma(self) -> None:
        assert mismatch_mass_clearing(28.0, 30.0, 0.0) == 1.0
        assert mismatch_mass_clearing(40.0, 30.0, 0.0) == 0.0


def _cells(coupling_takts, solvable_takts):
    return [
        (
            t,
            CellVerdict(
                solvable=t in solvable_takts,
                coupling_valid=t in coupling_takts,
                washout=t not in coupling_takts,
                indeterminate=False,
            ),
        )
        for t in (0.5, 1.0, 2.0, 5.0)
    ]


class TestCrossoverWindow:
    def test_window_is_intersection(self) -> None:
        w = two_crossover_window(_cells(coupling_takts={0.5, 1.0}, solvable_takts={1.0, 2.0, 5.0}))
        assert w.window_takts_s == [1.0]
        assert w.non_empty

    def test_empty_window(self) -> None:
        w = two_crossover_window(_cells(coupling_takts={0.5}, solvable_takts={2.0, 5.0}))
        assert not w.non_empty


class TestVerdict:
    def test_intrinsic_precedence(self) -> None:
        w = two_crossover_window(_cells(coupling_takts=set(), solvable_takts={2.0}))
        assert (
            classify_verdict(
                window=w,
                free_regrasp_gap_ci_low_pp=25.0,
                matched_solvable_at_realistic=True,
                any_indeterminate=False,
            )
            == VERDICT_COUPLING_VALID_INTRINSIC
        )

    def test_coupling_valid_overlaps_realistic(self) -> None:
        # window at takt 2.0 (inside the realistic band) -> COUPLING_VALID.
        w = two_crossover_window(_cells(coupling_takts={2.0}, solvable_takts={2.0}))
        assert REALISTIC_TAKT_BAND_S[0] <= 2.0 <= REALISTIC_TAKT_BAND_S[1]
        assert (
            classify_verdict(
                window=w,
                free_regrasp_gap_ci_low_pp=0.0,
                matched_solvable_at_realistic=True,
                any_indeterminate=False,
            )
            == VERDICT_COUPLING_VALID
        )

    def test_washout_for_real_cells_below_band(self) -> None:
        # window only at takt 0.5 (below the realistic band) -> WASHOUT_FOR_REAL_CELLS.
        w = two_crossover_window(_cells(coupling_takts={0.5}, solvable_takts={0.5}))
        assert (
            classify_verdict(
                window=w,
                free_regrasp_gap_ci_low_pp=0.0,
                matched_solvable_at_realistic=True,
                any_indeterminate=False,
            )
            == VERDICT_WASHOUT_FOR_REAL_CELLS
        )

    def test_washout_empty(self) -> None:
        w = two_crossover_window(_cells(coupling_takts=set(), solvable_takts={2.0}))
        assert (
            classify_verdict(
                window=w,
                free_regrasp_gap_ci_low_pp=0.0,
                matched_solvable_at_realistic=True,
                any_indeterminate=False,
            )
            == VERDICT_WASHOUT
        )

    def test_limb1_fail(self) -> None:
        w = two_crossover_window(_cells(coupling_takts=set(), solvable_takts=set()))
        assert (
            classify_verdict(
                window=w,
                free_regrasp_gap_ci_low_pp=0.0,
                matched_solvable_at_realistic=False,
                any_indeterminate=False,
            )
            == VERDICT_LIMB1_FAIL
        )

    def test_threshold_in_sigma_infinite_for_zero_sigma(self) -> None:
        assert math.isinf(threshold_in_sigma(28.0, 0.0))
