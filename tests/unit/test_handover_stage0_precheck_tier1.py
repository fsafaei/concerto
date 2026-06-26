# SPDX-License-Identifier: Apache-2.0
"""Tier-1 tests for the Stage-0 channel pre-check (ADR-026; Rev 2 §3).

Verifies the boundary derivation is a transparent function of the cited inputs, the
channel finding identifies grasp-pose as the re-grasp-forcing channel under the default
kinematics, lateral is translatable, and the sweep is deterministic.
"""

from __future__ import annotations

from chamber.spikes.handover_place_gate0.stage0_precheck import (
    GRASPABLE_FACE_SPAN_DEG,
    J5_PITCH_HALF_DEG,
    SEAT_APPROACH_CLEARANCE_FACTOR,
    derive_grasp_pose_boundaries,
    run_channel_precheck,
)


class TestBoundaryDerivation:
    def test_wrist_correction_is_pitch_times_clearance(self) -> None:
        wrist, reacquire = derive_grasp_pose_boundaries()
        assert wrist == J5_PITCH_HALF_DEG * SEAT_APPROACH_CLEARANCE_FACTOR
        assert reacquire == GRASPABLE_FACE_SPAN_DEG

    def test_wrist_below_reacquire(self) -> None:
        wrist, reacquire = derive_grasp_pose_boundaries()
        assert wrist < reacquire


class TestChannelFinding:
    def test_grasp_pose_is_the_binding_channel(self) -> None:
        finding = run_channel_precheck()
        assert finding["channel"] == "grasp_pose_orientation"
        assert finding["grasp_pose_channel"]["forces_regrasp"] is True
        assert finding["lateral_channel"]["forces_regrasp"] is False

    def test_lateral_fully_translatable(self) -> None:
        finding = run_channel_precheck()
        assert finding["lateral_channel"]["translatable_fraction"] >= 0.999

    def test_grasp_pose_fractions_sum_to_one(self) -> None:
        gp = run_channel_precheck()["grasp_pose_channel"]
        total = gp["in_grasp_fraction"] + gp["requires_regrasp_fraction"] + gp["intrinsic_fraction"]
        assert abs(total - 1.0) < 1e-9

    def test_finding_exposes_derived_boundaries(self) -> None:
        b = run_channel_precheck()["derived_boundaries"]
        assert b["wrist_correction_range_deg_headline"] > 0.0
        assert b["reacquire_range_deg"] > b["wrist_correction_range_deg_headline"]

    def test_clearance_factor_is_swept_not_a_point(self) -> None:
        finding = run_channel_precheck()
        sweep = finding["clearance_sweep"]
        assert set(sweep) == {"0.20", "0.35", "0.50", "0.70"}
        # wrist_correction is monotone increasing in the clearance factor.
        vals = [sweep[k] for k in sorted(sweep)]
        assert vals == sorted(vals)
        assert finding["inputs"]["clearance_factor_sweep"] == [0.2, 0.35, 0.5, 0.7]


class TestDeterminism:
    def test_byte_identical_across_runs(self) -> None:
        assert run_channel_precheck(root_seed=0) == run_channel_precheck(root_seed=0)
