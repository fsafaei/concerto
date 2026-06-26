# SPDX-License-Identifier: Apache-2.0
"""Tier-1 tests for the frozen scripted presenter partner (ADR-009; ADR-026).

Determinism, the presentation action shape, the matched/mismatched split (which lives
in the GRASP-POSE channel only), registry round-trip, and the black-box
``_FORBIDDEN_ATTRS`` shield that blocks any joint-training / policy-weight access.
"""

from __future__ import annotations

import numpy as np
import pytest

from chamber.envs.handover_place import HANDOVER_PRESENTATION_DIM
from chamber.partners.handover_presenter import (
    HANDOVER_PRESENTER_CLASS,
    HandoverPresenterPartner,
    presenter_spec,
)
from chamber.partners.interface import _FORBIDDEN_ATTRS
from chamber.partners.registry import load_partner

# Presentation layout: [lat_offset_x, lat_offset_y, grasp_pose_error_deg, timing_skew_s]
_GRASP_POSE_IDX = 2
_TIMING_IDX = 3


def _presentation(variant: str, seed: int, **params) -> np.ndarray:
    partner = HandoverPresenterPartner(presenter_spec(variant, **params))
    partner.reset(seed=seed)
    return np.asarray(partner.act({}))


class TestActionShape:
    def test_presentation_dim(self) -> None:
        action = _presentation("matched", seed=0)
        assert action.shape == (HANDOVER_PRESENTATION_DIM,)

    def test_timing_skew_nonnegative(self) -> None:
        action = _presentation("mismatched", seed=4)
        assert action[_TIMING_IDX] >= 0.0


class TestDeterminism:
    def test_same_seed_byte_identical(self) -> None:
        a = _presentation("mismatched", seed=11)
        b = _presentation("mismatched", seed=11)
        np.testing.assert_array_equal(a, b)

    def test_different_seed_differs(self) -> None:
        a = _presentation("mismatched", seed=11)
        b = _presentation("mismatched", seed=12)
        assert not np.array_equal(a, b)

    def test_act_before_reset_raises(self) -> None:
        partner = HandoverPresenterPartner(presenter_spec("matched"))
        with pytest.raises(RuntimeError):
            partner.act({})


class TestVariantSplit:
    def test_mismatch_is_in_the_grasp_pose_channel(self) -> None:
        # The whole point (Rev 2 B1): the mismatch biases the GRASP-POSE channel.
        matched = np.array(
            [abs(_presentation("matched", seed=s)[_GRASP_POSE_IDX]) for s in range(64)]
        )
        mismatched = np.array(
            [abs(_presentation("mismatched", seed=s)[_GRASP_POSE_IDX]) for s in range(64)]
        )
        assert mismatched.mean() > matched.mean()

    def test_lateral_channel_not_inflated_by_mismatch(self) -> None:
        # Lateral is success-side, NOT the mismatch: both variants present small lateral.
        matched_lat = np.array(
            [np.linalg.norm(_presentation("matched", seed=s)[0:2]) for s in range(64)]
        )
        mismatched_lat = np.array(
            [np.linalg.norm(_presentation("mismatched", seed=s)[0:2]) for s in range(64)]
        )
        # Comparable lateral magnitudes (mismatch does not inflate the lateral channel).
        assert mismatched_lat.mean() < 5.0 * matched_lat.mean() + 1e-6

    def test_invalid_variant_rejected(self) -> None:
        with pytest.raises(ValueError, match="variant must be"):
            presenter_spec("sideways")


class TestRegistryRoundTrip:
    def test_load_partner_builds_presenter(self) -> None:
        spec = presenter_spec("matched", seed=0)
        assert spec.class_name == HANDOVER_PRESENTER_CLASS
        partner = load_partner(spec)
        partner.reset(seed=0)
        action = np.asarray(partner.act({}))
        assert action.shape == (HANDOVER_PRESENTATION_DIM,)


class TestBlackBoxShield:
    @pytest.mark.parametrize("attr", sorted(_FORBIDDEN_ATTRS))
    def test_forbidden_attrs_blocked(self, attr: str) -> None:
        partner = HandoverPresenterPartner(presenter_spec("matched"))
        with pytest.raises(AttributeError):
            getattr(partner, attr)

    def test_only_reset_and_act_are_the_interface(self) -> None:
        partner = HandoverPresenterPartner(presenter_spec("matched"))
        assert callable(partner.reset)
        assert callable(partner.act)
