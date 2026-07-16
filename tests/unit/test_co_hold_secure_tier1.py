# SPDX-License-Identifier: Apache-2.0
# pyright: reportPrivateImportUsage=false, reportAttributeAccessIssue=false
"""Tier-1 (no-SAPIEN-scene) tests for the co-hold-secure env (ADR-029 §Decision).

Covers the pure-Python surface of :mod:`chamber.envs.co_hold_secure` that needs
no SAPIEN scene:

- module import on a Vulkan-less host (ADR-001 §Risks / P2).
- :func:`resolve_cohold_control` — the four precheck cells and the loud-fail.
- :func:`evaluate_cohold_success` — the ``seated ∧ pose_held ∧ within_force ∧
  static ∧ settled`` conjunction on synthetic inputs, including the new
  pose_held conjunct, and that the Gate-0-derived force budgets are required.
- the wedge-inverted design rule as code (:func:`cohold_wedge_limit_deg`,
  :func:`cohold_design_window_ok`) — the ADR-029 §Validation pin that a
  geometry edit silently leaving the window fails CI.
- the founder-confirmed geometry anchors and the S2-measured inputs.
- the part-box composition (:func:`_part_boxes`) — box count, open bore
  throat, blind floor at the engagement depth.
- the Tier-1 factory validation paths (bad control / horizon / clearance /
  the dormant route-(ii) pose-sequence hook), all raised before any SAPIEN
  import.

Tier-2 SAPIEN-gated coverage (construction of each control flag, reset/step
determinism, finite telemetry, the registry ``make`` round-trip) lives in
``tests/integration/test_co_hold_secure_real.py``.

ADR-029 §Decision / §Consequences / §Validation; ADR-027 §Admission.
"""

from __future__ import annotations

import numpy as np
import pytest

from chamber.envs.co_hold_secure import (
    COHOLD_ACHIEVABLE_TILT_CEILING_DEG,
    COHOLD_CHAMFER_M,
    COHOLD_CLEARANCE_SIDE_SET_M,
    COHOLD_COUPLE_FORCE_MAX_N,
    COHOLD_DEPTH_EPS_M,
    COHOLD_DETENT_CLICK_EPS_M,
    COHOLD_DETENT_FORCE_N,
    COHOLD_DETENT_TRAVEL_M,
    COHOLD_ENGAGE_DEPTH_M,
    COHOLD_PART_MASS_KG,
    COHOLD_PLUG_DIAMETER_M,
    COHOLD_POSE_HELD_TILT_TOL_DEG,
    COHOLD_POSE_HELD_TRANS_TOL_M,
    COHOLD_SECURE_FORCE_MAX_N,
    COHOLD_WEDGE_MARGIN_MIN,
    CoHoldControl,
    _part_boxes,
    _rot_from_quat_wxyz,
    cohold_bore_inner_radius,
    cohold_design_window_ok,
    cohold_nominal_pad_bottom_z_w,
    cohold_wedge_limit_deg,
    evaluate_cohold_success,
    make_co_hold_secure_env,
    resolve_cohold_control,
)

# Synthetic Gate-0-derived force limits for the predicate tests (real values
# are derived from the matched-pair distribution at Gate-0; test fixtures only).
_F_SECURE = 60.0
_F_COUPLE = 120.0


class TestResolveControl:
    """The four precheck control cells (ADR-029 §Decision)."""

    def test_matched(self) -> None:
        cfg = resolve_cohold_control("matched")
        assert isinstance(cfg, CoHoldControl)
        assert cfg.holder_active is True
        assert cfg.part_mode == "fixed_link"

    def test_limp_is_the_matched_rig(self) -> None:
        # C-limp differs from C-matched only in who drives the holder seat
        # (the PR #298 C2-style zero-action instrument) — identical build.
        limp, matched = resolve_cohold_control("limp"), resolve_cohold_control("matched")
        assert limp.holder_active is True
        assert limp.part_mode == matched.part_mode == "fixed_link"
        assert limp.expected_partner == "partner_ablated_zero"

    def test_none_is_free_part_on_stand(self) -> None:
        cfg = resolve_cohold_control("none")
        assert cfg.holder_active is False
        assert cfg.part_mode == "free_stand"

    def test_fixture_is_world_fixed(self) -> None:
        cfg = resolve_cohold_control("fixture")
        assert cfg.holder_active is False
        assert cfg.part_mode == "world_fixed"

    def test_unknown_control_raises_naming_options(self) -> None:
        with pytest.raises(ValueError, match="not one of the co-hold-secure"):
            resolve_cohold_control("cohold_bogus")
        with pytest.raises(ValueError, match="ADR-029"):
            resolve_cohold_control("")


class TestSuccessPredicate:
    """``seated ∧ pose_held ∧ within_force ∧ static ∧ settled`` (ADR-029)."""

    @staticmethod
    def _ok_kwargs() -> dict[str, object]:
        return {
            "seated_depth_m": COHOLD_ENGAGE_DEPTH_M,
            "axis_align_deg": 1.0,
            "pose_excursion_m": 0.002,
            "pose_tilt_deg": 0.5,
            "peak_secure_force_n": 30.0,
            "peak_couple_wrench_n": 40.0,
            "both_static": True,
            "settled": True,
            "f_secure_max": _F_SECURE,
            "f_couple_max": _F_COUPLE,
        }

    def test_all_conjuncts_pass_is_success(self) -> None:
        assert bool(evaluate_cohold_success(**self._ok_kwargs())) is True  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        ("key", "bad"),
        [
            ("seated_depth_m", COHOLD_ENGAGE_DEPTH_M / 2.0),
            ("axis_align_deg", 45.0),
            ("pose_excursion_m", COHOLD_POSE_HELD_TRANS_TOL_M + 0.001),
            ("pose_tilt_deg", COHOLD_POSE_HELD_TILT_TOL_DEG + 1.0),
            ("peak_secure_force_n", _F_SECURE + 1.0),
            ("peak_couple_wrench_n", _F_COUPLE + 1.0),
            ("both_static", False),
            ("settled", False),
        ],
    )
    def test_each_single_violation_fails(self, key: str, bad: object) -> None:
        kw = self._ok_kwargs()
        kw[key] = bad
        assert bool(evaluate_cohold_success(**kw)) is False  # type: ignore[arg-type]

    def test_pose_held_is_a_distinct_conjunct(self) -> None:
        # The limp-holder failure signature: the plug can seat while the part
        # escapes — success must still be False (ADR-029 §Consequences).
        kw = self._ok_kwargs()
        kw["pose_excursion_m"] = 0.05
        assert bool(evaluate_cohold_success(**kw)) is False  # type: ignore[arg-type]

    def test_broadcasts_over_batches(self) -> None:
        kw = self._ok_kwargs()
        kw["seated_depth_m"] = np.array([COHOLD_ENGAGE_DEPTH_M, 0.0])
        out = evaluate_cohold_success(**kw)  # type: ignore[arg-type]
        assert out.tolist() == [True, False]

    def test_force_limits_are_required_no_default(self) -> None:
        kw = self._ok_kwargs()
        del kw["f_secure_max"]
        with pytest.raises(TypeError):
            evaluate_cohold_success(**kw)  # type: ignore[arg-type]


class TestWedgeDesignRule:
    """The wedge-inverted design window (ADR-029 §Decision / §Validation)."""

    def test_wedge_law_matches_the_s2_anchor(self) -> None:
        # The S2 archives: seatable tilt < ~0.7 deg at 0.5 mm/side and 38 mm.
        assert cohold_wedge_limit_deg(0.5e-3, 0.038) == pytest.approx(0.7538, abs=1e-3)

    def test_wedge_limits_of_the_clearance_band(self) -> None:
        limits = [cohold_wedge_limit_deg(c) for c in COHOLD_CLEARANCE_SIDE_SET_M]
        assert limits[0] == pytest.approx(8.5308, abs=1e-3)
        assert limits[1] == pytest.approx(14.0362, abs=1e-3)
        assert limits[2] == pytest.approx(19.2900, abs=1e-3)

    def test_every_cell_is_inside_the_design_window(self) -> None:
        # The §Validation pin: a geometry edit that leaves the window fails CI.
        for clearance in COHOLD_CLEARANCE_SIDE_SET_M:
            assert cohold_design_window_ok(clearance) is True
            assert cohold_wedge_limit_deg(clearance) >= (
                COHOLD_WEDGE_MARGIN_MIN * COHOLD_ACHIEVABLE_TILT_CEILING_DEG
            )

    def test_the_coinsert_geometry_fails_the_window(self) -> None:
        # The co-insert cells (per-side {0.5, 0.25, 0.1} mm at 38 mm) are the
        # measured counter-example the rule was inverted from.
        for per_side in (0.5e-3, 0.25e-3, 0.1e-3):
            assert cohold_design_window_ok(per_side, 0.038) is False

    def test_monotone_in_clearance(self) -> None:
        a, b, c = (cohold_wedge_limit_deg(x) for x in (1.5e-3, 2.5e-3, 3.5e-3))
        assert a < b < c


class TestFrozenableConstants:
    """The founder-confirmed anchors and S2-measured inputs (ADR-029)."""

    def test_geometry_anchors(self) -> None:
        assert pytest.approx(0.010) == COHOLD_ENGAGE_DEPTH_M
        assert COHOLD_CLEARANCE_SIDE_SET_M == (1.5e-3, 2.5e-3, 3.5e-3)
        assert pytest.approx(0.002) == COHOLD_CHAMFER_M
        assert pytest.approx(0.016) == COHOLD_PLUG_DIAMETER_M
        assert pytest.approx(0.5) == COHOLD_PART_MASS_KG

    def test_detent_anchors(self) -> None:
        assert pytest.approx(40.0) == COHOLD_DETENT_FORCE_N
        assert pytest.approx(0.002) == COHOLD_DETENT_TRAVEL_M
        # seated must certify a genuine click-through: eps < travel, and the
        # click release margin is tighter still.
        assert COHOLD_DEPTH_EPS_M < COHOLD_DETENT_TRAVEL_M
        assert COHOLD_DETENT_CLICK_EPS_M < COHOLD_DEPTH_EPS_M

    def test_s2_measured_inputs(self) -> None:
        assert pytest.approx(2.8) == COHOLD_ACHIEVABLE_TILT_CEILING_DEG
        assert pytest.approx(2.0) == COHOLD_WEDGE_MARGIN_MIN

    def test_force_budgets_are_none_until_gate0(self) -> None:
        assert COHOLD_SECURE_FORCE_MAX_N is None
        assert COHOLD_COUPLE_FORCE_MAX_N is None


class TestPartGeometry:
    """Part composition from convex boxes (ADR-001 §Risks no-mesh; ADR-029)."""

    def test_bore_inner_radius_is_per_side(self) -> None:
        r = COHOLD_PLUG_DIAMETER_M / 2.0
        for clearance in COHOLD_CLEARANCE_SIDE_SET_M:
            assert cohold_bore_inner_radius(clearance) == pytest.approx(r + clearance)

    def test_box_count(self) -> None:
        # 1 floor + 12 wall facets + 12 chamfer facets + 1 base pad.
        assert len(_part_boxes(COHOLD_CLEARANCE_SIDE_SET_M[1])) == 26

    def test_bore_throat_is_open(self) -> None:
        # No box corner may protrude inside the bore radius through the throat
        # (mouth-to-floor) — the lip that would re-create the co-insert block.
        clearance = COHOLD_CLEARANCE_SIDE_SET_M[1]
        r_in = cohold_bore_inner_radius(clearance)
        for half, centre, quat in _part_boxes(clearance):
            rot = _rot_from_quat_wxyz(quat)
            for sx in (-1, 1):
                for sy in (-1, 1):
                    for sz in (-1, 1):
                        corner = np.asarray(centre) + rot @ (np.asarray(half) * [sx, sy, sz])
                        if -COHOLD_ENGAGE_DEPTH_M < corner[2] < COHOLD_CHAMFER_M:
                            assert float(np.hypot(corner[0], corner[1])) >= r_in - 1e-9

    def test_floor_caps_the_bore_at_the_engagement_depth(self) -> None:
        # The blind floor top face sits exactly at -engage_depth: the plug
        # bottoms out at the click boundary (the stable "clicked" state).
        half, centre, _quat = _part_boxes(COHOLD_CLEARANCE_SIDE_SET_M[1])[0]
        assert centre[2] + half[2] == pytest.approx(-COHOLD_ENGAGE_DEPTH_M)

    def test_pad_bottom_is_below_the_part(self) -> None:
        z = cohold_nominal_pad_bottom_z_w()
        assert np.isfinite(z)
        assert 0.0 < z < 0.38901  # above the table, below the nominal mouth


class TestFactoryTier1Validation:
    """Factory argument validation raised before any SAPIEN import (P2)."""

    def test_unknown_control_raises(self) -> None:
        with pytest.raises(ValueError, match="not one of the co-hold-secure"):
            make_co_hold_secure_env(control="bogus")

    def test_bad_horizon_raises(self) -> None:
        with pytest.raises(ValueError, match="quasi-static band"):
            make_co_hold_secure_env(episode_length=10)

    def test_nonpositive_clearance_raises(self) -> None:
        with pytest.raises(ValueError, match="clearance_side_m"):
            make_co_hold_secure_env(clearance_side_m=0.0)

    def test_multi_pose_hook_is_dormant(self) -> None:
        # Route (ii) — designed but dormant (ADR-029 §Decision).
        with pytest.raises(NotImplementedError, match="route-\\(ii\\)"):
            make_co_hold_secure_env(pose_sequence_len=2)

    def test_num_envs_other_than_one_fails_loudly(self) -> None:
        # The PR-A telemetry instruments are scalar per-env-0 (ADR-029 §Risks);
        # silent vectorisation would mis-attribute channels.
        with pytest.raises(ValueError, match="num_envs"):
            make_co_hold_secure_env(num_envs=2)


class TestHolderBlackBoxShield:
    """The holder seat enters through the FrozenPartner contract (ADR-009/I3)."""

    def test_holder_partner_is_black_box_frozen(self) -> None:
        from chamber.partners.api import PartnerSpec
        from chamber.partners.interface import _FORBIDDEN_ATTRS
        from chamber.partners.registry import load_partner

        holder = load_partner(
            PartnerSpec(
                "coinsert_reference_holder",
                0,
                None,
                None,
                {"uid": "panda_partner", "base_xyz": "0.62,0,0", "base_yaw_deg": "180"},
            )
        )
        # The shield: policy-access attributes raise; reset/act are the contract.
        for attr in _FORBIDDEN_ATTRS:
            with pytest.raises(AttributeError):
                getattr(holder, attr)
        holder.reset(seed=0)
        assert callable(holder.act)
