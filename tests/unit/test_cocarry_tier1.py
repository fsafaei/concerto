# SPDX-License-Identifier: Apache-2.0
# pyright: reportPrivateImportUsage=false, reportAttributeAccessIssue=false
"""Tier-1 (no-SAPIEN-scene) tests for the co-carry rig (ADR-026 §Decision 1-2).

Covers the pure-Python surface of :mod:`chamber.envs.cocarry` and
:mod:`chamber.partners.cocarry_impedance` that needs no SAPIEN scene:

- :func:`resolve_cocarry_condition` — the matched + single-arm positive
  control conditions, and the loud-fail on a bogus id.
- :func:`tilt_deg_from_quaternion` — level => 0 deg, pitched bar tilt,
  batched.
- :func:`evaluate_cocarry_success` — the joint success conjunction on
  synthetic metrics (placed + level + unstressed + static => success;
  each single violation => failure).
- Reward / geometry / threshold constants present and typed.
- :func:`cocarry_matched_controller_specs` — the controller geometry the
  matched pair reads, derived from the env constants.
- :class:`CoCarryImpedancePartner` — registered, parses its spec, and its
  ``act`` returns an 8-D action that steps the TCP toward the goal
  (the Jacobian/FK provider parses the URDF without SAPIEN — Tier-1-safe).

Tier-2 SAPIEN-gated coverage (real env construction, Rung-0 attach
stability, Rung-1 matched competence + the coupling positive-control) is
in ``tests/integration/test_cocarry_real.py``.

ADR-026 §Decision 1-2; ADR-009 §Decision; R-2026-06-B Rungs 0-1.
"""

from __future__ import annotations

import numpy as np
import pytest

from chamber.envs.cocarry import (
    COCARRY_BAR_LENGTH_M,
    COCARRY_BAR_MASS_KG,
    COCARRY_GOAL_THRESH_M,
    COCARRY_REWARD_LEVEL_COEFF,
    COCARRY_REWARD_NORMALIZER,
    COCARRY_REWARD_SETTLE_COEFF,
    COCARRY_REWARD_SUCCESS_BONUS,
    COCARRY_REWARD_TRANSPORT_COEFF,
    COCARRY_SETTLE_WINDOW_STEPS,
    COCARRY_STRESS_MAX_PROXY_N,
    COCARRY_TILT_MAX_DEG,
    CoCarryCondition,
    cocarry_matched_controller_specs,
    evaluate_cocarry_success,
    resolve_cocarry_condition,
    tilt_deg_from_quaternion,
)

_MATCHED = "cocarry_matched_panda_pair"
_SINGLE = "cocarry_single_arm_positive_control"


class TestResolveCondition:
    """Co-carry condition resolution (ADR-026 §Decision 1-2)."""

    def test_matched(self) -> None:
        cfg = resolve_cocarry_condition(_MATCHED)
        assert isinstance(cfg, CoCarryCondition)
        assert cfg.agent_uids == ("panda_wristcam", "panda_partner")
        assert cfg.single_arm is False

    def test_single_arm_positive_control(self) -> None:
        cfg = resolve_cocarry_condition(_SINGLE)
        assert cfg.agent_uids == ("panda_wristcam", "panda_partner")
        assert cfg.single_arm is True

    def test_unknown_condition_raises_naming_options(self) -> None:
        with pytest.raises(ValueError, match="not one of the co-carry"):
            resolve_cocarry_condition("cocarry_bogus")
        with pytest.raises(ValueError, match="ADR-026"):
            resolve_cocarry_condition("")


class TestTiltFromQuaternion:
    """Bar tilt from level via the pose quaternion (ADR-026 §Decision 1)."""

    def test_identity_is_level(self) -> None:
        assert float(tilt_deg_from_quaternion([1.0, 0.0, 0.0, 0.0])) == pytest.approx(0.0, abs=1e-6)

    def test_ninety_degree_pitch_is_vertical(self) -> None:
        # 90 deg rotation about y maps the bar's local-x long axis to world -z.
        q = [np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4), 0.0]
        assert float(tilt_deg_from_quaternion(q)) == pytest.approx(90.0, abs=1e-3)

    def test_forty_five_degree_pitch(self) -> None:
        q = [np.cos(np.pi / 8), 0.0, np.sin(np.pi / 8), 0.0]
        assert float(tilt_deg_from_quaternion(q)) == pytest.approx(45.0, abs=1e-3)

    def test_yaw_about_z_keeps_bar_level(self) -> None:
        # Rotation about world z keeps the long axis horizontal -> 0 tilt.
        q = [np.cos(np.pi / 6), 0.0, 0.0, np.sin(np.pi / 6)]
        assert float(tilt_deg_from_quaternion(q)) == pytest.approx(0.0, abs=1e-6)

    def test_batched(self) -> None:
        qs = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [np.cos(np.pi / 8), 0.0, np.sin(np.pi / 8), 0.0],
            ]
        )
        out = tilt_deg_from_quaternion(qs)
        assert out.shape == (2,)
        np.testing.assert_allclose(out, [0.0, 45.0], atol=1e-3)


class TestEvaluateSuccess:
    """Joint co-carry success predicate on synthetic metrics (ADR-026 §Decision 1)."""

    def _ok(
        self,
        *,
        centroid_to_goal_dist: float = 0.02,
        max_tilt_deg: float = 5.0,
        max_stress_proxy: float = 100.0,
        both_static: bool = True,
    ) -> np.ndarray:
        return evaluate_cocarry_success(
            centroid_to_goal_dist=centroid_to_goal_dist,
            max_tilt_deg=max_tilt_deg,
            max_stress_proxy=max_stress_proxy,
            both_static=both_static,
        )

    def test_placed_level_unstressed_static_is_success(self) -> None:
        assert bool(self._ok())

    def test_not_placed_fails(self) -> None:
        assert not bool(self._ok(centroid_to_goal_dist=0.5))

    def test_tilted_fails(self) -> None:
        assert not bool(self._ok(max_tilt_deg=COCARRY_TILT_MAX_DEG + 1.0))

    def test_overstressed_fails(self) -> None:
        assert not bool(self._ok(max_stress_proxy=COCARRY_STRESS_MAX_PROXY_N + 1.0))

    def test_non_static_fails(self) -> None:
        assert not bool(self._ok(both_static=False))

    def test_tilt_boundary_is_strict(self) -> None:
        # Exactly at the limit must fail (predicate uses strict <).
        assert not bool(self._ok(max_tilt_deg=COCARRY_TILT_MAX_DEG))


class TestConstants:
    """Pre-registration-grade constants present + typed (R-2026-06-B)."""

    def test_reward_coefficients_named_and_float(self) -> None:
        for c in (
            COCARRY_REWARD_TRANSPORT_COEFF,
            COCARRY_REWARD_LEVEL_COEFF,
            COCARRY_REWARD_SETTLE_COEFF,
            COCARRY_REWARD_SUCCESS_BONUS,
            COCARRY_REWARD_NORMALIZER,
        ):
            assert isinstance(c, float)
            assert np.isfinite(c)

    def test_geometry_and_threshold_constants(self) -> None:
        assert COCARRY_BAR_LENGTH_M > 0.0
        assert COCARRY_BAR_MASS_KG > 0.0
        assert 0.0 < COCARRY_GOAL_THRESH_M < COCARRY_BAR_LENGTH_M
        assert 0.0 < COCARRY_TILT_MAX_DEG < 90.0
        assert COCARRY_STRESS_MAX_PROXY_N > 0.0
        assert isinstance(COCARRY_SETTLE_WINDOW_STEPS, int)
        assert COCARRY_SETTLE_WINDOW_STEPS >= 0


class TestMatchedControllerSpecs:
    """Controller geometry derived from the env constants (ADR-026 §Decision 1)."""

    def test_specs_cover_both_uids_with_opposite_ends(self) -> None:
        specs = cocarry_matched_controller_specs()
        assert set(specs) == {"panda_wristcam", "panda_partner"}
        assert specs["panda_wristcam"]["end_sign"] == "1"
        assert specs["panda_partner"]["end_sign"] == "-1"
        assert specs["panda_partner"]["base_yaw_deg"] == "180"
        # bar_half_len is half the bar length.
        assert float(specs["panda_wristcam"]["bar_half_len"]) == pytest.approx(
            COCARRY_BAR_LENGTH_M / 2.0
        )


class TestImpedanceControllerTier1:
    """The matched controller's ``act`` on synthetic obs (ADR-026 §Decision 1; ADR-009).

    The Jacobian/FK provider parses the panda URDF via pytorch_kinematics
    without SAPIEN, so the controller's full control law is Tier-1-testable.
    """

    @staticmethod
    def _partner(uid: str, end_sign: str, yaw: str):
        import chamber.partners.cocarry_impedance  # noqa: F401 - register
        from chamber.partners.api import PartnerSpec
        from chamber.partners.registry import load_partner

        extra = {
            "uid": uid,
            "base_xyz": "-0.5,0,0" if uid == "panda_wristcam" else "0.5,0,0",
            "base_yaw_deg": yaw,
            "end_sign": end_sign,
            "bar_half_len": "0.115",
        }
        return load_partner(PartnerSpec("cocarry_impedance", 0, None, None, extra))

    @staticmethod
    def _obs(uid: str, qpos7: np.ndarray, goal: np.ndarray) -> dict:
        return {
            "agent": {uid: {"qpos": np.concatenate([qpos7, [0.04, 0.04]]).astype(np.float32)}},
            "extra": {"goal_pos": goal.astype(np.float32)},
        }

    def test_registered(self) -> None:
        import chamber.partners.cocarry_impedance  # noqa: F401
        from chamber.partners.registry import list_registered

        assert "cocarry_impedance" in list_registered()

    def test_act_returns_eight_dim_with_open_gripper(self) -> None:
        ctrl = self._partner("panda_wristcam", "1", "0")
        q = np.array([0.0, np.pi / 8, 0.0, -np.pi * 5 / 8, 0.0, np.pi * 3 / 4, -np.pi / 4])
        a = ctrl.act(self._obs("panda_wristcam", q, np.array([0.0, 0.1, 0.3])))
        assert a.shape == (8,)
        assert a.dtype == np.float32
        # Gripper held open (last component +1).
        assert a[7] == pytest.approx(1.0)
        assert np.all(np.abs(a) <= 1.0 + 1e-6)

    def test_act_holds_open_gripper_action_when_obs_incomplete(self) -> None:
        ctrl = self._partner("panda_wristcam", "1", "0")
        a = ctrl.act({"agent": {}, "extra": {}})
        assert a.shape == (8,)
        assert a[7] == pytest.approx(1.0)
        np.testing.assert_array_equal(a[:7], np.zeros(7, dtype=np.float32))

    def test_act_with_missing_goal_but_present_qpos_holds(self) -> None:
        """qpos present, goal absent -> the controller emits a hold (no arm command)."""
        ctrl = self._partner("panda_wristcam", "1", "0")
        qpos9 = np.concatenate([np.zeros(7), [0.04, 0.04]]).astype(np.float32)
        obs = {
            "agent": {"panda_wristcam": {"qpos": qpos9}},
            "extra": {},
        }
        a = ctrl.act(obs)
        np.testing.assert_array_equal(a[:7], np.zeros(7, dtype=np.float32))
        assert a[7] == pytest.approx(1.0)

    def test_act_with_short_goal_and_short_qpos_holds(self) -> None:
        """Malformed goal / qpos lengths fall through to the hold action (defensive)."""
        ctrl = self._partner("panda_wristcam", "1", "0")
        short_goal = {
            "agent": {"panda_wristcam": {"qpos": np.zeros(9, dtype=np.float32)}},
            "extra": {"goal_pos": np.zeros(2, dtype=np.float32)},
        }
        np.testing.assert_array_equal(ctrl.act(short_goal)[:7], np.zeros(7, dtype=np.float32))
        short_qpos = {
            "agent": {"panda_wristcam": {"qpos": np.zeros(5, dtype=np.float32)}},
            "extra": {"goal_pos": np.zeros(3, dtype=np.float32)},
        }
        np.testing.assert_array_equal(ctrl.act(short_qpos)[:7], np.zeros(7, dtype=np.float32))

    def test_bad_base_xyz_raises_value_error(self) -> None:
        from chamber.partners.api import PartnerSpec
        from chamber.partners.registry import load_partner

        with pytest.raises(ValueError, match="three floats"):
            load_partner(PartnerSpec("cocarry_impedance", 0, None, None, {"base_xyz": "1,2"}))
        with pytest.raises(ValueError, match="must be floats"):
            load_partner(PartnerSpec("cocarry_impedance", 0, None, None, {"base_xyz": "a,b,c"}))

    def test_act_steps_tcp_toward_a_higher_goal(self) -> None:
        """A goal above the current TCP should command a non-zero arm action upward.

        The controller is deterministic; with the TCP starting at the
        ready pose and the goal placed above it, the returned joint deltas
        must be non-trivial (the arm is asked to lift), and applying the
        FK at ``qpos + delta_scaled`` must reduce the base-frame error.
        """
        from chamber.agents.panda_jacobian import PandaJacobianProvider

        ctrl = self._partner("panda_wristcam", "1", "0")
        q = np.array([0.0, np.pi / 8, 0.0, -np.pi * 5 / 8, 0.0, np.pi * 3 / 4, -np.pi / 4])
        prov = PandaJacobianProvider()
        tcp0 = prov.fk_tcp_position(q)
        # Goal well above current TCP (world); ego base at (-0.5,0,0),
        # end_sign +1, half 0.115 -> target_base = goal + (0.5,0,0) + (0.115,0,0).
        goal = np.array([0.0, 0.0, tcp0[2] - 0.5 + 0.2])  # raise z by ~0.2 in base terms
        a = ctrl.act(self._obs("panda_wristcam", q, goal))
        delta = a[:7].astype(np.float64) * 0.1  # un-normalise to rad delta
        assert float(np.linalg.norm(delta)) > 1e-3
        tcp1 = prov.fk_tcp_position(q + delta)
        target_base = goal + np.array([0.5, 0.0, 0.0]) + np.array([0.115, 0.0, 0.0])
        assert np.linalg.norm(tcp1 - target_base) < np.linalg.norm(tcp0 - target_base)


class TestCoCarryCouplingTier1:
    """The Rung-4b compliant-coupling resolver (ADR-026 §Decision 4; R-2026-06-B §15 Rung 4b).

    Pure Tier-1: :func:`chamber.envs.cocarry.cocarry_coupling` resolves the
    dual-hold drive (stiffness, damping, force_limit) for the rigid default and
    the compliant Variant-A/B overrides, without any SAPIEN dependency.
    """

    @staticmethod
    def _coupling():
        from chamber.envs.cocarry import cocarry_coupling

        return cocarry_coupling

    def test_rigid_default_matches_module_constants(self) -> None:
        from chamber.envs.cocarry import (
            COCARRY_DRIVE_FORCE_LIMIT_UNBOUNDED,
            COCARRY_DRIVE_LINEAR_DAMPING,
            COCARRY_DRIVE_LINEAR_STIFFNESS,
        )

        k, c, fl = self._coupling()()
        assert k == COCARRY_DRIVE_LINEAR_STIFFNESS
        assert c == COCARRY_DRIVE_LINEAR_DAMPING
        assert fl == COCARRY_DRIVE_FORCE_LIMIT_UNBOUNDED

    def test_variant_a_lower_stiffness_derives_damping(self) -> None:
        from chamber.envs.cocarry import COCARRY_DRIVE_DAMPING_RATIO

        k, c, fl = self._coupling()(5000.0)
        assert k == 5000.0
        assert c == pytest.approx(5000.0 * COCARRY_DRIVE_DAMPING_RATIO)
        # Variant A: unbounded force (no cap).
        assert fl > 1e30

    def test_explicit_damping_and_force_limit_override(self) -> None:
        k, c, fl = self._coupling()(8000.0, 123.0, 110.0)
        assert (k, c, fl) == (8000.0, 123.0, 110.0)

    def test_damping_ratio_is_rigid_ratio(self) -> None:
        from chamber.envs.cocarry import (
            COCARRY_DRIVE_DAMPING_RATIO,
            COCARRY_DRIVE_LINEAR_DAMPING,
            COCARRY_DRIVE_LINEAR_STIFFNESS,
        )

        assert (
            pytest.approx(COCARRY_DRIVE_LINEAR_DAMPING / COCARRY_DRIVE_LINEAR_STIFFNESS)
            == COCARRY_DRIVE_DAMPING_RATIO
        )
