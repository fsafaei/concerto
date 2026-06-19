# SPDX-License-Identifier: Apache-2.0
# pyright: reportPrivateImportUsage=false, reportAttributeAccessIssue=false
"""Tier-1 (no-SAPIEN-scene) tests for the Rung-4 embodiment-shift surface (ADR-026 §D4; ADR-005).

Covers the pure-Python surface that needs no SAPIEN scene:

- :mod:`chamber.agents.xarm6_jacobian` — the xArm6 FK/Jacobian provider parses
  the URDF via pytorch_kinematics (no SAPIEN), returns the right shapes, is
  deterministic, and loud-fails on a wrong-arity qpos.
- :mod:`chamber.partners.cocarry_xarm6` — the compliant xArm6 teammate is
  registered, returns a finite/bounded/deterministic 7-D action, exercises its
  cooperative bar-leveling branch (reads ``obs["extra"]["bar_pose"]``), and
  holds defensively on incomplete obs.
- :mod:`chamber.envs.cocarry_obs` — the partner-observation adapter
  (:func:`_adapt_partner_vec` / :func:`_fix_width`) is the IDENTITY for a Panda
  partner (7 arm + 2 gripper, so the matched reference stays byte-identical)
  and maps the 12-DOF xArm6 into the fixed 18-D Panda ego-state layout.
- :mod:`chamber.envs.cocarry` — the ``cocarry_xarm6_partner`` condition + the
  xArm6 controller geometry spec resolve correctly.

ADR-026 §Decision 4; ADR-005 §Decision; ADR-009 §Decision; R-2026-06-B §15 Rung 4.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

_XARM6_READY6 = np.array([0.0, -0.084, -0.8, 0.0, 0.692, 0.0])

#: The xArm6 + Robotiq URDF is a separate ManiSkill asset (ADR-005) — unlike
#: the bundled panda, it is not present until downloaded
#: (``python -m mani_skill.utils.download_asset xarm6_robotiq``; CI fetches it).
#: Tests that build the real kinematic chain skip cleanly when it is absent,
#: so a clean checkout without the asset does not hard-error (mirrors the
#: Tier-2 ``sapien_gpu_available`` gating pattern).
_XARM6_URDF = os.path.expanduser("~/.maniskill/data/robots/xarm6/xarm6_robotiq.urdf")
_needs_xarm6_asset = pytest.mark.skipif(
    not os.path.exists(_XARM6_URDF),
    reason="xArm6 URDF asset absent (run: python -m mani_skill.utils.download_asset xarm6_robotiq)",
)


# --------------------------------------------------------------------------
# xArm6 FK/Jacobian provider.
# --------------------------------------------------------------------------


@_needs_xarm6_asset
class TestXArm6JacobianProvider:
    """The xArm6 provider parses the URDF and returns the right shapes (ADR-005; ADR-026 §D4)."""

    @staticmethod
    def _provider():
        from chamber.agents.xarm6_jacobian import XArm6JacobianProvider

        return XArm6JacobianProvider()

    def test_fk_shape_and_determinism(self) -> None:
        p = self._provider()
        a = p.fk_tcp_position(_XARM6_READY6)
        b = p.fk_tcp_position(_XARM6_READY6)
        assert a.shape == (3,)
        assert a.dtype == np.float64
        np.testing.assert_array_equal(a, b)

    def test_jacobian_shape(self) -> None:
        j = self._provider().jacobian(_XARM6_READY6)
        assert j.shape == (3, 6)
        assert np.all(np.isfinite(j))

    def test_wrong_qpos_arity_raises(self) -> None:
        p = self._provider()
        with pytest.raises(ValueError, match=r"\(6,\)"):
            p.fk_tcp_position(np.zeros(7))
        with pytest.raises(ValueError, match=r"\(6,\)"):
            p.jacobian(np.zeros(5))


# --------------------------------------------------------------------------
# xArm6 compliant controller.
# --------------------------------------------------------------------------


def _xarm6_partner():
    from chamber.envs.cocarry import cocarry_xarm6_controller_spec
    from chamber.partners.api import PartnerSpec
    from chamber.partners.registry import load_partner

    return load_partner(
        PartnerSpec("cocarry_xarm6_impedance", 0, None, None, dict(cocarry_xarm6_controller_spec()))
    )


def _obs(q6: np.ndarray, goal: np.ndarray, bar: np.ndarray | None = None) -> dict:
    extra: dict = {"goal_pos": goal.astype(np.float32)}
    if bar is not None:
        extra["bar_pose"] = bar.astype(np.float32)
    # xArm6 qpos = 6 arm + 6 Robotiq gripper joints.
    qpos = np.concatenate([q6, np.zeros(6)]).astype(np.float32)
    return {"agent": {"xarm6_robotiq": {"qpos": qpos}}, "extra": extra}


def _bar(center: np.ndarray | None = None, *, tilt_deg: float = 0.0) -> np.ndarray:
    c = np.array([0.0, 0.0, 0.17]) if center is None else center
    a = np.radians(tilt_deg) / 2.0
    quat = np.array([np.cos(a), 0.0, np.sin(a), 0.0])  # rotation about y -> the x-ends differ in z
    return np.concatenate([c, quat])


@_needs_xarm6_asset
class TestXArm6Controller:
    """The xArm6 teammate's action contract + leveling (ADR-026 §D4; ADR-009)."""

    def test_registered(self) -> None:
        import chamber.partners.cocarry_xarm6  # noqa: F401
        from chamber.partners.registry import list_registered

        assert "cocarry_xarm6_impedance" in list_registered()

    def test_act_seven_dim_bounded_deterministic(self) -> None:
        p1, p2 = _xarm6_partner(), _xarm6_partner()
        p1.reset(seed=0)
        p2.reset(seed=0)
        obs = _obs(_XARM6_READY6, np.array([0.0, 0.12, 0.28]), bar=_bar())
        a1, a2 = p1.act(obs), p2.act(obs)
        assert a1.shape == (7,)
        assert a1.dtype == np.float32
        assert np.all(np.abs(a1) <= 1.0 + 1e-6)
        assert np.all(np.isfinite(a1))
        assert a1[6] == pytest.approx(0.0)  # gripper held (no delta)
        np.testing.assert_array_equal(a1, a2)

    def test_leveling_branch_uses_bar_pose(self) -> None:
        # Isolate the cooperative-leveling term: a goal at the current bar
        # height (~zero goal-error, so the Cartesian command is unsaturated),
        # comparing a LEVEL bar to a slightly TILTED one. Only the leveling
        # z-correction (the gap between the two bar-end heights) differs, so a
        # different action proves the bar-pose leveling branch is live.
        goal = np.array([0.0, 0.0, 0.17])
        p = _xarm6_partner()
        p.reset(seed=0)
        a_level = p.act(_obs(_XARM6_READY6, goal, bar=_bar(tilt_deg=0.0)))
        p.reset(seed=0)
        a_tilt = p.act(_obs(_XARM6_READY6, goal, bar=_bar(tilt_deg=5.0)))
        assert not np.array_equal(a_level, a_tilt)

    def test_incomplete_obs_holds(self) -> None:
        p = _xarm6_partner()
        p.reset(seed=0)
        a = p.act({"agent": {}, "extra": {}})
        assert a.shape == (7,)
        np.testing.assert_array_equal(a[:6], np.zeros(6, dtype=np.float32))

    def test_short_qpos_holds(self) -> None:
        p = _xarm6_partner()
        p.reset(seed=0)
        obs = {
            "agent": {"xarm6_robotiq": {"qpos": np.zeros(4, dtype=np.float32)}},
            "extra": {"goal_pos": np.zeros(3, dtype=np.float32)},
        }
        np.testing.assert_array_equal(p.act(obs)[:6], np.zeros(6, dtype=np.float32))

    def test_act_without_bar_falls_back_to_goal_z(self) -> None:
        # goal + qpos present, NO bar_pose -> the controller drives to goal_z
        # (the leveling fallback) and still emits a valid bounded action.
        p = _xarm6_partner()
        p.reset(seed=0)
        a = p.act(_obs(_XARM6_READY6, np.array([0.0, 0.12, 0.28]), bar=None))
        assert a.shape == (7,)
        assert np.all(np.isfinite(a))
        assert np.all(np.abs(a) <= 1.0 + 1e-6)

    def test_short_goal_holds(self) -> None:
        p = _xarm6_partner()
        p.reset(seed=0)
        obs = {
            "agent": {"xarm6_robotiq": {"qpos": np.zeros(12, dtype=np.float32)}},
            "extra": {"goal_pos": np.zeros(2, dtype=np.float32)},  # < 3 -> hold
        }
        np.testing.assert_array_equal(p.act(obs)[:6], np.zeros(6, dtype=np.float32))

    def test_missing_agent_uid_holds(self) -> None:
        p = _xarm6_partner()
        p.reset(seed=0)
        obs = {"agent": {"other": {}}, "extra": {"goal_pos": np.zeros(3, dtype=np.float32)}}
        np.testing.assert_array_equal(p.act(obs)[:6], np.zeros(6, dtype=np.float32))


class TestXArm6ControllerHelpers:
    """The xArm6 controller's geometry helpers (ADR-026 §D4)."""

    def test_parse_vec3_rejects_arity_and_non_float(self) -> None:
        from chamber.partners.cocarry_xarm6 import _parse_vec3

        with pytest.raises(ValueError, match="x,y,z"):
            _parse_vec3("1,2")
        with pytest.raises(ValueError, match="floats"):
            _parse_vec3("a,b,c")

    def test_quat_zero_norm_guard_is_identity(self) -> None:
        from chamber.partners.cocarry_xarm6 import _quat_wxyz_to_matrix

        np.testing.assert_array_equal(_quat_wxyz_to_matrix(np.zeros(4)), np.eye(3))

    def test_to_numpy_flat_handles_torch(self) -> None:
        import torch

        from chamber.partners.cocarry_xarm6 import _to_numpy_flat

        out = _to_numpy_flat(torch.arange(6, dtype=torch.float32).reshape(1, 6))
        np.testing.assert_array_equal(out, np.arange(6))


# --------------------------------------------------------------------------
# Partner-observation adapter (the frozen-ego interface bridge).
# --------------------------------------------------------------------------


class TestPartnerAdapter:
    """The partner→ego-state adapter: Panda no-op, xArm6 embed (ADR-026 §D4; Rung 4)."""

    def test_fix_width_pad_and_truncate(self) -> None:
        from chamber.envs.cocarry_obs import _fix_width

        wide = np.arange(12, dtype=np.float32).reshape(1, 12)
        np.testing.assert_array_equal(
            _fix_width(wide, 7), np.arange(7, dtype=np.float32).reshape(1, 7)
        )
        narrow = np.arange(3, dtype=np.float32).reshape(1, 3)
        out = _fix_width(narrow, 7)
        assert out.shape == (1, 7)
        np.testing.assert_array_equal(out[0, :3], [0, 1, 2])
        np.testing.assert_array_equal(out[0, 3:], np.zeros(4))

    def test_panda_partner_adapter_is_identity(self) -> None:
        from chamber.envs.cocarry_obs import _adapt_partner_vec

        # Panda partner: 7 arm + 2 finger = 9. arm_dof=7 -> identity.
        v = np.arange(9, dtype=np.float32)
        out = _adapt_partner_vec(v, arm_dof=7)
        assert out.shape == (1, 9)
        np.testing.assert_array_equal(out[0], v)

    def test_xarm6_partner_adapter_embeds_into_9(self) -> None:
        from chamber.envs.cocarry_obs import _adapt_partner_vec

        # xArm6: 6 arm + 6 gripper = 12. arm_dof=6 -> [6 arm, 1 zero pad, 2 gripper] = 9.
        v = np.arange(12, dtype=np.float32)  # arm 0..5, gripper 6..11
        out = _adapt_partner_vec(v, arm_dof=6)
        assert out.shape == (1, 9)
        np.testing.assert_array_equal(out[0, :6], [0, 1, 2, 3, 4, 5])  # arm
        assert out[0, 6] == 0.0  # arm pad to 7
        np.testing.assert_array_equal(out[0, 7:9], [6, 7])  # first 2 gripper


# --------------------------------------------------------------------------
# Env condition + controller geometry spec.
# --------------------------------------------------------------------------


class TestXArm6EnvWiring:
    """The xArm6 condition + controller spec resolve correctly (ADR-026 §D4; ADR-005)."""

    def test_condition_resolves(self) -> None:
        from chamber.envs.cocarry import resolve_cocarry_condition

        c = resolve_cocarry_condition("cocarry_xarm6_partner")
        assert c.agent_uids == ("panda_wristcam", "xarm6_robotiq")
        assert c.single_arm is False

    def test_controller_spec_geometry(self) -> None:
        from chamber.envs.cocarry import cocarry_xarm6_controller_spec

        spec = cocarry_xarm6_controller_spec()
        assert spec["uid"] == "xarm6_robotiq"
        assert spec["base_yaw_deg"] == "180"
        assert spec["end_sign"] == "-1"
        # The xArm6 mounts closer than the Panda partner (0.35 vs 0.5).
        assert float(spec["base_xyz"].split(",")[0]) == pytest.approx(0.35)
