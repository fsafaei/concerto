# SPDX-License-Identifier: Apache-2.0
# pyright: reportPrivateImportUsage=false, reportAttributeAccessIssue=false
"""Tier-1 (no-SAPIEN-scene) tests for the co-insert S2 controllers (ADR-026 §Decision 1).

Covers the pure-Python control surface of
:mod:`chamber.partners.coinsert_impedance` — the structured base inserter and
the cooperative reference holder — that needs no SAPIEN scene (the controllers
build a CPU ``pytorch_kinematics`` chain and act on synthetic observation
dicts):

- registration + construction through the partner registry (ADR-009 §Decision).
- ``act`` returns a well-formed ``pd_joint_delta_pos`` action (8-D, finite,
  gripper held, arm deltas in ``[-1, 1]``) and degrades to a gripper-only action
  when the obs lacks the required leaves (defensive — keeps the rollout shape
  stable).
- the per-episode-reset contract (ADR-009 §Decision; the co-carry
  stateful-control regression guard) — ``reset`` clears the integral + step
  counter (+ the holder's captured nominal height); ``assert_episode_state_clear``
  passes right after ``reset`` and fires after an ``act``.
- the base inserter's frozen-at-S3 insertion-envelope numbers.
- the pure geometry helpers (quaternion→rotation, lateral basis).

Tier-2 SAPIEN-gated coverage (real insertion success, force distributions) is
the S2 measurement campaign under ``scripts/repro``; the per-episode-reset
property test lives in ``tests/property/test_coinsert_reset.py``.

ADR-026 §Decision 1; ADR-009 §Decision; ADR-004 §Decision; the co-insert design.
"""

from __future__ import annotations

import numpy as np
import pytest

from chamber.partners.api import PartnerSpec
from chamber.partners.coinsert_impedance import (
    CoInsertBaseInserter,
    CoInsertReferenceHolder,
    _lateral_basis,
    _quat_wxyz_to_rot,
)
from chamber.partners.registry import load_partner

_BASE_EXTRA = {
    "uid": "panda_wristcam",
    "base_xyz": "-0.5,0,0",
    "base_yaw_deg": "0",
    "peg_half_len": "0.04",
}
_HOLDER_EXTRA = {
    "uid": "panda_partner",
    "base_xyz": "0.5,0,0",
    "base_yaw_deg": "180",
    "peg_half_len": "0.04",
}


def _base() -> CoInsertBaseInserter:
    p = load_partner(PartnerSpec("coinsert_base_inserter", 0, None, None, dict(_BASE_EXTRA)))
    assert isinstance(p, CoInsertBaseInserter)
    return p


def _holder() -> CoInsertReferenceHolder:
    p = load_partner(PartnerSpec("coinsert_reference_holder", 0, None, None, dict(_HOLDER_EXTRA)))
    assert isinstance(p, CoInsertReferenceHolder)
    return p


def _obs(uid: str) -> dict:
    """Synthetic obs: peg above an upward-facing socket (the warm-started layout)."""
    return {
        "agent": {uid: {"qpos": np.zeros((1, 9), dtype=np.float32)}},
        "extra": {
            # peg pointing down (identity → local +z is world +z; tip-down handled
            # by the controller via the half-len offset), socket flipped to face up.
            "peg_pose": np.array([[0.0, 0.0, 0.30, 1.0, 0.0, 0.0, 0.0]]),
            "receptacle_pose": np.array([[0.0, 0.0, 0.15, 0.0, 1.0, 0.0, 0.0]]),
        },
    }


def test_controllers_register_and_construct() -> None:
    assert isinstance(_base(), CoInsertBaseInserter)
    assert isinstance(_holder(), CoInsertReferenceHolder)


@pytest.mark.parametrize("which", ["base", "holder"])
def test_act_returns_wellformed_action(which: str) -> None:
    ctrl = _base() if which == "base" else _holder()
    uid = "panda_wristcam" if which == "base" else "panda_partner"
    ctrl.reset(seed=0)
    action = np.asarray(ctrl.act(_obs(uid)))
    assert action.shape == (8,)
    assert np.all(np.isfinite(action))
    assert np.all(action[:7] >= -1.0)
    assert np.all(action[:7] <= 1.0)
    # gripper channel held constant (welded body → fingers inert).
    assert action[7] == pytest.approx(-1.0)


@pytest.mark.parametrize("which", ["base", "holder"])
def test_missing_obs_keys_degrade_to_gripper_only(which: str) -> None:
    ctrl = _base() if which == "base" else _holder()
    ctrl.reset(seed=0)
    action = np.asarray(ctrl.act({"agent": {}, "extra": {}}))
    assert action.shape == (8,)
    assert np.all(action[:7] == 0.0)
    assert action[7] == pytest.approx(-1.0)


@pytest.mark.parametrize("which", ["base", "holder"])
def test_per_episode_reset_clears_state(which: str) -> None:
    ctrl = _base() if which == "base" else _holder()
    uid = "panda_wristcam" if which == "base" else "panda_partner"
    ctrl.reset(seed=0)
    ctrl.assert_episode_state_clear()  # passes right after reset
    for _ in range(5):
        ctrl.act(_obs(uid))
    with pytest.raises(AssertionError):
        ctrl.assert_episode_state_clear()  # state accumulated → fires
    ctrl.reset(seed=0)
    ctrl.assert_episode_state_clear()  # cleared again


def test_holder_nominal_height_cleared_on_reset() -> None:
    h = _holder()
    h.reset(seed=0)
    h.act(_obs("panda_partner"))  # captures nominal height
    with pytest.raises(AssertionError):
        h.assert_episode_state_clear()
    h.reset(seed=0)
    assert h._nominal_mouth is None
    h.assert_episode_state_clear()


def test_insertion_envelope_numbers() -> None:
    from chamber.envs.coinsert import COINSERT_CHAMFER_M

    env = _base().insertion_envelope_m
    assert env["chamfer_capture_radius_m"] == pytest.approx(float(COINSERT_CHAMFER_M))
    assert env["spiral_search_amplitude_m"] > 0.0
    assert env["insertion_envelope_m"] == pytest.approx(
        env["spiral_search_amplitude_m"] + env["chamfer_capture_radius_m"]
    )


def test_quat_identity_is_identity_rotation() -> None:
    np.testing.assert_allclose(
        _quat_wxyz_to_rot(np.array([1.0, 0.0, 0.0, 0.0])), np.eye(3), atol=1e-12
    )


def test_lateral_basis_orthonormal_and_perpendicular() -> None:
    for axis in (np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]), np.array([0.3, -0.4, 0.86])):
        e1, e2 = _lateral_basis(axis)
        a = axis / np.linalg.norm(axis)
        assert abs(e1 @ a) < 1e-9
        assert abs(e2 @ a) < 1e-9
        assert abs(e1 @ e2) < 1e-9
        assert e1 @ e1 == pytest.approx(1.0)
        assert e2 @ e2 == pytest.approx(1.0)


def test_determinism_same_obs_same_action_across_reset() -> None:
    """Two identical episodes (reset → same obs sequence) yield byte-identical actions (P6)."""
    b1, b2 = _base(), _base()
    b1.reset(seed=0)
    b2.reset(seed=0)
    for _ in range(6):
        a1 = np.asarray(b1.act(_obs("panda_wristcam")))
        a2 = np.asarray(b2.act(_obs("panda_wristcam")))
        np.testing.assert_array_equal(a1, a2)
