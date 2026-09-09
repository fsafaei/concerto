# SPDX-License-Identifier: Apache-2.0
"""Tier-1 tests for ``concerto.contracts.actions`` (no SAPIEN; a fake env stands in)."""

from __future__ import annotations

import dataclasses

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium import spaces

from concerto import contracts
from concerto.contracts import ActionField, TaskContract
from concerto.contracts.actions import ScaleDeltaActions

# ManiSkill 3.0.1 panda ``pd_ee_delta_pose`` (agents/robots/panda/panda.py): pos +-0.1, rot +-0.1.
PANDA_POS_UPPER = 0.1
PANDA_ROT_LOWER = -0.1
OBS_DIM = 42

CASES = {
    "in_range": [0.3, -0.2, 0.9, 0.1, 0.0, -0.4, 0.5],
    "translation_out_of_range": [2.0, -3.0, 0.5, 0.0, 0.0, 0.0, 1.0],
    "rotation_norm_above_one": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, -1.0],
    "mixed": [7.0, 0.5, -7.0, 0.8, -0.8, 0.8, 5.0],
    "zero": [0.0] * 7,
}


class _FakeEnv(gym.Env):  # type: ignore[type-arg]
    """Records the action it receives; spaces shaped like the pickcube contract's."""

    def __init__(self) -> None:
        self.observation_space = spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (7,), np.float32)
        self.received: object = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(OBS_DIM, np.float32), {}

    def step(self, action):
        self.received = action
        return np.zeros(OBS_DIM, np.float32), 0.0, False, False, {}


def _maniskill_controller(a7: np.ndarray) -> np.ndarray:
    """Numpy port of ``PDEEPoseController._clip_and_scale_action`` plus the gripper's affine map."""
    pos = np.clip(a7[:3], -1.0, 1.0) * PANDA_POS_UPPER
    rot = a7[3:6].astype(float)
    norm = np.linalg.norm(rot)
    if norm > 1.0:
        rot = rot / norm
    rot = rot * PANDA_ROT_LOWER
    grip = 0.015 + 0.025 * np.clip(a7[6], -1.0, 1.0)
    return np.concatenate([pos, rot, [grip]])


def _contract_physical(c: TaskContract, a7: np.ndarray) -> np.ndarray:
    """What the contract says the action means (robolab's ``action_to_physical`` rule)."""
    sl = c.action_slices()
    pos_f, rot_f, grip_f = (c.action_field(n) for n in ("delta_pos", "delta_rot", "gripper"))
    pos = [pos_f.to_physical(float(v)) for v in a7[sl["delta_pos"]]]
    rot = a7[sl["delta_rot"]].astype(float)
    norm = np.linalg.norm(rot)
    if norm > 1.0:
        rot = rot / norm
    rot_phys = [rot_f.to_physical(float(v)) for v in rot]
    grip = grip_f.to_physical(float(a7[sl["gripper"]][0]))
    return np.array(pos + rot_phys + [grip])


def _wrap(c: TaskContract, pos_upper=PANDA_POS_UPPER, rot_lower=PANDA_ROT_LOWER):
    return ScaleDeltaActions(_FakeEnv(), c, pos_upper=pos_upper, rot_lower=rot_lower)


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("case", sorted(CASES))
def test_controller_after_wrapper_equals_contract_to_physical(version: int, case: str) -> None:
    c = contracts.get("pickcube", version)
    a = np.array(CASES[case], dtype=np.float32)
    sent = _wrap(c).action(a).numpy()
    np.testing.assert_allclose(_maniskill_controller(sent), _contract_physical(c, a), atol=1e-7)


def test_v2_scales_and_v1_identity() -> None:
    w2, w1 = _wrap(contracts.get("pickcube", 2)), _wrap(contracts.get("pickcube", 1))
    assert (w2.pos_scale, w2.rot_scale) == pytest.approx((0.05, 0.5))
    assert (w1.pos_scale, w1.rot_scale) == pytest.approx((1.0, 1.0))
    a = np.array(CASES["mixed"], dtype=np.float32)
    np.testing.assert_allclose(w1.action(a).numpy()[6], a[6])
    np.testing.assert_allclose(w1.action(a).numpy()[:3], np.clip(a[:3], -1, 1))


def test_accepts_numpy_and_torch_in_both_shapes_and_returns_float32_torch() -> None:
    w = _wrap(contracts.get("pickcube"))
    a = np.array(CASES["mixed"], dtype=np.float64)
    expected = w.action(a.astype(np.float32)).numpy()
    for raw in (a, a[None, :], torch.tensor(a), torch.tensor(a)[None, :]):
        out = w.action(raw)
        assert isinstance(out, torch.Tensor)
        assert out.dtype == torch.float32
        assert out.shape == tuple(raw.shape)
        np.testing.assert_allclose(out.reshape(-1).numpy(), expected, atol=1e-7)
    before = a.copy()
    w.action(a)
    np.testing.assert_array_equal(a, before)  # the input is never mutated
    with pytest.raises(ValueError, match="expected 7"):
        w.action(np.zeros(6, np.float32))


def test_step_hands_the_scaled_action_to_the_env() -> None:
    inner = _FakeEnv()
    w = ScaleDeltaActions(inner, contracts.get("pickcube"), pos_upper=0.1, rot_lower=-0.1)
    a = np.array(CASES["in_range"], dtype=np.float32)
    w.reset(seed=0)
    w.step(a)
    received = inner.received
    assert isinstance(received, torch.Tensor)
    np.testing.assert_allclose(received.numpy(), w.action(a).numpy())
    assert w.action_space == w.env.action_space
    assert w.get_wrapper_attr("contract") is contracts.get("pickcube")


def test_gripper_passes_through_unclipped() -> None:
    w = _wrap(contracts.get("pickcube"))
    out = w.action(np.array([0, 0, 0, 0, 0, 0, 5.0], dtype=np.float32)).numpy()
    assert out[6] == pytest.approx(5.0), "ManiSkill clips the gripper itself"


def test_refuses_a_contract_bound_above_the_controller() -> None:
    with pytest.raises(ValueError, match="pos_scale"):
        _wrap(contracts.get("pickcube", 2), pos_upper=0.001)


def test_refuses_a_rotation_sign_drift() -> None:
    with pytest.raises(ValueError, match="rot_scale"):
        _wrap(contracts.get("pickcube", 2), rot_lower=0.1)


def test_refuses_asymmetric_bounds() -> None:
    c = contracts.get("pickcube")
    fields = list(c.action_spec)
    fields[0] = ActionField("delta_pos", 3, -1.0, 1.0, -0.005, 0.01, "m per step", "base")
    skewed = dataclasses.replace(c, version=99, action_spec=tuple(fields))
    with pytest.raises(ValueError, match="symmetric"):
        _wrap(skewed)


def test_refuses_a_contract_without_the_three_fields() -> None:
    c = contracts.get("pickcube")
    fields = (ActionField("delta_pos", 3, -1.0, 1.0, -0.005, 0.005, "m", "base"),)
    with pytest.raises(ValueError, match="delta_rot"):
        _wrap(dataclasses.replace(c, version=99, action_spec=fields))
