# SPDX-License-Identifier: Apache-2.0
"""Tier-1 tests for ``concerto.contracts`` (no SAPIEN; always runs on CPU)."""

from __future__ import annotations

import dataclasses
import math

import pytest

from concerto import contracts
from concerto.contracts import ActionField, ObsField, TaskContract

PICKCUBE_OBS_ORDER = [
    ("qpos", 9),
    ("qvel", 9),
    ("is_grasped", 1),
    ("tcp_pose", 7),
    ("goal_pos", 3),
    ("obj_pose", 7),
    ("tcp_to_obj_pos", 3),
    ("obj_to_goal_pos", 3),
]


def test_get_pickcube_is_populated() -> None:
    c = contracts.get("pickcube")
    assert c.task_id == "pickcube"
    assert c.version >= 1
    assert c.env_id == "PickCube-v1"
    assert c.robot_uid == "panda"
    assert c.control_mode == "pd_ee_delta_pose"
    assert c.control_hz == 20
    assert c.max_episode_steps == 50
    assert c.obs_dim == 42
    assert c.action_dim == 7
    assert c.quaternion_order == "wxyz"
    assert len(c.reset_joint_config) == contracts.N_ARM_JOINTS
    assert c.reset_joint_config[3] == pytest.approx(-math.pi * 5 / 8)
    assert c.base_position_in_world == (-0.615, 0.0, 0.0)
    assert c.delta_pose_frame == "root_translation:root_aligned_body_rotation"
    assert c.notes


def test_pickcube_obs_order_and_slices_are_contiguous() -> None:
    c = contracts.get("pickcube")
    assert [(f.name, f.dim) for f in c.obs_spec] == PICKCUBE_OBS_ORDER
    slices = c.obs_slices()
    expected_start = 0
    for name, dim in PICKCUBE_OBS_ORDER:
        assert slices[name] == slice(expected_start, expected_start + dim)
        expected_start += dim
    assert expected_start == c.obs_dim


def test_pickcube_action_slices() -> None:
    c = contracts.get("pickcube")
    assert c.action_slices() == {
        "delta_pos": slice(0, 3),
        "delta_rot": slice(3, 6),
        "gripper": slice(6, 7),
    }
    with pytest.raises(KeyError, match="no action field"):
        c.action_field("nope")


def test_pickcube_action_physical_mapping() -> None:
    c = contracts.get("pickcube")
    pos, rot, grip = (c.action_field(n) for n in ("delta_pos", "delta_rot", "gripper"))
    assert pos.to_physical(1.0) == pytest.approx(0.1)
    assert pos.to_physical(-0.5) == pytest.approx(-0.05)
    assert pos.to_physical(7.0) == pytest.approx(0.1), "clips before scaling"
    # ManiSkill 3.0.1 multiplies the rotation action by rot_lower (-0.1): sign flip.
    assert rot.to_physical(1.0) == pytest.approx(-0.1)
    assert rot.to_physical(0.0) == pytest.approx(0.0)
    assert grip.to_physical(-1.0) == pytest.approx(-0.01)
    assert grip.to_physical(1.0) == pytest.approx(0.04)
    assert grip.to_physical(0.0) == pytest.approx(0.015)


def test_get_unknown_raises_keyerror_listing_known() -> None:
    with pytest.raises(KeyError, match="pickcube"):
        contracts.get("does_not_exist")


def test_list_contracts_contains_pickcube() -> None:
    assert "pickcube" in contracts.list_contracts()


def test_contract_is_frozen() -> None:
    c = contracts.get("pickcube")
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.version = 99  # type: ignore[misc]


def test_register_duplicate_raises() -> None:
    with pytest.raises(ValueError, match="already registered"):
        contracts.register(contracts.get("pickcube"))


def test_obs_field_rejects_zero_dim() -> None:
    with pytest.raises(ValueError, match="dim must be positive"):
        ObsField("x", 0, "m", "world")


def test_action_field_rejects_empty_range() -> None:
    with pytest.raises(ValueError, match="bad dim/bounds"):
        ActionField("x", 1, 1.0, 1.0, 0.0, 1.0, "m", "base")


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"quaternion_order": "xyzw"}, "quaternion_order"),
        ({"reset_joint_config": (0.0,) * 6}, "reset_joint_config"),
        ({"base_position_in_world": (0.0, 0.0)}, "base_position_in_world"),
        ({"control_hz": 0}, "control_hz"),
        ({"obs_spec": ()}, "obs_spec is empty"),
        (
            {"action_spec": (ActionField("a", 1, -1.0, 1.0, 0.0, 1.0, "m", "base"),) * 2},
            "duplicate field names",
        ),
    ],
)
def test_contract_validation(changes: dict[str, object], message: str) -> None:
    base = contracts.get("pickcube")
    with pytest.raises(ValueError, match=message):
        dataclasses.replace(base, **changes)  # pyright: ignore[reportArgumentType]


def test_minimal_contract_constructs() -> None:
    c = TaskContract(
        task_id="t",
        version=0,
        env_id="E-v0",
        robot_uid="panda",
        obs_mode="state",
        control_mode="pd_ee_delta_pose",
        control_hz=10,
        max_episode_steps=1,
        obs_spec=(ObsField("a", 2, "m", "world"),),
        action_spec=(ActionField("b", 1, -1.0, 1.0, -0.1, 0.1, "m", "base"),),
        quaternion_order="wxyz",
        reset_joint_config=(0.0,) * 7,
        base_position_in_world=(0.0, 0.0, 0.0),
        delta_pose_frame="root_translation:root_aligned_body_rotation",
    )
    assert (c.obs_dim, c.action_dim) == (2, 1)
    assert c.notes == ""
