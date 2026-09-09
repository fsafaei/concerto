# SPDX-License-Identifier: Apache-2.0
"""Contracts for single-arm ``PickCube-v1`` (panda, state obs, ``pd_ee_delta_pose``).

Every number here was read from the installed ManiSkill 3.0.1 package:
``envs/tasks/tabletop/pick_cube.py`` (obs order, robot placement),
``agents/robots/panda/panda.py`` (controller bounds, rest keyframe),
``agents/controllers/pd_ee_pose.py`` (delta-pose frame and scaling) and
``utils/structs/types.py`` (default sim/control rates).

Version 1 is the env as ManiSkill ships it. Version 2 keeps everything but asks
for 20x smaller translation steps, 2x smaller rotation steps and a 4x longer
horizon, realised in the sim by :class:`concerto.contracts.actions.ScaleDeltaActions`
in front of the unchanged controller; both stay registered.
"""

from __future__ import annotations

import dataclasses
import math

from concerto.contracts.spec import ActionField, ObsField, TaskContract, register

_NOTES_V1 = (
    "Flat state obs = flatten(agent{qpos, qvel}) ++ flatten(extra{...}) in the listed order. "
    "Poses are [x y z qw qx qy qz]; world frame = base frame shifted by base_position_in_world. "
    "Arm action: the translation delta is added in the robot root (base) frame; the rotation "
    "action is clipped to unit norm, multiplied by rot_lower (= -0.1, so a positive action "
    "rotates negatively), read as XYZ Euler angles (R = Rx Ry Rz) and left-multiplied onto the "
    "current orientation (rotation about base-aligned axes). use_target=False: deltas apply to "
    "the instantaneous TCP pose, not the previous target. The gripper action maps affinely to a "
    "finger-joint target in metres; gripper width is twice the finger value."
)

_NOTES_V2 = (
    "Version 2 differs from version 1 only in the per-step action bounds and the horizon: +1 "
    "means 0.005 m of translation (was 0.1 m) and -0.05 rad of rotation (was -0.1 rad; the sign "
    "flip stays); the gripper field is unchanged; max_episode_steps is 200 (was 50). The sim "
    "realises the smaller bounds with concerto.contracts.actions.ScaleDeltaActions in front of "
    "the unchanged ManiSkill controller (pos +-0.1 m, rot +-0.1 rad): the wrapper clips exactly "
    "as the controller does (per element for the translation, unit norm for the rotation), then "
    "multiplies by 0.05 and 0.5, so the controller's own clip never binds and the physical delta "
    "equals ActionField.to_physical of the raw action. Why: robolab's conservative safety "
    "envelope passes at most 0.005 m and 0.05 rad per step at speed scale 1.0, so a policy "
    "trained on the version-1 bounds would be clamped 20x on the real arm. " + _NOTES_V1
)

PICKCUBE_V1: TaskContract = register(
    TaskContract(
        task_id="pickcube",
        version=1,
        env_id="PickCube-v1",
        robot_uid="panda",
        obs_mode="state",
        control_mode="pd_ee_delta_pose",
        control_hz=20,
        max_episode_steps=50,
        obs_spec=(
            ObsField("qpos", 9, "rad (7 arm joints), m (2 finger joints)", "joint"),
            ObsField("qvel", 9, "rad/s (7 arm joints), m/s (2 finger joints)", "joint"),
            ObsField("is_grasped", 1, "bool as float", "none"),
            ObsField("tcp_pose", 7, "m, unit quaternion wxyz", "world"),
            ObsField("goal_pos", 3, "m", "world"),
            ObsField("obj_pose", 7, "m, unit quaternion wxyz", "world"),
            ObsField("tcp_to_obj_pos", 3, "m", "world"),
            ObsField("obj_to_goal_pos", 3, "m", "world"),
        ),
        action_spec=(
            ActionField("delta_pos", 3, -1.0, 1.0, -0.1, 0.1, "m per step", "base"),
            ActionField(
                "delta_rot", 3, -1.0, 1.0, 0.1, -0.1, "rad per step (sign-flipped)", "base"
            ),
            ActionField("gripper", 1, -1.0, 1.0, -0.01, 0.04, "m (finger joint target)", "none"),
        ),
        quaternion_order="wxyz",
        reset_joint_config=(
            0.0,
            math.pi / 8,
            0.0,
            -math.pi * 5 / 8,
            0.0,
            math.pi * 3 / 4,
            math.pi / 4,
        ),
        base_position_in_world=(-0.615, 0.0, 0.0),
        delta_pose_frame="root_translation:root_aligned_body_rotation",
        notes=_NOTES_V1,
    )
)

PICKCUBE: TaskContract = register(
    dataclasses.replace(
        PICKCUBE_V1,
        version=2,
        max_episode_steps=200,
        action_spec=(
            ActionField("delta_pos", 3, -1.0, 1.0, -0.005, 0.005, "m per step", "base"),
            ActionField(
                "delta_rot", 3, -1.0, 1.0, 0.05, -0.05, "rad per step (sign-flipped)", "base"
            ),
            ActionField("gripper", 1, -1.0, 1.0, -0.01, 0.04, "m (finger joint target)", "none"),
        ),
        notes=_NOTES_V2,
    )
)
"""The current pickcube contract (version 2)."""
