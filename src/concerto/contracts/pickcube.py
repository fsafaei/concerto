# SPDX-License-Identifier: Apache-2.0
"""Contract for single-arm ``PickCube-v1`` (panda, state obs, ``pd_ee_delta_pose``).

Every number here was read from the installed ManiSkill 3.0.1 package:
``envs/tasks/tabletop/pick_cube.py`` (obs order, robot placement),
``agents/robots/panda/panda.py`` (controller bounds, rest keyframe),
``agents/controllers/pd_ee_pose.py`` (delta-pose frame and scaling) and
``utils/structs/types.py`` (default sim/control rates).
"""

from __future__ import annotations

import math

from concerto.contracts.spec import ActionField, ObsField, TaskContract, register

_NOTES = (
    "Flat state obs = flatten(agent{qpos, qvel}) ++ flatten(extra{...}) in the listed order. "
    "Poses are [x y z qw qx qy qz]; world frame = base frame shifted by base_position_in_world. "
    "Arm action: the translation delta is added in the robot root (base) frame; the rotation "
    "action is clipped to unit norm, multiplied by rot_lower (= -0.1, so a positive action "
    "rotates negatively), read as XYZ Euler angles (R = Rx Ry Rz) and left-multiplied onto the "
    "current orientation (rotation about base-aligned axes). use_target=False: deltas apply to "
    "the instantaneous TCP pose, not the previous target. The gripper action maps affinely to a "
    "finger-joint target in metres; gripper width is twice the finger value."
)

PICKCUBE: TaskContract = register(
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
        notes=_NOTES,
    )
)
