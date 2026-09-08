# SPDX-License-Identifier: Apache-2.0
"""Tier-2: the real ManiSkill ``PickCube-v1`` env matches ``concerto.contracts.get("pickcube")``.

Gated on :func:`chamber.utils.device.sapien_gpu_available`; skipped on
Vulkan-less hosts. On a SAPIEN-capable host it pins the spaces, the field
layout, the controller frame and bounds, the reset configuration, and the
sign of the delta-pose actions the contract documents.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from chamber.utils.device import sapien_gpu_available
from concerto import contracts

_N_STEPS = 5
_MIN_MOVE_M = 0.01
_MIN_TURN_RAD = 0.02


def _tcp_pose(env: object) -> tuple[np.ndarray, np.ndarray]:
    raw = env.unwrapped.agent.tcp_pose.raw_pose[0].cpu().numpy()  # type: ignore[attr-defined]
    return raw[:3].astype(np.float64), raw[3:].astype(np.float64)


def _rotmat_wxyz(q: np.ndarray) -> np.ndarray:
    return Rotation.from_quat(q, scalar_first=True).as_matrix()


@pytest.mark.smoke
@pytest.mark.gpu
@pytest.mark.skipif(
    not sapien_gpu_available(),
    reason="Requires Vulkan/GPU (SAPIEN); skipped on CPU-only machines",
)
class TestPickCubeContractMatchesSim:
    @pytest.fixture
    def env(self):
        from concerto.contracts.sim import make_sim_env

        env = make_sim_env("pickcube", sim_backend="physx_cpu", robot_init_qpos_noise=0.0)
        yield env
        env.close()

    def test_spaces_and_rates(self, env) -> None:
        c = contracts.get("pickcube")
        # ManiSkill batches ``observation_space`` even for one env; the single spaces are unbatched.
        obs_space = env.unwrapped.single_observation_space
        act_space = env.unwrapped.single_action_space
        assert obs_space.shape == (c.obs_dim,)
        assert act_space.shape == (c.action_dim,)
        assert np.all(act_space.low == -1.0)
        assert np.all(act_space.high == 1.0)
        assert env.unwrapped.control_freq == c.control_hz
        assert env.spec is not None
        assert env.spec.max_episode_steps == c.max_episode_steps

    def test_controller_config_matches_contract(self, env) -> None:
        c = contracts.get("pickcube")
        arm = env.unwrapped.agent.controller.controllers["arm"].config
        grip = env.unwrapped.agent.controller.controllers["gripper"].config
        assert arm.frame == c.delta_pose_frame
        assert arm.use_target is False
        pos, rot, gripper = (c.action_field(n) for n in ("delta_pos", "delta_rot", "gripper"))
        assert (arm.pos_lower, arm.pos_upper) == (pos.physical_low, pos.physical_high)
        # ManiSkill scales the rotation action by rot_lower; the contract encodes that sign.
        assert arm.rot_lower == rot.to_physical(1.0)
        assert (grip.lower, grip.upper) == (gripper.physical_low, gripper.physical_high)

    def test_reset_and_obs_layout(self, env) -> None:
        c = contracts.get("pickcube")
        obs, _ = env.reset(seed=0)
        obs = np.asarray(obs, dtype=np.float64).reshape(-1)
        sl = c.obs_slices()
        qpos = env.unwrapped.agent.robot.get_qpos()[0].cpu().numpy()
        np.testing.assert_allclose(obs[sl["qpos"]], qpos, atol=1e-6)
        np.testing.assert_allclose(qpos[:7], c.reset_joint_config, atol=1e-6)
        p, q = _tcp_pose(env)
        np.testing.assert_allclose(obs[sl["tcp_pose"]][:3], p, atol=1e-6)
        np.testing.assert_allclose(obs[sl["tcp_pose"]][3:], q, atol=1e-6)
        assert abs(np.linalg.norm(q) - 1.0) < 1e-5
        base = env.unwrapped.agent.robot.pose.p[0].cpu().numpy()
        np.testing.assert_allclose(base, c.base_position_in_world, atol=1e-6)
        cube = env.unwrapped.cube.pose.p[0].cpu().numpy()
        np.testing.assert_allclose(obs[sl["tcp_to_obj_pos"]], cube - p, atol=1e-5)

    def test_plus_x_action_moves_tcp_along_plus_x(self, env) -> None:
        env.reset(seed=0)
        p0, _ = _tcp_pose(env)
        action = np.zeros(7, dtype=np.float32)
        action[0] = 1.0
        for _ in range(_N_STEPS):
            env.step(action)
        p1, _ = _tcp_pose(env)
        delta = p1 - p0
        assert delta[0] > _MIN_MOVE_M
        assert abs(delta[1]) < delta[0] / 4
        assert abs(delta[2]) < delta[0] / 4

    def test_plus_z_rotation_action_turns_negatively_about_base_z(self, env) -> None:
        env.reset(seed=0)
        _, q0 = _tcp_pose(env)
        action = np.zeros(7, dtype=np.float32)
        action[5] = 1.0
        for _ in range(_N_STEPS):
            env.step(action)
        _, q1 = _tcp_pose(env)
        # Left-multiplied delta: R1 = R_delta @ R0, so R_delta is expressed in base axes.
        r_delta = _rotmat_wxyz(q1) @ _rotmat_wxyz(q0).T
        rotvec = Rotation.from_matrix(r_delta).as_rotvec()
        assert rotvec[2] < -_MIN_TURN_RAD
        assert abs(rotvec[0]) < abs(rotvec[2]) / 4
        assert abs(rotvec[1]) < abs(rotvec[2]) / 4
