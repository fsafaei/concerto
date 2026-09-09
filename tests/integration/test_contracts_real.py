# SPDX-License-Identifier: Apache-2.0
"""Tier-2: the real ManiSkill ``PickCube-v1`` env matches every ``pickcube`` contract version.

Gated on :func:`chamber.utils.device.sapien_gpu_available`; skipped on
Vulkan-less hosts. On a SAPIEN-capable host it pins, per contract version,
the spaces and horizon, the field layout, the controller frame and bounds
(times the ``ScaleDeltaActions`` scale), the reset configuration, and the
sign and size of the delta-pose actions. It also builds the registered
Gymnasium id the way a plain-Gymnasium trainer does and checks what that
consumer sees.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from chamber.utils.device import sapien_gpu_available
from concerto import contracts

_TRACKING_FRACTION = 0.5  # the PD controller must realise at least this much of a commanded delta
_ATOL_M = 1e-3
_ATOL_RAD = 1e-3


def _tcp_pose(env: object) -> tuple[np.ndarray, np.ndarray]:
    raw = env.unwrapped.agent.tcp_pose.raw_pose[0].cpu().numpy()  # type: ignore[attr-defined]
    return raw[:3].astype(np.float64), raw[3:].astype(np.float64)


def _rotmat_wxyz(q: np.ndarray) -> np.ndarray:
    return Rotation.from_quat(q, scalar_first=True).as_matrix()


def _n_steps(c: contracts.TaskContract) -> int:
    return 5 if c.version == 1 else 10


def _run_out(env: gym.Env, action: np.ndarray) -> tuple[int, dict]:
    """Step ``action`` until the episode ends; ``(steps, last info)``."""
    steps = 0
    while True:
        _, _, terminated, truncated, info = env.step(action)
        steps += 1
        if terminated or truncated:
            return steps, info


@pytest.mark.smoke
@pytest.mark.gpu
@pytest.mark.skipif(
    not sapien_gpu_available(),
    reason="Requires Vulkan/GPU (SAPIEN); skipped on CPU-only machines",
)
class TestPickCubeContractMatchesSim:
    @pytest.fixture(params=[1, 2], ids=lambda v: f"v{v}")
    def contract(self, request):
        return contracts.get("pickcube", request.param)

    @pytest.fixture
    def env(self, contract):
        from concerto.contracts.sim import make_sim_env

        env = make_sim_env(contract, sim_backend="physx_cpu", robot_init_qpos_noise=0.0)
        yield env
        env.close()

    def test_spaces_and_rates(self, env, contract) -> None:
        from mani_skill.utils.gym_utils import find_max_episode_steps_value

        c = contract
        # ManiSkill batches ``observation_space`` even for one env; the single spaces are unbatched.
        obs_space = env.unwrapped.single_observation_space
        act_space = env.unwrapped.single_action_space
        assert obs_space.shape == (c.obs_dim,)
        assert act_space.shape == (c.action_dim,)
        assert np.all(act_space.low == -1.0)
        assert np.all(act_space.high == 1.0)
        assert env.unwrapped.control_freq == c.control_hz
        # gym.make(max_episode_steps=...) leaves env.spec unset; the time-limit wrapper knows.
        assert find_max_episode_steps_value(env) == c.max_episode_steps
        assert env.get_wrapper_attr("contract") is c

    def test_controller_bounds_times_wrapper_scale_match_contract(self, env, contract) -> None:
        c = contract
        arm = env.unwrapped.agent.controller.controllers["arm"].config
        grip = env.unwrapped.agent.controller.controllers["gripper"].config
        assert arm.frame == c.delta_pose_frame
        assert arm.use_target is False
        pos, rot, gripper = (c.action_field(n) for n in ("delta_pos", "delta_rot", "gripper"))
        assert arm.pos_upper * env.pos_scale == pytest.approx(pos.to_physical(1.0))
        assert arm.pos_lower * env.pos_scale == pytest.approx(pos.to_physical(-1.0))
        # ManiSkill multiplies the unit-norm-clipped rotation action by rot_lower; the contract
        # encodes that sign and the wrapper only shrinks the magnitude.
        assert arm.rot_lower * env.rot_scale == pytest.approx(rot.to_physical(1.0))
        assert (grip.lower, grip.upper) == (gripper.physical_low, gripper.physical_high)

    def test_reset_and_obs_layout(self, env, contract) -> None:
        c = contract
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

    def test_plus_x_action_moves_tcp_along_plus_x(self, env, contract) -> None:
        n = _n_steps(contract)
        bound = contract.action_field("delta_pos").to_physical(1.0)
        env.reset(seed=0)
        p0, _ = _tcp_pose(env)
        action = np.zeros(7, dtype=np.float32)
        action[0] = 1.0
        for _ in range(n):
            env.step(action)
        p1, _ = _tcp_pose(env)
        delta = p1 - p0
        print(f"v{contract.version}: {delta[0] / n * 1e3:.2f} mm per step of {bound * 1e3:.1f} mm")
        assert delta[0] <= n * bound + _ATOL_M, "the sim never moves more than the contract says"
        assert delta[0] >= _TRACKING_FRACTION * n * bound
        assert abs(delta[1]) < delta[0] / 4
        assert abs(delta[2]) < delta[0] / 4

    def test_plus_z_rotation_action_turns_negatively_about_base_z(self, env, contract) -> None:
        n = _n_steps(contract)
        bound = -contract.action_field("delta_rot").to_physical(1.0)  # +1 turns negatively
        assert bound > 0
        env.reset(seed=0)
        _, q0 = _tcp_pose(env)
        action = np.zeros(7, dtype=np.float32)
        action[5] = 1.0
        for _ in range(n):
            env.step(action)
        _, q1 = _tcp_pose(env)
        # Left-multiplied delta: R1 = R_delta @ R0, so R_delta is expressed in base axes.
        r_delta = _rotmat_wxyz(q1) @ _rotmat_wxyz(q0).T
        rotvec = Rotation.from_matrix(r_delta).as_rotvec()
        turned = -rotvec[2]
        print(f"v{contract.version}: {turned / n:.4f} rad per step of {bound:.3f} rad")
        assert turned <= n * bound + _ATOL_RAD, "the sim never turns more than the contract says"
        assert turned >= _TRACKING_FRACTION * n * bound
        assert abs(rotvec[0]) < turned / 4
        assert abs(rotvec[1]) < turned / 4

    def test_registered_gym_id_builds_the_training_env(self, contract) -> None:
        import concerto.contracts.sim  # noqa: F401  (registers the ids)

        c = contract
        env = gym.make(c.gym_id, robot_init_qpos_noise=0.0)
        try:
            assert env.spec is not None
            assert env.spec.id == c.gym_id
            assert env.get_wrapper_attr("contract") is c
            assert env.action_space == gym.spaces.Box(-1.0, 1.0, (c.action_dim,), np.float32)
            assert env.observation_space.shape == (c.obs_dim,)
            obs, _ = env.reset(seed=0)
            assert isinstance(obs, np.ndarray)
            assert obs.dtype == np.float32
            assert obs.shape == (c.obs_dim,)
            obs, reward, terminated, truncated, info = env.step(np.zeros(c.action_dim, np.float32))
            assert isinstance(obs, np.ndarray)
            assert isinstance(reward, float)
            assert isinstance(terminated, bool | np.bool_)
            assert isinstance(truncated, bool | np.bool_)
            assert "success" in info
            # Zero actions never succeed, so the horizon is what ends the episode.
            steps, _ = _run_out(env, np.zeros(c.action_dim, np.float32))
            assert steps + 1 == c.max_episode_steps
        finally:
            env.close()

    def test_roborl_wrapper_stack_reports_episode_statistics(self, contract) -> None:
        import concerto.contracts.sim  # noqa: F401  (registers the ids)

        c = contract
        # roborl's make_env adds RecordEpisodeStatistics; FlashSAC adds RescaleAction onto [-1, 1].
        env = gym.wrappers.RescaleAction(
            gym.wrappers.RecordEpisodeStatistics(gym.make(c.gym_id, robot_init_qpos_noise=0.0)),
            min_action=np.float32(-1.0),
            max_action=np.float32(1.0),
        )
        try:
            env.reset(seed=0)
            steps, info = _run_out(env, np.zeros(c.action_dim, np.float32))
            assert steps == c.max_episode_steps
            assert int(info["episode"]["l"]) == c.max_episode_steps
            assert np.isfinite(float(info["episode"]["r"]))
        finally:
            env.close()
