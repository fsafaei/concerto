# SPDX-License-Identifier: Apache-2.0
# pyright: reportPrivateImportUsage=false, reportAttributeAccessIssue=false
#
# Tier-2 tests poke at ManiSkill v3 BaseEnv internals (``env.agent``,
# ``env.obs_mode``, ``env.action_space.spaces``) that aren't on the public
# :class:`gym.Env` Protocol; torch's stubs likewise omit some public
# symbols. Suppressed file-locally for readability.
"""Tier-2 SAPIEN-gated tests for :class:`Stage1PickPlaceEnv` (ADR-007 §Stage 1b).

Real ManiSkill v3 env construction + rollout coverage that needs a
Vulkan-capable host. Mirrors
:mod:`tests.integration.test_stage0_adapter_real`'s skipif pattern;
gated by :func:`chamber.utils.device.sapien_gpu_available` so CI runners
without a GPU pass without running these (the entire module is skipped
in that case).

The Tier-1 surface (condition resolution, control-model dispatch,
URDF-parse Jacobian agreement, OM channel filter shape preservation) is
covered without SAPIEN by
:mod:`tests.unit.test_stage1_pickplace_tier1`. This file's job is to
verify the **real** env behaves correctly when SAPIEN + the panda /
fetch / panda_partner URDFs are present:

1. **Per-condition env construction.** All four ``condition_id`` strings
   actually construct a SAPIEN scene; the agents_dict carries the
   expected bare uids (the strip-suffix shim works for both the
   two-panda and panda+fetch tuples).
2. **Per-condition action_space shape.** The trained policy adapter
   (P1.04 / P1.05) reads ``env.action_space[uid].shape[0]``; this test
   pins the shapes so a future ManiSkill point release that changes the
   panda or fetch controller config breaks loudly here, not silently in
   a training run.
3. **10-step rollout finite-reward smoke.** The reward path
   (panda-routed ``evaluate`` + ``compute_normalized_dense_reward``)
   returns finite values across 10 steps with zero-actions — verifies
   the multi-agent rewrite didn't introduce NaN / Inf.
4. **Synthesised force_torque under the OM-hetero condition.** The
   ``obs["extra"]["force_torque"]`` channel is present, shape ``(N, 6)``,
   and finite. Zero-mean sigma=:data:`SYNTHESISED_FT_NOISE_SIGMA_N` is the
   pin; magnitude variance across steps verifies the noise injection.
5. **Real-Jacobian / live-qpos round-trip.** Construct the env, step
   once, then invoke the panda's
   ``build_control_models()["panda_wristcam"].action_to_cartesian_accel``
   with a non-zero action and check the output is a finite Cartesian-
   acceleration vector — exercises the closure-via-env-reference
   workaround end-to-end (Q3 mod 3 ADR-004 §Open questions).

The 100-step + oracle-action success-detection test the founder's
prompt §2 Task D names is **out of scope for this PR**: it requires the
P1.04 trained-policy factory or a hand-crafted joint-space trajectory
that solves the cube-on-target task. Both are P1.04+ slices; this
file covers the env's structural correctness only.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch

from chamber.envs.stage1_pickplace import (
    SYNTHESISED_FT_NOISE_SIGMA_N,
    make_stage1_pickplace_env,
)
from chamber.utils.device import sapien_gpu_available
from concerto.safety.api import JacobianControlModel
from concerto.safety.cbf_qp import AgentSnapshot

pytestmark = [
    pytest.mark.slow,
    pytest.mark.gpu,
    pytest.mark.skipif(
        not sapien_gpu_available(),
        reason="Requires Vulkan/GPU (SAPIEN); skipped on CPU-only machines",
    ),
]

_AS_HOMO = "stage1_pickplace_panda_only_mappo_shared_param"
_AS_HETERO = "stage1_pickplace_panda_plus_fetch_ego_aht_happo_per_agent"
_OM_HOMO = "stage1_pickplace_vision_only"
_OM_HETERO = "stage1_pickplace_vision_plus_force_torque_plus_proprio"


class TestPerConditionConstruction:
    """All four condition_ids construct a SAPIEN scene (ADR-007 §Stage 1b)."""

    def test_as_homo_two_pandas_load(self) -> None:
        env = make_stage1_pickplace_env(condition_id=_AS_HOMO, episode_length=10)
        try:
            assert set(env.agent.agents_dict.keys()) == {
                "panda_wristcam",
                "panda_partner",
            }
        finally:
            env.close()

    def test_as_hetero_panda_plus_fetch_load(self) -> None:
        env = make_stage1_pickplace_env(condition_id=_AS_HETERO, episode_length=10)
        try:
            assert set(env.agent.agents_dict.keys()) == {"panda_wristcam", "fetch"}
        finally:
            env.close()

    def test_om_homo_loads_with_rgb_depth_obs_mode(self) -> None:
        env = make_stage1_pickplace_env(condition_id=_OM_HOMO, episode_length=10)
        try:
            assert "rgb" in env.obs_mode
            assert "depth" in env.obs_mode
        finally:
            env.close()

    def test_om_hetero_loads_with_rgb_depth_obs_mode(self) -> None:
        env = make_stage1_pickplace_env(condition_id=_OM_HETERO, episode_length=10)
        try:
            assert "rgb" in env.obs_mode
            assert "depth" in env.obs_mode
        finally:
            env.close()


class TestActionSpacePerUidShape:
    """Per-uid action_space shapes match URDF expectations (regression pin)."""

    def test_as_homo_two_pandas_action_shapes(self) -> None:
        env = make_stage1_pickplace_env(condition_id=_AS_HOMO, episode_length=10)
        try:
            assert isinstance(env.action_space, gym.spaces.Dict)
            assert set(env.action_space.spaces) == {"panda_wristcam", "panda_partner"}
            # Both pandas use pd_joint_delta_pos: 7 arm joints + 1 mimic
            # gripper = 8-D action.
            assert env.action_space.spaces["panda_wristcam"].shape == (8,)
            assert env.action_space.spaces["panda_partner"].shape == (8,)
        finally:
            env.close()

    def test_as_hetero_panda_eight_fetch_thirteen(self) -> None:
        env = make_stage1_pickplace_env(condition_id=_AS_HETERO, episode_length=10)
        try:
            assert env.action_space.spaces["panda_wristcam"].shape == (8,)
            # Fetch pd_joint_delta_pos:
            #   7 (arm) + 1 (gripper, mimic-joint config collapses 2 → 1) +
            #   3 (body: head_pan, head_tilt, torso_lift) +
            #   2 (base: PDBaseForwardVelControllerConfig compacts the 3
            #      root joints to forward + angular)
            # = 13-D action (mani_skill 3.0.1 fetch.py:101-105 body_joint_names;
            # confirmed 2026-05-18 by Tier-2 SAPIEN smoke on the RTX 2080 box).
            assert env.action_space.spaces["fetch"].shape == (13,)
        finally:
            env.close()


class TestTenStepRolloutFiniteReward:
    """Zero-action rollout returns finite obs + reward across 10 steps."""

    @pytest.mark.parametrize(
        "condition_id",
        [_AS_HOMO, _AS_HETERO, _OM_HOMO, _OM_HETERO],
    )
    def test_finite_reward_under_zero_action(self, condition_id: str) -> None:
        env = make_stage1_pickplace_env(condition_id=condition_id, episode_length=10, root_seed=0)
        try:
            obs, _ = env.reset(seed=0)
            assert obs is not None
            zero_action = {
                uid: np.zeros(box.shape, dtype=np.float32)
                for uid, box in env.action_space.spaces.items()
            }
            for _ in range(10):
                _, reward, terminated, truncated, _ = env.step(zero_action)
                reward_t = torch.as_tensor(reward).flatten().detach().cpu().numpy()
                assert np.all(np.isfinite(reward_t)), f"reward not finite at {condition_id}"
                if terminated or truncated:
                    break
        finally:
            env.close()


class TestEpisodeLengthTruncation:
    """The env emits ``truncated=True`` at ``episode_length`` (P-RC; ADR-007 §Stage 1b).

    Regression pin for root cause ENV-NO-TRUNCATION (firing
    ``2026-05-24-p1-05-9``): ManiSkill v3 ``BaseEnv.step`` hard-codes
    ``truncated=False`` (``mani_skill/envs/sapien_env.py`` L1069), so without
    :meth:`Stage1PickPlaceEnv.step`'s ``_elapsed_steps >= episode_length``
    override the training loop (which resets only on
    ``terminated | truncated``) never bounds an episode — the GAE then sees a
    non-terminating MDP. These pins assert the boundary fires at exactly
    ``episode_length`` and resets per episode.
    """

    @staticmethod
    def _first_truncation_step(env: object, *, max_steps: int) -> int | None:
        obs, _ = env.reset(seed=0)  # type: ignore[attr-defined]
        assert obs is not None
        zero_action = {
            uid: np.zeros(box.shape, dtype=np.float32)
            for uid, box in env.action_space.spaces.items()  # type: ignore[attr-defined]
        }
        for step_idx in range(max_steps):
            _, _, _, truncated, _ = env.step(zero_action)  # type: ignore[attr-defined]
            if bool(torch.as_tensor(truncated).flatten()[0]):
                return step_idx
        return None

    @pytest.mark.parametrize("episode_length", [5, 12])
    def test_truncates_at_episode_length(self, episode_length: int) -> None:
        env = make_stage1_pickplace_env(
            condition_id=_AS_HETERO, episode_length=episode_length, root_seed=0
        )
        try:
            first = self._first_truncation_step(env, max_steps=episode_length + 5)
            # 0-indexed: the ``episode_length``-th step is index ``episode_length - 1``.
            assert first == episode_length - 1, (
                f"expected first truncation at step index {episode_length - 1}, got {first}"
            )
        finally:
            env.close()

    def test_truncation_boundary_resets_per_episode(self) -> None:
        horizon = 6
        env = make_stage1_pickplace_env(
            condition_id=_AS_HETERO, episode_length=horizon, root_seed=0
        )
        try:
            first = self._first_truncation_step(env, max_steps=horizon + 3)
            assert first == horizon - 1, (
                f"first episode truncated at {first}, expected {horizon - 1}"
            )
            # A fresh reset must restore the horizon (BaseEnv resets _elapsed_steps);
            # if it did not reset, the second episode would not truncate at the horizon.
            second = self._first_truncation_step(env, max_steps=horizon + 3)
            assert second == horizon - 1, (
                f"second episode truncated at {second}, expected {horizon - 1}"
            )
        finally:
            env.close()


class TestSynthesisedForceTorque:
    """OM-hetero exposes the synthesised FT channel (Q2; ADR-007 §Revision history)."""

    def test_force_torque_in_obs_extra_with_six_dim_per_env(self) -> None:
        env = make_stage1_pickplace_env(condition_id=_OM_HETERO, episode_length=10)
        try:
            obs, _ = env.reset(seed=0)
            extra = obs["extra"]
            assert "force_torque" in extra, "synthesised FT channel missing under OM-hetero"
            ft = np.asarray(extra["force_torque"])
            # Shape (N_envs, 6) — 3-D force per finger x 2 fingers.
            assert ft.shape[-1] == 6
            assert np.all(np.isfinite(ft))
        finally:
            env.close()

    def test_force_torque_noise_injection_changes_across_steps(self) -> None:
        """sigma>0 means the FT channel varies step-to-step even at rest."""
        env = make_stage1_pickplace_env(condition_id=_OM_HETERO, episode_length=5, root_seed=0)
        try:
            obs, _ = env.reset(seed=0)
            zero_action = {
                uid: np.zeros(box.shape, dtype=np.float32)
                for uid, box in env.action_space.spaces.items()
            }
            ft_history: list[np.ndarray] = []
            ft_history.append(np.asarray(obs["extra"]["force_torque"]).copy())
            for _ in range(5):
                obs, _, _, _, _ = env.step(zero_action)
                ft_history.append(np.asarray(obs["extra"]["force_torque"]).copy())
            stacked = np.stack(ft_history, axis=0)
            # At least one component should differ across the 6 reads
            # given sigma=0.5 N noise; tolerate the no-contact case (no
            # contact force -> no signal beyond pure noise) by checking
            # any variation across steps.
            assert stacked.std() > 0.0, (
                f"FT channel did not vary across 6 reads at "
                f"sigma={SYNTHESISED_FT_NOISE_SIGMA_N} N — noise injection not firing"
            )
        finally:
            env.close()


class TestJacobianClosureRoundTrip:
    """End-to-end JacobianControlModel with closure-via-env-reference (Q3 mod 3)."""

    def test_panda_jacobian_action_to_cartesian_accel_finite_after_reset(self) -> None:
        """Live qpos -> Jacobian -> Cartesian acceleration round-trip is finite.

        Constructs the env, resets it (which populates ``_latest_qpos``
        via :meth:`_before_simulation_step`), then exercises the panda's
        :class:`JacobianControlModel.action_to_cartesian_accel` with a
        small joint-space action. The output must be a finite
        Cartesian-acceleration vector — the alternative is that the
        closure couldn't read live qpos, which would surface here as a
        RuntimeError ("panda_wristcam qpos not yet cached").
        """
        env = make_stage1_pickplace_env(condition_id=_AS_HETERO, episode_length=10, root_seed=0)
        try:
            env.reset(seed=0)
            # Step once so _before_simulation_step has run and the qpos
            # cache is primed.
            zero_action = {
                uid: np.zeros(box.shape, dtype=np.float32)
                for uid, box in env.action_space.spaces.items()
            }
            env.step(zero_action)
            models = env.build_control_models()  # type: ignore[attr-defined]
            panda_model = models["panda_wristcam"]
            assert isinstance(panda_model, JacobianControlModel)
            assert panda_model.jacobian_fn is not None
            # Small joint-space action — full 8-D pd_joint_delta_pos
            # action vector (7 arm joints + 1 mimic-gripper). The
            # padded Jacobian's gripper column is zero by construction
            # (chamber.envs.stage1_pickplace._panda_jacobian), so the
            # mimic-gripper component drops out of the Cartesian-
            # acceleration output regardless of its value; we still
            # need an 8-element vector to satisfy the matmul shape.
            action = np.zeros(panda_model.action_dim, dtype=np.float64)
            action[0] = 0.01
            snap = AgentSnapshot(
                position=np.zeros(3, dtype=np.float64),
                velocity=np.zeros(3, dtype=np.float64),
                radius=0.05,
            )
            accel = panda_model.action_to_cartesian_accel(snap, action)
            assert accel.shape == (3,)
            assert np.all(np.isfinite(accel)), "Cartesian accel not finite"
        finally:
            env.close()
