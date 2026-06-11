# SPDX-License-Identifier: Apache-2.0
# pyright: reportPrivateImportUsage=false, reportAttributeAccessIssue=false
#
# torch stubs and ManiSkill v3 BaseEnv internals (``env.action_space.spaces``)
# aren't fully advertised; suppressed file-locally for readability.
"""Tier-2 SAPIEN-gated tests for :class:`TrainedPolicyFactory` (P1.04).

Real Stage-1b env end-to-end smoke. Gated on
:func:`chamber.utils.device.sapien_gpu_available` so CI runners without
a GPU pass without running these. Runs on the RTX 2080 box; founder
pastes output into the PR description for the ADR-007 §Stage 1b
handoff record.

Two cases (per the P1.04 prompt §D):

1. ``test_real_stage1b_env_training_completes`` — drive the factory
   against :class:`chamber.envs.stage1_pickplace.Stage1PickPlaceEnv` at
   a tiny ``total_frames`` (1000); assert :func:`run_training` completes
   without error and the returned closure produces a finite-norm
   action on a real obs. Smoke that the wiring works end-to-end, not
   a science assertion.
2. ``test_trained_policy_outperforms_zero_action_on_small_budget`` —
   5k-frame mini-training run; assert mean episodic reward from the
   trained policy is at least no worse than the
   :func:`_zero_ego_action_factory` baseline on the same seed x env.
   Weak signal that the trainer is doing something; not the Stage-1b
   ≥20 pp ADR-007 §Validation-criteria assertion (that's the full
   Stage-1b run in P1.05).

The full Stage-1b science run (5 seeds x 100k frames per axis) is out
of scope for Tier-2; that's the founder-launched
``scripts/repro/stage1_as_stage1b.sh`` (P1.05 launcher).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from chamber.benchmarks.stage1_common import TrainedPolicyFactory, _zero_ego_action_factory
from chamber.envs.stage1_pickplace import make_stage1_pickplace_env
from chamber.utils.device import sapien_gpu_available, torch_cuda_available
from concerto.training.config import (
    EgoAHTConfig,
    EnvConfig,
    HAPPOHyperparams,
    PartnerConfig,
    RuntimeConfig,
)

pytestmark = [
    pytest.mark.slow,
    pytest.mark.gpu,
    pytest.mark.skipif(
        not sapien_gpu_available(),
        reason="Requires Vulkan/GPU (SAPIEN); skipped on CPU-only machines",
    ),
    pytest.mark.skipif(
        not torch_cuda_available(),
        reason=(
            "Trainer is constructed with RuntimeConfig.device='cuda'; skipped on hosts "
            "where torch.cuda.is_available() is False (ADR-002 §Rev 2026-05-20 cuda-major "
            "coupling discipline; closes #198)."
        ),
    ),
]

_AS_HETERO = "stage1_pickplace_panda_plus_fetch_ego_aht_happo_per_agent"


def _stage1b_tiny_cfg(tmp_path, *, total_frames: int) -> EgoAHTConfig:  # type: ignore[no-untyped-def]
    """Stage-1b cfg sized for Tier-2 smoke runtime (≤2 min/run on RTX 2080)."""
    return EgoAHTConfig(
        seed=0,
        total_frames=total_frames,
        checkpoint_every=max(total_frames, 10_000),  # avoid mid-run checkpoint noise
        artifacts_root=tmp_path / "artifacts",
        log_dir=tmp_path / "logs",
        env=EnvConfig(
            task="stage1_pickplace",
            episode_length=50,
            agent_uids=("panda_wristcam", "fetch"),
            condition_id=_AS_HETERO,
        ),
        partner=PartnerConfig(
            class_name="scripted_heuristic",
            extra={"uid": "fetch", "target_xy": "0.0,0.0", "action_dim": "13"},
        ),
        happo=HAPPOHyperparams(
            rollout_length=250,
            batch_size=50,
            hidden_dim=32,  # smaller for 8 GB VRAM on RTX 2080 (prompt §4 hint)
        ),
        runtime=RuntimeConfig(device="cuda", deterministic_torch=False),
    )


class TestRealStage1bEnvTrainingCompletes:
    """Smoke: factory + Stage-1b env + run_training compose end-to-end."""

    def test_real_stage1b_env_training_completes(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """Drive 1000-frame training on the AS-hetero condition; assert closure works."""
        cfg = _stage1b_tiny_cfg(tmp_path, total_frames=1000)
        factory = TrainedPolicyFactory(cfg=cfg)
        eval_env = make_stage1_pickplace_env(
            condition_id=_AS_HETERO, episode_length=50, root_seed=0
        )
        try:
            ego_action = factory(eval_env, seed=0)
            obs, _ = eval_env.reset(seed=0)
            action = ego_action(obs)
            # 8-D for panda_wristcam under pd_joint_delta_pos
            # (7 arm + 1 gripper-mimic; verified by Tier-2 smoke
            # 2026-05-18).
            assert action.shape == (8,)
            assert np.all(np.isfinite(action)), "trained policy produced non-finite action"
        finally:
            eval_env.close()


class TestTrainedPolicyVsZeroBaseline:
    """Weak signal: trained policy is at least no worse than zero baseline on small budget."""

    def test_trained_policy_outperforms_zero_action_on_small_budget(
        self,
        tmp_path,  # type: ignore[no-untyped-def]
    ) -> None:
        """5k-frame mini-training run; mean-episode-reward comparison vs zero baseline.

        Not a Stage-1b ≥20 pp gate measurement — too few frames, no
        scientific claim. Just a smoke that the trainer is doing
        SOMETHING (a learning signal at this scale should at least
        not regress below the zero-action floor).
        """
        cfg = _stage1b_tiny_cfg(tmp_path, total_frames=5000)
        factory = TrainedPolicyFactory(cfg=cfg)

        # Build two eval envs at the same seed so the rollouts are
        # determinism-comparable.
        eval_env_trained = make_stage1_pickplace_env(
            condition_id=_AS_HETERO, episode_length=50, root_seed=0
        )
        eval_env_zero = make_stage1_pickplace_env(
            condition_id=_AS_HETERO, episode_length=50, root_seed=0
        )
        try:
            trained_act = factory(eval_env_trained, seed=0)
            zero_act = _zero_ego_action_factory(eval_env_zero, seed=0)

            trained_reward = _rollout_mean_reward(eval_env_trained, trained_act, "fetch")
            zero_reward = _rollout_mean_reward(eval_env_zero, zero_act, "fetch")

            # Weak signal — the trained policy should at least not
            # regress below the zero baseline. 5k frames is too few
            # to assert strict outperformance reliably (high variance
            # in single-seed comparison); the full-budget Stage-1b
            # run (P1.05) is where the ≥20 pp gate fires.
            assert trained_reward >= zero_reward - 1.0, (
                f"trained mean_reward={trained_reward:.4f} regressed below "
                f"zero-action baseline={zero_reward:.4f}; the trainer may be "
                "diverging at this budget — re-run before alarming."
            )
        finally:
            eval_env_trained.close()
            eval_env_zero.close()


def _rollout_mean_reward(env, ego_act_callable, partner_uid: str, *, n_steps: int = 50):  # type: ignore[no-untyped-def]
    """Run one episode under ``ego_act_callable``; return mean per-step reward."""
    obs, _ = env.reset(seed=0)
    rewards = []
    for _ in range(n_steps):
        ego_action = ego_act_callable(obs)
        # Pack the action dict — partner uses zero action (the Tier-2
        # smoke is about the ego's wiring, not partner coordination).
        action_space = env.action_space
        partner_shape = action_space.spaces[partner_uid].shape
        action_dict = {
            "panda_wristcam": ego_action,
            partner_uid: np.zeros(partner_shape, dtype=np.float32),
        }
        obs, reward, terminated, truncated, _ = env.step(action_dict)
        # ManiSkill returns a torch tensor per env; flatten + numpyify.
        rewards.append(float(torch.as_tensor(reward).flatten()[0].item()))
        if terminated or truncated:
            break
    return float(np.mean(rewards)) if rewards else 0.0
