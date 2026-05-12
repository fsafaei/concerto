# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``chamber.benchmarks.training_runner`` (T4b.11).

Covers ADR-002 §Decisions (env / partner construction) + plan/05 §3.5
(the chamber-side bridge between concrete env / partner classes and
the ``concerto.training.ego_aht.train`` loop).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from chamber.benchmarks.stage0_smoke import _SMOKE_ROBOT_UIDS
from chamber.benchmarks.training_runner import (
    build_env,
    build_partner,
    run_training,
)
from chamber.envs.errors import ChamberEnvCompatibilityError
from chamber.envs.mpe_cooperative_push import MPECooperativePushEnv
from chamber.partners.heuristic import ScriptedHeuristicPartner
from chamber.utils.device import sapien_gpu_available
from concerto.training.config import (
    EgoAHTConfig,
    EnvConfig,
    HAPPOHyperparams,
    PartnerConfig,
    RuntimeConfig,
)
from concerto.training.ego_aht import RewardCurve

if TYPE_CHECKING:
    from pathlib import Path


def _tiny_cfg(tmp_path: Path, *, total_frames: int = 50) -> EgoAHTConfig:
    return EgoAHTConfig(
        seed=0,
        total_frames=total_frames,
        checkpoint_every=50,
        artifacts_root=tmp_path / "artifacts",
        log_dir=tmp_path / "logs",
        env=EnvConfig(task="mpe_cooperative_push", episode_length=20),
        partner=PartnerConfig(
            class_name="scripted_heuristic",
            extra={"uid": "partner", "target_xy": "0.0,0.0", "action_dim": "2"},
        ),
        happo=HAPPOHyperparams(rollout_length=25),
        # Pin device=cpu for unit tests (M4b-9b): the default RuntimeConfig
        # device="auto" resolves to MPS on Mac, which has known numerical
        # quirks under PPO that the test fixtures aren't tuned for.
        runtime=RuntimeConfig(device="cpu"),
    )


class TestBuildEnv:
    def test_mpe_cooperative_push(self) -> None:
        """plan/05 §3.5: the empirical-guarantee env constructs."""
        env = build_env(EnvConfig(task="mpe_cooperative_push"), root_seed=0)
        assert isinstance(env, MPECooperativePushEnv)
        obs, _ = env.reset(seed=0)
        assert "agent" in obs

    def test_stage0_smoke_rejects_non_smoke_uids(self) -> None:
        """T4b.3: the Stage-0 adapter rejects agent_uids that aren't ADR-001 rig robots."""
        with pytest.raises(ValueError, match="_SMOKE_ROBOT_UIDS"):
            build_env(EnvConfig(task="stage0_smoke"), root_seed=0)

    @pytest.mark.skipif(
        sapien_gpu_available(),
        reason="Vulkan is available; CPU-only error path not reachable here",
    )
    def test_stage0_smoke_raises_compat_error_without_gpu(self) -> None:
        """ADR-001 §Risks: build_env for stage0_smoke surfaces the SAPIEN-missing error."""
        with pytest.raises(ChamberEnvCompatibilityError, match="SAPIEN/Vulkan"):
            build_env(
                EnvConfig(
                    task="stage0_smoke",
                    episode_length=10,
                    agent_uids=(_SMOKE_ROBOT_UIDS[0], _SMOKE_ROBOT_UIDS[1]),
                ),
                root_seed=0,
            )

    def test_unknown_task_raises(self) -> None:
        """ADR-002 §Decisions: unknown task names fail loud."""
        with pytest.raises(ValueError, match="Unknown env task"):
            build_env(EnvConfig(task="not_a_task"), root_seed=0)

    def test_root_seed_routes_to_env(self) -> None:
        """P6: the root_seed flows through to the env's deterministic RNG."""
        env_a = build_env(EnvConfig(task="mpe_cooperative_push"), root_seed=42)
        env_b = build_env(EnvConfig(task="mpe_cooperative_push"), root_seed=42)
        obs_a, _ = env_a.reset(seed=0)
        obs_b, _ = env_b.reset(seed=0)
        # Identical root_seed → identical initial state.
        for uid in obs_a["agent"]:
            assert obs_a["agent"][uid]["state"].tolist() == obs_b["agent"][uid]["state"].tolist()


class TestBuildPartner:
    def test_scripted_heuristic(self) -> None:
        """ADR-009 §Decision: PartnerConfig builds via the M4a registry."""
        partner = build_partner(
            PartnerConfig(
                class_name="scripted_heuristic",
                extra={"uid": "partner", "target_xy": "0.0,0.0", "action_dim": "2"},
            )
        )
        assert isinstance(partner, ScriptedHeuristicPartner)
        # The runner returns PartnerLike; narrow with isinstance to access spec.
        assert partner.spec.class_name == "scripted_heuristic"

    def test_unknown_class_raises_keyerror(self) -> None:
        """ADR-009 §Decision: unknown class names raise KeyError listing knowns."""
        with pytest.raises(KeyError, match="not_registered"):
            build_partner(PartnerConfig(class_name="not_registered", extra={"uid": "x"}))

    def test_extra_round_trips_via_partner_spec(self) -> None:
        """plan/04 §3.1: extra metadata propagates from config to spec."""
        partner = build_partner(
            PartnerConfig(
                class_name="scripted_heuristic",
                extra={"uid": "partner", "target_xy": "0.5,-0.5", "action_dim": "4"},
            )
        )
        # The runner returns the structural PartnerLike type; cast to
        # the concrete chamber side for spec-attribute access.
        assert isinstance(partner, ScriptedHeuristicPartner)
        assert partner.spec.extra["target_xy"] == "0.5,-0.5"
        assert partner.spec.extra["action_dim"] == "4"


class TestRunTrainingEndToEnd:
    def test_run_training_produces_reward_curve(self, tmp_path: Path) -> None:
        """T4b.11: end-to-end run wires env + partner + train()."""
        cfg = _tiny_cfg(tmp_path, total_frames=50)
        curve = run_training(cfg, repo_root=tmp_path)
        assert isinstance(curve, RewardCurve)
        assert len(curve.per_step_ego_rewards) == 50

    def test_run_training_emits_checkpoint(self, tmp_path: Path) -> None:
        """T4b.12: the chamber-side runner produces .pt + sidecar artefacts."""
        cfg = _tiny_cfg(tmp_path, total_frames=50)
        curve = run_training(cfg, repo_root=tmp_path)
        assert len(curve.checkpoint_paths) == 1
        ckpt = curve.checkpoint_paths[0]
        assert ckpt.exists()
        assert (ckpt.parent / (ckpt.name + ".json")).exists()

    def test_run_training_seed_reproduces_reward_curve(self, tmp_path: Path) -> None:
        """plan/05 §6 criterion 7: end-to-end determinism — same seed → same rewards.

        Real env + real heuristic partner + RandomEgoTrainer all derived
        from cfg.seed. Two runs at the same seed produce byte-identical
        per-step reward curves.
        """
        cfg_a = _tiny_cfg(tmp_path / "a", total_frames=40)
        cfg_b = _tiny_cfg(tmp_path / "b", total_frames=40)
        curve_a = run_training(cfg_a, repo_root=tmp_path)
        curve_b = run_training(cfg_b, repo_root=tmp_path)
        assert curve_a.per_step_ego_rewards == curve_b.per_step_ego_rewards


class TestPublicSurface:
    def test_module_exports(self) -> None:
        from chamber.benchmarks import training_runner

        for name in ("build_env", "build_partner", "run_training"):
            assert hasattr(training_runner, name)
