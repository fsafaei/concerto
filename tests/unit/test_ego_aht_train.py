# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``concerto.training.ego_aht`` (T4b.11).

Covers ADR-002 §Decisions (the canonical training entry point) and
plan/05 §3.5 (the loop that T4b.13's empirical-guarantee experiment
runs against; the loop that T4b.14's zoo-seed run produces the
``happo_seed7_step50k.pt`` artefact through).

Tests for the chamber-side env / partner builders live in
``tests/unit/test_training_runner.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import numpy as np
import pytest

from concerto.training.checkpoints import load_checkpoint
from concerto.training.config import (
    EgoAHTConfig,
    EnvConfig,
    HAPPOHyperparams,
    PartnerConfig,
)
from concerto.training.ego_aht import (
    EnvLike,
    PartnerLike,
    RandomEgoTrainer,
    RewardCurve,
    train,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


class _FakeEnv:
    """Minimal :class:`EnvLike` for trainer-loop tests (no chamber import)."""

    def __init__(
        self,
        *,
        agent_uids: tuple[str, str] = ("ego", "partner"),
        episode_length: int = 20,
    ) -> None:
        self._uids = agent_uids
        self._episode_length = episode_length
        self._step_count = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        del seed, options
        self._step_count = 0
        return self._obs(), {}

    def step(
        self,
        action: Mapping[str, Any],
    ) -> tuple[Mapping[str, Any], float, bool, bool, Mapping[str, Any]]:
        del action
        self._step_count += 1
        truncated = self._step_count >= self._episode_length
        return self._obs(), -1.0, False, truncated, {}

    def _obs(self) -> Mapping[str, Any]:
        return {"agent": {uid: {"state": np.zeros(10, dtype=np.float32)} for uid in self._uids}}


class _FakePartner:
    """Minimal :class:`PartnerLike` for trainer-loop tests."""

    def __init__(self) -> None:
        self.spec: Any = None
        self.reset_calls = 0
        self.act_calls = 0

    def reset(self, *, seed: int | None = None) -> None:
        del seed
        self.reset_calls += 1

    def act(
        self,
        obs: Mapping[str, Any],
        *,
        deterministic: bool = True,
    ) -> np.ndarray:
        del obs, deterministic
        self.act_calls += 1
        return np.zeros(2, dtype=np.float32)


def _tiny_cfg(tmp_path: Path, *, total_frames: int = 100) -> EgoAHTConfig:
    """Build a minimal config that exercises the loop in <1 second."""
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
    )


class TestProtocols:
    def test_envlike_is_runtime_checkable(self) -> None:
        """T4b.11: structural typing — chamber's MPE env satisfies EnvLike."""
        env: Any = _FakeEnv()
        assert isinstance(env, EnvLike)

    def test_partnerlike_is_runtime_checkable(self) -> None:
        """T4b.11: structural typing — M4a's FrozenPartner satisfies PartnerLike."""
        partner: Any = _FakePartner()
        assert isinstance(partner, PartnerLike)


class TestRandomEgoTrainer:
    def test_act_returns_correct_shape_and_dtype(self) -> None:
        """T4b.11: reference trainer outputs match the env's action shape."""
        trainer = RandomEgoTrainer(ego_uid="ego", action_dim=2, root_seed=0)
        action = trainer.act({})
        assert action.shape == (2,)
        assert action.dtype == np.float32

    def test_act_deterministic_returns_zeros(self) -> None:
        """deterministic=True is the canonical evaluation mode."""
        trainer = RandomEgoTrainer(ego_uid="ego", action_dim=3, root_seed=0)
        action = trainer.act({}, deterministic=True)
        np.testing.assert_array_equal(action, np.zeros(3, dtype=np.float32))

    def test_act_components_in_unit_box(self) -> None:
        """plan/05 §3.5: random sampling stays in the env's [-1, 1] action box."""
        trainer = RandomEgoTrainer(ego_uid="ego", action_dim=2, root_seed=42)
        for _ in range(50):
            action = trainer.act({})
            assert np.all(action >= -1.0)
            assert np.all(action <= 1.0)

    def test_same_seed_reproduces_actions(self) -> None:
        """P6: identical root_seed → byte-identical action stream."""
        a = RandomEgoTrainer(ego_uid="ego", action_dim=2, root_seed=7)
        b = RandomEgoTrainer(ego_uid="ego", action_dim=2, root_seed=7)
        for _ in range(10):
            np.testing.assert_array_equal(a.act({}), b.act({}))

    def test_distinct_seeds_diverge(self) -> None:
        a = RandomEgoTrainer(ego_uid="ego", action_dim=2, root_seed=0)
        b = RandomEgoTrainer(ego_uid="ego", action_dim=2, root_seed=1)
        assert not np.array_equal(a.act({}), b.act({}))

    def test_observe_and_update_are_no_ops(self) -> None:
        """Reference trainer is stateless; observe/update do not raise."""
        trainer = RandomEgoTrainer(ego_uid="ego", action_dim=2, root_seed=0)
        trainer.observe({}, reward=-1.0, done=False)
        trainer.update()

    def test_state_dict_is_empty(self) -> None:
        """Reference trainer has no learnable parameters."""
        trainer = RandomEgoTrainer(ego_uid="ego", action_dim=2, root_seed=0)
        assert trainer.state_dict() == {}


class TestRewardCurve:
    def test_default_construction(self) -> None:
        curve = RewardCurve(run_id="0" * 16)
        assert curve.per_step_ego_rewards == []
        assert curve.per_episode_ego_rewards == []
        assert curve.checkpoint_paths == []

    def test_frozen(self) -> None:
        """Immutability: the curve is the run's reproducible artefact."""
        import dataclasses

        curve = RewardCurve(run_id="0" * 16)
        with pytest.raises(dataclasses.FrozenInstanceError):
            curve.run_id = "x"  # type: ignore[misc]


class TestTrainEndToEnd:
    def test_train_produces_reward_curve_with_random_trainer(self, tmp_path: Path) -> None:
        """T4b.11: the canonical loop runs end-to-end with the reference trainer."""
        cfg = _tiny_cfg(tmp_path, total_frames=100)
        curve = train(cfg, env=_FakeEnv(), partner=_FakePartner(), repo_root=tmp_path)
        assert isinstance(curve, RewardCurve)
        assert len(curve.per_step_ego_rewards) == 100
        assert len(curve.per_episode_ego_rewards) >= 1

    def test_train_emits_checkpoints_on_cadence(self, tmp_path: Path) -> None:
        """T4b.11: checkpoint_every fires the save_checkpoint path."""
        cfg = _tiny_cfg(tmp_path, total_frames=100)
        # checkpoint_every=50 → expect 2 checkpoints over 100 frames.
        curve = train(cfg, env=_FakeEnv(), partner=_FakePartner(), repo_root=tmp_path)
        assert len(curve.checkpoint_paths) == 2
        for path in curve.checkpoint_paths:
            assert path.exists()
            assert (path.parent / (path.name + ".json")).exists()

    def test_train_writes_jsonl_log(self, tmp_path: Path) -> None:
        """T4b.10: every step / rollout / checkpoint emits a JSONL line."""
        cfg = _tiny_cfg(tmp_path, total_frames=50)
        curve = train(cfg, env=_FakeEnv(), partner=_FakePartner(), repo_root=tmp_path)
        jsonl_path = cfg.log_dir / f"{curve.run_id}.jsonl"
        assert jsonl_path.exists()
        lines = jsonl_path.read_text(encoding="utf-8").splitlines()
        events = [line for line in lines if line]
        assert len(events) >= 4  # start + at least one rollout + checkpoint + end

    def test_train_checkpoint_round_trips_via_load_checkpoint(self, tmp_path: Path) -> None:
        """T4b.12 contract: checkpoints saved by train() load back cleanly."""
        cfg = _tiny_cfg(tmp_path, total_frames=50)
        curve = train(cfg, env=_FakeEnv(), partner=_FakePartner(), repo_root=tmp_path)
        assert len(curve.checkpoint_paths) >= 1
        ckpt_path = curve.checkpoint_paths[0]
        relative = ckpt_path.relative_to(cfg.artifacts_root)
        uri = f"local://{relative}"
        state_dict, metadata = load_checkpoint(uri=uri, artifacts_root=cfg.artifacts_root)
        assert metadata.run_id == curve.run_id
        assert metadata.seed == cfg.seed
        assert state_dict == {}

    def test_train_partner_reset_called_at_start(self, tmp_path: Path) -> None:
        """ADR-009 §Decision: partner.reset(seed) fires at training start."""
        cfg = _tiny_cfg(tmp_path, total_frames=10)
        partner = _FakePartner()
        train(cfg, env=_FakeEnv(), partner=partner, repo_root=tmp_path)
        assert partner.reset_calls >= 1
        assert partner.act_calls == 10

    def test_train_seed_reproduces_per_step_rewards(self, tmp_path: Path) -> None:
        """plan/05 §6 criterion 7: same cfg.seed → byte-identical reward curve.

        With the fake env returning -1.0 always, the test pins the LOOP
        determinism rather than env determinism — so what we exercise is
        that two identical seed runs produce the same per-step reward
        list shape and values.
        """
        cfg_a = _tiny_cfg(tmp_path / "a", total_frames=50)
        cfg_b = _tiny_cfg(tmp_path / "b", total_frames=50)
        curve_a = train(cfg_a, env=_FakeEnv(), partner=_FakePartner(), repo_root=tmp_path)
        curve_b = train(cfg_b, env=_FakeEnv(), partner=_FakePartner(), repo_root=tmp_path)
        assert curve_a.per_step_ego_rewards == curve_b.per_step_ego_rewards


class TestTrainerFactorySeam:
    def test_factory_is_called_with_cfg_and_env(self, tmp_path: Path) -> None:
        """T4b.11: the factory seam receives (cfg, env=, ego_uid=) per the Protocol."""
        captured: dict[str, Any] = {}

        def factory(cfg, *, env, ego_uid):  # type: ignore[no-untyped-def]
            captured["cfg"] = cfg
            captured["env"] = env
            captured["ego_uid"] = ego_uid
            return RandomEgoTrainer(ego_uid=ego_uid, action_dim=2, root_seed=cfg.seed)

        cfg = _tiny_cfg(tmp_path, total_frames=50)
        env = _FakeEnv()
        train(
            cfg,
            env=env,
            partner=_FakePartner(),
            trainer_factory=factory,
            repo_root=tmp_path,
        )
        assert captured["cfg"] is cfg
        assert captured["ego_uid"] == "ego"
        assert captured["env"] is env

    def test_factory_trainer_methods_get_called(self, tmp_path: Path) -> None:
        """T4b.11: act() / observe() / update() / state_dict() all fire on the loop."""
        from concerto.training import ego_aht

        with (
            patch.object(ego_aht.RandomEgoTrainer, "act", autospec=True) as mock_act,
            patch.object(ego_aht.RandomEgoTrainer, "observe", autospec=True) as mock_obs,
            patch.object(ego_aht.RandomEgoTrainer, "update", autospec=True) as mock_upd,
            patch.object(ego_aht.RandomEgoTrainer, "state_dict", autospec=True) as mock_sd,
        ):
            mock_act.return_value = np.zeros(2, dtype=np.float32)
            mock_sd.return_value = {}
            cfg = _tiny_cfg(tmp_path, total_frames=50)
            train(cfg, env=_FakeEnv(), partner=_FakePartner(), repo_root=tmp_path)

        # 50 frames → 50 act() + 50 observe() calls.
        assert mock_act.call_count == 50
        assert mock_obs.call_count == 50
        # rollout_length=25 → 2 update() calls at frame 25 and 50.
        assert mock_upd.call_count == 2
        # checkpoint_every=50 → 1 state_dict call.
        assert mock_sd.call_count == 1


class TestPublicSurface:
    def test_module_exports(self) -> None:
        from concerto.training import ego_aht

        for name in (
            "EgoTrainer",
            "EnvLike",
            "PartnerLike",
            "RandomEgoTrainer",
            "RewardCurve",
            "TrainerFactory",
            "train",
        ):
            assert hasattr(ego_aht, name)
