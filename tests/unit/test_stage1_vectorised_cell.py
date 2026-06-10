# SPDX-License-Identifier: Apache-2.0
"""Tier-1 tests for the vectorised Stage-1b training cell (P1.05.10).

ADR-007 §Stage 1b regime-alignment revision: ``num_envs > 1`` cells run
ManiSkill GPU-parallel with the chamber-side auto-reset wrapper, batched
obs synthesis, a batched partner heuristic, a batched trainer
(act / observe / per-env GAE), and the vectorised rollout loop in
:func:`concerto.training.ego_aht.train`. These tests pin the pure-Python
contracts against fakes (no SAPIEN / Vulkan); the Tier-2 counterpart at
``tests/integration/test_stage1_vectorised_real.py`` exercises the real
GPU env.

Also pinned: the safety-stack conflict loud-fail (ADR-004 §Decision
snapshot contract is single-env; vectorised cells require the
``safety.enabled=false`` operator override) and the ADR-002 single-env
byte-identity guard (num_envs=1 never routes through the vector paths).
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import pytest

from chamber.benchmarks.ego_ppo_trainer import EgoPPOTrainer
from chamber.envs.stage1_vector import Stage1AutoResetWrapper
from chamber.partners.api import PartnerSpec
from chamber.partners.heuristic import ScriptedHeuristicPartner
from concerto.training.config import (
    EgoAHTConfig,
    EnvConfig,
    HAPPOHyperparams,
    PartnerConfig,
    RuntimeConfig,
    SafetyConfig,
)
from concerto.training.ego_aht import train

_N_ENVS = 4
_OBS_DIM = 6
_ACT_DIM = 3
_PARTNER_ACT_DIM = 2


def _batched_obs(n: int, *, base: float = 0.0) -> dict[str, Any]:
    """Build a fake vectorised obs dict in the chamber state layout."""
    state = np.tile(np.arange(_OBS_DIM, dtype=np.float32), (n, 1)) + np.float32(base)
    return {
        "agent": {
            "ego": {"state": state.copy()},
            "partner": {"state": state.copy()[:, :4]},
        }
    }


class _FakeVectorEnv:
    """Fake auto-reset vector env satisfying the loop's EnvLike surface.

    Truncates every ``horizon`` steps on every env simultaneously
    (simplest deterministic done pattern) and surfaces the pre-reset
    obs via ``info["final_observation"]`` — the
    :class:`chamber.envs.stage1_vector.Stage1AutoResetWrapper`
    contract.
    """

    def __init__(self, *, n: int = _N_ENVS, horizon: int = 5) -> None:
        self._n = n
        self._horizon = horizon
        self._t = 0
        self.n_steps = 0

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        del seed, options
        self._t = 0
        return _batched_obs(self._n), {}

    def step(self, action: dict[str, Any]) -> tuple[dict[str, Any], Any, Any, Any, dict[str, Any]]:
        assert action["ego"].shape == (self._n, _ACT_DIM)
        assert action["partner"].shape == (self._n, _PARTNER_ACT_DIM)
        self._t += 1
        self.n_steps += 1
        reward = np.full(self._n, 0.5, dtype=np.float32)
        truncated = np.full(self._n, self._t >= self._horizon, dtype=np.bool_)
        terminated = np.zeros(self._n, dtype=np.bool_)
        info: dict[str, Any] = {}
        obs = _batched_obs(self._n, base=float(self._t))
        if truncated.any():
            info["final_observation"] = obs
            self._t = 0
            obs = _batched_obs(self._n)
        return obs, reward, terminated, truncated, info


class _RecordingTrainer:
    """Fake EgoTrainer that records the vectorised observe/update cadence."""

    def __init__(self) -> None:
        self.observe_calls: list[dict[str, Any]] = []
        self.n_updates = 0

    def act(self, obs: Any, *, deterministic: bool = False) -> np.ndarray:  # type: ignore[type-arg]
        del deterministic
        n = np.asarray(obs["agent"]["ego"]["state"]).shape[0]
        return np.zeros((n, _ACT_DIM), dtype=np.float32)

    def observe(
        self,
        obs: Any,
        reward: Any,
        done: Any,
        *,
        truncated: Any = False,
        final_obs: Any = None,
    ) -> None:
        del obs
        self.observe_calls.append(
            {"reward": reward, "done": done, "truncated": truncated, "final_obs": final_obs}
        )

    def update(self) -> None:
        self.n_updates += 1

    def state_dict(self) -> dict[str, Any]:
        return {}


def _vector_cfg(
    tmp_path: Any,
    *,
    num_envs: int = _N_ENVS,
    total_frames: int = 64,
    rollout_length: int = 4,
    checkpoint_every: int = 32,
    safety_enabled: bool = False,
) -> EgoAHTConfig:
    return EgoAHTConfig(
        seed=0,
        total_frames=total_frames,
        checkpoint_every=checkpoint_every,
        artifacts_root=tmp_path / "artifacts",
        log_dir=tmp_path / "logs",
        env=EnvConfig(
            task="mpe_cooperative_push",
            episode_length=5,
            num_envs=num_envs,
        ),
        partner=PartnerConfig(
            class_name="scripted_heuristic",
            extra={"uid": "partner", "target_xy": "0.0,0.0", "action_dim": "2"},
        ),
        happo=HAPPOHyperparams(
            rollout_length=rollout_length,
            batch_size=rollout_length,
            n_epochs=1,
            hidden_dim=16,
        ),
        runtime=RuntimeConfig(device="cpu"),
        safety=SafetyConfig(enabled=safety_enabled),
    )


def _factory_of(trainer: Any) -> Any:
    """Wrap a pre-built fake trainer in the TrainerFactory call shape."""

    def _factory(
        cfg: Any,
        *,
        env: Any,
        partner: Any,
        ego_uid: str,
        logger: Any = None,
    ) -> Any:
        del cfg, env, partner, ego_uid, logger
        return trainer

    return _factory


def _build_partner(action_dim: int = _PARTNER_ACT_DIM) -> ScriptedHeuristicPartner:
    return ScriptedHeuristicPartner(
        spec=PartnerSpec(
            class_name="scripted_heuristic",
            seed=0,
            checkpoint_step=None,
            weights_uri=None,
            extra={"uid": "partner", "target_xy": "0.0,0.0", "action_dim": str(action_dim)},
        )
    )


class TestEnvConfigNumEnvs:
    def test_default_is_one(self) -> None:
        """ADR-002: the historical single-env cell is the default."""
        assert EnvConfig(task="mpe_cooperative_push").num_envs == 1

    def test_zero_rejected(self) -> None:
        """num_envs >= 1 is a validation invariant (P1.05.10)."""
        with pytest.raises(ValueError, match="num_envs"):
            EnvConfig(task="mpe_cooperative_push", num_envs=0)


class TestVectorisedTrainLoop:
    def test_safety_conflict_loud_fails(self, tmp_path: Any) -> None:
        """num_envs > 1 with safety.enabled=True raises (ADR-007 regime-alignment).

        The training-time safety stack is single-env in this slice;
        silently dropping it would hide the posture change from the
        audit trail. The operator override safety.enabled=false is the
        documented vectorised-cell posture.
        """
        cfg = _vector_cfg(tmp_path, safety_enabled=True)
        with pytest.raises(ValueError, match=r"num_envs.*safety|safety.*num_envs"):
            train(
                cfg,
                env=_FakeVectorEnv(),  # type: ignore[arg-type]
                partner=_build_partner(),
                trainer_factory=_factory_of(_RecordingTrainer()),
            )

    def test_cadence_episodes_and_checkpoints(self, tmp_path: Any) -> None:
        """Vector loop: frames / updates / episodes / checkpoints all batch-aware.

        total_frames=64 at num_envs=4 → 16 iterations; rollout_length=4
        → 4 updates; horizon=5 → 3 full episode boundaries per env
        (iterations 5, 10, 15) → 12 completed episodes + 4 tail
        accumulators; checkpoint_every=32 frames → buckets at 32 and 64
        → 2 checkpoints.
        """
        cfg = _vector_cfg(tmp_path)
        fake_env = _FakeVectorEnv()
        trainer = _RecordingTrainer()
        result = train(
            cfg,
            env=fake_env,  # type: ignore[arg-type]
            partner=_build_partner(),
            trainer_factory=_factory_of(trainer),
        )
        assert fake_env.n_steps == 16
        assert trainer.n_updates == 4
        assert len(trainer.observe_calls) == 16
        # Every observe carried per-env vectors.
        first = trainer.observe_calls[0]
        assert np.asarray(first["reward"]).shape == (_N_ENVS,)
        assert np.asarray(first["done"]).shape == (_N_ENVS,)
        # Truncation steps carried the pre-reset final obs.
        trunc_calls = [c for c in trainer.observe_calls if np.asarray(c["truncated"]).any()]
        assert len(trunc_calls) == 3
        assert all(c["final_obs"] is not None for c in trunc_calls)
        # 12 completed episodes + 4 tail accumulators; each completed
        # episode is 5 steps x 0.5 reward.
        assert len(result.curve.per_episode_ego_rewards) == 12 + _N_ENVS
        np.testing.assert_allclose(result.curve.per_episode_ego_rewards[:12], 2.5)
        # Per-step curve records the batch-mean per iteration.
        assert len(result.curve.per_step_ego_rewards) == 16
        assert len(result.curve.checkpoint_paths) == 2

    def test_single_env_path_does_not_enter_vector_loop(self, tmp_path: Any) -> None:
        """num_envs=1 keeps the historical scalar loop (ADR-002 byte-identity guard).

        The scalar loop calls env.reset at every episode boundary; the
        vector loop never does. A fake that counts resets distinguishes
        the two paths structurally.
        """

        class _CountingScalarEnv:
            def __init__(self) -> None:
                self.n_resets = 0

            def reset(
                self, *, seed: int | None = None, options: dict[str, Any] | None = None
            ) -> tuple[dict[str, Any], dict[str, Any]]:
                del seed, options
                self.n_resets += 1
                return _batched_obs(1), {}

            def step(
                self, action: dict[str, Any]
            ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
                del action
                return _batched_obs(1), 0.0, False, True, {}

        class _ScalarTrainer(_RecordingTrainer):
            def act(self, obs: Any, *, deterministic: bool = False) -> np.ndarray:  # type: ignore[type-arg]
                del obs, deterministic
                return np.zeros(_ACT_DIM, dtype=np.float32)

        cfg = _vector_cfg(tmp_path, num_envs=1, total_frames=8, rollout_length=8)
        env = _CountingScalarEnv()
        train(
            cfg,
            env=env,  # type: ignore[arg-type]
            partner=_build_partner(),
            trainer_factory=_factory_of(_ScalarTrainer()),
        )
        # Initial reset + one per truncation boundary (every step) —
        # the vector loop would have left this at exactly 1.
        assert env.n_resets == 9


class TestAutoResetWrapper:
    def test_partial_reset_and_final_observation(self) -> None:
        """Done envs are partially reset; the pre-reset obs is surfaced (Pardo 2017).

        ADR-007 §Stage 1b regime-alignment: the wrapper owns the
        autoreset contract so the training loop never resets mid-run.
        """

        class _FakeBatchedInner(gym.Env):  # type: ignore[type-arg]
            observation_space = gym.spaces.Dict(
                {"x": gym.spaces.Box(-np.inf, np.inf, (_N_ENVS, 2), np.float32)}
            )
            action_space = gym.spaces.Box(-1, 1, (_N_ENVS, 2), np.float32)

            def __init__(self) -> None:
                self.reset_idx_calls: list[np.ndarray] = []  # type: ignore[type-arg]
                self._step = 0

            def reset(
                self, *, seed: int | None = None, options: dict[str, Any] | None = None
            ) -> tuple[dict[str, Any], dict[str, Any]]:
                del seed
                if options is not None and "env_idx" in options:
                    self.reset_idx_calls.append(np.asarray(options["env_idx"]))
                return {"x": np.full((_N_ENVS, 2), -1.0, dtype=np.float32)}, {}

            def step(self, action: Any) -> tuple[dict[str, Any], Any, Any, Any, dict[str, Any]]:
                del action
                self._step += 1
                obs = {"x": np.full((_N_ENVS, 2), float(self._step), dtype=np.float32)}
                terminated = np.zeros(_N_ENVS, dtype=np.bool_)
                truncated = np.zeros(_N_ENVS, dtype=np.bool_)
                truncated[1] = True
                truncated[3] = True
                return obs, np.zeros(_N_ENVS, dtype=np.float32), terminated, truncated, {}

        inner = _FakeBatchedInner()
        wrap = Stage1AutoResetWrapper(inner)
        obs, reward, terminated, truncated, info = wrap.step(np.zeros((_N_ENVS, 2)))
        del reward, terminated
        # Pre-reset obs surfaced; post-reset obs returned.
        assert "final_observation" in info
        np.testing.assert_allclose(info["final_observation"]["x"], 1.0)
        np.testing.assert_allclose(obs["x"], -1.0)
        # Exactly the done idx were reset.
        assert len(inner.reset_idx_calls) == 1
        np.testing.assert_array_equal(inner.reset_idx_calls[0], [1, 3])
        # Flags pass through pre-reset.
        np.testing.assert_array_equal(np.asarray(truncated), [False, True, False, True])

    def test_no_done_no_reset(self) -> None:
        """Steps without boundaries pass through untouched (no spurious resets)."""

        class _QuietInner(gym.Env):  # type: ignore[type-arg]
            observation_space = gym.spaces.Dict(
                {"x": gym.spaces.Box(-np.inf, np.inf, (2, 2), np.float32)}
            )
            action_space = gym.spaces.Box(-1, 1, (2, 2), np.float32)
            n_resets = 0

            def reset(self, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
                del kwargs
                self.n_resets += 1
                return {"x": np.zeros((2, 2), dtype=np.float32)}, {}

            def step(self, action: Any) -> tuple[dict[str, Any], Any, Any, Any, dict[str, Any]]:
                del action
                return (
                    {"x": np.ones((2, 2), dtype=np.float32)},
                    np.zeros(2, dtype=np.float32),
                    np.zeros(2, dtype=np.bool_),
                    np.zeros(2, dtype=np.bool_),
                    {},
                )

        inner = _QuietInner()
        wrap = Stage1AutoResetWrapper(inner)
        _, _, _, _, info = wrap.step(np.zeros((2, 2)))
        assert "final_observation" not in info
        assert inner.n_resets == 0


class TestBatchedPartnerHeuristic:
    def test_batched_state_yields_batched_actions(self) -> None:
        """(num_envs, dim) state → (num_envs, action_dim) planar-reach actions."""
        partner = _build_partner(action_dim=13)
        state = np.zeros((3, 6), dtype=np.float32)
        state[0, :2] = [0.4, -0.2]
        state[1, :2] = [2.0, 2.0]  # beyond clip
        state[2, :2] = [0.0, 0.0]
        obs = {"agent": {"partner": {"state": state}}}
        action = partner.act(obs)
        assert action.shape == (3, 13)
        np.testing.assert_allclose(action[0, :2], [-0.4, 0.2], atol=1e-6)
        np.testing.assert_allclose(action[1, :2], [-1.0, -1.0])
        np.testing.assert_allclose(action[2, :2], [0.0, 0.0])
        np.testing.assert_allclose(action[:, 2:], 0.0)

    def test_single_env_path_unchanged(self) -> None:
        """1-D state keeps the historical scalar action shape (ADR-002 guard)."""
        partner = _build_partner(action_dim=2)
        obs = {"agent": {"partner": {"state": np.asarray([0.25, -0.5, 0.0], dtype=np.float32)}}}
        action = partner.act(obs)
        assert action.shape == (2,)
        np.testing.assert_allclose(action, [-0.25, 0.5])


class _FakeBoxEnv:
    """Minimal env exposing batched spaces for EgoPPOTrainer.from_config."""

    def __init__(self, n: int) -> None:
        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Dict(
                    {
                        "ego": gym.spaces.Dict(
                            {"state": gym.spaces.Box(-np.inf, np.inf, (n, _OBS_DIM), np.float32)}
                        )
                    }
                )
            }
        )
        self.action_space = gym.spaces.Dict(
            {"ego": gym.spaces.Box(-1.0, 1.0, (n, _ACT_DIM), np.float32)}
        )


class TestVectorisedEgoPPOTrainer:
    def _build(self, tmp_path: Any) -> EgoPPOTrainer:
        cfg = _vector_cfg(tmp_path, rollout_length=6, total_frames=24)
        return EgoPPOTrainer.from_config(
            cfg,
            env=_FakeBoxEnv(_N_ENVS),  # type: ignore[arg-type]
            partner=_build_partner(),
            ego_uid="ego",
        )

    def test_from_config_strips_batched_spaces(self, tmp_path: Any) -> None:
        """Batched (num_envs, dim) spaces size the nets per env (P1.05.10)."""
        trainer = self._build(tmp_path)
        assert trainer._obs_dim == _OBS_DIM
        assert trainer._act_dim == _ACT_DIM

    def test_act_returns_batched_actions(self, tmp_path: Any) -> None:
        """act on (num_envs, obs_dim) state returns (num_envs, act_dim)."""
        trainer = self._build(tmp_path)
        action = trainer.act(_batched_obs(_N_ENVS))
        assert np.asarray(action).shape == (_N_ENVS, _ACT_DIM)
        # Deterministic mode also batched.
        action_det = trainer.act(_batched_obs(_N_ENVS), deterministic=True)
        assert np.asarray(action_det).shape == (_N_ENVS, _ACT_DIM)

    def test_observe_requires_final_obs_at_truncation(self, tmp_path: Any) -> None:
        """A vector truncation without final_obs loud-fails (issue #62; Pardo 2017)."""
        trainer = self._build(tmp_path)
        trainer.act(_batched_obs(_N_ENVS))
        truncated = np.zeros(_N_ENVS, dtype=np.bool_)
        truncated[2] = True
        with pytest.raises(ValueError, match="final_obs"):
            trainer.observe(
                _batched_obs(_N_ENVS),
                np.zeros(_N_ENVS, dtype=np.float32),
                truncated.copy(),
                truncated=truncated,
            )

    def test_full_rollout_update_cycle(self, tmp_path: Any) -> None:
        """act/observe/update over a batched rollout with truncation boundaries.

        Exercises the per-env GAE path end-to-end: T=6 steps x N=4 envs
        with a mid-rollout truncation on all envs at t=3 (final_obs
        supplied). update() must consume the (T, N) buffers and clear
        them.
        """
        trainer = self._build(tmp_path)
        horizon = 3
        for t in range(6):
            trainer.act(_batched_obs(_N_ENVS, base=float(t)))
            truncated = np.full(_N_ENVS, (t + 1) % horizon == 0, dtype=np.bool_)
            final_obs = _batched_obs(_N_ENVS, base=float(t) + 0.5) if truncated.any() else None
            trainer.observe(
                _batched_obs(_N_ENVS, base=float(t + 1)),
                np.full(_N_ENVS, 0.1, dtype=np.float32),
                truncated.copy(),
                truncated=truncated,
                final_obs=final_obs,
            )
        assert len(trainer._buf_rewards) == 6
        assert np.asarray(trainer._buf_rewards[0]).shape == (_N_ENVS,)
        trainer.update()
        assert not trainer._buf_rewards
        # A second cycle must also work (buffer state fully reset).
        trainer.act(_batched_obs(_N_ENVS))
        trainer.observe(
            _batched_obs(_N_ENVS),
            np.zeros(_N_ENVS, dtype=np.float32),
            np.zeros(_N_ENVS, dtype=np.bool_),
            truncated=np.zeros(_N_ENVS, dtype=np.bool_),
        )
        trainer.update()
        assert not trainer._buf_rewards
