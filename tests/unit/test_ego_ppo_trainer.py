# SPDX-License-Identifier: Apache-2.0
# pyright: reportPrivateImportUsage=false
#
# Suppresses pyright's complaints about ``torch.allclose`` / ``torch.equal``
# (public API, missing from torch's stub ``__all__``). Same rationale as
# ``src/chamber/benchmarks/ego_ppo_trainer.py``.
"""Unit tests for ``chamber.benchmarks.ego_ppo_trainer`` (M4b-8a).

Covers:

- ADR-002 §Decisions: the ego-AHT trainer that drives the empirical-
  guarantee experiment (T4b.13 / M4b-8b).
- plan/05 §3.5: rollout / GAE / advantage-normalization / PPO update
  on the CONCERTO side, not in the HARL fork.
- :class:`~concerto.training.ego_aht.EgoTrainer` Protocol conformance:
  ``act`` / ``observe`` / ``update`` / ``state_dict`` shapes + types.

The property-test counterpart at ``tests/property/test_advantage_decomposition.py``
covers the GAE math itself; this module covers the trainer's wiring.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import pytest
import torch

# HARL is scoped to the ``[dependency-groups].train`` PEP 735 group
# (ADR-002 §Revision-history 2026-05-16; #131). Every test in this
# module constructs :class:`EgoPPOTrainer.from_config`, which lazily
# imports ``harl.algorithms.actors.happo.HAPPO`` inside ``__init__``.
# Skip the whole module if the train group is not installed so
# ``uv sync --group dev``-only environments do not see spurious
# failures pointing at the trainer's helpful error.
pytest.importorskip(
    "harl.algorithms.actors.happo",
    reason="HARL not installed; run `uv sync --group train` to enable HARL tests.",
)

from chamber.benchmarks.ego_ppo_trainer import (
    EgoPPOTrainer,
    _build_harl_args,
    _flat_ego_obs,
)
from chamber.envs.mpe_cooperative_push import MPECooperativePushEnv
from chamber.partners.api import PartnerSpec
from chamber.partners.heuristic import ScriptedHeuristicPartner
from concerto.training.config import (
    EgoAHTConfig,
    EnvConfig,
    HAPPOHyperparams,
    PartnerConfig,
    RuntimeConfig,
)


def _tiny_cfg(*, seed: int = 0, rollout_length: int = 16) -> EgoAHTConfig:
    """Construct a Phase-0 config small enough for fast-CI unit tests."""
    return EgoAHTConfig(
        seed=seed,
        total_frames=rollout_length,
        checkpoint_every=rollout_length,
        env=EnvConfig(task="mpe_cooperative_push", episode_length=rollout_length),
        partner=PartnerConfig(
            class_name="scripted_heuristic",
            extra={"uid": "partner", "target_xy": "0.0,0.0", "action_dim": "2"},
        ),
        happo=HAPPOHyperparams(
            rollout_length=rollout_length,
            batch_size=rollout_length,
            n_epochs=1,
            hidden_dim=32,
        ),
        # Pin device=cpu so the unit tests don't auto-promote to MPS on
        # the Mac dev box; MPS has known numerical quirks under PPO that
        # the test fixtures aren't tuned for. M4b-9b.
        runtime=RuntimeConfig(device="cpu"),
    )


def _build_trainer(*, seed: int = 0, rollout_length: int = 16) -> EgoPPOTrainer:
    cfg = _tiny_cfg(seed=seed, rollout_length=rollout_length)
    env = MPECooperativePushEnv(root_seed=seed)
    return EgoPPOTrainer.from_config(cfg, env=env, partner=_build_partner(), ego_uid="ego")


def _build_partner() -> ScriptedHeuristicPartner:
    return ScriptedHeuristicPartner(
        spec=PartnerSpec(
            class_name="scripted_heuristic",
            seed=0,
            checkpoint_step=None,
            weights_uri=None,
            extra={"uid": "partner", "target_xy": "0.0,0.0", "action_dim": "2"},
        )
    )


def _drive_one_rollout(
    trainer: EgoPPOTrainer,
    env: MPECooperativePushEnv,
    partner: ScriptedHeuristicPartner,
    *,
    n_steps: int,
) -> None:
    """Step the env n times, feeding the trainer's act/observe contract."""
    obs, _ = env.reset(seed=0)
    partner.reset(seed=0)
    for _ in range(n_steps):
        ego_a = trainer.act(obs)
        partner_a = partner.act(obs)
        obs, reward, terminated, truncated, _ = env.step({"ego": ego_a, "partner": partner_a})
        trainer.observe(obs, reward, terminated or truncated, truncated=truncated)


class TestBuildHarlArgs:
    def test_returns_18_keys_with_required_fields(self) -> None:
        """ADR-002 §Decisions: HARL constructor needs every key present."""
        args = _build_harl_args(happo=HAPPOHyperparams())
        # Spot-check the keys HARL's HAPPO + StochasticPolicy + MLPBase read.
        for key in (
            "lr",
            "opti_eps",
            "weight_decay",
            "clip_param",
            "ppo_epoch",
            "actor_num_mini_batch",
            "entropy_coef",
            "use_max_grad_norm",
            "max_grad_norm",
            "data_chunk_length",
            "use_recurrent_policy",
            "use_naive_recurrent_policy",
            "use_policy_active_masks",
            "action_aggregation",
            "hidden_sizes",
            "gain",
            "initialization_method",
            "recurrent_n",
            "use_feature_normalization",
            "activation_func",
            "std_x_coef",
            "std_y_coef",
        ):
            assert key in args, f"missing required HARL args key: {key!r}"

    def test_lr_clip_eps_and_n_epochs_pass_through(self) -> None:
        """plan/05 §3.5: project hyperparams override HARL defaults."""
        happo = HAPPOHyperparams(lr=1e-5, clip_eps=0.05, n_epochs=7)
        args = _build_harl_args(happo=happo)
        assert args["lr"] == pytest.approx(1e-5)
        assert args["clip_param"] == pytest.approx(0.05)
        assert args["ppo_epoch"] == 7

    def test_actor_num_mini_batch_clamped_to_one_when_batch_exceeds_rollout(self) -> None:
        """plan/05 §3.5: batch_size > rollout_length collapses to single-batch SGD."""
        happo = HAPPOHyperparams(rollout_length=100, batch_size=1024)
        args = _build_harl_args(happo=happo)
        assert args["actor_num_mini_batch"] == 1

    def test_actor_num_mini_batch_uses_floor_division(self) -> None:
        """plan/05 §3.5: rollout_length // batch_size, never silently rounds up."""
        happo = HAPPOHyperparams(rollout_length=1000, batch_size=256)
        args = _build_harl_args(happo=happo)
        assert args["actor_num_mini_batch"] == 1000 // 256

    def test_hidden_sizes_inflated_to_two_layers(self) -> None:
        """ADR-002 §Decisions: two-hidden-layer MLP matches the published Bi-DexHands HAPPO."""
        happo = HAPPOHyperparams(hidden_dim=128)
        args = _build_harl_args(happo=happo)
        assert args["hidden_sizes"] == [128, 128]


class TestFlatEgoObs:
    def test_extracts_ego_state_only(self) -> None:
        """plan/04 §3.4: ``obs['agent'][uid]['state']`` is the canonical state vector."""
        obs = {
            "agent": {
                "ego": {"state": np.array([1.0, 2.0, 3.0], dtype=np.float32)},
                "partner": {"state": np.array([9.0, 9.0, 9.0], dtype=np.float32)},
            }
        }
        flat = _flat_ego_obs(obs, "ego")
        np.testing.assert_array_equal(flat, np.array([1.0, 2.0, 3.0], dtype=np.float32))

    def test_returns_float32_dtype(self) -> None:
        """ADR-002 §Decisions: HARL networks expect float32 inputs."""
        obs = {"agent": {"ego": {"state": [1, 2, 3]}, "partner": {"state": [0, 0, 0]}}}
        flat = _flat_ego_obs(obs, "ego")
        assert flat.dtype == np.float32


class TestFromConfig:
    def test_constructs_against_real_env(self) -> None:
        """ADR-002 §Decisions: the factory wires obs+act spaces from the env."""
        cfg = _tiny_cfg()
        env = MPECooperativePushEnv()
        trainer = EgoPPOTrainer.from_config(cfg, env=env, partner=_build_partner(), ego_uid="ego")
        assert isinstance(trainer, EgoPPOTrainer)

    def test_rejects_non_box_obs_state(self) -> None:
        """Loud-fail when the env's ego state isn't a Box (defensive against schema drift)."""

        class _BadStateEnv:
            observation_space: Any = gym.spaces.Dict(
                {
                    "agent": gym.spaces.Dict(
                        {
                            "ego": gym.spaces.Dict(
                                {"state": gym.spaces.Discrete(2)}  # not Box
                            ),
                            "partner": gym.spaces.Dict({"state": gym.spaces.Discrete(2)}),
                        }
                    )
                }
            )
            action_space: Any = gym.spaces.Dict(
                {
                    "ego": gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
                    "partner": gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
                }
            )

            def reset(self, **_: Any) -> tuple[dict[str, Any], dict[str, Any]]:
                return {}, {}

            def step(
                self, _action: dict[str, Any]
            ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
                return {}, 0.0, False, False, {}

        cfg = _tiny_cfg()
        with pytest.raises(TypeError, match=r"gym\.spaces\.Box"):
            EgoPPOTrainer.from_config(
                cfg,
                env=_BadStateEnv(),  # type: ignore[arg-type]
                partner=_build_partner(),
                ego_uid="ego",
            )

    def test_rejects_non_box_action_space(self) -> None:
        """Loud-fail when the env's ego action isn't a Box (defensive)."""

        class _BadActionEnv:
            observation_space: Any = gym.spaces.Dict(
                {
                    "agent": gym.spaces.Dict(
                        {
                            "ego": gym.spaces.Dict(
                                {
                                    "state": gym.spaces.Box(
                                        low=-1, high=1, shape=(10,), dtype=np.float32
                                    )
                                }
                            ),
                            "partner": gym.spaces.Dict(
                                {
                                    "state": gym.spaces.Box(
                                        low=-1, high=1, shape=(10,), dtype=np.float32
                                    )
                                }
                            ),
                        }
                    )
                }
            )
            action_space: Any = gym.spaces.Dict(
                {"ego": gym.spaces.Discrete(4), "partner": gym.spaces.Discrete(4)}
            )

            def reset(self, **_: Any) -> tuple[dict[str, Any], dict[str, Any]]:
                return {}, {}

            def step(
                self, _action: dict[str, Any]
            ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
                return {}, 0.0, False, False, {}

        cfg = _tiny_cfg()
        with pytest.raises(TypeError, match=r"gym\.spaces\.Box"):
            EgoPPOTrainer.from_config(
                cfg,
                env=_BadActionEnv(),  # type: ignore[arg-type]
                partner=_build_partner(),
                ego_uid="ego",
            )


class TestActSurface:
    def test_returns_correct_shape_and_dtype(self) -> None:
        """ADR-002 §Decisions: act() emits shape (act_dim,) float32."""
        trainer = _build_trainer()
        env = MPECooperativePushEnv(root_seed=0)
        obs, _ = env.reset(seed=0)
        action = trainer.act(obs)
        assert action.shape == (2,)
        assert action.dtype == np.float32

    def test_caches_pre_step_state(self) -> None:
        """plan/05 §3.5: act() populates the pre-step cache for observe()."""
        trainer = _build_trainer()
        env = MPECooperativePushEnv(root_seed=0)
        obs, _ = env.reset(seed=0)
        assert trainer._pending is None
        trainer.act(obs)
        # All four sub-fields populated together (frozen dataclass).
        assert trainer._pending is not None
        assert trainer._pending.action is not None
        assert trainer._pending.log_prob is not None
        assert trainer._pending.value is not None

    def test_deterministic_flag_returns_distribution_mode(self) -> None:
        """ADR-002 §Decisions: deterministic=True is reproducible across calls.

        For a diag-Gaussian policy the mode equals the mean, so two
        ``act(obs, deterministic=True)`` calls on the same obs at the
        same network state must return identical actions.
        """
        trainer = _build_trainer()
        env = MPECooperativePushEnv(root_seed=0)
        obs, _ = env.reset(seed=0)
        a1 = trainer.act(obs, deterministic=True)
        a2 = trainer.act(obs, deterministic=True)
        np.testing.assert_array_equal(a1, a2)


class TestObserveSurface:
    def test_observe_without_prior_act_is_noop(self) -> None:
        """Defensive: stray observe() before any act() does not crash or insert."""
        trainer = _build_trainer()
        env = MPECooperativePushEnv(root_seed=0)
        obs, _ = env.reset(seed=0)
        trainer.observe(obs, 0.0, False)
        assert len(trainer._buf_rewards) == 0

    def test_observe_inserts_one_tuple_per_act(self) -> None:
        """plan/05 §3.5: act/observe pairing — one rollout step per call pair."""
        trainer = _build_trainer()
        env = MPECooperativePushEnv(root_seed=0)
        partner = _build_partner()
        _drive_one_rollout(trainer, env, partner, n_steps=8)
        assert len(trainer._buf_rewards) == 8
        assert len(trainer._buf_obs) == 8
        assert len(trainer._buf_actions) == 8
        assert len(trainer._buf_log_probs) == 8
        assert len(trainer._buf_values) == 8
        assert len(trainer._buf_terminated) == 8
        assert len(trainer._buf_truncated) == 8
        assert len(trainer._buf_truncation_bootstraps) == 8

    def test_observe_clears_pending_cache(self) -> None:
        """plan/05 §3.5: observe() consumes the pre-step cache exactly once."""
        trainer = _build_trainer()
        env = MPECooperativePushEnv(root_seed=0)
        obs, _ = env.reset(seed=0)
        trainer.act(obs)
        assert trainer._pending is not None
        next_obs, reward, term, trunc, _ = env.step(
            {"ego": np.zeros(2, dtype=np.float32), "partner": np.zeros(2, dtype=np.float32)}
        )
        trainer.observe(next_obs, reward, term or trunc, truncated=trunc)
        assert trainer._pending is None

    def test_observe_records_truncation_bootstrap_at_time_limit(self) -> None:
        """Pardo 2017 / issue #62: at truncation, V(s_truncated_final) is cached for GAE."""
        trainer = _build_trainer(rollout_length=2)
        env = MPECooperativePushEnv(episode_length=2, root_seed=0)
        partner = _build_partner()
        obs, _ = env.reset(seed=0)
        partner.reset(seed=0)
        # Step 1: not a boundary.
        trainer.act(obs)
        next_obs, r, term, trunc, _ = env.step(
            {"ego": np.zeros(2, dtype=np.float32), "partner": partner.act(obs)}
        )
        trainer.observe(next_obs, r, term or trunc, truncated=trunc)
        obs = next_obs
        assert trainer._buf_truncated[-1] is False
        assert trainer._buf_truncation_bootstraps[-1] == 0.0
        # Step 2: truncation boundary (episode_length=2).
        trainer.act(obs)
        next_obs, r, term, trunc, _ = env.step(
            {"ego": np.zeros(2, dtype=np.float32), "partner": partner.act(obs)}
        )
        assert trunc is True
        assert term is False
        trainer.observe(next_obs, r, term or trunc, truncated=trunc)
        assert trainer._buf_truncated[-1] is True
        assert trainer._buf_terminated[-1] is False
        # The truncation bootstrap should be a real critic value, not 0.0
        # (the critic has bias terms so V(.) is generically nonzero).
        # Defensive: pin only that it's a finite float, not a specific value.
        bootstrap = trainer._buf_truncation_bootstraps[-1]
        assert isinstance(bootstrap, float)
        assert np.isfinite(bootstrap)


class TestUpdateSurface:
    def test_update_without_buffer_is_noop(self) -> None:
        """ADR-002 §Decisions: empty-buffer update() is a safe no-op."""
        trainer = _build_trainer()
        # Should not raise.
        trainer.update()

    def test_update_clears_buffer(self) -> None:
        """plan/05 §3.5: update() consumes the rollout — buffer empty afterward."""
        trainer = _build_trainer(rollout_length=8)
        env = MPECooperativePushEnv(root_seed=0)
        partner = _build_partner()
        _drive_one_rollout(trainer, env, partner, n_steps=8)
        assert len(trainer._buf_rewards) == 8
        trainer.update()
        assert len(trainer._buf_rewards) == 0

    def test_update_changes_actor_parameters(self) -> None:
        """ADR-002 risk-mitigation #1: PPO update actually updates the actor.

        If the gradient pipeline is broken (e.g., HAPPO.update sample shape
        wrong, no backward pass), the actor parameters would be unchanged.
        """
        trainer = _build_trainer(rollout_length=8)
        env = MPECooperativePushEnv(root_seed=0)
        partner = _build_partner()
        before = [p.detach().clone() for p in trainer._happo.actor.parameters()]
        _drive_one_rollout(trainer, env, partner, n_steps=8)
        trainer.update()
        after = list(trainer._happo.actor.parameters())
        # At least one parameter tensor must have moved.
        assert any(not torch.allclose(b, a) for b, a in zip(before, after, strict=True)), (
            "actor parameters did not change after PPO update"
        )

    def test_update_changes_critic_parameters(self) -> None:
        """ADR-002 §Decisions: critic update via MSE on returns moves the critic."""
        trainer = _build_trainer(rollout_length=8)
        env = MPECooperativePushEnv(root_seed=0)
        partner = _build_partner()
        before = [p.detach().clone() for p in trainer._critic.parameters()]
        _drive_one_rollout(trainer, env, partner, n_steps=8)
        trainer.update()
        after = list(trainer._critic.parameters())
        assert any(not torch.allclose(b, a) for b, a in zip(before, after, strict=True)), (
            "critic parameters did not change after MSE update"
        )


class TestStateDict:
    def test_returns_actor_critic_optimizer_subkeys(self) -> None:
        """ADR-002 §Decisions / T4b.12: state_dict bundles the four trainable pieces."""
        trainer = _build_trainer()
        sd = trainer.state_dict()
        assert set(sd.keys()) == {"actor", "actor_optim", "critic", "critic_optim"}

    def test_round_trip_preserves_actor_weights(self) -> None:
        """T4b.12: the .pt artefact + load_state_dict cycle restores the actor."""
        trainer_a = _build_trainer(seed=7)
        sd = trainer_a.state_dict()
        # Build a fresh trainer (different RNG state due to torch.manual_seed
        # overwrite) and load the saved state.
        trainer_b = _build_trainer(seed=99)
        trainer_b._happo.actor.load_state_dict(sd["actor"])
        trainer_b._critic.load_state_dict(sd["critic"])
        # After the load, every actor + critic param tensor must be equal.
        for pa, pb in zip(
            trainer_a._happo.actor.parameters(),
            trainer_b._happo.actor.parameters(),
            strict=True,
        ):
            assert torch.equal(pa, pb)
        for pa, pb in zip(
            trainer_a._critic.parameters(), trainer_b._critic.parameters(), strict=True
        ):
            assert torch.equal(pa, pb)


class TestProtocolConformance:
    def test_exposes_required_methods(self) -> None:
        """plan/05 §3.5: structural typing — EgoPPOTrainer carries the EgoTrainer surface.

        :class:`EgoTrainer` is a Protocol but not ``runtime_checkable`` (its
        :meth:`state_dict` returns a non-Protocol-friendly type), so we
        verify conformance by attribute inspection instead.
        """
        trainer = _build_trainer()
        for name in ("act", "observe", "update", "state_dict"):
            assert callable(getattr(trainer, name)), f"missing required method {name}"


class TestDeterminism:
    """P6 / ADR-002 §Decisions: seeded construction reproduces network state.

    These tests deliberately avoid asserting on action streams across two
    trainers in the same process: torch's *global* RNG is the action-
    sampling source, and any intervening torch op (the second trainer's
    construction itself, for instance) shifts the shared RNG state. The
    contract we *do* enforce: identical seeds yield identical initial
    network weights; distinct seeds yield distinct initial weights.
    Action-stream determinism across separate processes is verified by
    ``test_run_training_seed_reproduces_reward_curve`` in
    ``test_training_runner.py``.
    """

    def test_same_seed_initializes_identical_actor_weights(self) -> None:
        """P6: two trainers at the same seed have bit-equal initial actor weights."""
        trainer_a = _build_trainer(seed=11)
        trainer_b = _build_trainer(seed=11)
        for pa, pb in zip(
            trainer_a._happo.actor.parameters(),
            trainer_b._happo.actor.parameters(),
            strict=True,
        ):
            assert torch.equal(pa, pb)

    def test_distinct_seeds_initialize_distinct_actor_weights(self) -> None:
        """P6: distinct seeds must produce distinct initial actor weights.

        A seed swap that left the weights unchanged would mean the
        determinism harness is leaking a constant seed regardless of
        ``cfg.seed``.
        """
        trainer_a = _build_trainer(seed=1)
        trainer_b = _build_trainer(seed=2)
        # Iterate parameters until we find one that differs — at least one
        # tensor must, otherwise the seeds aren't routing through.
        any_differ = False
        for pa, pb in zip(
            trainer_a._happo.actor.parameters(),
            trainer_b._happo.actor.parameters(),
            strict=True,
        ):
            if not torch.equal(pa, pb):
                any_differ = True
                break
        assert any_differ, "distinct seeds yielded identical actor weights"


class TestDeviceResolution:
    """ADR-002 §Decisions: RuntimeConfig.device resolves to a concrete torch.device."""

    def test_auto_routes_through_torch_device(self) -> None:
        """device='auto' defers to chamber.utils.device.torch_device()."""
        from chamber.benchmarks.ego_ppo_trainer import _resolve_device
        from chamber.utils.device import torch_device

        resolved = _resolve_device("auto")
        assert isinstance(resolved, torch.device)
        assert resolved.type == torch_device()

    def test_explicit_cpu(self) -> None:
        """device='cpu' resolves to torch.device('cpu') unconditionally."""
        from chamber.benchmarks.ego_ppo_trainer import _resolve_device

        assert _resolve_device("cpu").type == "cpu"

    def test_cuda_raises_with_adr_cite_when_unavailable(self) -> None:
        """device='cuda' loud-fails with the ADR-002 cite when CUDA isn't available."""
        from chamber.benchmarks.ego_ppo_trainer import _resolve_device

        if torch.cuda.is_available():
            pytest.skip("CUDA available; not-available path not reachable")
        with pytest.raises(RuntimeError, match=r"ADR-002"):
            _resolve_device("cuda")

    def test_mps_raises_with_adr_cite_when_unavailable(self) -> None:
        """device='mps' loud-fails with the ADR-002 cite when MPS isn't available."""
        from chamber.benchmarks.ego_ppo_trainer import _resolve_device

        if torch.backends.mps.is_available():
            pytest.skip("MPS available; not-available path not reachable")
        with pytest.raises(RuntimeError, match=r"ADR-002"):
            _resolve_device("mps")

    def test_from_config_calls_torch_determinism_on_cpu(self) -> None:
        """plan/08 §4: trainer construction applies the determinism flag on CPU."""
        from unittest.mock import patch

        from chamber.benchmarks.stage0_smoke import _SMOKE_ROBOT_UIDS  # noqa: F401
        from chamber.envs.mpe_cooperative_push import MPECooperativePushEnv

        cfg = _tiny_cfg()
        env = MPECooperativePushEnv(root_seed=0)
        # Force device='cpu' + deterministic_torch=True via model_copy.
        cfg_cpu = cfg.model_copy(
            update={"runtime": cfg.runtime.model_copy(update={"device": "cpu"})}
        )
        with patch("chamber.benchmarks.ego_ppo_trainer._set_torch_determinism") as mock_set:
            EgoPPOTrainer.from_config(cfg_cpu, env=env, partner=_build_partner(), ego_uid="ego")
            mock_set.assert_called_once_with(True)

    def test_from_config_skips_torch_determinism_when_flag_off(self) -> None:
        """plan/08 §4: deterministic_torch=False skips the strict-mode flag."""
        from unittest.mock import patch

        from chamber.envs.mpe_cooperative_push import MPECooperativePushEnv

        cfg = _tiny_cfg()
        env = MPECooperativePushEnv(root_seed=0)
        cfg_no_det = cfg.model_copy(
            update={
                "runtime": cfg.runtime.model_copy(
                    update={"device": "cpu", "deterministic_torch": False}
                )
            }
        )
        with patch("chamber.benchmarks.ego_ppo_trainer._set_torch_determinism") as mock_set:
            EgoPPOTrainer.from_config(cfg_no_det, env=env, partner=_build_partner(), ego_uid="ego")
            mock_set.assert_not_called()


class TestEgoPPOTrainerPublicSurface:
    def test_module_exports(self) -> None:
        """plan/10 §2: __all__ pins the public surface."""
        from chamber.benchmarks import ego_ppo_trainer

        assert "EgoPPOTrainer" in ego_ppo_trainer.__all__
        assert "compute_gae" in ego_ppo_trainer.__all__
        assert "normalize_advantages" in ego_ppo_trainer.__all__

    def test_class_marker_signals_module_gae_routing(self) -> None:
        """plan/05 §5: ``USES_MODULE_GAE`` is the contract pin for the property test."""
        assert EgoPPOTrainer.USES_MODULE_GAE is True
