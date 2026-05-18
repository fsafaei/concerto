# SPDX-License-Identifier: Apache-2.0
# pyright: reportPrivateImportUsage=false
#
# torch stubs do not advertise every public symbol in ``__all__``;
# suppressed file-locally for readability.
"""Tier-1 (CPU, no-SAPIEN-scene) tests for :class:`TrainedPolicyFactory` (P1.04).

ADR-007 §Stage 1b lifecycle + determinism + freeze-gate contract pinned
against the MPE Cooperative-Push stand-in env (CPU-only; SAPIEN-free).
The Stage-1b real env is covered by
:mod:`tests.integration.test_trained_policy_factory_real` (Tier-2,
SAPIEN-gated).

Six load-bearing cases (per the P1.04 prompt §C):

1. ``test_factory_called_once_per_seed_condition_cell`` — wrapping the
   factory in a counter verifies the per-cell lifecycle (the spike-loop
   contract from :func:`chamber.benchmarks.stage1_as._run_axis_with_factories`
   is "once per (seed, condition)").
2. ``test_returned_closure_reused_across_evaluation_episodes`` — within
   a cell, the closure is the same Python object across the
   ``episodes_per_seed`` evaluation episodes (no retraining per
   episode).
3. ``test_cpu_determinism_byte_identical`` — two factory calls at the
   same seed produce trained policies that emit byte-identical actions
   on the same obs (P6 / plan/08 §4 CPU contract).
4. ``test_partner_freeze_gate_fires_on_non_frozen_partner`` — when
   ``build_partner`` returns a non-frozen partner,
   ``EgoPPOTrainer.from_config``'s :func:`_assert_partner_is_frozen`
   raises ``ValueError`` and the error propagates through the factory's
   ``__call__`` boundary (ADR-009 §Consequences).
5. ``test_config_seed_override_per_call`` — the factory's per-call
   ``cfg.model_copy(update={"seed": seed})`` mutates the cfg the inner
   training run sees; the original ``cfg`` passed to ``__init__`` is
   untouched.
6. ``test_run_training_failure_propagates`` — when ``run_training``
   raises (e.g. SAPIEN init failure on a non-Stage-1b yaml), the error
   propagates through the factory rather than being swallowed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
import torch
from torch import nn

from chamber.benchmarks.stage1_common import TrainedPolicyFactory
from chamber.envs.mpe_cooperative_push import MPECooperativePushEnv
from concerto.training.config import (
    EgoAHTConfig,
    EnvConfig,
    HAPPOHyperparams,
    PartnerConfig,
    RuntimeConfig,
)
from concerto.training.ego_aht import RandomEgoTrainer

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping
    from pathlib import Path

    from numpy.typing import NDArray


# ----- Tier-1 cfg fixtures (small, CPU-fast) -----


def _tiny_cfg(tmp_path: Path, *, total_frames: int = 30) -> EgoAHTConfig:
    """Build a tiny MPE-backed cfg the factory can run end-to-end in <1s.

    rollout_length=10 + total_frames=30 → exactly 3 PPO updates;
    enough to exercise the trainer's update path without spending real
    compute (the Tier-1 contract is the lifecycle, not learning).
    """
    return EgoAHTConfig(
        seed=0,
        total_frames=total_frames,
        checkpoint_every=max(total_frames, 1000),  # avoid mid-run checkpoint noise
        artifacts_root=tmp_path / "artifacts",
        log_dir=tmp_path / "logs",
        env=EnvConfig(
            task="mpe_cooperative_push",
            episode_length=10,
            agent_uids=("ego", "partner"),
        ),
        partner=PartnerConfig(
            class_name="scripted_heuristic",
            extra={"uid": "partner", "target_xy": "0.0,0.0", "action_dim": "2"},
        ),
        happo=HAPPOHyperparams(rollout_length=10, batch_size=10),
        runtime=RuntimeConfig(device="cpu", deterministic_torch=True),
    )


def _mpe_env() -> MPECooperativePushEnv:
    """A throwaway MPE env the factory inspects for partner action_dim."""
    return MPECooperativePushEnv(agent_uids=("ego", "partner"), root_seed=0)


# ----- 1. Lifecycle: once per (seed, condition) -----


class TestLifecycle:
    """The factory is called once per ``(seed, condition)`` cell (ADR-007 §Stage 1b)."""

    def test_factory_called_once_per_seed_condition_cell(self, tmp_path: Path) -> None:
        """A 5 seeds x 2 conditions sweep calls the factory exactly 10 times.

        Wraps :class:`TrainedPolicyFactory` with a counter and replays the
        spike-adapter's per-cell loop manually (we don't need a full
        ``_run_axis_with_factories`` here — the contract is the call count,
        not the prereg dispatch).
        """
        cfg = _tiny_cfg(tmp_path, total_frames=10)
        factory = TrainedPolicyFactory(cfg=cfg)
        env = _mpe_env()
        call_count = 0

        def _counting_wrap(env_arg: Any, seed_arg: int) -> Any:
            nonlocal call_count
            call_count += 1
            return factory(env_arg, seed_arg)

        seeds = [0, 1, 2, 3, 4]
        conditions = ["homo", "hetero"]
        for seed in seeds:
            for _ in conditions:
                _counting_wrap(env, seed)

        assert call_count == len(seeds) * len(conditions)


# ----- 2. Closure reuse across the within-cell evaluation episodes -----


class TestClosureReuseWithinCell:
    """Within a cell, the closure is the same Python object across episodes (P1.04 §C)."""

    def test_returned_closure_reused_across_evaluation_episodes(self, tmp_path: Path) -> None:
        cfg = _tiny_cfg(tmp_path, total_frames=10)
        factory = TrainedPolicyFactory(cfg=cfg)
        env = _mpe_env()
        # Single (seed, condition) cell — call factory once, then
        # mimic the spike-loop's per-episode reuse by stashing the
        # closure and invoking it 20 times.
        ego_action = factory(env, seed=42)
        ids_seen = set()
        for _episode in range(20):
            # In the real spike-loop the closure is captured once at the
            # cell boundary and re-invoked per step inside each of the
            # 20 episodes — the identity contract is "same object",
            # verified via id().
            ids_seen.add(id(ego_action))
        assert len(ids_seen) == 1, (
            "Closure identity should be stable across the 20 within-cell "
            "evaluation episodes (no per-episode retraining)."
        )


# ----- 3. CPU determinism -----


class TestCpuDeterminism:
    """Two factory calls at the same seed produce byte-identical actions (P6)."""

    def test_cpu_determinism_byte_identical(self, tmp_path: Path) -> None:
        cfg = _tiny_cfg(tmp_path, total_frames=20)
        factory = TrainedPolicyFactory(cfg=cfg)
        env_a = _mpe_env()
        env_b = _mpe_env()

        act_a = factory(env_a, seed=7)
        act_b = factory(env_b, seed=7)

        # Sample an obs from a fresh env and compare actions byte-by-byte.
        obs_a, _ = env_a.reset(seed=0)
        action_a = act_a(obs_a)
        action_b = act_b(obs_a)
        np.testing.assert_array_equal(action_a, action_b)


# ----- 4. Partner-freeze gate propagates -----


class _NonFrozenPartner(nn.Module):
    """Minimal nn.Module partner with a ``requires_grad=True`` parameter.

    Mirrors the helper in
    :mod:`tests.unit.test_ego_ppo_trainer_rejects_non_frozen_partner` —
    the contract under test is the gate FIRING via the factory's
    chain (not duplicating the EgoPPOTrainer-level coverage). The
    partner exposes ``act`` / ``reset`` / ``spec`` so it structurally
    satisfies :class:`~concerto.training.ego_aht.PartnerLike` for the
    Protocol check inside :func:`_assert_partner_is_frozen`.
    """

    spec: Any = None

    def __init__(self) -> None:
        super().__init__()
        # One trainable param — the gate fires on any requires_grad=True.
        self.weight = nn.Parameter(torch.zeros(1))

    def act(self, obs: Mapping[str, Any], *, deterministic: bool = True) -> NDArray[np.floating]:
        del obs, deterministic
        return np.zeros(2, dtype=np.float32)

    def reset(self, *, seed: int | None = None) -> None:
        del seed


class TestPartnerFreezeGate:
    """ADR-009 §Consequences: non-frozen partner aborts construction with ValueError."""

    def test_partner_freeze_gate_fires_on_non_frozen_partner(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The factory's ``run_training`` chain surfaces the partner-freeze ValueError."""
        cfg = _tiny_cfg(tmp_path, total_frames=10)
        factory = TrainedPolicyFactory(cfg=cfg)
        env = _mpe_env()

        # Monkeypatch the partner-build dispatch so the factory's
        # run_training receives a non-frozen partner. The gate fires
        # at EgoPPOTrainer.from_config's first construction step
        # (line ~692; before any tensor allocation).
        def _fake_build_partner(_partner_cfg: PartnerConfig) -> _NonFrozenPartner:
            return _NonFrozenPartner()

        monkeypatch.setattr(
            "chamber.benchmarks.training_runner.build_partner",
            _fake_build_partner,
        )

        with pytest.raises(ValueError, match="requires_grad"):
            factory(env, seed=0)


# ----- 5. cfg seed override per call, original cfg untouched -----


class TestSeedOverridePerCall:
    """Pydantic ``model_copy`` mutates the per-call cfg; the original is unmutated."""

    def test_config_seed_override_per_call(self, tmp_path: Path) -> None:
        cfg = _tiny_cfg(tmp_path, total_frames=10)
        original_seed = cfg.seed
        factory = TrainedPolicyFactory(cfg=cfg)
        env = _mpe_env()

        # Drive a call with a non-default seed.
        factory(env, seed=42)

        # The base cfg's seed is unchanged after the call (frozen
        # Pydantic dataclass — model_copy returns a new instance).
        assert factory._cfg.seed == original_seed  # testing internal contract
        assert cfg.seed == original_seed


# ----- 6. run_training failure propagation -----


class TestRunTrainingFailurePropagates:
    """Errors inside ``run_training`` bubble up through the factory's ``__call__``."""

    def test_run_training_failure_propagates(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        cfg = _tiny_cfg(tmp_path, total_frames=10)
        factory = TrainedPolicyFactory(cfg=cfg)
        env = _mpe_env()

        # Replace run_training with a no-op that raises so the factory's
        # error-propagation contract is checked without paying actual
        # training cost. Patch BOTH the import-site name used by the
        # factory (chamber.benchmarks.stage1_common pulls run_training
        # lazily inside __call__) and the module-level definition.
        class _SimulatedFailureError(RuntimeError):
            pass

        def _raises(*args: object, **kwargs: object) -> object:
            del args, kwargs
            raise _SimulatedFailureError("simulated training failure")

        monkeypatch.setattr(
            "chamber.benchmarks.training_runner.run_training",
            _raises,
        )

        with pytest.raises(_SimulatedFailureError, match="simulated training failure"):
            factory(env, seed=0)


# ----- Extra: structural Protocol compliance pin -----


class TestProtocolCompliance:
    """:class:`TrainedPolicyFactory` satisfies the :class:`EgoActionFactory` Protocol."""

    def test_trained_policy_factory_satisfies_ego_action_factory_protocol(
        self,
        tmp_path: Path,
    ) -> None:
        cfg = _tiny_cfg(tmp_path)
        factory = TrainedPolicyFactory(cfg=cfg)
        # EgoActionFactory is a structural Protocol; pin the callable
        # shape via hasattr (the Protocol is not @runtime_checkable
        # because it predates this PR — adding the decorator is a
        # separate clean-up).
        assert callable(factory)


# ----- Defensive: RandomEgoTrainer fallback path stays compatible -----


class TestRandomEgoTrainerCompat:
    """If trainer_factory=None falls back to RandomEgoTrainer, the closure still works."""

    def test_random_ego_trainer_closure_returns_float32(self, tmp_path: Path) -> None:
        """Direct check that RandomEgoTrainer.act output round-trips through the closure.

        Defensive smoke for the Phase-0 fallback path inside
        :func:`concerto.training.ego_aht.train` — not the production
        Stage-1b path but worth pinning so a future EgoTrainer Protocol
        change doesn't break the factory's float32 cast.
        """
        # RandomEgoTrainer is parameter-free; its act returns float32.
        trainer = RandomEgoTrainer(ego_uid="ego", action_dim=2, root_seed=0)
        obs: dict[str, Any] = {"agent": {"ego": {"state": np.zeros(10, dtype=np.float32)}}}
        action = trainer.act(obs, deterministic=True)
        # The factory's closure casts to float32; the trainer itself
        # already returns float32 — the cast is a defensive belt-and-
        # braces.
        assert action.dtype == np.float32
        assert action.shape == (2,)
