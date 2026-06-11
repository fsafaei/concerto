# SPDX-License-Identifier: Apache-2.0
"""Tier-1 integration test for P1.05.11 / ADR-017 observability slice.

This module is the **keystone** test the slice plan §3.7 named: a
small PPO loop with both sinks enabled in offline mode that asserts:

1. The per-cell ``<run_id>.jsonl`` parses and carries the expected
   line types (``training_start``, ``scalar`` with
   ``metric_namespace="train"``, ``training_end``).
2. The new trainer metric emission contract is satisfied: every
   ``scalar`` line carries the full PPO-health key set
   (``policy_loss``, ``value_loss``, ``dist_entropy``, ``approx_kl``,
   ``clip_fraction``, …).
3. ``chamber-analyze summary`` against the produced JSONL returns the
   expected fields (the agent-side reader contract).
4. The **deterministic-seed equivalence** smoke (slice §9.1; mandatory
   §11 checkpoint 3): the same MPE cell run twice with
   ``observability=off`` vs ``observability=on`` (offline W&B mode)
   produces identical per-step reward curves modulo wall-clock
   timestamps. This proves the observability writes do not perturb
   the science — the critical property D9 from ADR-017 names.

The test runs on the Tier-1 MPE env (no SAPIEN, no GPU). The Stage-1b
deterministic-equivalence smoke that uses the real
:class:`Stage1PickPlaceEnv` is gated by ``pytest.mark.gpu`` and the
host driver/library mismatch documented in Stage 0 probe 2; it lives
alongside this test as an xpass-eligible follow-up.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest  # noqa: TC002 - public test-helper API used in type annotations

if TYPE_CHECKING:  # pragma: no cover
    from pathlib import Path

from chamber.benchmarks.ego_ppo_trainer import EgoPPOTrainer
from chamber.cli.analyze import main as analyze_main
from chamber.envs.mpe_cooperative_push import MPECooperativePushEnv
from chamber.partners.api import PartnerSpec
from chamber.partners.heuristic import ScriptedHeuristicPartner
from concerto.training.config import (
    EgoAHTConfig,
    EnvConfig,
    HAPPOHyperparams,
    PartnerConfig,
    RuntimeConfig,
    WandbConfig,
)
from concerto.training.ego_aht import train


def _cfg(
    tmp_path: Path,
    *,
    wandb_enabled: bool,
    total_frames: int = 30,
) -> EgoAHTConfig:
    """Tiny MPE config that runs in <5s."""
    return EgoAHTConfig(
        seed=0,
        total_frames=total_frames,
        checkpoint_every=max(total_frames, 1000),
        artifacts_root=tmp_path / "artifacts",
        log_dir=tmp_path / "logs",
        wandb=WandbConfig(enabled=wandb_enabled),
        env=EnvConfig(
            task="mpe_cooperative_push",
            episode_length=10,
            agent_uids=("ego", "partner"),
        ),
        partner=PartnerConfig(
            class_name="scripted_heuristic",
            extra={"uid": "partner", "target_xy": "0.0,0.0", "action_dim": "2"},
        ),
        happo=HAPPOHyperparams(rollout_length=10, batch_size=10, n_epochs=1),
        runtime=RuntimeConfig(device="cpu", deterministic_torch=True),
    )


def _mpe_env() -> MPECooperativePushEnv:
    return MPECooperativePushEnv(agent_uids=("ego", "partner"), root_seed=0)


def _scripted_partner() -> ScriptedHeuristicPartner:
    spec = PartnerSpec(
        class_name="scripted_heuristic",
        seed=0,
        checkpoint_step=None,
        weights_uri=None,
        extra={"uid": "partner", "target_xy": "0.0,0.0", "action_dim": "2"},
    )
    return ScriptedHeuristicPartner(spec)


def _trainer_factory_with_logger(cfg, *, env, partner, ego_uid, logger=None):  # type: ignore[no-untyped-def]
    """Wrapper that threads the optional logger kwarg into EgoPPOTrainer.from_config."""
    return EgoPPOTrainer.from_config(cfg, env=env, partner=partner, ego_uid=ego_uid, logger=logger)


class TestPPOObservabilityIntegration:
    """End-to-end: train an MPE cell + assert JSONL emission + chamber-analyze parse."""

    def test_jsonl_carries_scalar_event_with_train_namespace(self, tmp_path: Path) -> None:
        """ADR-017 §Schema: scalar events from EgoPPOTrainer hit the per-cell JSONL."""
        cfg = _cfg(tmp_path, wandb_enabled=False, total_frames=20)
        env = _mpe_env()
        partner = _scripted_partner()
        result = train(
            cfg,
            env=env,
            partner=partner,
            trainer_factory=_trainer_factory_with_logger,
            repo_root=tmp_path,
        )
        jsonl = next((tmp_path / "logs").glob("*.jsonl"))
        lines = [json.loads(ln) for ln in jsonl.read_text().splitlines() if ln.strip()]
        events = {ln.get("event") for ln in lines}
        assert "training_start" in events
        assert "scalar" in events, "trainer did not emit any event=scalar lines"
        assert "training_end" in events
        scalar_lines = [ln for ln in lines if ln.get("event") == "scalar"]
        # The trainer emits PPO-health scalars in the "train" namespace and
        # task-progress scalars (e.g. mean_reward) in the "rollout" namespace
        # (ADR-017 §Schema; task-progress instrumentation). Assert the train
        # namespace carries the PPO health keys; the rollout namespace is a
        # separate task-progress channel and is checked below.
        train_scalars = [ln for ln in scalar_lines if ln.get("metric_namespace") == "train"]
        assert train_scalars, "trainer did not emit any train-namespace scalar lines"
        for line in train_scalars:
            for key in ("policy_loss", "value_loss", "dist_entropy", "approx_kl"):
                assert key in line, f"scalar missing {key}: {line}"
            for key in ("policy_loss", "value_loss"):
                assert np.isfinite(line[key]), f"non-finite {key}={line[key]!r}"
        assert any(ln.get("metric_namespace") == "rollout" for ln in scalar_lines), (
            "trainer did not emit any rollout-namespace task-progress scalars"
        )
        # The result carries the curve the existing assertions touch.
        assert result.curve is not None

    def test_chamber_analyze_summary_returns_expected_fields(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """ADR-017 §Decisions D5: chamber-analyze reads the produced JSONL.

        End-to-end: train one cell with observability on, then invoke
        ``chamber-analyze summary <run_id>`` against the produced
        ``<run_id>.jsonl``. The summary must surface ``run_id``,
        ``seed``, ``task``, and the lambda steady-state (None on this
        cell since safety is disabled, but the FIELD must be present).
        """
        cfg = _cfg(tmp_path, wandb_enabled=False, total_frames=20)
        env = _mpe_env()
        partner = _scripted_partner()
        train(
            cfg,
            env=env,
            partner=partner,
            trainer_factory=_trainer_factory_with_logger,
            repo_root=tmp_path,
        )
        jsonl = next((tmp_path / "logs").glob("*.jsonl"))
        run_id = jsonl.stem
        # structlog's default fallback writes to stdout too; clear the
        # buffer so we only capture the analyze subcommand's output.
        capsys.readouterr()
        rc = analyze_main(
            [
                "summary",
                run_id,
                "--archive-root",
                str(tmp_path / "logs"),
                "--json",
            ]
        )
        assert rc == 0
        out = json.loads(capsys.readouterr().out)
        assert out["run_id"] == run_id
        assert out["seed"] == 0
        assert out["task"] == "mpe_cooperative_push"
        # n_steps non-null — at least one scalar event with a step field.
        assert out["n_steps"] is not None
        assert out["n_steps"] > 0


class TestDeterministicSeedEquivalence:
    """ADR-017 D9 + slice §9.1: observability writes do not perturb training.

    The gating smoke for the slice. Runs the same MPE cell twice with
    ``WandbConfig.enabled=False`` then ``=True`` (WANDB_MODE=offline so
    no network) and asserts the per-step reward curve is byte-identical
    between the two runs.

    Per founder direction (Stage 0 probe 2 outcome; ADR-017 §Decisions D6):
    ``rollout_recorder.enabled=False`` in BOTH runs for this PR — the
    host driver/library mismatch prevents env.render() from returning
    frames, so the recorder equivalence cannot be proved here. Logger
    emission equivalence is what this smoke proves today. Post-reboot,
    the same smoke can be re-run with recorder.enabled=True in both
    runs to prove recorder equivalence (and the env-render xfail flips).
    """

    def _run_cell(
        self,
        tmp_path: Path,
        *,
        wandb_enabled: bool,
        monkeypatch: pytest.MonkeyPatch,
    ) -> list[float]:
        # Force offline mode so wandb.init() does not hit the network
        # even when the cell sets wandb.enabled=true.
        monkeypatch.setenv("WANDB_MODE", "offline")
        cfg = _cfg(tmp_path, wandb_enabled=wandb_enabled, total_frames=20)
        env = _mpe_env()
        partner = _scripted_partner()
        result = train(
            cfg,
            env=env,
            partner=partner,
            trainer_factory=_trainer_factory_with_logger,
            repo_root=tmp_path,
        )
        # Per-step ego rewards are the canonical determinism witness.
        return list(result.curve.per_step_ego_rewards)

    def test_per_step_reward_curve_byte_identical_across_observability_flag(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Slice §9.1: stripping wall_time, the curve is byte-identical.

        We do NOT compare the JSONL files directly (those carry
        run-time-dependent wall_time fields). The canonical determinism
        witness per ADR-002 §Decisions is the per-step reward curve.
        """
        run_off = tmp_path / "run_off"
        run_off.mkdir()
        run_on = tmp_path / "run_on"
        run_on.mkdir()
        curve_off = self._run_cell(run_off, wandb_enabled=False, monkeypatch=monkeypatch)
        curve_on = self._run_cell(run_on, wandb_enabled=True, monkeypatch=monkeypatch)
        assert len(curve_off) == len(curve_on), (
            f"curve length drift: off={len(curve_off)} on={len(curve_on)}; "
            "observability flag must not change the rollout step count."
        )
        for i, (a, b) in enumerate(zip(curve_off, curve_on, strict=True)):
            assert a == b, (
                f"per-step reward divergence at step {i}: off={a!r} on={b!r}. "
                "Observability writes are perturbing the science. STOP per ADR-017 §12.1."
            )
