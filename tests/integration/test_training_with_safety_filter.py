# SPDX-License-Identifier: Apache-2.0
"""Tier-1 (CPU, no SAPIEN scene) — safety stack wired into the training loop (P1.04.5).

Pins the contract that :func:`concerto.training.ego_aht.train`'s
additive safety kwargs work correctly: (i) all-``None`` is byte-
identical to pre-P1.04.5; (ii) intent-mismatch validation
loud-fails; (iii) when the safety filter is wired the JSONL gets
``safety_telemetry`` + ``safety_telemetry_final`` events; (iv)
``cfg.safety.enabled=False`` + all-``None``-kwargs runs unfiltered
and the final summary reports ``safety_enabled=False``.

Uses :class:`MPECooperativePushEnv` + ``RandomEgoTrainer`` so the
test runs in <2 seconds on CPU.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

if TYPE_CHECKING:
    from pathlib import Path

from chamber.envs.mpe_cooperative_push import MPECooperativePushEnv
from chamber.partners.api import PartnerSpec
from chamber.partners.heuristic import ScriptedHeuristicPartner
from concerto.safety.api import Bounds, DoubleIntegratorControlModel, SafetyState
from concerto.safety.cbf_qp import AgentSnapshot, ExpCBFQP
from concerto.safety.conformal import reset_on_partner_swap
from concerto.training.config import (
    EgoAHTConfig,
    EnvConfig,
    HAPPOHyperparams,
    PartnerConfig,
    RuntimeConfig,
    SafetyConfig,
)
from concerto.training.ego_aht import train


def _tiny_cfg(tmp_path: Path, *, safety_enabled: bool, total_frames: int = 30) -> EgoAHTConfig:
    """Build a tiny MPE-backed cfg the test can run end-to-end in <1s."""
    return EgoAHTConfig(
        seed=0,
        total_frames=total_frames,
        checkpoint_every=max(total_frames, 1000),
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
        safety=SafetyConfig(enabled=safety_enabled),
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


def _build_test_safety_wire(
    env: MPECooperativePushEnv,  # noqa: ARG001 - signature stable across the test fixture's evolution; env arg may be consumed by future variants that read env.action_space
) -> tuple[Any, SafetyState, Bounds, Any, float]:
    """Hand-build the five safety kwargs for the Tier-1 test."""
    control_models = {
        "ego": DoubleIntegratorControlModel(uid="ego", action_dim=2),
        "partner": DoubleIntegratorControlModel(uid="partner", action_dim=2),
    }
    safety_filter = ExpCBFQP.ego_only(control_models=control_models)
    state = SafetyState(lambda_={("ego", "partner"): 0.0})
    reset_on_partner_swap(state, uids=("ego", "partner"), lambda_safe=0.0, n_warmup_steps=5)
    bounds = Bounds(
        action_linf_component=1.0,
        cartesian_accel_capacity=1.0,
        action_rate=10.0,
        comm_latency_ms=0.0,
        force_limit=50.0,
    )

    def snapshot_builder(_env: object) -> dict[str, AgentSnapshot]:
        """Per-agent snapshot — MPE positions/actions are 2-D; keep snapshot dim aligned."""
        snaps: dict[str, AgentSnapshot] = {}
        for uid, pos2d in _env._positions.items():  # type: ignore[attr-defined]
            snaps[uid] = AgentSnapshot(
                position=np.asarray(pos2d, dtype=np.float64).reshape(2),
                velocity=np.zeros(2, dtype=np.float64),
                radius=0.05,
            )
        return snaps

    return safety_filter, state, bounds, snapshot_builder, 0.1


# ----- 1. Byte-identical: all-None ⇒ pre-P1.04.5 behaviour -----


class TestSafetyKwargsAllNoneIsByteIdentical:
    """All-``None`` safety kwargs ⇒ training loop runs unfiltered, byte-identical to P1.04."""

    def test_all_none_runs_unfiltered_and_returns_curve(self, tmp_path: Path) -> None:
        cfg = _tiny_cfg(tmp_path, safety_enabled=False)
        result = train(cfg, env=_mpe_env(), partner=_scripted_partner())
        assert len(result.curve.per_step_ego_rewards) == cfg.total_frames

    def test_all_none_jsonl_has_no_safety_events(self, tmp_path: Path) -> None:
        """No safety_telemetry / safety_telemetry_final events when disabled."""
        cfg = _tiny_cfg(tmp_path, safety_enabled=False)
        result = train(cfg, env=_mpe_env(), partner=_scripted_partner())
        jsonl_path = cfg.log_dir / f"{result.curve.run_id}.jsonl"
        assert jsonl_path.exists()
        events = [json.loads(line) for line in jsonl_path.read_text().splitlines()]
        event_names = {e.get("event") for e in events}
        assert "safety_telemetry" not in event_names
        assert "safety_telemetry_final" not in event_names


# ----- 2. Intent-mismatch validation -----


class TestSafetyKwargsIntentMismatch:
    """train() loud-fails on cfg ↔ kwargs intent mismatch (P1.04.5)."""

    def test_cfg_enabled_no_kwargs_raises(self, tmp_path: Path) -> None:
        cfg = _tiny_cfg(tmp_path, safety_enabled=True)
        with pytest.raises(ValueError, match="kwargs were not passed"):
            train(cfg, env=_mpe_env(), partner=_scripted_partner())

    def test_cfg_disabled_kwargs_passed_raises(self, tmp_path: Path) -> None:
        cfg = _tiny_cfg(tmp_path, safety_enabled=False)
        env = _mpe_env()
        filt, state, bounds, builder, dt = _build_test_safety_wire(env)
        with pytest.raises(ValueError, match="safety-stack kwargs were passed"):
            train(
                cfg,
                env=env,
                partner=_scripted_partner(),
                safety_filter=filt,
                safety_state=state,
                safety_bounds=bounds,
                safety_snapshot_builder=builder,
                safety_dt=dt,
            )

    def test_partial_kwargs_raises(self, tmp_path: Path) -> None:
        cfg = _tiny_cfg(tmp_path, safety_enabled=True)
        env = _mpe_env()
        filt, state, _, _, _ = _build_test_safety_wire(env)
        with pytest.raises(ValueError, match="complete set"):
            train(
                cfg,
                env=env,
                partner=_scripted_partner(),
                safety_filter=filt,
                safety_state=state,
                # missing bounds / builder / dt
            )


# ----- 3. Safety wired: JSONL events emitted -----


class TestSafetyTelemetryEmitsExpectedEvents:
    """When safety is wired, JSONL gets per-rollout + final events."""

    def test_jsonl_contains_safety_telemetry_window_events(self, tmp_path: Path) -> None:
        cfg = _tiny_cfg(tmp_path, safety_enabled=True, total_frames=30)
        env = _mpe_env()
        filt, state, bounds, builder, dt = _build_test_safety_wire(env)
        result = train(
            cfg,
            env=env,
            partner=_scripted_partner(),
            safety_filter=filt,
            safety_state=state,
            safety_bounds=bounds,
            safety_snapshot_builder=builder,
            safety_dt=dt,
        )
        jsonl_path = cfg.log_dir / f"{result.curve.run_id}.jsonl"
        events = [json.loads(line) for line in jsonl_path.read_text().splitlines()]
        # rollout_length=10 + total_frames=30 → 3 window events
        window_events = [e for e in events if e.get("event") == "safety_telemetry"]
        assert len(window_events) == 3

    def test_jsonl_contains_exactly_one_final_summary(self, tmp_path: Path) -> None:
        cfg = _tiny_cfg(tmp_path, safety_enabled=True, total_frames=30)
        env = _mpe_env()
        filt, state, bounds, builder, dt = _build_test_safety_wire(env)
        result = train(
            cfg,
            env=env,
            partner=_scripted_partner(),
            safety_filter=filt,
            safety_state=state,
            safety_bounds=bounds,
            safety_snapshot_builder=builder,
            safety_dt=dt,
        )
        jsonl_path = cfg.log_dir / f"{result.curve.run_id}.jsonl"
        events = [json.loads(line) for line in jsonl_path.read_text().splitlines()]
        finals = [e for e in events if e.get("event") == "safety_telemetry_final"]
        assert len(finals) == 1
        final = finals[0]
        # Schema pin: every audit-gate predicate input is present.
        for key in (
            "safety_enabled",
            "predictor_kind",
            "lambda_mean",
            "lambda_var",
            "lambda_steady_state",
            "lambda_max_observed",
            "cartesian_accel_capacity",
            "saturation_threshold",
            "saturated",
            "n_filter_calls",
            "n_fallback_fires",
            "n_qp_infeasible",
            # P1.04.6 forward-compat fields:
            "n_braking_fires",
            "braking_fire_rate",
        ):
            assert key in final, f"safety_telemetry_final missing field {key!r}"
        assert final["safety_enabled"] is True
        assert final["predictor_kind"] == "constant_velocity"
        assert final["n_filter_calls"] == cfg.total_frames
