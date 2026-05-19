# SPDX-License-Identifier: Apache-2.0
"""Tier-1 (CPU, no SAPIEN) — braking-fallback parity in the training loop (P1.04.6).

Pins the contract that :func:`concerto.training.ego_aht.train` now
calls :func:`concerto.safety.braking.maybe_brake` in front of the
CBF-QP outer filter (ADR-007 §Stage 1b Revision 8). Asserts:

1. The training loop runs end-to-end with ``safety_emergency_controllers``
   wired (CBF + braking composition).
2. When the synthetic snapshot builder reports an imminent collision
   (TTC well under ``cfg.safety.tau_brake``), the
   ``safety_telemetry_final`` event carries
   ``n_braking_fires > 0`` and a matching ``braking_fire_rate``.
3. When the snapshots place agents far apart (TTC = +inf), braking
   never fires and the cell behaves exactly as the P1.04.5 CBF-only
   path did.
4. ``cfg.safety.tau_brake`` is consumed: lowering it past the cell's
   actual TTC suppresses fires; raising it above the cell's TTC
   fires every step.

The synthetic snapshot builder bypasses MPE's positions so the test
is deterministic + fast; the production wiring lives in
:func:`chamber.benchmarks.training_runner.build_safety_snapshot_builder`
+ :meth:`Stage1PickPlaceEnv.build_agent_snapshots`.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

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
from concerto.safety.emergency import CartesianAccelEmergencyController
from concerto.training.config import (
    EgoAHTConfig,
    EnvConfig,
    HAPPOHyperparams,
    PartnerConfig,
    RuntimeConfig,
    SafetyConfig,
)
from concerto.training.ego_aht import train


def _cfg(tmp_path: Path, *, tau_brake: float, total_frames: int = 30) -> EgoAHTConfig:
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
        safety=SafetyConfig(enabled=True, tau_brake=tau_brake),
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


def _build_safety_wire():  # type: ignore[no-untyped-def]
    """Hand-build the five safety kwargs (filter / state / bounds / builder / dt)."""
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
    return safety_filter, state, bounds, 0.1


def _emergency_controllers() -> dict[str, CartesianAccelEmergencyController]:
    return {
        "ego": CartesianAccelEmergencyController(),
        "partner": CartesianAccelEmergencyController(),
    }


def _imminent_collision_builder(_env: object) -> dict[str, AgentSnapshot]:
    """Snapshot builder that always reports the two agents on a collision course.

    Positions 1 cm apart along x; velocities closing at 1 m/s. Pairwise
    TTC ≈ 0.01 s, well under any reasonable ``tau_brake``.
    """
    return {
        "ego": AgentSnapshot(
            position=np.array([0.0, 0.0], dtype=np.float64),
            velocity=np.array([1.0, 0.0], dtype=np.float64),
            radius=0.0,
        ),
        "partner": AgentSnapshot(
            position=np.array([0.01, 0.0], dtype=np.float64),
            velocity=np.array([-1.0, 0.0], dtype=np.float64),
            radius=0.0,
        ),
    }


def _far_apart_builder(_env: object) -> dict[str, AgentSnapshot]:
    """Snapshot builder that always reports agents stationary, far apart.

    No closing motion → TTC = +inf → braking never fires.
    """
    return {
        "ego": AgentSnapshot(
            position=np.array([0.0, 0.0], dtype=np.float64),
            velocity=np.zeros(2, dtype=np.float64),
            radius=0.05,
        ),
        "partner": AgentSnapshot(
            position=np.array([10.0, 0.0], dtype=np.float64),
            velocity=np.zeros(2, dtype=np.float64),
            radius=0.05,
        ),
    }


def _read_final_event(cfg: EgoAHTConfig, run_id: str) -> dict[str, object]:
    """Load the per-cell ``safety_telemetry_final`` event from the JSONL."""
    jsonl_path = cfg.log_dir / f"{run_id}.jsonl"
    events = [json.loads(line) for line in jsonl_path.read_text().splitlines()]
    finals = [e for e in events if e.get("event") == "safety_telemetry_final"]
    assert len(finals) == 1, f"expected 1 final event; got {len(finals)}"
    return finals[0]


class TestBrakingFiresOnImminentCollision:
    """When TTC < tau_brake every step, n_braking_fires == n_filter_calls."""

    def test_braking_fires_every_step_under_imminent_collision(self, tmp_path: Path) -> None:
        cfg = _cfg(tmp_path, tau_brake=0.100, total_frames=30)
        filt, state, bounds, dt = _build_safety_wire()
        result = train(
            cfg,
            env=_mpe_env(),
            partner=_scripted_partner(),
            safety_filter=filt,
            safety_state=state,
            safety_bounds=bounds,
            safety_snapshot_builder=_imminent_collision_builder,
            safety_dt=dt,
            safety_emergency_controllers=_emergency_controllers(),
        )
        final = _read_final_event(cfg, result.curve.run_id)
        # Every step fires braking; QP is skipped on those steps.
        assert final["n_braking_fires"] == cfg.total_frames
        assert final["n_filter_calls"] == cfg.total_frames
        assert final["braking_fire_rate"] == pytest.approx(1.0)
        # CBF-QP own fallback flag stays at 0 (the QP didn't run on
        # braking-fired steps).
        assert final["n_fallback_fires"] == 0


class TestBrakingSilentWhenSafe:
    """When TTC = +inf every step, n_braking_fires == 0 and the cell matches CBF-only."""

    def test_braking_never_fires_when_agents_are_stationary_and_far_apart(
        self, tmp_path: Path
    ) -> None:
        cfg = _cfg(tmp_path, tau_brake=0.100, total_frames=30)
        filt, state, bounds, dt = _build_safety_wire()
        result = train(
            cfg,
            env=_mpe_env(),
            partner=_scripted_partner(),
            safety_filter=filt,
            safety_state=state,
            safety_bounds=bounds,
            safety_snapshot_builder=_far_apart_builder,
            safety_dt=dt,
            safety_emergency_controllers=_emergency_controllers(),
        )
        final = _read_final_event(cfg, result.curve.run_id)
        assert final["n_braking_fires"] == 0
        assert final["braking_fire_rate"] == 0.0


class TestTauBrakeIsConsumed:
    """``cfg.safety.tau_brake`` controls the fire rate."""

    def test_tiny_tau_brake_suppresses_fires(self, tmp_path: Path) -> None:
        """tau_brake = 1 µs is below the synthetic 10 ms TTC → never fires."""
        cfg = _cfg(tmp_path, tau_brake=1e-6, total_frames=20)
        filt, state, bounds, dt = _build_safety_wire()
        result = train(
            cfg,
            env=_mpe_env(),
            partner=_scripted_partner(),
            safety_filter=filt,
            safety_state=state,
            safety_bounds=bounds,
            safety_snapshot_builder=_imminent_collision_builder,
            safety_dt=dt,
            safety_emergency_controllers=_emergency_controllers(),
        )
        final = _read_final_event(cfg, result.curve.run_id)
        assert final["n_braking_fires"] == 0


class TestEmergencyControllerKwargIsOptional:
    """Omitting ``safety_emergency_controllers`` falls back to per-uid Cartesian default."""

    def test_runs_without_explicit_controllers(self, tmp_path: Path) -> None:
        """``safety_emergency_controllers=None`` ⇒ maybe_brake builds its own per-uid default."""
        cfg = _cfg(tmp_path, tau_brake=0.100, total_frames=20)
        filt, state, bounds, dt = _build_safety_wire()
        # Note: no safety_emergency_controllers kwarg passed.
        result = train(
            cfg,
            env=_mpe_env(),
            partner=_scripted_partner(),
            safety_filter=filt,
            safety_state=state,
            safety_bounds=bounds,
            safety_snapshot_builder=_imminent_collision_builder,
            safety_dt=dt,
        )
        final = _read_final_event(cfg, result.curve.run_id)
        # Cartesian default is the right shape for 2-D MPE; braking
        # still fires under imminent-collision snapshots.
        assert final["n_braking_fires"] == cfg.total_frames


class TestBrakingDeterminism:
    """ADR-002 P6: two runs at the same seed produce identical braking counts."""

    def test_two_runs_identical_braking_counts(self, tmp_path: Path) -> None:
        def _run(tag: str) -> dict[str, object]:
            cfg = _cfg(tmp_path / tag, tau_brake=0.100, total_frames=30)
            filt, state, bounds, dt = _build_safety_wire()
            result = train(
                cfg,
                env=_mpe_env(),
                partner=_scripted_partner(),
                safety_filter=filt,
                safety_state=state,
                safety_bounds=bounds,
                safety_snapshot_builder=_imminent_collision_builder,
                safety_dt=dt,
                safety_emergency_controllers=_emergency_controllers(),
            )
            return _read_final_event(cfg, result.curve.run_id)

        final_a = _run("a")
        final_b = _run("b")
        assert final_a["n_braking_fires"] == final_b["n_braking_fires"]
        assert final_a["braking_fire_rate"] == final_b["braking_fire_rate"]
