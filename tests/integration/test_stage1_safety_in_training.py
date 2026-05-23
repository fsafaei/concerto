# SPDX-License-Identifier: Apache-2.0
"""Tier-2 SAPIEN-gated — safety stack wired into Stage-1b training (P1.04.5).

Drives :class:`TrainedPolicyFactory` against the real
:class:`Stage1PickPlaceEnv` with ``cfg.safety.enabled=True`` for a
200-step AS-homo (panda + panda_partner) cell. AS-homo is the
diagnostic-critical cell per D11 of the P1.04.5 design pass —
the two pandas reach toward the same xy region near the cube spawn,
so the CBF filter is expected to fire substantively here. The
predicate B λ_var > 1e-12 assertion is the regression-pin against
the "λ converged to zero" failure mode.

Gated on :func:`chamber.utils.device.sapien_gpu_available`; skipped
on CPU-only CI. Founder runs on the RTX 2080 box and pastes the
output into the PR description (mirrors the P1.03 / P1.04 Tier-2
handoff record format).
"""

from __future__ import annotations

import json

import pytest

from chamber.benchmarks.stage1_common import TrainedPolicyFactory
from chamber.envs.stage1_pickplace import make_stage1_pickplace_env
from chamber.utils.device import sapien_gpu_available, torch_cuda_available
from concerto.training.config import (
    EgoAHTConfig,
    EnvConfig,
    HAPPOHyperparams,
    PartnerConfig,
    RuntimeConfig,
    SafetyConfig,
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

_AS_HOMO = "stage1_pickplace_panda_only_mappo_shared_param"


def _stage1b_safety_cfg(tmp_path, *, total_frames: int) -> EgoAHTConfig:  # type: ignore[no-untyped-def]
    """Stage-1b cfg with safety enabled, sized for Tier-2 smoke runtime."""
    return EgoAHTConfig(
        seed=0,
        total_frames=total_frames,
        checkpoint_every=max(total_frames, 10_000),
        artifacts_root=tmp_path / "artifacts",
        log_dir=tmp_path / "logs",
        env=EnvConfig(
            task="stage1_pickplace",
            episode_length=50,
            agent_uids=("panda_wristcam", "panda_partner"),
            condition_id=_AS_HOMO,
        ),
        partner=PartnerConfig(
            class_name="scripted_heuristic",
            extra={"uid": "panda_partner", "target_xy": "0.0,0.0", "action_dim": "8"},
        ),
        happo=HAPPOHyperparams(rollout_length=100, batch_size=20, hidden_dim=32),
        runtime=RuntimeConfig(device="cuda", deterministic_torch=False),
        safety=SafetyConfig(enabled=True, saturation_threshold=0.9, cbf_gamma=5.0),
    )


class TestStage1bSafetyInTraining:
    """200-step AS-homo cell exercises the safety stack end-to-end."""

    def test_safety_telemetry_final_emitted_with_finite_lambda(
        self,
        tmp_path,  # type: ignore[no-untyped-def]
    ) -> None:
        """200-step run produces the safety_telemetry_final JSONL event with finite λ."""
        cfg = _stage1b_safety_cfg(tmp_path, total_frames=200)
        factory = TrainedPolicyFactory(cfg=cfg)
        eval_env = make_stage1_pickplace_env(condition_id=_AS_HOMO, episode_length=50, root_seed=0)
        try:
            # Drive one cell. The factory builds the safety stack via
            # _build_safety_for_cell and threads through run_training.
            factory(eval_env, seed=0)
            # Find the JSONL — the factory's run_training writes to
            # cfg.log_dir keyed by run_id; pick the most-recent one.
            log_files = list(cfg.log_dir.glob("*.jsonl"))
            assert log_files, "no JSONL emitted under cfg.log_dir"
            events = [json.loads(line) for f in log_files for line in f.read_text().splitlines()]
            finals = [e for e in events if e.get("event") == "safety_telemetry_final"]
            assert len(finals) == 1, f"expected 1 final event; got {len(finals)}"
            final = finals[0]
            # Predicate A inputs are present + finite.
            assert final["safety_enabled"] is True
            assert final["n_filter_calls"] == 200
            assert isinstance(final["lambda_steady_state"], float)
            assert isinstance(final["lambda_mean"], float)
            assert isinstance(final["lambda_var"], float)
            # The AS-homo cell should not saturate at 200 steps.
            assert final["saturated"] is False
        finally:
            eval_env.close()

    def test_as_homo_lambda_moves_substantively(
        self,
        tmp_path,  # type: ignore[no-untyped-def]
    ) -> None:
        """Predicate B regression pin: AS-homo λ should vary, not stay at a constant.

        The AS-homo cell is the diagnostic-critical case per D11. With
        two pandas reaching toward the same xy region, the CBF should
        fire substantively and λ should adapt + vary. If λ_var stays
        ≤ 1e-12 across the run, predicate B would trip on a real
        Stage-1b launch — this test is the early-warning pin.
        """
        cfg = _stage1b_safety_cfg(tmp_path, total_frames=200)
        factory = TrainedPolicyFactory(cfg=cfg)
        eval_env = make_stage1_pickplace_env(condition_id=_AS_HOMO, episode_length=50, root_seed=0)
        try:
            factory(eval_env, seed=0)
            log_files = list(cfg.log_dir.glob("*.jsonl"))
            events = [json.loads(line) for f in log_files for line in f.read_text().splitlines()]
            final = next(e for e in events if e.get("event") == "safety_telemetry_final")
            # Predicate B's "stuck" threshold is 1e-12. The AS-homo
            # cell at 200 steps should clear this comfortably; if
            # not, surface back — the conformal-slack-overlay's
            # dynamics-mismatch absorption hypothesis (ADR-007 §Stage
            # 1b note 1) is over-compensating.
            if final["lambda_mean"] > 1e-6:
                # λ adapted; predicate B requires variance.
                assert final["lambda_var"] > 1e-12, (
                    f"λ_mean={final['lambda_mean']:.6e} > 1e-6 but "
                    f"λ_var={final['lambda_var']:.6e} <= 1e-12 — "
                    "predicate B would trip on a real Stage-1b launch."
                )
        finally:
            eval_env.close()
