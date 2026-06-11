# SPDX-License-Identifier: Apache-2.0
"""Programmatic driver for ``braking_parity_comparative_audit.sh`` (P1.04.6).

Runs one 200-step AS-homo Stage-1b training cell with the
``tau_brake`` value sourced from the
``CONCERTO_BRAKING_AUDIT_TAU_BRAKE`` env var and copies the resulting
JSONL to the path named in ``CONCERTO_BRAKING_AUDIT_JSONL``. The bash
wrapper invokes this twice (one CBF-only, one CBF + braking) and reads
the two JSONLs back to compute the comparative summary.

The driver is kept in ``scripts/repro/`` (alongside its bash wrapper)
rather than under ``src/chamber/cli/`` because it is a one-shot audit
tool, not a public CLI surface. Underscore-prefixed so it is treated
as a private helper.

Requires SAPIEN + Vulkan (the Stage-1b env is the test target).
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from chamber.benchmarks.stage1_common import TrainedPolicyFactory
from chamber.envs.stage1_pickplace import make_stage1_pickplace_env
from chamber.utils.device import sapien_gpu_available
from concerto.training.config import (
    EgoAHTConfig,
    EnvConfig,
    HAPPOHyperparams,
    PartnerConfig,
    RuntimeConfig,
    SafetyConfig,
)

_AS_HOMO: str = "stage1_pickplace_panda_only_mappo_shared_param"


def _cfg(tmp_dir: Path, *, tau_brake: float, total_frames: int = 200) -> EgoAHTConfig:
    """Stage-1b AS-homo cfg sized for the 200-step comparative audit."""
    return EgoAHTConfig(
        seed=0,
        total_frames=total_frames,
        checkpoint_every=max(total_frames, 10_000),
        artifacts_root=tmp_dir / "artifacts",
        log_dir=tmp_dir / "logs",
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
        safety=SafetyConfig(
            enabled=True, saturation_threshold=0.9, cbf_gamma=5.0, tau_brake=tau_brake
        ),
    )


def main() -> int:
    """Drive one 200-step cell + copy the JSONL to the audit output path."""
    if not sapien_gpu_available():
        sys.stderr.write(
            "_braking_parity_audit_driver: SAPIEN/Vulkan unavailable; aborting. "
            "Run on a GPU host with `uv sync --group train`.\n"
        )
        return 2

    tau_brake_str = os.environ.get("CONCERTO_BRAKING_AUDIT_TAU_BRAKE")
    target_jsonl_str = os.environ.get("CONCERTO_BRAKING_AUDIT_JSONL")
    if tau_brake_str is None or target_jsonl_str is None:
        sys.stderr.write(
            "_braking_parity_audit_driver: missing env vars. Set "
            "CONCERTO_BRAKING_AUDIT_TAU_BRAKE and CONCERTO_BRAKING_AUDIT_JSONL "
            "(typically via scripts/repro/braking_parity_comparative_audit.sh).\n"
        )
        return 2

    tau_brake = float(tau_brake_str)
    target_jsonl = Path(target_jsonl_str)
    target_jsonl.parent.mkdir(parents=True, exist_ok=True)

    work_dir = target_jsonl.parent / f"_work_{target_jsonl.stem}"
    cfg = _cfg(work_dir, tau_brake=tau_brake)

    factory = TrainedPolicyFactory(cfg=cfg)
    eval_env = make_stage1_pickplace_env(condition_id=_AS_HOMO, episode_length=50, root_seed=0)
    try:
        factory(eval_env, seed=0)
    finally:
        eval_env.close()

    # Copy the emitted JSONL to the audit output path. The factory's
    # run_training keys the JSONL by run_id under cfg.log_dir; pick the
    # most-recent file.
    jsonl_files = sorted(cfg.log_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not jsonl_files:
        sys.stderr.write(f"_braking_parity_audit_driver: no JSONL produced under {cfg.log_dir}.\n")
        return 3
    shutil.copyfile(jsonl_files[0], target_jsonl)
    sys.stdout.write(f"_braking_parity_audit_driver: tau_brake={tau_brake} → {target_jsonl}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
