# SPDX-License-Identifier: Apache-2.0
"""Rung-5 Stage-3 from-scratch re-freeze — TRAINING launch (ADR-026 §D4; ADR-007).

Trains the matched Panda co-carry ego FROM SCRATCH on the FIXED task
(configs/training/ego_aht_happo/cocarry_rung5_compliant.yaml: compliant K=8000,
coupling instrument, stress_max 365.6, penalty 303/62.6, transport PBRS) and
records the saved-checkpoint manifest. This is the HEAVY step; the selection +
held-out validation + freeze (manifest v2) run in the separate Stage-3b driver
over the checkpoints recorded here.

Budget is the pre-registered hard early-STOP cap: TOTAL_FRAMES defaults to
300000 (the 300k early-STOP budget). Stage-3b evaluates the saved checkpoints on
S; if >= 1 is constraint-clean and reaches matched joint-success >= C_min (0.75),
the from-scratch primary is on track (select the earliest constraint-clean
max-success; continue to the 900k cap only if not yet at max). If none clears by
300k -> STOP the from-scratch primary, trigger the residual-on-impedance
escalation. Run does NOT freeze anything; it only trains + records checkpoints.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def main() -> int:
    from chamber.benchmarks.training_runner import run_training
    from concerto.training.config import load_config

    config = "configs/training/ego_aht_happo/cocarry_rung5_compliant.yaml"
    total_frames = int(os.environ.get("TOTAL_FRAMES", "300000"))
    seed = int(os.environ.get("SEED", "0"))
    out_path = os.environ.get(
        "OUT_JSON", "spikes/results/cocarry/rung5/cocarry_rung5_refreeze_train.json"
    )
    overrides = [f"seed={seed}", f"total_frames={total_frames}"]
    if os.environ.get("CHECKPOINT_EVERY"):
        overrides.append(f"checkpoint_every={int(os.environ['CHECKPOINT_EVERY'])}")
    print(f"    Rung-5 re-freeze TRAINING: {config}  seed={seed}  total_frames={total_frames}")
    cfg = load_config(config_path=Path(config), overrides=overrides)
    result = run_training(cfg)

    paths = [str(p) for p in result.curve.checkpoint_paths]
    # The run_id is the {run_id}_stepN.pt stem prefix.
    run_id = None
    if paths:
        run_id = Path(paths[-1]).name.split("_step")[0]
    artifact = {
        "schema": "cocarry_rung5_refreeze_train/v1",
        "stage": "Rung-5 Stage-3 from-scratch re-freeze (training only)",
        "config": config,
        "seed": seed,
        "total_frames": total_frames,
        "run_id": run_id,
        "checkpoint_paths": paths,
        "n_checkpoints": len(paths),
        "note": (
            "Training only — no selection/validation/freeze here. Stage-3b "
            "(_cocarry_rung5_select_freeze) evaluates these checkpoints on S/V, "
            "applies the earliest-constraint-clean-max-success selection rule, and "
            "builds the freeze manifest v2."
        ),
    }
    Path(out_path).write_text(json.dumps(artifact, sort_keys=True, indent=2), encoding="utf-8")
    print(f"    run_id={run_id}  checkpoints={len(paths)}")
    for p in paths:
        print(f"      {p}")
    print(f"    artifact -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
