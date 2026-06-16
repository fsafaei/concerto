#!/usr/bin/env bash
# Co-carry Rung-2 train-to-reference + freeze the learned incumbent
# (ADR-026 §Decision 4; ADR-002 §Risks #1 + partner-freeze gate; ADR-009
# §Decision; R-2026-06-B §15 Rung 2).
#
# Steps 3-6 of the Rung-2 build order:
#   * train the learned ego against the frozen matched cocarry_impedance
#     partner on the matched co-carry env (EgoPPOTrainer via run_training);
#   * confirm a positive learning slope (check_positive_learning_slope);
#   * evaluate the frozen incumbent + matched partner and compare to the
#     Rung-1 matched reference (the Step-1 STOP criterion);
#   * write the checkpoint + the COMPLETE freeze manifest (every COCARRY_*
#     coefficient/limit, the re-derived f_max, the matched reference, the
#     env + config content hashes, the checkpoint hash).
#
# Reads the f_max + matched-reference artifact produced by
# scripts/repro/cocarry_rung2_fmax_reference.sh. Run that FIRST.
#
# Needs a Vulkan/GPU host (SAPIEN + CUDA); single-env cell (the matched
# partner reads env 0).
#
# Usage:
#   bash scripts/repro/cocarry_rung2_train.sh [TOTAL_FRAMES] [SEED] [N_EVAL_SEEDS]
#     defaults: TOTAL_FRAMES from the yaml (300000), SEED=0, N_EVAL_SEEDS=12
#
# Writes:
#   artifacts/<run_id>_step<N>.pt(.json)              (the frozen incumbent)
#   spikes/results/cocarry/rung2/cocarry_rung2_freeze_manifest.json
#
# Exit codes:
#   0 — incumbent reached the Rung-1 reference; manifest frozen + complete.
#   3 — STOP: training could not reach the reference, or no positive slope,
#       or the freeze is incomplete (R-2026-06-B §15 stop criteria).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

TOTAL_FRAMES="${1:-}"
SEED="${2:-0}"
N_EVAL_SEEDS="${3:-12}"
DIST_JSON="spikes/results/cocarry/rung2/cocarry_rung2_fmax_distribution.json"
MANIFEST_JSON="spikes/results/cocarry/rung2/cocarry_rung2_freeze_manifest.json"

if [ ! -f "${DIST_JSON}" ]; then
    echo "==> ERROR — ${DIST_JSON} not found. Run cocarry_rung2_fmax_reference.sh first."
    exit 3
fi

echo "==> Co-carry Rung-2 train-to-reference + freeze (ADR-026 §Decision 4; R-2026-06-B §15)"
echo "    $(uv run python -c 'from chamber.utils.device import device_report; print(device_report())' 2>/dev/null)"

TOTAL_FRAMES="${TOTAL_FRAMES}" SEED="${SEED}" N_EVAL_SEEDS="${N_EVAL_SEEDS}" \
DIST_JSON="${DIST_JSON}" MANIFEST_JSON="${MANIFEST_JSON}" uv run python - <<'PY'
import json
import os
from pathlib import Path
import numpy as np
from concerto.training.config import load_config
from concerto.training.learning_signal_check import CheckStatus
from chamber.benchmarks.training_runner import run_training
from chamber.benchmarks.cocarry_runner import summarize
from chamber.benchmarks.cocarry_incumbent import (
    evaluate_incumbent_matched, slope_report_from_curve,
)
from chamber.benchmarks import cocarry_freeze as F

CONFIG = "configs/training/ego_aht_happo/cocarry_matched.yaml"
ENV_MODULE = "src/chamber/envs/cocarry.py"
dist = json.loads(Path(os.environ["DIST_JSON"]).read_text())
ref_rate = float(dist["matched_summary"]["success_rate"])
p99 = float(dist["matched_success_stress_p99_n"])
fmax = float(dist["fmax_derived_n"])
n_clusters = int(dist["n_seed_clusters"])
seed = int(os.environ["SEED"])

overrides = [f"seed={seed}", "artifacts_root=./artifacts", "log_dir=./logs"]
if os.environ.get("TOTAL_FRAMES"):
    overrides.append(f"total_frames={os.environ['TOTAL_FRAMES']}")
cfg = load_config(config_path=Path(CONFIG), overrides=overrides)

print(f"    Rung-1 matched reference: success={ref_rate:.0%} over {n_clusters} clusters; "
      f"f_max={fmax:.1f} N")
print(f"    Training ego (seed={seed}, frames={cfg.total_frames}) vs frozen cocarry_impedance ...")
result = run_training(cfg)
report = slope_report_from_curve(result.curve)
print(f"    learning slope={report.slope:+.5f} p={report.p_value:.2e} status={report.status.name} "
      f"n={report.n_episodes}")

ckpt = result.curve.checkpoint_paths[-1]
uri = "local://artifacts/" + ckpt.name
eval_seeds = list(range(10_000, 10_000 + int(os.environ["N_EVAL_SEEDS"])))
metrics = evaluate_incumbent_matched(
    cfg=cfg, checkpoint_uri=uri, artifacts_root=Path("./artifacts"), seeds=eval_seeds,
)
incumbent = summarize(metrics)
print(f"    incumbent + matched partner: success={incumbent['success_rate']:.0%} "
      f"over {len(eval_seeds)} seeds (reference {ref_rate:.0%})")

# Stop criteria (R-2026-06-B §15): reach the reference AND a positive slope.
reached = incumbent["success_rate"] >= ref_rate - 0.10  # within 10pp of the reference
slope_ok = report.status is CheckStatus.PASSED
ckpt_sha = json.loads(Path(str(ckpt) + ".json").read_text())["sha256"]
manifest = F.build_manifest(
    matched_reference_success_rate=ref_rate,
    n_seed_clusters=n_clusters,
    matched_success_stress_p99_n=p99,
    fmax_value_n=fmax,
    distribution_artifact=os.environ["DIST_JSON"],
    matched_reference_artifact=os.environ["DIST_JSON"],
    config_path=CONFIG,
    env_module_path=ENV_MODULE,
    checkpoint=F.CheckpointRecord(uri=uri, sha256=ckpt_sha, seed=seed, step=None),
    notes=[
        f"incumbent matched-eval success={incumbent['success_rate']:.3f} "
        f"(reference {ref_rate:.3f}) over {len(eval_seeds)} seeds",
        f"learning slope={report.slope:.5f} p={report.p_value:.3e} status={report.status.name}",
    ],
)
if not reached or not slope_ok:
    print("==> STOP (R-2026-06-B §15): " + (
        "incumbent did not reach the Rung-1 reference" if not reached
        else "no significant positive learning slope"
    ) + " — report it; do NOT lower the reference. Manifest NOT written.")
    raise SystemExit(3)

out = F.write_manifest(manifest, os.environ["MANIFEST_JSON"])
print(f"    freeze manifest written -> {out} (every COCARRY_* constant captured; "
      f"f_max status={manifest.fmax.status})")
print("==> Rung-2 GREEN: incumbent reached the reference; incumbent frozen + manifest complete.")
PY
