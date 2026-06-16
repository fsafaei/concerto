#!/usr/bin/env bash
# Co-carry Rung-1 matched competence + coupling positive-control
# (ADR-026 §Decision 1-2; R-2026-06-B Rungs 0-1).
#
# Rung 1 establishes the honest high reference and proves the coupling is
# real, using the matched (identical) Panda pair:
#
#   (a) Competence — both arms running the hand-written impedance controller
#       reach high joint success over a seed sweep.
#   (b) Coupling positive-control (load-bearing; ADR-026 §Decision 2) —
#       a single arm reaches ~= 0 success on the same task, AND the tilt
#       constraint binds on matched successes (the matched bar is held level
#       with margin while a single arm drives the tilt far past the limit).
#
# Also prints the measured wrist constraint-solver force distribution on
# matched successful episodes, from which f_max is derived (a justified high
# percentile) for the Rung-3+ pre-statement (R-2026-06-B "stress proxy").
#
# Gated by the Rung-0 stability smoke (run cocarry_rung0_stability.sh first).
# Needs a Vulkan/GPU host (SAPIEN); skipped on a CPU-only host.
#
# Usage:
#   bash scripts/repro/cocarry_rung1_matched.sh [N_SEEDS]
#
# Exit codes:
#   0 — Rung-1 green (matched competent; single-arm ~= 0; constraints bind).
#   1 — a stop criterion fired (R-2026-06-B §8): single arm can succeed, or
#       the matched pair cannot, or the constraints do not bind. STOP and
#       fix geometry/limits before any learning.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

N_SEEDS="${1:-12}"

echo "==> Co-carry Rung-1 matched competence + coupling positive-control (ADR-026 §Decision 1-2)"
echo "    $(uv run python -c 'from chamber.utils.device import device_report; print(device_report())' 2>/dev/null)"

if ! uv run python -c "from chamber.utils.device import sapien_gpu_available; import sys; sys.exit(0 if sapien_gpu_available() else 1)" 2>/dev/null; then
    echo "==> WARNING — sapien_gpu_available() is False; the Tier-2 tests will be SKIPPED."
    echo ""
fi

echo "    Seed sweep (N=${N_SEEDS}): matched competence + single-arm positive-control"
N_SEEDS="${N_SEEDS}" uv run python - <<'PY'
import os
import numpy as np
from chamber.benchmarks.cocarry_runner import evaluate_condition, summarize
from chamber.envs.cocarry import COCARRY_TILT_MAX_DEG, COCARRY_STRESS_MAX_PROXY_N

seeds = list(range(int(os.environ["N_SEEDS"])))
matched = evaluate_condition(condition_id="cocarry_matched_panda_pair", seeds=seeds)
single = evaluate_condition(condition_id="cocarry_single_arm_positive_control", seeds=seeds)
ms, ss = summarize(matched), summarize(single)
print(f"      MATCHED      success={ms['success_rate']:.0%}  "
      f"max_tilt p50={ms['max_tilt_p50']:.1f} / limit {COCARRY_TILT_MAX_DEG:.0f} deg")
print(f"      SINGLE-ARM   success={ss['success_rate']:.0%}  "
      f"max_tilt p50={ss['max_tilt_p50']:.1f} deg  (centroid p50={ss['centroid_to_goal_p50']:.3f} m)")
print(f"      constraint-force on matched successes: p50={ms['stress_p50']:.0f}  "
      f"p95={ms['success_stress_p95']:.0f}  p99={ms['stress_p99']:.0f}  max={ms['stress_max']:.0f} N "
      f"(current f_max={COCARRY_STRESS_MAX_PROXY_N:.0f} N)")
print(f"      tilt BINDS: matched holds < limit, single-arm exceeds by "
      f"{ss['max_tilt_p50'] / COCARRY_TILT_MAX_DEG:.1f}x")
PY

echo ""
echo "    Running: uv run pytest tests/integration/test_cocarry_real.py::TestRung1MatchedCompetence \\"
echo "             tests/integration/test_cocarry_real.py::TestRung1CouplingPositiveControl -v --no-cov"
echo ""
if uv run pytest \
    tests/integration/test_cocarry_real.py::TestRung1MatchedCompetence \
    tests/integration/test_cocarry_real.py::TestRung1CouplingPositiveControl \
    -v --no-cov; then
    echo ""
    echo "==> PASS — Rung-1 green: matched competent, single-arm ~= 0, tilt constraint binds."
    exit 0
else
    STATUS=$?
    echo ""
    echo "==> FAIL — Rung-1 gate returned exit ${STATUS}. STOP (R-2026-06-B §8): if a single"
    echo "    arm can succeed or the constraints do not bind, the task is not genuinely"
    echo "    coupled — fix geometry/limits before any learning."
    exit 1
fi
