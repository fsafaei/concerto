#!/usr/bin/env bash
# Co-carry Rung-0 attach-stability smoke (ADR-026 §Decision 1; R-2026-06-B Rungs 0-1).
#
# The mandatory Rung-0 gate: two Panda arms attached to one rigid bar via the
# dual-hold 6-DOF SAPIEN drive, held under a zero-action hold across seeds.
# Asserts the attach is stable — the bar is held, telemetry stays finite (no
# constraint-solver blow-up), and the wrist constraint-solver force stays
# bounded. This gate MUST pass before Rung 1 (R-2026-06-B "rigid-joint
# stability check gates Phase A").
#
# Runs the Tier-2 stability test + prints a per-seed telemetry summary.
# Needs a Vulkan/GPU host (SAPIEN); on a CPU-only host the test is skipped.
#
# Usage:
#   bash scripts/repro/cocarry_rung0_stability.sh
#
# Exit codes:
#   0 — Rung-0 gate green (or skipped on a non-GPU host, warned loudly).
#   1 — Rung-0 gate failed: fix the drive/contact model before Rung 1.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

echo "==> Co-carry Rung-0 attach-stability smoke (ADR-026 §Decision 1; R-2026-06-B)"
echo "    $(uv run python -c 'from chamber.utils.device import device_report; print(device_report())' 2>/dev/null)"

if ! uv run python -c "from chamber.utils.device import sapien_gpu_available; import sys; sys.exit(0 if sapien_gpu_available() else 1)" 2>/dev/null; then
    echo "==> WARNING — sapien_gpu_available() is False on this host."
    echo "    The Tier-2 stability test will be SKIPPED, not run. Re-run from a Vulkan host."
    echo ""
fi

echo "    Per-seed zero-action hold telemetry:"
uv run python - <<'PY'
from chamber.benchmarks.cocarry_runner import rollout_hold
for seed in range(5):
    m = rollout_hold(seed=seed, n_steps=80)
    print(f"      seed{m.seed} finite={m.finite} "
          f"max_tilt={m.max_tilt_deg:.1f}deg max_stress={m.max_stress_proxy:.0f}N")
PY

echo ""
echo "    Running: uv run pytest tests/integration/test_cocarry_real.py::TestRung0Stability -v --no-cov"
echo ""
if uv run pytest tests/integration/test_cocarry_real.py::TestRung0Stability -v --no-cov; then
    echo ""
    echo "==> PASS — Rung-0 attach is stable. Cleared to build Rung 1."
    exit 0
else
    STATUS=$?
    echo ""
    echo "==> FAIL — Rung-0 stability gate returned exit ${STATUS}. STOP: fix the"
    echo "    drive/contact model before Rung 1 (R-2026-06-B §8 stop criterion)."
    exit 1
fi
