#!/usr/bin/env bash
# Co-carry Rung-4 mixed-embodiment stability gate + capability calibration
# (ADR-026 §Decision 4; ADR-005; ADR-009; R-2026-06-B §15 Rung 4).
#
# The two cheap-fail gates BEFORE any embodiment-heterogeneity measurement:
# (1) the Panda ego + xArm6 partner rigid-bar chain is stable, then (2) the
# xArm6 teammate is capability-calibrated against the cooperative Panda ego at
# C_min. If the xArm6 (and the gripper-less ur_10e backup) cannot clear the
# gate, the measurement does not run — itself the EH result. C_min is NEVER
# weakened to force admission.
#
# Needs a Vulkan/GPU host (SAPIEN); aborts on a CPU-only host.
#
# Usage: bash scripts/repro/cocarry_rung4_eh_calibrate.sh
# Writes: spikes/results/cocarry/rung4/cocarry_rung4_calibration_roster.json
#
# Exit codes:
#   0 — >= 1 embodiment teammate cleared the gate (proceed to pre-register + measure).
#   2 — no embodiment teammate cleared the gate: STOP, report the feasibility finding.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

echo "==> Co-carry Rung-4 EH stability + calibration (ADR-026 §D4; R-2026-06-B §15 Rung 4)"
if ! uv run python -c "from chamber.utils.device import sapien_gpu_available; import sys; sys.exit(0 if sapien_gpu_available() else 1)" 2>/dev/null; then
    echo "==> ERROR — sapien_gpu_available() is False; needs a Vulkan/GPU host. Aborting."
    exit 1
fi

RC=0
uv run python scripts/repro/_cocarry_rung4_eh_calibrate.py || RC=$?
echo ""
if [ "${RC}" -eq 0 ]; then
    echo "==> >= 1 embodiment teammate cleared the gate. Next: pre-register + measure."
    exit 0
else
    echo "==> FEASIBILITY FINDING: no embodiment teammate cleared the capability gate."
    echo "    The measurement does not run (R-2026-06-B §15 Rung 4). Do NOT weaken C_min."
    exit 2
fi
