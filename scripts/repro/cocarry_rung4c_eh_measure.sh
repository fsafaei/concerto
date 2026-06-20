#!/usr/bin/env bash
# Co-carry Rung-4c EH measurement under the embodiment-invariant coupling-force
# instrument (ADR-026 §D4; ADR-005; ADR-009; R-2026-06-B §15 Rung 4c).
#
# Run AFTER committing + tagging the pre-registration (cocarry_rung4c_eh_prereg.json).
# All conditions use stress_measure="coupling"; f_max is re-derived from the
# matched-Panda-pair coupling distribution then held; the frozen incumbent is
# eval-only (no retrain — its policy is stress-measure-agnostic at inference).
#
# Needs a Vulkan/GPU host (SAPIEN); aborts on a CPU-only host.
#
# Usage: bash scripts/repro/cocarry_rung4c_eh_measure.sh
# Writes: spikes/results/cocarry/rung4c/cocarry_rung4c_eh_measurement.json (+ _invariance.json)
#
# Exit codes: 0 — measurement complete (verdict in the artifact); 3 — STOP
# (matched reference did not reconfirm under the coupling measure).

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

echo "==> Co-carry Rung-4c EH measurement (ADR-026 §D4; R-2026-06-B §15 Rung 4c)"
if ! uv run python -c "from chamber.utils.device import sapien_gpu_available; import sys; sys.exit(0 if sapien_gpu_available() else 1)" 2>/dev/null; then
    echo "==> ERROR — sapien_gpu_available() is False; needs a Vulkan/GPU host. Aborting."
    exit 1
fi

RC=0
uv run python scripts/repro/_cocarry_rung4c_eh_measure.py || RC=$?
echo ""
if [ "${RC}" -eq 0 ]; then
    echo "==> Measurement complete — read the verdict in cocarry_rung4c_eh_measurement.json + the report."
else
    echo "==> STOP (RC=${RC}) — see the artifact."
fi
exit "${RC}"
