#!/usr/bin/env bash
# Co-carry Rung-4b Stage-1 compliant-coupling sweep + gate
# (ADR-026 §Decision 4; ADR-005; ADR-009; R-2026-06-B §15 Rung 4b).
#
# Run AFTER committing + tagging the pre-registration
# (cocarry_rung4b_coupling_prereg.json). Measures C1-C4 over the pre-registered
# grid (Variant A linear + Variant B force-saturated) + the static-hold wrist
# baseline, applies the selection/gate rule, writes the verdict. The compliant
# coupling is the ONLY task change; predicate/f_max/radius/C_min unchanged.
#
# Needs a Vulkan/GPU host (SAPIEN); aborts on a CPU-only host.
#
# Usage: bash scripts/repro/cocarry_rung4b_coupling_sweep.sh
# Writes: spikes/results/cocarry/rung4b/cocarry_rung4b_coupling_sweep.json
#
# Exit codes:
#   0 — PROCEED: a setting satisfies C1-C4 (continue to Stage 2).
#   2 — STOP (honest fallback): no setting satisfies C1-C4 (report the finding).

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

echo "==> Co-carry Rung-4b Stage-1 coupling sweep (ADR-026 §D4; R-2026-06-B §15 Rung 4b)"
if ! uv run python -c "from chamber.utils.device import sapien_gpu_available; import sys; sys.exit(0 if sapien_gpu_available() else 1)" 2>/dev/null; then
    echo "==> ERROR — sapien_gpu_available() is False; needs a Vulkan/GPU host. Aborting."
    exit 1
fi

RC=0
uv run python scripts/repro/_cocarry_rung4b_coupling_sweep.py || RC=$?
echo ""
if [ "${RC}" -eq 0 ]; then
    echo "==> PROCEED — a compliant coupling satisfies C1-C4. Next: Stage 2 (incumbent transfer/re-freeze)."
else
    echo "==> STOP (honest fallback) — see the verdict in cocarry_rung4b_coupling_sweep.json + the report."
fi
exit "${RC}"
