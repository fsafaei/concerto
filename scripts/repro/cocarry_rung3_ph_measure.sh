#!/usr/bin/env bash
# Co-carry Rung-3 policy-heterogeneity (PH) measurement — the heterogeneity
# result (ADR-026 §Decision 4 + §Validation criteria; ADR-009 §Decision;
# R-2026-06-B §15 Rung 3).
#
# RUN ONLY AFTER the pre-registration (cocarry_rung3_ph_prereg.json) + the
# calibration roster are committed and git-tagged. Holds the Rung-2 incumbent
# FROZEN and measures whether pairing it with each capability-matched
# policy-shift teammate degrades cooperation vs the matched reference. Computes
# the per-teammate + pooled Δ with the cluster-bootstrap one-sided CI (IQM
# secondary, cluster-robust binomial GEE confirmatory) and applies the
# pre-committed decision + null rule. STOPs if the matched reference does not
# reconfirm ~100%.
#
# Needs a Vulkan/GPU host (SAPIEN); aborts on a CPU-only host.
#
# Usage: bash scripts/repro/cocarry_rung3_ph_measure.sh
# Writes: spikes/results/cocarry/rung3/cocarry_rung3_ph_measurement.json
#
# Exit codes:
#   0 — measurement complete (verdict in the artifact: drop / null / indeterminate).
#   3 — STOP: the matched reference did not reconfirm ~100% (something changed;
#       investigate before trusting the shifted arm).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

echo "==> Co-carry Rung-3 PH measurement (ADR-026 §Decision 4; R-2026-06-B §15 Rung 3)"

if ! uv run python -c "from chamber.utils.device import sapien_gpu_available; import sys; sys.exit(0 if sapien_gpu_available() else 1)" 2>/dev/null; then
    echo "==> ERROR — sapien_gpu_available() is False; this measurement needs a Vulkan/GPU host. Aborting."
    exit 1
fi

uv run python scripts/repro/_cocarry_rung3_ph_measure.py
RC=$?

echo ""
if [ "${RC}" -eq 0 ]; then
    echo "==> Measurement complete. Read the verdict in cocarry_rung3_ph_measurement.json"
    echo "    and the report COCARRY_RUNG3_PH_REPORT.md. STOP at the open PR for founder review."
    exit 0
else
    echo "==> STOP (RC=${RC}): the matched reference did not reconfirm ~100%. Investigate the rig"
    echo "    / determinism / checkpoint before trusting the shifted measurement."
    exit "${RC}"
fi
