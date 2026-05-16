#!/usr/bin/env bash
# Reproduce the Stage-1 OM axis spike
# (ADR-007 §Implementation staging Stage 1; plan/07 §5).
#
# Twins scripts/repro/stage1_as.sh — same shape, different axis. See
# that script's header for the operating contract, the PARTIAL-row
# rationale, and the UTC-date convention. The only changes here are
# the axis label (OM), the pre-registration path
# (spikes/preregistration/OM.yaml), and the result-archive prefix
# (spikes/results/stage1-OM-<date>/).
#
# Usage:
#   bash scripts/repro/stage1_om.sh
#
# Exit codes:
#   0  — verify-prereg + run + eval all succeeded.
#   4  — pre-registration SHA mismatch (chamber-spike verify-prereg).
#   6  — Stage-1 OM adapter not yet implemented (chamber-spike run).
#   anything else — bubbled up from the underlying tool.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

echo "==> Stage-1 OM spike (ADR-007 §Implementation staging; plan/07 §5)"
echo "    $(uv run python -c 'from chamber.utils.device import device_report; print(device_report())' 2>/dev/null)"
echo ""

echo "==> Verifying pre-registration (ADR-007 §Discipline)"
uv run chamber-spike verify-prereg --spike spikes/preregistration/OM.yaml

RUN_DIR="spikes/results/stage1-OM-$(date -u +%Y%m%d)"
mkdir -p "${RUN_DIR}"
SPIKE_JSON="${RUN_DIR}/spike_om.json"
LEADERBOARD_JSON="${RUN_DIR}/leaderboard.json"

echo ""
echo "==> Launching spike — output: ${SPIKE_JSON}"
uv run chamber-spike run --axis OM --output "${SPIKE_JSON}"

echo ""
echo "==> Running ADR-008 evaluation pipeline — output: ${LEADERBOARD_JSON}"
uv run chamber-eval "${SPIKE_JSON}" --output "${LEADERBOARD_JSON}"

echo ""
echo "==> PASS — Stage-1 OM spike complete."
echo "    SpikeRun:    ${SPIKE_JSON}"
echo "    Leaderboard: ${LEADERBOARD_JSON}"
echo "    Commit the result archive under ${RUN_DIR}/ per plan/07 §2."
