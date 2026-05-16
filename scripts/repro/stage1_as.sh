#!/usr/bin/env bash
# Reproduce the Stage-1 AS axis spike
# (ADR-007 §Implementation staging Stage 1; plan/07 §5).
#
# Mirrors the structure of scripts/repro/stage0_smoke.sh: a thin shell
# wrapper that pre-flights the on-disk pre-registration against its
# committed git tag (ADR-007 §Discipline), then dispatches to the
# chamber-side adapter and the ADR-008 evaluation pipeline. The script
# does NOT touch GPU directly; the underlying chamber-spike run reads
# CONCERTO_DEVICE / the env's own device-resolution per plan/05 §3.5.
#
# Usage:
#   bash scripts/repro/stage1_as.sh
#
# Exit codes:
#   0  — verify-prereg + run + eval all succeeded.
#   4  — pre-registration SHA mismatch (chamber-spike verify-prereg).
#   6  — Stage-1 AS adapter not yet implemented (chamber-spike run).
#   anything else — bubbled up from the underlying tool.
#
# The resulting artefacts land under spikes/results/stage1-AS-<date>/;
# the closing summary names both paths. Commit that directory under
# the canonical spikes/results/<tag>/ path per plan/07 §2.
#
# `chamber-eval` is invoked with a single SpikeRun and emits a PARTIAL
# leaderboard row by design — the canonical Stage-1 archive is per-axis
# (plan/07 §5). The Month-3 summarise step (plan/07 §T5b.10) is what
# combines the six per-axis archives into a complete HRS vector.
#
# The <date> in RUN_DIR is taken from UTC so two operators on opposite
# sides of the dateline land in the same directory.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

echo "==> Stage-1 AS spike (ADR-007 §Implementation staging; plan/07 §5)"
echo "    $(uv run python -c 'from chamber.utils.device import device_report; print(device_report())' 2>/dev/null)"
echo ""

echo "==> Verifying pre-registration (ADR-007 §Discipline)"
uv run chamber-spike verify-prereg --spike spikes/preregistration/AS.yaml

RUN_DIR="spikes/results/stage1-AS-$(date -u +%Y%m%d)"
mkdir -p "${RUN_DIR}"
SPIKE_JSON="${RUN_DIR}/spike_as.json"
LEADERBOARD_JSON="${RUN_DIR}/leaderboard.json"

echo ""
echo "==> Launching spike — output: ${SPIKE_JSON}"
uv run chamber-spike run --axis AS --output "${SPIKE_JSON}"

echo ""
echo "==> Running ADR-008 evaluation pipeline — output: ${LEADERBOARD_JSON}"
uv run chamber-eval "${SPIKE_JSON}" --output "${LEADERBOARD_JSON}"

echo ""
echo "==> PASS — Stage-1 AS spike complete."
echo "    SpikeRun:    ${SPIKE_JSON}"
echo "    Leaderboard: ${LEADERBOARD_JSON}"
echo "    Commit the result archive under ${RUN_DIR}/ per plan/07 §2."
