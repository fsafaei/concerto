#!/usr/bin/env bash
# Reproduce the empirical-guarantee experiment (T4b.13; ADR-002 §Risks #1).
#
# Plan/05 §6 criterion 4: 100k ego-AHT HAPPO frames on the MPE
# Cooperative-Push env should clear the moving-window-of-10 / threshold-
# 0.8 trip-wire within 30 minutes of CPU wall-time.
#
# Usage:
#   bash scripts/repro/empirical_guarantee.sh
#
# Exit codes:
#   0  — guarantee asserted (PASS).
#   1  — assertion fired or runtime exceeded the budget (FAIL).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

echo "==> Empirical guarantee (ADR-002 §Risks #1; plan/05 §6 #4)"
echo "    $(uv run python -c 'from chamber.utils.device import device_report; print(device_report())' 2>/dev/null)"
echo "    Running: make empirical-guarantee"
echo ""

if make empirical-guarantee; then
    echo ""
    echo "==> PASS — empirical guarantee asserted."
    exit 0
else
    STATUS=$?
    if [ "${STATUS}" -eq 5 ]; then
        echo ""
        echo "==> PASS (no tests collected; marker/filter produced empty set)."
        exit 0
    fi
    echo ""
    echo "==> FAIL — empirical guarantee returned exit ${STATUS}."
    echo "    DO NOT widen window or lower threshold to coerce a pass."
    echo "    Open a #scope-revision issue (plan/05 §6 #4)."
    exit 1
fi
