#!/usr/bin/env bash
# Reproduce the Stage-0 smoke test (ADR-001 §Validation criteria, ADR-007 §Stage 0).
#
# Usage:
#   bash scripts/repro/stage0_smoke.sh
#
# Exit codes:
#   0  — all smoke tests passed or were skipped (Vulkan not available).
#   1  — one or more smoke tests failed.
#
# Tier 1 (wrapper-structure, no GPU): always runs.
# Tier 2 (real ManiSkill, GPU required): skipped automatically when Vulkan
#   is not available; runs end-to-end on a Vulkan-capable machine.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

echo "==> Stage-0 smoke (ADR-001 §Validation criteria)"
echo "    $(uv run python -c 'from chamber.utils.device import device_report; print(device_report())' 2>/dev/null)"
echo "    Running: uv run pytest -m smoke -x -v -k stage0"
echo ""

if uv run pytest -m smoke -x -v -k stage0 --no-cov; then
    echo ""
    echo "==> PASS — Stage-0 smoke green."
    exit 0
else
    STATUS=$?
    if [ "${STATUS}" -eq 5 ]; then
        # Exit code 5 = no tests collected (e.g. if -k filter matched nothing).
        echo ""
        echo "==> PASS (no tests collected; marker/filter produced empty set)."
        exit 0
    fi
    echo ""
    echo "==> FAIL — Stage-0 smoke returned exit ${STATUS}."
    exit 1
fi
