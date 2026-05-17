#!/usr/bin/env bash
# Stage-1b pick-place env Tier-1 (CPU) smoke (ADR-007 §Stage 1b; P1.03).
#
# Runs ``tests/unit/test_stage1_pickplace_tier1.py`` — the no-SAPIEN-scene
# surface for :class:`chamber.envs.stage1_pickplace`. Founder-runnable on
# the M5 MacBook with no GPU host.
#
# Usage:
#   bash scripts/repro/stage1_pickplace_cpu_smoke.sh
#
# Exit codes:
#   0  — all Tier-1 tests passed.
#   1  — one or more Tier-1 tests failed.
#
# Tier-2 (SAPIEN + Vulkan) coverage lives in
# ``tests/integration/test_stage1_pickplace_real.py`` and runs only on a
# Vulkan-capable host — use ``scripts/repro/stage1_pickplace_real_smoke.sh``
# from the RTX 2080 box to exercise it.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

echo "==> Stage-1b pick-place env Tier-1 smoke (ADR-007 §Stage 1b; P1.03)"
echo "    $(uv run python -c 'from chamber.utils.device import device_report; print(device_report())' 2>/dev/null)"
echo "    Running: uv run pytest tests/unit/test_stage1_pickplace_tier1.py -v --no-cov"
echo ""

if uv run pytest tests/unit/test_stage1_pickplace_tier1.py -v --no-cov; then
    echo ""
    echo "==> PASS — Stage-1b Tier-1 (CPU) smoke green on this host."
    echo ""
    echo "    Tier-2 (SAPIEN + Vulkan) coverage is gated; run"
    echo "    'bash scripts/repro/stage1_pickplace_real_smoke.sh' on a"
    echo "    Vulkan-capable host (e.g. the RTX 2080 box) to exercise it."
    exit 0
else
    STATUS=$?
    echo ""
    echo "==> FAIL — Stage-1b Tier-1 smoke returned exit ${STATUS}."
    exit 1
fi
