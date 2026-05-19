#!/usr/bin/env bash
# Stage-1b pick-place env Tier-2 (SAPIEN + Vulkan) smoke (ADR-007 §Stage 1b; P1.03).
#
# Runs ``tests/integration/test_stage1_pickplace_real.py`` — the real env
# coverage that needs a Vulkan-capable host. Must be run from the RTX 2080
# box (or any host where ``chamber.utils.device.sapien_gpu_available()``
# returns True); on a Vulkan-less host the tests are skipped automatically
# by the module's ``pytestmark = pytest.mark.skipif(...)``.
#
# Usage:
#   bash scripts/repro/stage1_pickplace_real_smoke.sh
#
# Exit codes:
#   0  — all Tier-2 tests passed (or all were skipped, which the script
#        warns about loudly so the founder catches a forgotten GPU host).
#   1  — one or more Tier-2 tests failed.
#
# Tier-1 (CPU-only) coverage is at
# ``scripts/repro/stage1_pickplace_cpu_smoke.sh``.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

echo "==> Stage-1b pick-place env Tier-2 (SAPIEN + Vulkan) smoke (ADR-007 §Stage 1b)"
echo "    $(uv run python -c 'from chamber.utils.device import device_report; print(device_report())' 2>/dev/null)"
echo ""

if ! uv run python -c "from chamber.utils.device import sapien_gpu_available; import sys; sys.exit(0 if sapien_gpu_available() else 1)" 2>/dev/null; then
    echo "==> WARNING — sapien_gpu_available() returned False on this host."
    echo "    The Tier-2 tests will be SKIPPED, not run. Re-run from a Vulkan-capable host"
    echo "    (e.g. the RTX 2080 box) to exercise the real-env coverage."
    echo ""
fi

echo "    Running: uv run pytest tests/integration/test_stage1_pickplace_real.py -v --no-cov"
echo ""

if uv run pytest tests/integration/test_stage1_pickplace_real.py -v --no-cov; then
    echo ""
    echo "==> PASS — Stage-1b Tier-2 smoke green on this host."
    echo "    Paste the test output above into the PR description (ADR-007 §Stage 1b"
    echo "    handoff: ≥4-week trigger guardrail past Month-3 lock review)."
    exit 0
else
    STATUS=$?
    echo ""
    echo "==> FAIL — Stage-1b Tier-2 smoke returned exit ${STATUS}."
    exit 1
fi
