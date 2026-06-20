#!/usr/bin/env bash
# Co-carry Rung-4c same-ego EH-vs-control-style floor (committed generator;
# ADR-026 §D4; R-2026-06-B §15 Rung 4c). Reproduces the headline floor table
# from committed code (replaces the PR-#252 development /tmp script) and
# reconciles the 1867-vs-1773 N seed-set discrepancy. f_max is read from the
# committed measurement artifact (NOT re-derived).
#
# Needs a Vulkan/GPU host (SAPIEN). Usage: bash scripts/repro/cocarry_rung4c_sameego_floor.sh
# Writes: spikes/results/cocarry/rung4c/cocarry_rung4c_sameego_floor.json
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "${SCRIPT_DIR}/../.." && pwd)"
echo "==> Co-carry Rung-4c same-ego floor (ADR-026 §D4; R-2026-06-B §15 Rung 4c)"
if ! uv run python -c "from chamber.utils.device import sapien_gpu_available; import sys; sys.exit(0 if sapien_gpu_available() else 1)" 2>/dev/null; then
    echo "==> ERROR — sapien_gpu_available() is False; needs a Vulkan/GPU host. Aborting."
    exit 1
fi
uv run python scripts/repro/_cocarry_rung4c_sameego_floor.py
