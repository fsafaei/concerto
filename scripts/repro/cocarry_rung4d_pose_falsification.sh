#!/usr/bin/env bash
# Co-carry Rung-4d Stage-A1 fair carry-pose falsification of the Rung-4c EH
# headline (ADR-026 §D4; ADR-005; R-2026-06-B §15 Rung 4d). Kinematic
# min-compliance search + env static/active coupling for the default vs the
# optimised xArm6 pose. Needs a Vulkan/GPU host.
# Usage: bash scripts/repro/cocarry_rung4d_pose_falsification.sh
# Writes: spikes/results/cocarry/rung4d/cocarry_rung4d_pose_falsification.json
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$(cd "${SCRIPT_DIR}/../.." && pwd)"
if ! uv run python -c "from chamber.utils.device import sapien_gpu_available; import sys; sys.exit(0 if sapien_gpu_available() else 1)" 2>/dev/null; then
    echo "==> ERROR — needs a Vulkan/GPU host."; exit 1; fi
uv run python scripts/repro/_cocarry_rung4d_pose_falsification.py
