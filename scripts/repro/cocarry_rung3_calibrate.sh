#!/usr/bin/env bash
# Co-carry Rung-3 capability calibration — Step 1 of the PH measurement
# (ADR-026 §Decision 4 + §Validation criteria; ADR-009 §Decision; R-2026-06-B §15 Rung 3).
#
# The MANDATORY capability-calibration gate that defuses the "weaker teammate"
# confound BEFORE any reference/shifted measurement: each policy-shift candidate
# (chamber.partners.cocarry_policy_shift) is paired with a COOPERATIVE REFERENCE
# ego (the matched cocarry_impedance on the ego seat — NOT the frozen incumbent)
# and must clear C_min (chamber.benchmarks.cocarry_ph.C_MIN, grounded on the
# matched reference, fixed at pre-statement). A candidate that cannot be
# capability-matched is EXCLUDED and the exclusion is archived (the excluded
# roster is itself a finding). The matched teammate's calibrated score M is
# measured here so C_min = max(0.75, M - 0.25) is auditable.
#
# Needs a Vulkan/GPU host (SAPIEN); aborts on a CPU-only host.
#
# Usage: bash scripts/repro/cocarry_rung3_calibrate.sh
# Writes: spikes/results/cocarry/rung3/cocarry_rung3_calibration_roster.json
#
# Exit codes:
#   0 — calibration complete; >= 3 candidates cleared the gate (PH measurement
#       may proceed to pre-registration).
#   2 — FEWER THAN 3 candidates cleared the gate: STOP and report (do NOT
#       weaken C_min to pass teammates — R-2026-06-B §15 Rung 3 stop criterion).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

OUT_DIR="spikes/results/cocarry/rung3"
OUT_JSON="${OUT_DIR}/cocarry_rung3_calibration_roster.json"
mkdir -p "${OUT_DIR}"

echo "==> Co-carry Rung-3 capability calibration (ADR-026 §Decision 4; R-2026-06-B §15 Rung 3)"

if ! uv run python -c "from chamber.utils.device import sapien_gpu_available; import sys; sys.exit(0 if sapien_gpu_available() else 1)" 2>/dev/null; then
    echo "==> ERROR — sapien_gpu_available() is False; this measurement needs a Vulkan/GPU host. Aborting."
    exit 1
fi

OUT_JSON="${OUT_JSON}" uv run python scripts/repro/_cocarry_rung3_calibrate.py
RC=$?

echo ""
if [ "${RC}" -eq 0 ]; then
    echo "==> >= 3 candidates cleared the capability gate. Next: commit the pre-registration"
    echo "    (cocarry_rung3_ph_prereg.json) + git tag BEFORE measuring, then"
    echo "    bash scripts/repro/cocarry_rung3_ph_measure.sh"
    exit 0
else
    echo "==> FEWER THAN 3 candidates cleared the gate — STOP and report (R-2026-06-B §15 Rung 3)."
    echo "    The excluded roster is the finding. Do NOT weaken C_min to pass teammates."
    exit 2
fi
