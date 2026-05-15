#!/usr/bin/env bash
# Reproduce the M4a draft-zoo seed (T4b.14; ADR-009 §Decision).
#
# Runs the 100k-frame ego-AHT HAPPO experiment on the ADR-001 Stage-0
# rig-validated env on a Vulkan-capable Linux box, then publishes the
# step-50000 checkpoint as the canonical
# ``local://artifacts/happo_seed7_step50k.pt`` the M4a FrozenHARLPartner
# adapter (T4.6) loads in M4 Phase 3.
#
# This script is GPU-only. CPU-only hosts get a loud failure at the
# device probe; reproduction requires the same Linux + CUDA environment
# the user-facing Docker recipe in docs/how-to/run-on-gpu.md sets up.
#
# Usage:
#   bash scripts/repro/zoo_seed.sh
#
# Exit codes:
#   0  — training completed + the published artefact's SHA-256 was
#        written.
#   1  — training failed or the published artefact wasn't found.
#   2  — CPU-only host (torch_device() != "cuda"). Loud-fail before
#        any work happens.
#
# This script targets the Stage-0 rig-validation env (zero reward by
# design — see ADR-001 §Validation criteria), so the empirical-
# guarantee slope test is intentionally NOT invoked here: a flat
# zero-reward curve has slope = 0 and trips the gate, even though the
# rig validation itself succeeded. The ``--check-guarantee`` flag (the
# ADR-002 §Risks #1 trip-wire) belongs on real-task spike runners, not
# on this rig-validation reproduction. Phase-1+ Stage-1 task spikes
# will wire the flag into their own runner scripts.
#
# plan/05 §6 #5 + #6, plan/08 §10–§11.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

CONFIG="configs/training/ego_aht_happo/stage0_smoke.yaml"
PUBLISHED_NAME="happo_seed7_step50k.pt"
INNER_STEP_TAG="_step50000.pt"
ARTIFACTS_ROOT="${REPO_ROOT}/artifacts"
INNER_ARTIFACTS_DIR="${ARTIFACTS_ROOT}/artifacts"
PUBLISHED_PT="${INNER_ARTIFACTS_DIR}/${PUBLISHED_NAME}"
PUBLISHED_SIDECAR="${PUBLISHED_PT}.json"
SHA_OUT_DIR="${SCRIPT_DIR}/artifacts"
SHA_OUT_PATH="${SHA_OUT_DIR}/${PUBLISHED_NAME}.sha256"

echo "==> Zoo-seed run (T4b.14; ADR-009 §Decision)"
echo "    $(uv run python -c 'from chamber.utils.device import device_report; print(device_report())' 2>/dev/null)"

DEVICE="$(uv run python -c 'from chamber.utils.device import torch_device; print(torch_device())' 2>/dev/null)"
if [ "${DEVICE}" != "cuda" ]; then
    echo ""
    echo "==> FAIL — torch_device() reported '${DEVICE}', not 'cuda'."
    echo "    The zoo-seed run is GPU-only. Reproduce on a Linux box with a"
    echo "    CUDA-capable GPU per docs/how-to/run-on-gpu.md."
    exit 2
fi

echo "    Running: uv run chamber-spike train --config ${CONFIG}"
echo ""

# --check-guarantee is intentionally NOT passed here — the Stage-0
# smoke env is zero-reward by design (ADR-001 §Validation criteria) and
# the slope test always trips on a flat reward curve. Real-task spikes
# wire the trip-wire into their own runner scripts.
if uv run chamber-spike train --config "${CONFIG}"; then
    echo ""
    echo "==> Training completed."
else
    STATUS=$?
    echo ""
    echo "==> FAIL — chamber-spike train returned exit ${STATUS}."
    exit "${STATUS}"
fi

# Publish the step-50000 checkpoint under the M4a-contract name. The
# trainer's internal naming uses ``{run_id}_step50000.pt`` (per
# concerto.training.ego_aht._save_run_checkpoint); the published name
# is the stable artefact identifier the M4a FrozenHARLPartner adapter
# (T4.6) loads.
echo ""
echo "==> Publishing step-50000 checkpoint as ${PUBLISHED_NAME}"
shopt -s nullglob
inner_pts=( "${INNER_ARTIFACTS_DIR}"/*"${INNER_STEP_TAG}" )
shopt -u nullglob
if [ "${#inner_pts[@]}" -ne 1 ]; then
    echo "==> FAIL — expected exactly 1 step-50000 checkpoint, found ${#inner_pts[@]}:"
    printf '    %s\n' "${inner_pts[@]}"
    exit 1
fi
source_pt="${inner_pts[0]}"
source_sidecar="${source_pt}.json"
if [ ! -f "${source_sidecar}" ]; then
    echo "==> FAIL — sidecar ${source_sidecar} not found alongside ${source_pt}."
    exit 1
fi
cp "${source_pt}" "${PUBLISHED_PT}"
cp "${source_sidecar}" "${PUBLISHED_SIDECAR}"
echo "    Published: ${PUBLISHED_PT}"
echo "    Sidecar:   ${PUBLISHED_SIDECAR}"

# Compute the SHA-256 of the published .pt and write it under
# scripts/repro/artifacts/ for Branch 5 to commit.
mkdir -p "${SHA_OUT_DIR}"
if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "${PUBLISHED_PT}" | awk '{print $1}' > "${SHA_OUT_PATH}"
elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "${PUBLISHED_PT}" | awk '{print $1}' > "${SHA_OUT_PATH}"
else
    echo "==> FAIL — neither sha256sum nor shasum available on PATH."
    exit 1
fi
SHA_VALUE="$(cat "${SHA_OUT_PATH}")"

echo ""
echo "==> PASS — zoo-seed artefact published."
echo "    SHA-256: ${SHA_VALUE}"
echo "    Written: ${SHA_OUT_PATH}"
echo ""
echo "    Next step: commit ${SHA_OUT_PATH#${REPO_ROOT}/}"
echo "    and the published .pt + .pt.json artefacts (out-of-tree per"
echo "    plan/04 §3.8) per the Branch 5 manifest PR (T4b.14)."
exit 0
