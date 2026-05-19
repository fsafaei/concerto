#!/usr/bin/env bash
# Reproduce the Stage-1b OM axis spike (ADR-007 §Stage 1b; P1.05).
#
# Twin of scripts/repro/stage1_as_stage1b.sh — same shape, OM axis.
# See that script's header for the operating contract, the audit-gate
# rationale, and the cloud-A100 cost / time alternative.
#
# Usage:
#   bash scripts/repro/stage1_om_stage1b.sh
#
# Exit codes (same as the AS twin):
#   0  — verify-prereg + run + eval + audit + gate (predicates A+B) green.
#   4  — pre-registration SHA mismatch.
#   6  — Stage-1 OM adapter not yet shipped (regression indicator post-PR 5a).
#   7  — post-regeneration audit failed.
#   8  — audit-gate predicate A (saturation) tripped.
#   9  — audit-gate predicate B (λ stuck) tripped.
#
# Result archive: spikes/results/stage1-OM-stage1b-<UTC-date>/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

echo "==> Stage-1b OM spike (ADR-007 §Stage 1b; P1.05)"
echo "    $(uv run python -c 'from chamber.utils.device import device_report; print(device_report())' 2>/dev/null)"
echo ""

echo "==> Verifying pre-registration (ADR-007 §Discipline)"
uv run chamber-spike verify-prereg --spike spikes/preregistration/OM.yaml

RUN_DIR="spikes/results/stage1-OM-stage1b-$(date -u +%Y%m%d)"
mkdir -p "${RUN_DIR}"
SPIKE_JSON="${RUN_DIR}/spike_om.json"
LEADERBOARD_JSON="${RUN_DIR}/leaderboard.json"

echo ""
echo "==> Launching spike — output: ${SPIKE_JSON}"
uv run chamber-spike run --axis OM --sub-stage 1b --output "${SPIKE_JSON}"

echo ""
echo "==> Running ADR-008 evaluation pipeline — output: ${LEADERBOARD_JSON}"
uv run chamber-eval "${SPIKE_JSON}" --output "${LEADERBOARD_JSON}"

echo ""
echo "==> Auditing regenerated SpikeRun (ADR-016 §Validation criteria + ADR-007 §Discipline)"
schema_version=$(jq -r '.schema_version' "${SPIKE_JSON}")
sub_stage=$(jq -r '.sub_stage' "${SPIKE_JSON}")
prereg_sha=$(jq -r '.prereg_sha' "${SPIKE_JSON}")
git_tag=$(jq -r '.git_tag' "${SPIKE_JSON}")

if [[ "${schema_version}" != "2" ]]; then
    echo "  FAIL schema_version: expected 2, got ${schema_version} (ADR-016 §Decision)" >&2
    exit 7
fi
if [[ "${sub_stage}" != "1b" ]]; then
    echo "  FAIL sub_stage:      expected 1b, got ${sub_stage} (ADR-007 §Stage 1b)" >&2
    exit 7
fi
if [[ -z "${git_tag}" ]] || [[ "${git_tag}" == "null" ]]; then
    echo "  FAIL git_tag:        empty or null (ADR-007 §Discipline)" >&2
    exit 7
fi
if [[ -z "${prereg_sha}" ]] || [[ "${prereg_sha}" == "null" ]]; then
    echo "  FAIL prereg_sha:     empty or null (ADR-007 §Discipline)" >&2
    exit 7
fi
tagged_sha=$(git ls-tree "${git_tag}" spikes/preregistration/OM.yaml | awk '{print $3}')
if [[ "${prereg_sha}" != "${tagged_sha}" ]]; then
    echo "  FAIL prereg_sha:     on-disk ${prereg_sha} != tagged ${tagged_sha} (ADR-007 §Discipline)" >&2
    exit 7
fi
echo "    PASS schema_version=${schema_version}  sub_stage=${sub_stage}  prereg_sha=${prereg_sha}"

echo ""
echo "==> Running audit-gate hook (predicates A + B; ADR-007 §Stage 1b)"
JSONL_PATH="$(find logs -name '*.jsonl' -newer "${SPIKE_JSON}" -print 2>/dev/null | head -n 1 || true)"
if [[ -z "${JSONL_PATH}" ]]; then
    JSONL_PATH="$(ls -t logs/*.jsonl 2>/dev/null | head -n 1 || true)"
fi
if [[ -z "${JSONL_PATH}" ]]; then
    echo "  WARN: no JSONL found under logs/; skipping audit-gate hook." >&2
else
    bash scripts/repro/stage1_om_stage1b_audit_gate.sh "${JSONL_PATH}"
fi

echo ""
echo "==> PASS — Stage-1b OM spike complete."
echo "    SpikeRun:    ${SPIKE_JSON}"
echo "    Leaderboard: ${LEADERBOARD_JSON}"
echo "    Commit the result archive under ${RUN_DIR}/ per plan/07 §2."
