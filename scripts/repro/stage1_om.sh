#!/usr/bin/env bash
# Reproduce the Stage-1 OM axis spike
# (ADR-007 §Implementation staging Stage 1; plan/07 §5).
#
# Twins scripts/repro/stage1_as.sh — same shape, different axis. See
# that script's header for the operating contract, the PARTIAL-row
# rationale, and the UTC-date convention. The only changes here are
# the axis label (OM), the pre-registration path
# (spikes/preregistration/OM.yaml), and the result-archive prefix
# (spikes/results/stage1-OM-<date>/).
#
# Usage:
#   bash scripts/repro/stage1_om.sh
#
# Exit codes:
#   0  — verify-prereg + run + eval + post-regeneration audit all succeeded.
#   4  — pre-registration SHA mismatch (chamber-spike verify-prereg).
#   6  — Stage-1 OM adapter not yet implemented (chamber-spike run).
#   7  — post-regeneration audit failed (schema_version, sub_stage, or
#        prereg_sha did not match the ADR-016 / ADR-007 contract).
#   anything else — bubbled up from the underlying tool.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

echo "==> Stage-1 OM spike (ADR-007 §Implementation staging; plan/07 §5)"
echo "    $(uv run python -c 'from chamber.utils.device import device_report; print(device_report())' 2>/dev/null)"
echo ""

echo "==> Verifying pre-registration (ADR-007 §Discipline)"
uv run chamber-spike verify-prereg --spike spikes/preregistration/OM.yaml

RUN_DIR="spikes/results/stage1-OM-$(date -u +%Y%m%d)"
mkdir -p "${RUN_DIR}"
SPIKE_JSON="${RUN_DIR}/spike_om.json"
LEADERBOARD_JSON="${RUN_DIR}/leaderboard.json"

echo ""
echo "==> Launching spike — output: ${SPIKE_JSON}"
uv run chamber-spike run --axis OM --output "${SPIKE_JSON}"

echo ""
echo "==> Running ADR-008 evaluation pipeline — output: ${LEADERBOARD_JSON}"
uv run chamber-eval "${SPIKE_JSON}" --output "${LEADERBOARD_JSON}"

echo ""
echo "==> Auditing regenerated SpikeRun (ADR-016 §Validation criteria + ADR-007 §Discipline)"
# See scripts/repro/stage1_as.sh for the audit-block rationale; this
# is the OM mirror with the OM-specific pre-registration path.
schema_version=$(jq -r '.schema_version' "${SPIKE_JSON}")
sub_stage=$(jq -r '.sub_stage' "${SPIKE_JSON}")
prereg_sha=$(jq -r '.prereg_sha' "${SPIKE_JSON}")
git_tag=$(jq -r '.git_tag' "${SPIKE_JSON}")

if [[ "${schema_version}" != "2" ]]; then
    echo "  FAIL schema_version: expected 2, got ${schema_version} (ADR-016 §Decision)" >&2
    exit 7
fi
if [[ "${sub_stage}" != "1a" ]]; then
    echo "  FAIL sub_stage:      expected 1a, got ${sub_stage} (ADR-007 §Stage 1a)" >&2
    exit 7
fi
# Guard git_tag before the git ls-tree call below — under set -e, an
# empty / null git_tag would exit with git's native 128 instead of
# the orderly audit-failure path (reviewer P-N1 on PR 3).
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
echo "==> PASS — Stage-1 OM spike complete."
echo "    SpikeRun:    ${SPIKE_JSON}"
echo "    Leaderboard: ${LEADERBOARD_JSON}"
echo "    Commit the result archive under ${RUN_DIR}/ per plan/07 §2."
