#!/usr/bin/env bash
# Reproduce the Stage-1 AS axis spike
# (ADR-007 §Implementation staging Stage 1; plan/07 §5).
#
# Mirrors the structure of scripts/repro/stage0_smoke.sh: a thin shell
# wrapper that pre-flights the on-disk pre-registration against its
# committed git tag (ADR-007 §Discipline), then dispatches to the
# chamber-side adapter and the ADR-008 evaluation pipeline. The script
# does NOT touch GPU directly; the underlying chamber-spike run reads
# CONCERTO_DEVICE / the env's own device-resolution per plan/05 §3.5.
#
# Usage:
#   bash scripts/repro/stage1_as.sh
#
# Exit codes:
#   0  — verify-prereg + run + eval + post-regeneration audit all succeeded.
#   4  — pre-registration SHA mismatch (chamber-spike verify-prereg).
#   6  — Stage-1 AS adapter not yet implemented (chamber-spike run).
#   7  — post-regeneration audit failed (schema_version, sub_stage, or
#        prereg_sha did not match the ADR-016 / ADR-007 contract).
#   anything else — bubbled up from the underlying tool.
#
# The resulting artefacts land under spikes/results/stage1-AS-<date>/;
# the closing summary names both paths. Commit that directory under
# the canonical spikes/results/<tag>/ path per plan/07 §2.
#
# `chamber-eval` is invoked with a single SpikeRun and emits a PARTIAL
# leaderboard row by design — the canonical Stage-1 archive is per-axis
# (plan/07 §5). The Month-3 summarise step (plan/07 §T5b.10) is what
# combines the six per-axis archives into a complete HRS vector.
#
# The <date> in RUN_DIR is taken from UTC so two operators on opposite
# sides of the dateline land in the same directory.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

echo "==> Stage-1 AS spike (ADR-007 §Implementation staging; plan/07 §5)"
echo "    $(uv run python -c 'from chamber.utils.device import device_report; print(device_report())' 2>/dev/null)"
echo ""

echo "==> Verifying pre-registration (ADR-007 §Discipline)"
uv run chamber-spike verify-prereg --spike spikes/preregistration/AS.yaml

RUN_DIR="spikes/results/stage1-AS-$(date -u +%Y%m%d)"
mkdir -p "${RUN_DIR}"
SPIKE_JSON="${RUN_DIR}/spike_as.json"
LEADERBOARD_JSON="${RUN_DIR}/leaderboard.json"

echo ""
echo "==> Launching spike — output: ${SPIKE_JSON}"
uv run chamber-spike run --axis AS --output "${SPIKE_JSON}"

echo ""
echo "==> Running ADR-008 evaluation pipeline — output: ${LEADERBOARD_JSON}"
uv run chamber-eval "${SPIKE_JSON}" --output "${LEADERBOARD_JSON}"

echo ""
echo "==> Auditing regenerated SpikeRun (ADR-016 §Validation criteria + ADR-007 §Discipline)"
# Three audit assertions ride on the regenerated archive before it
# can be committed:
#   (1) schema_version == 2 — pins the ADR-016 §Decision bump.
#   (2) sub_stage == "1a"   — pins the Stage-1a routing signal that
#                              ADR-007 §Stage 1a (Phase-0 MPE stand-in)
#                              and the summarize-month3 Defer-routing
#                              depend on.
#   (3) prereg_sha matches  — the on-disk SpikeRun's prereg_sha equals
#                              the blob SHA of the locked YAML at the
#                              tagged commit (ADR-007 §Discipline).
# Each check fails loudly with a one-line diagnostic and exit code 7.
# The block is inlined rather than factored into a shared helper to
# match the inline structure of the verify-prereg step above and keep
# each repro script self-contained (matching the AS/OM twin policy).
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
tagged_sha=$(git ls-tree "${git_tag}" spikes/preregistration/AS.yaml | awk '{print $3}')
if [[ "${prereg_sha}" != "${tagged_sha}" ]]; then
    echo "  FAIL prereg_sha:     on-disk ${prereg_sha} != tagged ${tagged_sha} (ADR-007 §Discipline)" >&2
    exit 7
fi
echo "    PASS schema_version=${schema_version}  sub_stage=${sub_stage}  prereg_sha=${prereg_sha}"

echo ""
echo "==> PASS — Stage-1 AS spike complete."
echo "    SpikeRun:    ${SPIKE_JSON}"
echo "    Leaderboard: ${LEADERBOARD_JSON}"
echo "    Commit the result archive under ${RUN_DIR}/ per plan/07 §2."
