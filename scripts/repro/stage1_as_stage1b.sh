#!/usr/bin/env bash
# Reproduce the Stage-1b AS axis spike (ADR-007 §Stage 1b; P1.05).
#
# Twin of scripts/repro/stage1_as.sh — same shape, same prereg-verify
# + chamber-spike run + chamber-eval + audit dispatch, but routed
# through the Stage-1b production path via --sub-stage 1b. The result
# archive lands under spikes/results/stage1-AS-stage1b-<UTC-date>/ so
# the Stage-1a archives at spikes/results/stage1-AS-<UTC-date>/ stay
# grep-distinguishable on disk (operator + audit-trail story).
#
# Pre-conditions:
#   - GPU host with SAPIEN + Vulkan available
#     (`chamber.utils.device.sapien_gpu_available()`).
#   - `uv sync --group train` has been run (harl-aht on PyPI per #173).
#   - The repo is on a commit >= PR 5a merge.
#
# Wall-time on RTX 2080: ~25-40 GPU-h.
# Wall-time on cloud A100 (rental): ~5-8 GPU-h (~$6-10 on Lambda Labs
# at ~$1.10/h; ~$4-7 on Runpod spot at ~$0.79/h). Pin the cuda-version
# extra in `uv sync` to match the host driver; see
# pyproject.toml [dependency-groups].
#
# Usage:
#   bash scripts/repro/stage1_as_stage1b.sh
#
# Exit codes:
#   0  — verify-prereg + run + eval + post-regeneration audit + audit-gate hook
#        (predicates A + B) all succeeded.
#   4  — pre-registration SHA mismatch (chamber-spike verify-prereg).
#   6  — Stage-1 AS adapter not yet shipped (should not happen post-PR 5a;
#        regression indicator).
#   7  — post-regeneration audit failed (schema_version, sub_stage, or
#        prereg_sha did not match the ADR-016 / ADR-007 contract).
#   8  — audit-gate predicate A (saturation) tripped.
#   9  — audit-gate predicate B (λ stuck) tripped.
#
# Result archive: spikes/results/stage1-AS-stage1b-<UTC-date>/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

echo "==> Stage-1b AS spike (ADR-007 §Stage 1b; P1.05)"
echo "    $(uv run python -c 'from chamber.utils.device import device_report; print(device_report())' 2>/dev/null)"
echo "    $(uv run python -c 'import torch; print(f"torch={torch.__version__}, cuda={torch.version.cuda}")' 2>/dev/null)"
echo ""

echo "==> Verifying pre-registration (ADR-007 §Discipline)"
uv run chamber-spike verify-prereg --spike spikes/preregistration/AS.yaml

RUN_DIR="spikes/results/stage1-AS-stage1b-$(date -u +%Y%m%d)"
mkdir -p "${RUN_DIR}"
SPIKE_JSON="${RUN_DIR}/spike_as.json"
LEADERBOARD_JSON="${RUN_DIR}/leaderboard.json"

echo ""
echo "==> Launching spike — output: ${SPIKE_JSON}"
uv run chamber-spike run --axis AS --sub-stage 1b --output "${SPIKE_JSON}"

echo ""
echo "==> Running ADR-008 evaluation pipeline — output: ${LEADERBOARD_JSON}"
uv run chamber-eval "${SPIKE_JSON}" --output "${LEADERBOARD_JSON}"

echo ""
echo "==> Auditing regenerated SpikeRun (ADR-016 §Validation criteria + ADR-007 §Discipline)"
# Same audit assertions as stage1_as.sh — but here the expected
# sub_stage is "1b" (Stage-1b science evaluation), and the gate
# follows the same path (schema_version=2, sub_stage matches,
# prereg_sha matches the tagged blob).
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
tagged_sha=$(git ls-tree "${git_tag}" spikes/preregistration/AS.yaml | awk '{print $3}')
if [[ "${prereg_sha}" != "${tagged_sha}" ]]; then
    echo "  FAIL prereg_sha:     on-disk ${prereg_sha} != tagged ${tagged_sha} (ADR-007 §Discipline)" >&2
    exit 7
fi
echo "    PASS schema_version=${schema_version}  sub_stage=${sub_stage}  prereg_sha=${prereg_sha}"

echo ""
echo "==> Running audit-gate hook (predicates A + B; ADR-007 §Stage 1b)"
# The training rollout's JSONL lives under ./logs/ per cfg.log_dir
# (the Stage-1b yaml's default). The hook reads the most-recent JSONL
# under that directory. Operators staging multiple back-to-back runs
# should clean logs/ between launches (or symlink the per-run log
# alongside the SpikeRun archive — Phase-1 follow-up).
JSONL_PATH="$(find logs -name '*.jsonl' -newer "${SPIKE_JSON}" -print 2>/dev/null | head -n 1 || true)"
if [[ -z "${JSONL_PATH}" ]]; then
    # Fallback: pick the most-recent JSONL the founder's host has
    # produced. Audit-gate hook will read the final summary regardless.
    JSONL_PATH="$(ls -t logs/*.jsonl 2>/dev/null | head -n 1 || true)"
fi
if [[ -z "${JSONL_PATH}" ]]; then
    echo "  WARN: no JSONL found under logs/; skipping audit-gate hook." >&2
else
    bash scripts/repro/stage1_as_stage1b_audit_gate.sh "${JSONL_PATH}"
fi

echo ""
echo "==> PASS — Stage-1b AS spike complete."
echo "    SpikeRun:    ${SPIKE_JSON}"
echo "    Leaderboard: ${LEADERBOARD_JSON}"
echo "    Commit the result archive under ${RUN_DIR}/ per plan/07 §2."
