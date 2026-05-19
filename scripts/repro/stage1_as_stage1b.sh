#!/usr/bin/env bash
# Stage-1b AS axis spike (ADR-007 §Stage 1b; P1.04.5 + P1.05).
#
# This script is the host of the audit-gate hook (predicates A + B
# from the P1.04.5 design pass). The actual Stage-1b run invocation
# (chamber-spike run --axis AS --sub-stage 1b ...) is shipped in P1.05;
# this P1.04.5 vintage ships only the audit-gate machinery so the
# predicate can be reviewed independently from the dispatch swap.
#
# Audit-gate predicates (ADR-007 §Stage 1b implementation-details Rev 7):
#
#   Predicate A (saturation guard, always evaluated):
#     λ_steady_state < 0.9 × cartesian_accel_capacity   → exit 0
#     λ_steady_state >= 0.9 × cartesian_accel_capacity  → exit 8
#
#   Predicate B (adaptation invariant, conditional on λ_mean > 1e-6):
#     λ adapted away from 0 AND λ_var > 1e-12           → exit 0
#     λ adapted away from 0 AND λ_var <= 1e-12          → exit 9 (λ stuck)
#     λ stayed at 0 (vacuously OK)                      → exit 0
#
# Predicate B captures the underlying invariant rather than codifying
# condition-specific expectations — capturing "if λ moved, it should
# have varied" is robust to per-cell asymmetries (AS-homo will likely
# fire the filter substantively; AS-hetero / OM-* may stay at λ ≈ 0
# vacuously). The cell-aware variant would have been the same class of
# foot-gun ADR-016 §Decision closed via the typed sub_stage field.
#
# Safety-disabled path:
#   When the JSONL summary records safety_enabled=false (operator
#   override via `cfg.safety.enabled=false`), both predicates are
#   skipped and the gate emits a non-failing "safety disabled by
#   operator override; gate skipped" message. Exit 0.
#
# Usage:
#   bash scripts/repro/stage1_as_stage1b.sh <path/to/run.jsonl>
#
# Exit codes:
#   0  — gate passed (or safety disabled).
#   8  — λ saturated against cartesian_accel_capacity (predicate A).
#   9  — λ adapted but stuck (predicate B; λ_mean > ε but λ_var ≈ 0).
#   2  — usage error (missing argument or non-existent JSONL).
#   3  — JSONL malformed (no safety_telemetry_final event found).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

JSONL_PATH="${1:-}"
if [[ -z "${JSONL_PATH}" ]]; then
    echo "Usage: bash scripts/repro/stage1_as_stage1b.sh <path/to/run.jsonl>" >&2
    exit 2
fi
if [[ ! -f "${JSONL_PATH}" ]]; then
    echo "stage1_as_stage1b: JSONL not found: ${JSONL_PATH}" >&2
    exit 2
fi

# Read the final-summary event (last line that matches event="safety_telemetry_final").
# The training loop emits exactly one such event per cell at end-of-training.
FINAL_SUMMARY="$(jq -c 'select(.event == "safety_telemetry_final")' "${JSONL_PATH}" | tail -n 1)"
if [[ -z "${FINAL_SUMMARY}" ]]; then
    echo "stage1_as_stage1b: no safety_telemetry_final event in ${JSONL_PATH}" >&2
    echo "  (cfg.safety.enabled may have been false during the run, OR the run" >&2
    echo "   crashed before completion. Inspect the JSONL for training_end events" >&2
    echo "   and re-run if necessary.)" >&2
    exit 3
fi

# Safety-disabled path: emit non-failing message and exit 0.
SAFETY_ENABLED="$(jq -r '.safety_enabled' <<<"${FINAL_SUMMARY}")"
if [[ "${SAFETY_ENABLED}" == "false" ]]; then
    echo "stage1_as_stage1b: safety disabled by operator override (cfg.safety.enabled=false); gate skipped."
    exit 0
fi

# Extract predicate inputs.
LAMBDA_STEADY_STATE="$(jq -r '.lambda_steady_state' <<<"${FINAL_SUMMARY}")"
LAMBDA_MEAN="$(jq -r '.lambda_mean' <<<"${FINAL_SUMMARY}")"
LAMBDA_VAR="$(jq -r '.lambda_var' <<<"${FINAL_SUMMARY}")"
CAPACITY="$(jq -r '.cartesian_accel_capacity' <<<"${FINAL_SUMMARY}")"
SATURATION_THRESHOLD="$(jq -r '.saturation_threshold' <<<"${FINAL_SUMMARY}")"

# Predicate A: λ_steady_state < saturation_threshold * cartesian_accel_capacity?
# Use awk for the float comparison (bash arithmetic is integer-only; jq
# arithmetic compiles but its boolean → bash translation is awkward).
THRESHOLD_VALUE="$(awk -v t="${SATURATION_THRESHOLD}" -v c="${CAPACITY}" 'BEGIN { printf "%.6f", t * c }')"
if awk -v l="${LAMBDA_STEADY_STATE}" -v t="${THRESHOLD_VALUE}" 'BEGIN { exit !(l >= t) }'; then
    echo "stage1_as_stage1b: FAIL predicate A (saturation guard)" >&2
    echo "  λ_steady_state=${LAMBDA_STEADY_STATE} >= threshold=${THRESHOLD_VALUE}" >&2
    echo "  (saturation_threshold=${SATURATION_THRESHOLD} × cartesian_accel_capacity=${CAPACITY})" >&2
    echo "  λ saturated against the cell's bounds; the QP is running" >&2
    echo "  without safety margin. ADR-007 §Stage 1b: saturating cells" >&2
    echo "  force slice P1.05.5 (AgentSnapshot.qpos extension) to launch" >&2
    echo "  mandatorily before Stage 2." >&2
    exit 8
fi

# Predicate B: adaptation-conditional. Only fires when λ moved away
# from zero (λ_mean > 1e-6). The vacuous case (λ stayed at 0) passes.
if awk -v m="${LAMBDA_MEAN}" 'BEGIN { exit !(m > 1e-6) }'; then
    # λ adapted away from zero; assert it also varied.
    if awk -v v="${LAMBDA_VAR}" 'BEGIN { exit !(v <= 1e-12) }'; then
        echo "stage1_as_stage1b: FAIL predicate B (λ adapted but stuck)" >&2
        echo "  λ_mean=${LAMBDA_MEAN} > 1e-6 (adapted away from 0) but" >&2
        echo "  λ_var=${LAMBDA_VAR} <= 1e-12 (didn't vary). The conformal" >&2
        echo "  slack overlay converged to a non-zero constant — the" >&2
        echo "  filter is firing but the learning signal is degenerate." >&2
        echo "  Likely cause: predictor or env state-noise too low to" >&2
        echo "  drive update_lambda_from_predictor's loss vector." >&2
        exit 9
    fi
fi

echo "stage1_as_stage1b: PASS audit gate (predicates A + B)."
echo "  λ_steady_state=${LAMBDA_STEADY_STATE} (threshold=${THRESHOLD_VALUE})"
echo "  λ_mean=${LAMBDA_MEAN}, λ_var=${LAMBDA_VAR}"
exit 0
