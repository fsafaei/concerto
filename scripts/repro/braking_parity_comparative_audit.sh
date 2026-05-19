#!/usr/bin/env bash
# CBF-only vs CBF+braking comparative audit (P1.04.6; ADR-007 §Stage 1b Rev 8).
#
# Runs a 200-step AS-homo Stage-1b training cell twice — once with the
# safety stack at the P1.04.5 vintage (cfg.safety.tau_brake = 1e-9, so
# maybe_brake never fires; CBF-only) and once with the production
# vintage (cfg.safety.tau_brake = 0.100; CBF + braking). Reads the
# safety_telemetry_final event from each run's JSONL and prints the
# λ-steady-state diff so the founder can quote it in the PR
# description without manual jq surgery.
#
# This is the "comparative audit" the ADR-007 Rev 7 P1.04.6 bullet
# commits to: "the diff is diagnostic for whether the braking-fallback
# changes the adaptation equilibrium meaningfully". A small or zero
# diff means braking-fallback fires rarely on the AS-homo geometry +
# the conformal-slack overlay already absorbs the dynamics-mismatch
# error without needing the braking backstop empirically. A large
# diff means the two layers are interacting; surface back before
# Stage-1b launch.
#
# Pre-conditions:
#   - GPU host with SAPIEN + Vulkan (`chamber.utils.device.sapien_gpu_available()`).
#   - `uv sync --group train` has been run.
#
# Wall-time: ~2-4 min on RTX 2080 for the 200-step pair.
#
# Usage:
#   bash scripts/repro/braking_parity_comparative_audit.sh
#
# Exit codes:
#   0  — both runs completed; comparative summary printed.
#   2  — usage / pre-condition failure.
#   3  — JSONL malformed or final event missing on either side.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

OUT_DIR="${OUT_DIR:-spikes/results/_braking_parity_comparative_audit_$(date -u +%Y%m%d_%H%M%S)}"
mkdir -p "${OUT_DIR}"

echo "==> Braking-parity comparative audit (P1.04.6; ADR-007 §Stage 1b Rev 8)"
echo "    output dir: ${OUT_DIR}"
echo ""

# Run A — CBF-only (tau_brake = 1e-9, below any synthetic TTC; braking
# never fires). Uses the existing tests/integration/test_stage1_safety_in_training.py
# entry point as a programmatic driver — same env (AS-homo,
# panda + panda_partner), same SafetyAggregator wiring, but reading
# tau_brake from an env override so we can flip the layer on/off
# without editing the yaml.
echo "==> Run A (CBF-only; tau_brake=1e-9)"
LOG_A="${OUT_DIR}/run_a_cbf_only.jsonl"
CONCERTO_BRAKING_AUDIT_TAU_BRAKE=1e-9 \
CONCERTO_BRAKING_AUDIT_JSONL="${LOG_A}" \
    uv run python scripts/repro/_braking_parity_audit_driver.py

# Run B — CBF + braking (tau_brake = 0.100; production vintage).
echo ""
echo "==> Run B (CBF + braking; tau_brake=0.100)"
LOG_B="${OUT_DIR}/run_b_cbf_plus_braking.jsonl"
CONCERTO_BRAKING_AUDIT_TAU_BRAKE=0.100 \
CONCERTO_BRAKING_AUDIT_JSONL="${LOG_B}" \
    uv run python scripts/repro/_braking_parity_audit_driver.py

# Read both final events and emit a side-by-side diff.
echo ""
echo "==> Comparative summary"

FINAL_A="$(jq -c 'select(.event == "safety_telemetry_final")' "${LOG_A}" | tail -n 1)"
FINAL_B="$(jq -c 'select(.event == "safety_telemetry_final")' "${LOG_B}" | tail -n 1)"
if [[ -z "${FINAL_A}" ]] || [[ -z "${FINAL_B}" ]]; then
    echo "FAIL: no safety_telemetry_final event in one or both runs." >&2
    echo "  Run A JSONL: ${LOG_A}" >&2
    echo "  Run B JSONL: ${LOG_B}" >&2
    exit 3
fi

SS_A="$(jq -r '.lambda_steady_state' <<<"${FINAL_A}")"
SS_B="$(jq -r '.lambda_steady_state' <<<"${FINAL_B}")"
NBR_A="$(jq -r '.n_braking_fires' <<<"${FINAL_A}")"
NBR_B="$(jq -r '.n_braking_fires' <<<"${FINAL_B}")"
RATE_A="$(jq -r '.braking_fire_rate' <<<"${FINAL_A}")"
RATE_B="$(jq -r '.braking_fire_rate' <<<"${FINAL_B}")"
LMEAN_A="$(jq -r '.lambda_mean' <<<"${FINAL_A}")"
LMEAN_B="$(jq -r '.lambda_mean' <<<"${FINAL_B}")"
LVAR_A="$(jq -r '.lambda_var' <<<"${FINAL_A}")"
LVAR_B="$(jq -r '.lambda_var' <<<"${FINAL_B}")"

# awk for the float diff (bash arithmetic is integer-only).
DIFF="$(awk -v a="${SS_A}" -v b="${SS_B}" 'BEGIN { printf "%.6f", b - a }')"

cat <<SUMMARY
                            Run A (CBF-only)    Run B (CBF + braking)
  n_braking_fires           ${NBR_A}                  ${NBR_B}
  braking_fire_rate         ${RATE_A}              ${RATE_B}
  lambda_steady_state       ${SS_A}              ${SS_B}
  lambda_mean               ${LMEAN_A}              ${LMEAN_B}
  lambda_var                ${LVAR_A}              ${LVAR_B}

  Δ lambda_steady_state (B - A): ${DIFF}

  Interpretation:
    - |Δ| < 0.05: braking fires rarely; the conformal overlay alone
      absorbs the dynamics-mismatch error. Safe to launch Stage-1b
      with the production tau_brake = 0.100.
    - 0.05 ≤ |Δ| < 0.5: the two layers interact moderately. Stage-1b
      results from CBF-only and CBF+braking would diverge but neither
      should saturate alone; surface in the PR description.
    - |Δ| ≥ 0.5 or Run A saturates: hard problem — the conformal
      overlay is over-loaded without the braking backstop. ADR-007
      §Open questions #8 (slice P1.05.5) may need to fire pre-Stage-2.
SUMMARY

echo ""
echo "==> PASS — comparative audit complete."
echo "    Run A JSONL: ${LOG_A}"
echo "    Run B JSONL: ${LOG_B}"
echo "    Quote the summary block above in the PR description."
