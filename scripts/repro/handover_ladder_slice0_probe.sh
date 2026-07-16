#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Handover ladder Slice-0 oracle-headroom probe (eval-only, CPU-only, NON-gating; I1).
# Executes the pre-stated rule of
#   spikes/results/handover-ladder-probe-2026-07-16/PROBE_PRESTATEMENT.md
#   (tag probe-handover-ladder-slice0-2026-07-16; rule committed before any run).
# Recomputes REF against the SHA-verified committed bundle
#   spikes/results/benchmark/handover-v1/ref-script-2026-07-06/ (hard reconciliation
#   gate: a mismatch aborts with no verdict), then searches the ego action space
#   per state for the achievable ceiling and the three fixed-policy comparators.
set -euo pipefail
cd "$(dirname "$0")/../.."
exec uv run python scripts/repro/_handover_ladder_slice0_probe.py "$@"
