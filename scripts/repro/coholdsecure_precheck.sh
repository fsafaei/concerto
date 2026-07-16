#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Co-hold-secure PR-A engineering precheck (ADR-029 §Decision; pre-stated,
# NON-REGISTERED, non-gating — no prereg tag exists or is rotated here; the
# founder-signed Gate-0 pre-registration is PR-B). Rule-before-result: the
# decision rule is committed in
# spikes/results/coholdsecure/precheck-2026-07-16/PRECHECK_PRESTATEMENT.md
# BEFORE any measured episode. Verdict: PROCEED iff P1 ∧ P2 ∧ P4.
# Deterministic (P6/ADR-002): fixed seeds; rewrites precheck_results.json
# byte-identically. Requires a SAPIEN GPU/Vulkan host (see REPRO.txt).
set -euo pipefail
cd "$(dirname "$0")/../.."
exec uv run python scripts/repro/_coholdsecure_precheck.py "$@"
