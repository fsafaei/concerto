#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Phase-2 co-carry decouple ablation (ADR-026 §Decision 1-2 / §Open-questions; R-2026-06-C
# condition 5). Eval-only, no training: the SAME matched base pair, coupling swept DOWN to
# zero (C3, endpoints C0/C1) plus a zero-action-partner arm (C2), to test whether base
# success at K=8000 is cooperation-contingent or coupling-trivial.
# Pre-registration: spikes/preregistration/cocarry_decouple_ablation_prereg.json
#   (tag prereg-cocarry-decouple-ablation-2026-07-11).
# The driver REFUSES a dirty tree (exit 7) or a prereg blob-SHA mismatch vs the tag (exit 4).
set -euo pipefail
cd "$(dirname "$0")/../.."
exec uv run python scripts/repro/_cocarry_decouple_ablation.py "$@"
