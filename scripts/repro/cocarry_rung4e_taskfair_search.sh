#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Rung-4e - the task-fair embodiment search (ADR-026 §Decision 2/4; R-2026-06-B §15 Rung 4e).
# Reproduces the pose x controller search against the full joint-success criterion, the
# stress<->tilt Pareto frontier, and the pre-committed falsifiable-both-ways verdict.
# Pre-registration: spikes/results/cocarry/rung4e/cocarry_rung4e_taskfair_prereg.json
#   (tag prereg-cocarry-rung4e-taskfair-2026-06-20).
set -euo pipefail
cd "$(dirname "$0")/../.."
exec uv run python scripts/repro/_cocarry_rung4e_taskfair_search.py
