#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Phase-2 co-carry base-difficulty probe (ADR-026 §Decision 1-2 / §Open-questions; R-2026-06-C).
# Eval-only, no training: sweeps task difficulty (coupling stiffness + tightened goal-radius/tilt)
# on the competent MATCHED pair to find whether a graceful hard-but-feasible regime exists, or
# whether stiffness only ever cliffs into numerical over-coupling (the Rung-4 wall).
# Pre-registration: spikes/results/cocarry/base_probe/cocarry_base_probe_prereg.json
#   (tag prereg-cocarry-base-probe-2026-06-22).
set -euo pipefail
cd "$(dirname "$0")/../.."
exec uv run python scripts/repro/_cocarry_base_probe.py
