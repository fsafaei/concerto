#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# ADR-028 §Validation criteria 1 smoke: produce a v3 bundle with
# `chamber-eval run`, admit it with `chamber-eval verify`, then corrupt
# one byte of one episode file and assert `verify` rejects it.
# CPU-only; target < 5 minutes (measured: seconds).
#
# Usage:
#   bash scripts/smoke_eval.sh                # clean tree (CI)
#   SMOKE_EVAL_ALLOW_DIRTY=1 bash scripts/smoke_eval.sh   # local dev loop
#
# Exit codes: 0 pass; 1 any step failed (including "verify passed on a
# tampered bundle", the failure this smoke exists to catch).
set -euo pipefail

allow_dirty=()
if [ -n "${SMOKE_EVAL_ALLOW_DIRTY:-}" ]; then
  allow_dirty=(--allow-dirty)
fi

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

uv run chamber-eval run --task mpe_cooperative_push --policy random \
  --partner scripted_heuristic --seeds 2 --episodes 5 \
  --out "$tmp/bundle" "${allow_dirty[@]}"

uv run chamber-eval verify "$tmp/bundle" "${allow_dirty[@]}"

uv run python - "$tmp/bundle/episodes_seed0.jsonl" <<'PY'
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
data = bytearray(path.read_bytes())
data[10] ^= 0xFF
path.write_bytes(bytes(data))
print(f"smoke-eval: flipped one byte of {path.name}")
PY

if uv run chamber-eval verify "$tmp/bundle" "${allow_dirty[@]}"; then
  echo "smoke-eval: FAIL — verify passed on a tampered bundle" >&2
  exit 1
fi

echo "smoke-eval: PASS (fresh bundle verifies; tampered bundle rejected)"
