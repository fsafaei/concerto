#!/usr/bin/env bash
# Co-carry Rung-2 f_max re-derivation + matched-reference measurement
# (ADR-026 §Decision 4 + §Validation criteria; R-2026-06-B §15 Rung 2).
#
# Step-1 of the Rung-2 build order: re-run the matched constraint-force
# distribution at the pre-registered seed count, COMMIT it as the artifact
# of record, derive f_max = 1.25 x matched-success p99, and CLASSIFY it
# against the pre-stated consistency bands (fixed in
# chamber.benchmarks.cocarry_freeze, before any measurement — no forking
# path). Also records the matched-pair reference success rate (the Rung-2
# training target) and re-scores the single-arm positive-control under the
# (possibly new) f_max so the coupling check still holds.
#
# PR #245 set COCARRY_STRESS_MAX_PROXY_N = 130 N from a 15-seed run recorded
# only in a docstring. This re-derivation is the authority going forward
# (if consistent); a MATERIAL DIVERGENCE is a STOP-and-report condition.
#
# Needs a Vulkan/GPU host (SAPIEN); skipped on a CPU-only host.
#
# Usage:
#   bash scripts/repro/cocarry_rung2_fmax_reference.sh [N_SEEDS]   # default 12
#
# Writes: spikes/results/cocarry/rung2/cocarry_rung2_fmax_distribution.json
#
# Exit codes:
#   0 — measurement complete; f_max CONSISTENT with the pre-stated bands.
#   2 — f_max MATERIALLY DIVERGENT: STOP, report both distributions, freeze
#       nothing until the cause (determinism / seed-set / rig change) is
#       understood (R-2026-06-B §15).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

N_SEEDS="${1:-12}"
OUT_DIR="spikes/results/cocarry/rung2"
OUT_JSON="${OUT_DIR}/cocarry_rung2_fmax_distribution.json"
mkdir -p "${OUT_DIR}"

echo "==> Co-carry Rung-2 f_max re-derivation + matched reference (ADR-026 §Decision 4)"
echo "    $(uv run python -c 'from chamber.utils.device import device_report; print(device_report())' 2>/dev/null)"

if ! uv run python -c "from chamber.utils.device import sapien_gpu_available; import sys; sys.exit(0 if sapien_gpu_available() else 1)" 2>/dev/null; then
    echo "==> ERROR — sapien_gpu_available() is False; this measurement needs a Vulkan/GPU host. Aborting."
    exit 1
fi

echo "    Matched + single-arm sweep (N=${N_SEEDS} seed-clusters; ADR-026 §Validation criteria)"
N_SEEDS="${N_SEEDS}" OUT_JSON="${OUT_JSON}" uv run python - <<'PY'
import json
import os
import numpy as np
from chamber.benchmarks.cocarry_runner import evaluate_condition, summarize
from chamber.benchmarks import cocarry_freeze as F
from chamber.envs.cocarry import (
    COCARRY_STRESS_MAX_PROXY_N,
    COCARRY_TILT_MAX_DEG,
    evaluate_cocarry_success,
)

n = int(os.environ["N_SEEDS"])
seeds = list(range(n))
matched = evaluate_condition(condition_id="cocarry_matched_panda_pair", seeds=seeds)
single = evaluate_condition(condition_id="cocarry_single_arm_positive_control", seeds=seeds)
ms, ss = summarize(matched), summarize(single)

# f_max is derived from the matched-SUCCESS distribution p99 (the documented
# rule). summarize() reports success_stress_p95; for the p99 we recompute it
# from the successful matched episodes directly.
succ_stress = np.asarray([m.max_stress_proxy for m in matched if m.success])
matched_success_p99 = float(np.percentile(succ_stress, 99)) if succ_stress.size else float("nan")
fmax = F.derive_fmax_from_p99(matched_success_p99)
status = F.classify_fmax(fmax_value=fmax, matched_success_stress_p99=matched_success_p99)

# Re-score the single-arm positive-control + matched competence under the
# (possibly new) f_max: matched successes only change via the stress conjunct
# (tilt/centroid/static unchanged), so recompute the stress gate on each
# episode. The coupling is carried by tilt, so single-arm stays ~0.
def rescore(metrics, fmax_n):
    ok = 0
    for m in metrics:
        s = evaluate_cocarry_success(
            centroid_to_goal_dist=m.centroid_to_goal,
            max_tilt_deg=m.max_tilt_deg,
            max_stress_proxy=m.max_stress_proxy,
            both_static=bool(m.success) or (m.max_tilt_deg < COCARRY_TILT_MAX_DEG),
            stress_max=fmax_n,
        )
        ok += int(bool(s))
    return ok / max(1, len(metrics))

matched_rate_new = rescore(matched, fmax)
single_rate_new = rescore(single, fmax)

artifact = {
    "schema": "cocarry_rung2_fmax_distribution/v1",
    "n_seed_clusters": n,
    "seeds": seeds,
    "matched_summary": ms,
    "single_arm_summary": ss,
    "matched_success_stress_proxy_n": [float(x) for x in succ_stress],
    "matched_success_stress_p99_n": matched_success_p99,
    "fmax_derived_n": fmax,
    "fmax_multiplier": F.FMAX_P99_MULTIPLIER,
    "fmax_consistency_band_n": list(F.FMAX_CONSISTENCY_BAND_N),
    "matched_success_p99_band_n": list(F.MATCHED_SUCCESS_P99_BAND_N),
    "fmax_status": status,
    "provisional_fmax_n": COCARRY_STRESS_MAX_PROXY_N,
    "rescore_under_new_fmax": {
        "matched_success_rate": matched_rate_new,
        "single_arm_success_rate": single_rate_new,
    },
}
out = os.environ["OUT_JSON"]
with open(out, "w", encoding="utf-8") as fh:
    json.dump(artifact, fh, sort_keys=True, indent=2)

print(f"      matched  success={ms['success_rate']:.0%}  tilt p50={ms['max_tilt_p50']:.1f} deg")
print(f"      single   success={ss['success_rate']:.0%}  tilt p50={ss['max_tilt_p50']:.1f} deg")
print(f"      matched-success stress p99={matched_success_p99:.1f} N "
      f"(band {F.MATCHED_SUCCESS_P99_BAND_N})")
print(f"      derived f_max = 1.25 x p99 = {fmax:.1f} N "
      f"(band {F.FMAX_CONSISTENCY_BAND_N}; provisional was {COCARRY_STRESS_MAX_PROXY_N:.0f} N)")
print(f"      re-score under new f_max: matched={matched_rate_new:.0%}  single={single_rate_new:.0%}")
print(f"      VERDICT: f_max {status.upper()}")
print(f"      artifact -> {out}")
import sys
sys.exit(0 if status == F.FMAX_CONSISTENT else 2)
PY
RC=$?

echo ""
if [ "${RC}" -eq 0 ]; then
    echo "==> CONSISTENT — freeze the re-derived f_max (commit ${OUT_JSON}); 130 N was provisional."
    echo "    Next: bash scripts/repro/cocarry_rung2_train.sh"
    exit 0
else
    echo "==> MATERIAL DIVERGENCE — STOP (R-2026-06-B §15). Report both distributions"
    echo "    side-by-side (this run vs the #245 15-seed p50=90/p99=104/max=105 N) with a"
    echo "    determinism / seed-set / rig diagnosis. Freeze nothing until understood."
    exit 2
fi
