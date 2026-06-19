# SPDX-License-Identifier: Apache-2.0
"""Driver for the Rung-3 capability-calibration gate (ADR-026 §Decision 4; R-2026-06-B §15 Rung 3).

Invoked by ``scripts/repro/cocarry_rung3_calibrate.sh``. Pairs each
policy-shift candidate (and the matched ``cocarry_impedance`` teammate, to
measure the calibrated reference score M) with the cooperative-reference ego
over the pre-registered calibration seeds, gates each against ``C_MIN``, and
writes the calibration roster (admitted + excluded) as the artifact of record.
Exits 2 if fewer than 3 candidates clear the gate (the Rung-3 STOP criterion).
"""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime

from chamber.benchmarks import cocarry_ph as ph
from chamber.partners.cocarry_policy_shift import COCARRY_POLICY_SHIFT_CANDIDATES

_RENDER_BACKEND = "none"
_MIN_ADMITTED = 3
_EPS = 1e-9


def _calibrate(class_name: str) -> dict:
    seeds = list(ph.CALIBRATION_SEEDS)
    metrics = ph.evaluate_calibration(
        candidate_class=class_name, seeds=seeds, render_backend=_RENDER_BACKEND
    )
    rate = ph.success_rate({m.seed: m.success for m in metrics})
    summary = ph.conjunct_failure_summary(metrics)
    return {
        "class_name": class_name,
        "calibration_success_rate": rate,
        "passes_gate": ph.passes_capability_gate(rate),
        "conjunct_summary": summary,
        "per_seed": [
            {
                "seed": m.seed,
                "success": m.success,
                "is_placed": m.is_placed,
                "is_level": m.is_level,
                "is_unstressed": m.is_unstressed,
                "both_static": m.both_static,
                "centroid_to_goal": round(m.centroid_to_goal, 4),
                "max_tilt_deg": round(m.max_tilt_deg, 3),
                "max_stress_proxy": round(m.max_stress_proxy, 1),
            }
            for m in metrics
        ],
    }


def main() -> int:
    out_path = os.environ.get(
        "OUT_JSON", "spikes/results/cocarry/rung3/cocarry_rung3_calibration_roster.json"
    )
    print(
        f"    Cooperative reference ego = {ph.COOPERATIVE_REFERENCE_EGO_CLASS} (NOT the incumbent)"
    )
    print(f"    C_min = {ph.C_MIN} over calibration seeds {list(ph.CALIBRATION_SEEDS)}")

    # The matched teammate (cocarry_impedance on the partner seat) calibrated
    # against the cooperative ego = the matched pair = the calibrated reference
    # score M. Measured first so C_min = max(0.75, M - 0.25) is auditable.
    print(f"    [matched] calibrating {ph.MATCHED_PARTNER_CLASS} (the reference teammate) ...")
    matched_row = _calibrate(ph.MATCHED_PARTNER_CLASS)
    m_score = matched_row["calibration_success_rate"]
    print(f"      matched teammate calibrated score M = {m_score:.0%}")

    candidates = []
    for cls in COCARRY_POLICY_SHIFT_CANDIDATES:
        print(f"    [candidate] calibrating {cls} ...")
        row = _calibrate(cls)
        cs = row["conjunct_summary"]
        print(
            f"      {cls}: success={row['calibration_success_rate']:.0%} "
            f"gate={'PASS' if row['passes_gate'] else 'EXCLUDE'}  "
            f"(fail placed/level/stress/static = "
            f"{cs.get('fail_placed')}/{cs.get('fail_level')}/"
            f"{cs.get('fail_unstressed')}/{cs.get('fail_static')})"
        )
        candidates.append(row)

    admitted = [c["class_name"] for c in candidates if c["passes_gate"]]
    excluded = [c["class_name"] for c in candidates if not c["passes_gate"]]

    roster = {
        "schema": "cocarry_rung3_calibration_roster/v1",
        "date": datetime.now(UTC).strftime("%Y-%m-%d"),
        "purpose": (
            "Capability-calibration gate for the Rung-3 policy-heterogeneity (PH) "
            "measurement. Each policy-shift candidate is paired with a cooperative "
            "reference ego (the matched impedance, NOT the frozen incumbent) and must "
            "clear C_min to enter the test. Defuses the weaker-teammate confound; the "
            "excluded roster is itself a finding (ADR-026 §Decision 4; R-2026-06-B §15)."
        ),
        "cooperative_reference_ego": ph.COOPERATIVE_REFERENCE_EGO_CLASS,
        "matched_partner": ph.MATCHED_PARTNER_CLASS,
        "c_min": ph.C_MIN,
        "c_min_derivation": ph.C_MIN_DERIVATION,
        "calibration_band": list(ph.CALIBRATION_BAND),
        "calibration_seeds": list(ph.CALIBRATION_SEEDS),
        "matched_teammate_calibrated_score_M": m_score,
        "c_min_check_against_M": {
            "rule": "C_min = max(0.75, M - 0.25)",
            "M": m_score,
            "M_minus_0p25": round(m_score - 0.25, 4),
            "binding_c_min": max(0.75, round(m_score - 0.25, 4)),
            "consistent": abs(max(0.75, m_score - 0.25) - ph.C_MIN) < _EPS,
        },
        "matched_teammate_calibration": matched_row,
        "candidates": candidates,
        "admitted": admitted,
        "excluded": excluded,
        "n_admitted": len(admitted),
        "min_admitted_required": _MIN_ADMITTED,
        "stop_triggered": len(admitted) < _MIN_ADMITTED,
        "notes": [
            "Calibration is the candidate paired with the cooperative reference ego "
            "(matched cocarry_impedance), so the score isolates the candidate's task "
            "competence, NOT how the frozen incumbent reacts to it.",
            "EXCLUSIONS BIAS Δ TOWARD ZERO: the gate truncates heterogeneity from "
            "above. If any candidate is excluded, a downstream Δ≈0 may NOT be read as "
            "'PH does not couple to cooperation' (R-2026-06-B §15 Rung 3).",
            "C_min is fixed at pre-statement and grounded on the matched reference; it "
            "is NEVER weakened to pass teammates. Fewer than 3 admitted ⇒ STOP.",
        ],
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(roster, fh, sort_keys=True, indent=2)
    print(f"      admitted={admitted}")
    print(f"      excluded={excluded}")
    print(f"      roster -> {out_path}")
    return 0 if len(admitted) >= _MIN_ADMITTED else 2


if __name__ == "__main__":
    sys.exit(main())
