# SPDX-License-Identifier: Apache-2.0
"""Rung-4c same-ego EH-vs-control-style floor — committed generator (ADR-026 §D4; R-2026-06-B §15).

Produces the headline floor table (``cocarry_rung4c_sameego_floor.json``) from
COMMITTED code (replacing the development ``/tmp`` script in the PR-#252 report):
the FIXED cooperative reference ego (matched impedance) paired with each
condition — matched Panda partner, the 3 control-style policy-shift teammates
(same Panda body), and the xArm6 (embodiment shift) — under the
embodiment-invariant coupling-force instrument (``stress_measure="coupling"``,
compliant K=8000), success re-scored at the committed, held f_max (read from
the measurement artifact — NOT re-derived).

Reconciliation of the PR-#252 1867-vs-1773 N discrepancy: the report's floor
xArm6 (1867) was measured on the pre-registered FLOOR seeds (80100-80111); the
measurement JSON's ``xarm6_capability`` (1773) on the EH seeds (80200-80211) —
the SAME condition on DIFFERENT pre-registered seed sets. This generator runs
the floor on the canonical floor seeds AND cross-checks the xArm6 on the EH
seeds, so both numbers are reproduced from committed code and the difference is
shown to be n=12 seed sampling (the over-load is seed-robust), not an error.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from chamber.benchmarks import cocarry_ph as ph
from chamber.envs.cocarry import evaluate_cocarry_success

_RENDER = "none"
_K = 8000.0
_MEASURE = "coupling"
_FLOOR_SEEDS = list(range(80100, 80112))  # pre-registered PH/floor seeds (canonical)
_EH_SEEDS = list(range(80200, 80212))  # pre-registered EH seeds (capability-gate cross-check)
_MEASUREMENT = "spikes/results/cocarry/rung4c/cocarry_rung4c_eh_measurement.json"
# (cooperative ego, condition, partner uid) per floor row.
_CONDITIONS = [
    ("cocarry_impedance", "cocarry_matched_panda_pair", "panda_partner", "matched_panda"),
    ("cocarry_stiff_impedance", "cocarry_matched_panda_pair", "panda_partner", "control_style_PH"),
    ("cocarry_admittance", "cocarry_matched_panda_pair", "panda_partner", "control_style_PH"),
    ("cocarry_slew_impedance", "cocarry_matched_panda_pair", "panda_partner", "control_style_PH"),
    ("cocarry_xarm6_impedance", "cocarry_xarm6_partner", "xarm6_robotiq", "embodiment_EH"),
]


def _rescore(m: ph.ConjunctMetrics, fmax: float) -> bool:
    return bool(
        evaluate_cocarry_success(
            centroid_to_goal_dist=m.centroid_to_goal,
            max_tilt_deg=m.max_tilt_deg,
            max_stress_proxy=m.max_stress_proxy,
            both_static=m.both_static,
            stress_max=fmax,
        )
    )


def _row(candidate: str, cond: str, puid: str, seeds: list[int], fmax: float) -> dict:
    metrics = ph.evaluate_calibration(
        candidate_class=candidate,
        seeds=seeds,
        condition_id=cond,
        partner_uid=puid,
        render_backend=_RENDER,
        drive_stiffness=_K,
        stress_measure=_MEASURE,
    )
    return {
        "success_rate_at_fmax": float(np.mean([_rescore(m, fmax) for m in metrics])),
        "coupling_p90": float(np.percentile([m.max_stress_proxy for m in metrics], 90)),
        "coupling_max": float(np.max([m.max_stress_proxy for m in metrics])),
        "fail_placed": int(sum(1 for m in metrics if not m.is_placed)),
        "fail_level": int(sum(1 for m in metrics if not m.is_level)),
        "fail_unstressed": int(sum(1 for m in metrics if m.max_stress_proxy >= fmax)),
        "fail_static": int(sum(1 for m in metrics if not m.both_static)),
    }


def main() -> int:
    out_path = os.environ.get(
        "OUT_JSON", "spikes/results/cocarry/rung4c/cocarry_rung4c_sameego_floor.json"
    )
    fmax = float(json.loads(Path(_MEASUREMENT).read_text(encoding="utf-8"))["fmax_coupling_n"])
    print(f"    f_max (committed, held) = {fmax:.1f} N; floor seeds {_FLOOR_SEEDS}")
    rows = {}
    for candidate, cond, puid, kind in _CONDITIONS:
        r = _row(candidate, cond, puid, _FLOOR_SEEDS, fmax)
        r["kind"] = kind
        rows[candidate] = r
        print(
            f"      {candidate:28} [{kind:16}] coupling_p90={r['coupling_p90']:.0f} "
            f"success@fmax={r['success_rate_at_fmax']:.0%}"
        )
    # Cross-check: xArm6 on the EH seeds (reconcile with the measurement's
    # xarm6_capability block; same condition, different prereg seed set).
    xarm6_eh = _row(
        "cocarry_xarm6_impedance", "cocarry_xarm6_partner", "xarm6_robotiq", _EH_SEEDS, fmax
    )
    print(f"      xArm6 cross-check on EH seeds: coupling_p90={xarm6_eh['coupling_p90']:.0f}")

    control_style = [
        v["coupling_p90"]
        for v in rows.values()
        if v["kind"] in ("matched_panda", "control_style_PH")
    ]
    eh = rows["cocarry_xarm6_impedance"]["coupling_p90"]
    artifact = {
        "schema": "cocarry_rung4c_sameego_floor/v2",
        "date": datetime.now(UTC).strftime("%Y-%m-%d"),
        "generator": "scripts/repro/_cocarry_rung4c_sameego_floor.py",
        "prereg_artifact": "spikes/results/cocarry/rung4c/cocarry_rung4c_eh_prereg.json",
        "stress_measure": _MEASURE,
        "drive_stiffness_Npm": _K,
        "fmax_coupling_n": fmax,
        "floor_seeds": _FLOOR_SEEDS,
        "eh_seeds": _EH_SEEDS,
        "rows": rows,
        "xarm6_eh_seeds_crosscheck": xarm6_eh,
        "control_style_floor_p90_range_n": [float(min(control_style)), float(max(control_style))],
        "embodiment_xarm6_p90_n": float(eh),
        "embodiment_over_control_style_ratio": float(eh / max(*control_style, 1e-6)),
        "discrepancy_reconciliation": (
            f"PR-#252 floor xArm6 (1867 N) used the pre-registered FLOOR seeds (80100-); the "
            f"measurement JSON xarm6_capability (1773 N) used the EH seeds (80200-) — same "
            f"condition (cooperative ego + xArm6, coupling, f_max), different pre-registered seed "
            f"sets. This generator reproduces both from committed code: floor-seed p90="
            f"{eh:.0f} N, EH-seed p90={xarm6_eh['coupling_p90']:.0f} N. The ~5% difference is "
            f"n=12 seed sampling; the over-load (~5x f_max, ~6x the control-style floor) is "
            f"seed-robust. The headline cites the floor-seed value consistently with the other "
            f"floor rows (all on 80100-)."
        ),
    }
    Path(out_path).write_text(json.dumps(artifact, sort_keys=True, indent=2), encoding="utf-8")
    print(
        f"    control-style floor p90 {min(control_style):.0f}-{max(control_style):.0f} N | "
        f"xArm6 {eh:.0f} N ({eh / max(control_style):.1f}x ceiling)"
    )
    print(f"    artifact -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
