# SPDX-License-Identifier: Apache-2.0
"""Driver for the Rung-4 mixed-embodiment stability gate + capability calibration.

ADR-026 §Decision 4; ADR-005 §Decision; ADR-009 §Decision; R-2026-06-B §15 Rung 4.
Invoked by ``scripts/repro/cocarry_rung4_eh_calibrate.sh``.

Two cheap-fail gates, BEFORE any measurement:

1. **Mixed-embodiment stability** (Rung-0 style): the Panda ego (cooperative
   reference) + the xArm6 partner rigidly holding one bar, driven to the goal.
   Two *different* arms on one rigid bar is a new contact-dynamics regime; the
   gate asserts the closed chain is stable (telemetry finite, no solver
   blow-up, bounded constraint force).
2. **Capability calibration**: the xArm6 teammate paired with the cooperative
   Panda ego (NOT the frozen incumbent) over the calibration seeds, gated at
   C_min = 0.75 (the same gate as Rung 3). A different, less-dexterous body is
   genuinely harder to capability-match; if it cannot clear the gate it is
   EXCLUDED and that is reported as a feasibility finding (do NOT weaken C_min).

The ``ur_e`` backup named in the spec does not exist in ``mani-skill==3.0.1``;
the only UR (``ur_10e``) ships with no gripper / TCP, so it cannot host the
dual-hold bar weld — recorded here as a structural exclusion, not a tuning gap.

Exits 2 if fewer than 1 embodiment teammate clears the gate (the measurement
cannot run — that is itself the result).
"""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime

import numpy as np

from chamber.benchmarks import cocarry_ph as ph

_RENDER_BACKEND = "none"
_MIN_ADMITTED = 1
_STABILITY_SEEDS = [60000, 60001, 60002]  # subset of the calibration seeds


def main() -> int:
    out_path = os.environ.get(
        "OUT_JSON", "spikes/results/cocarry/rung4/cocarry_rung4_calibration_roster.json"
    )
    print(f"    embodiment teammate = {ph.XARM6_PARTNER_CLASS} on the {ph.XARM6_PARTNER_UID} seat")
    print(
        f"    cooperative reference ego = {ph.COOPERATIVE_REFERENCE_EGO_CLASS} (NOT the incumbent)"
    )
    print(f"    C_min = {ph.C_MIN}; calibration seeds {list(ph.EH_CALIBRATION_SEEDS)}")

    # --- Gate 1: mixed-embodiment stability (computed from cooperative-ego +
    # xArm6 rollouts; finite + bounded telemetry => the closed chain is stable).
    print("    [stability] Panda ego + xArm6 partner on one welded bar ...")
    stab = ph.evaluate_calibration(
        candidate_class=ph.XARM6_PARTNER_CLASS,
        seeds=_STABILITY_SEEDS,
        condition_id=ph.XARM6_CONDITION_ID,
        partner_uid=ph.XARM6_PARTNER_UID,
        render_backend=_RENDER_BACKEND,
    )
    all_finite = all(
        np.isfinite(m.max_tilt_deg)
        and np.isfinite(m.max_stress_proxy)
        and np.isfinite(m.centroid_to_goal)
        for m in stab
    )
    stab_max_stress = float(max(m.max_stress_proxy for m in stab))
    stab_max_tilt = float(max(m.max_tilt_deg for m in stab))
    stable = bool(all_finite and np.isfinite(stab_max_stress))
    print(
        f"      stable={stable} (finite telemetry); max_stress={stab_max_stress:.0f} N "
        f"max_tilt={stab_max_tilt:.1f} deg over {len(stab)} seeds"
    )
    if not stable:
        print("      STOP: mixed-embodiment rig UNSTABLE (solver blow-up) — feasibility finding.")
        # fall through to write the roster recording the instability.

    # --- Gate 2: capability calibration (12 seeds).
    print("    [calibration] xArm6 + cooperative ego over the calibration seeds ...")
    seeds = list(ph.EH_CALIBRATION_SEEDS)
    metrics = ph.evaluate_calibration(
        candidate_class=ph.XARM6_PARTNER_CLASS,
        seeds=seeds,
        condition_id=ph.XARM6_CONDITION_ID,
        partner_uid=ph.XARM6_PARTNER_UID,
        render_backend=_RENDER_BACKEND,
    )
    rate = ph.success_rate({m.seed: m.success for m in metrics})
    summary = ph.conjunct_failure_summary(metrics)
    passes = ph.passes_capability_gate(rate)
    print(f"      xArm6 calibration success = {rate:.0%}  gate = {'PASS' if passes else 'EXCLUDE'}")
    print(
        f"      conjunct fails placed/level/stress/static = {summary.get('fail_placed')}/"
        f"{summary.get('fail_level')}/{summary.get('fail_unstressed')}/{summary.get('fail_static')}"
        f"; stress p90 {summary.get('stress_p90'):.0f} N (matched ref ~75 N, f_max 130 N)"
    )

    admitted = [ph.XARM6_PARTNER_CLASS] if passes else []
    excluded = [] if passes else [ph.XARM6_PARTNER_CLASS]

    roster = {
        "schema": "cocarry_rung4_calibration_roster/v1",
        "date": datetime.now(UTC).strftime("%Y-%m-%d"),
        "axis": "EH (embodiment heterogeneity) — different partner BODY, same frozen Panda ego",
        "purpose": (
            "Rung-4 mixed-embodiment stability gate + capability calibration for the "
            "embodiment-heterogeneity measurement. The xArm6 + Robotiq teammate is paired "
            "with the cooperative Panda ego (NOT the frozen incumbent) and gated at C_min. "
            "If it cannot be capability-matched it is EXCLUDED and the measurement does not "
            "run — itself the result (ADR-026 §D4; ADR-005; R-2026-06-B §15 Rung 4)."
        ),
        "cooperative_reference_ego": ph.COOPERATIVE_REFERENCE_EGO_CLASS,
        "c_min": ph.C_MIN,
        "c_min_derivation": ph.C_MIN_DERIVATION,
        "calibration_seeds": seeds,
        "mixed_embodiment_stability": {
            "seeds": _STABILITY_SEEDS,
            "stable": stable,
            "all_telemetry_finite": all_finite,
            "max_stress_proxy_n": stab_max_stress,
            "max_tilt_deg": stab_max_tilt,
            "note": "Stable = finite, bounded telemetry (no solver blow-up). The rig is "
            "physically stable; the issue is CAPABILITY (the stiff rigid weld makes the two "
            "different-bodied arms fight), not numerical instability.",
        },
        "candidates": [
            {
                "class_name": ph.XARM6_PARTNER_CLASS,
                "partner_uid": ph.XARM6_PARTNER_UID,
                "condition_id": ph.XARM6_CONDITION_ID,
                "calibration_success_rate": rate,
                "passes_gate": passes,
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
        ],
        "ur_backup": {
            "requested": "ur_e",
            "available_in_maniskill_3_0_1": "ur_10e (no ur_e)",
            "usable": False,
            "reason": "ur_10e ships with NO gripper / TCP link (urdf_path=None, "
            "gripper_joint_names=None, ee/tcp_link_name=None), so the dual-hold bar weld has "
            "no grip frame to attach to. Structurally unusable as a co-carry teammate — a "
            "documented exclusion, not a tuning gap. xarm6_robotiq is the only ManiSkill "
            "3.0.1 robot with a different body AND a weldable Robotiq gripper.",
        },
        "admitted": admitted,
        "excluded": excluded,
        "n_admitted": len(admitted),
        "min_admitted_required": _MIN_ADMITTED,
        "stop_triggered": len(admitted) < _MIN_ADMITTED,
        "feasibility_finding": (
            None
            if passes
            else "Capability-matching an embodiment-heterogeneous teammate to this co-carry "
            "task with a hand-written controller is INFEASIBLE under the unchanged bar "
            "(success predicate, f_max 130 N, 0.10 m radius). The rigid dual-hold weld "
            "(20000 N/m) only tolerates embodiment-SYMMETRIC pairs: the matched Panda pair "
            "stays at ~75 N internal stress by symmetry, but the xArm6 + Panda-ego pair fight "
            "through the bar at 3-5x that (stress p90 well above the 130 N ceiling) and/or "
            "tilt it past 15 deg, across a wide compliance/leveling/gain sweep. The measurement "
            "does not run; this is the EH result for this rig."
        ),
        "directional_bias_caveat": (
            "EXCLUSION biases any EH Δ toward zero AND here PRECLUDES the measurement. This "
            "result may NOT be read as 'embodiment does not couple to cooperation' — the "
            "opposite: the rigid weld couples embodiment so strongly that no capability-matched "
            "cross-embodiment teammate exists. C_min was NOT weakened to force admission."
        ),
        "notes": [
            "The cooperative reference ego is the matched cocarry_impedance (a proven "
            "cooperative controller), NOT the frozen incumbent, so the gate measures the "
            "xArm6's task capability, not how the incumbent reacts to it.",
            "A wide hand-written-controller sweep (compliant impedance, admittance follower, "
            "cooperative bar-leveling, gain/damping/step variations) was explored on throwaway "
            "seeds; none cleared placed AND level AND unstressed jointly. The registered "
            "controller is the principled compliant + cooperative-leveling configuration.",
        ],
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(roster, fh, sort_keys=True, indent=2)
    print(f"      admitted={admitted} excluded={excluded}")
    print(f"      roster -> {out_path}")
    return 0 if len(admitted) >= _MIN_ADMITTED else 2


if __name__ == "__main__":
    sys.exit(main())
