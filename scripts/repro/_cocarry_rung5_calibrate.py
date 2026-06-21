# SPDX-License-Identifier: Apache-2.0
# pyright: reportPrivateImportUsage=false
"""Rung-5 Stage-2 capability calibration (pre-registered; ADR-026 §D4; ADR-007; ADR-009).

Independent-competence gate for the Rung-5 co-design measurement, re-run FRESH
on this rung's fixed substrate (compliant K=8000, the coupling instrument,
f_max 365.6) — Rung-3's admissions do NOT transfer. Each candidate (on the
partner seat) is paired with the COOPERATIVE-REFERENCE EGO (matched impedance,
NOT the frozen incumbent — calibrating against the incumbent would gate on the
very outcome being measured, circular), so the score isolates the candidate's
task competence and a downstream joint drop is attributable to mismatch /
non-co-design, not to a weak teammate.

- Arm A (co-designed control-style family): stiff / admittance / slew /
  nullspace impedance.
- Arm B (NON-co-designed, the decisive arm): selfish_goal / _effort / _station
  (matched-impedance class, matched gains, objective-only shift).

C_min = 0.75. A candidate clears the gate iff its joint success (judged at
f_max 365.6) over the calibration seeds >= C_min; else EXCLUDED (archived, with
the directional-bias caveat). PRE-REGISTERED STOP: fewer than 2 Arm-B variants
competent => the >= 2-variant replication requirement cannot be met => STOP /
feasibility-note (no CD-axis verdict). Exit 0 = proceed; exit 3 = STOP.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

import numpy as np

import chamber.envs.cocarry as cc
from chamber.benchmarks import cocarry_ph as ph
from chamber.envs.cocarry_obs import make_cocarry_training_env
from chamber.partners.api import PartnerSpec
from chamber.partners.cocarry_policy_shift import COCARRY_POLICY_SHIFT_CANDIDATES
from chamber.partners.cocarry_selfish import COCARRY_SELFISH_CANDIDATES
from chamber.partners.registry import load_partner

# Locked Stage-0 constants (cocarry_rung5_codesign_prereg.json).
_FMAX = 365.6
_K = 8000.0
_MEASURE = "coupling"
_EPISODE = cc.COCARRY_DEFAULT_EPISODE_LENGTH  # 320
_C_MIN = ph.C_MIN  # 0.75
_RENDER = "none"
_CALIB_SEEDS = list(range(53000, 53012))  # 12
_MIN_ARM_B_COMPETENT = 2


def _calibration_episode(candidate_class: str, seed: int) -> ph.ConjunctMetrics:
    """Pair the candidate (partner seat) with the cooperative reference ego (ego seat)."""
    env = make_cocarry_training_env(
        condition_id="cocarry_matched_panda_pair",
        root_seed=seed,
        render_backend=_RENDER,
        drive_stiffness=_K,
        stress_measure=_MEASURE,
        stress_max=_FMAX,
    )
    try:
        ego = ph.build_cooperative_ego(seed=seed)
        ego.reset(seed=seed)
        partner = load_partner(
            PartnerSpec(
                candidate_class,
                seed,
                None,
                None,
                dict(cc.cocarry_matched_controller_specs()["panda_partner"]),
            )
        )
        return ph.rollout_pair(
            env=env,
            ego_act=lambda o, _e=ego: np.asarray(_e.act(o), dtype=np.float32),
            partner=partner,
            seed=seed,
            episode_length=_EPISODE,
        )
    finally:
        env.close()


def _calibrate(candidate_class: str) -> dict[str, Any]:
    """Calibrate one candidate over the calibration seeds; return its roster entry."""
    metrics = [_calibration_episode(candidate_class, s) for s in _CALIB_SEEDS]
    summary = ph.conjunct_failure_summary(metrics)
    rate = summary["success_rate"]
    passes = ph.passes_capability_gate(rate, c_min=_C_MIN)
    print(
        f"      {candidate_class:28s} success={rate:.3f}  passes(C_min={_C_MIN})={passes}  "
        f"(fail placed/level/unstress/static = "
        f"{summary['fail_placed']}/{summary['fail_level']}/"
        f"{summary['fail_unstressed']}/{summary['fail_static']})"
    )
    return {
        "class_name": candidate_class,
        "calibration_success_rate": rate,
        "passes_gate": bool(passes),
        "conjunct_summary": summary,
        "per_seed": [
            {
                "seed": m.seed,
                "success": m.success,
                "stress": m.max_stress_proxy,
                "tilt": m.max_tilt_deg,
            }
            for m in metrics
        ],
    }


def main() -> int:
    out_path = os.environ.get(
        "OUT_JSON", "spikes/results/cocarry/rung5/cocarry_rung5_calibration_roster.json"
    )
    arm_a = list(COCARRY_POLICY_SHIFT_CANDIDATES)
    arm_b = list(COCARRY_SELFISH_CANDIDATES)

    print(f"    [Arm A] co-designed control-style family ({len(arm_a)} candidates, K={_K})...")
    roster_a = [_calibrate(c) for c in arm_a]
    print(f"    [Arm B] non-co-designed self-interested variants ({len(arm_b)} candidates)...")
    roster_b = [_calibrate(c) for c in arm_b]

    admitted_a = [r["class_name"] for r in roster_a if r["passes_gate"]]
    excluded_a = [r["class_name"] for r in roster_a if not r["passes_gate"]]
    admitted_b = [r["class_name"] for r in roster_b if r["passes_gate"]]
    excluded_b = [r["class_name"] for r in roster_b if not r["passes_gate"]]

    stops: list[str] = []
    if len(admitted_b) < _MIN_ARM_B_COMPETENT:
        stops.append(
            f"only {len(admitted_b)} Arm-B variant(s) competent (need >= {_MIN_ARM_B_COMPETENT}); "
            "the >= 2-variant replication requirement cannot be met -> STOP / feasibility-note "
            "(no CD-axis verdict)."
        )
    verdict = "STOP" if stops else "PROCEED"

    artifact = {
        "schema": "cocarry_rung5_calibration_roster/v1",
        "stage": "Rung-5 Stage-2 capability calibration",
        "constants": {
            "c_min": _C_MIN,
            "fmax_coupling_n": _FMAX,
            "drive_stiffness": _K,
            "stress_measure": _MEASURE,
            "calibration_seeds": _CALIB_SEEDS,
            "min_arm_b_competent": _MIN_ARM_B_COMPETENT,
        },
        "pairing": (
            "candidate (partner seat) vs cooperative-reference ego "
            "(cocarry_impedance, NOT the incumbent — non-circular)"
        ),
        "arm_a": {"roster": roster_a, "admitted": admitted_a, "excluded": excluded_a},
        "arm_b": {"roster": roster_b, "admitted": admitted_b, "excluded": excluded_b},
        "directional_bias_caveat": (
            "The capability gate truncates heterogeneity from above (excludes teammates the "
            "cooperative reference cannot succeed with), biasing Δ toward zero. On a null, report "
            "'among capability-matched partners, no qualifying drop' — never 'co-design is inert'."
        ),
        "verdict": verdict,
        "stops": stops,
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(artifact, fh, sort_keys=True, indent=2)
    print(f"    Arm A admitted: {admitted_a}  excluded: {excluded_a}")
    print(f"    Arm B admitted: {admitted_b}  excluded: {excluded_b}")
    print(f"    verdict = {verdict}")
    for s in stops:
        print(f"      STOP: {s}")
    print(f"    artifact -> {out_path}")
    return 3 if stops else 0


if __name__ == "__main__":
    sys.exit(main())
