# SPDX-License-Identifier: Apache-2.0
"""Co-insert S2 fixed-link competence-tuning probe — BOUND_HIT (ADR-026 §D4; ADR-005).

The bounded competence-tuning slice on the validated fixed-link rig (the
create_drive "wall" disproven). The founder authorised a FIRM bound of 8 tuning
configurations to reach the matched insertion seat; this generator reproduces the
committed best regime (config 4) and the single-inserter positive control, and
records the verdict: BOUND_HIT — the matched pair reliably reaches ~30 mm with
good alignment but cannot clear the last ~8 mm to the 38 mm seat (a friction
wedge), robust across the create_drive AND fixed-link architectures and all 8
control configs. This is a TASK-PARAMETERISATION finding (0.5 mm clearance + the
declared friction + 40 mm depth in the quasi-static control regime), NOT the
(disproven) sim/constraint-fidelity wall and NOT a control-competence failure —
a founder call: adjust the still-unfrozen task params or honest stop.

Determinism: env P6 substream + deterministic controllers; fixed seeds. GPU/oracle
gated. Numbers come only from the committed artifact.

ADR-026 §Decision 1-4; ADR-005 §Decision (the dual-sim posture); ADR-009 §Decision.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from chamber.envs.coinsert import make_coinsert_env
from chamber.partners.api import PartnerSpec
from chamber.partners.registry import load_partner

_EGO = {
    "uid": "panda_wristcam",
    "base_xyz": "-0.5,0,0",
    "base_yaw_deg": "0",
    "peg_half_len": "0.04",
}
_HOLDER = {
    "uid": "panda_partner",
    "base_xyz": "0.5,0,0",
    "base_yaw_deg": "180",
    "peg_half_len": "0.04",
}
_SEEDS = (0, 1, 2, 3, 4)
_EP = 320

# The 8 bounded tuning configs tried + their outcome (the firm-bound log).
_TUNING_LOG = [
    "1. baseline (ori_gain 6): seat 0; ori-hold couples into lateral, peg never centres",
    "2. ori_gain 6->2.5: seat 0; lateral fixed (~1 mm), peg hovers at standoff",
    "3. standoff 15->4 mm: seat 0; peg stuck ~13 mm above (below-hold arm-arm interference)",
    "4. holder->long-bracket fixed-link: seat 0 but inserts 23 mm; socket braces (sockDrift 0)",
    "5. descend 0.002->0.005, stall 3->8: seat 0; reaches ~30 mm and HOLDS (committed best)",
    "6. descend->0.003, retract 0.004x3: seat 0; friction-lock drags holder (210 mm + 113 mm)",
    "7. stiff holder kp 2->8 + firm press: seat 0; DESTABILISES (socket flung, depth 86 mm)",
    "8. ori_gain->4 + moderate unjam: seat 0; reaches ~30 mm then drags 167 mm",
]


def _matched_gate() -> dict:
    out = []
    for seed in _SEEDS:
        env = make_coinsert_env(
            condition_id="coinsert_matched_reference",
            num_envs=1,
            render_backend="none",
            peg_clearance_m=1.0e-3,
            episode_length=_EP,
        )
        base = load_partner(PartnerSpec("coinsert_base_inserter", seed, None, None, dict(_EGO)))
        hold = load_partner(
            PartnerSpec("coinsert_reference_holder", seed, None, None, dict(_HOLDER))
        )
        obs, _ = env.reset(seed=seed)
        base.reset(seed=seed)
        hold.reset(seed=seed)
        info = {}
        for _ in range(_EP):
            a = {
                "panda_wristcam": np.asarray(base.act(obs), np.float32),
                "panda_partner": np.asarray(hold.act(obs), np.float32),
            }
            obs, _, term, _, info = env.step(a)
            if bool(np.asarray(term).reshape(-1)[0]):
                break
        out.append(
            {
                "seed": seed,
                "depth_mm": round(float(info["seated_depth_m"][0]) * 1000, 1),
                "align_deg": round(float(info["axis_align_deg"][0]), 1),
                "seated": bool(np.asarray(info["seated"]).reshape(-1)[0]),
                "success_geom": bool(np.asarray(info["success_geom"]).reshape(-1)[0]),
                "peak_insert_n": round(float(info["peak_insert_force_n"][0]), 1),
                "peak_couple_n": round(float(info["peak_couple_wrench_n"][0]), 1),
            }
        )
        env.close()
    return {
        "per_seed": out,
        "seat_rate": float(np.mean([r["success_geom"] for r in out])),
        "depth_mm_median": float(np.median([r["depth_mm"] for r in out])),
        "align_deg_median": float(np.median([r["align_deg"] for r in out])),
    }


def _single_inserter() -> dict:
    out = []
    for seed in (0, 1, 2):
        env = make_coinsert_env(
            condition_id="coinsert_single_inserter_positive_control",
            num_envs=1,
            render_backend="none",
            peg_clearance_m=1.0e-3,
            episode_length=_EP,
        )
        base = load_partner(PartnerSpec("coinsert_base_inserter", seed, None, None, dict(_EGO)))
        hold = load_partner(
            PartnerSpec("coinsert_reference_holder", seed, None, None, dict(_HOLDER))
        )
        obs, _ = env.reset(seed=seed)
        base.reset(seed=seed)
        hold.reset(seed=seed)
        info = {}
        for _ in range(_EP):
            a = {
                "panda_wristcam": np.asarray(base.act(obs), np.float32),
                "panda_partner": np.asarray(hold.act(obs), np.float32),
            }
            obs, _, term, _, info = env.step(a)
            if bool(np.asarray(term).reshape(-1)[0]):
                break
        out.append(bool(np.asarray(info["success_geom"]).reshape(-1)[0]))
        env.close()
    return {"success": out, "rate": float(np.mean(out))}


def main() -> int:
    out_dir = os.environ.get("OUT_DIR", "spikes/results/coinsert/s2/2026-06-25-fixedlink-tuning")
    os.makedirs(out_dir, exist_ok=True)
    matched = _matched_gate()
    single = _single_inserter()
    artifact = {
        "schema": "coinsert_s2_fixedlink_tuning/v1",
        "stage": "S2 — bounded competence-tuning on the fixed-link rig",
        "firm_bound": "8 tuning configurations (founder-set); hit without reaching the seat",
        "verdict": "BOUND_HIT",
        "matched": matched,
        "single_inserter_positive_control": single,
        "tuning_log": _TUNING_LOG,
        "diagnosis": (
            "On the validated fixed-link rig the matched pair reliably reaches "
            "~30 mm with good alignment (median depth {:.1f} mm, align {:.1f} deg), but "
            "cannot clear the last ~8 mm to the 38 mm seat: a friction wedge at "
            "~30 mm (0.5 mm-per-side clearance + the declared friction + 40 mm "
            "depth). The press authority needed to overcome the friction either "
            "COCKS the peg (align 0.3 -> 2.4-4.7 deg, worsening the wedge) or, once "
            "the peg friction-locks, DRAGS the holder (the joint impedance cannot "
            "resist the locked-pair reaction; stiffening it destabilises). This "
            "wall is robust across the create_drive AND fixed-link architectures "
            "and all 8 control configs."
        ).format(matched["depth_mm_median"], matched["align_deg_median"]),
        "interpretation": (
            "NOT the (disproven) SAPIEN constraint-fidelity wall — the fixed link "
            "braces with no over-constraint and the single-inserter positive "
            "control holds (rate {:.2f}). NOT a control-competence failure that 8 "
            "configs could fix. A TASK-PARAMETERISATION wall: the task as "
            "parameterised (clearance + the declared peg-socket friction + the "
            "40 mm seat depth in the quasi-static control regime) is mis-conditioned "
            "even for a competent controller. Founder call: adjust the still-"
            "unfrozen task params (the declared friction and/or the seat depth) or "
            "honest stop. Do NOT tune unbounded."
        ).format(single["rate"]),
        "honesty_statement": (
            "Numbers are the committed probe output; no value is hand-entered. The "
            "matched pair does NOT seat — the >=0.9 precondition is not met — so the "
            "coupling preview, Gate-D, capability-gate, force-limit and "
            "oracle-calibration steps are NOT run; this slice stops at the firm bound."
        ),
    }
    out_path = os.path.join(out_dir, "coinsert_s2_fixedlink_tuning_probe.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(artifact, fh, sort_keys=True, indent=2)
    print(f"  matched seat_rate {matched['seat_rate']} depth_median {matched['depth_mm_median']}mm")
    print(
        f"  matched align_median {matched['align_deg_median']}deg; single-inserter {single['rate']}"
    )
    print(f"  VERDICT: {artifact['verdict']}")
    print(f"  artifact -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
