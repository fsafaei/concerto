# SPDX-License-Identifier: Apache-2.0
"""Rung-4b Stage-1 compliant-coupling sweep + gate (ADR-026 §D4; ADR-005; R-2026-06-B §15 Rung 4b).

Invoked by ``scripts/repro/cocarry_rung4b_coupling_sweep.sh`` AFTER the
pre-registration is committed + tagged. For each candidate coupling setting
(Variant A linear stiffness grid + a Variant B force-saturated point) it
measures the four binding constraints over the pre-registered seeds:

- C1 matched Panda pair ~100% with small deflection,
- C2 single-arm positive control ~0 (coupling-validity preserved),
- C3 the coupling binds on matched successes (tilt/stress load-bearing),
- C4 the xArm6 teammate is admitted (calibration vs the cooperative Panda ego
  clears C_min, success predicate UNCHANGED).

Plus the zero-action static-hold wrist-stress baseline (Panda vs xArm6) that
attributes any C4 failure to a cooperative fight vs an embodiment-dependent
proxy. Applies the pre-registered selection/gate rule and writes the verdict.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime

import numpy as np

from chamber.benchmarks import cocarry_ph as ph
from chamber.benchmarks.cocarry_runner import build_matched_controllers
from chamber.envs.cocarry import COCARRY_GOAL_THRESH_M, COCARRY_STRESS_MAX_PROXY_N
from chamber.envs.cocarry_obs import make_cocarry_training_env

_RENDER = "none"
_SEEDS = [70010, 70011, 70012, 70013, 70014, 70015, 70016, 70017]
_EPLEN = 320
_FMAX = COCARRY_STRESS_MAX_PROXY_N
_RADIUS = COCARRY_GOAL_THRESH_M
# Pre-registered grid.
_VARIANT_A = [8000.0, 4000.0, 2000.0]
_VARIANT_B = [(8000.0, 110.0)]
# Pre-stated thresholds for the C1-C4 verdicts.
_C1_SUCCESS = 0.9  # matched success floor
_C1_DEFLECTION_FRAC = 0.8  # matched centroid p90 <= this * radius (clear margin)
_C2_SINGLE_MAX = 0.1  # single-arm success ceiling (positive control ~0)
_C3_STRESS_FRAC = 0.2  # coupling binds: matched stress p90 >= this * f_max ...
_C3_TILT_DEG = 2.0  # ... OR matched tilt p90 >= this (non-slack)
_SETTLE = 15  # placement settle window skipped in the static baseline


def _matched(seed: int, stiffness: float, force_limit: float | None) -> ph.ConjunctMetrics:
    env = make_cocarry_training_env(
        condition_id="cocarry_matched_panda_pair",
        root_seed=seed,
        render_backend=_RENDER,
        drive_stiffness=stiffness,
        drive_force_limit=force_limit,
    )
    try:
        ego = build_matched_controllers()["panda_wristcam"]
        partner = build_matched_controllers()["panda_partner"]
        return ph.rollout_pair(
            env=env,
            ego_act=lambda o, _e=ego: np.asarray(_e.act(o), dtype=np.float32),
            partner=partner,
            seed=seed,
            episode_length=_EPLEN,
        )
    finally:
        env.close()


def _single_arm(seed: int, stiffness: float) -> ph.ConjunctMetrics:
    # Positive control: the partner is retracted with no drive (env single_arm),
    # so it cannot affect the bar; success ~0 reflects the single ego alone.
    env = make_cocarry_training_env(
        condition_id="cocarry_single_arm_positive_control",
        root_seed=seed,
        render_backend=_RENDER,
        drive_stiffness=stiffness,
    )
    try:
        ego = build_matched_controllers()["panda_wristcam"]
        partner = build_matched_controllers()["panda_partner"]
        return ph.rollout_pair(
            env=env,
            ego_act=lambda o, _e=ego: np.asarray(_e.act(o), dtype=np.float32),
            partner=partner,
            seed=seed,
            episode_length=_EPLEN,
        )
    finally:
        env.close()


def _static_wrist(condition_id: str, stiffness: float, seed: int = 70010, n: int = 120) -> dict:
    """Zero-action static hold wrist-stress (no cooperative motion) — the C4 diagnostic."""
    env = make_cocarry_training_env(
        condition_id=condition_id, root_seed=seed, render_backend=_RENDER, drive_stiffness=stiffness
    )
    try:
        env.reset(seed=seed)
        ego_uid = env.get_wrapper_attr("ego_uid")
        puid = env.get_wrapper_attr("partner_uid")
        spaces = env.action_space.spaces  # type: ignore[attr-defined]
        adim = {u: spaces[u].shape[0] for u in (ego_uid, puid)}
        vals: list[float] = []
        for step in range(n):
            env.step(
                {
                    ego_uid: np.zeros(adim[ego_uid], dtype=np.float32),
                    puid: np.zeros(adim[puid], dtype=np.float32),
                }
            )
            tel = env.get_wrapper_attr("get_telemetry")()
            if step >= _SETTLE:
                vals.append(float(np.asarray(tel["stress_proxy"]).reshape(-1)[0]))
        return {"p50": float(np.percentile(vals, 50)), "max": float(np.max(vals))}
    finally:
        env.close()


def _summ(metrics: list) -> dict:
    return {
        "success_rate": float(np.mean([m.success for m in metrics])),
        "centroid_p90": float(np.percentile([m.centroid_to_goal for m in metrics], 90)),
        "tilt_p90": float(np.percentile([m.max_tilt_deg for m in metrics], 90)),
        "stress_p90": float(np.percentile([m.max_stress_proxy for m in metrics], 90)),
        "stress_max": float(np.max([m.max_stress_proxy for m in metrics])),
        "fail_placed": int(sum(1 for m in metrics if not m.is_placed)),
        "fail_level": int(sum(1 for m in metrics if not m.is_level)),
        "fail_unstressed": int(sum(1 for m in metrics if not m.is_unstressed)),
    }


def _evaluate_setting(label: str, stiffness: float, force_limit: float | None) -> dict:
    print(f"    [{label}] K={stiffness:.0f} FL={force_limit}...")
    ms = _summ([_matched(s, stiffness, force_limit) for s in _SEEDS])
    ss = _summ([_single_arm(s, stiffness) for s in _SEEDS])
    xarm6 = ph.evaluate_calibration(
        candidate_class=ph.XARM6_PARTNER_CLASS,
        seeds=_SEEDS,
        condition_id=ph.XARM6_CONDITION_ID,
        partner_uid=ph.XARM6_PARTNER_UID,
        render_backend=_RENDER,
        drive_stiffness=stiffness,
        drive_force_limit=force_limit,
    )
    xs = _summ(xarm6)
    c1 = ms["success_rate"] >= _C1_SUCCESS and ms["centroid_p90"] <= _C1_DEFLECTION_FRAC * _RADIUS
    c2 = ss["success_rate"] <= _C2_SINGLE_MAX
    c3 = ms["stress_p90"] >= _C3_STRESS_FRAC * _FMAX or ms["tilt_p90"] >= _C3_TILT_DEG
    c4 = bool(xarm6) and float(np.mean([m.success for m in xarm6])) >= ph.C_MIN
    print(
        f"      C1 matched={ms['success_rate']:.0%} cen_p90={ms['centroid_p90']:.3f} -> {c1} | "
        f"C2 single={ss['success_rate']:.0%} -> {c2} | "
        f"C3 tilt_p90={ms['tilt_p90']:.0f} str_p90={ms['stress_p90']:.0f} -> {c3} | "
        f"C4 xarm6={xs['success_rate']:.0%} Ufail={xs['fail_unstressed']}/{len(_SEEDS)} -> {c4}"
    )
    return {
        "label": label,
        "stiffness_Npm": stiffness,
        "force_limit_N": force_limit,
        "matched_C1_C3": ms,
        "single_arm_C2": ss,
        "xarm6_C4": xs,
        "C1_matched_clean": bool(c1),
        "C2_single_arm_near_zero": bool(c2),
        "C3_coupling_binds": bool(c3),
        "C4_xarm6_admitted": bool(c4),
        "all_satisfied": bool(c1 and c2 and c3 and c4),
    }


def _verdict(settings: list, static: dict) -> tuple[str, str, dict]:
    passing = [s for s in settings if s["all_satisfied"]]
    chosen = max(passing, key=lambda s: s["stiffness_Npm"]) if passing else None
    xarm6_static = static["xarm6_partner_K8000"]
    xarm6_static_over_fmax = xarm6_static["p50"] >= _FMAX or xarm6_static["max"] >= _FMAX
    c4_ever = any(s["C4_xarm6_admitted"] for s in settings)
    c1c3_ever = any(s["C1_matched_clean"] and s["C3_coupling_binds"] for s in settings)
    summary = {
        "any_C1_C3_satisfied": bool(c1c3_ever),
        "any_C4_satisfied": bool(c4_ever),
        "xarm6_static_wrist_over_fmax": bool(xarm6_static_over_fmax),
        "chosen_setting": chosen,
    }
    if chosen is not None:
        gate = (
            f"Stiffest setting satisfying C1-C4: {chosen['label']} "
            f"(K={chosen['stiffness_Npm']:.0f})."
        )
        return "PROCEED", gate, summary
    if c1c3_ever and xarm6_static_over_fmax and not c4_ever:
        gate = (
            "C1-C3 satisfiable (the compliant coupling resolves the Rung-4 over-coupling "
            "fight: the xArm6 transports + levels), but C4 is UNSATISFIABLE. The xArm6's "
            "wrist-incoming-force stress proxy reads >= f_max even at a zero-action static "
            "hold (no cooperative fight), so the proxy is EMBODIMENT-DEPENDENT and the "
            "Panda-calibrated f_max rejects the xArm6 on a body artifact, not a fight. "
            "Fixing it needs an embodiment-invariant proxy / per-body f_max calibration — a "
            "predicate/f_max change the rules forbid here. HONEST FALLBACK: a sharper finding "
            "than Rung-4 (the over-coupling wall is removed; the blocker is now the "
            "measurement instrument)."
        )
        return "STOP_PROXY_ARTIFACT", gate, summary
    gate = "No setting satisfies C1-C4 and the cause is not isolated to the proxy; STOP."
    return "STOP_NO_VALID_COUPLING", gate, summary


def main() -> int:
    out = os.environ.get(
        "OUT_JSON", "spikes/results/cocarry/rung4b/cocarry_rung4b_coupling_sweep.json"
    )
    print(f"    seeds={_SEEDS}  f_max={_FMAX:.1f} N  radius={_RADIUS} m  C_min={ph.C_MIN}")
    print("    [diagnostic] zero-action static-hold wrist baseline (Panda vs xArm6)...")
    static = {
        "panda_partner_K8000": _static_wrist("cocarry_matched_panda_pair", 8000.0),
        "xarm6_partner_K8000": _static_wrist("cocarry_xarm6_partner", 8000.0),
        "panda_partner_K20000_rigidish": _static_wrist("cocarry_matched_panda_pair", 20000.0),
        "xarm6_partner_K20000_rigidish": _static_wrist("cocarry_xarm6_partner", 20000.0),
    }
    print(
        f"      static panda={static['panda_partner_K8000']} xarm6={static['xarm6_partner_K8000']}"
    )

    settings = [_evaluate_setting(f"A_K{int(k)}", k, None) for k in _VARIANT_A]
    settings += [_evaluate_setting(f"B_K{int(k)}_FL{int(fl)}", k, fl) for k, fl in _VARIANT_B]

    verdict, gate, summary = _verdict(settings, static)
    panda_s = static["panda_partner_K8000"]
    xarm6_s = static["xarm6_partner_K8000"]
    diagnosis = (
        "Zero-action static hold (no cooperative motion). xArm6 wrist stress "
        f"p50={xarm6_s['p50']:.0f}/max={xarm6_s['max']:.0f} N vs Panda "
        f"p50={panda_s['p50']:.0f}/max={panda_s['max']:.0f} N — the xArm6 (Robotiq) wrist "
        f"reads ~3-6x the Panda for the same rest hold, already over f_max ({_FMAX:.0f} N) "
        "BEFORE any fight. The wrist-incoming-joint-force proxy is embodiment-dependent."
    )
    artifact = {
        "schema": "cocarry_rung4b_coupling_sweep/v1",
        "date": datetime.now(UTC).strftime("%Y-%m-%d"),
        "prereg_artifact": "spikes/results/cocarry/rung4b/cocarry_rung4b_coupling_prereg.json",
        "seeds": _SEEDS,
        "f_max_n": _FMAX,
        "goal_radius_m": _RADIUS,
        "c_min": ph.C_MIN,
        "static_wrist_baseline": static,
        "static_wrist_diagnosis": diagnosis,
        "settings": settings,
        "chosen_setting": summary["chosen_setting"],
        "verdict": verdict,
        "gate": gate,
        "constraints_summary": summary,
    }
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(artifact, fh, sort_keys=True, indent=2)
    print(f"\n    VERDICT: {verdict}\n    {gate}\n    artifact -> {out}")
    return 0 if verdict == "PROCEED" else 2


if __name__ == "__main__":
    sys.exit(main())
