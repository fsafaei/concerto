# SPDX-License-Identifier: Apache-2.0
# pyright: reportPrivateImportUsage=false
"""Rung-4e - the task-fair embodiment search (the bounded, conclusive last EH shot).

ADR-026 §Decision 2 / §Decision 4 / §Validation criteria; ADR-005; ADR-009;
R-2026-06-B §15 Rung 4e. Rung-4c's embodiment headline was retracted as a pose
artifact (Rung-4d): a stiffness-optimised xArm6 carries the bar IN-BAND on
coupling, so the over-load was an unoptimised default pose. But that same
optimisation revealed a CANDIDATE - the stiffness-optimal pose then over-tilts
(active tilt p90 26.3 deg > 15). This driver runs ONE bounded, task-fair search
over the xArm6's POSE x CONTROLLER against the FULL joint-success criterion (not
a single proxy) and decides by the pre-committed, falsifiable-both-ways rule in
``spikes/results/cocarry/rung4e/cocarry_rung4e_taskfair_prereg.json`` (tag
``prereg-cocarry-rung4e-taskfair-2026-06-20``), arbitrated by the stress<->tilt
Pareto frontier with a search-adequacy criterion.

Everything except the xArm6 pose x controller is held fixed: compliant coupling
(K=8000), the invariant coupling stress measure, f_max 365.6 N, the predicate,
the 0.10 m radius, the 15 deg tilt ceiling, and the fixed cooperative reference
ego (matched impedance) across the matched-Panda and xArm6 conditions. The
controller search stays in the matched compliant family (kp fixed; only the
leveling/yielding levers level_gain x admit_bias and the pose vary) so a drop
cannot be manufactured by stiffening the partner.

Decision (pre-committed):
  * best xArm6 config reaches joint success >= C_min on the confirm seeds
    -> EH ROBUSTLY NOT ESTABLISHED (a fair different body co-carries as well).
  * no config in-band on all conjuncts AND the stress<->tilt Pareto frontier has
    no in-band-on-both point AND search-adequacy holds
    -> EH ESTABLISHED via a multi-objective infeasibility.
  * otherwise -> INDETERMINATE (report honestly; do not claim either way).
"""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from chamber.agents.panda_jacobian import PandaJacobianProvider
from chamber.agents.xarm6_jacobian import XArm6JacobianProvider
from chamber.benchmarks import cocarry_ph as ph
from chamber.envs.cocarry_obs import make_cocarry_training_env
from chamber.partners.api import PartnerSpec
from chamber.partners.registry import load_partner

if TYPE_CHECKING:
    from chamber.benchmarks.cocarry_ph import ConjunctMetrics

# --- Fixed task (REUSED verbatim; never weakened) ---------------------------
_K = 8000.0
_MEASURE = "coupling"
_FMAX = 365.6
_TILT_MAX = 15.0
_GOAL = 0.10
_EP_LEN = 320
_RENDER = "none"

# --- Search space -----------------------------------------------------------
_BASES = (0.30, 0.35, 0.40, 0.45)
_RESTARTS = 80
_LEVEL_GAINS = (0.5, 1.0, 1.5)
_ADMIT_BIASES = (0.7, 1.0)
_DEFAULT_Q6 = np.array([0.0, -0.084, -0.8, 0.0, 0.692, 0.0])
_PANDA_CARRY_Q = np.array([0.0, np.pi / 8, 0.0, -np.pi * 5 / 8, 0.0, np.pi * 3 / 4, -np.pi / 4])
_GRIPPER_PAD = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# --- Seeds (disjoint from every prior set) ----------------------------------
_SEARCH_SEEDS = [81000, 81001, 81002, 81003]
_CONFIRM_SEEDS = list(range(81100, 81112))

# --- Decision + adequacy (pre-committed) ------------------------------------
_C_MIN = 0.75
_ADEQ_PLATEAU_EPS = 0.02
_ADEQ_TAIL_FRAC = 0.40
_ADEQ_MIN_FRONTIER = 4
_ADEQ_MARGIN = 0.15
_MIN_RATE = 0.5  # a config counts as placed/static for the frontier if the rate clears this

# --- Kinematics -------------------------------------------------------------
_IK_TOL = 1e-5
_IK_SUCCESS = 1e-3
_LIM = np.array(
    [
        [-6.283, 6.283],
        [-2.059, 2.094],
        [-3.800, 0.192],
        [-6.283, 6.283],
        [-1.693, 3.142],
        [-6.283, 6.283],
    ]
)
_TARGET_W = np.array([-0.115, 0.0, 0.1698])
_BAR_HALF = 0.115
_P = XArm6JacobianProvider()


def _rotz_t(deg: float) -> np.ndarray:
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1.0]])


def _vcomp(q6: np.ndarray) -> float:
    j = _P.jacobian(q6)
    return float(np.linalg.inv(j @ j.T + 1e-9 * np.eye(3))[2, 2])


def _ik(tgt_base: np.ndarray, q0: np.ndarray, iters: int = 150) -> tuple[np.ndarray, float]:
    q = np.clip(q0, _LIM[:, 0], _LIM[:, 1])
    for _ in range(iters):
        e = tgt_base - _P.fk_tcp_position(q)
        if np.linalg.norm(e) < _IK_TOL:
            break
        j = _P.jacobian(q)
        step = np.clip(j.T @ np.linalg.solve(j @ j.T + 1e-4 * np.eye(3), e), -0.1, 0.1)
        q = np.clip(q + step, _LIM[:, 0], _LIM[:, 1])
    return q, float(np.linalg.norm(tgt_base - _P.fk_tcp_position(q)))


def _diverse_poses() -> list[dict]:
    """Per-base min-vertical-compliance feasible carry pose + the default anchor."""
    rng = np.random.default_rng(0)
    poses: list[dict] = []
    for bx in _BASES:
        tgt = _rotz_t(180.0) @ (_TARGET_W - np.array([bx, 0, 0]))
        best: tuple[float, np.ndarray] | None = None
        for _ in range(_RESTARTS):
            q, err = _ik(tgt, rng.uniform(_LIM[:, 0], _LIM[:, 1]))
            feasible = bool(np.all(q >= _LIM[:, 0]) and np.all(q <= _LIM[:, 1]))
            if err < _IK_SUCCESS and feasible:
                vc = _vcomp(q)
                if best is None or vc < best[0]:
                    best = (vc, q.copy())
        if best is not None:
            poses.append(
                {
                    "label": f"minvc_base{bx}",
                    "base_x": float(bx),
                    "q6": [round(x, 5) for x in best[1].tolist()],
                    "vcomp": round(best[0], 3),
                }
            )
    poses.append(
        {
            "label": "default_base0.35",
            "base_x": 0.35,
            "q6": [round(x, 5) for x in _DEFAULT_Q6.tolist()],
            "vcomp": round(_vcomp(_DEFAULT_Q6), 3),
        }
    )
    return poses


def _panda_vcomp() -> float:
    pj = PandaJacobianProvider()(None, _PANDA_CARRY_Q)  # type: ignore[arg-type]
    return float(np.linalg.inv(pj @ pj.T + 1e-9 * np.eye(3))[2, 2])


def _agg(mets: list[ConjunctMetrics]) -> dict:
    """Aggregate per-seed metrics; the unstressed conjunct judged at the coupling f_max."""
    stress = np.asarray([m.max_stress_proxy for m in mets])
    tilt = np.asarray([m.max_tilt_deg for m in mets])
    cen = np.asarray([m.centroid_to_goal for m in mets])
    placed = np.asarray([bool(m.is_placed) for m in mets])
    level = np.asarray([bool(m.is_level) for m in mets])
    static = np.asarray([bool(m.both_static) for m in mets])
    unstressed = stress < _FMAX
    joint = placed & level & unstressed & static
    return {
        "n": len(mets),
        "stress_p90": round(float(np.percentile(stress, 90)), 1),
        "tilt_p90": round(float(np.percentile(tilt, 90)), 1),
        "centroid_p90": round(float(np.percentile(cen, 90)), 4),
        "placed_rate": round(float(np.mean(placed)), 3),
        "level_rate": round(float(np.mean(level)), 3),
        "unstressed_rate": round(float(np.mean(unstressed)), 3),
        "static_rate": round(float(np.mean(static)), 3),
        "joint_success": round(float(np.mean(joint)), 3),
    }


def _deficit(agg: dict) -> float:
    """Smooth JOINT-success deficit over ALL conjuncts (the search guide, not a single proxy)."""
    return round(
        max(0.0, agg["stress_p90"] / _FMAX - 1.0)
        + max(0.0, agg["tilt_p90"] / _TILT_MAX - 1.0)
        + max(0.0, agg["centroid_p90"] / _GOAL - 1.0)
        + (1.0 - agg["static_rate"]),
        4,
    )


def _eval_xarm6(
    base_x: float, q6: list[float], level_gain: float, ab: float, seeds: list[int]
) -> dict:
    mets: list[ConjunctMetrics] = []
    for s in seeds:
        env = make_cocarry_training_env(
            condition_id="cocarry_xarm6_partner",
            episode_length=_EP_LEN,
            root_seed=s,
            render_backend=_RENDER,
            drive_stiffness=_K,
            stress_measure=_MEASURE,
            xarm6_base_x=base_x,
            xarm6_ready_qpos=[*q6, *_GRIPPER_PAD],
        )
        try:
            coop = ph.build_cooperative_ego(seed=s)
            coop.reset(seed=s)
            extra = {
                "uid": "xarm6_robotiq",
                "base_xyz": f"{base_x},0.0,0.0",
                "base_yaw_deg": "180",
                "end_sign": "-1",
                "bar_half_len": repr(_BAR_HALF),
                "level_gain": repr(level_gain),
                "admit_bias": repr(ab),
            }
            par = load_partner(PartnerSpec("cocarry_xarm6_impedance", s, None, None, extra))
            mets.append(
                ph.rollout_pair(
                    env=env,
                    ego_act=lambda o, _e=coop: np.asarray(_e.act(o), dtype=np.float32),
                    partner=par,
                    seed=s,
                    episode_length=_EP_LEN,
                )
            )
        finally:
            env.close()
    return _agg(mets)


def _eval_matched(seeds: list[int]) -> dict:
    mets = ph.evaluate_calibration(
        candidate_class="cocarry_impedance",
        seeds=seeds,
        episode_length=_EP_LEN,
        render_backend=_RENDER,
        condition_id="cocarry_matched_panda_pair",
        partner_uid="panda_partner",
        drive_stiffness=_K,
        stress_measure=_MEASURE,
    )
    return _agg(mets)


def _eligible(configs: list[dict]) -> list[dict]:
    """Configs that already satisfy placed AND static (the two non-tradeoff conjuncts)."""
    return [
        c
        for c in configs
        if c["agg"]["placed_rate"] >= _MIN_RATE and c["agg"]["static_rate"] >= _MIN_RATE
    ]


def _pareto_front(configs: list[dict]) -> list[dict]:
    """Non-dominated (lower stress_p90, lower tilt_p90) configs that are placed AND static."""
    elig = _eligible(configs)
    front: list[dict] = []
    for c in elig:
        cs, ct = c["agg"]["stress_p90"], c["agg"]["tilt_p90"]
        dominated = any(
            (o["agg"]["stress_p90"] <= cs and o["agg"]["tilt_p90"] <= ct)
            and (o["agg"]["stress_p90"] < cs or o["agg"]["tilt_p90"] < ct)
            for o in elig
        )
        if not dominated:
            front.append(c)
    return sorted(front, key=lambda c: c["agg"]["stress_p90"])


def _closest_box_approach(configs: list[dict]) -> float:
    """Min normalised L2 distance from any placed+static config to the in-band box corner region."""
    elig = _eligible(configs)
    if not elig:
        return float("inf")
    dists = []
    for c in elig:
        ds = max(0.0, c["agg"]["stress_p90"] / _FMAX - 1.0)
        dt = max(0.0, c["agg"]["tilt_p90"] / _TILT_MAX - 1.0)
        dists.append(float(np.hypot(ds, dt)))
    return round(min(dists), 4)


def _has_inband_both(configs: list[dict]) -> bool:
    return any(
        c["agg"]["stress_p90"] < _FMAX and c["agg"]["tilt_p90"] < _TILT_MAX
        for c in _eligible(configs)
    )


def main() -> int:
    out = os.environ.get(
        "OUT_JSON", "spikes/results/cocarry/rung4e/cocarry_rung4e_taskfair_search.json"
    )
    print("    [1] pose set (joint-limit-clamped min-compliance per base + default)...")
    poses = _diverse_poses()
    panda_vc = _panda_vcomp()
    for p in poses:
        ratio = p["vcomp"] / panda_vc
        print(f"      {p['label']}: base {p['base_x']} vcomp {p['vcomp']} ({ratio:.2f}x panda)")

    print("    [2] task-fair pose x controller search (4 search seeds)...")
    configs: list[dict] = []
    deficit_trace: list[float] = []
    best_deficit = float("inf")
    for p in poses:
        for lg in _LEVEL_GAINS:
            for ab in _ADMIT_BIASES:
                agg = _eval_xarm6(p["base_x"], p["q6"], lg, ab, _SEARCH_SEEDS)
                cfg = {
                    "pose": p["label"],
                    "base_x": p["base_x"],
                    "q6": p["q6"],
                    "vcomp": p["vcomp"],
                    "level_gain": lg,
                    "admit_bias": ab,
                    "agg": agg,
                    "deficit": _deficit(agg),
                }
                configs.append(cfg)
                best_deficit = min(best_deficit, cfg["deficit"])
                deficit_trace.append(round(best_deficit, 4))
                print(
                    f"      {p['label']:>16} lg={lg} ab={ab} -> "
                    f"stress_p90={agg['stress_p90']} tilt_p90={agg['tilt_p90']} "
                    f"js={agg['joint_success']} deficit={cfg['deficit']}"
                )

    best = min(configs, key=lambda c: c["deficit"])
    print(
        f"    [3] best search config: {best['pose']} lg={best['level_gain']} "
        f"ab={best['admit_bias']} (deficit {best['deficit']}, js {best['agg']['joint_success']})"
    )
    print("    [4] confirm: best config + matched Panda anchor on 12 confirm seeds...")
    best_confirm = _eval_xarm6(
        best["base_x"], best["q6"], best["level_gain"], best["admit_bias"], _CONFIRM_SEEDS
    )
    matched = _eval_matched(_CONFIRM_SEEDS)
    print(f"      xArm6 best confirm: {best_confirm}")
    print(f"      matched Panda anchor: {matched}")

    # --- Adequacy (pre-committed) -------------------------------------------
    tail = max(1, round(len(deficit_trace) * _ADEQ_TAIL_FRAC))
    tail_improvement = round(deficit_trace[-tail] - deficit_trace[-1], 4)
    plateau = bool(tail_improvement <= _ADEQ_PLATEAU_EPS)
    front = _pareto_front(configs)
    closest = _closest_box_approach(configs)
    best_confirm_inband = (
        best_confirm["stress_p90"] < _FMAX and best_confirm["tilt_p90"] < _TILT_MAX
    )
    inband_exists = _has_inband_both(configs) or best_confirm_inband
    density_ok = bool(len(front) >= _ADEQ_MIN_FRONTIER)
    margin_ok = bool(closest > _ADEQ_MARGIN)
    adequacy = {
        "plateau": plateau,
        "tail_frac": _ADEQ_TAIL_FRAC,
        "tail_deficit_improvement": tail_improvement,
        "plateau_eps": _ADEQ_PLATEAU_EPS,
        "frontier_points": len(front),
        "frontier_min": _ADEQ_MIN_FRONTIER,
        "density_ok": density_ok,
        "closest_box_approach_norm": closest,
        "closest_margin": _ADEQ_MARGIN,
        "margin_ok": margin_ok,
    }

    # --- Decision (pre-committed, falsifiable both ways) --------------------
    null_met = bool(best_confirm["joint_success"] >= _C_MIN)
    established_met = bool(
        (not null_met) and (not inband_exists) and plateau and density_ok and margin_ok
    )
    if null_met:
        verdict = "EH_ROBUSTLY_NOT_ESTABLISHED"
        js = best_confirm["joint_success"]
        finding = (
            f"A fairly posed+controlled xArm6 ({best['pose']}, level_gain "
            f"{best['level_gain']}, admit_bias {best['admit_bias']}) reaches joint "
            f"success {js} >= C_min {_C_MIN} on the confirm seeds (stress_p90 "
            f"{best_confirm['stress_p90']} < f_max {_FMAX:.0f}, tilt_p90 "
            f"{best_confirm['tilt_p90']} < {_TILT_MAX:.0f}), comparable to the matched "
            f"Panda anchor (js {matched['joint_success']}). The different body co-carries "
            f"as well as the matched body under fair pose+control optimisation -> EH "
            f"ROBUSTLY NOT ESTABLISHED on this task. The Rung-4d candidate (stress<->tilt "
            f"tradeoff) is RESOLVED by a fair search: a config is in-band on both."
        )
    elif established_met:
        verdict = "EH_ESTABLISHED_MULTIOBJECTIVE_INFEASIBILITY"
        finding = (
            f"No fairly posed+controlled xArm6 config gets in-band on all conjuncts at once (best "
            f"confirm joint success {best_confirm['joint_success']} < C_min {_C_MIN}); the "
            f"stress<->tilt Pareto frontier over {len(front)} non-dominated placed+static configs "
            f"contains NO point in-band on both (closest normalised approach to the in-band box "
            f"{closest} > margin {_ADEQ_MARGIN}, plateau improvement {tail_improvement} <= "
            f"{_ADEQ_PLATEAU_EPS}). The matched Panda anchor sits in the box (stress_p90 "
            f"{matched['stress_p90']}, tilt_p90 {matched['tilt_p90']}, js "
            f"{matched['joint_success']}). The 6-DOF arm cannot trade off stress vs tilt "
            f"the way the 7-DOF Panda can -> EH "
            f"ESTABLISHED via a multi-objective infeasibility (the mechanistic embodiment finding)."
        )
    else:
        verdict = "INDETERMINATE"
        finding = (
            f"The search did not converge to a pre-committed verdict: null_met={null_met}, "
            f"inband_exists={inband_exists}, plateau={plateau}, density_ok={density_ok}, "
            f"margin_ok={margin_ok} (closest approach {closest}, frontier {len(front)} pts, best "
            f"confirm js {best_confirm['joint_success']}). Report honestly; collect more "
            f"configs/seeds before claiming either way."
        )

    artifact = {
        "schema": "cocarry_rung4e_taskfair_search/v1",
        "date": datetime.now(UTC).strftime("%Y-%m-%d"),
        "generator": "scripts/repro/_cocarry_rung4e_taskfair_search.py",
        "prereg": "spikes/results/cocarry/rung4e/cocarry_rung4e_taskfair_prereg.json",
        "fmax_coupling_n": _FMAX,
        "tilt_max_deg": _TILT_MAX,
        "goal_radius_m": _GOAL,
        "drive_stiffness_Npm": _K,
        "c_min": _C_MIN,
        "search_seeds": _SEARCH_SEEDS,
        "confirm_seeds": _CONFIRM_SEEDS,
        "panda_vcompliance": round(panda_vc, 3),
        "poses": poses,
        "search_configs": configs,
        "deficit_trace": deficit_trace,
        "best_config": {
            "pose": best["pose"],
            "base_x": best["base_x"],
            "q6": best["q6"],
            "vcomp": best["vcomp"],
            "level_gain": best["level_gain"],
            "admit_bias": best["admit_bias"],
            "search_agg": best["agg"],
            "confirm_agg": best_confirm,
        },
        "matched_panda_anchor": matched,
        "pareto_frontier": [
            {
                "pose": c["pose"],
                "level_gain": c["level_gain"],
                "admit_bias": c["admit_bias"],
                "stress_p90": c["agg"]["stress_p90"],
                "tilt_p90": c["agg"]["tilt_p90"],
                "joint_success": c["agg"]["joint_success"],
            }
            for c in front
        ],
        "adequacy": adequacy,
        "verdict": verdict,
        "finding": finding,
    }
    Path(out).write_text(json.dumps(artifact, sort_keys=True, indent=2), encoding="utf-8")
    print(f"\n    VERDICT: {verdict}\n    {finding}\n    artifact -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
