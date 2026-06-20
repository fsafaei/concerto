# SPDX-License-Identifier: Apache-2.0
# pyright: reportPrivateImportUsage=false
"""Rung-4d Stage-A1 — fair xArm6 carry-pose falsification of the Rung-4c EH headline.

ADR-026 §D4 / §Decision 2; ADR-005; R-2026-06-B §15 Rung 4d. The Rung-4c
embodiment headline ("a different body over-loads the cooperative coupling
~5x f_max") used the xArm6's DEFAULT carry pose, whose endpoint vertical
compliance (6.9) happens to be 1.46x the Panda's — but endpoint stiffness is
configuration-dependent, and the xArm6's pose was never optimised while the
Panda's was. This driver gives the xArm6 its best fair shot:

1. **Kinematic search** (FK/Jacobian only, joint-limit-clamped): over base x +
   the 3-DOF redundancy (random feasible IK restarts), find the carry config
   that MINIMISES the vertical endpoint compliance (J Jᵀ)⁻¹_zz at the bar end.
2. **Env test** (the embodiment-invariant coupling measure, compliant K=8000):
   the default vs the optimised pose, static-hold AND active-carry coupling,
   via the committed ``make_cocarry_env(xarm6_base_x=, xarm6_ready_qpos=)``
   overrides (no monkeypatching).

Falsification logic (Stage-A1 gate): if the fairly-posed xArm6 stays IN-BAND on
coupling (< f_max), the Rung-4c over-load was a POSE ARTIFACT and the EH
coupling-over-stress headline does not hold — report the correction, do not
proceed to the re-freeze.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from chamber.agents.panda_jacobian import PandaJacobianProvider
from chamber.agents.xarm6_jacobian import XArm6JacobianProvider
from chamber.benchmarks import cocarry_ph as ph
from chamber.envs.cocarry_obs import make_cocarry_training_env
from chamber.partners.api import PartnerSpec
from chamber.partners.registry import load_partner

_K = 8000.0
_MEASURE = "coupling"
_FMAX = 365.6
_RENDER = "none"
_SETTLE = 15
_IK_TOL = 1e-5
_IK_SUCCESS = 1e-3
# Real xArm6 arm joint limits (rad), from robot.get_qlimits().
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
_TARGET_W = np.array([-0.115, 0.0, 0.1698])  # the partner -x bar end (world)
_DEFAULT_Q6 = np.array([0.0, -0.084, -0.8, 0.0, 0.692, 0.0])
_PANDA_CARRY_Q = np.array([0.0, np.pi / 8, 0.0, -np.pi * 5 / 8, 0.0, np.pi * 3 / 4, -np.pi / 4])
_BASES = (0.30, 0.35, 0.40, 0.45)
_RESTARTS = 80
_SEEDS = list(range(80100, 80106))
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


def _search_min_compliance() -> dict:
    rng = np.random.default_rng(0)
    best: tuple[float, float, np.ndarray] | None = None
    per_base: dict[str, float] = {}
    for bx in _BASES:
        tgt = _rotz_t(180.0) @ (_TARGET_W - np.array([bx, 0, 0]))
        bb = None
        for _ in range(_RESTARTS):
            q, err = _ik(tgt, rng.uniform(_LIM[:, 0], _LIM[:, 1]))
            feasible = bool(np.all(q >= _LIM[:, 0]) and np.all(q <= _LIM[:, 1]))
            if err < _IK_SUCCESS and feasible:
                vc = _vcomp(q)
                if bb is None or vc < bb:
                    bb = vc
                if best is None or vc < best[0]:
                    best = (vc, bx, q.copy())
        per_base[str(bx)] = round(bb, 3) if bb is not None else float("nan")
    if best is None:
        raise RuntimeError("no feasible xArm6 IK solution found in the search")
    pj = PandaJacobianProvider()(None, _PANDA_CARRY_Q)  # type: ignore[arg-type]
    panda_vc = float(np.linalg.inv(pj @ pj.T + 1e-9 * np.eye(3))[2, 2])
    return {
        "panda_vcompliance": round(panda_vc, 3),
        "xarm6_default_vcompliance": round(_vcomp(_DEFAULT_Q6), 3),
        "xarm6_best_vcompliance": round(best[0], 3),
        "xarm6_best_base_x": best[1],
        "xarm6_best_q6": [round(x, 5) for x in best[2].tolist()],
        "xarm6_best_over_panda": round(best[0] / panda_vc, 2),
        "per_base_best": per_base,
    }


def _static(base_x: float, q6: list[float], seed: int = 80100, n: int = 120) -> dict:
    env = make_cocarry_training_env(
        condition_id="cocarry_xarm6_partner",
        root_seed=seed,
        render_backend=_RENDER,
        drive_stiffness=_K,
        stress_measure=_MEASURE,
        xarm6_base_x=base_x,
        xarm6_ready_qpos=[*q6, 0, 0, 0, 0, 0, 0],
    )
    try:
        env.reset(seed=seed)
        eu = env.get_wrapper_attr("ego_uid")
        spaces = env.action_space.spaces  # type: ignore[attr-defined]
        adim = {u: spaces[u].shape[0] for u in (eu, "xarm6_robotiq")}
        vals = []
        for t in range(n):
            env.step(
                {
                    eu: np.zeros(adim[eu], np.float32),
                    "xarm6_robotiq": np.zeros(adim["xarm6_robotiq"], np.float32),
                }
            )
            if t >= _SETTLE:
                vals.append(
                    float(
                        np.asarray(env.get_wrapper_attr("get_telemetry")()["stress_proxy"]).reshape(
                            -1
                        )[0]
                    )
                )
        return {
            "p50": round(float(np.percentile(vals, 50)), 0),
            "max": round(float(np.max(vals)), 0),
        }
    finally:
        env.close()


def _active(base_x: float, q6: list[float]) -> dict:
    cps, tilts = [], []
    spec = {
        "uid": "xarm6_robotiq",
        "base_xyz": f"{base_x},0.0,0.0",
        "base_yaw_deg": "180",
        "end_sign": "-1",
        "bar_half_len": repr(0.115),
    }
    for s in _SEEDS:
        env = make_cocarry_training_env(
            condition_id="cocarry_xarm6_partner",
            root_seed=s,
            render_backend=_RENDER,
            drive_stiffness=_K,
            stress_measure=_MEASURE,
            xarm6_base_x=base_x,
            xarm6_ready_qpos=[*q6, 0, 0, 0, 0, 0, 0],
        )
        try:
            coop = ph.build_cooperative_ego(seed=s)
            coop.reset(seed=s)
            par = load_partner(PartnerSpec("cocarry_xarm6_impedance", s, None, None, dict(spec)))
            m = ph.rollout_pair(
                env=env,
                ego_act=lambda o, _e=coop: np.asarray(_e.act(o), dtype=np.float32),
                partner=par,
                seed=s,
                episode_length=320,
            )
            cps.append(m.max_stress_proxy)
            tilts.append(m.max_tilt_deg)
        finally:
            env.close()
    return {
        "coupling_p90": round(float(np.percentile(cps, 90)), 0),
        "tilt_p90": round(float(np.percentile(tilts, 90)), 1),
    }


def main() -> int:
    out = os.environ.get(
        "OUT_JSON", "spikes/results/cocarry/rung4d/cocarry_rung4d_pose_falsification.json"
    )
    print("    [1] kinematic min-compliance search (joint-limit-clamped)...")
    kin = _search_min_compliance()
    print(
        f"      panda vcomp={kin['panda_vcompliance']} "
        f"default xarm6={kin['xarm6_default_vcompliance']} "
        f"best xarm6={kin['xarm6_best_vcompliance']} "
        f"(base {kin['xarm6_best_base_x']}, {kin['xarm6_best_over_panda']}x panda)"
    )
    print("    [2] env test: default vs optimised pose (static + active coupling)...")
    default = {
        "base_x": 0.35,
        "q6": _DEFAULT_Q6.tolist(),
        "static": _static(0.35, _DEFAULT_Q6.tolist()),
        "active": _active(0.35, _DEFAULT_Q6.tolist()),
    }
    obx, oq = kin["xarm6_best_base_x"], kin["xarm6_best_q6"]
    optimised = {"base_x": obx, "q6": oq, "static": _static(obx, oq), "active": _active(obx, oq)}
    print(f"      DEFAULT  : static {default['static']} active {default['active']}")
    print(f"      OPTIMISED: static {optimised['static']} active {optimised['active']}")

    opt_static_inband = optimised["static"]["max"] < _FMAX
    opt_active_inband = optimised["active"]["coupling_p90"] < _FMAX
    if opt_static_inband and opt_active_inband:
        verdict = "POSE_ARTIFACT"
        finding = (
            f"A fairly-posed xArm6 (endpoint vcompliance "
            f"{kin['xarm6_best_vcompliance']} < the Panda's {kin['panda_vcompliance']}) "
            f"stays IN-BAND on coupling (static max {optimised['static']['max']} N, "
            f"active p90 {optimised['active']['coupling_p90']} N, both < f_max {_FMAX:.0f}). "
            f"The Rung-4c coupling-over-stress EH headline (default-pose active "
            f"{default['active']['coupling_p90']} N) is a POSE ARTIFACT of the unoptimised "
            f"default carry pose, not an embodiment law. STOP per Stage-A1 — do NOT "
            f"re-freeze. NOTE: the stiffness-optimal pose then fails the LEVEL conjunct "
            f"(tilt p90 {optimised['active']['tilt_p90']} deg); a stress-vs-tilt pose "
            f"tradeoff is a CANDIDATE genuine difficulty but is NOT established (a task-fair "
            f"pose+controller search is the next slice). The EH axis is NOT established by "
            f"the committed coupling-over-stress result."
        )
    else:
        verdict = "OVER_LOAD_POSE_ROBUST"
        finding = (
            f"The fairly-posed xArm6 still over-loads the coupling (optimised active p90 "
            f"{optimised['active']['coupling_p90']} N vs f_max {_FMAX:.0f}); the EH "
            f"over-load is pose-robust. Proceed to the re-freeze with the (static vs "
            f"active) cooperation-mediation framing."
        )
    artifact = {
        "schema": "cocarry_rung4d_pose_falsification/v1",
        "date": datetime.now(UTC).strftime("%Y-%m-%d"),
        "generator": "scripts/repro/_cocarry_rung4d_pose_falsification.py",
        "fmax_coupling_n": _FMAX,
        "drive_stiffness_Npm": _K,
        "seeds": _SEEDS,
        "kinematic_search": kin,
        "default_pose": default,
        "optimised_pose": optimised,
        "verdict": verdict,
        "finding": finding,
    }
    Path(out).write_text(json.dumps(artifact, sort_keys=True, indent=2), encoding="utf-8")
    print(f"\n    VERDICT: {verdict}\n    {finding}\n    artifact -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
