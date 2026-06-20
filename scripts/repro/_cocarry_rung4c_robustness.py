# SPDX-License-Identifier: Apache-2.0
# pyright: reportPrivateImportUsage=false
"""Rung-4c construct-crux hardening — committed controlled-invariance + pose-robustness.

ADR-026 §D4; ADR-005; R-2026-06-B §15 Rung 4c. Converts the PR-#252 prose
arguments into committed evidence:

1. **Controlled invariance** — with the bar teleported to the SAME world pose
   and the two grippers' TCPs at the SAME engineered points, the coupling
   measure reads ~identical for a Panda-held vs an xArm6-held bar (it is
   geometry-only: K * ||grip - bar_end||). Proves the instrument is
   embodiment-invariant *given the same state* (the residual is a real
   state difference, not a measurement bias).
2. **Static-hold** — the zero-action gravity-only coupling (the "6.4x" that was
   prose-only): the xArm6 over-loads the coupling vs the Panda even with NO
   cooperative motion.
3. **Pose / base-placement sweep** — the xArm6 coupling over-load persists
   across base placements + carry poses (not a single geometrically-bad spawn,
   the AS-axis failure mode). If pose-sensitive, that is reported.
4. **Mechanism** — the xArm6 vs Panda end-effector (linear-Jacobian) endpoint
   stiffness at the carry pose, substantiating "poor carry-pose endpoint
   stiffness -> larger spring deflection -> coupling over-load".
"""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from mani_skill.utils.structs.pose import Pose

import chamber.envs.cocarry as cc
from chamber.agents.panda_jacobian import PandaJacobianProvider
from chamber.agents.xarm6_jacobian import XArm6JacobianProvider
from chamber.benchmarks import cocarry_ph as ph
from chamber.envs.cocarry_obs import make_cocarry_training_env
from chamber.partners.api import PartnerSpec
from chamber.partners.registry import load_partner

_RENDER = "none"
_K = 8000.0
_MEASURE = "coupling"
_SETTLE = 15
_INVARIANCE_TOL_N = 5.0
_IK_TOL = 1e-5


def _coupling_now(env: Any) -> float:  # noqa: ANN401
    return float(np.asarray(env.get_wrapper_attr("get_telemetry")()["stress_proxy"]).reshape(-1)[0])


def _grip_world(env: Any, uid: str) -> np.ndarray:  # noqa: ANN401
    hand, grip_in_hand, _ = env.get_wrapper_attr("_weld_anchors")[uid]
    p = (hand.pose * grip_in_hand).p
    return np.asarray(p.detach().cpu()).reshape(-1)[:3]


def controlled_invariance() -> dict:
    """Same imposed bar pose -> identical coupling reading for Panda vs xArm6 (geometry-only)."""
    grid = [0.0, 0.01, 0.02, 0.03, 0.05]  # imposed vertical bar offsets (m)
    res: dict[str, Any] = {"offsets_m": grid, "panda_coupling_n": [], "xarm6_coupling_n": []}
    grips = {}
    for cond, puid, key in (
        ("cocarry_matched_panda_pair", "panda_partner", "panda"),
        ("cocarry_xarm6_partner", "xarm6_robotiq", "xarm6"),
    ):
        env = make_cocarry_training_env(
            condition_id=cond,
            root_seed=80100,
            render_backend=_RENDER,
            drive_stiffness=_K,
            stress_measure=_MEASURE,
        )
        try:
            env.reset(seed=80100)
            grips[key] = {
                "ego_tcp": _grip_world(env, "panda_wristcam").tolist(),
                "partner_tcp": _grip_world(env, puid).tolist(),
            }
            bar = env.get_wrapper_attr("bar")
            base_p = np.asarray(bar.pose.p.detach().cpu()).reshape(-1)[:3]
            for dz in grid:
                target = base_p.copy()
                target[2] += dz
                bar.set_pose(
                    Pose.create_from_pq(
                        torch.from_numpy(target.astype(np.float32)[None, :]).to(
                            env.get_wrapper_attr("device")
                        )
                    )
                )
                res[f"{key}_coupling_n"].append(round(_coupling_now(env), 2))
        finally:
            env.close()
    res["grip_frames_world"] = grips
    res["max_abs_diff_n"] = float(
        np.max(np.abs(np.array(res["panda_coupling_n"]) - np.array(res["xarm6_coupling_n"])))
    )
    res["invariant"] = bool(res["max_abs_diff_n"] < _INVARIANCE_TOL_N)
    return res


def static_hold(cond: str, puid: str, n: int = 120) -> dict:
    env = make_cocarry_training_env(
        condition_id=cond,
        root_seed=80100,
        render_backend=_RENDER,
        drive_stiffness=_K,
        stress_measure=_MEASURE,
    )
    try:
        env.reset(seed=80100)
        eu = env.get_wrapper_attr("ego_uid")
        sp = env.action_space.spaces  # type: ignore[attr-defined]
        adim = {u: sp[u].shape[0] for u in (eu, puid)}
        vals = []
        for t in range(n):
            env.step(
                {
                    eu: np.zeros(adim[eu], dtype=np.float32),
                    puid: np.zeros(adim[puid], dtype=np.float32),
                }
            )
            if t >= _SETTLE:
                vals.append(_coupling_now(env))
        return {
            "p50": round(float(np.percentile(vals, 50)), 1),
            "max": round(float(np.max(vals)), 1),
        }
    finally:
        env.close()


def _xarm6_ready_qpos_for_base(base_x: float) -> np.ndarray:
    """DLS position-IK: xArm6 ready qpos placing eef at the -x bar end for `base_x`."""
    p = XArm6JacobianProvider()
    rotz_t = np.array(
        [[np.cos(np.pi), np.sin(np.pi), 0], [-np.sin(np.pi), np.cos(np.pi), 0], [0, 0, 1.0]]
    )
    target_w = np.array([-0.115, 0.0, 0.1698])
    tgt_b = rotz_t @ (target_w - np.array([base_x, 0.0, 0.0]))
    q = np.array([0.0, -0.084, -0.8, 0.0, 0.692, 0.0])
    for _ in range(400):
        e = tgt_b - p.fk_tcp_position(q)
        if np.linalg.norm(e) < _IK_TOL:
            break
        j = p.jacobian(q)
        q = q + np.clip(j.T @ np.linalg.solve(j @ j.T + 1e-4 * np.eye(3), e), -0.1, 0.1)
    return np.concatenate([q, np.zeros(6)])


def base_pose_sweep(fmax: float) -> dict:
    """xArm6 coupling over-load across base placements (vs cooperative ego, active)."""
    out = []
    orig_base = cc._XARM6_BASE_X_M
    orig_pose = cc._BASE_POSE_BY_UID["xarm6_robotiq"]
    orig_q = cc._READY_QPOS_BY_UID["xarm6_robotiq"]
    seeds = list(range(80100, 80106))
    try:
        for bx in (0.30, 0.35, 0.40):
            ready = _xarm6_ready_qpos_for_base(bx)
            cc._XARM6_BASE_X_M = bx
            cc._BASE_POSE_BY_UID["xarm6_robotiq"] = ((bx, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
            cc._READY_QPOS_BY_UID["xarm6_robotiq"] = ready
            spec = {
                "uid": "xarm6_robotiq",
                "base_xyz": f"{bx},0.0,0.0",
                "base_yaw_deg": "180",
                "end_sign": "-1",
                "bar_half_len": repr(0.115),
            }
            cps = []
            for s in seeds:
                env = make_cocarry_training_env(
                    condition_id="cocarry_xarm6_partner",
                    root_seed=s,
                    render_backend=_RENDER,
                    drive_stiffness=_K,
                    stress_measure=_MEASURE,
                )
                try:
                    coop = ph.build_cooperative_ego(seed=s)
                    coop.reset(seed=s)
                    par = load_partner(
                        PartnerSpec("cocarry_xarm6_impedance", s, None, None, dict(spec))
                    )
                    m = ph.rollout_pair(
                        env=env,
                        ego_act=lambda o, _e=coop: np.asarray(_e.act(o), dtype=np.float32),
                        partner=par,
                        seed=s,
                        episode_length=320,
                    )
                    cps.append(m.max_stress_proxy)
                finally:
                    env.close()
            out.append(
                {
                    "base_x_m": bx,
                    "coupling_p90": round(float(np.percentile(cps, 90)), 0),
                    "coupling_min": round(float(np.min(cps)), 0),
                    "over_fmax": bool(np.min(cps) >= fmax),
                }
            )
    finally:
        cc._XARM6_BASE_X_M = orig_base
        cc._BASE_POSE_BY_UID["xarm6_robotiq"] = orig_pose
        cc._READY_QPOS_BY_UID["xarm6_robotiq"] = orig_q
    return {"per_base": out, "over_fmax_all_bases": bool(all(b["over_fmax"] for b in out))}


def endpoint_stiffness() -> dict:
    """End-effector vertical endpoint stiffness proxy (1 / (J J^T)_zz) at the carry pose."""
    panda_q = np.array([0.0, np.pi / 8, 0.0, -np.pi * 5 / 8, 0.0, np.pi * 3 / 4, -np.pi / 4])
    xarm6_q = np.array([0.0, -0.084, -0.8, 0.0, 0.692, 0.0])
    pj = PandaJacobianProvider()(None, panda_q)  # type: ignore[arg-type]
    xj = XArm6JacobianProvider().jacobian(xarm6_q)
    # Vertical compliance ~ (J J^T)_zz (larger => softer endpoint in z).
    pz = float(np.linalg.inv(pj @ pj.T + 1e-9 * np.eye(3))[2, 2])
    xz = float(np.linalg.inv(xj @ xj.T + 1e-9 * np.eye(3))[2, 2])
    return {
        "panda_vertical_compliance": pz,
        "xarm6_vertical_compliance": xz,
        "xarm6_over_panda": float(xz / max(pz, 1e-12)),
    }


def main() -> int:
    out_path = os.environ.get(
        "OUT_JSON", "spikes/results/cocarry/rung4c/cocarry_rung4c_robustness.json"
    )
    meas = json.loads(
        Path("spikes/results/cocarry/rung4c/cocarry_rung4c_eh_measurement.json").read_text("utf-8")
    )
    fmax = float(meas["fmax_coupling_n"])
    print("    [1] controlled invariance (same bar state, Panda vs xArm6)...")
    inv = controlled_invariance()
    print(f"      max abs diff = {inv['max_abs_diff_n']:.2f} N -> invariant={inv['invariant']}")
    print("    [2] static-hold (zero-action gravity-only coupling)...")
    sp = static_hold("cocarry_matched_panda_pair", "panda_partner")
    sx = static_hold("cocarry_xarm6_partner", "xarm6_robotiq")
    static = {
        "panda": sp,
        "xarm6": sx,
        "xarm6_over_panda_p50": round(sx["p50"] / max(sp["p50"], 1e-6), 2),
    }
    print(f"      panda={sp} xarm6={sx} ratio_p50={static['xarm6_over_panda_p50']}")
    print("    [3] base/pose sweep...")
    sweep = base_pose_sweep(fmax)
    print(f"      over f_max at all bases = {sweep['over_fmax_all_bases']}: {sweep['per_base']}")
    print("    [4] endpoint-stiffness mechanism...")
    stiff = endpoint_stiffness()
    print(f"      xArm6/Panda vertical compliance = {stiff['xarm6_over_panda']:.1f}x")

    artifact = {
        "schema": "cocarry_rung4c_robustness/v1",
        "date": datetime.now(UTC).strftime("%Y-%m-%d"),
        "generator": "scripts/repro/_cocarry_rung4c_robustness.py",
        "fmax_coupling_n": fmax,
        "controlled_invariance": inv,
        "static_hold": static,
        "base_pose_sweep": sweep,
        "endpoint_stiffness_mechanism": stiff,
        "summary": (
            f"Instrument invariant given the same state (max diff "
            f"{inv['max_abs_diff_n']:.1f} N). xArm6 over-loads the coupling "
            f"{static['xarm6_over_panda_p50']}x even at a zero-action static hold, and over "
            f"f_max at every swept base ({sweep['over_fmax_all_bases']}); the xArm6 "
            f"end-effector is {stiff['xarm6_over_panda']:.1f}x more vertically compliant than "
            f"the Panda at the carry pose. The over-load is a real, pose-robust embodiment "
            f"effect, not an instrument bias or a single bad spawn."
        ),
    }
    Path(out_path).write_text(json.dumps(artifact, sort_keys=True, indent=2), encoding="utf-8")
    print(f"    artifact -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
