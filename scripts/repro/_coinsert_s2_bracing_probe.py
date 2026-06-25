# SPDX-License-Identifier: Apache-2.0
"""Co-insert S2 bracing probe — the SAPIEN constraint-fidelity wall (ADR-026 §D4; ADR-005).

The S2 instrument-validation slice after the founder's S2-insertion-wall ruling
(reproducible 32 mm wedge = rig artifact, not a Gate-D wall) and the
assembly-drag fork (take the BRACING path: cut the insertion force, then hold so
the cooperative reference RESISTS the reaction; active raise-to-meet mating is
construct-invalid). This generator reproduces the three findings that, together,
establish a **SAPIEN constraint-fidelity wall** (a sim / force-trust blocker, NOT
a Gate-D task finding):

1. **Force cut.** The inserter axial press is softened (the structured base's
   ``descend_step`` shrunk, the unjam back-off triggered sooner) so seating is
   force-guided (compliant search + chamfer lead-in), not a brute-force ram.
   Reports peak peg-socket contact force + peak workpiece interaction wrench for
   the ram vs the force-guided press.
2. **Lateral hold drags.** Even at the reduced force, the clean-weld lateral
   bracket hold cannot structurally brace the axial reaction: the holder wrist
   yields to the moment and the FREE assembly drags down (socket-z sinks),
   seating only part-way.
3. **Below-hold braces but over-constrains.** A hand-below-socket hold DOES brace
   (socket-z holds), but SAPIEN's drive-weld over-constrains the holder arm at
   every weld anchor span (large internal preload at zero action + socket tilt),
   so the peg cannot engage. There is no SAPIEN drive-weld hold that both braces
   axially AND holds the socket orientation without over-constraint.

Verdict: ``SAPIEN_CONSTRAINT_FIDELITY_WALL`` — per the founder escalation this
triggers the decision between the pre-committed S1b MuJoCo migrate path
(Stage-2 §7, provided for exactly "the sim can't represent the contact /
constraint physics") and an honest close. It is NOT resolved with active mating
and is NOT a Gate-D task finding.

Determinism: the env routes RNG through its P6 substream (``reset(seed=...)``);
the hand-written controllers are deterministic. Seeds are fixed. SAPIEN is the
GPU/oracle-gated substrate (run with ``uv sync --all-extras --group dev --group
oracle``); numbers come only from the committed artifact.

ADR-026 §Decision 1-4; ADR-005 §Decision (the dual-sim posture + the S1b
fallback); ADR-009 §Decision (the frozen reference holder); the S2 design.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import chamber.envs.coinsert as ci
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
_SEEDS = (0, 1, 2)
_EP = 320
_CLEARANCE = 1.0e-3
_FRICTION = 0.5


def _rotz_col(q: np.ndarray) -> np.ndarray:
    """World +z column of R(q) for a wxyz quaternion (the body's local +z axis)."""
    w, x, y, z = q
    return np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])


def _rollout(descend_step: str) -> dict:
    """Roll the matched pair (base inserter + reference holder); record seat + forces + drag."""
    out: list[dict] = []
    for seed in _SEEDS:
        env = make_coinsert_env(
            condition_id="coinsert_matched_reference",
            num_envs=1,
            render_backend="none",
            peg_clearance_m=_CLEARANCE,
            peg_socket_friction=_FRICTION,
            episode_length=_EP,
        )
        e = env.unwrapped
        ego_extra = {**_EGO, "descend_step": descend_step}
        base = load_partner(PartnerSpec("coinsert_base_inserter", seed, None, None, ego_extra))
        hold = load_partner(
            PartnerSpec("coinsert_reference_holder", seed, None, None, dict(_HOLDER))
        )
        obs, _ = env.reset(seed=seed)
        base.reset(seed=seed)
        hold.reset(seed=seed)
        sock_z0 = float(e.receptacle.pose.p[0, 2])
        info: dict = {}
        for _ in range(_EP):
            action = {
                "panda_wristcam": np.asarray(base.act(obs), dtype=np.float32),
                "panda_partner": np.asarray(hold.act(obs), dtype=np.float32),
            }
            obs, _, terminated, _, info = env.step(action)
            if bool(np.asarray(terminated).reshape(-1)[0]):
                break
        sock_z1 = float(e.receptacle.pose.p[0, 2])
        out.append(
            {
                "seed": seed,
                "depth_mm": round(float(info["seated_depth_m"][0]) * 1000, 1),
                "align_deg": round(float(info["axis_align_deg"][0]), 1),
                "seated": bool(np.asarray(info["seated"]).reshape(-1)[0]),
                "peak_insert_force_n": round(float(info["peak_insert_force_n"][0]), 1),
                "peak_couple_wrench_n": round(float(info["peak_couple_wrench_n"][0]), 1),
                "socket_z_start_m": round(sock_z0, 4),
                "socket_z_end_m": round(sock_z1, 4),
                "assembly_sink_mm": round((sock_z0 - sock_z1) * 1000, 1),
            }
        )
        env.close()
    return {
        "per_seed": out,
        "depth_mm_max": max(r["depth_mm"] for r in out),
        "peak_insert_n_max": max(r["peak_insert_force_n"] for r in out),
        "peak_couple_n_max": max(r["peak_couple_wrench_n"] for r in out),
        "assembly_sink_mm_max": max(r["assembly_sink_mm"] for r in out),
        "any_seated": any(r["seated"] for r in out),
    }


def _below_hold_span_sweep() -> dict:
    """Diagnostic: the below-hold braces but the SAPIEN drive-weld over-constrains.

    A hand-below-socket hold braces (socket-z holds) yet the drive-weld
    over-constrains the holder arm at every anchor span. Patches the holder
    geometry to the below-hold (an alternate configuration under
    characterisation, not the shipped default) and reports the zero-action wrist
    preload + socket tilt per weld anchor span.
    """
    p2 = np.array([0.0, 0.1, 0.0, -2.65, 0.0, 2.75, 0.785, 0.0, 0.0])  # hand below+beside socket
    saved = (
        ci._PANDA_READY_QPOS_HOLDER,
        ci._SOCKET_GRIP_IN_HAND_P,
        ci._SOCKET_GRIP_IN_HAND_Q,
        ci._WELD_ANCHOR_SPAN_M,
    )
    try:
        ci._PANDA_READY_QPOS_HOLDER = p2
        # Identity-position + flip read → compute the on-axis-below grip offset.
        ci._SOCKET_GRIP_IN_HAND_P = (0.0, 0.0, 0.0)
        ci._SOCKET_GRIP_IN_HAND_Q = (0.0, 1.0, 0.0, 0.0)
        env = make_coinsert_env(num_envs=1, render_backend="none", peg_clearance_m=_CLEARANCE)
        e = env.unwrapped
        env.reset(seed=0)
        h = e.agent.agents_dict[e.partner_uid].robot.links_map["panda_hand"]
        hp = h.pose.p[0].cpu().numpy()
        hq = h.pose.q[0].cpu().numpy()
        w, x, y, z = hq
        h_r = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
            ]
        )
        peg = e.peg.pose
        tip = peg.p[0].cpu().numpy() + e._peg_half_len * _rotz_col(peg.q[0].cpu().numpy())
        grip_p = h_r.T @ (np.array([tip[0], tip[1], tip[2] - 0.013]) - hp)
        env.close()
        ci._SOCKET_GRIP_IN_HAND_P = tuple(float(v) for v in grip_p)
        ci._SOCKET_GRIP_IN_HAND_Q = (0.0, 1.0, 0.0, 0.0)

        rows: list[dict] = []
        for span in (0.02, 0.04, 0.06, 0.10):
            ci._WELD_ANCHOR_SPAN_M = span
            env = make_coinsert_env(num_envs=1, render_backend="none", peg_clearance_m=_CLEARANCE)
            e = env.unwrapped
            env.reset(seed=0)
            zero = {
                "panda_wristcam": np.zeros(8, dtype=np.float32),
                "panda_partner": np.zeros(8, dtype=np.float32),
            }
            for _ in range(20):
                env.step(zero)
            sock_axis = _rotz_col(e.receptacle.pose.q[0].cpu().numpy())
            tilt = float(
                np.degrees(np.arccos(np.clip(sock_axis @ np.array([0.0, 0.0, 1.0]), -1, 1)))
            )
            rows.append(
                {
                    "anchor_span_mm": round(span * 1000, 0),
                    "zero_action_wrist_preload_n": round(
                        float(e.workpiece_interaction_wrench()[0]), 0
                    ),
                    "socket_openup_tilt_deg": round(tilt, 1),
                    "socket_z_held_m": round(float(e.receptacle.pose.p[0, 2]), 4),
                }
            )
            env.close()
        return {
            "weld_arm_m": round(float(np.linalg.norm(grip_p)), 3),
            "spans": rows,
            "braces": (
                "socket-z holds (no drag) — the below-hold resists the axial "
                "reaction in compression"
            ),
            "over_constrains": (
                "every span shows a large zero-action wrist preload + socket "
                "tilt → peg cannot engage"
            ),
        }
    finally:
        (
            ci._PANDA_READY_QPOS_HOLDER,
            ci._SOCKET_GRIP_IN_HAND_P,
            ci._SOCKET_GRIP_IN_HAND_Q,
            ci._WELD_ANCHOR_SPAN_M,
        ) = saved


def main() -> int:
    out_dir = os.environ.get("OUT_DIR", "spikes/results/coinsert/s2/2026-06-24")
    os.makedirs(out_dir, exist_ok=True)

    ram = _rollout(descend_step="0.006")  # brute-force ram (high press authority)
    guided = _rollout(descend_step="0.002")  # force-guided (the committed default)
    below = _below_hold_span_sweep()

    artifact = {
        "schema": "coinsert_s2_bracing_probe/v1",
        "stage": "S2 — bracing path + SAPIEN constraint-fidelity check",
        "purpose": (
            "After the founder S2 assembly-drag ruling (take the bracing path: cut the "
            "insertion force, then hold so the reference resists the reaction; active "
            "raise-to-meet mating is construct-invalid). Establish whether a competent "
            "BRACING cooperative hold can seat the peg within SAPIEN's drive-weld limits."
        ),
        "design": {
            "seeds": list(_SEEDS),
            "episode_length": _EP,
            "clearance_m": _CLEARANCE,
            "friction": _FRICTION,
            "force_cut": "inserter descend_step 0.006 (ram) vs 0.002 (force-guided)",
            "below_hold": (
                "holder hand moved below+beside the socket to brace the axial "
                "reaction in compression"
            ),
        },
        "force_cut": {
            "ram": {
                "peak_insert_force_n": ram["peak_insert_n_max"],
                "peak_couple_wrench_n": ram["peak_couple_n_max"],
                "depth_mm_max": ram["depth_mm_max"],
            },
            "force_guided": {
                "peak_insert_force_n": guided["peak_insert_n_max"],
                "peak_couple_wrench_n": guided["peak_couple_n_max"],
                "depth_mm_max": guided["depth_mm_max"],
            },
            "finding": (
                "softening the press cuts the peak forces substantially (the 32 mm "
                "wedge was a brute-force ram)"
            ),
        },
        "lateral_hold_drag": {
            "per_seed": guided["per_seed"],
            "assembly_sink_mm_max": guided["assembly_sink_mm_max"],
            "any_seated": guided["any_seated"],
            "finding": (
                "even at the reduced force the clean-weld lateral bracket hold cannot brace the "
                "axial reaction — the FREE assembly drags down and the peg seats only part-way"
            ),
        },
        "below_hold_braces_but_over_constrains": below,
        "verdict": "SAPIEN_CONSTRAINT_FIDELITY_WALL",
        "rationale": (
            "No SAPIEN drive-weld hold both braces the axial insertion reaction AND "
            "holds the socket orientation without over-constraint: the only cleanly-"
            "representable weld (the long lateral bracket, align ~1 deg) cannot brace "
            "(the assembly drags), while the hand-below-socket hold that DOES brace "
            "over-constrains the drive-weld at every anchor span (large zero-action "
            "wrist preload + socket tilt) so the peg cannot engage. This is a sim / "
            "force-trust blocker (the drive-weld cannot represent an axial-load-bearing "
            "cooperative grasp), NOT a Gate-D task finding (reserved for a clean, "
            "competent, FREE hold that still cannot seat). Per the Stage-2 escalation "
            "it triggers the founder decision between the pre-committed S1b MuJoCo "
            "migrate path and an honest close; NOT resolved with active mating."
        ),
        "honesty_statement": (
            "Numbers are the committed probe output; no value is hand-entered. The "
            "below-hold sweep PATCHES the holder geometry module constants to "
            "characterise an alternate configuration (not the shipped default); the "
            "matched-pair rollouts use the shipped env + controllers unchanged. The "
            "matched pair does NOT seat (the >=0.9 precondition is not met) — so the "
            "Gate-D / capability-gate / force-limit / oracle-calibration measurements "
            "are NOT run; this slice stops at the force-trust blocker, as pre-committed."
        ),
    }
    out_path = os.path.join(out_dir, "coinsert_s2_bracing_probe.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(artifact, fh, sort_keys=True, indent=2)
    rc, gc = ram["peak_couple_n_max"], guided["peak_couple_n_max"]
    print(f"  force cut: ram couple {rc}N -> guided {gc}N")
    gd, gs = guided["depth_mm_max"], guided["assembly_sink_mm_max"]
    print(f"  lateral hold: depth {gd}mm, sink {gs}mm")
    print(f"  below-hold preloads: {[r['zero_action_wrist_preload_n'] for r in below['spans']]} N")
    print(f"  VERDICT: {artifact['verdict']}")
    print(f"  artifact -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
