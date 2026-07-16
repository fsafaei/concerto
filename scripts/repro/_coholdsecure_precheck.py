# SPDX-License-Identifier: Apache-2.0
"""Co-hold-secure PR-A engineering precheck driver (ADR-029 §Decision; non-registered).

Runs the pre-stated, **non-gating** solvability / coupling-liveness /
stress-channel precheck whose rule is committed in
``spikes/results/coholdsecure/precheck-2026-07-16/PRECHECK_PRESTATEMENT.md``
BEFORE any measured episode (rule-before-result; the #298/#309 pattern — no
prereg tag is created or rotated, PR-B owns that). Verdict: PROCEED iff
P1 ∧ P2 ∧ P4; STOP otherwise.

Cells (all deterministic — fixed seeds, P6 substreams, no wall-clock RNG):

- **P1 solvability**: C-matched (base securer + cooperative reference holder)
  on the three clearance cells x seeds {0, 1, 2} at the 40 N detent anchor.
- **P2 coupling-liveness**: C-limp (zero-action holder, grasp maintained) on
  every cell, and C-none (no holder; part on the passive stand) on every cell.
- **P3 fixture pre-read** (reported, not gating): C-fixture per-cell outcomes
  and wrench beside C-matched.
- **P4 stress-channel liveness**: the low-authority creep press on the middle
  cell across the detent sweep {20, 40, 60} N; statistic = p90 of the holder
  workpiece-wrench over detent-active steps.

Usage::

    uv run python scripts/repro/_coholdsecure_precheck.py            # run + write JSON
    uv run python scripts/repro/_coholdsecure_precheck.py --render   # tables from JSON

The run rewrites ``precheck_results.json`` byte-identically (P6 / ADR-002).
GPU/Vulkan host required (SAPIEN); CI cannot see this — see REPRO.txt.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from chamber.envs.co_hold_secure import (
    COHOLD_CLEARANCE_SIDE_SET_M,
    COHOLD_DETENT_FORCE_N,
    COHOLD_DETENT_TRAVEL_M,
    COHOLD_ENGAGE_DEPTH_M,
    COHOLD_POSE_HELD_TRANS_TOL_M,
    make_co_hold_secure_env,
)
from chamber.partners.api import PartnerSpec
from chamber.partners.registry import load_partner

_ARCHIVE = Path("spikes/results/coholdsecure/precheck-2026-07-16")
_RESULTS_JSON = _ARCHIVE / "precheck_results.json"

_EPISODE_STEPS = 320
_SEEDS = (0, 1, 2)
_CELLS_MM = tuple(round(c * 1000, 1) for c in COHOLD_CLEARANCE_SIDE_SET_M)
_MIDDLE_CELL_M = COHOLD_CLEARANCE_SIDE_SET_M[1]
_DETENT_SWEEP_N = (20.0, 40.0, 60.0)

#: P4 artifact bound: 3x the working force limit, pre-stated as 3x the largest
#: sweep setting (the S2 over-constraint lesson was 573-1306 N phantom
#: preloads against ~40-90 N working forces).
_P4_WRENCH_BOUND_N = 3.0 * max(_DETENT_SWEEP_N)

#: Frozen driver seat configs (PR-A bring-up; gains are instrument tuning —
#: free before the Gate-0 prereg — the geometry and the rule are not).
_EGO_PRESS = {
    "uid": "panda_wristcam",
    "base_xyz": "-0.62,0,0",
    "base_yaw_deg": "0",
    "peg_half_len": "0.02",
    "depth_target": str(COHOLD_ENGAGE_DEPTH_M),
    "descend_step": "0.001",
    "step_max": "0.025",
    "gripper_action": "1.0",
    "stall_steps": "100000",
    "seat_eps": "0.0003",
    "press_ratchet": "1",
}
#: P4 creep press: low authority so the ramp traversal is quasi-static and the
#: wrench samples near-equilibrium (many detent-active steps).
_EGO_CREEP = {**_EGO_PRESS, "descend_step": "0.0005", "step_max": "0.008"}
_HOLDER_REF = {
    "uid": "panda_partner",
    "base_xyz": "0.62,0,0",
    "base_yaw_deg": "180",
    "peg_half_len": "0.02",
    "gripper_action": "1.0",
    "kp_axial": "2.5",
    "kp_lateral": "2.5",
    "ori_hold_gain": "3.5",
}


def _holder_for(control: str, seed: int) -> Any:
    """Holder-seat driver: cooperative reference (matched) or zero-action."""
    if control == "matched":
        return load_partner(
            PartnerSpec("coinsert_reference_holder", seed, None, None, dict(_HOLDER_REF))
        )
    return load_partner(
        PartnerSpec("partner_ablated_zero", seed, None, None, {"action_dim": "8"})
    )


def _episode(
    *, control: str, clearance_m: float, detent_n: float, seed: int, creep: bool
) -> dict[str, Any]:
    """One deterministic episode; returns the per-episode record."""
    env: Any = make_co_hold_secure_env(
        control=control,
        clearance_side_m=clearance_m,
        detent_force_n=detent_n,
        render_backend="none",
        root_seed=seed,
    )
    ego_extra = _EGO_CREEP if creep else _EGO_PRESS
    base = load_partner(PartnerSpec("coinsert_base_inserter", seed, None, None, dict(ego_extra)))
    hold = _holder_for(control, seed)
    try:
        obs, _ = env.reset(seed=seed)
        base.reset(seed=seed)
        hold.reset(seed=seed)
        entry = COHOLD_ENGAGE_DEPTH_M - COHOLD_DETENT_TRAVEL_M
        wrench_active: list[float] = []
        info: dict[str, Any] = {}
        for _ in range(_EPISODE_STEPS):
            ego_a = np.asarray(base.act(obs), dtype=np.float32)
            par_a = np.asarray(hold.act(obs), dtype=np.float32)
            obs, _r, _te, _tr, info = env.step(
                {"panda_wristcam": ego_a, "panda_partner": par_a}
            )
            depth = float(info["seated_depth_m"][0])
            latched = bool(env._detent_latched[0]) if env._detent_latched else False  # noqa: SLF001
            if entry - 0.0005 <= depth and not latched:
                wrench_active.append(float(env.holder_workpiece_wrench()[0]))

        def _f(key: str) -> float:
            return float(np.asarray(info[key]).reshape(-1)[0])

        def _b(key: str) -> bool:
            return bool(np.asarray(info[key]).reshape(-1)[0])

        return {
            "control": control,
            "clearance_side_mm": round(clearance_m * 1000, 1),
            "detent_force_n": detent_n,
            "seed": seed,
            "creep_probe": creep,
            "success": _b("success"),
            "seated": _b("seated"),
            "pose_held": _b("pose_held"),
            "within_force": _b("within_force"),
            "both_static": _b("both_static"),
            "settled": _b("settled"),
            "seated_depth_mm": round(_f("seated_depth_m") * 1000, 2),
            "axis_align_deg": round(_f("axis_align_deg"), 2),
            "pose_excursion_mm": round(_f("pose_excursion_m") * 1000, 2),
            "pose_tilt_deg": round(_f("pose_tilt_deg"), 2),
            "peak_secure_force_n": round(_f("peak_secure_force_n"), 1),
            "peak_couple_wrench_n": round(_f("peak_couple_wrench_n"), 1),
            "detent_active_steps": len(wrench_active),
            "wrench_active_p90_n": (
                round(float(np.percentile(wrench_active, 90)), 2) if wrench_active else 0.0
            ),
            "wrench_active_max_n": (
                round(max(wrench_active), 2) if wrench_active else 0.0
            ),
            "all_channels_finite": bool(
                np.isfinite(
                    [
                        _f("seated_depth_m"),
                        _f("axis_align_deg"),
                        _f("pose_excursion_m"),
                        _f("pose_tilt_deg"),
                        _f("peak_secure_force_n"),
                        _f("peak_couple_wrench_n"),
                    ]
                ).all()
            ),
        }
    finally:
        env.close()


def _cell_success(records: list[dict[str, Any]], control: str, cell_mm: float) -> list[bool]:
    return [
        r["success"]
        for r in records
        if r["control"] == control
        and r["clearance_side_mm"] == cell_mm
        and not r["creep_probe"]
    ]


def compute_verdict(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the pre-stated P1/P2/P4 rule to the per-episode records.

    Pure function of the committed records (the report's tables and verdict
    are recomputed from ``precheck_results.json``, never hand-copied).
    """
    # P1: C-matched success = 1.0 on >= 2 of 3 cells INCLUDING the middle.
    p1_cells_pass = {
        cell: bool(all(_cell_success(records, "matched", cell)))
        for cell in _CELLS_MM
    }
    p1 = sum(p1_cells_pass.values()) >= 2 and p1_cells_pass[_CELLS_MM[1]]

    # P2a: C-limp success = 0.0 on every matched-passing cell, mechanism =
    # part-pose escape (excursion > tolerance) with finite channels.
    p2a_checks: dict[str, bool] = {}
    for cell in _CELLS_MM:
        if not p1_cells_pass[cell]:
            continue
        limp = [
            r
            for r in records
            if r["control"] == "limp" and r["clearance_side_mm"] == cell
        ]
        p2a_checks[str(cell)] = bool(
            limp
            and all(not r["success"] for r in limp)
            and all(
                r["pose_excursion_mm"] > COHOLD_POSE_HELD_TRANS_TOL_M * 1000 for r in limp
            )
            and all(r["all_channels_finite"] for r in limp)
        )
    p2a = bool(p2a_checks) and all(p2a_checks.values())

    # P2b: C-none success = 0.0 everywhere (two-robot necessity).
    none_records = [r for r in records if r["control"] == "none"]
    p2b = bool(none_records) and all(not r["success"] for r in none_records)
    p2 = p2a and p2b

    # P4: creep-probe wrench p90 strictly increasing across the detent sweep
    # on the middle cell, all samples finite, max below the artifact bound.
    p4_stats: dict[str, float] = {}
    p4_max = 0.0
    p4_finite = True
    for force in _DETENT_SWEEP_N:
        rows = [
            r
            for r in records
            if r["creep_probe"] and r["detent_force_n"] == force and r["seed"] == 0
        ]
        p4_stats[str(force)] = rows[0]["wrench_active_p90_n"] if rows else float("nan")
        for r in rows:
            p4_max = max(p4_max, r["wrench_active_max_n"])
            p4_finite = p4_finite and r["all_channels_finite"]
    vals = [p4_stats[str(f)] for f in _DETENT_SWEEP_N]
    p4 = (
        all(np.isfinite(vals))
        and vals[0] < vals[1] < vals[2]
        and p4_max < _P4_WRENCH_BOUND_N
        and p4_finite
    )

    # P3 is reported, never gating.
    fixture_rows = [r for r in records if r["control"] == "fixture"]
    p3_report = {
        "fixture_success_per_cell": {
            str(cell): all(_cell_success(records, "fixture", cell)) for cell in _CELLS_MM
        },
        "fixture_count": len(fixture_rows),
    }

    verdict = "PROCEED" if (p1 and p2 and p4) else "STOP"
    return {
        "p1_solvability": {"pass": p1, "cells": {str(k): v for k, v in p1_cells_pass.items()}},
        "p2_coupling_liveness": {
            "pass": p2,
            "limp_per_matched_cell": p2a_checks,
            "none_all_fail": p2b,
        },
        "p3_fixture_pre_read": p3_report,
        "p4_stress_channel": {
            "pass": p4,
            "wrench_p90_by_force_n": p4_stats,
            "wrench_max_n": p4_max,
            "bound_n": _P4_WRENCH_BOUND_N,
        },
        "verdict": verdict,
    }


def _run() -> int:
    records: list[dict[str, Any]] = []
    plan: list[tuple[str, float, float, bool]] = []
    for cell in COHOLD_CLEARANCE_SIDE_SET_M:
        for control in ("matched", "limp", "none", "fixture"):
            plan.append((control, cell, COHOLD_DETENT_FORCE_N, False))
    for force in _DETENT_SWEEP_N:
        plan.append(("matched", _MIDDLE_CELL_M, force, True))
    total = len(plan) * len(_SEEDS)
    done = 0
    for control, cell, force, creep in plan:
        for seed in _SEEDS:
            rec = _episode(
                control=control, clearance_m=cell, detent_n=force, seed=seed, creep=creep
            )
            records.append(rec)
            done += 1
            print(
                f"[{done}/{total}] {control:8s} clr={rec['clearance_side_mm']}mm "
                f"F={force:4.1f}N seed={seed} creep={creep} -> "
                f"success={rec['success']} d={rec['seated_depth_mm']}mm "
                f"exc={rec['pose_excursion_mm']}mm",
                flush=True,
            )
    verdict = compute_verdict(records)
    payload = {
        "precheck": "coholdsecure-precheck-2026-07-16",
        "adr": "ADR-029 §Decision (pre-stated, non-registered, non-gating)",
        "episode_steps": _EPISODE_STEPS,
        "seeds": list(_SEEDS),
        "detent_anchor_n": COHOLD_DETENT_FORCE_N,
        "detent_sweep_n": list(_DETENT_SWEEP_N),
        "driver_ego_press": _EGO_PRESS,
        "driver_ego_creep": _EGO_CREEP,
        "driver_holder_reference": _HOLDER_REF,
        "records": records,
        "verdict_block": verdict,
    }
    _ARCHIVE.mkdir(parents=True, exist_ok=True)
    _RESULTS_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {_RESULTS_JSON}")
    print(f"VERDICT: {verdict['verdict']}")
    return 0


def _render() -> int:
    payload = json.loads(_RESULTS_JSON.read_text())
    records = payload["records"]
    verdict = compute_verdict(records)
    print("## Per-cell precheck table (recomputed from precheck_results.json)\n")
    print(
        "| control | clearance (mm/side) | detent (N) | seeds | success | depth (mm) "
        "| align (deg) | excursion (mm) | tilt (deg) | peak wrench (N) |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")
    seen: set[tuple[str, float, float, bool]] = set()
    for r in records:
        key = (r["control"], r["clearance_side_mm"], r["detent_force_n"], r["creep_probe"])
        if key in seen:
            continue
        seen.add(key)
        rows = [
            x
            for x in records
            if (x["control"], x["clearance_side_mm"], x["detent_force_n"], x["creep_probe"])
            == key
        ]
        succ = sum(x["success"] for x in rows)
        tag = " (creep)" if r["creep_probe"] else ""
        print(
            f"| {r['control']}{tag} | {r['clearance_side_mm']} | {r['detent_force_n']} "
            f"| {len(rows)} | {succ}/{len(rows)} | {rows[0]['seated_depth_mm']} "
            f"| {rows[0]['axis_align_deg']} | {rows[0]['pose_excursion_mm']} "
            f"| {rows[0]['pose_tilt_deg']} | {rows[0]['peak_couple_wrench_n']} |"
        )
    print("\n## P4 creep-probe wrench (p90 over detent-active steps)\n")
    for force, p90 in verdict["p4_stress_channel"]["wrench_p90_by_force_n"].items():
        print(f"- F = {force} N -> p90 = {p90} N")
    print(json.dumps(verdict, indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point (ADR-029 §Decision; the pre-stated precheck)."""
    parser = argparse.ArgumentParser(prog="coholdsecure-precheck", description=__doc__)
    parser.add_argument(
        "--render",
        action="store_true",
        help="render tables + verdict from the committed JSON (no episodes)",
    )
    args = parser.parse_args(argv)
    return _render() if args.render else _run()


if __name__ == "__main__":
    sys.exit(main())
