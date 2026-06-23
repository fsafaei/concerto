# SPDX-License-Identifier: Apache-2.0
# pyright: reportPrivateImportUsage=false
"""EXPLORATORY (NON-pre-registered) stiffness-boundary characterisation (ADR-026 §Open-questions).

A cheap addendum to the LOCKED base-difficulty probe (cocarry_base_probe_prereg).
The pre-registered geometric stiffness grid jumps trivial (K=8000, 100%) ->
stress-ceiling-failure (K=16000, 0%, fail_unstressed 12/12) with no point in
between. This characterises that gap to answer the strategic question the board
decision hinges on: when the matched base degrades as coupling stiffness rises,
is it GRACEFUL COORDINATION-STRAIN or merely OVER-COUPLING ONSET (the failing
conjunct is `unstressed`, i.e. the coupling force F = K * deflection crossing
f_max with the bar still placed + level)?

This is NOT pre-registered and does NOT change the locked verdict; it only
characterises the stiffness knob's boundary. Same matched pair, same 12 seeds,
same coupling instrument / f_max as the locked probe. Eval-only, no training.

ADR-026 §Open-questions (coupling stiffness is a task parameter). Do not merge.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

import chamber.envs.cocarry as cc
from chamber.benchmarks import cocarry_ph as ph
from chamber.envs.cocarry_obs import make_cocarry_training_env

_REPO = Path(__file__).resolve().parents[2]
_PROBE_DIR = _REPO / "spikes/results/cocarry/base_probe"
_PREREG = json.loads((_PROBE_DIR / "cocarry_base_probe_prereg.json").read_text("utf-8"))
_OUT = _PROBE_DIR / "cocarry_base_probe_stiffness_boundary_exploratory.json"

_FMAX = float(_PREREG["fixed_base_config"]["fmax_coupling_n"])
_SEEDS = list(_PREREG["seeds"])
_EPISODE = int(_PREREG["fixed_base_config"]["episode_length"])
# Fill the pre-registered geometric gap between the last trivial (8000) and the
# first failing (16000) setting, to locate the f_max-crossing and its mechanism.
_K_GRID = [9000.0, 10000.0, 11000.0, 12000.0, 13000.0, 14000.0]


def _rollout(seed: int, k: float) -> dict[str, Any]:
    env = make_cocarry_training_env(
        condition_id="cocarry_matched_panda_pair",
        episode_length=_EPISODE,
        root_seed=seed,
        render_backend="none",
        drive_stiffness=k,
        stress_measure="coupling",
    )
    try:
        ego = ph.build_cooperative_ego(seed=seed)
        partner = ph.build_partner_seat("cocarry_impedance", seed=seed, partner_uid="panda_partner")
        m = ph.rollout_pair(
            env=env,
            ego_act=lambda o, _e=ego: np.asarray(_e.act(o), dtype=np.float32),
            partner=partner,
            seed=seed,
            episode_length=_EPISODE,
        )
    finally:
        env.close()
    success = bool(
        cc.evaluate_cocarry_success(
            centroid_to_goal_dist=m.centroid_to_goal,
            max_tilt_deg=m.max_tilt_deg,
            max_stress_proxy=m.max_stress_proxy,
            both_static=m.both_static,
            goal_thresh=0.10,
            tilt_max_deg=15.0,
            stress_max=_FMAX,
        )
    )
    return {
        "success": success,
        "is_placed": bool(m.centroid_to_goal <= 0.10),  # noqa: PLR2004
        "is_level": bool(m.max_tilt_deg < 15.0),  # noqa: PLR2004
        "is_unstressed": bool(m.max_stress_proxy < _FMAX),
        "both_static": bool(m.both_static),
        "max_stress_proxy": float(m.max_stress_proxy),
        "max_tilt_deg": float(m.max_tilt_deg),
    }


def main() -> int:
    results: dict[str, Any] = {}
    print(
        f"  exploratory stiffness boundary (8000<K<16000), {len(_SEEDS)} seeds, f_max={_FMAX:.1f}"
    )
    for k in _K_GRID:
        rs = [_rollout(s, k) for s in _SEEDS]
        sr = float(np.mean([r["success"] for r in rs]))
        stress = np.array([r["max_stress_proxy"] for r in rs])
        fail_unstr = int(sum(1 for r in rs if not r["is_unstressed"]))
        fail_level = int(sum(1 for r in rs if not r["is_level"]))
        fail_place = int(sum(1 for r in rs if not r["is_placed"]))
        # The dominant failing conjunct => the mechanism of any sub-1.0 band.
        mechanism = "none"
        if sr < 1.0:
            mechanism = (
                "over_coupling(unstressed)"
                if fail_unstr >= max(fail_level, fail_place)
                else ("leveling" if fail_level >= fail_place else "placement")
            )
        results[f"{k:.0f}"] = {
            "drive_stiffness_npm": k,
            "success_rate": sr,
            "fail_unstressed": fail_unstr,
            "fail_level": fail_level,
            "fail_placed": fail_place,
            "stress_p50": float(np.percentile(stress, 50)),
            "stress_p90": float(np.percentile(stress, 90)),
            "dominant_failing_conjunct": mechanism,
        }
        print(
            f"    K={k:>7.0f}  success={sr:.3f}  stress_p50/p90={np.percentile(stress, 50):.0f}/"
            f"{np.percentile(stress, 90):.0f}  fail(u/l/p)={fail_unstr}/{fail_level}/{fail_place}"
            f"  mechanism={mechanism}"
        )

    graceful = {
        k: r
        for k, r in results.items()
        if 0.10 <= r["success_rate"] <= 0.90  # noqa: PLR2004
    }
    # A stiffness "band" is genuine coordination-strain only if its degradation is
    # NOT dominated by the over-coupling (unstressed) conjunct.
    coordination_strain = {
        k: r
        for k, r in graceful.items()
        if r["dominant_failing_conjunct"] != "over_coupling(unstressed)"
    }
    finding = (
        "OVER_COUPLING_ONSET_ONLY"
        if graceful and not coordination_strain
        else ("GRACEFUL_COORDINATION_STRAIN_BAND" if coordination_strain else "NO_SUB1_BAND_IN_GAP")
    )
    note = {
        "OVER_COUPLING_ONSET_ONLY": (
            "Every sub-1.0 stiffness setting in the gap degrades via the `unstressed` conjunct "
            "(coupling force F = K*deflection crossing f_max with the bar still placed + level). "
            "This is over-coupling onset (the Rung-4 wall mechanism at lower magnitude), NOT "
            "graceful coordination-strain. The stiffness knob has no coordination-difficulty band."
        ),
        "GRACEFUL_COORDINATION_STRAIN_BAND": (
            "A sub-1.0 stiffness setting degrades via leveling/placement (not over-coupling) -- a "
            "candidate graceful coordination-strain band in the coupling physics. Investigate."
        ),
        "NO_SUB1_BAND_IN_GAP": "No sub-1.0 setting in the gap at this resolution.",
    }[finding]

    out = {
        "schema": "cocarry_base_probe_stiffness_boundary_exploratory/v1",
        "pre_registered": False,
        "note": "EXPLORATORY addendum to the locked base-difficulty probe; does NOT change the "
        "pre-registered verdict. Characterises the geometric-grid gap (8000<K<16000) to classify "
        "the stiffness knob's degradation mechanism.",
        "fmax_coupling_n": _FMAX,
        "seeds": _SEEDS,
        "by_drive_stiffness_npm": results,
        "graceful_settings": list(graceful),
        "coordination_strain_settings": list(coordination_strain),
        "finding": finding,
        "finding_note": note,
    }
    _OUT.write_text(json.dumps(out, sort_keys=True, indent=2), encoding="utf-8")
    print(f"\n  FINDING = {finding}")
    print(f"  {note}")
    print(f"  -> {_OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
