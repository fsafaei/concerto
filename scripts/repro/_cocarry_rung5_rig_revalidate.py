# SPDX-License-Identifier: Apache-2.0
# pyright: reportPrivateImportUsage=false
"""Rung-5 Stage-1 rig re-validation (pre-registered; ADR-026 §D4; ADR-007).

Re-validates the FIXED co-carry substrate (compliant coupling K=8000, the
embodiment-invariant coupling-stress instrument, f_max 365.6) BEFORE the heavy
Stage-3 re-freeze, against the LOCKED protocol
(spikes/results/cocarry/rung5/cocarry_rung5_codesign_prereg.json):

1. Controlled invariance — re-confirm the coupling instrument is geometry-only
   (Panda vs xArm6 at the SAME imposed bar state read within ~1 N); the measure
   code is byte-unchanged since Rung-4c, so this re-runs that check directly.
2. Matched reference — the matched Panda cooperative pair (both seats
   cocarry_impedance) on the compliant coupling at f_max 365.6 succeeds ~1.0,
   and its success-stress p99 reconfirms f_max 365.6 consistent (p99 in
   [234, 318]) -> grounds the Stage-3 penalty (threshold = round(1.05 x p99),
   scale = 365.6 - threshold). p99 OUTSIDE the band => STOP (substrate drift
   invalidating the fixed f_max; do NOT re-derive).
3. Single-arm positive control — only the ego holds; success ~0 (the coupling
   is real: the tilt/stress conjuncts only hold when two holds share the load).

This is eval-only and incumbent-free (the trained incumbent does not exist
until Stage 3); the ego seat runs the hand-built cooperative-impedance
controller, the substrate's known-good reference. Exit 0 = PASS; exit 3 = STOP.
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
from chamber.partners.registry import load_partner

# Locked Stage-0 constants (cocarry_rung5_codesign_prereg.json).
_FMAX = 365.6
_K = 8000.0
_MEASURE = "coupling"
_EPISODE = cc.COCARRY_DEFAULT_EPISODE_LENGTH  # 320
_FMAX_BAND = (234.0, 318.0)  # matched-coupling success-stress p99 reconfirm band
_RENDER = "none"
# Seeds (prereg): the matched-coupling distribution + reference set; a disjoint
# subset drives the single-arm positive control (a control, not a measurement).
_MATCHED_SEEDS = list(range(52000, 52012))  # 12
_SINGLE_SEEDS = list(range(52000, 52006))  # 6


def _matched_ego_partner(seed: int) -> tuple[Any, Any]:
    """Build the matched cooperative pair (both seats = cocarry_impedance)."""
    specs = cc.cocarry_matched_controller_specs()
    ego = load_partner(
        PartnerSpec("cocarry_impedance", seed, None, None, dict(specs["panda_wristcam"]))
    )
    partner = load_partner(
        PartnerSpec("cocarry_impedance", seed, None, None, dict(specs["panda_partner"]))
    )
    return ego, partner


def _episode(seed: int, condition_id: str) -> ph.ConjunctMetrics:
    """One matched-pair episode at K=8000 + coupling measure, success judged at f_max 365.6."""
    env = make_cocarry_training_env(
        condition_id=condition_id,
        root_seed=seed,
        render_backend=_RENDER,
        drive_stiffness=_K,
        stress_measure=_MEASURE,
        stress_max=_FMAX,  # the env predicate judges is_unstressed at the coupling f_max
    )
    try:
        ego, partner = _matched_ego_partner(seed)
        ego.reset(seed=seed)
        return ph.rollout_pair(
            env=env,
            ego_act=lambda o, _e=ego: np.asarray(_e.act(o), dtype=np.float32),
            partner=partner,
            seed=seed,
            episode_length=_EPISODE,
        )
    finally:
        env.close()


def main() -> int:
    out_path = os.environ.get(
        "OUT_JSON", "spikes/results/cocarry/rung5/cocarry_rung5_rig_revalidation.json"
    )

    # [1] Controlled invariance — re-run the Rung-4c check (measure byte-unchanged).
    print("    [1] controlled invariance (same bar state, Panda vs xArm6)...")
    from scripts.repro._cocarry_rung4c_robustness import controlled_invariance

    invariance = controlled_invariance()
    _diff = invariance["max_abs_diff_n"]
    print(f"      max abs diff = {_diff:.2f} N -> invariant={invariance['invariant']}")

    # [2] Matched reference + success-stress p99 -> f_max reconfirm + penalty grounding.
    print(f"    [2] matched reference ({len(_MATCHED_SEEDS)} seeds, K={_K}, f_max={_FMAX})...")
    matched = [_episode(s, "cocarry_matched_panda_pair") for s in _MATCHED_SEEDS]
    matched_summary = ph.conjunct_failure_summary(matched)
    matched_rate = matched_summary["success_rate"]
    success_stress = [m.max_stress_proxy for m in matched if m.success]
    p99 = float(np.percentile(success_stress, 99)) if success_stress else float("nan")
    p99_in_band = bool(_FMAX_BAND[0] <= p99 <= _FMAX_BAND[1])
    penalty_threshold = round(1.05 * p99) if p99_in_band else None
    penalty_scale = (_FMAX - penalty_threshold) if penalty_threshold is not None else None
    print(
        f"      matched success={matched_rate:.3f}  success-stress p99={p99:.1f} N  "
        f"in_band[{_FMAX_BAND[0]:.0f},{_FMAX_BAND[1]:.0f}]={p99_in_band}"
    )
    if penalty_threshold is not None:
        print(
            f"      penalty grounding: threshold={penalty_threshold} N  scale={penalty_scale:.1f} N"
        )

    # [3] Single-arm positive control — success ~0 (the coupling is real).
    print(f"    [3] single-arm positive control ({len(_SINGLE_SEEDS)} seeds)...")
    single = [_episode(s, "cocarry_single_arm_positive_control") for s in _SINGLE_SEEDS]
    single_rate = float(np.mean([m.success for m in single]))
    print(f"      single-arm success={single_rate:.3f}")

    # Pre-registered gates.
    stops: list[str] = []
    if not invariance["invariant"]:
        stops.append(
            f"controlled invariance failed (max abs diff {invariance['max_abs_diff_n']:.2f} N)"
        )
    if matched_rate < 0.90:
        stops.append(f"matched reference {matched_rate:.3f} < 0.90 (substrate not ready)")
    if not p99_in_band:
        stops.append(
            f"matched-coupling p99 {p99:.1f} N outside [{_FMAX_BAND[0]:.0f},{_FMAX_BAND[1]:.0f}] "
            "(substrate drift invalidating the fixed f_max; do NOT re-derive)"
        )
    if single_rate > 0.10:
        stops.append(f"single-arm positive control {single_rate:.3f} > 0.10 (coupling not binding)")

    verdict = "STOP" if stops else "PASS"
    artifact = {
        "schema": "cocarry_rung5_rig_revalidation/v1",
        "stage": "Rung-5 Stage-1 rig re-validation",
        "constants": {
            "fmax_coupling_n": _FMAX,
            "drive_stiffness": _K,
            "stress_measure": _MEASURE,
            "episode_length": _EPISODE,
            "fmax_reconfirm_band_n": list(_FMAX_BAND),
        },
        "controlled_invariance": invariance,
        "matched_reference": {
            "seeds": _MATCHED_SEEDS,
            "success_rate": matched_rate,
            "success_stress_p99_n": p99,
            "p99_in_band": p99_in_band,
            "penalty_grounding": {
                "rule": "threshold = round(1.05 x p99); scale = f_max - threshold",
                "threshold_n": penalty_threshold,
                "scale_n": penalty_scale,
            },
            "summary": matched_summary,
        },
        "single_arm_positive_control": {"seeds": _SINGLE_SEEDS, "success_rate": single_rate},
        "gates": {
            "invariance_holds": bool(invariance["invariant"]),
            "matched_reconfirms": bool(matched_rate >= 0.90),
            "p99_in_band": p99_in_band,
            "single_arm_near_zero": bool(single_rate <= 0.10),
        },
        "verdict": verdict,
        "stops": stops,
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(artifact, fh, sort_keys=True, indent=2)
    print(f"    verdict = {verdict}")
    if stops:
        for s in stops:
            print(f"      STOP: {s}")
    print(f"    artifact -> {out_path}")
    return 3 if stops else 0


if __name__ == "__main__":
    sys.exit(main())
