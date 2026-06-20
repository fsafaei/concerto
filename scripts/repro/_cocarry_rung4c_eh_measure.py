# SPDX-License-Identifier: Apache-2.0
"""Rung-4c EH measurement under the embodiment-invariant coupling-force instrument.

ADR-026 §D4; ADR-005; ADR-009; R-2026-06-B §15 Rung 4c. Invoked by
``scripts/repro/cocarry_rung4c_eh_measure.sh`` AFTER the pre-registration is
committed + tagged.

All conditions run with ``stress_measure="coupling"`` (the bar coupling spring
force — embodiment-invariant by construction). f_max is re-derived from the
matched-Panda-pair coupling distribution, then HELD; success is re-scored in
Python at that f_max (the env's built-in 130.6 N unstressed conjunct is for the
wrist proxy and is bypassed here, exactly as the rigid f_max re-derivation
rescored). The frozen incumbent is eval-only (its policy input carries no
stress term, so the stress measure does not change its actions — no retrain).

Pipeline: invariance table -> f_max -> reference reconfirm -> single-arm ->
xArm6 capability gate -> PH floor -> EH graded Δ + CI + GEE + decision.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from chamber.benchmarks import cocarry_ph as ph
from chamber.benchmarks.cocarry_runner import build_matched_controllers
from chamber.envs.cocarry import cocarry_xarm6_controller_spec, evaluate_cocarry_success
from chamber.envs.cocarry_obs import make_cocarry_training_env
from chamber.partners.api import PartnerSpec
from chamber.partners.registry import load_partner
from concerto.training.config import load_config

_CONFIG = Path("configs/training/ego_aht_happo/cocarry_matched.yaml")
_CKPT = "local://artifacts/f6dad85ec9df4d58_step100000.pt"
_ART = Path("artifacts")
_RENDER = "none"
_MEASURE = "coupling"
_K = 8000.0
_FMAX_MULT = 1.25
_DELTA_MIN = 0.20
_REACH = 0.90
_SINGLE_ARM_MAX = 0.1
_PH_TEAMMATES = ["cocarry_stiff_impedance", "cocarry_admittance", "cocarry_slew_impedance"]
_FMAX_SEEDS = list(range(80000, 80012))
_PH_SEEDS = list(range(80100, 80112))
_EH_SEEDS = list(range(80200, 80212))
_INV_SEEDS = [70010, 70011, 70012, 70013]


def _rescore(m: ph.ConjunctMetrics, fmax: float) -> bool:
    """Success re-scored at the coupling f_max (placed & level & coupling<fmax & static)."""
    return bool(
        evaluate_cocarry_success(
            centroid_to_goal_dist=m.centroid_to_goal,
            max_tilt_deg=m.max_tilt_deg,
            max_stress_proxy=m.max_stress_proxy,
            both_static=m.both_static,
            stress_max=fmax,
        )
    )


def _matched_pair(seed: int) -> ph.ConjunctMetrics:
    """Matched controller pair (both impedance) under the coupling measure."""
    env = make_cocarry_training_env(
        condition_id="cocarry_matched_panda_pair",
        root_seed=seed,
        render_backend=_RENDER,
        drive_stiffness=_K,
        stress_measure=_MEASURE,
    )
    try:
        ctr = build_matched_controllers()
        return ph.rollout_pair(
            env=env,
            ego_act=lambda o, _e=ctr["panda_wristcam"]: np.asarray(_e.act(o), dtype=np.float32),
            partner=ctr["panda_partner"],
            seed=seed,
            episode_length=320,
        )
    finally:
        env.close()


def _incumbent_vs(partner_class: str, seeds: list[int], cond: str, puid: str) -> list:
    cfg = load_config(config_path=_CONFIG, overrides=[])
    return ph.evaluate_incumbent_vs_partner(
        cfg=cfg,
        checkpoint_uri=_CKPT,
        artifacts_root=_ART,
        partner_class=partner_class,
        seeds=seeds,
        render_backend=_RENDER,
        condition_id=cond,
        partner_uid=puid,
        drive_stiffness=_K,
        stress_measure=_MEASURE,
    )


def _invariance() -> dict:
    """Wrist vs coupling, Panda vs xArm6 (active cooperative state) — the instrument finding."""
    out = {}
    for measure in ("wrist", "coupling"):
        mp = [
            make_active(seed, "cocarry_matched_panda_pair", "panda_partner", measure)
            for seed in _INV_SEEDS
        ]
        xp = [
            make_active(seed, "cocarry_xarm6_partner", "xarm6_robotiq", measure)
            for seed in _INV_SEEDS
        ]
        mp90 = float(np.percentile([m.max_stress_proxy for m in mp], 90))
        xp90 = float(np.percentile([m.max_stress_proxy for m in xp], 90))
        out[measure] = {
            "matched_p90": mp90,
            "xarm6_p90": xp90,
            "ratio": xp90 / max(mp90, 1e-6),
        }
    return out


def make_active(seed: int, cond: str, puid: str, measure: str) -> ph.ConjunctMetrics:
    """Active cooperative rollout: cooperative Panda ego + the partner on `cond`."""
    env = make_cocarry_training_env(
        condition_id=cond,
        root_seed=seed,
        render_backend=_RENDER,
        drive_stiffness=_K,
        stress_measure=measure,
    )
    try:
        coop = ph.build_cooperative_ego(seed=seed)
        coop.reset(seed=seed)
        if puid == "xarm6_robotiq":
            partner = load_partner(
                PartnerSpec(
                    "cocarry_xarm6_impedance",
                    seed,
                    None,
                    None,
                    dict(cocarry_xarm6_controller_spec()),
                )
            )
        else:
            partner = ph.build_partner_seat("cocarry_impedance", seed=seed, partner_uid=puid)
        return ph.rollout_pair(
            env=env,
            ego_act=lambda o, _e=coop: np.asarray(_e.act(o), dtype=np.float32),
            partner=partner,
            seed=seed,
            episode_length=320,
        )
    finally:
        env.close()


def _rate(metrics: list, fmax: float) -> dict[int, bool]:
    return {m.seed: _rescore(m, fmax) for m in metrics}


def _conj(metrics: list, fmax: float) -> dict:
    n = len(metrics)
    return {
        "n": n,
        "success_rate": float(np.mean([_rescore(m, fmax) for m in metrics])),
        "fail_placed": int(sum(1 for m in metrics if not m.is_placed)),
        "fail_level": int(sum(1 for m in metrics if not m.is_level)),
        "fail_unstressed": int(sum(1 for m in metrics if m.max_stress_proxy >= fmax)),
        "fail_static": int(sum(1 for m in metrics if not m.both_static)),
        "coupling_p90": float(np.percentile([m.max_stress_proxy for m in metrics], 90)),
        "coupling_max": float(np.max([m.max_stress_proxy for m in metrics])),
    }


def main() -> int:  # noqa: PLR0915 - one linear measurement pipeline
    out_path = os.environ.get(
        "OUT_JSON", "spikes/results/cocarry/rung4c/cocarry_rung4c_eh_measurement.json"
    )
    print("    [1] invariance investigation (active; wrist vs coupling, Panda vs xArm6)...")
    invariance = _invariance()
    Path("spikes/results/cocarry/rung4c/cocarry_rung4c_invariance.json").write_text(
        json.dumps(invariance, sort_keys=True, indent=2), encoding="utf-8"
    )
    w_r = invariance["wrist"]["ratio"]
    c_r = invariance["coupling"]["ratio"]
    print(f"      wrist ratio={w_r:.1f}  coupling ratio={c_r:.1f}")

    print("    [2] f_max from matched-Panda-pair coupling distribution...")
    fmax_metrics = [_matched_pair(s) for s in _FMAX_SEEDS]
    # matched-success coupling distribution (success on placed&level&static, pre-f_max).
    succ_coupling = [
        m.max_stress_proxy for m in fmax_metrics if m.is_placed and m.is_level and m.both_static
    ]
    p99 = float(np.percentile(succ_coupling, 99)) if succ_coupling else float("nan")
    fmax = _FMAX_MULT * p99
    print(f"      matched coupling success p99={p99:.0f} -> f_max={fmax:.0f} N (1.25x)")
    # Now matched-controller pair success rate at this f_max (sanity ~100%).
    matched_rate = float(np.mean([_rescore(m, fmax) for m in fmax_metrics]))
    print(f"      matched-controller pair success @ f_max = {matched_rate:.0%}")

    print("    [3] reference reconfirm: frozen incumbent + matched partner...")
    ref = _incumbent_vs(
        "cocarry_impedance", _EH_SEEDS, "cocarry_matched_panda_pair", "panda_partner"
    )
    reference = _rate(ref, fmax)
    ref_rate = ph.success_rate(reference)
    print(f"      reference success = {ref_rate:.0%}")

    print("    [4] single-arm positive control (coupling)...")
    single = [_matched_single(s) for s in _FMAX_SEEDS[:6]]
    single_rate = float(np.mean([_rescore(m, fmax) for m in single]))
    print(f"      single-arm success = {single_rate:.0%}")

    print("    [5] xArm6 capability gate (vs cooperative ego, coupling + f_max)...")
    calib = ph.evaluate_calibration(
        candidate_class="cocarry_xarm6_impedance",
        seeds=_EH_SEEDS,
        condition_id="cocarry_xarm6_partner",
        partner_uid="xarm6_robotiq",
        render_backend=_RENDER,
        drive_stiffness=_K,
        stress_measure=_MEASURE,
    )
    calib_rate = float(np.mean([_rescore(m, fmax) for m in calib]))
    calib_admitted = calib_rate >= ph.C_MIN
    print(f"      xArm6 calibration success @ f_max = {calib_rate:.0%}  admitted={calib_admitted}")

    print("    [6] PH floor: frozen incumbent + admitted policy-shift teammates...")
    ph_shifted = {}
    for cls in _PH_TEAMMATES:
        m = _incumbent_vs(cls, _PH_SEEDS, "cocarry_matched_panda_pair", "panda_partner")
        ph_shifted[cls] = _rate(m, fmax)
        print(f"      {cls}: {ph.success_rate(ph_shifted[cls]):.0%}")
    ph_ref = _rate(
        _incumbent_vs(
            "cocarry_impedance", _PH_SEEDS, "cocarry_matched_panda_pair", "panda_partner"
        ),
        fmax,
    )
    ph_boot = ph.cluster_bootstrap_delta(ph_ref, ph_shifted, root_seed=0)
    print(f"      PH pooled Δ = {ph_boot['pooled_mean_delta']:+.3f}")

    print("    [7] EH measurement: frozen incumbent + xArm6 vs matched reference...")
    eh_metrics = _incumbent_vs(
        "cocarry_xarm6_impedance", _EH_SEEDS, "cocarry_xarm6_partner", "xarm6_robotiq"
    )
    eh_shifted = {"cocarry_xarm6_impedance": _rate(eh_metrics, fmax)}
    eh_boot = ph.cluster_bootstrap_delta(reference, eh_shifted, root_seed=0)
    eh_glmm = ph.cluster_robust_glmm(reference, eh_shifted)
    decision = ph.decide(
        pooled_mean_delta=eh_boot["pooled_mean_delta"],
        pooled_ci_lower_one_sided=eh_boot["pooled_ci_lower_one_sided"],
        pooled_ci_upper_for_null=eh_boot["pooled_ci_two_sided"][1],
        positive_control_holds=single_rate <= _SINGLE_ARM_MAX,
        any_excluded=False,
        delta_min=_DELTA_MIN,
    )
    eh_d = eh_boot["pooled_mean_delta"]
    eh_lo = eh_boot["pooled_ci_lower_one_sided"]
    print(f"      EH pooled Δ = {eh_d:+.3f}  CI_lower={eh_lo:+.3f}")
    print(f"      VERDICT: {decision['verdict']}")

    artifact = {
        "schema": "cocarry_rung4c_eh_measurement/v1",
        "date": datetime.now(UTC).strftime("%Y-%m-%d"),
        "prereg_artifact": "spikes/results/cocarry/rung4c/cocarry_rung4c_eh_prereg.json",
        "stress_measure": _MEASURE,
        "drive_stiffness_Npm": _K,
        "invariance": invariance,
        "fmax_coupling_n": fmax,
        "fmax_matched_success_p99_n": p99,
        "fmax_multiplier": _FMAX_MULT,
        "matched_controller_pair_rate_at_fmax": matched_rate,
        "reference": {
            "success_rate": ref_rate,
            "conjunct": _conj(ref, fmax),
            "reconfirms": bool(ref_rate >= _REACH),
        },
        "single_arm_rate": single_rate,
        "xarm6_capability": {
            "rate": calib_rate,
            "admitted": bool(calib_admitted),
            "conjunct": _conj(calib, fmax),
        },
        "ph_floor": {
            "pooled_mean_delta": ph_boot["pooled_mean_delta"],
            "per_teammate": ph_boot["per_teammate"],
        },
        "eh": {
            "shifted_rate": ph.success_rate(eh_shifted["cocarry_xarm6_impedance"]),
            "conjunct": _conj(eh_metrics, fmax),
            "pooled_mean_delta": eh_boot["pooled_mean_delta"],
            "pooled_ci_lower_one_sided": eh_boot["pooled_ci_lower_one_sided"],
            "pooled_ci_two_sided": list(eh_boot["pooled_ci_two_sided"]),
            "glmm": eh_glmm,
            "decision": decision,
            "eh_vs_ph_floor": (
                f"EH Δ={eh_boot['pooled_mean_delta']:+.3f} vs PH control-style floor "
                f"Δ={ph_boot['pooled_mean_delta']:+.3f}; an EH drop beyond the PH spread is "
                "attributed to embodiment."
            ),
        },
        "seeds": {"fmax": _FMAX_SEEDS, "ph": _PH_SEEDS, "eh": _EH_SEEDS},
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(artifact, fh, sort_keys=True, indent=2)
    print(f"    artifact -> {out_path}")
    if not (ref_rate >= _REACH):
        print("    STOP: matched reference did not reconfirm under the coupling measure.")
        return 3
    return 0


def _matched_single(seed: int) -> ph.ConjunctMetrics:
    env = make_cocarry_training_env(
        condition_id="cocarry_single_arm_positive_control",
        root_seed=seed,
        render_backend=_RENDER,
        drive_stiffness=_K,
        stress_measure=_MEASURE,
    )
    try:
        ctr = build_matched_controllers()
        return ph.rollout_pair(
            env=env,
            ego_act=lambda o, _e=ctr["panda_wristcam"]: np.asarray(_e.act(o), dtype=np.float32),
            partner=ctr["panda_partner"],
            seed=seed,
            episode_length=320,
        )
    finally:
        env.close()


if __name__ == "__main__":
    sys.exit(main())
