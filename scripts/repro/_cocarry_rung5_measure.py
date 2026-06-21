# SPDX-License-Identifier: Apache-2.0
# pyright: reportPrivateImportUsage=false
"""Rung-5 Stage 4-5 co-design (CD) measurement (pre-registered; ADR-026 §D4; ADR-007).

Eval-only against the FROZEN residual incumbent (no more training). Measures the
incumbent's joint-success drop vs two partner sets differing ONLY in co-design,
on the fixed task (compliant K=8000, coupling instrument, stress_max 365.6,
residual-base ego). All partners are the same Panda body, calibrated competent
(Stage-2 roster) vs the cooperative reference:

- Arm A (co-designed control-style): stiff / admittance / nullspace impedance.
- Arm B (NON-co-designed): selfish_goal / selfish_effort.

Per the locked prereg: paired within-seed over the 28 measurement seeds; pooled
Δ = reference_rate - mean_over_partners(shifted_rate); cluster (seed) bootstrap
one-sided 95% CI (n_boot=10000, P6); decide() = drop iff pooled Δ >= 0.20 AND
CI lower > 0. Arm B additionally requires replication across >= 2 variants.
Isolation (arms identical except objective) is the mediation guarantee; the
coupling-stress profile is reported. Then the comparative verdict matrix.

Reference must reconfirm >= 0.90 first (else STOP). Exit 0 = measured; 3 = STOP.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import chamber.envs.cocarry as cc

# Register the partner families (decorator registration on import).
import chamber.partners.cocarry_impedance
import chamber.partners.cocarry_policy_shift
import chamber.partners.cocarry_selfish  # noqa: F401
from chamber.benchmarks import cocarry_ph as ph
from chamber.benchmarks.cocarry_incumbent import load_frozen_incumbent
from chamber.benchmarks.cocarry_runner import build_matched_controllers
from chamber.envs.cocarry_obs import make_cocarry_training_env
from chamber.partners.api import PartnerSpec
from chamber.partners.registry import load_partner
from concerto.training.config import load_config

_FMAX = 365.6
_K = 8000.0
_MEASURE = "coupling"
_EPISODE = 320
_DELTA_MIN = ph.DELTA_MIN  # 0.20
_REACH = 0.90
_RENDER = "none"
_CONFIG = "configs/training/ego_aht_happo/cocarry_rung5_residual.yaml"
_ARTIFACTS = Path("./artifacts")
_M_SEEDS = list(range(54000, 54028))  # 28 (prereg)
_MATCHED = "cocarry_impedance"
_N_BOOT = ph.N_BOOTSTRAP  # 10000
_MIN_REPLICATION = 2  # an Arm-B CD-axis drop must replicate across >= 2 variants (prereg)


def _partner(partner_class: str, seed: int) -> Any:  # noqa: ANN401
    return load_partner(
        PartnerSpec(
            partner_class,
            seed,
            None,
            None,
            dict(cc.cocarry_matched_controller_specs()["panda_partner"]),
        )
    )


def _rollout(ego_act: Any, partner_class: str, seed: int) -> ph.ConjunctMetrics:  # noqa: ANN401
    """One episode: frozen residual incumbent (ego) + a partner, judged at f_max 365.6."""
    env = make_cocarry_training_env(
        condition_id="cocarry_matched_panda_pair",
        episode_length=_EPISODE,
        root_seed=seed,
        render_backend=_RENDER,
        drive_stiffness=_K,
        stress_measure=_MEASURE,
        stress_max=_FMAX,
        residual_base_ego=True,  # the incumbent IS base + residual
    )
    try:
        return ph.rollout_pair(
            env=env,
            ego_act=ego_act,
            partner=_partner(partner_class, seed),
            seed=seed,
            episode_length=_EPISODE,
        )
    finally:
        env.close()


def _measure_partner(ego_act: Any, partner_class: str) -> tuple[dict[int, bool], dict[str, Any]]:  # noqa: ANN401
    metrics = [_rollout(ego_act, partner_class, s) for s in _M_SEEDS]
    success = {m.seed: bool(m.success) for m in metrics}
    summary = ph.conjunct_failure_summary(metrics)
    print(
        f"      {partner_class:28s} success={summary['success_rate']:.3f}  "
        f"tilt_p90={summary['max_tilt_p90']:.1f}  stress_p90={summary['stress_p90']:.0f}  "
        f"(fail p/l/u/s={summary['fail_placed']}/{summary['fail_level']}/"
        f"{summary['fail_unstressed']}/{summary['fail_static']})"
    )
    return success, summary


def _measure_arm(
    ego_act: Any,  # noqa: ANN401
    reference: dict[int, bool],
    partners: list[str],
) -> dict[str, Any]:
    shifted: dict[str, dict[int, bool]] = {}
    summaries: dict[str, Any] = {}
    for pc in partners:
        shifted[pc], summaries[pc] = _measure_partner(ego_act, pc)
    boot = ph.cluster_bootstrap_delta(reference, shifted, n_boot=_N_BOOT, root_seed=0)
    dec = ph.decide(
        pooled_mean_delta=boot["pooled_mean_delta"],
        pooled_ci_lower_one_sided=boot["pooled_ci_lower_one_sided"],
        pooled_ci_upper_for_null=boot["pooled_ci_two_sided"][1],
        positive_control_holds=True,  # Stage-1 single-arm 0/6
        any_excluded=True,  # slew (A) / selfish_station (B) excluded at calibration
        delta_min=_DELTA_MIN,
    )
    # Per-variant qualifying drops (for Arm-B replication).
    per_variant_drop = {
        name: bool(pt["mean_delta"] >= _DELTA_MIN and pt["ci_lower_one_sided"] > 0.0)
        for name, pt in boot["per_teammate"].items()
    }
    return {
        "delta": boot,
        "decision": dec,
        "per_variant_drop": per_variant_drop,
        "n_variants_drop": int(sum(per_variant_drop.values())),
        "summaries": summaries,
    }


def main() -> int:
    out_path = os.environ.get(
        "OUT_JSON", "spikes/results/cocarry/rung5/cocarry_rung5_cd_measurement.json"
    )
    manifest = json.loads(
        Path("spikes/results/cocarry/rung5/cocarry_rung5_freeze_manifest.json").read_text("utf-8")
    )
    roster = json.loads(
        Path("spikes/results/cocarry/rung5/cocarry_rung5_calibration_roster.json").read_text(
            "utf-8"
        )
    )
    incumbent_uri = manifest["checkpoint"]["uri"]
    arm_a = roster["arm_a"]["admitted"]
    arm_b = roster["arm_b"]["admitted"]

    # Load the frozen residual incumbent once (SHA auto-verified; policy reused).
    print(f"    loading frozen incumbent {incumbent_uri} ...")
    cfg = load_config(config_path=Path(_CONFIG), overrides=["seed=0"])
    env0 = make_cocarry_training_env(
        condition_id="cocarry_matched_panda_pair",
        episode_length=_EPISODE,
        root_seed=_M_SEEDS[0],
        render_backend=_RENDER,
        drive_stiffness=_K,
        stress_measure=_MEASURE,
        stress_max=_FMAX,
        residual_base_ego=True,
    )
    try:
        ego_act = load_frozen_incumbent(
            cfg=cfg,
            env=env0,
            partner=build_matched_controllers()["panda_partner"],
            checkpoint_uri=incumbent_uri,
            artifacts_root=_ARTIFACTS,
        )
    finally:
        env0.close()

    # Reference: incumbent + matched partner. Reconfirm >= 0.90 first.
    print(f"    [ref] incumbent + matched ({len(_M_SEEDS)} seeds)...")
    reference, ref_summary = _measure_partner(ego_act, _MATCHED)
    ref_rate = ph.success_rate(reference)
    if ref_rate < _REACH:
        out = {
            "schema": "cocarry_rung5_cd_measurement/v1",
            "verdict": "STOP_REFERENCE_NOT_RECONFIRMED",
            "reason": f"reference {ref_rate:.3f} < {_REACH}; substrate changed -> STOP.",
            "reference_summary": ref_summary,
        }
        Path(out_path).write_text(json.dumps(out, sort_keys=True, indent=2), encoding="utf-8")
        print(f"    verdict = STOP_REFERENCE_NOT_RECONFIRMED ({ref_rate:.3f} < {_REACH})")
        return 3

    print(f"    [Arm A] co-designed control-style ({len(arm_a)} partners)...")
    a = _measure_arm(ego_act, reference, arm_a)
    print(f"    [Arm B] non-co-designed ({len(arm_b)} partners)...")
    b = _measure_arm(ego_act, reference, arm_b)

    arm_a_drop = a["decision"]["verdict"] == ph.VERDICT_DROP
    arm_b_qualifies = (
        b["decision"]["verdict"] == ph.VERDICT_DROP and b["n_variants_drop"] >= _MIN_REPLICATION
    )
    if arm_a_drop:
        matrix = "ARM_A_DROP_INVESTIGATE_SUBSTRATE"
        matrix_note = (
            "Unexpected: a co-designed control-style drop most likely signals a "
            "substrate/incumbent regression -> re-check Stage-1/Stage-3 before any CD claim."
        )
    elif arm_b_qualifies:
        matrix = "THESIS_LANDS_CODESIGN_LOAD_BEARING"
        matrix_note = (
            "Arm A null AND Arm B qualifying replicated drop: co-design / shared cooperation "
            "is the load-bearing variable; the matched arm is the control."
        )
    else:
        matrix = "CONCLUSIVE_NEGATIVE_ROUTE_HARDER_TASK"
        matrix_note = (
            "Arm A null AND Arm B null. Read WITH the incumbent caveat: the residual incumbent "
            "is a ROBUST cooperator (success delta=0 vs base, even MORE stress-robust), so this "
            "is 'the task is too forgiving for a robust trained incumbent' -> the pre-committed "
            "harder-task regime, NOT 'co-design is inert'. A rigorous publishable negative."
        )

    out = {
        "schema": "cocarry_rung5_cd_measurement/v1",
        "stage": "Rung-5 Stage 4-5 co-design (CD) measurement",
        "incumbent_uri": incumbent_uri,
        "incumbent_sha256": manifest["checkpoint"]["sha256"],
        "measurement_seeds": _M_SEEDS,
        "n_seed_clusters": len(_M_SEEDS),
        "delta_min": _DELTA_MIN,
        "reference": {"partner": _MATCHED, "success_rate": ref_rate, "summary": ref_summary},
        "reference_reconfirms": True,
        "arm_a": {
            "label": "co-designed control-style (the control)",
            "partners": arm_a,
            "pooled_mean_delta": a["delta"]["pooled_mean_delta"],
            "pooled_ci_lower_one_sided": a["delta"]["pooled_ci_lower_one_sided"],
            "pooled_ci_two_sided": a["delta"]["pooled_ci_two_sided"],
            "per_teammate": a["delta"]["per_teammate"],
            "verdict": a["decision"]["verdict"],
            "decision": a["decision"],
            "summaries": a["summaries"],
        },
        "arm_b": {
            "label": "non-co-designed self-interested (the decisive arm)",
            "partners": arm_b,
            "pooled_mean_delta": b["delta"]["pooled_mean_delta"],
            "pooled_ci_lower_one_sided": b["delta"]["pooled_ci_lower_one_sided"],
            "pooled_ci_two_sided": b["delta"]["pooled_ci_two_sided"],
            "per_teammate": b["delta"]["per_teammate"],
            "verdict": b["decision"]["verdict"],
            "decision": b["decision"],
            "per_variant_drop": b["per_variant_drop"],
            "n_variants_drop": b["n_variants_drop"],
            "replication_required": _MIN_REPLICATION,
            "replicated_drop": arm_b_qualifies,
            "summaries": b["summaries"],
        },
        "comparative_verdict": matrix,
        "comparative_note": matrix_note,
        "null_caveat": (
            "Calibration excluded slew (A) + selfish_station (B), truncating heterogeneity from "
            "above (biases delta toward zero): a null is 'among capability-matched partners, no "
            "qualifying drop', never 'co-design is inert'."
        ),
        "incumbent_caveat": (
            "The frozen incumbent is a robust cooperator (base+residual; success delta=0 vs base, "
            "~13% lower coupling stress): an A-null AND B-null routes to the harder-task regime, "
            "per the verdict matrix."
        ),
    }
    Path(out_path).write_text(json.dumps(out, sort_keys=True, indent=2), encoding="utf-8")
    print(
        f"    Arm A: pooled Δ={a['delta']['pooled_mean_delta']:+.3f} "
        f"CI_low={a['delta']['pooled_ci_lower_one_sided']:+.3f} -> {a['decision']['verdict']}"
    )
    print(
        f"    Arm B: pooled Δ={b['delta']['pooled_mean_delta']:+.3f} "
        f"CI_low={b['delta']['pooled_ci_lower_one_sided']:+.3f} "
        f"variants_drop={b['n_variants_drop']}/2 -> {b['decision']['verdict']}"
    )
    print(f"    COMPARATIVE VERDICT = {matrix}")
    print(f"    measurement -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
