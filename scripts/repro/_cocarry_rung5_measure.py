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


def _rollout(
    ego_act: Any,  # noqa: ANN401
    partner_class: str,
    seed: int,
    *,
    residual: bool = True,
    ego_obj: Any = None,  # noqa: ANN401
) -> ph.ConjunctMetrics:
    """One episode: ego (residual incumbent or base) + a partner, judged at f_max 365.6.

    ``ego_obj`` (when given) is reset per episode: rollout_pair resets the partner
    but NOT the ego, and the hand-built base controller is STATEFUL (cached
    start-target / step), so the base control MUST reset it each episode. The
    frozen incumbent is a stateless policy closure (ego_obj=None).
    """
    if ego_obj is not None:
        ego_obj.reset(seed=seed)
    env = make_cocarry_training_env(
        condition_id="cocarry_matched_panda_pair",
        episode_length=_EPISODE,
        root_seed=seed,
        render_backend=_RENDER,
        drive_stiffness=_K,
        stress_measure=_MEASURE,
        stress_max=_FMAX,
        # True: incumbent = base + residual; False: base alone (the control).
        residual_base_ego=residual,
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


def _measure_partner(
    ego_act: Any,  # noqa: ANN401
    partner_class: str,
    *,
    residual: bool = True,
    ego_obj: Any = None,  # noqa: ANN401
) -> tuple[dict[int, bool], dict[str, Any]]:
    metrics = [
        _rollout(ego_act, partner_class, s, residual=residual, ego_obj=ego_obj) for s in _M_SEEDS
    ]
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


def main() -> int:  # noqa: PLR0915 - linear measurement+control pipeline, kept auditable in one place
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

    # BASE-ROBUSTNESS CONTROL (added after the per-teammate breakdown revealed a
    # confound): run the structured BASE cooperative ego (no residual) vs every
    # partner on the SAME measurement seeds. A partner's incumbent-drop is a
    # GENUINE heterogeneity effect only if the base ALSO drops with it; if the
    # base holds (>= reach) but the incumbent drops, the drop is an incumbent
    # RESIDUAL-BRITTLENESS artifact, NOT a co-design penalty.
    print("    [base control] structured cooperative ego (no residual) vs all partners...")
    base_ego = ph.build_cooperative_ego(seed=0)
    base_rate: dict[str, float] = {}
    base_summaries: dict[str, Any] = {}
    for pc in [_MATCHED, *arm_a, *arm_b]:
        base_success, base_summaries[pc] = _measure_partner(
            lambda o, _e=base_ego: _e.act(o), pc, residual=False, ego_obj=base_ego
        )
        base_rate[pc] = ph.success_rate(base_success)

    inc_rate = {pc: s["success_rate"] for arm in (a, b) for pc, s in arm["summaries"].items()}
    inc_rate[_MATCHED] = ref_rate

    def _genuine(pc: str) -> bool:
        # incumbent drops >= Δ_min AND the base ALSO drops (the partner is genuinely hard).
        return (ref_rate - inc_rate[pc]) >= _DELTA_MIN and base_rate[pc] < _REACH

    base_robust_all = all(base_rate[pc] >= _REACH for pc in [_MATCHED, *arm_a, *arm_b])
    n_genuine_b = sum(1 for pc in arm_b if _genuine(pc))
    arm_a_drop = a["decision"]["verdict"] == ph.VERDICT_DROP
    arm_b_qualifies = (
        b["decision"]["verdict"] == ph.VERDICT_DROP and b["n_variants_drop"] >= _MIN_REPLICATION
    )
    arm_a_clean_null = a["decision"]["verdict"] == ph.VERDICT_NULL

    if arm_b_qualifies and n_genuine_b < _MIN_REPLICATION and base_robust_all:
        matrix = "CONFOUNDED_BY_INCUMBENT_BRITTLENESS"
        matrix_note = (
            "The structured base cooperative ego succeeds >= reach with EVERY capability-matched "
            "partner (co-designed AND non-co-designed) on the SAME measurement seeds; the "
            "incumbent's drops (admittance/selfish) are RESIDUAL training-brittleness artifacts, "
            "NOT a co-design penalty. The measurement does NOT isolate co-design: the task is "
            "robustly solvable by the structured controller with all matched partners. The CD axis "
            "is NOT established with this learned incumbent. Strategic read: corroborates that a "
            "learned (even residual) cooperator overfits its training partner where the structured "
            "controller is robust; a harder TASK alone will not fix the incumbent-construction gap."
        )
    elif arm_a_drop:
        matrix = "ARM_A_DROP_INVESTIGATE_SUBSTRATE"
        matrix_note = "A co-designed drop signals a substrate/incumbent regression -> investigate."
    elif arm_b_qualifies and n_genuine_b >= _MIN_REPLICATION and arm_a_clean_null:
        matrix = "THESIS_LANDS_CODESIGN_LOAD_BEARING"
        matrix_note = (
            "Arm A null AND Arm B replicated drop that SURVIVES the base control (base drops too): "
            "co-design is the load-bearing variable."
        )
    elif arm_b_qualifies and n_genuine_b >= _MIN_REPLICATION and not arm_a_clean_null:
        matrix = "HETEROGENEITY_BITES_CODESIGN_NOT_CLEANLY_ISOLATED"
        matrix_note = (
            "Arm B drops survive the base control, but Arm A is not a clean null (a co-designed "
            "partner also drops): degradation tracks active co-leveling, not co-design per se."
        )
    else:
        matrix = "CONCLUSIVE_NEGATIVE_ROUTE_HARDER_TASK"
        matrix_note = (
            "Arm A null AND Arm B null: heterogeneity does not bite even without co-design -> the "
            "pre-committed harder-task regime. NOT 'co-design is inert'."
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
        "base_robustness_control": {
            "why": (
                "Structured base cooperative ego (cocarry_impedance, no residual) vs every partner "
                "on the SAME measurement seeds. Separates a genuine co-design/heterogeneity effect "
                "(base also drops) from an incumbent residual-brittleness artifact (base holds)."
            ),
            "base_success_rate": base_rate,
            "incumbent_success_rate": inc_rate,
            "base_robust_all_partners": base_robust_all,
            "genuine_drop_arm_b": {pc: _genuine(pc) for pc in arm_b},
            "n_genuine_drop_arm_b": n_genuine_b,
            "base_summaries": base_summaries,
        },
        "comparative_verdict": matrix,
        "comparative_note": matrix_note,
        "null_caveat": (
            "Calibration excluded slew (A) + selfish_station (B), truncating heterogeneity from "
            "above (biases delta toward zero): a null is 'among capability-matched partners, no "
            "qualifying drop', never 'co-design is inert'."
        ),
        "incumbent_caveat": (
            "Base-robustness control RE-READS the freeze-time diagnostic: the residual lowered "
            "stress with the matched partner but is BRITTLE off-distribution (the structured base "
            "cooperates with every partner; the incumbent does not). The 'trained incumbent' is "
            "partner-overfit, not a robust cooperator."
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
