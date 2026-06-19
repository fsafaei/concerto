# SPDX-License-Identifier: Apache-2.0
"""Driver for the Rung-3 PH measurement (ADR-026 §Decision 4; R-2026-06-B §15 Rung 3).

Invoked by ``scripts/repro/cocarry_rung3_ph_measure.sh`` AFTER the
pre-registration is committed + git-tagged. Loads the admitted teammate set
from the committed calibration roster, evaluates the frozen incumbent against
the matched reference partner and each admitted teammate over the
pre-registered measurement seeds, computes the per-teammate + pooled Δ with
the cluster-bootstrap one-sided CI (IQM secondary, cluster-robust binomial
confirmatory), applies the pre-committed decision rule, and writes the
measurement artifact. STOPs (exit 3) if the matched reference does not
reconfirm ~100% (something changed — investigate before trusting the shifted
arm).
"""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from chamber.benchmarks import cocarry_ph as ph
from chamber.benchmarks.cocarry_runner import summarize as _summarize
from concerto.training.config import load_config


def summarize(metrics: list) -> dict:
    """Aggregate ConjunctMetrics via the shared runner ``summarize``.

    ``ConjunctMetrics`` is a structural superset of ``EpisodeMetrics`` (it
    carries every field ``summarize`` reads); the cast bridges ``list``
    invariance for pyright without copying.
    """
    return _summarize(cast("Any", metrics))


_CONFIG = Path("configs/training/ego_aht_happo/cocarry_matched.yaml")
_CKPT_URI = "local://artifacts/f6dad85ec9df4d58_step100000.pt"
_ARTIFACTS_ROOT = Path("artifacts")
_RENDER_BACKEND = "none"
_ROSTER = "spikes/results/cocarry/rung3/cocarry_rung3_calibration_roster.json"
_REFERENCE_RECONFIRM_FLOOR = 0.90  # the matched-reference reach band (UNCHANGED bar)


def _metrics_to_rows(metrics: list) -> list[dict]:
    return [
        {
            "seed": m.seed,
            "success": m.success,
            "is_placed": m.is_placed,
            "is_level": m.is_level,
            "is_unstressed": m.is_unstressed,
            "both_static": m.both_static,
            "centroid_to_goal": round(m.centroid_to_goal, 4),
            "max_tilt_deg": round(m.max_tilt_deg, 3),
            "max_stress_proxy": round(m.max_stress_proxy, 1),
        }
        for m in metrics
    ]


def main() -> int:
    out_path = os.environ.get(
        "OUT_JSON", "spikes/results/cocarry/rung3/cocarry_rung3_ph_measurement.json"
    )
    roster = json.loads(Path(_ROSTER).read_text(encoding="utf-8"))
    admitted = list(roster["admitted"])
    excluded = list(roster["excluded"])
    any_excluded = len(excluded) > 0
    seeds = list(ph.MEASUREMENT_SEEDS)
    cfg = load_config(config_path=_CONFIG, overrides=[])

    print(f"    admitted teammates = {admitted}")
    print(f"    measurement seeds  = {seeds}")

    # 1. Reference: frozen incumbent + matched partner. Reconfirm ~100%.
    print(f"    [reference] frozen incumbent + {ph.MATCHED_PARTNER_CLASS} ...")
    ref_metrics = ph.evaluate_incumbent_vs_partner(
        cfg=cfg,
        checkpoint_uri=_CKPT_URI,
        artifacts_root=_ARTIFACTS_ROOT,
        partner_class=ph.MATCHED_PARTNER_CLASS,
        seeds=seeds,
        render_backend=_RENDER_BACKEND,
    )
    reference = {m.seed: m.success for m in ref_metrics}
    ref_rate = ph.success_rate(reference)
    print(f"      reference success = {ref_rate:.0%}")

    # 2. Shifted: frozen incumbent + each admitted teammate.
    shifted_metrics: dict[str, list] = {}
    shifted_by_teammate: dict[str, dict[int, bool]] = {}
    for cls in admitted:
        print(f"    [shifted] frozen incumbent + {cls} ...")
        m = ph.evaluate_incumbent_vs_partner(
            cfg=cfg,
            checkpoint_uri=_CKPT_URI,
            artifacts_root=_ARTIFACTS_ROOT,
            partner_class=cls,
            seeds=seeds,
            render_backend=_RENDER_BACKEND,
        )
        shifted_metrics[cls] = m
        shifted_by_teammate[cls] = {x.seed: x.success for x in m}
        print(f"      {cls} success = {ph.success_rate(shifted_by_teammate[cls]):.0%}")

    # 3. Statistics: pooled + per-teammate Δ, IQM, GLMM, decision.
    boot = ph.cluster_bootstrap_delta(reference, shifted_by_teammate, root_seed=0)
    ref_iqm = ph.iqm([float(v) for v in reference.values()])
    iqm_per_teammate = {
        cls: ph.iqm([float(v) for v in sb.values()]) for cls, sb in shifted_by_teammate.items()
    }
    pooled_shifted_iqm = float(sum(iqm_per_teammate.values()) / max(1, len(iqm_per_teammate)))
    glmm = ph.cluster_robust_glmm(reference, shifted_by_teammate)

    # The positive control (single-arm ~0) is established at Rung 1; the bar is
    # unchanged, so it holds by construction (recorded as the pre-committed input).
    positive_control_holds = True
    decision = ph.decide(
        pooled_mean_delta=boot["pooled_mean_delta"],
        pooled_ci_lower_one_sided=boot["pooled_ci_lower_one_sided"],
        pooled_ci_upper_for_null=boot["pooled_ci_two_sided"][1],
        positive_control_holds=positive_control_holds,
        any_excluded=any_excluded,
    )

    artifact = {
        "schema": "cocarry_rung3_ph_measurement/v1",
        "date": datetime.now(UTC).strftime("%Y-%m-%d"),
        "axis": "PH (policy heterogeneity)",
        "prereg_artifact": "spikes/results/cocarry/rung3/cocarry_rung3_ph_prereg.json",
        "calibration_roster_artifact": _ROSTER,
        "freeze_manifest_artifact": (
            "spikes/results/cocarry/rung2/cocarry_rung2_freeze_manifest.json"
        ),
        "checkpoint_uri": _CKPT_URI,
        "measurement_seeds": seeds,
        "admitted_teammates": admitted,
        "excluded_teammates": excluded,
        "reference": {
            "partner_class": ph.MATCHED_PARTNER_CLASS,
            "success_rate": ref_rate,
            "summary": summarize(ref_metrics),
            "conjunct_summary": ph.conjunct_failure_summary(ref_metrics),
            "per_seed": _metrics_to_rows(ref_metrics),
        },
        "reference_reconfirm_floor": _REFERENCE_RECONFIRM_FLOOR,
        "reference_reconfirms": bool(ref_rate >= _REFERENCE_RECONFIRM_FLOOR),
        "shifted": {
            cls: {
                "success_rate": ph.success_rate(shifted_by_teammate[cls]),
                "summary": summarize(shifted_metrics[cls]),
                "conjunct_summary": ph.conjunct_failure_summary(shifted_metrics[cls]),
                "per_seed": _metrics_to_rows(shifted_metrics[cls]),
            }
            for cls in admitted
        },
        "delta": {
            "pooled_mean_delta": boot["pooled_mean_delta"],
            "pooled_ci_lower_one_sided": boot["pooled_ci_lower_one_sided"],
            "pooled_ci_two_sided": list(boot["pooled_ci_two_sided"]),
            "per_teammate": boot["per_teammate"],
            "n_seed_clusters": boot["n_seed_clusters"],
            "n_boot": boot["n_boot"],
            "alpha": boot["alpha"],
            "pooling_method": (
                "pooled = reference_rate - mean_over_teammates(shifted_rate); each teammate "
                "weighted equally; per-teammate Δ also reported. Cluster (seed) bootstrap "
                "resamples the 12 paired seed-clusters."
            ),
        },
        "iqm_secondary": {
            "reference_iqm": ref_iqm,
            "per_teammate_iqm": iqm_per_teammate,
            "pooled_shifted_iqm": pooled_shifted_iqm,
            "iqm_pooled_delta": ref_iqm - pooled_shifted_iqm,
            "note": "IQM is the secondary, NON-gate-bearing estimator (R-2026-06-B §15).",
        },
        "glmm_confirmatory": glmm,
        "positive_control_holds": positive_control_holds,
        "decision": decision,
        "delta_min": ph.DELTA_MIN,
        "null_rule": ph.NULL_RULE,
        "null_caveat_applies": any_excluded,
        "null_caveat": ph.NULL_CAVEAT if any_excluded else None,
        "unchanged_bar": {
            "success_predicate": "is_placed & is_level & is_unstressed & both_static & is_settled "
            "(cocarry.py); UNCHANGED",
            "fmax_n": 130.56970529556276,
            "goal_radius_m": 0.10,
            "matched_reference_success_rate": 1.0,
        },
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(artifact, fh, sort_keys=True, indent=2)

    print("")
    print(
        f"    pooled mean Δ = {boot['pooled_mean_delta']:+.3f}  "
        f"one-sided 95% CI lower = {boot['pooled_ci_lower_one_sided']:+.3f}"
    )
    for cls in admitted:
        pt = boot["per_teammate"][cls]
        print(
            f"      {cls}: Δ={pt['mean_delta']:+.3f} (shifted {pt['shifted_rate']:.0%}) "
            f"ci_lower={pt['ci_lower_one_sided']:+.3f}"
        )
    print(
        f"    GLMM confirmatory: {glmm.get('status')} "
        f"coef={glmm.get('coef_shifted_logodds', 'n/a')} p={glmm.get('p_value', 'n/a')}"
    )
    print(f"    VERDICT: {decision['verdict']}")
    print(f"    artifact -> {out_path}")

    if not (ref_rate >= _REFERENCE_RECONFIRM_FLOOR):
        print(
            "    STOP: matched reference did NOT reconfirm ~100% — investigate before "
            "trusting the shifted arm (R-2026-06-B §15 Rung 3 stop criterion)."
        )
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
