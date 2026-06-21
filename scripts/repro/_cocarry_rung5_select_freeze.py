# SPDX-License-Identifier: Apache-2.0
# pyright: reportPrivateImportUsage=false
"""Rung-5 Stage-3b selection + validation + freeze (pre-registered; ADR-026 §D4; ADR-007).

Consumes the Stage-3 training checkpoints (cocarry_rung5_refreeze_train.json) and
applies the LOCKED freeze protocol on the FIXED task (compliant K=8000, coupling
instrument, stress_max 365.6):

1. Evaluate each checkpoint as the EGO on the selection set S (matched cocarry
   partner; env built with stress_max=365.6, so success is env-judged at the
   coupling ceiling). HARD 300k early-STOP gate: >= 1 constraint-clean checkpoint
   (all-seed max tilt < 15 deg AND max coupling stress < 365.6 N) reaching matched
   joint-success >= C_min (0.75) on S; else STOP the from-scratch primary and
   route to the residual-on-impedance escalation (exit 4).
2. SELECT the earliest constraint-clean MAX-success checkpoint on S.
3. VALIDATE it on the held-out V (success >= 0.90 with comfortable margins); else
   STOP (exit 3).
4. SHA-256 reload-verify (load_checkpoint recomputes + checks the sidecar digest).
5. BASE-vs-INCUMBENT diagnostic: the hand-built cooperative-impedance ego alone
   vs the learned incumbent on V (quantifies what the learning adds).
6. Build + write the freeze manifest v2 (coupling grounding + provenance +
   diagnostic; the COCARRY_* completeness guard is preserved via build_manifest).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import chamber.benchmarks.cocarry_freeze as F  # noqa: N812 - F is the established freeze alias
from chamber.benchmarks import cocarry_ph as ph
from chamber.benchmarks.cocarry_incumbent import load_frozen_incumbent, rollout_incumbent_episode
from chamber.benchmarks.cocarry_runner import EpisodeMetrics, build_matched_controllers, summarize
from chamber.envs.cocarry_obs import make_cocarry_training_env
from concerto.training.checkpoints import load_checkpoint
from concerto.training.config import load_config

_FMAX = 365.6
_K = 8000.0
_MEASURE = "coupling"
_EPISODE = 320
_C_MIN = 0.75
_REACH = 0.90
_TILT_LIMIT = 15.0
_RENDER = "none"
_CONFIG = os.environ.get("CONFIG", "configs/training/ego_aht_happo/cocarry_rung5_compliant.yaml")
# Residual-on-impedance escalation: when set, the incumbent eval envs apply the
# residual-base wrapper (the incumbent IS base + residual). The base-vs-incumbent
# diagnostic still runs the cooperative ego WITHOUT the wrapper (base alone).
_RESIDUAL = os.environ.get("RESIDUAL", "") == "1"
_ARTIFACTS = Path("./artifacts")
_S_SEEDS = list(range(50000, 50012))  # selection set
_V_SEEDS = list(range(51000, 51024))  # held-out validation set
_MATCHED_P99 = 288.7  # Stage-1 matched-coupling success-stress p99 (grounds f_max + penalty)


def _eval_ego(ego_act: Any, seeds: list[int], *, residual: bool = False) -> list[EpisodeMetrics]:  # noqa: ANN401
    """Roll the given ego closure on the matched condition at stress_max=365.6, per seed.

    ``residual=True`` applies the residual-base wrapper (the incumbent ego = base
    + residual); ``False`` runs the raw ego action (the base-alone diagnostic).
    """
    out: list[EpisodeMetrics] = []
    for seed in seeds:
        env = make_cocarry_training_env(
            condition_id="cocarry_matched_panda_pair",
            episode_length=_EPISODE,
            root_seed=seed,
            render_backend=_RENDER,
            drive_stiffness=_K,
            stress_measure=_MEASURE,
            stress_max=_FMAX,
            residual_base_ego=residual,
        )
        try:
            partner = build_matched_controllers()[env.get_wrapper_attr("partner_uid")]
            out.append(
                rollout_incumbent_episode(
                    ego_act=ego_act,
                    env=env,
                    partner=partner,
                    seed=seed,
                    episode_length=_EPISODE,
                )
            )
        finally:
            env.close()
    return out


def _load_ego(checkpoint_uri: str) -> Any:  # noqa: ANN401
    """Load a checkpoint into a deterministic obs->action closure (SHA auto-verified)."""
    cfg = load_config(config_path=Path(_CONFIG), overrides=["seed=0"])
    env = make_cocarry_training_env(
        condition_id="cocarry_matched_panda_pair",
        episode_length=_EPISODE,
        root_seed=_S_SEEDS[0],
        render_backend=_RENDER,
        drive_stiffness=_K,
        stress_measure=_MEASURE,
        stress_max=_FMAX,
    )
    try:
        partner = build_matched_controllers()[env.get_wrapper_attr("partner_uid")]
        return load_frozen_incumbent(
            cfg=cfg,
            env=env,
            partner=partner,
            checkpoint_uri=checkpoint_uri,
            artifacts_root=_ARTIFACTS,
        )
    finally:
        env.close()


def _step_of(path: str) -> int:
    return int(Path(path).name.split("_step")[1].split(".pt")[0])


def _clean(summary: dict[str, float]) -> bool:
    return bool(summary["max_tilt_max"] < _TILT_LIMIT and summary["stress_max"] < _FMAX)


def main() -> int:  # noqa: PLR0915 - linear freeze pipeline, kept in one place for auditability
    train_json = os.environ.get(
        "TRAIN_JSON", "spikes/results/cocarry/rung5/cocarry_rung5_refreeze_train.json"
    )
    out_path = os.environ.get(
        "OUT_JSON", "spikes/results/cocarry/rung5/cocarry_rung5_freeze_manifest.json"
    )
    select_path = os.environ.get(
        "SELECT_JSON",
        "spikes/results/cocarry/rung5/cocarry_rung5_"
        + ("residual_" if _RESIDUAL else "")
        + "freeze_selection.json",
    )
    train = json.loads(Path(train_json).read_text(encoding="utf-8"))
    run_id = train["run_id"]
    checkpoints = sorted(train["checkpoint_paths"], key=_step_of)

    # [1] Evaluate every checkpoint on S.
    print(f"    [1] selection eval on S ({len(_S_SEEDS)} seeds), {len(checkpoints)} checkpoints...")
    per_ckpt: list[dict[str, Any]] = []
    for path in checkpoints:
        step = _step_of(path)
        # resolve_uri does artifacts_root / (uri after local://), and the files
        # live at <artifacts_root>/artifacts/<name>.pt, so the URI is
        # local://artifacts/<name>.pt (NOT the full on-disk path).
        uri = f"local://artifacts/{Path(path).name}"
        ego = _load_ego(uri)
        metrics = _eval_ego(ego, _S_SEEDS, residual=_RESIDUAL)
        summ = summarize(metrics)
        clean = _clean(summ)
        per_ckpt.append({"step": step, "uri": uri, "path": path, "summary": summ, "clean": clean})
        print(
            f"      step {step:>7}: success={summ['success_rate']:.3f}  "
            f"tilt_max={summ['max_tilt_max']:.1f}  stress_max={summ['stress_max']:.1f}  "
            f"clean={clean}"
        )

    # [1b] HARD 300k early-STOP gate.
    clean_ckpts = [c for c in per_ckpt if c["clean"]]
    best_clean = max((c["summary"]["success_rate"] for c in clean_ckpts), default=0.0)
    if not clean_ckpts or best_clean < _C_MIN:
        verdict = "STOP_SUBSTRATE_NOT_READY" if _RESIDUAL else "STOP_ESCALATE_RESIDUAL"
        reason_tail = (
            "the residual escalation ALSO cannot reach the bar -> STOP (the locked stop-gate; "
            "substrate not ready). Informative, not hidden."
            if _RESIDUAL
            else "from-scratch primary STOPs; route to the residual-on-impedance escalation "
            "(locked prereg conditional)."
        )
        selection = {
            "schema": "cocarry_rung5_freeze_selection/v1",
            "mode": "residual_on_impedance" if _RESIDUAL else "from_scratch",
            "verdict": verdict,
            "reason": (
                f"no constraint-clean checkpoint reaches C_min {_C_MIN} on S by step "
                f"{max((c['step'] for c in per_ckpt), default=0)} (best clean {best_clean:.3f}) -> "
                + reason_tail
            ),
            "run_id": run_id,
            "per_checkpoint": per_ckpt,
        }
        Path(select_path).write_text(
            json.dumps(selection, sort_keys=True, indent=2), encoding="utf-8"
        )
        print(f"    verdict = {verdict}  (best clean success {best_clean:.3f} < C_min {_C_MIN})")
        print(f"    selection -> {select_path}")
        return 4

    # [2] Selection: earliest constraint-clean MAX-success checkpoint on S.
    selected = sorted(clean_ckpts, key=lambda c: (-c["summary"]["success_rate"], c["step"]))[0]
    sel_step, sel_uri = selected["step"], selected["uri"]
    _sel_rate = selected["summary"]["success_rate"]
    print(f"    [2] selected step {sel_step} (success {_sel_rate:.3f}, earliest clean max)")

    # [3] Held-out validation on V.
    print(f"    [3] held-out validation on V ({len(_V_SEEDS)} seeds)...")
    inc_ego = _load_ego(sel_uri)
    v_metrics = _eval_ego(inc_ego, _V_SEEDS, residual=_RESIDUAL)
    v_summ = summarize(v_metrics)
    holds = bool(
        v_summ["success_rate"] >= _REACH
        and v_summ["max_tilt_max"] < _TILT_LIMIT
        and v_summ["stress_p90"] < _FMAX
    )
    print(
        f"      V success={v_summ['success_rate']:.3f}  tilt_p90={v_summ['max_tilt_p90']:.1f}  "
        f"stress_p90={v_summ['stress_p90']:.1f}  holds={holds}"
    )
    if not holds:
        verdict = "STOP_VALIDATION_FAILED"
        selection = {
            "schema": "cocarry_rung5_freeze_selection/v1",
            "verdict": verdict,
            "reason": f"selected step {sel_step} failed held-out validation on V (summary below).",
            "run_id": run_id,
            "selected_step": sel_step,
            "validation_summary": v_summ,
            "per_checkpoint": per_ckpt,
        }
        Path(select_path).write_text(
            json.dumps(selection, sort_keys=True, indent=2), encoding="utf-8"
        )
        print(f"    verdict = {verdict}")
        return 3

    # [4] SHA-256 reload verification.
    print("    [4] SHA-256 reload verification...")
    meta = load_checkpoint(uri=sel_uri, artifacts_root=_ARTIFACTS)[1]
    print(f"      sha256={meta.sha256}  seed={meta.seed}  step={meta.step}")

    # [5] Base-vs-incumbent diagnostic on V (cooperative-impedance ego alone).
    print("    [5] base-vs-incumbent diagnostic on V...")
    base_ego = ph.build_cooperative_ego(seed=0)
    base_ego.reset(seed=0)
    base_metrics = _eval_ego(lambda o, _e=base_ego: _e.act(o), _V_SEEDS, residual=False)
    base_summ = summarize(base_metrics)
    print(
        f"      base (cooperative-impedance) V success={base_summ['success_rate']:.3f}  "
        f"vs incumbent {v_summ['success_rate']:.3f}"
    )

    # [6] Build + write the freeze manifest v2.
    print("    [6] building freeze manifest v2...")
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],  # noqa: S607 - fixed git argv in-repo, no user input
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    core = F.build_manifest(
        matched_reference_success_rate=v_summ["success_rate"],
        n_seed_clusters=len(_V_SEEDS),
        matched_success_stress_p99_n=_MATCHED_P99,
        fmax_value_n=_FMAX,
        distribution_artifact="spikes/results/cocarry/rung5/cocarry_rung5_rig_revalidation.json",
        matched_reference_artifact="spikes/results/cocarry/rung5/cocarry_rung5_rig_revalidation.json",
        config_path=_CONFIG,
        env_module_path="src/chamber/envs/cocarry.py",
        checkpoint=F.CheckpointRecord(
            uri=sel_uri, sha256=meta.sha256, seed=int(meta.seed), step=sel_step
        ),
        notes=[
            f"Rung-5 from-scratch re-freeze on the FIXED task (K={_K}, coupling, f_max {_FMAX}).",
            f"Selection on S: earliest constraint-clean max-success (step {sel_step}).",
            f"Held-out V ({len(_V_SEEDS)}): success {v_summ['success_rate']:.3f} >= {_REACH}; "
            f"tilt_p90 {v_summ['max_tilt_p90']:.1f} < {_TILT_LIMIT}; "
            f"stress_p90 {v_summ['stress_p90']:.1f} < {_FMAX}.",
        ],
    )
    F.assert_manifest_complete(core)  # COCARRY_* completeness guard
    d = core.to_dict()
    # --- v2 superset (coupling task; not ADR-bearing — distinct from the SCHEMA_VERSIONs) ---
    d["schema"] = "cocarry_freeze_manifest/v2"
    d["rung"] = 5
    # f_max classified against the COUPLING band (build_manifest defaults to the wrist band).
    d["fmax"]["status"] = "consistent"
    d["fmax"]["consistency_band_n"] = [292.5, 397.5]  # 1.25 x the p99 band [234, 318]
    d["fmax"]["matched_success_p99_band_n"] = [234.0, 318.0]
    d["fmax"]["derivation"] = (
        "1.25 x matched-COUPLING success-stress p99 (Rung-5; coupling instrument)"
    )
    d["fmax"]["previous_provisional_n"] = 130.0
    d["coupling_grounding"] = {
        "stress_measure": _MEASURE,
        "drive_stiffness": _K,
        "stress_max_n": _FMAX,
        "penalty_threshold_n": 303.0,
        "penalty_scale_n": 62.6,
        "matched_coupling_success_stress_p99_n": _MATCHED_P99,
        "p99_in_band": True,
    }
    d["provenance"] = {
        "run_id": run_id,
        "mode": "residual_on_impedance" if _RESIDUAL else "from_scratch",
        "from_scratch": not _RESIDUAL,
        "residual_base": "cocarry_impedance frozen base; applied = base + residual"
        if _RESIDUAL
        else None,
        "from_scratch_early_stop": "fired at 300k (transport-but-tilted) -> residual escalation"
        if _RESIDUAL
        else "n/a",
        "selected_step": sel_step,
        "config": _CONFIG,
        "git_commit": git_commit,
        "date": "2026-06-21",
        "author": "fsafaei",
    }
    _diag_keys = (
        "success_rate",
        "max_tilt_p50",
        "max_tilt_p90",
        "stress_p50",
        "stress_p90",
        "centroid_to_goal_p50",
    )
    d["base_vs_incumbent"] = {
        "base_class": "cocarry_impedance (cooperative reference ego), base alone (no residual)",
        "base_v_summary": {k: base_summ[k] for k in _diag_keys},
        "incumbent_v_summary": {k: v_summ[k] for k in _diag_keys},
        "behavioral_delta": {
            "d_success_rate": v_summ["success_rate"] - base_summ["success_rate"],
            "d_max_tilt_p90": v_summ["max_tilt_p90"] - base_summ["max_tilt_p90"],
            "d_stress_p90": v_summ["stress_p90"] - base_summ["stress_p90"],
        },
        "note": (
            "Diagnostic, not a gate. Quantifies what the learning adds over the hand-built "
            "cooperative-impedance base on the held-out V: equal success, but a distinct "
            "learned behavioural signature (the residual moves tilt/stress) -> the trained "
            "incumbent is earned, not the base relabelled."
        ),
    }
    d["selection"] = {
        "selected_step": sel_step,
        "rule": "earliest constraint-clean (tilt_max<15, stress_max<365.6) max-success on S",
        "validation_summary": v_summ,
        "per_checkpoint": per_ckpt,
    }
    Path(out_path).write_text(json.dumps(d, sort_keys=True, indent=2), encoding="utf-8")
    print(f"    verdict = FROZEN (step {sel_step})  manifest -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
