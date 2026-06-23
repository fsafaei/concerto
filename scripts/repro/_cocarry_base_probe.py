# SPDX-License-Identifier: Apache-2.0
# pyright: reportPrivateImportUsage=false
"""Phase-2 co-carry base-difficulty probe (pre-registered; ADR-026; R-2026-06-C).

Eval-only, NO training, NO residual, NO heterogeneity. One competent hand-built
MATCHED pair (cooperative-impedance ego + matched impedance partner -- the
'base' config of the Rung-5 base_robustness_control); the ONLY thing varied is
task difficulty. Answers the board base-failure criterion: does a harder
co-carry regime exist where the matched base itself stops trivially solving the
task AND degrades for coordination reasons (graceful coordination-strain)
rather than numerical blow-up (the Rung-4 over-coupling wall)?

The grid, seeds, fixed base config, artifact-vs-real classification thresholds,
and the falsifiable-both-ways decision rule are read from the LOCKED, git-tagged
pre-registration (spikes/results/cocarry/base_probe/cocarry_base_probe_prereg.json,
tag prereg-cocarry-base-probe-2026-06-22) -- this driver only executes it.

On main, make_cocarry_training_env has no stress_max param (a Rung-5 addition),
so the env-internal success flag gates at the wrist ceiling 130 N and is IGNORED;
success is computed POST-HOC via the pure chamber.envs.cocarry.evaluate_cocarry_
success(...) at the grounded coupling f_max (365.6 N) on the recorded raw metrics.
The matched hand-built controllers ignore reward, so rollout physics are
byte-identical regardless of the (unused) stress_max/penalty params.

Primary stiffness sweep: real rollouts at each K, per-step trajectories recorded
(the artifact-vs-real test). Secondary goal-radius + tilt sweeps: post-hoc
re-evaluation of the frozen predicate on the K=8000 base rollouts (physics
unchanged => cannot be numerically unstable).

ADR-026 §Decision 1-2 (coupling-validity + the joint predicate, never weakened);
ADR-026 §Open-questions (coupling stiffness is a task parameter); ADR-026
§Validation-criteria. Phase-2, NON-gating. Do not merge -- founder review.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

import chamber.envs.cocarry as cc

# cocarry_ph imports the matched impedance controller family at module level
# (decorator registration), so `cocarry_impedance` is resolvable here.
from chamber.benchmarks import cocarry_ph as ph
from chamber.envs.cocarry_obs import make_cocarry_training_env

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[2]
_PROBE_DIR = _REPO / "spikes/results/cocarry/base_probe"
_PREREG = _PROBE_DIR / "cocarry_base_probe_prereg.json"
_OUT = _PROBE_DIR / "cocarry_base_probe_measurement.json"
_TRAJ_OUT = _PROBE_DIR / "cocarry_base_probe_trajectories.json"
_RENDER = "none"

# Frozen predicate thresholds (NEVER weakened; the secondary sweeps only tighten
# these for the post-hoc re-evaluation). f_max is the Rung-4c-grounded coupling
# ceiling. These mirror chamber.envs.cocarry's module constants / Rung-4c.
_FROZEN_GOAL_THRESH_M = 0.10
_FROZEN_TILT_MAX_DEG = 15.0
_FMAX_COUPLING_N = 365.6080997467041


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rollout_with_trajectory(
    *,
    env: Any,  # noqa: ANN401
    ego: Any,  # noqa: ANN401
    partner: Any,  # noqa: ANN401
    seed: int,
    episode_length: int,
) -> dict[str, Any]:
    """One matched-pair episode, recording per-step instantaneous trajectories.

    Returns the episode metrics the frozen predicate reads (final centroid, the
    settle-windowed running maxima of tilt + coupling stress, both_static) plus
    the per-step instantaneous (tilt_deg, coupling stress, centroid) arrays the
    artifact-vs-real test needs to tell graceful strain from numerical blow-up.
    """
    ego_uid = env.get_wrapper_attr("ego_uid")
    partner_uid = env.get_wrapper_attr("partner_uid")
    get_tel = env.get_wrapper_attr("get_telemetry")
    obs, _ = env.reset(seed=seed)
    ego.reset(seed=seed)
    partner.reset(seed=seed)
    tilt_steps: list[float] = []
    stress_steps: list[float] = []
    centroid_steps: list[float] = []
    info: dict[str, Any] = {}
    n_steps = 0
    for _ in range(episode_length):
        action = {
            ego_uid: np.asarray(ego.act(obs), dtype=np.float32),
            partner_uid: partner.act(obs),
        }
        obs, _, terminated, truncated, info = env.step(action)
        tel = get_tel()
        tilt_steps.append(float(np.asarray(tel["tilt_deg"].detach().cpu()).reshape(-1)[0]))
        stress_steps.append(float(np.asarray(tel["stress_proxy"].detach().cpu()).reshape(-1)[0]))
        centroid_steps.append(
            float(np.asarray(tel["centroid_to_goal"].detach().cpu()).reshape(-1)[0])
        )
        n_steps += 1
        if bool(np.asarray(terminated).reshape(-1)[0]) or bool(
            np.asarray(truncated).reshape(-1)[0]
        ):
            break
    tel = get_tel()
    return {
        "seed": seed,
        "centroid_to_goal": ph._to_float(tel["centroid_to_goal"]),
        "max_tilt_deg": ph._to_float(tel["max_tilt_deg"]),
        "max_stress_proxy": ph._to_float(tel["max_stress_proxy"]),
        "both_static": ph._info_bool(info, "both_static"),
        "n_steps": n_steps,
        "tilt_steps": tilt_steps,
        "stress_steps": stress_steps,
        "centroid_steps": centroid_steps,
    }


def _success_at(
    ep: dict[str, Any], *, goal_thresh: float, tilt_max_deg: float, stress_max: float
) -> bool:
    """Apply the frozen joint predicate to one episode's raw metrics (post-hoc)."""
    return bool(
        cc.evaluate_cocarry_success(
            centroid_to_goal_dist=ep["centroid_to_goal"],
            max_tilt_deg=ep["max_tilt_deg"],
            max_stress_proxy=ep["max_stress_proxy"],
            both_static=ep["both_static"],
            goal_thresh=goal_thresh,
            tilt_max_deg=tilt_max_deg,
            stress_max=stress_max,
        )
    )


def _episode_finite(ep: dict[str, Any]) -> bool:
    return bool(
        np.all(np.isfinite(ep["tilt_steps"]))
        and np.all(np.isfinite(ep["stress_steps"]))
        and np.isfinite(ep["max_tilt_deg"])
        and np.isfinite(ep["max_stress_proxy"])
        and np.isfinite(ep["centroid_to_goal"])
    )


def _setting_summary(eps: list[dict[str, Any]], successes: list[bool]) -> dict[str, Any]:
    max_tilts = np.array([e["max_tilt_deg"] for e in eps], dtype=np.float64)
    max_stress = np.array([e["max_stress_proxy"] for e in eps], dtype=np.float64)
    centroids = np.array([e["centroid_to_goal"] for e in eps], dtype=np.float64)
    return {
        "n": len(eps),
        "success_rate": float(np.mean(successes)),
        "n_success": int(np.sum(successes)),
        "fail_placed_at_010": int(
            sum(1 for e in eps if e["centroid_to_goal"] > _FROZEN_GOAL_THRESH_M)
        ),
        "fail_level_at_15": int(sum(1 for e in eps if e["max_tilt_deg"] >= _FROZEN_TILT_MAX_DEG)),
        "fail_unstressed_at_fmax": int(
            sum(1 for e in eps if e["max_stress_proxy"] >= _FMAX_COUPLING_N)
        ),
        "fail_static": int(sum(1 for e in eps if not e["both_static"])),
        "centroid_to_goal_p50": float(np.percentile(centroids, 50)),
        "centroid_to_goal_p90": float(np.percentile(centroids, 90)),
        "centroid_to_goal_max": float(np.max(centroids)),
        "max_tilt_deg_p50": float(np.percentile(max_tilts, 50)),
        "max_tilt_deg_p90": float(np.percentile(max_tilts, 90)),
        "max_tilt_deg_max": float(np.max(max_tilts)),
        "stress_p50": float(np.percentile(max_stress, 50)),
        "stress_p90": float(np.percentile(max_stress, 90)),
        "stress_max": float(np.max(max_stress)),
        "stress_median_of_episode_max": float(np.median(max_stress)),
        "tilt_median_of_episode_max": float(np.median(max_tilts)),
    }


def _classify_stable(success_rate: float) -> str:
    if success_rate >= 0.95:  # noqa: PLR2004 (pre-registered band edge)
        return "FEASIBLE_TRIVIAL"
    if success_rate > 0.90:  # noqa: PLR2004
        return "MARGINAL"
    if success_rate >= 0.10:  # noqa: PLR2004
        return "HARD_BUT_FEASIBLE"
    return "STABLE_INFEASIBLE"


def _classify_setting(
    summary: dict[str, Any], *, all_finite: bool, thresholds: dict[str, Any]
) -> dict[str, Any]:
    """Artifact-vs-real classification per the locked prereg."""
    stress_artifact_n = thresholds["stress_artifact_n"]
    tilt_artifact_deg = thresholds["tilt_artifact_deg"]
    nonfinite = not all_finite
    catastrophic_stress = summary["stress_median_of_episode_max"] >= stress_artifact_n
    dropped_bar = summary["tilt_median_of_episode_max"] >= tilt_artifact_deg
    artifact = bool(nonfinite or catastrophic_stress or dropped_bar)
    if artifact:
        reasons = []
        if nonfinite:
            reasons.append("non-finite tilt/stress")
        if catastrophic_stress:
            reasons.append(
                f"median episode-max stress {summary['stress_median_of_episode_max']:.0f} N "
                f">= {stress_artifact_n:.0f} N (3x f_max)"
            )
        if dropped_bar:
            reasons.append(
                f"median episode-max tilt {summary['tilt_median_of_episode_max']:.1f} deg "
                f">= {tilt_artifact_deg:.0f} deg"
            )
        return {
            "physics_stable": False,
            "classification": "NUMERICAL_ARTIFACT",
            "reasons": reasons,
        }
    return {
        "physics_stable": True,
        "classification": _classify_stable(summary["success_rate"]),
        "reasons": [],
    }


def _aggregate_trajectory(eps: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-step median/p90/max across seeds (compact, distinguishes graceful vs blow-up)."""
    max_len = max(len(e["tilt_steps"]) for e in eps)

    def _pad(arr: list[float]) -> np.ndarray:
        a = np.array(arr, dtype=np.float64)
        if len(a) < max_len:  # episode terminated early: hold the last value
            a = np.concatenate([a, np.full(max_len - len(a), a[-1] if len(a) else np.nan)])
        return a

    tilt = np.vstack([_pad(e["tilt_steps"]) for e in eps])
    stress = np.vstack([_pad(e["stress_steps"]) for e in eps])
    centroid = np.vstack([_pad(e["centroid_steps"]) for e in eps])
    return {
        "n_steps": max_len,
        "tilt_deg_p50": np.nanpercentile(tilt, 50, axis=0).round(4).tolist(),
        "tilt_deg_p90": np.nanpercentile(tilt, 90, axis=0).round(4).tolist(),
        "tilt_deg_max": np.nanmax(tilt, axis=0).round(4).tolist(),
        "stress_n_p50": np.nanpercentile(stress, 50, axis=0).round(3).tolist(),
        "stress_n_p90": np.nanpercentile(stress, 90, axis=0).round(3).tolist(),
        "stress_n_max": np.nanmax(stress, axis=0).round(3).tolist(),
        "centroid_m_p50": np.nanpercentile(centroid, 50, axis=0).round(5).tolist(),
    }


def main() -> int:  # noqa: PLR0912, PLR0915 - linear measurement pipeline, kept auditable here
    prereg = json.loads(_PREREG.read_text("utf-8"))
    base = prereg["fixed_base_config"]
    seeds: list[int] = list(prereg["seeds"])
    episode = int(base["episode_length"])
    fmax = float(base["fmax_coupling_n"])
    stress_measure = base["stress_measure"]
    condition = base["condition_id"]
    thresholds = prereg["artifact_vs_real_classification"]
    k_grid = list(prereg["sweeps"]["primary_stiffness"]["grid"])
    radius_grid = list(prereg["sweeps"]["secondary_goal_radius"]["grid"])
    tilt_grid = list(prereg["sweeps"]["secondary_tilt"]["grid"])
    base_k = float(base["baseline_drive_stiffness"])

    print(f"  base-difficulty probe: matched pair, {len(seeds)} seeds, f_max={fmax:.1f} N")
    print(f"  primary stiffness grid: {k_grid}")

    # --- Primary stiffness sweep (real rollouts; per-step trajectories) ---
    stiffness_results: dict[str, Any] = {}
    trajectories: dict[str, Any] = {}
    base_k_eps: list[dict[str, Any]] = []
    t0 = time.time()
    for k in k_grid:
        tk = time.time()
        eps: list[dict[str, Any]] = []
        for s in seeds:
            env = make_cocarry_training_env(
                condition_id=condition,
                episode_length=episode,
                root_seed=s,
                render_backend=_RENDER,
                drive_stiffness=float(k),
                stress_measure=stress_measure,
            )
            try:
                ego = ph.build_cooperative_ego(seed=s)
                partner = ph.build_partner_seat(
                    "cocarry_impedance", seed=s, partner_uid="panda_partner"
                )
                eps.append(
                    _rollout_with_trajectory(
                        env=env, ego=ego, partner=partner, seed=s, episode_length=episode
                    )
                )
            finally:
                env.close()
        successes = [
            _success_at(e, goal_thresh=0.10, tilt_max_deg=15.0, stress_max=fmax) for e in eps
        ]
        summary = _setting_summary(eps, successes)
        all_finite = all(_episode_finite(e) for e in eps)
        cls = _classify_setting(summary, all_finite=all_finite, thresholds=thresholds)
        key = f"{k:.0f}"
        stiffness_results[key] = {
            "drive_stiffness_npm": float(k),
            "summary": summary,
            "all_finite": all_finite,
            **cls,
            "per_seed": [
                {
                    "seed": e["seed"],
                    "success": successes[i],
                    "centroid_to_goal": round(e["centroid_to_goal"], 5),
                    "max_tilt_deg": round(e["max_tilt_deg"], 3),
                    "max_stress_proxy": round(e["max_stress_proxy"], 2),
                    "both_static": e["both_static"],
                    "n_steps": e["n_steps"],
                }
                for i, e in enumerate(eps)
            ],
        }
        trajectories[key] = _aggregate_trajectory(eps)
        if float(k) == base_k:
            base_k_eps = eps
        print(
            f"    K={k:>8.0f}  success={summary['success_rate']:.3f}  "
            f"stress(p50/p90/max)={summary['stress_p50']:.0f}/{summary['stress_p90']:.0f}/"
            f"{summary['stress_max']:.0f}  tilt_p90={summary['max_tilt_deg_p90']:.1f}  "
            f"-> {cls['classification']}  [{time.time() - tk:.0f}s]"
        )

    if not base_k_eps:
        msg = f"base stiffness {base_k} not in grid {k_grid}; cannot run secondary sweeps"
        raise RuntimeError(msg)

    # --- Secondary goal-radius sweep (post-hoc on the K=8000 base rollouts) ---
    radius_results: dict[str, Any] = {}
    for r in radius_grid:
        succ = [
            _success_at(e, goal_thresh=float(r), tilt_max_deg=_FROZEN_TILT_MAX_DEG, stress_max=fmax)
            for e in base_k_eps
        ]
        sr = float(np.mean(succ))
        radius_results[f"{r:.2f}"] = {
            "goal_thresh_m": float(r),
            "success_rate": sr,
            "n_success": int(np.sum(succ)),
            "physics_stable": True,
            "classification": _classify_stable(sr),
        }
        print(f"    radius={r:.2f} m  success={sr:.3f}  -> {_classify_stable(sr)}")

    # --- Secondary tilt-limit sweep (post-hoc on the K=8000 base rollouts) ---
    tilt_results: dict[str, Any] = {}
    for t in tilt_grid:
        succ = [
            _success_at(
                e, goal_thresh=_FROZEN_GOAL_THRESH_M, tilt_max_deg=float(t), stress_max=fmax
            )
            for e in base_k_eps
        ]
        sr = float(np.mean(succ))
        tilt_results[f"{t:.0f}"] = {
            "tilt_max_deg": float(t),
            "success_rate": sr,
            "n_success": int(np.sum(succ)),
            "physics_stable": True,
            "classification": _classify_stable(sr),
        }
        print(f"    tilt={t:.0f} deg  success={sr:.3f}  -> {_classify_stable(sr)}")

    # --- Verdict (pre-committed, falsifiable both ways) ---
    bands: list[dict[str, Any]] = []
    for key, res in stiffness_results.items():
        if res["classification"] == "HARD_BUT_FEASIBLE":
            bands.append(
                {
                    "knob": "drive_stiffness",
                    "setting": f"K={key} N/m",
                    "kind": "physics-gets-hard",
                    "success_rate": res["summary"]["success_rate"],
                }
            )
    for key, res in radius_results.items():
        if res["classification"] == "HARD_BUT_FEASIBLE":
            bands.append(
                {
                    "knob": "goal_radius",
                    "setting": f"radius={key} m",
                    "kind": "precision-tolerance",
                    "success_rate": res["success_rate"],
                }
            )
    for key, res in tilt_results.items():
        if res["classification"] == "HARD_BUT_FEASIBLE":
            bands.append(
                {
                    "knob": "tilt_limit",
                    "setting": f"tilt={key} deg",
                    "kind": "precision-tolerance",
                    "success_rate": res["success_rate"],
                }
            )

    any_artifact = any(
        r["classification"] == "NUMERICAL_ARTIFACT" for r in stiffness_results.values()
    )
    stiffness_band = [b for b in bands if b["knob"] == "drive_stiffness"]
    if bands:
        verdict = "HARD_BUT_FEASIBLE_BAND_EXISTS"
        verdict_note = (
            f"{len(bands)} hard-but-feasible (graceful, success in [0.10,0.90], stable) setting(s) "
            f"found: {[b['setting'] for b in bands]}. A genuinely-hard regime exists; testing "
            "heterogeneity there is meaningful (the full Option-2 test, properly instrumented per "
            "R-2026-06-C). Band kind(s): "
            + ", ".join(sorted({b["kind"] for b in bands}))
            + (
                ". NOTE: the coupling-STIFFNESS knob itself "
                + ("DID" if stiffness_band else "did NOT")
                + " produce a graceful band -- "
                + (
                    "the physics genuinely strains the matched controller."
                    if stiffness_band
                    else "the band(s) come only from tightening the success-predicate tolerance "
                    "(precision), not from the coupling physics getting harder."
                )
            )
        )
    else:
        verdict = "NO_FEASIBLE_HARD_REGIME_VIA_THESE_KNOBS"
        verdict_note = (
            "No setting (stiffness OR predicate-tightening) put the matched base into a graceful "
            "hard-but-feasible band (success in [0.10,0.90], stable). "
            + (
                "Stiffness transitioned trivial -> NUMERICAL_ARTIFACT (over-coupling "
                "blow-up), never a graceful band. "
                if any_artifact
                else "Stiffness stayed feasible-trivial across the whole grid. "
            )
            + "No buildable co-carry regime makes coordination strain a competent controller "
            "via these knobs -> supports concluding + reframing (a contact-rich DIFFERENT task "
            "class is a separate, bigger, gated decision -- not more stiffness)."
        )

    out = {
        "schema": "cocarry_base_probe_measurement/v1",
        "stage": prereg["stage"],
        "prereg": "spikes/results/cocarry/base_probe/cocarry_base_probe_prereg.json",
        "prereg_tag": "prereg-cocarry-base-probe-2026-06-22",
        "prereg_sha256": _sha256_file(_PREREG),
        "env_module_sha256": _sha256_file(_REPO / "src/chamber/envs/cocarry.py"),
        "fmax_coupling_n": fmax,
        "seeds": seeds,
        "n_seeds": len(seeds),
        "primary_stiffness_sweep": stiffness_results,
        "secondary_goal_radius_sweep": radius_results,
        "secondary_tilt_sweep": tilt_results,
        "hard_but_feasible_bands": bands,
        "verdict": verdict,
        "verdict_note": verdict_note,
        "trajectories_artifact": (
            "spikes/results/cocarry/base_probe/cocarry_base_probe_trajectories.json"
        ),
        "governance": (
            "Eval-only; no training/residual/heterogeneity; matched pair; predicate never "
            "weakened. Phase-2, NON-gating. Prior artifacts immutable (I8); no schema bump (I9)."
        ),
    }
    _OUT.write_text(json.dumps(out, sort_keys=True, indent=2), encoding="utf-8")
    traj_out = {
        "schema": "cocarry_base_probe_trajectories/v1",
        "note": (
            "Per-stiffness-setting per-step aggregated (across seeds) instantaneous "
            "trajectories for the artifact-vs-real test: tilt (deg), coupling stress (N), "
            "centroid (m). Graceful coordination-strain = bounded, smoothly-rising; numerical "
            "over-coupling artifact = catastrophic spikes / divergence."
        ),
        "fmax_coupling_n": fmax,
        "by_drive_stiffness_npm": trajectories,
    }
    _TRAJ_OUT.write_text(json.dumps(traj_out, sort_keys=True, indent=2), encoding="utf-8")

    print(f"\n  VERDICT = {verdict}")
    print(f"  {verdict_note}")
    print(f"  measurement -> {_OUT}")
    print(f"  trajectories -> {_TRAJ_OUT}")
    print(f"  total wall {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
