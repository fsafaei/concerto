# SPDX-License-Identifier: Apache-2.0
"""Measured two-limb runner for the Gate-0 handover-and-place spike (ADR-026; Rev 2 §6).

Runs the matched + mismatched conditions across the clearance x takt grid for BOTH
arm-basis endpoints (fast/slow) plus the free-re-grasp diagnostic endpoint, against the
FROZEN, tagged prereg. Emits a standard :class:`chamber.evaluation.results.SpikeRun`
(schema_version 2; no bump) plus the crossover curves and the clearance-threshold /
mismatch-overlay readout (decision #1), and classifies the verdict via
:mod:`chamber.spikes.handover_place_gate0.decision`.

This module is committed at PR B as TOOLING; it performs NO measurement until invoked
against the tagged prereg at PR C (the immutable archive is written then). Determinism
(P6 / ADR-002): every env/presenter draw routes through ``derive_substream``; episodes
are paired across conditions on a shared ``initial_state_seed`` so the paired-cluster
bootstrap matches matched vs mismatched on identical env initial conditions.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from chamber.agents.handover_ego_scripted import ScriptedHandoverEgo
from chamber.envs.handover_place import (
    HANDOVER_DELTA_MIN_PP,
    HANDOVER_MATCHED_REFERENCE,
    HANDOVER_PRESENTATION_MISMATCH,
    HANDOVER_TAU_SOLV,
    make_handover_place_env,
)
from chamber.evaluation.bootstrap import (
    build_paired_episodes,
    cluster_bootstrap,
    pacluster_bootstrap,
)
from chamber.evaluation.results import ConditionPair, EpisodeResult, SpikeRun
from chamber.partners.handover_presenter import HandoverPresenterPartner, presenter_spec
from chamber.spikes.handover_place_gate0.decision import (
    REALISTIC_TAKT_BAND_S,
    binding_threshold_deg,
    classify_cell,
    classify_verdict,
    clearance_mismatch_overlay,
    mismatch_mass_clearing,
    threshold_in_sigma,
    two_crossover_window,
)
from concerto.training.seeding import derive_substream

if TYPE_CHECKING:
    from collections.abc import Sequence

#: J5 pitch half-range (deg) — the cited binding wrist axis (mirrors the Stage-0 finding).
_J5_PITCH_HALF_DEG: float = 125.0


def _condition_id(variant: str) -> str:
    return HANDOVER_MATCHED_REFERENCE if variant == "matched" else HANDOVER_PRESENTATION_MISMATCH


def run_cell_episodes(
    *,
    variant: str,
    clearance_factor: float,
    takt_s: float,
    arm_basis: str,
    place_cycle_s: float,
    regrasp_duration_s: float,
    free_regrasp: bool,
    seeds: Sequence[int],
    episodes_per_seed: int,
    env_params: dict[str, Any],
    presenter_params: dict[str, float],
    mismatch_bias_deg: float | None = None,
) -> list[EpisodeResult]:
    """Run one (variant, clearance, takt, arm-basis) cell and return its episodes (ADR-026).

    The wrist-correction range is derived per clearance level
    (``J5_pitch_half * clearance_factor``); the budget is ``takt - place_cycle`` (or
    infinite under ``free_regrasp``). Episodes are keyed by a shared
    ``initial_state_seed`` so matched/mismatched pair on identical env initial conditions.
    """
    wrist_correction_deg = _J5_PITCH_HALF_DEG * clearance_factor
    env = make_handover_place_env(
        condition_id=_condition_id(variant),
        free_regrasp=free_regrasp,
        regrasp_budget_s=takt_s - place_cycle_s,
        regrasp_duration_s=regrasp_duration_s,
        wrist_correction_deg=wrist_correction_deg,
        **env_params,
    )
    # presenter_params carry only grasp-pose distribution keys (never `seed`).
    spec = presenter_spec(variant, **presenter_params)  # type: ignore[arg-type]
    presenter = HandoverPresenterPartner(spec)
    ego = ScriptedHandoverEgo(
        translation_range_m=env_params["translation_range_m"],
        wrist_correction_deg=wrist_correction_deg,
    )
    cond = _condition_id(variant)
    results: list[EpisodeResult] = []
    for s in seeds:
        for e in range(episodes_per_seed):
            iss = int(s) * 1000 + e
            obs, _ = env.reset(seed=iss)
            presenter.reset(seed=iss)
            obs, _, _, _, _ = env.step(presenter.act(obs))
            _, _, _, _, info = env.step(ego.act(obs))
            results.append(
                EpisodeResult(
                    seed=int(s),
                    episode_idx=e,
                    initial_state_seed=iss,
                    success=bool(info["success"]),
                    metadata={
                        "condition": cond,
                        "clearance_factor": clearance_factor,
                        "takt_s": takt_s,
                        "arm_basis": arm_basis,
                        "free_regrasp": free_regrasp,
                        "mismatch_bias_deg": mismatch_bias_deg,
                        "binding_conjunct": info["binding_conjunct"],
                        "failure_mode": info["failure_mode"],
                    },
                )
            )
    return results


def _iqm_ci(
    successes_by_seed: dict[int, list[float]], *, n_boot: int, rng_name: str
) -> tuple[float, float]:
    rng = derive_substream(rng_name, root_seed=0).default_rng()
    ci = cluster_bootstrap(successes_by_seed, n_resamples=n_boot, rng=rng)
    return ci.iqm, ci.ci_low


def _gap_ci_pp(
    matched: list[EpisodeResult], mismatched: list[EpisodeResult], *, n_boot: int, rng_name: str
) -> tuple[float, float, float]:
    """Paired-cluster gap (matched - mismatched), in percentage points, with CI (ADR-008)."""
    run = SpikeRun(
        spike_id="handover_place_gate0_cell",
        prereg_sha="cell-analysis",
        git_tag="cell-analysis",
        axis="handover_place",
        sub_stage="2",
        condition_pair=ConditionPair(
            homogeneous_id=HANDOVER_MATCHED_REFERENCE,
            heterogeneous_id=HANDOVER_PRESENTATION_MISMATCH,
        ),
        seeds=sorted({ep.seed for ep in matched}),
        episode_results=[*matched, *mismatched],
    )
    pairs = build_paired_episodes(run)
    rng = derive_substream(rng_name, root_seed=0).default_rng()
    ci = pacluster_bootstrap(pairs, n_resamples=n_boot, rng=rng)
    return ci.iqm * 100.0, ci.ci_low * 100.0, ci.ci_high * 100.0


def _successes_by_seed(episodes: list[EpisodeResult]) -> dict[int, list[float]]:
    out: dict[int, list[float]] = {}
    for ep in episodes:
        out.setdefault(ep.seed, []).append(1.0 if ep.success else 0.0)
    return out


def clearance_threshold_overlay(
    *,
    clearance_factors: Sequence[float],
    angular_window_deg: float,
    matched_grasp_pose_sigma_deg: float,
    mismatched_grasp_pose_bias_deg: float,
    mismatched_grasp_pose_sigma_deg: float,
) -> list[dict[str, float]]:
    """Per-clearance binding threshold (deg + sigma) + realistic mismatch mass (decision #1)."""
    rows: list[dict[str, float]] = []
    for f in clearance_factors:
        wrist = _J5_PITCH_HALF_DEG * f
        thr_deg = binding_threshold_deg(wrist, angular_window_deg)
        rows.append(
            {
                "clearance_factor": f,
                "wrist_correction_deg": wrist,
                "binding_threshold_deg": thr_deg,
                "binding_threshold_sigma": threshold_in_sigma(
                    thr_deg, matched_grasp_pose_sigma_deg
                ),
                "mismatch_mass_clearing": mismatch_mass_clearing(
                    thr_deg, mismatched_grasp_pose_bias_deg, mismatched_grasp_pose_sigma_deg
                ),
            }
        )
    return rows


def analyze(
    episodes: list[EpisodeResult],
    *,
    clearance_factors: Sequence[float],
    takt_grid_s: Sequence[float],
    arm_bases: Sequence[str],
    n_boot: int,
    overlay: list[dict[str, Any]],
    mismatch_biases_deg: Sequence[float] = (),
    tau_solv: float = HANDOVER_TAU_SOLV,
    delta_min_pp: float = HANDOVER_DELTA_MIN_PP,
) -> dict[str, Any]:
    """Build the per-cell crossover curves + verdict from the raw episodes (ADR-026 §Decision).

    Cells are keyed by (mismatch_bias, arm, clearance, takt) — the clearance and mismatch
    knobs are BOTH swept (decision #2). The matched reference is shared across biases; the
    gap pairs the matched set with each bias's mismatched set on identical initial
    conditions. Also returns the (clearance x mismatch) measured coupling region.
    """

    def _sel(eps: list[EpisodeResult], filt: dict[str, Any]) -> list[EpisodeResult]:
        return [e for e in eps if all(e.metadata.get(k) == v for k, v in filt.items())]

    biases: Sequence[float] = mismatch_biases_deg or (math.nan,)
    curves: list[dict[str, Any]] = []
    any_indeterminate = False
    free_gap_low = -math.inf
    matched_solvable_at_realistic = False
    lo, hi = REALISTIC_TAKT_BAND_S

    for bias in biases:
        for arm in arm_bases:
            for f in clearance_factors:
                takt_cells: list[tuple[float, Any]] = []
                for t in takt_grid_s:
                    mb = {
                        "arm_basis": arm,
                        "clearance_factor": f,
                        "takt_s": t,
                        "free_regrasp": False,
                    }
                    matched = _sel(episodes, {**mb, "condition": HANDOVER_MATCHED_REFERENCE})
                    mm_filt = {**mb, "condition": HANDOVER_PRESENTATION_MISMATCH}
                    if mismatch_biases_deg:
                        mm_filt["mismatch_bias_deg"] = bias
                    mismatched = _sel(episodes, mm_filt)
                    if not matched or not mismatched:
                        continue
                    m_iqm, m_low = _iqm_ci(
                        _successes_by_seed(matched),
                        n_boot=n_boot,
                        rng_name=f"eval.matched.{arm}.{f}.{t}",
                    )
                    gap_iqm, gap_low, gap_high = _gap_ci_pp(
                        matched,
                        mismatched,
                        n_boot=n_boot,
                        rng_name=f"eval.gap.{bias}.{arm}.{f}.{t}",
                    )
                    verdict = classify_cell(
                        matched_ci_low=m_low,
                        gap_ci_low_pp=gap_low,
                        gap_ci_high_pp=gap_high,
                        tau_solv=tau_solv,
                        delta_min_pp=delta_min_pp,
                    )
                    any_indeterminate = any_indeterminate or verdict.indeterminate
                    if verdict.solvable and lo <= t <= hi:
                        matched_solvable_at_realistic = True
                    takt_cells.append((t, verdict))
                    curves.append(
                        {
                            "mismatch_bias_deg": bias,
                            "arm_basis": arm,
                            "clearance_factor": f,
                            "takt_s": t,
                            "matched_iqm": m_iqm,
                            "matched_ci_low": m_low,
                            "gap_pp": gap_iqm,
                            "gap_ci_low_pp": gap_low,
                            "gap_ci_high_pp": gap_high,
                            "solvable": verdict.solvable,
                            "coupling_valid": verdict.coupling_valid,
                            "washout": verdict.washout,
                            "indeterminate": verdict.indeterminate,
                        }
                    )
                curves.append(
                    {
                        "mismatch_bias_deg": bias,
                        "arm_basis": arm,
                        "clearance_factor": f,
                        "crossover_window": two_crossover_window(takt_cells)._asdict(),
                    }
                )

    # free-re-grasp diagnostic (intrinsic if it persists): max free gap-low over biases.
    for bias in biases:
        fmm_filt: dict[str, Any] = {
            "condition": HANDOVER_PRESENTATION_MISMATCH,
            "free_regrasp": True,
        }
        if mismatch_biases_deg:
            fmm_filt["mismatch_bias_deg"] = bias
        fm = _sel(episodes, {"condition": HANDOVER_MATCHED_REFERENCE, "free_regrasp": True})
        fmm = _sel(episodes, fmm_filt)
        if fm and fmm:
            _, gl, _ = _gap_ci_pp(fm, fmm, n_boot=n_boot, rng_name=f"eval.gap.free.{bias}")
            free_gap_low = max(free_gap_low, gl)

    # (clearance x mismatch) measured coupling region: coupling-valid at any takt/arm.
    region: list[dict[str, Any]] = []
    for f in clearance_factors:
        for bias in biases:
            cells = [
                c
                for c in curves
                if c.get("clearance_factor") == f
                and c.get("mismatch_bias_deg") == bias
                and "takt_s" in c
            ]
            region.append(
                {
                    "clearance_factor": f,
                    "mismatch_bias_deg": bias,
                    "coupling_valid_any_takt": any(c["coupling_valid"] for c in cells),
                    "max_gap_ci_low_pp": max((c["gap_ci_low_pp"] for c in cells), default=math.nan),
                }
            )

    all_takt_cells: list[tuple[float, Any]] = [
        (
            c["takt_s"],
            classify_cell(
                matched_ci_low=c["matched_ci_low"],
                gap_ci_low_pp=c["gap_ci_low_pp"],
                gap_ci_high_pp=c["gap_ci_high_pp"],
                tau_solv=tau_solv,
                delta_min_pp=delta_min_pp,
            ),
        )
        for c in curves
        if "takt_s" in c
    ]
    overall_window = two_crossover_window(all_takt_cells)
    verdict = classify_verdict(
        window=overall_window,
        free_regrasp_gap_ci_low_pp=free_gap_low,
        matched_solvable_at_realistic=matched_solvable_at_realistic,
        any_indeterminate=any_indeterminate,
        delta_min_pp=delta_min_pp,
    )
    return {
        "curves": curves,
        "overall_window": overall_window._asdict(),
        "free_regrasp_gap_ci_low_pp": free_gap_low,
        "clearance_threshold_overlay": overlay,
        "mismatch_coupling_region": region,
        "verdict": verdict,
    }


def run_gate0(params: dict[str, Any]) -> tuple[SpikeRun, dict[str, Any]]:
    """Run the full Gate-0 grid and analyse it (ADR-026; Rev 2 §6). NO archive write.

    ``params`` is the frozen prereg's committed numbers (loaded by :func:`main`). Runs
    matched + mismatched across the clearance x takt grid for both arm-basis endpoints,
    plus the free-re-grasp diagnostic per clearance, and returns the
    ``(SpikeRun, analysis)`` pair. Determinism is inherited from the env/presenter
    substreams; this function performs measurement and is invoked ONLY at PR C against
    the tagged prereg.
    """
    seeds: Sequence[int] = params["seeds"]
    k: int = params["episodes_per_seed"]
    clearances: Sequence[float] = params["clearance_factor_sweep"]
    takts: Sequence[float] = params["takt_grid_s"]
    arm_bases: dict[str, dict[str, float]] = params["arm_bases"]
    env_params: dict[str, Any] = params["env_params"]
    n_boot: int = params["n_boot"]

    mismatch_bias_sweep: Sequence[float] = params["mismatched_grasp_pose_bias_sweep_deg"]
    matched_pp: dict[str, float] = params["matched_presenter_params"]
    mismatched_pp_base: dict[str, float] = params["mismatched_presenter_params"]
    free_regrasp_duration_s = arm_bases["fast"]["regrasp_duration_s"]

    episodes: list[EpisodeResult] = []
    # Matched reference (shared across mismatch biases).
    for arm, ab in arm_bases.items():
        for f in clearances:
            for t in takts:
                episodes += run_cell_episodes(
                    variant="matched",
                    clearance_factor=f,
                    takt_s=t,
                    arm_basis=arm,
                    place_cycle_s=ab["place_cycle_s"],
                    regrasp_duration_s=ab["regrasp_duration_s"],
                    free_regrasp=False,
                    seeds=seeds,
                    episodes_per_seed=k,
                    env_params=env_params,
                    presenter_params=matched_pp,
                )
    for f in clearances:  # matched free-re-grasp endpoint (arm-basis-independent)
        episodes += run_cell_episodes(
            variant="matched",
            clearance_factor=f,
            takt_s=takts[0],
            arm_basis="free",
            place_cycle_s=0.0,
            regrasp_duration_s=free_regrasp_duration_s,
            free_regrasp=True,
            seeds=seeds,
            episodes_per_seed=k,
            env_params=env_params,
            presenter_params=matched_pp,
        )
    # Mismatched, swept over the grasp-pose bias (decision #2).
    for bias in mismatch_bias_sweep:
        pp = {**mismatched_pp_base, "grasp_pose_bias_deg": bias}
        for arm, ab in arm_bases.items():
            for f in clearances:
                for t in takts:
                    episodes += run_cell_episodes(
                        variant="mismatched",
                        clearance_factor=f,
                        takt_s=t,
                        arm_basis=arm,
                        place_cycle_s=ab["place_cycle_s"],
                        regrasp_duration_s=ab["regrasp_duration_s"],
                        free_regrasp=False,
                        seeds=seeds,
                        episodes_per_seed=k,
                        env_params=env_params,
                        presenter_params=pp,
                        mismatch_bias_deg=bias,
                    )
        for f in clearances:
            episodes += run_cell_episodes(
                variant="mismatched",
                clearance_factor=f,
                takt_s=takts[0],
                arm_basis="free",
                place_cycle_s=0.0,
                regrasp_duration_s=free_regrasp_duration_s,
                free_regrasp=True,
                seeds=seeds,
                episodes_per_seed=k,
                env_params=env_params,
                presenter_params=pp,
                mismatch_bias_deg=bias,
            )

    overlay = clearance_mismatch_overlay(
        clearance_factors=list(clearances),
        mismatch_biases_deg=list(mismatch_bias_sweep),
        angular_window_deg=env_params["angular_window_deg"],
        matched_sigma_deg=params["matched_grasp_pose_sigma_deg"],
        mismatch_sigma_deg=params["mismatched_grasp_pose_sigma_deg"],
        j5_pitch_half_deg=_J5_PITCH_HALF_DEG,
    )
    analysis = analyze(
        episodes,
        clearance_factors=clearances,
        takt_grid_s=takts,
        arm_bases=list(arm_bases),
        mismatch_biases_deg=mismatch_bias_sweep,
        n_boot=n_boot,
        overlay=overlay,
    )
    spike_run = SpikeRun(
        spike_id="handover_place_gate0",
        prereg_sha=params["prereg_sha"],
        git_tag=params["git_tag"],
        axis="handover_place",
        sub_stage="2",
        condition_pair=ConditionPair(
            homogeneous_id=HANDOVER_MATCHED_REFERENCE,
            heterogeneous_id=HANDOVER_PRESENTATION_MISMATCH,
        ),
        seeds=list(seeds),
        episode_results=episodes,
    )
    return spike_run, analysis
