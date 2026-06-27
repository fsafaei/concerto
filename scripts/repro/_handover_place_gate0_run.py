# SPDX-License-Identifier: Apache-2.0
"""Gate-0 handover-and-place MEASURED two-limb slice + immutable archive (ADR-026; PR C).

Runs the FROZEN, tagged prereg (``prereg-handover-place-gate0-rev2-2026-06-26``): blob-SHA
verified before the first measurement, deterministic via ``derive_substream`` (P6), launch
code SHA captured once. Writes the write-once archive (I8) and the verdict report.

Decision rule + parameters are read from ``spikes/preregistration/handover_place/gate0.yaml``
and NOT re-derived here. The single canonical cell rule (two-sided 95% CI; COUPLING_VALID
iff gap CI_lower >= delta_min, WASHOUT iff gap CI_upper < delta_min) is applied verbatim by
``chamber.spikes.handover_place_gate0.decision.classify_cell`` via the runner.

Repro:  uv run python scripts/repro/_handover_place_gate0_run.py
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml

from chamber.envs.handover_place import (
    HANDOVER_DEFAULT_ANGULAR_STIFFNESS_N_PER_DEG,
    HANDOVER_DEFAULT_CONTACT_STIFFNESS_N_PER_M,
    HANDOVER_DEFAULT_TRANSLATION_RANGE_M,
    HANDOVER_MATCHED_REFERENCE,
    HANDOVER_PRESENTATION_MISMATCH,
    HANDOVER_TAU_SOLV,
)
from chamber.spikes.handover_place_gate0.decision import REALISTIC_TAKT_BAND_S
from chamber.spikes.handover_place_gate0.runner import run_gate0
from concerto.training.logging import compute_run_metadata

if TYPE_CHECKING:
    from collections.abc import Callable

    from chamber.evaluation.results import EpisodeResult, SpikeRun

_REPO = Path(__file__).resolve().parents[2]
_PREREG = _REPO / "spikes/preregistration/handover_place/gate0.yaml"
_TAG = "prereg-handover-place-gate0-rev2-2026-06-26"
_DATE = "2026-06-26"
_ARCHIVE = _REPO / f"spikes/results/handover-place-gate0-{_DATE}"
_HIGH_BRACKET_SIGMA_U = 1.0  # the sigma_u=1.0 power-sim bracket n=20 was sized against
_MIN_SEEDS_FOR_STD = 2
_BAND = list(REALISTIC_TAKT_BAND_S)
_MATCHED = HANDOVER_MATCHED_REFERENCE
_MISMATCH = HANDOVER_PRESENTATION_MISMATCH


def _git(*args: str) -> str:
    return subprocess.run(  # noqa: S603 (fixed argv, no shell, trusted git args)
        ["git", *args],  # noqa: S607 (git resolved from PATH; standard repo tooling)
        cwd=_REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def verify_prereg_blob() -> str:
    """Blob-SHA lock (ADR-007 §Discipline): on-disk prereg blob == the tag's blob.

    Returns the verified blob SHA; raises ``RuntimeError`` on mismatch / missing tag (the
    doc-form prereg is verified by THIS mechanism, not the axis-locked loader).
    """
    on_disk = _git("hash-object", str(_PREREG))
    rel = _PREREG.relative_to(_REPO).as_posix()
    tag_blob = _git("rev-parse", f"{_TAG}:{rel}")
    if on_disk != tag_blob:
        raise RuntimeError(
            f"prereg blob-SHA MISMATCH: on-disk={on_disk} tag={tag_blob} (tag {_TAG}). STOP."
        )
    return on_disk


def load_params(prereg: dict[str, Any], *, prereg_sha: str, seeds: list[int]) -> dict[str, Any]:
    """Build the run_gate0 params from the frozen prereg (no re-derivation; ADR-026)."""
    st = prereg["success_tolerance"]
    pm = prereg["partner"]["matched"]
    px = prereg["partner"]["mismatched"]
    env_params: dict[str, Any] = {
        "lateral_window_m": float(st["lateral_window_m"]),
        "angular_window_deg": float(st["angular_window_central_deg"]),
        "seating_force_limit_n": float(st["seating_force_ceiling_n"]),
        # lateral is success-side (translated away) -> non-binding reach:
        "translation_range_m": HANDOVER_DEFAULT_TRANSLATION_RANGE_M,
        "reacquire_range_deg": float(prereg["intrinsic_boundary"]["reacquire_range_deg"]),
        "contact_stiffness_n_per_m": HANDOVER_DEFAULT_CONTACT_STIFFNESS_N_PER_M,
        "angular_stiffness_n_per_deg": HANDOVER_DEFAULT_ANGULAR_STIFFNESS_N_PER_DEG,
    }
    matched_pp: dict[str, float] = {
        "lateral_offset_x_m": 0.0,
        "lateral_offset_y_m": 0.0,
        "lateral_sigma_m": float(pm["lateral_offset_sigma_m"]),
        "grasp_pose_bias_deg": 0.0,
        "grasp_pose_sigma_deg": float(pm["grasp_pose_sigma_deg"]),
        "timing_skew_bias_s": 0.0,
        "timing_skew_sigma_s": float(pm["timing_skew_sigma_s"]),
    }
    mismatched_pp: dict[str, float] = {
        "lateral_offset_x_m": 0.0,
        "lateral_offset_y_m": 0.0,
        "lateral_sigma_m": float(px["lateral_offset_sigma_m"]),
        "grasp_pose_sigma_deg": float(px["grasp_pose_sigma_deg"]),
        "timing_skew_bias_s": float(px["timing_skew_bias_s"]),
        "timing_skew_sigma_s": float(px["timing_skew_sigma_s"]),
    }
    arm_bases = {
        name: {
            "place_cycle_s": float(prereg["arm_bases"][name]["place_cycle_s"]),
            "regrasp_duration_s": float(prereg["arm_bases"][name]["regrasp_duration_s"]),
        }
        for name in ("fast", "slow")
    }
    return {
        "seeds": seeds,
        "episodes_per_seed": int(prereg["decision_rule"]["episodes_per_seed"]),
        "clearance_factor_sweep": list(prereg["in_grasp_correction"]["clearance_factor_sweep"]),
        "takt_grid_s": list(prereg["takt_grid_s"]),
        "arm_bases": arm_bases,
        "env_params": env_params,
        "n_boot": int(prereg["decision_rule"]["bootstrap"]["n_boot"]),
        "matched_presenter_params": matched_pp,
        "mismatched_presenter_params": mismatched_pp,
        "mismatched_grasp_pose_bias_sweep_deg": [float(b) for b in px["grasp_pose_bias_deg_sweep"]],
        "matched_grasp_pose_sigma_deg": float(pm["grasp_pose_sigma_deg"]),
        "mismatched_grasp_pose_sigma_deg": float(px["grasp_pose_sigma_deg"]),
        "prereg_sha": prereg_sha,
        "git_tag": _TAG,
    }


def _per_seed_logit_std(episodes: list[EpisodeResult]) -> float:
    by_seed: dict[int, list[float]] = defaultdict(list)
    for e in episodes:
        by_seed[e.seed].append(1.0 if e.success else 0.0)
    if len(by_seed) < _MIN_SEEDS_FOR_STD:
        return 0.0
    rates = [float(np.mean(v)) for v in by_seed.values()]
    k = float(np.mean([len(v) for v in by_seed.values()]))
    clip = 1.0 / (2.0 * k)
    logits = [math.log(_c / (1 - _c)) for _c in (min(max(r, clip), 1 - clip) for r in rates)]
    return float(np.std(np.array(logits), ddof=1))


def estimate_measured_sigma_u(episodes: list[EpisodeResult]) -> dict[str, Any]:
    """Measured between-seed dispersion on the latent logit scale vs the high bracket (ADR-026).

    Pools each condition's per-seed success rate over the (shared) grid — every seed runs the
    SAME cells, so the between-seed std of the per-seed pooled rate isolates the seed latent —
    and maps to a logit-sigma_u. Conservatively takes the max over conditions and compares to
    the sigma_u=1.0 bracket n=20 was sized against.
    """
    matched = [
        e
        for e in episodes
        if e.metadata.get("condition") == _MATCHED and not e.metadata.get("free_regrasp")
    ]
    mismatched = [
        e
        for e in episodes
        if e.metadata.get("condition") == _MISMATCH and not e.metadata.get("free_regrasp")
    ]
    su_m = _per_seed_logit_std(matched)
    su_x = _per_seed_logit_std(mismatched)
    measured = max(su_m, su_x)
    return {
        "matched_sigma_u_hat": su_m,
        "mismatched_sigma_u_hat": su_x,
        "measured_sigma_u_hat": measured,
        "high_bracket_sigma_u": _HIGH_BRACKET_SIGMA_U,
        "exceeds_high_bracket": measured > _HIGH_BRACKET_SIGMA_U,
        "method": (
            "per-condition per-seed pooled success rate over the shared grid -> logit -> "
            "between-seed std (ddof=1); max over {matched, mismatched-budgeted}; vs sigma_u=1.0"
        ),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _episode_record(e: EpisodeResult) -> dict[str, Any]:
    m = e.metadata
    return {
        "seed": e.seed,
        "episode_idx": e.episode_idx,
        "initial_state_seed": e.initial_state_seed,
        "success": e.success,
        "clearance_factor": m.get("clearance_factor"),
        "takt_s": m.get("takt_s"),
        "arm_basis": m.get("arm_basis"),
        "mismatch_bias_deg": m.get("mismatch_bias_deg"),
        "binding_conjunct": m.get("binding_conjunct"),
        "failure_mode": m.get("failure_mode"),
    }


def _binding_conjunct_log(episodes: list[EpisodeResult]) -> dict[str, Any]:
    """T3: which conjunct drove each place failure, matched vs mismatched."""

    def tally(cond: str) -> dict[str, int]:
        out: dict[str, int] = defaultdict(int)
        for e in episodes:
            if e.metadata.get("condition") == cond and not e.success:
                out[e.metadata.get("binding_conjunct", "?")] += 1
        return dict(out)

    return {
        "matched_failures_by_conjunct": tally(_MATCHED),
        "mismatched_failures_by_conjunct": tally(_MISMATCH),
    }


def _limb1(curves: list[dict[str, Any]]) -> dict[str, Any]:
    """T1: matched solvability + Limb-1 headroom over the realistic-takt cells."""
    lo, hi = REALISTIC_TAKT_BAND_S
    cells = [c for c in curves if "takt_s" in c and lo <= c["takt_s"] <= hi]
    if not cells:
        return {
            "solvable": False,
            "min_matched_iqm": None,
            "min_matched_ci_low": None,
            "headroom_pp": None,
        }
    min_ci_low = min(c["matched_ci_low"] for c in cells)
    return {
        "min_matched_iqm": min(c["matched_iqm"] for c in cells),
        "min_matched_ci_low": min_ci_low,
        "tau_solv": HANDOVER_TAU_SOLV,
        "solvable": min_ci_low >= HANDOVER_TAU_SOLV,
        "headroom_pp": (min_ci_low - HANDOVER_TAU_SOLV) * 100.0,
    }


def _branch_for(verdict: str, branches: dict[str, str]) -> str:
    for key, text in branches.items():
        if verdict in [k.strip() for k in key.split("/")]:
            return text
    return "(no branch mapping)"


def build_verdict_report(  # noqa: PLR0915 (linear Markdown report builder)
    *,
    prereg: dict[str, Any],
    prereg_sha: str,
    code_sha: str,
    analysis: dict[str, Any],
    variance: dict[str, Any],
    conjunct_log: dict[str, Any],
    matched_sigma_deg: float,
    extension_invoked: bool,
) -> str:
    verdict = analysis["verdict"]
    branch = _branch_for(verdict, prereg["branches"])
    limb1 = _limb1(analysis["curves"])
    cells = [c for c in analysis["curves"] if "takt_s" in c]
    coupling_cells = [c for c in cells if c["coupling_valid"]]
    windows = [c for c in analysis["curves"] if "crossover_window" in c]
    region = analysis["mismatch_coupling_region"]
    free_low = analysis["free_regrasp_gap_ci_low_pp"]
    delta_min = float(prereg["decision_rule"]["coupling_delta_min_pp"])
    intrinsic = free_low >= delta_min
    lines: list[str] = []
    a = lines.append

    a(f"# Gate-0 handover-and-place — VERDICT REPORT ({_DATE})")
    a("")
    a("Non-gating Phase-2 ADR-026 coupling-validity spike (invariant I1). Measured two-limb")
    a("slice against the FROZEN, tagged pre-registration.")
    a("")
    a("## Provenance")
    a(f"- prereg tag: `{_TAG}`")
    a(f"- prereg blob-SHA (verified == tag blob): `{prereg_sha}`")
    a(f"- code SHA (launch, captured once): `{code_sha}`")
    a("- decision rule: `decision_rule.canonical_cell_rule` — two-sided 95% CI; COUPLING_VALID")
    a("  iff gap CI_lower >= delta_min; WASHOUT iff gap CI_upper < delta_min; INDETERMINATE else.")
    a("- every number below cites its archive JSON.")
    a("")
    a("## HEADLINE")
    a(f"- **Verdict: `{verdict}`** (verdict_space; spike_handover_place_gate0_{_DATE}.json +")
    a("  crossover_curves.json)")
    a(f"- **Branch:** {branch}")
    band_note = (
        "EXCEEDS -> seed extension invoked"
        if variance["exceeds_high_bracket"]
        else "within bracket (n=20 holds)"
    )
    a(f"- **Variance check:** sigma_u_hat = {variance['measured_sigma_u_hat']:.3f} vs high")
    a(
        f"  bracket {variance['high_bracket_sigma_u']:.1f} -> {band_note}; seed extension "
        f"{'INVOKED' if extension_invoked else 'not invoked'} (spike JSON variance_check)."
    )
    a("")
    a("## Limb 1 — solvability (T1)")
    a(f"- matched grasp_pose_sigma = {matched_sigma_deg:.1f} deg (reported in degrees, T2).")
    a(f"- realistic takt band {_BAND} s: min matched IQM = {limb1['min_matched_iqm']:.3f},")
    a(f"  min matched CI_lower = {limb1['min_matched_ci_low']:.3f} (crossover_curves.json).")
    a(f"- tau_solv = {limb1['tau_solv']}; **SOLVABLE = {limb1['solvable']}**; Limb-1 headroom =")
    a(f"  {limb1['headroom_pp']:+.1f} pp.")
    a("")
    a("## Limb 2 — coupling validity (canonical rule; crossover_curves.json)")
    a(f"- coupling-valid cells (gap CI_lower >= delta_min): {len(coupling_cells)} of {len(cells)}.")
    if coupling_cells:
        a("- coupling-valid cells (clearance / bias / arm / takt : gap_pp [CI_lo, CI_hi]):")
        for c in sorted(
            coupling_cells,
            key=lambda c: (
                c["clearance_factor"],
                c["mismatch_bias_deg"],
                c["arm_basis"],
                c["takt_s"],
            ),
        ):
            a(
                f"    - clr {c['clearance_factor']}/{c['mismatch_bias_deg']:.0f}d/"
                f"{c['arm_basis']}/{c['takt_s']}s : {c['gap_pp']:.1f} "
                f"[{c['gap_ci_low_pp']:.1f}, {c['gap_ci_high_pp']:.1f}]"
            )
    a("")
    a("## Crossover band mapped to takt (both arms; crossover_curves.json)")
    for w in windows:
        cw = w["crossover_window"]
        a(
            f"- {w['mismatch_bias_deg']:.0f}deg / clr {w['clearance_factor']} / {w['arm_basis']}: "
            f"window_takts_s={cw['window_takts_s']} (floor={cw['solvability_floor_takt_s']}, "
            f"ceiling={cw['coupling_ceiling_takt_s']})"
        )
    a(f"- realistic takt band = {_BAND} s; COUPLING_VALID fires iff a window overlaps it.")
    a("")
    a("## Intrinsic vs budget-mediated split (free-re-grasp endpoint)")
    split = (
        "INTRINSIC (persists at free re-grasp)"
        if intrinsic
        else "BUDGET-MEDIATED (vanishes at free re-grasp)"
    )
    a(
        f"- free-re-grasp gap CI_lower = {free_low:.1f} pp -> **{split}** (spike JSON free episodes)."
    )
    a("")
    a("## (clearance x mismatch) MEASURED coupling region")
    a("  (clearance_mismatch_region_measured.json)")
    a("- clearance / bias / coupling_valid_any_takt / max_gap_CI_lower_pp:")
    for r in region:
        mg = r["max_gap_ci_low_pp"]
        mg_s = "nan" if (isinstance(mg, float) and math.isnan(mg)) else f"{mg:.1f}"
        a(
            f"    - clr {r['clearance_factor']} / {r['mismatch_bias_deg']:.0f}deg : "
            f"coupling={r['coupling_valid_any_takt']} / max_gap_CI_lower={mg_s} pp"
        )
    a("")
    a("## Binding-conjunct log (T3 — lateral success-side, grasp-pose coupling-side)")
    a(f"- matched failures by conjunct: {conjunct_log['matched_failures_by_conjunct']}")
    a(f"- mismatched failures by conjunct: {conjunct_log['mismatched_failures_by_conjunct']}")
    a("  (expected: mismatched failures dominated by 'angular' (grasp-pose); 'lateral' absent.)")
    a("")
    a("## Selected branch")
    a(f"- **{verdict}** -> {branch}")
    if verdict in ("WASHOUT", "WASHOUT_FOR_REAL_CELLS"):
        a("- The pre-named escalation to cross-modal co-hold-and-secure is now ARMED (a separate")
        a("  founder decision; NOT executed here).")
    a("")
    a("_No prose-only numbers: every figure traces to a committed JSON in this archive._")
    return "\n".join(lines) + "\n"


def write_archive(
    *,
    spike_run: SpikeRun,
    analysis: dict[str, Any],
    variance: dict[str, Any],
    conjunct_log: dict[str, Any],
    verdict_report: str,
    cli_text: str,
) -> list[str]:
    """Write the write-once archive (I8) + SHA256SUMS. Returns the file names."""
    _ARCHIVE.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    def _w(name: str, content: str) -> None:
        (_ARCHIVE / name).write_text(content, encoding="utf-8")
        written.append(name)

    spike_obj = json.loads(spike_run.model_dump_json())
    spike_obj["variance_check"] = variance
    spike_obj["binding_conjunct_log"] = conjunct_log
    _w(f"spike_handover_place_gate0_{_DATE}.json", json.dumps(spike_obj, indent=2, sort_keys=True))

    cells = [c for c in analysis["curves"] if "takt_s" in c]
    crossovers = [c for c in analysis["curves"] if "crossover_window" in c]
    _w(
        "crossover_curves.json",
        json.dumps(
            {
                "cells": cells,
                "crossover_windows": crossovers,
                "overall_window": analysis["overall_window"],
                "free_regrasp_gap_ci_low_pp": analysis["free_regrasp_gap_ci_low_pp"],
                "verdict": analysis["verdict"],
            },
            indent=2,
            sort_keys=True,
        ),
    )
    _w(
        "clearance_mismatch_region_measured.json",
        json.dumps(
            {
                "measured_region": analysis["mismatch_coupling_region"],
                "design_overlay": analysis["clearance_threshold_overlay"],
            },
            indent=2,
            sort_keys=True,
        ),
    )

    def _is(cond: str, *, free: bool) -> Callable[[EpisodeResult], bool]:
        def pred(e: EpisodeResult) -> bool:
            return (
                e.metadata.get("condition") == cond and bool(e.metadata.get("free_regrasp")) is free
            )

        return pred

    classes = {
        "matched.jsonl": _is(_MATCHED, free=False),
        "mismatched_budgeted.jsonl": _is(_MISMATCH, free=False),
        "mismatched_free.jsonl": _is(_MISMATCH, free=True),
    }
    for fname, pred in classes.items():
        eps = sorted(
            (e for e in spike_run.episode_results if pred(e)),
            key=lambda e: (
                e.seed,
                e.metadata.get("clearance_factor") or 0.0,
                e.metadata.get("takt_s") or 0.0,
                str(e.metadata.get("arm_basis")),
                e.metadata.get("mismatch_bias_deg") or -1.0,
                e.episode_idx,
            ),
        )
        _w(fname, "\n".join(json.dumps(_episode_record(e), sort_keys=True) for e in eps) + "\n")

    _w(f"GATE_VERDICT_REPORT_{_DATE}.md", verdict_report)
    _w("repro_command.txt", "uv run python scripts/repro/_handover_place_gate0_run.py\n")
    _w("next_stage_measurement.txt", cli_text)

    sums = "\n".join(f"{_sha256(_ARCHIVE / n)}  {n}" for n in sorted(written)) + "\n"
    (_ARCHIVE / "SHA256SUMS.txt").write_text(sums, encoding="utf-8")
    written.append("SHA256SUMS.txt")
    return written


def main() -> int:
    """Verify, run, archive (ADR-026; ADR-007; I8). Returns 0 on success."""
    log: list[str] = []

    def out(msg: str) -> None:
        print(msg)
        log.append(msg)

    prereg_sha = verify_prereg_blob()
    out(f"verify-prereg: PASS (blob {prereg_sha} == tag {_TAG} blob)")
    prereg = yaml.safe_load(_PREREG.read_text(encoding="utf-8"))
    run_ctx = compute_run_metadata(seed=0, run_kind="handover_place_gate0", repo_root=_REPO)
    out(f"code SHA (launch): {run_ctx.git_sha}  run_id: {run_ctx.run_id}")

    seeds = [int(s) for s in prereg["decision_rule"]["seeds"]]
    k = prereg["decision_rule"]["episodes_per_seed"]
    out(f"running n={len(seeds)} seed-clusters x {k} episodes ...")
    params = load_params(prereg, prereg_sha=prereg_sha, seeds=seeds)
    spike_run, analysis = run_gate0(params)
    variance = estimate_measured_sigma_u(spike_run.episode_results)
    out(
        f"variance check: measured sigma_u_hat={variance['measured_sigma_u_hat']:.3f} "
        f"(matched {variance['matched_sigma_u_hat']:.3f} / "
        f"mismatched {variance['mismatched_sigma_u_hat']:.3f}) vs high bracket "
        f"{variance['high_bracket_sigma_u']}"
    )

    extension_invoked = False
    if variance["exceeds_high_bracket"]:
        ext_seeds = seeds + [int(s) for s in prereg["decision_rule"]["seeds_extension"]]
        out(
            f"measured sigma_u exceeds the high bracket -> invoking pre-registered seed "
            f"extension ONCE (n={len(ext_seeds)}); declaring after."
        )
        params = load_params(prereg, prereg_sha=prereg_sha, seeds=ext_seeds)
        spike_run, analysis = run_gate0(params)
        variance = estimate_measured_sigma_u(spike_run.episode_results)
        variance["seed_extension_invoked"] = True
        extension_invoked = True

    conjunct_log = _binding_conjunct_log(spike_run.episode_results)
    out(f"episodes scored: {len(spike_run.episode_results)}")
    out(f"VERDICT: {analysis['verdict']}")

    cli_text = "\n".join(log) + "\nexit_code: 0\n"
    report = build_verdict_report(
        prereg=prereg,
        prereg_sha=prereg_sha,
        code_sha=run_ctx.git_sha,
        analysis=analysis,
        variance=variance,
        conjunct_log=conjunct_log,
        matched_sigma_deg=params["matched_grasp_pose_sigma_deg"],
        extension_invoked=extension_invoked,
    )
    written = write_archive(
        spike_run=spike_run,
        analysis=analysis,
        variance=variance,
        conjunct_log=conjunct_log,
        verdict_report=report,
        cli_text=cli_text,
    )
    print(f"archive -> {_ARCHIVE.relative_to(_REPO)}  ({len(written)} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
