# SPDX-License-Identifier: Apache-2.0
# pyright: reportPrivateImportUsage=false
"""Phase-2 co-carry decouple ablation (pre-registered; ADR-026; R-2026-06-C condition 5).

Eval-only, NO training, NO residual, NO heterogeneity. One competent hand-built
MATCHED pair (cooperative-impedance ego + matched impedance partner -- the
'base' config of the Rung-5 base_robustness_control); the ONLY thing varied is
the coupling: DOWN to zero (the sibling of the base-probe's upward sweep), plus
a partner-removed arm (zero-action partner seat, coupling intact). Answers the
R-2026-06-C condition-5 question: is the base ego's near-perfect success
COOPERATION-CONTINGENT (collapses when decoupled => the base_robustness_control
is a valid cooperation demonstration and the co-carry CONFOUNDED verdict holds)
or COUPLING-TRIVIAL (survives decoupling => the control never demonstrated
cooperation and the headline must be re-interpreted)?

The conditions, seeds, fixed base config, estimator, decision-rule bounds, and
tripwires are read from the LOCKED, git-tagged pre-registration
(spikes/preregistration/cocarry_decouple_ablation_prereg.json, tag
prereg-cocarry-decouple-ablation-2026-07-11) -- this driver only executes it.
It REFUSES to run on a dirty working tree (exit 7) or when the on-disk prereg
blob SHA differs from the blob stored at the tag (exit 4, the chamber-eval
PREREG_MISMATCH gate; no bypass). No episode runs before the tag exists.

Success is computed POST-HOC via the pure chamber.envs.cocarry.
evaluate_cocarry_success(...) at the frozen coupling f_max (365.6 N) on the
per-episode raw metrics RE-READ from the committed JSONL -- never from
in-memory values (the pre-headline recompute gate). The env-internal success
flag is ignored, exactly as in the base probe.

ADR-026 §Decision 1-2 (coupling-validity + the joint predicate, never
weakened); ADR-026 §Open-questions (coupling stiffness is a task parameter).
Phase-2, NON-gating. Do not merge -- founder review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
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
from chamber.evaluation.bundles import DIRTY_TREE_EXIT_CODE
from chamber.evaluation.prereg import PREREG_MISMATCH_EXIT_CODE
from chamber.partners.ablation import PARTNER_ABLATED_ZERO_CLASS
from chamber.partners.api import PartnerSpec
from chamber.partners.registry import load_partner

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[2]
_PREREG = _REPO / "spikes/preregistration/cocarry_decouple_ablation_prereg.json"
_PREREG_TAG = "prereg-cocarry-decouple-ablation-2026-07-11"
_OUT_DIR = _REPO / "spikes/results/cocarry/decouple_ablation"
_OUT = _OUT_DIR / "cocarry_decouple_ablation_measurement.json"
_EPISODES = _OUT_DIR / "cocarry_decouple_ablation_episodes.jsonl"
_TRAJ_OUT = _OUT_DIR / "cocarry_decouple_ablation_trajectories.json"
_STOP_OUT = _OUT_DIR / "cocarry_decouple_ablation_stop.json"
_SHASUMS = _OUT_DIR / "SHA256SUMS"
_REPRO_TXT = _OUT_DIR / "REPRO.txt"
_RENDER = "none"
_REPRO_COMMAND = "uv run --no-sync python scripts/repro/_cocarry_decouple_ablation.py"

#: Tripwire-STOP exit code (substrate drift / non-finite / compute-cap breach).
#: Matches the chamber-eval training trip-wire code; distinct from dirty-tree 7
#: and prereg-mismatch 4.
_TRIPWIRE_STOP_EXIT_CODE = 3

#: pd_joint_delta_pos action width of the Panda partner seat (7 arm + gripper)
#: -- the action_dim the registered zero-action instrument is bound to, same
#: as the admission A2 cell (partners.json: action_dim "8").
_PANDA_ACTION_DIM = 8

# Frozen predicate thresholds (NEVER weakened). f_max is the Rung-4c-grounded
# coupling ceiling. These mirror chamber.envs.cocarry / the base probe.
_FROZEN_GOAL_THRESH_M = 0.10
_FROZEN_TILT_MAX_DEG = 15.0
_FMAX_COUPLING_N = 365.6080997467041

#: Every prior co-carry seed set (committed artifacts; training blocks are
#: conservative 1000-wide windows around the recorded block starts). The
#: pre-registered block [71000..71019] must be disjoint from ALL of these;
#: :func:`verify_seed_disjointness` checks it in code before any rollout.
_PRIOR_COCARRY_SEEDS: dict[str, frozenset[int]] = {
    "rung2_train_S_block": frozenset(range(10000, 11000)),
    "rung2_val_V_block": frozenset(range(20000, 21000)),
    "rung3_measurement": frozenset(range(30000, 30012)),
    "rung3_calibration_block": frozenset(range(40000, 41000)),
    "rung4_eh_block": frozenset(range(50000, 51000)),
    "rung5_measurement_block": frozenset(range(54000, 55000)),
    "rung4_eh_block_2": frozenset(range(60000, 61000)),
    "base_probe": frozenset(range(70000, 70012)),
    "rung4b_coupling_sweep": frozenset(range(70010, 70018)),
    "rung4d_pose_falsification": frozenset(range(80100, 80106)),
    "rung4e_search": frozenset(range(81000, 81004)),
    "rung4e_confirm": frozenset(range(81100, 81112)),
    "admission_2026_07_05": frozenset(range(90000, 90007)),
}


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Tripwire 1 -- prereg-first, tag-before-measurement (the chamber-eval gate).
# ---------------------------------------------------------------------------


def _git(*args: str) -> str:
    git = shutil.which("git")
    if git is None:
        msg = "git executable not found on PATH; cannot verify the pre-registration gate"
        raise RuntimeError(msg)
    result = subprocess.run(  # noqa: S603 - git binary resolved via shutil.which
        [git, "-C", str(_REPO), *args],
        capture_output=True,
        check=True,
        text=True,
    )
    return result.stdout.strip()


def _refuse_dirty_tree(*, allow_dirty: bool) -> None:
    """Refuse a dirty working tree (mirror chamber-eval; exit DIRTY_TREE_EXIT_CODE)."""
    status = _git("status", "--porcelain")
    if not status:
        return
    if allow_dirty:
        print("  WARNING: dirty working tree tolerated (--allow-dirty); recorded in the artifact")
        return
    print(
        "REFUSED: dirty working tree -- the pre-registered run must launch from a clean,\n"
        "tagged state (commit or stash first; --allow-dirty only for re-runs over\n"
        "already-written results).",
        file=sys.stderr,
    )
    raise SystemExit(DIRTY_TREE_EXIT_CODE)


def _verify_prereg_tag() -> str:
    """Verify the on-disk prereg matches the blob at the locked tag (no bypass).

    Returns the verified blob SHA. Exits ``PREREG_MISMATCH_EXIT_CODE`` (4) when
    the tag is missing or the SHAs disagree -- the chamber-eval
    ``PREREG_MISMATCH`` gate: any post-tag edit to the prereg's bytes shifts
    the on-disk blob SHA but not the tagged one.
    """
    rel = _PREREG.relative_to(_REPO)
    try:
        _git("rev-parse", "--verify", f"refs/tags/{_PREREG_TAG}")
    except subprocess.CalledProcessError:
        print(
            f"PREREG_MISMATCH: tag {_PREREG_TAG!r} does not exist -- no episode runs before\n"
            "the founder cuts the SSH-signed annotated tag over the committed prereg.",
            file=sys.stderr,
        )
        raise SystemExit(PREREG_MISMATCH_EXIT_CODE) from None
    on_disk = _git("hash-object", "--", str(_PREREG))
    at_tag = _git("rev-parse", f"{_PREREG_TAG}:{rel}")
    if on_disk != at_tag:
        print(
            f"PREREG_MISMATCH: on-disk prereg blob {on_disk} != tagged blob {at_tag}\n"
            f"(tag={_PREREG_TAG}, file={rel}). The prereg was edited after the tag was cut.",
            file=sys.stderr,
        )
        raise SystemExit(PREREG_MISMATCH_EXIT_CODE)
    return on_disk


def verify_seed_disjointness(seeds: list[int]) -> dict[str, list[int]]:
    """Return the per-prior-block collisions of ``seeds`` (empty dict = disjoint)."""
    seed_set = set(seeds)
    return {
        name: sorted(seed_set & prior)
        for name, prior in _PRIOR_COCARRY_SEEDS.items()
        if seed_set & prior
    }


# ---------------------------------------------------------------------------
# Condition wiring (unit-tested Tier-1: C1 spring-off + C2 zero-action seat).
# ---------------------------------------------------------------------------


def build_noop_partner(*, seed: int) -> Any:  # noqa: ANN401 - FrozenPartner Protocol
    """Build the C2 partner-removed seat: the registered zero-action instrument.

    LIMP-BUT-COUPLED (distinct from the A2 'retracted' anchor): the partner ARM
    stays present and coupled (K=8000); its pd_joint_delta_pos action stream is
    identically zero (holds the ready pose, never cooperates) --
    :class:`chamber.partners.ablation.PartnerAblatedZero`, the same registered
    instrument the admission A2 cell used (``action_dim`` 8), so the cell's
    partner identity is registry-resolvable, not an ad-hoc lambda.
    """
    return load_partner(
        PartnerSpec(
            PARTNER_ABLATED_ZERO_CLASS, seed, None, None, {"action_dim": str(_PANDA_ACTION_DIM)}
        )
    )


def decouple_verdict(
    *,
    c0_rate: float,
    c1_rate: float,
    c2_rate: float,
    gap_ci_lower: float,
    c0_min: float,
    c2_max: float,
    gap_ci_lower_min: float,
    trivial_min: float,
) -> str:
    """Apply the pre-committed, falsifiable-both-ways decision rule (prereg §decision_rule).

    Founder-confirmed 2026-07-14 form. COUPLING_TRIVIAL iff C2 >= trivial_min
    OR C1 >= trivial_min (the base wins with no functioning partner, or with no
    coupling at all -- either is a trivial-solve; takes PRECEDENCE, since any
    decoupled trivial-solve invalidates the control regardless of the other
    cell). COOPERATION_CONTINGENT iff the coupled anchor holds (>= c0_min), the
    limp-but-coupled partner-removed cell collapses (C2 <= c2_max), AND the
    (C0 - C2) gap's one-sided lower confidence bound clears gap_ci_lower_min;
    C2 is the PRIMARY instrument (cooperation removed at fixed K=8000, bar
    still supported); C1/C3 are the corroborating dose-response, not part of
    this contrast (a C1-only failure is weaker evidence -- at K=0 the bar is
    unsupported). INDETERMINATE otherwise -- report the split (e.g. C2 low,
    C1 mid), propose the next cheap discriminator, never self-authorize it.
    """
    if c2_rate >= trivial_min or c1_rate >= trivial_min:
        return "COUPLING_TRIVIAL"
    if c0_rate >= c0_min and c2_rate <= c2_max and gap_ci_lower >= gap_ci_lower_min:
        return "COOPERATION_CONTINGENT"
    return "INDETERMINATE"


def bootstrap_rate_ci(
    per_seed: dict[int, bool],
    *,
    n_boot: int = 10000,
    alpha: float = 0.05,
    root_seed: int = 0,
) -> tuple[float, float]:
    """Two-sided seed-bootstrap percentile CI on one cell's success rate (P6 RNG)."""
    from concerto.training.seeding import derive_substream  # noqa: PLC0415 - lazy

    rng = derive_substream(
        "cocarry.decouple_ablation.rate_bootstrap", root_seed=root_seed
    ).default_rng()
    vals = np.array([float(per_seed[s]) for s in sorted(per_seed)], dtype=np.float64)
    n = len(vals)
    boots = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        boots[b] = float(np.mean(vals[rng.integers(0, n, size=n)]))
    return (
        float(np.percentile(boots, 100.0 * (alpha / 2.0))),
        float(np.percentile(boots, 100.0 * (1.0 - alpha / 2.0))),
    )


# ---------------------------------------------------------------------------
# Rollout + post-hoc predicate (verbatim from the base probe, the template).
# ---------------------------------------------------------------------------


def _rollout_with_trajectory(
    *,
    env: Any,  # noqa: ANN401
    ego: Any,  # noqa: ANN401
    partner: Any,  # noqa: ANN401
    seed: int,
    episode_length: int,
) -> dict[str, Any]:
    """One pair episode, recording per-step instantaneous trajectories.

    Returns the episode metrics the frozen predicate reads (final centroid, the
    settle-windowed running maxima of tilt + coupling stress, both_static) plus
    the per-step instantaneous (tilt_deg, coupling stress, centroid) arrays.
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
            partner_uid: np.asarray(partner.act(obs), dtype=np.float32),
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


def _success_at(ep: dict[str, Any]) -> bool:
    """Apply the frozen joint predicate to one episode's raw metrics (post-hoc)."""
    return bool(
        cc.evaluate_cocarry_success(
            centroid_to_goal_dist=ep["centroid_to_goal"],
            max_tilt_deg=ep["max_tilt_deg"],
            max_stress_proxy=ep["max_stress_proxy"],
            both_static=ep["both_static"],
            goal_thresh=_FROZEN_GOAL_THRESH_M,
            tilt_max_deg=_FROZEN_TILT_MAX_DEG,
            stress_max=_FMAX_COUPLING_N,
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


def _cell_summary(eps: list[dict[str, Any]], successes: list[bool]) -> dict[str, Any]:
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


def _classify_cell(
    summary: dict[str, Any], *, all_finite: bool, thresholds: dict[str, Any]
) -> dict[str, Any]:
    """Artifact-vs-real classification per the locked prereg (base-probe rule)."""
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
    """Per-step median/p90/max across seeds (compact; graceful vs blow-up vs free-fall)."""
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


# ---------------------------------------------------------------------------
# Measurement pipeline.
# ---------------------------------------------------------------------------


def _stop(reason: str, *, details: dict[str, Any] | None = None) -> int:
    """Tripwire STOP: write the stop record, report, exit 3 -- never extend."""
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "cocarry_decouple_ablation_stop/v1",
        "prereg_tag": _PREREG_TAG,
        "stop_reason": reason,
        "details": details or {},
        "note": (
            "Pre-registered tripwire STOP (prereg §tripwires): no verdict is issued; "
            "no condition beyond the recorded ones ran. Founder review required."
        ),
    }
    _STOP_OUT.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
    print(f"\n  STOP: {reason}")
    print(f"  stop record -> {_STOP_OUT}")
    return _TRIPWIRE_STOP_EXIT_CODE


def _run_cell(
    *,
    drive_stiffness: float,
    partner_kind: str,
    seeds: list[int],
    episode_length: int,
    condition: str,
    stress_measure: str,
) -> list[dict[str, Any]]:
    """Roll one cell (12 seeds; fresh env per seed, mirroring the base probe)."""
    eps: list[dict[str, Any]] = []
    for s in seeds:
        env = make_cocarry_training_env(
            condition_id=condition,
            episode_length=episode_length,
            root_seed=s,
            render_backend=_RENDER,
            drive_stiffness=drive_stiffness,
            stress_measure=stress_measure,
        )
        try:
            ego = ph.build_cooperative_ego(seed=s)
            if partner_kind == "noop":
                partner = build_noop_partner(seed=s)
            else:
                partner = ph.build_partner_seat(
                    "cocarry_impedance", seed=s, partner_uid="panda_partner"
                )
            eps.append(
                _rollout_with_trajectory(
                    env=env, ego=ego, partner=partner, seed=s, episode_length=episode_length
                )
            )
        finally:
            env.close()
    return eps


def _episode_rows(cell_key: str, cell_cfg: dict[str, Any], eps: list[dict[str, Any]]) -> list[str]:
    """Serialise one cell's episodes as JSONL rows of RAW metrics (no success field).

    Success is deliberately NOT written: the pre-headline recompute gate
    re-derives it from these raw metrics via evaluate_cocarry_success, so the
    headline can never come from an in-memory or prose value.
    """
    return [
        json.dumps(
            {
                "cell": cell_key,
                "drive_stiffness_npm": cell_cfg["drive_stiffness_npm"],
                "partner": cell_cfg["partner"],
                "seed": e["seed"],
                "centroid_to_goal": e["centroid_to_goal"],
                "max_tilt_deg": e["max_tilt_deg"],
                "max_stress_proxy": e["max_stress_proxy"],
                "both_static": e["both_static"],
                "n_steps": e["n_steps"],
            },
            sort_keys=True,
        )
        for e in eps
    ]


def main() -> int:  # noqa: PLR0915 - linear measurement pipeline, kept auditable here
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help=(
            "Tolerate a dirty working tree (chamber-eval style; for re-runs over "
            "already-written results). The prereg blob-SHA gate has NO bypass."
        ),
    )
    args = parser.parse_args()

    # --- Tripwire 1: prereg-first, tag-before-measurement ---
    _refuse_dirty_tree(allow_dirty=bool(args.allow_dirty))
    prereg_blob_sha = _verify_prereg_tag()
    print(f"  prereg gate OK: tag {_PREREG_TAG} -> blob {prereg_blob_sha}")

    prereg = json.loads(_PREREG.read_text("utf-8"))
    base = prereg["fixed_base_config"]
    seeds: list[int] = [int(s) for s in prereg["seeds"]]
    episode = int(base["episode_length"])
    stress_measure = str(base["stress_measure"])
    condition = str(base["condition_id"])
    base_k = float(base["baseline_drive_stiffness"])
    k_grid = [float(k) for k in prereg["conditions"]["C3_down_sweep"]["drive_stiffness_grid_npm"]]
    thresholds = prereg["artifact_vs_real_classification"]
    bounds = prereg["decision_rule"]["bounds"]
    tripwires = prereg["tripwires"]
    c0_reproduce_min = float(tripwires["c0_reproduce_min"])
    max_episodes = int(tripwires["compute_cap"]["max_episodes"])
    max_wall_s = float(tripwires["compute_cap"]["max_wall_clock_s"])

    # --- Seed-disjointness check (in code, printed, before any rollout) ---
    collisions = verify_seed_disjointness(seeds)
    if collisions:
        return _stop(
            "SEED_COLLISION: the pre-registered block overlaps a prior co-carry seed set",
            details={"collisions": collisions},
        )
    print(
        f"  seed disjointness OK: block [{min(seeds)}..{max(seeds)}] (n={len(seeds)}) vs "
        f"{len(_PRIOR_COCARRY_SEEDS)} prior co-carry seed sets -- no overlap"
    )

    planned_episodes = len(seeds) * (len(k_grid) + 1)  # down-sweep cells + C2
    if planned_episodes > max_episodes:
        return _stop(
            "COMPUTE_CAP: planned episodes exceed the pre-registered cap",
            details={"planned": planned_episodes, "max_episodes": max_episodes},
        )
    print(
        f"  decouple ablation: {len(k_grid)} down-sweep cells + C2, {len(seeds)} seeds, "
        f"{planned_episodes} episodes (cap {max_episodes}), f_max={_FMAX_COUPLING_N:.1f} N"
    )

    # Cell table: the K=8000 down-sweep cell IS C0; the K=0 cell IS C1; C2 is
    # the zero-action-partner cell at K=8000. Each (cell, seed) runs once.
    cells: dict[str, dict[str, Any]] = {}
    for k in k_grid:
        cells[f"K{k:.0f}"] = {
            "drive_stiffness_npm": float(k),
            "partner": "cocarry_impedance",
            "partner_kind": "matched",
            "aliases": (["C0"] if float(k) == base_k else []) + (["C1"] if k == 0.0 else []),
        }
    cells["C2_partner_removed"] = {
        "drive_stiffness_npm": base_k,
        "partner": PARTNER_ABLATED_ZERO_CLASS,
        "partner_kind": "noop",
        "aliases": ["C2"],
    }

    t0 = time.time()
    eps_by_cell: dict[str, list[dict[str, Any]]] = {}

    # --- C0 first (tripwire 5: the substrate-drift kill criterion) ---
    c0_key = f"K{base_k:.0f}"
    run_order = [c0_key, *[k for k in cells if k != c0_key]]
    for cell_key in run_order:
        if time.time() - t0 > max_wall_s:
            return _stop(
                "COMPUTE_CAP: wall-clock ceiling breached",
                details={"elapsed_s": time.time() - t0, "max_wall_clock_s": max_wall_s},
            )
        cfg = cells[cell_key]
        tk = time.time()
        eps = _run_cell(
            drive_stiffness=cfg["drive_stiffness_npm"],
            partner_kind=cfg["partner_kind"],
            seeds=seeds,
            episode_length=episode,
            condition=condition,
            stress_measure=stress_measure,
        )
        eps_by_cell[cell_key] = eps
        if not all(_episode_finite(e) for e in eps):
            bad = [e["seed"] for e in eps if not _episode_finite(e)]
            return _stop(
                "NON_FINITE: non-finite tilt/stress/centroid in a rollout",
                details={"cell": cell_key, "seeds": bad},
            )
        successes = [_success_at(e) for e in eps]
        rate = float(np.mean(successes))
        print(
            f"    {cell_key:>20}  success={rate:.3f}  "
            f"stress_p90={float(np.percentile([e['max_stress_proxy'] for e in eps], 90)):.0f}  "
            f"tilt_p90={float(np.percentile([e['max_tilt_deg'] for e in eps], 90)):.1f}  "
            f"[{time.time() - tk:.0f}s]"
        )
        if cell_key == c0_key and rate < c0_reproduce_min:
            return _stop(
                "SUBSTRATE_DRIFT: C0 did not reproduce the base_robustness_control "
                f"matched cell (success {rate:.3f} < c0_reproduce_min {c0_reproduce_min})",
                details={
                    "c0_success_rate": rate,
                    "per_seed": [
                        {"seed": e["seed"], "success": successes[i]} for i, e in enumerate(eps)
                    ],
                },
            )

    # --- Commit the per-episode raw metrics + trajectories ---
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[str] = []
    trajectories: dict[str, Any] = {}
    for cell_key, cfg in cells.items():
        rows.extend(_episode_rows(cell_key, cfg, eps_by_cell[cell_key]))
        trajectories[cell_key] = _aggregate_trajectory(eps_by_cell[cell_key])
    _EPISODES.write_text("\n".join(rows) + "\n", encoding="utf-8")
    _TRAJ_OUT.write_text(
        json.dumps(
            {
                "schema": "cocarry_decouple_ablation_trajectories/v1",
                "note": (
                    "Per-cell per-step aggregated (across seeds) instantaneous trajectories: "
                    "tilt (deg), coupling stress (N), centroid (m). Distinguishes graceful "
                    "strain / free-fall (decoupled) / blow-up."
                ),
                "fmax_coupling_n": _FMAX_COUPLING_N,
                "by_cell": trajectories,
            },
            sort_keys=True,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --- Pre-headline recompute gate: RE-READ the committed JSONL; recompute
    # every success via the pure predicate; the verdict reads only this. ---
    committed = [json.loads(line) for line in _EPISODES.read_text("utf-8").splitlines() if line]
    by_cell: dict[str, dict[int, dict[str, Any]]] = {}
    for row in committed:
        by_cell.setdefault(row["cell"], {})[int(row["seed"])] = row
    cell_results: dict[str, Any] = {}
    per_seed_success: dict[str, dict[int, bool]] = {}
    for cell_key, cfg in cells.items():
        cell_rows = by_cell[cell_key]
        succ = {s: _success_at(cell_rows[s]) for s in sorted(cell_rows)}
        per_seed_success[cell_key] = succ
        eps_list = [cell_rows[s] for s in sorted(cell_rows)]
        succ_list = [succ[s] for s in sorted(cell_rows)]
        summary = _cell_summary(eps_list, succ_list)
        all_finite = all(_episode_finite(eps_by_cell[cell_key][i]) for i in range(len(eps_list)))
        cls = _classify_cell(summary, all_finite=all_finite, thresholds=thresholds)
        ci_lo, ci_hi = bootstrap_rate_ci(succ)
        cell_results[cell_key] = {
            "drive_stiffness_npm": cfg["drive_stiffness_npm"],
            "partner": cfg["partner"],
            "aliases": cfg["aliases"],
            "summary": summary,
            "success_ci95": [ci_lo, ci_hi],
            "all_finite": all_finite,
            **cls,
            "per_seed": [
                {
                    "seed": s,
                    "success": succ[s],
                    "centroid_to_goal": round(cell_rows[s]["centroid_to_goal"], 5),
                    "max_tilt_deg": round(cell_rows[s]["max_tilt_deg"], 3),
                    "max_stress_proxy": round(cell_rows[s]["max_stress_proxy"], 2),
                    "both_static": cell_rows[s]["both_static"],
                    "n_steps": cell_rows[s]["n_steps"],
                }
                for s in sorted(cell_rows)
            ],
        }

    # --- Estimator (prereg §estimator): the C0-vs-C2 primary contrast; C1/C3
    # reported as the corroborating coupling dose-response (2026-07-14 rule) ---
    c1_key = "K0"
    c0_map = per_seed_success[c0_key]
    c1_map = per_seed_success[c1_key]
    c2_map = per_seed_success["C2_partner_removed"]
    c0_rate = float(np.mean([float(v) for v in c0_map.values()]))
    c1_rate = float(np.mean([float(v) for v in c1_map.values()]))
    c2_rate = float(np.mean([float(v) for v in c2_map.values()]))
    gap = ph.cluster_bootstrap_delta(c0_map, {"C2_partner_removed": c2_map}, root_seed=0)
    verdict = decouple_verdict(
        c0_rate=c0_rate,
        c1_rate=c1_rate,
        c2_rate=c2_rate,
        gap_ci_lower=float(gap["pooled_ci_lower_one_sided"]),
        c0_min=float(bounds["c0_min"]),
        c2_max=float(bounds["c2_max"]),
        gap_ci_lower_min=float(bounds["gap_ci_lower_min"]),
        trivial_min=float(bounds["trivial_min"]),
    )

    out = {
        "schema": "cocarry_decouple_ablation_measurement/v1",
        "stage": prereg["stage"],
        "prereg": "spikes/preregistration/cocarry_decouple_ablation_prereg.json",
        "prereg_tag": _PREREG_TAG,
        "prereg_blob_sha": prereg_blob_sha,
        "prereg_sha256": _sha256_file(_PREREG),
        "env_module_sha256": _sha256_file(_REPO / "src/chamber/envs/cocarry.py"),
        "fmax_coupling_n": _FMAX_COUPLING_N,
        "seeds": seeds,
        "n_seeds": len(seeds),
        "cells": cell_results,
        "c0_success_rate": c0_rate,
        "c1_decoupled_rate": c1_rate,
        "c2_partner_removed_rate": c2_rate,
        "c0_minus_c2_gap": {
            "point": gap["pooled_mean_delta"],
            "ci_lower_one_sided_95": gap["pooled_ci_lower_one_sided"],
            "ci_two_sided_95": list(gap["pooled_ci_two_sided"]),
            "n_seed_clusters": gap["n_seed_clusters"],
            "n_boot": gap["n_boot"],
        },
        "decision_rule_bounds": bounds,
        "verdict": verdict,
        "verdict_note": (
            "Pre-committed rule (prereg §decision_rule, founder-confirmed 2026-07-14), "
            "recomputed from the committed per-episode JSONL via evaluate_cocarry_success "
            f"-- never from in-memory values. C0={c0_rate:.3f}, C2={c2_rate:.3f}, "
            f"C0-C2 gap lower bound {gap['pooled_ci_lower_one_sided']:.3f}; corroborating "
            f"dose-response C1(K=0)={c1_rate:.3f} (C3 per-cell rates in `cells`)."
        ),
        "episodes_artifact": (
            "spikes/results/cocarry/decouple_ablation/cocarry_decouple_ablation_episodes.jsonl"
        ),
        "trajectories_artifact": (
            "spikes/results/cocarry/decouple_ablation/cocarry_decouple_ablation_trajectories.json"
        ),
        "governance": (
            "Eval-only; no training/residual/heterogeneity; predicate + f_max frozen; only "
            "the coupling varied. Phase-2, NON-gating. Prior artifacts immutable (I8); no "
            "schema bump (I9); no Phase-1 / M10 code touched (I1)."
        ),
    }
    _OUT.write_text(json.dumps(out, sort_keys=True, indent=2), encoding="utf-8")
    _REPRO_TXT.write_text(
        f"{_REPRO_COMMAND}\n# prereg tag: {_PREREG_TAG}\n# prereg blob SHA: {prereg_blob_sha}\n",
        encoding="utf-8",
    )
    shasums = "".join(
        f"{_sha256_file(p)}  {p.name}\n" for p in (_OUT, _EPISODES, _TRAJ_OUT, _REPRO_TXT)
    )
    _SHASUMS.write_text(shasums, encoding="utf-8")

    # --- Print-back (recomputed table, then the verdict, then the SHA) ---
    print("\n  recomputed success table (from the committed per-episode JSONL):")
    for cell_key, res in cell_results.items():
        alias = f" ({'/'.join(res['aliases'])})" if res["aliases"] else ""
        lo, hi = res["success_ci95"]
        print(
            f"    {cell_key:>20}{alias:<6} success={res['summary']['success_rate']:.3f} "
            f"[{lo:.3f}, {hi:.3f}]  -> {res['classification']}"
        )
    print(
        f"    primary contrast C0-C2 = {gap['pooled_mean_delta']:.3f} (one-sided 95% lower "
        f"bound {gap['pooled_ci_lower_one_sided']:.3f}); corroborating dose-response "
        f"C1(K=0) = {c1_rate:.3f}"
    )
    print(f"\n  VERDICT = {verdict}")
    print(f"  measurement -> {_OUT}")
    print(f"  measurement SHA256 = {_sha256_file(_OUT)}")
    print(f"  total wall {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
