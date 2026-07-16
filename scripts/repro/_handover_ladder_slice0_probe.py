# SPDX-License-Identifier: Apache-2.0
"""Handover ladder Slice-0 oracle-headroom probe (eval-only, CPU-only, NON-gating).

Executes the pre-stated probe of
``spikes/results/handover-ladder-probe-2026-07-16/PROBE_PRESTATEMENT.md`` (the
rule-before-result record; ADR-007 §Discipline in pre-statement form — no prereg
tag is rotated, no gate claim is made, invariant I1 holds: nothing here gates
Phase-1/M10). The question: can a learned ego ladder on ``handover_place``
discriminate at all — is the nominal room above REF-SCRIPT *reachable* (any
action succeeds in those states) and *state-dependent* (a policy has something
to learn beyond a constant rule)?

Protocol (replication of the committed schedule, ADR-011 as amended; ADR-028):

* Cells: ``presenter_mismatch_30`` / ``presenter_mismatch_45`` from
  ``handover_place_partners@v1`` at the committed coupling-valid anchor
  (:data:`chamber.benchmarks.partner_probe.HANDOVER_PROBE_ENV_PARAMS`).
* Draws: the exact committed grid — seeds 0-4 x 50 episodes/partner,
  ``initial_state_seed = seed*1000 + episode`` — so the recomputed REF
  reconciles per-episode against the SHA-verified committed bundle
  ``spikes/results/benchmark/handover-v1/ref-script-2026-07-06/`` (I8: the
  committed archive is read-only; a reconciliation mismatch aborts with no
  verdict).
* Oracle: per-state search of the ego phase-1 action under the exact success
  predicate, driven through the real env step. The translate optimum is
  analytic (exact cancellation; presented offsets are ~3e-4 m against a 0.10 m
  range, and the residual-lateral norm enters the lateral window and the force
  proxy monotonically); ``reorient_deg x regrasp_flag`` is searched on a dense
  grid (the env clips wrist correction analytically, so reorient invariance is
  measured and reported rather than assumed).
* Fixed policies (learnable-structure comparators): always-regrasp,
  never-regrasp, and the scripted rule, each with the standard translate/wrist
  corrections.

Determinism (P6 / ADR-002): every draw routes through
``concerto.training.seeding.derive_substream`` (env reset + presenter reset per
``initial_state_seed``); the driver itself uses no wall-clock-dependent values
and no unseeded RNG anywhere in the measurement path.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from chamber.agents.handover_ego_scripted import ScriptedHandoverEgo
from chamber.benchmarks.handover_eval import run_handover_episodes_for_set
from chamber.benchmarks.partner_probe import HANDOVER_PROBE_ENV_PARAMS
from chamber.envs.handover_place import HandoverPlaceEnv, make_handover_place_env
from chamber.evaluation.bundles import compute_summary
from chamber.partners.registry import load_partner
from chamber.partners.sets import (
    PartnerMemberSpec,
    PartnerSetSpec,
    get_partner_set,
    resolve_set_members,
)

if TYPE_CHECKING:
    from chamber.evaluation.results import EpisodeResult
    from chamber.partners.api import FrozenPartner

_ARCHIVE_DIR = Path("spikes/results/handover-ladder-probe-2026-07-16")
_PRESTATEMENT = _ARCHIVE_DIR / "PROBE_PRESTATEMENT.md"
_RESULTS_OUT = _ARCHIVE_DIR / "probe_results.json"
_REPRO_TXT = _ARCHIVE_DIR / "REPRO.txt"
_PROBE_TAG = "probe-handover-ladder-slice0-2026-07-16"

_REF_BUNDLE_DIR = Path("spikes/results/benchmark/handover-v1/ref-script-2026-07-06")

_PARTNER_SET_SLUG = "handover_place_partners"
_PARTNER_SET_VERSION = 1
_EVAL_MEMBERS = ("presenter_mismatch_30", "presenter_mismatch_45")
_SEEDS = (0, 1, 2, 3, 4)
_EPISODES_PER_SEED = 50
_ROOT_SEED = 0
_N_RESAMPLES = 2000

#: Pre-stated decision-rule bounds (founder-confirmed at kickoff 2026-07-16).
_HEADROOM_BOUND = 0.15
_LEARNABLE_STRUCTURE_BOUND = 0.05

#: Dense reorient grid (the env clips the wrist correction analytically, so the
#: expectation is exact invariance across this axis — measured, not assumed).
_REORIENT_GRID_POINTS = 11
_REGRASP_FLAGS = (0.0, 1.0)

_FLOAT_ATOL = 1e-9

_REPRO_COMMAND = "uv run --no-sync python scripts/repro/_handover_ladder_slice0_probe.py"


def _sha256_file(path: Path) -> str:
    """SHA-256 hex digest of one file (streamed)."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_committed_bundle() -> dict[str, Any]:
    """SHA-verify the committed REF bundle before reading it (I8; ADR-018 custody).

    Every file named in the bundle's own ``SHA256SUMS.txt`` is re-hashed; a
    mismatch aborts the probe (the committed archive is the reconciliation
    anchor and must be byte-intact).
    """
    sums_path = _REF_BUNDLE_DIR / "SHA256SUMS.txt"
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, name = line.split()
        actual = _sha256_file(_REF_BUNDLE_DIR / name)
        if actual != expected:
            msg = f"committed bundle SHA mismatch for {name}: {actual} != {expected}"
            raise RuntimeError(msg)
    bundle = json.loads((_REF_BUNDLE_DIR / "bundle.json").read_text(encoding="utf-8"))
    return dict(bundle)


def _load_committed_episodes() -> dict[tuple[str, int, int], dict[str, Any]]:
    """Committed per-episode records keyed by (member, seed, initial_state_seed)."""
    records: dict[tuple[str, int, int], dict[str, Any]] = {}
    for seed in _SEEDS:
        path = _REF_BUNDLE_DIR / f"episodes_seed{seed}.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            key = (rec["metadata"]["member"], int(rec["seed"]), int(rec["initial_state_seed"]))
            records[key] = rec
    return records


def _resolve_members() -> tuple[PartnerSetSpec, list[tuple[PartnerMemberSpec, dict[str, str]]]]:
    """Resolve the two coupling-valid eval members in committed set order."""
    set_spec = get_partner_set(_PARTNER_SET_SLUG, version=_PARTNER_SET_VERSION)
    members = resolve_set_members(set_spec, include_private=False)
    members = [(m, p) for m, p in members if m.member_name in _EVAL_MEMBERS]
    found = [m.member_name for m, _ in members]
    if found != list(_EVAL_MEMBERS):
        msg = f"eval members resolved as {found}, expected {list(_EVAL_MEMBERS)}"
        raise RuntimeError(msg)
    return set_spec, members


def _recompute_ref(
    set_spec: PartnerSetSpec, members: list[tuple[PartnerMemberSpec, dict[str, str]]]
) -> list[EpisodeResult]:
    """Drive the actual scripted ego over the committed grid (never copied numbers)."""
    episodes_by_seed, _, _ = run_handover_episodes_for_set(
        policy_id="ref_script_handover_ego",
        set_spec=set_spec,
        members=members,
        seeds=list(_SEEDS),
        episodes_per_seed=_EPISODES_PER_SEED,
        root_seed=_ROOT_SEED,
    )
    return [ep for seed in _SEEDS for ep in episodes_by_seed[seed]]


def _reconcile_ref(
    recomputed: list[EpisodeResult],
    committed: dict[tuple[str, int, int], dict[str, Any]],
    committed_summary: dict[str, Any],
) -> dict[str, Any]:
    """The hard reconciliation gate: recomputed REF must match the committed row.

    Checks per-episode equality (success, force, residuals) against the
    committed JSONL and recomputes the bundle summary with the committed
    estimator settings. Any divergence raises — the pre-statement forbids
    proceeding to a verdict on a diverged protocol.
    """
    if len(recomputed) != len(committed):
        msg = f"episode count mismatch: recomputed {len(recomputed)} vs committed {len(committed)}"
        raise RuntimeError(msg)
    for ep in recomputed:
        member = str(ep.metadata["member"])
        rec = committed.get((member, ep.seed, ep.initial_state_seed))
        if rec is None:
            msg = f"no committed record for ({member}, seed {ep.seed}, iss {ep.initial_state_seed})"
            raise RuntimeError(msg)
        recomputed_force = float("nan") if ep.force_peak is None else float(ep.force_peak)
        same = (
            bool(rec["success"]) == ep.success
            and abs(float(rec["force_peak"]) - recomputed_force) <= _FLOAT_ATOL
            and abs(
                float(rec["metadata"]["residual_angular_deg"])
                - float(ep.metadata["residual_angular_deg"])
            )
            <= _FLOAT_ATOL
            and str(rec["metadata"]["failure_mode"]) == str(ep.metadata["failure_mode"])
        )
        if not same:
            msg = (
                f"per-episode reconciliation mismatch at ({member}, seed {ep.seed}, "
                f"iss {ep.initial_state_seed}): recomputed "
                f"(success={ep.success}, force={ep.force_peak!r}) vs committed "
                f"(success={rec['success']}, force={rec['force_peak']!r})"
            )
            raise RuntimeError(msg)
    summary = compute_summary(recomputed, n_resamples=_N_RESAMPLES, bootstrap_root_seed=_ROOT_SEED)
    for field, value in (
        ("success_mean", summary.success_mean),
        ("success_iqm", summary.success_iqm),
    ):
        if value is None or abs(float(value) - float(committed_summary[field])) > _FLOAT_ATOL:
            msg = (
                f"summary reconciliation mismatch on {field}: recomputed {value!r} "
                f"vs committed {committed_summary[field]!r}"
            )
            raise RuntimeError(msg)
    return {
        "per_episode_matches": len(recomputed),
        "per_episode_mismatches": 0,
        "recomputed_success_mean": summary.success_mean,
        "recomputed_success_iqm": summary.success_iqm,
        "committed_success_mean": float(committed_summary["success_mean"]),
        "committed_success_iqm": float(committed_summary["success_iqm"]),
    }


def _build_env(*, free_regrasp: bool = False) -> HandoverPlaceEnv:
    """The committed coupling-valid anchor env (CB-04 numbers, verbatim)."""
    p = HANDOVER_PROBE_ENV_PARAMS
    return make_handover_place_env(
        free_regrasp=free_regrasp,
        lateral_window_m=p["lateral_window_m"],
        angular_window_deg=p["angular_window_deg"],
        seating_force_limit_n=p["seating_force_limit_n"],
        translation_range_m=p["translation_range_m"],
        wrist_correction_deg=p["wrist_correction_deg"],
        reacquire_range_deg=p["reacquire_range_deg"],
        contact_stiffness_n_per_m=p["contact_stiffness_n_per_m"],
        angular_stiffness_n_per_deg=p["angular_stiffness_n_per_deg"],
        regrasp_budget_s=p["regrasp_budget_s"],
        regrasp_duration_s=p["regrasp_duration_s"],
    )


def _replay(
    env: HandoverPlaceEnv,
    presenter: FrozenPartner,
    iss: int,
    ego_action: np.ndarray,
) -> dict[str, Any]:
    """One full deterministic episode replay: reset, presentation, ego action."""
    obs, _ = env.reset(seed=iss)
    presenter.reset(seed=iss)
    obs, _, _, _, _ = env.step(np.asarray(presenter.act(obs), dtype=np.float64))
    _, _, _, _, info = env.step(np.asarray(ego_action, dtype=np.float64))
    return info


def _probe_member(
    member: PartnerMemberSpec,
    params: dict[str, str],
    env: HandoverPlaceEnv,
    env_free: HandoverPlaceEnv,
) -> list[dict[str, Any]]:
    """Oracle search + fixed policies for one presenter cell over the committed draws."""
    variant = "matched" if float(params.get("grasp_pose_bias_deg", "0")) == 0.0 else "mismatched"
    member_spec = member.partner_spec(params=params, seat_extra={"variant": variant})
    presenter = load_partner(member_spec)
    scripted = ScriptedHandoverEgo(
        translation_range_m=env.translation_range_m,
        wrist_correction_deg=env.wrist_correction_deg,
    )
    wrist = env.wrist_correction_deg
    reorient_grid = np.linspace(-wrist, wrist, _REORIENT_GRID_POINTS)

    records: list[dict[str, Any]] = []
    for seed in _SEEDS:
        for episode in range(_EPISODES_PER_SEED):
            iss = seed * 1000 + episode
            # Presented state (phase-0 driven by the registered presenter,
            # exactly as the committed eval does).
            obs, _ = env.reset(seed=iss)
            presenter.reset(seed=iss)
            presentation = np.asarray(presenter.act(obs), dtype=np.float64)
            obs1, _, _, _, _ = env.step(presentation)
            lateral = np.asarray(obs1["lateral_offset"], dtype=np.float64)
            err_deg = float(obs1["grasp_pose_error_deg"])
            skew_s = float(presentation[3])

            # Analytic translate optimum: exact cancellation (|offset| ~ 3e-4 m
            # against a 0.10 m range; residual norm is monotone in both the
            # lateral window and the force proxy).
            tx, ty = float(-lateral[0]), float(-lateral[1])

            # Oracle: dense grid over reorient x regrasp, each candidate driven
            # through the real env step on the same draw.
            best: dict[str, Any] | None = None
            outcomes_by_flag: dict[float, set[tuple[bool, float]]] = {}
            for flag in _REGRASP_FLAGS:
                for reorient in reorient_grid:
                    action = np.array([tx, ty, float(reorient), flag], dtype=np.float64)
                    info = _replay(env, presenter, iss, action)
                    outcomes_by_flag.setdefault(flag, set()).add(
                        (bool(info["success"]), round(float(info["seating_force_proxy_n"]), 12))
                    )
                    candidate = {
                        "success": bool(info["success"]),
                        "seating_force_proxy_n": float(info["seating_force_proxy_n"]),
                        "residual_angular_deg": float(info["residual_angular_deg"]),
                        "regrasp_flag": float(flag),
                        "regrasp_executed": bool(info["regrasp_executed"]),
                        "regrasp_budget_blocked": bool(info["regrasp_budget_blocked"]),
                        "failure_mode": str(info["failure_mode"]),
                    }
                    if best is None or (
                        (candidate["success"], -candidate["seating_force_proxy_n"])
                        > (best["success"], -best["seating_force_proxy_n"])
                    ):
                        best = candidate
            if best is None:  # pragma: no cover - grid is never empty
                raise RuntimeError("empty oracle candidate grid")
            reorient_invariant = all(len(v) == 1 for v in outcomes_by_flag.values())

            # Free-re-grasp counterfactual (budget removed) for the
            # infeasibility decomposition.
            free_action = np.array([tx, ty, 0.0, 1.0], dtype=np.float64)
            free_info = _replay(env_free, presenter, iss, free_action)
            if best["success"]:
                classification = "feasible_and_found"
            elif bool(free_info["success"]):
                classification = "infeasible_by_budget"
            else:
                classification = "infeasible_by_window"

            # Fixed policies on the same draw.
            scripted_action = scripted.act(obs1)
            fixed_actions = {
                "scripted_rule": np.asarray(scripted_action, dtype=np.float64),
                "always_regrasp": np.array(
                    [tx, ty, float(np.clip(-err_deg, -wrist, wrist)), 1.0], dtype=np.float64
                ),
                "never_regrasp": np.array(
                    [tx, ty, float(np.clip(-err_deg, -wrist, wrist)), 0.0], dtype=np.float64
                ),
            }
            fixed = {
                name: {
                    "success": bool(info["success"]),
                    "seating_force_proxy_n": float(info["seating_force_proxy_n"]),
                }
                for name, info in (
                    (name, _replay(env, presenter, iss, action))
                    for name, action in fixed_actions.items()
                )
            }

            records.append(
                {
                    "member": member.member_name,
                    "seed": seed,
                    "episode": episode,
                    "initial_state_seed": iss,
                    "presented": {
                        "lateral_offset_m": [float(lateral[0]), float(lateral[1])],
                        "grasp_pose_error_deg": err_deg,
                        "timing_skew_s": skew_s,
                    },
                    "oracle": best,
                    "reorient_invariant": reorient_invariant,
                    "classification": classification,
                    "free_regrasp_success": bool(free_info["success"]),
                    "fixed": fixed,
                }
            )
    return records


def _cell_table(
    records: list[dict[str, Any]], ref_episodes: list[EpisodeResult]
) -> dict[str, dict[str, Any]]:
    """Per-cell means, decomposition, and seating-force summaries."""
    ref_by_member: dict[str, list[EpisodeResult]] = {}
    for ep in ref_episodes:
        ref_by_member.setdefault(str(ep.metadata["member"]), []).append(ep)

    cells: dict[str, dict[str, Any]] = {}
    for member in _EVAL_MEMBERS:
        recs = [r for r in records if r["member"] == member]
        refs = ref_by_member[member]
        n = len(recs)
        oracle_success = [float(r["oracle"]["success"]) for r in recs]
        fixed_means = {
            name: float(np.mean([float(r["fixed"][name]["success"]) for r in recs]))
            for name in ("always_regrasp", "never_regrasp", "scripted_rule")
        }
        decomposition = {
            label: sum(1 for r in recs if r["classification"] == label)
            for label in ("feasible_and_found", "infeasible_by_budget", "infeasible_by_window")
        }
        oracle_forces = np.asarray([r["oracle"]["seating_force_proxy_n"] for r in recs])
        ref_forces = np.asarray([ep.force_peak for ep in refs])
        cells[member] = {
            "n_episodes": n,
            "oracle_mean": float(np.mean(oracle_success)),
            "ref_mean": float(np.mean([1.0 if ep.success else 0.0 for ep in refs])),
            "fixed_means": fixed_means,
            "best_fixed_mean": max(fixed_means.values()),
            "decomposition": decomposition,
            "reorient_invariant_all": all(r["reorient_invariant"] for r in recs),
            "seating_force_proxy_n": {
                "oracle": {
                    "mean": float(np.mean(oracle_forces)),
                    "p50": float(np.percentile(oracle_forces, 50)),
                    "p90": float(np.percentile(oracle_forces, 90)),
                },
                "ref": {
                    "mean": float(np.mean(ref_forces)),
                    "p50": float(np.percentile(ref_forces, 50)),
                    "p90": float(np.percentile(ref_forces, 90)),
                },
            },
        }
    return cells


def _verdict(cells: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Apply the pre-stated decision rule verbatim (GO iff both conditions hold)."""
    headroom = {m: c["oracle_mean"] - c["ref_mean"] for m, c in cells.items()}
    learnable = {m: c["oracle_mean"] - c["best_fixed_mean"] for m, c in cells.items()}
    headroom_pass = any(v >= _HEADROOM_BOUND for v in headroom.values())
    learnable_pass = any(v >= _LEARNABLE_STRUCTURE_BOUND for v in learnable.values())
    return {
        "headroom_delta_by_cell": headroom,
        "headroom_bound": _HEADROOM_BOUND,
        "headroom_condition_holds": headroom_pass,
        "learnable_structure_delta_by_cell": learnable,
        "learnable_structure_bound": _LEARNABLE_STRUCTURE_BOUND,
        "learnable_structure_condition_holds": learnable_pass,
        "verdict": "GO" if (headroom_pass and learnable_pass) else "NO-GO",
    }


def main() -> int:
    """Run the pre-stated Slice-0 probe end to end and write the archive artifacts."""
    t0 = time.time()
    if not _PRESTATEMENT.exists():
        msg = f"pre-statement missing at {_PRESTATEMENT}; the rule must precede the result"
        raise RuntimeError(msg)

    print("verifying the committed REF bundle (SHA-256) ...")
    bundle = _verify_committed_bundle()
    committed = _load_committed_episodes()
    set_spec, members = _resolve_members()

    print("recomputing REF over the committed draws (live scripted ego) ...")
    ref_episodes = _recompute_ref(set_spec, members)
    reconciliation = _reconcile_ref(ref_episodes, committed, bundle["summary"])
    print(
        "  reconciliation OK: "
        f"{reconciliation['per_episode_matches']}/500 episodes match; "
        f"mean {reconciliation['recomputed_success_mean']:.3f} "
        f"(committed {reconciliation['committed_success_mean']:.3f}), "
        f"IQM {reconciliation['recomputed_success_iqm']:.3f} "
        f"(committed {reconciliation['committed_success_iqm']:.3f})"
    )

    env = _build_env()
    env_free = _build_env(free_regrasp=True)
    records: list[dict[str, Any]] = []
    for member, params in members:
        print(f"probing cell {member.member_name} (oracle grid + fixed policies) ...")
        records.extend(_probe_member(member, params, env, env_free))

    cells = _cell_table(records, ref_episodes)
    verdict = _verdict(cells)

    out = {
        "probe": "handover-ladder-slice0-oracle-headroom",
        "prestatement": {
            "path": str(_PRESTATEMENT),
            "sha256": _sha256_file(_PRESTATEMENT),
            "probe_tag": _PROBE_TAG,
        },
        "committed_ref_bundle": {
            "path": str(_REF_BUNDLE_DIR),
            "git_sha": bundle["git_sha"],
            "prereg_git_tag": bundle["prereg_git_tag"],
            "summary": bundle["summary"],
            "sha_verified": True,
        },
        "env_params": dict(HANDOVER_PROBE_ENV_PARAMS),
        "schedule": {
            "seeds": list(_SEEDS),
            "episodes_per_seed_per_partner": _EPISODES_PER_SEED,
            "pairing": "initial_state_seed = seed*1000 + episode",
            "eval_members": list(_EVAL_MEMBERS),
        },
        "oracle_search": {
            "translate": "analytic exact cancellation (see pre-statement)",
            "reorient_grid_points": _REORIENT_GRID_POINTS,
            "regrasp_flags": list(_REGRASP_FLAGS),
        },
        "reconciliation": reconciliation,
        "cells": cells,
        "decision": verdict,
        "governance": (
            "Eval-only, CPU-only, NON-gating (I1). No src/ behaviour change; the "
            "committed archives are read SHA-verified and never modified (I8); no "
            "schema constants; no prereg-tag rotation. Informs the v1.1 learned-"
            "ladder scope decision only."
        ),
        "per_episode": records,
    }
    _ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    _RESULTS_OUT.write_text(json.dumps(out, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    _REPRO_TXT.write_text(
        f"{_REPRO_COMMAND}\n"
        f"# probe tag (pre-statement commit): {_PROBE_TAG}\n"
        f"# committed REF bundle: {_REF_BUNDLE_DIR} (git {bundle['git_sha']})\n"
        f"# reproduce at the results commit of this archive; the run is\n"
        f"# deterministic (P6/ADR-002) and rewrites probe_results.json byte-identically.\n",
        encoding="utf-8",
    )

    print("\n  per-cell table (oracle vs REF vs fixed):")
    for member, cell in cells.items():
        fixed = cell["fixed_means"]
        print(
            f"    {member:>22}  oracle={cell['oracle_mean']:.3f}  ref={cell['ref_mean']:.3f}  "
            f"always={fixed['always_regrasp']:.3f}  never={fixed['never_regrasp']:.3f}  "
            f"scripted={fixed['scripted_rule']:.3f}"
        )
        print(
            f"    {'':>22}  decomposition={cell['decomposition']}  "
            f"reorient_invariant={cell['reorient_invariant_all']}"
        )
    print(f"\n  VERDICT (pre-stated rule) = {verdict['verdict']}")
    print(f"  headroom deltas: {verdict['headroom_delta_by_cell']}")
    print(f"  learnable-structure deltas: {verdict['learnable_structure_delta_by_cell']}")
    print(f"  results -> {_RESULTS_OUT}")
    print(f"  results SHA256 = {_sha256_file(_RESULTS_OUT)}")
    print(f"  total wall {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
