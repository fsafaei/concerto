# SPDX-License-Identifier: Apache-2.0
r"""Admission-cell runners + wrap extractors (ADR-027 §Admission protocol).

The measured half of :mod:`chamber.evaluation.admission`: each cell
runner drives one preregistered policy/partner configuration of one
task through its **existing** evaluation machinery — the co-carry
matched-controller rig (ADR-026 §Decision 1-2), the Stage-1 pick-place
env with the REF-SCRIPT scripted oracle (ADR-011 as amended), the
handover-place kinematic resolver — and returns the raw
:class:`chamber.evaluation.results.EpisodeResult` records that
:func:`chamber.evaluation.admission.run_admission` wraps into
``chamber-eval verify``-passing v3 bundles (ADR-028 §Decision 1).

Wrap extractors are the committed-evidence half (I8): they re-extract
statistics from SHA-verified immutable archive files (the
handover-place Gate-0 archive) — wrapping, never re-running, never
hand-copying numbers.

Registry style mirrors ADR-009 §Decision: module-level tables, loud
``KeyError`` listing the known keys. SAPIEN / ManiSkill imports stay
inside runner bodies (P2 wrapper-only; ADR-001 §Risks) so this module
is Tier-1-importable on a Vulkan-less host.
"""

from __future__ import annotations

import gzip
import json
from typing import TYPE_CHECKING, Any

import numpy as np

from chamber.evaluation.admission import (
    AdmissionCellSpec,
    AdmissionError,
    CellRun,
    WrappedEvidenceSpec,
)
from chamber.evaluation.results import EpisodeResult
from chamber.partners.ablation import PARTNER_ABLATED_ZERO_CLASS
from chamber.partners.api import PartnerSpec
from chamber.partners.cocarry_blind import COCARRY_BLIND_IMPEDANCE_CLASS
from chamber.partners.registry import load_partner

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable
    from pathlib import Path

    from chamber.partners.api import FrozenPartner

# Imports for the @register_partner side effects so every admission
# partner name resolves when this module is the entry point.
import chamber.partners.ablation
import chamber.partners.cocarry_blind
import chamber.partners.cocarry_impedance
import chamber.partners.heuristic  # noqa: F401

#: ``derive_substream`` labels recorded in cell seed schedules
#: (ADR-002 P6). The env-side substreams are owned by the envs
#: themselves; these labels document them in the bundle.
COCARRY_SUBSTREAM_LABELS: tuple[str, ...] = ("env.cocarry", "evaluation.bundle_bootstrap")
PICKPLACE_SUBSTREAM_LABELS: tuple[str, ...] = (
    "envs.stage1_pickplace",
    "evaluation.bundle_bootstrap",
)
HANDOVER_SUBSTREAM_LABELS: tuple[str, ...] = (
    "env.handover_place",
    "evaluation.bundle_bootstrap",
)


def _to_float(value: Any) -> float:  # noqa: ANN401 - torch/np scalar
    """Coerce a torch / numpy scalar-or-(1,) value to a Python float (env 0)."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return float(np.asarray(value).reshape(-1)[0])


def _partner_material(name: str, spec: PartnerSpec) -> dict[str, object]:
    """Serialize one partner-spec entry for ``partners.json`` (ADR-028 §Decision 1)."""
    return {
        "name": name,
        "class_name": spec.class_name,
        "seed": spec.seed,
        "checkpoint_step": spec.checkpoint_step,
        "weights_uri": spec.weights_uri,
        "extra": dict(spec.extra),
    }


# ---------------------------------------------------------------------------
# Co-carry cells (ADR-026 §Decision 1-2 rig; ADR-027 §Admission protocol).
# ---------------------------------------------------------------------------


def run_cocarry_cell(
    *,
    cell: AdmissionCellSpec,
    seeds: list[int],
    episodes_per_seed: int,
    root_seed: int,
    render_backend: str | None = None,
) -> CellRun:
    """Run one co-carry admission cell (ADR-027 §Admission protocol; ADR-026 §Decision 1-2).

    ``cell.params``:

    - ``condition_id`` — ``cocarry_matched_panda_pair`` (A1/A3) or
      ``cocarry_single_arm_positive_control`` (A2).
    - ``ego`` — ``"impedance"`` (the matched reference,
      :class:`chamber.partners.cocarry_impedance.CoCarryImpedancePartner`)
      or ``"blind"`` (B-BLIND,
      :class:`chamber.partners.cocarry_blind.CoCarryBlindImpedancePartner`).
    - ``episode_length`` — optional truncation override.

    **One fresh env per episode** (``root_seed = seed * 1000 + episode``
    — the Gate-0 ``initial_state_seed`` convention): re-using a stepped
    co-carry env across resets is a measured rig artifact (the dual-hold
    attach carries state into the next episode and produces spurious
    bar-fight failures — tilt 29-33° on draws that succeed cleanly from
    a fresh build), and every rung measurement drove this env one
    stepped episode per build. Two cells sharing a seed schedule see
    identical ``initial_state_seed`` sequences — the A3 pairing
    precondition. ``force_peak`` records the canonical wrist stress
    instrument's episode maximum (ADR-027 §Versioning).
    """
    del root_seed  # per ADR-002 the co-carry substream keys on the cluster seed
    from chamber.envs.cocarry import (
        COCARRY_DEFAULT_EPISODE_LENGTH,
        cocarry_matched_controller_specs,
        make_cocarry_env,
    )

    condition_id = str(cell.params.get("condition_id", "cocarry_matched_panda_pair"))
    ego_kind = str(cell.params.get("ego", "impedance"))
    episode_length = int(cell.params.get("episode_length", COCARRY_DEFAULT_EPISODE_LENGTH))
    ego_class = {
        "impedance": "cocarry_impedance",
        "blind": COCARRY_BLIND_IMPEDANCE_CLASS,
    }.get(ego_kind)
    if ego_class is None:
        msg = f"unknown co-carry ego kind {ego_kind!r}; known: impedance, blind"
        raise AdmissionError(msg)

    episodes_by_seed: dict[int, list[EpisodeResult]] = {}
    material: list[dict[str, object]] = []
    hashes: dict[str, str] = {}
    controller_specs = cocarry_matched_controller_specs()
    for cluster_seed in seeds:
        records: list[EpisodeResult] = []
        for episode in range(episodes_per_seed):
            iss = int(cluster_seed) * 1000 + episode
            env = make_cocarry_env(
                condition_id=condition_id,
                episode_length=episode_length,
                root_seed=iss,
                render_backend=render_backend,
            )
            try:
                single_arm = bool(env.single_arm)  # type: ignore[attr-defined]
                ego_uid = str(env.ego_uid)  # type: ignore[attr-defined]
                partner_uid = str(env.partner_uid)  # type: ignore[attr-defined]
                ego_spec = PartnerSpec(ego_class, 0, None, None, dict(controller_specs[ego_uid]))
                ego = load_partner(ego_spec)
                partner_dim = int(env.action_space.spaces[partner_uid].shape[0])  # type: ignore[attr-defined,index]
                if single_arm:
                    partner_spec = PartnerSpec(
                        PARTNER_ABLATED_ZERO_CLASS, 0, None, None, {"action_dim": str(partner_dim)}
                    )
                else:
                    partner_spec = PartnerSpec(
                        "cocarry_impedance", 0, None, None, dict(controller_specs[partner_uid])
                    )
                partner = load_partner(partner_spec)
                if not material:
                    material = [
                        _partner_material(f"ego:{ego_class}", ego_spec),
                        _partner_material(f"partner:{partner_spec.class_name}", partner_spec),
                    ]
                    hashes = {
                        f"ego:{ego_class}": ego_spec.partner_id,
                        f"partner:{partner_spec.class_name}": partner_spec.partner_id,
                    }
                obs, _ = env.reset(seed=iss)
                ego.reset(seed=iss)
                partner.reset(seed=iss)
                info: dict[str, Any] = {}
                for _ in range(episode_length):
                    action = {ego_uid: ego.act(obs), partner_uid: partner.act(obs)}
                    obs, _, terminated, truncated, info = env.step(action)
                    if _to_float(terminated) or _to_float(truncated):
                        break
                tel = env.get_telemetry()  # type: ignore[attr-defined]
                records.append(
                    EpisodeResult(
                        seed=cluster_seed,
                        episode_idx=episode,
                        initial_state_seed=iss,
                        success=bool(_to_float(info["success"])),
                        force_peak=_to_float(tel["max_stress_proxy"]),
                        metadata={
                            "condition": condition_id,
                            "ego": ego_class,
                            "partner": partner_spec.class_name,
                            "max_tilt_deg": _to_float(tel["max_tilt_deg"]),
                            "centroid_to_goal": _to_float(tel["centroid_to_goal"]),
                        },
                    )
                )
            finally:
                env.close()
        episodes_by_seed[cluster_seed] = records
    return CellRun(
        episodes_by_seed=episodes_by_seed,
        partner_material=material,
        partner_hashes=hashes,
        substream_labels=list(COCARRY_SUBSTREAM_LABELS),
    )


# ---------------------------------------------------------------------------
# Stage-1 pick-place cells (ADR-026 §Decision 3 control; ADR-011 REF-SCRIPT).
# ---------------------------------------------------------------------------


def run_pickplace_cell(
    *,
    cell: AdmissionCellSpec,
    seeds: list[int],
    episodes_per_seed: int,
    root_seed: int,
    render_backend: str | None = None,
) -> CellRun:
    """Run one Stage-1 pick-place admission cell (ADR-027 §Admission protocol).

    ``cell.params``:

    - ``condition_id`` — a ``_CONDITION_TABLE`` key (default the
      canonical heterogeneous AS cell, ADR-027 §Versioning).
    - ``variant`` — ``"reference"`` (REF-SCRIPT ego + the task's
      scripted partner), ``"partner_ablated"`` (A2: the partner seat
      zeroed by construction), or ``"partner_blind"`` (A3: the same
      ego with the partner obs subtree masked — B-BLIND enforcement).

    The ego is the scripted-competent oracle
    (:class:`chamber.agents.pickplace_ego_scripted.ScriptedPickPlaceEgo`);
    on this task it reads no partner leaf in any variant — the measured
    construct fact ADR-026 §Decision 3 recorded, which is exactly what
    the A2/A3 cells demonstrate with numbers. The task has no stress
    channel (``stress_channel=None``, ADR-027), so ``force_peak`` stays
    ``None``.
    """
    del root_seed, render_backend  # single-env CPU-side stepping; env owns its P6 streams
    from chamber.agents.pickplace_ego_scripted import (
        PICKPLACE_EGO_UID,
        ScriptedPickPlaceEgo,
    )
    from chamber.envs.stage1_pickplace import (
        make_stage1_pickplace_env,
    )

    condition_id = str(
        cell.params.get("condition_id", "stage1_pickplace_panda_plus_fetch_ego_aht_happo_per_agent")
    )
    variant = str(cell.params.get("variant", "reference"))
    if variant not in ("reference", "partner_ablated", "partner_blind"):
        msg = (
            f"unknown pick-place variant {variant!r}; "
            "known: reference, partner_ablated, partner_blind"
        )
        raise AdmissionError(msg)

    episodes_by_seed: dict[int, list[EpisodeResult]] = {}
    material: list[dict[str, object]] = []
    hashes: dict[str, str] = {}
    for cluster_seed in seeds:
        env = make_stage1_pickplace_env(condition_id=condition_id, root_seed=cluster_seed)
        try:
            uids = list(env.action_space.spaces)  # type: ignore[attr-defined]
            ego_uid = PICKPLACE_EGO_UID
            (partner_uid,) = [u for u in uids if u != ego_uid]
            partner_dim = int(env.action_space.spaces[partner_uid].shape[0])  # type: ignore[attr-defined,index]
            ego = ScriptedPickPlaceEgo(
                mask_partner_obs=(variant == "partner_blind"), partner_uid=partner_uid
            )
            if variant == "partner_ablated":
                partner_spec = PartnerSpec(
                    PARTNER_ABLATED_ZERO_CLASS, 0, None, None, {"action_dim": str(partner_dim)}
                )
            else:
                partner_spec = PartnerSpec(
                    "scripted_heuristic",
                    0,
                    None,
                    None,
                    {"action_dim": str(partner_dim), "target_xy": "0.0,0.0"},
                )
            partner: FrozenPartner = load_partner(partner_spec)
            if not material:
                material = [_partner_material(f"partner:{partner_spec.class_name}", partner_spec)]
                hashes = {f"partner:{partner_spec.class_name}": partner_spec.partner_id}
            records: list[EpisodeResult] = []
            max_steps = int(cell.params.get("episode_length", 100))
            for episode in range(episodes_per_seed):
                obs, _ = env.reset(seed=episode)
                ego.reset(seed=episode)
                partner.reset(seed=episode)
                info: dict[str, Any] = {}
                for _ in range(max_steps):
                    action = {ego_uid: ego.act(obs), partner_uid: partner.act(obs)}
                    obs, _, terminated, truncated, info = env.step(action)
                    if _to_float(terminated) or _to_float(truncated):
                        break
                records.append(
                    EpisodeResult(
                        seed=cluster_seed,
                        episode_idx=episode,
                        initial_state_seed=episode,
                        success=bool(_to_float(info.get("success", False))),
                        metadata={
                            "condition": condition_id,
                            "variant": variant,
                            "partner": partner_spec.class_name,
                            "end_phase": ego.phase,
                        },
                    )
                )
            episodes_by_seed[cluster_seed] = records
        finally:
            env.close()
    return CellRun(
        episodes_by_seed=episodes_by_seed,
        partner_material=material,
        partner_hashes=hashes,
        substream_labels=list(PICKPLACE_SUBSTREAM_LABELS),
    )


# ---------------------------------------------------------------------------
# Handover-place presenter-ablated cell (ADR-027 A2; ADR-026 §Decision 2).
# ---------------------------------------------------------------------------


def run_handover_ablated_cell(
    *,
    cell: AdmissionCellSpec,
    seeds: list[int],
    episodes_per_seed: int,
    root_seed: int,
    render_backend: str | None = None,
) -> CellRun:
    """Run the presenter-ablated handover-place A2 cell (ADR-027 §Admission protocol).

    ``cell.params`` carry the committed env keywords verbatim (the
    Gate-0 prereg numbers — windows, budget, ranges, stiffnesses; see
    :func:`chamber.envs.handover_place.make_handover_place_env`). The
    env is built with ``presenter_ablated=True``: the phase-0
    presentation event carries no part, so the part never enters the
    ego's workspace and the placement resolves honestly against the
    out-of-reach staging offset (success ≈ 0 structurally, demonstrated
    rather than asserted). ``force_peak`` records the seating-force
    proxy — the task's canonical stress channel (ADR-027 §Versioning).
    """
    del root_seed, render_backend  # pure-Python kinematic env; P6 seeds per episode below
    from chamber.agents.handover_ego_scripted import (
        ScriptedHandoverEgo,
    )
    from chamber.envs.handover_place import (
        make_handover_place_env,
    )

    env_params = {str(k): v for k, v in dict(cell.params).items()}
    env = make_handover_place_env(presenter_ablated=True, **env_params)
    ego = ScriptedHandoverEgo(
        translation_range_m=env.translation_range_m,
        wrist_correction_deg=env.wrist_correction_deg,
    )
    ablated_spec = PartnerSpec(PARTNER_ABLATED_ZERO_CLASS, 0, None, None, {"action_dim": "4"})
    presenter_seat = load_partner(ablated_spec)
    episodes_by_seed: dict[int, list[EpisodeResult]] = {}
    for cluster_seed in seeds:
        records: list[EpisodeResult] = []
        for episode in range(episodes_per_seed):
            iss = int(cluster_seed) * 1000 + episode  # the Gate-0 pairing convention
            obs, _ = env.reset(seed=iss)
            presenter_seat.reset(seed=iss)
            ego.reset(seed=iss)
            obs, _, _, _, _ = env.step(presenter_seat.act(obs))
            _, _, _, _, info = env.step(ego.act(obs))
            records.append(
                EpisodeResult(
                    seed=int(cluster_seed),
                    episode_idx=episode,
                    initial_state_seed=iss,
                    success=bool(info["success"]),
                    force_peak=float(info["seating_force_proxy_n"]),
                    metadata={
                        "condition": "handover_presenter_ablated",
                        "residual_lateral_m": float(info["residual_lateral_m"]),
                        "residual_angular_deg": float(info["residual_angular_deg"]),
                        "binding_conjunct": str(info["binding_conjunct"]),
                        "failure_mode": str(info["failure_mode"]),
                    },
                )
            )
        episodes_by_seed[int(cluster_seed)] = records
    return CellRun(
        episodes_by_seed=episodes_by_seed,
        partner_material=[
            _partner_material(f"presenter:{PARTNER_ABLATED_ZERO_CLASS}", ablated_spec)
        ],
        partner_hashes={f"presenter:{PARTNER_ABLATED_ZERO_CLASS}": ablated_spec.partner_id},
        substream_labels=list(HANDOVER_SUBSTREAM_LABELS),
    )


# ---------------------------------------------------------------------------
# Wrap extractors — the handover-place Gate-0 archive (I8).
# ---------------------------------------------------------------------------


def _load_crossover_cells(repo_path: Path, spec: WrappedEvidenceSpec) -> list[dict[str, Any]]:
    """Load the Gate-0 ``crossover_curves.json`` cell list from the SHA-verified archive."""
    rel = str(spec.params.get("crossover_file", f"{spec.archive}/crossover_curves.json"))
    if rel not in spec.files:
        msg = f"wrap extractor requires {rel} to be SHA-pinned in the prereg files map"
        raise AdmissionError(msg)
    path = repo_path / rel
    opener = gzip.open if rel.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as fh:  # type: ignore[operator]
        payload = json.load(fh)
    cells = payload.get("cells")
    if not isinstance(cells, list):
        msg = f"{rel}: no 'cells' list — not a Gate-0 crossover file"
        raise AdmissionError(msg)
    return cells


def extract_handover_gate0_limb1(*, repo_path: Path, spec: WrappedEvidenceSpec) -> dict[str, float]:
    """A1 ← Gate-0 Limb 1 (matched solvability on the realistic takt band; I8).

    Re-extracts the per-cell matched IQM / CI-lower minima over the
    cells inside ``params["takt_band_s"]`` from the SHA-verified
    ``crossover_curves.json`` — the numbers PR #263 committed, never
    hand-copied (ADR-016/I8).
    """
    band = spec.params.get("takt_band_s", [1.0, 5.0])
    lo, hi = float(band[0]), float(band[1])
    cells = [c for c in _load_crossover_cells(repo_path, spec) if lo <= float(c["takt_s"]) <= hi]
    if not cells:
        msg = f"no Gate-0 cells inside the takt band [{lo}, {hi}]"
        raise AdmissionError(msg)
    iqms = [float(c["matched_iqm"]) for c in cells]
    ci_lows = [float(c["matched_ci_low"]) for c in cells]
    return {
        "n_cells": float(len(cells)),
        "success_iqm": min(iqms),
        "success_ci_low": min(ci_lows),
        # The archive records no matched CI upper bound; the band-max IQM
        # is the conservative stand-in (only used to separate FAIL from
        # INDETERMINATE, never to pass a check).
        "success_ci_high": max(iqms),
    }


def extract_handover_gate0_limb2(*, repo_path: Path, spec: WrappedEvidenceSpec) -> dict[str, float]:
    """A3 ← Gate-0 Limb 2 restricted to the coupling-valid cell family (ADR-027 A3; I8).

    Re-extracts the matched-vs-mismatched gap CIs over the pinned
    coupling-valid family — ``params``: ``clearance_factor`` (0.2),
    ``mismatch_bias_deg`` ([30, 45]), optional ``takt_band_s`` — from
    the SHA-verified ``crossover_curves.json``. Gaps are converted from
    percentage points to fractions; the family *minimum* CI lower bound
    is reported (the weakest family member must clear ``delta_min``).
    """
    clearance = float(spec.params.get("clearance_factor", 0.2))
    biases = {float(b) for b in spec.params.get("mismatch_bias_deg", [30.0, 45.0])}
    band = spec.params.get("takt_band_s", [1.0, 5.0])
    lo, hi = float(band[0]), float(band[1])
    cells = [
        c
        for c in _load_crossover_cells(repo_path, spec)
        if float(c["clearance_factor"]) == clearance
        and float(c["mismatch_bias_deg"]) in biases
        and lo <= float(c["takt_s"]) <= hi
        and bool(c["coupling_valid"])
    ]
    if not cells:
        msg = (
            f"no coupling-valid Gate-0 cells at clearance {clearance} / "
            f"biases {sorted(biases)} inside takt band [{lo}, {hi}]"
        )
        raise AdmissionError(msg)
    gap_lows = [float(c["gap_ci_low_pp"]) / 100.0 for c in cells]
    gap_highs = [float(c["gap_ci_high_pp"]) / 100.0 for c in cells]
    gaps = [float(c["gap_pp"]) / 100.0 for c in cells]
    return {
        "n_cells": float(len(cells)),
        "delta_iqm": min(gaps),
        "delta_ci_low": min(gap_lows),
        "delta_ci_high": max(gap_highs),
        "delta_max": max(gaps),
    }


#: Cell-runner registry (ADR-009 §Decision registry style).
CELL_RUNNERS: dict[str, Callable[..., CellRun]] = {
    "cocarry_scripted": run_cocarry_cell,
    "pickplace_scripted": run_pickplace_cell,
    "handover_presenter_ablated": run_handover_ablated_cell,
}

#: Wrap-extractor registry (ADR-009 §Decision registry style; I8).
WRAP_EXTRACTORS: dict[str, Callable[..., dict[str, float]]] = {
    "handover_gate0_limb1": extract_handover_gate0_limb1,
    "handover_gate0_limb2": extract_handover_gate0_limb2,
}


def resolve_cell_runner(name: str) -> Callable[..., CellRun]:
    """Resolve a cell runner, loud-failing with the known keys (ADR-009 §Decision)."""
    try:
        return CELL_RUNNERS[name]
    except KeyError:
        known = ", ".join(sorted(CELL_RUNNERS)) or "<none>"
        msg = f"unknown admission cell runner {name!r}; known: {known}"
        raise KeyError(msg) from None


def resolve_wrap_extractor(name: str) -> Callable[..., dict[str, float]]:
    """Resolve a wrap extractor, loud-failing with the known keys (ADR-009 §Decision)."""
    try:
        return WRAP_EXTRACTORS[name]
    except KeyError:
        known = ", ".join(sorted(WRAP_EXTRACTORS)) or "<none>"
        msg = f"unknown admission wrap extractor {name!r}; known: {known}"
        raise KeyError(msg) from None


__all__ = [
    "CELL_RUNNERS",
    "COCARRY_SUBSTREAM_LABELS",
    "HANDOVER_SUBSTREAM_LABELS",
    "PICKPLACE_SUBSTREAM_LABELS",
    "WRAP_EXTRACTORS",
    "extract_handover_gate0_limb1",
    "extract_handover_gate0_limb2",
    "resolve_cell_runner",
    "resolve_wrap_extractor",
    "run_cocarry_cell",
    "run_handover_ablated_cell",
    "run_pickplace_cell",
]
