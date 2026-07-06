# SPDX-License-Identifier: Apache-2.0
r"""Partner-set probe suite — fingerprints + the competence floor (ADR-009 as amended 2026-07-05).

Runs one set member against the task's **reference ego** on the
committed probe schedule (:attr:`~chamber.partners.sets.PartnerSetSpec.probe_seeds`
x :attr:`~chamber.partners.sets.PartnerSetSpec.probe_episodes_per_seed`,
~20 episodes) and returns the raw
:class:`chamber.evaluation.results.EpisodeResult` records that (a) the
behavioural **fingerprint** aggregates over — summary statistics of the
member's action distribution and the dyad's stress channel, so a reader
can tell members apart without policy access (ADR-018/I3) — and (b) the
committed **matched-pair floor** is checked against before a member is
admitted to the set (a member no ego can work with measures nothing;
the co-insert Gate-0 lesson: a member that freezes is a wall, not a
partner).

Per-task reference egos (the same rigs the CB-04 admission cells drove,
:mod:`chamber.benchmarks.admission_cells`):

- ``cocarry`` — the default-gain matched impedance controller on the
  ego seat (ADR-026 §Decision 1); one fresh env per episode (the
  measured dual-hold rig artifact rule).
- ``stage1_pickplace_as`` — the REF-SCRIPT scripted oracle (ADR-011 as
  amended).
- ``handover_place`` — the scripted analytic corrector at the canonical
  coupling-valid anchor cell (the CB-04 A2 committed env numbers);
  the ``free_regrasp`` floor variant removes the re-grasp budget
  (Gate-0's endpoint that splits budget-mediated from intrinsic
  binding) so competence is judged with budget pressure off.

Registry style mirrors ADR-009 §Decision (loud ``KeyError`` with known
keys); SAPIEN imports stay inside runner bodies (P2 wrapper-only,
ADR-001 §Risks) so the module is Tier-1-importable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from chamber.evaluation.admission import CellRun
from chamber.evaluation.results import EpisodeResult
from chamber.partners.api import PartnerSpec
from chamber.partners.registry import load_partner

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Mapping

    from numpy.typing import NDArray

    from chamber.partners.sets import PartnerMemberSpec, PartnerSetSpec

#: Probe variants (ADR-009 as amended): the fingerprint probe, and the
#: handover free-re-grasp floor probe (see
#: :attr:`chamber.partners.sets.PartnerSetSpec.floor_probe`).
ProbeVariant = Literal["fingerprint", "free_regrasp"]

#: Reference-ego policy ids recorded in probe bundles, per task
#: (ADR-011 as amended naming; matches the CB-04 admission cells).
REFERENCE_EGO_IDS: dict[str, str] = {
    "cocarry": "ref_script_cocarry_impedance",
    "stage1_pickplace_as": "ref_script_pickplace",
    "handover_place": "ref_script_handover_ego",
}

#: The handover-place probe env — the canonical coupling-valid anchor
#: cell's committed numbers, verbatim from the CB-04 admission prereg
#: (tag ``prereg-admission-handover-place-2026-07-05``: clearance 0.2 →
#: wrist_correction 25°, fast arm basis, takt 1.5 s → budget 1.13 s;
#: windows/stiffnesses from the tagged Gate-0 prereg).
HANDOVER_PROBE_ENV_PARAMS: dict[str, float] = {
    "lateral_window_m": 1.0e-3,
    "angular_window_deg": 5.0,
    "seating_force_limit_n": 75.0,
    "translation_range_m": 0.10,
    "wrist_correction_deg": 25.0,
    "reacquire_range_deg": 170.0,
    "contact_stiffness_n_per_m": 3.75e4,
    "angular_stiffness_n_per_deg": 7.5,
    "regrasp_budget_s": 1.13,
    "regrasp_duration_s": 1.06,
}

#: Substream-label documentation per task (ADR-002 P6; recorded in the
#: probe bundle's seed schedule like the admission cells do).
PROBE_SUBSTREAM_LABELS: dict[str, tuple[str, ...]] = {
    "cocarry": ("env.cocarry", "evaluation.bundle_bootstrap"),
    "stage1_pickplace_as": ("envs.stage1_pickplace", "evaluation.bundle_bootstrap"),
    "handover_place": ("env.handover_place", "evaluation.bundle_bootstrap"),
}


def _to_float(value: Any) -> float:  # noqa: ANN401 - torch/np scalar
    """Coerce a torch / numpy scalar-or-(1,) value to a Python float (env 0)."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return float(np.asarray(value).reshape(-1)[0])


def _action_statistics(actions: list[NDArray[np.floating]]) -> dict[str, float]:
    """Per-episode action-distribution summary for the fingerprint (ADR-009 as amended).

    Distribution-level only — mean absolute command, dispersion, and the
    p90 absolute command — deliberately too coarse to reconstruct the
    member's policy or exact parameters (ADR-018/I3: behavioural
    characterisation without policy access).
    """
    if not actions:
        return {}
    flat = np.concatenate([np.asarray(a, dtype=np.float64).reshape(-1) for a in actions])
    return {
        "action_abs_mean": round(float(np.mean(np.abs(flat))), 6),
        "action_std": round(float(np.std(flat)), 6),
        "action_abs_p90": round(float(np.percentile(np.abs(flat), 90)), 6),
    }


def member_material(
    member: PartnerMemberSpec, spec: PartnerSpec, *, redact: bool
) -> dict[str, object]:
    """Serialize a member for ``partners.json`` (ADR-028 §Decision 1; ADR-018 custody).

    Private members are **redacted**: ``extra`` (which carries the
    behavioural parameter values) is replaced by the withheld marker.
    The custody hash still verifies —
    :attr:`~chamber.partners.api.PartnerSpec.partner_id` deliberately
    excludes ``extra``; identity rides on the ``member://`` URI's
    committed digest (published hashes, withheld parameters).
    """
    extra: dict[str, str] = (
        {"withheld_params_sha256": member.params_sha256} if redact else dict(spec.extra)
    )
    return {
        "name": member.member_name,
        "class_name": spec.class_name,
        "seed": spec.seed,
        "checkpoint_step": spec.checkpoint_step,
        "weights_uri": spec.weights_uri,
        "extra": extra,
    }


def _ego_material(name: str, spec: PartnerSpec) -> dict[str, object]:
    return {
        "name": name,
        "class_name": spec.class_name,
        "seed": spec.seed,
        "checkpoint_step": spec.checkpoint_step,
        "weights_uri": spec.weights_uri,
        "extra": dict(spec.extra),
    }


def _run_cocarry_probe(
    set_spec: PartnerSetSpec,
    member: PartnerMemberSpec,
    params: Mapping[str, str],
    *,
    variant: ProbeVariant,
    render_backend: str | None,
) -> CellRun:
    """Co-carry probe: default-gain reference ego + the member's partner seat (ADR-026)."""
    if variant != "fingerprint":
        msg = f"co-carry probe supports only the fingerprint variant; got {variant!r}"
        raise ValueError(msg)
    from chamber.envs.cocarry import (
        COCARRY_DEFAULT_EPISODE_LENGTH,
        cocarry_matched_controller_specs,
        make_cocarry_env,
    )

    controller_specs = cocarry_matched_controller_specs()
    episodes_by_seed: dict[int, list[EpisodeResult]] = {}
    material: list[dict[str, object]] = []
    hashes: dict[str, str] = {}
    for cluster_seed in set_spec.probe_seeds:
        records: list[EpisodeResult] = []
        for episode in range(set_spec.probe_episodes_per_seed):
            iss = int(cluster_seed) * 1000 + episode
            env = make_cocarry_env(
                condition_id="cocarry_matched_panda_pair",
                episode_length=COCARRY_DEFAULT_EPISODE_LENGTH,
                root_seed=iss,
                render_backend=render_backend,
            )
            try:
                ego_uid = str(env.ego_uid)  # type: ignore[attr-defined]
                partner_uid = str(env.partner_uid)  # type: ignore[attr-defined]
                ego_spec = PartnerSpec(
                    "cocarry_impedance", 0, None, None, dict(controller_specs[ego_uid])
                )
                seat_extra = dict(controller_specs[partner_uid])
                if member.registry_class == "frozen_cocarry_joint":
                    # The jointly-trained seat assembles its symmetric
                    # full state from raw leaves and needs the opposite
                    # uid (ADR-011 as amended; mirrors cocarry_eval).
                    seat_extra["other_uid"] = ego_uid
                member_spec = member.partner_spec(params=params, seat_extra=seat_extra)
                ego = load_partner(ego_spec)
                partner = load_partner(member_spec)
                if not material:
                    material = [
                        _ego_material("ego:cocarry_impedance", ego_spec),
                        member_material(member, member_spec, redact=member.split == "private"),
                    ]
                    hashes = {
                        "ego:cocarry_impedance": ego_spec.partner_id,
                        member.member_name: member_spec.partner_id,
                    }
                obs, _ = env.reset(seed=iss)
                ego.reset(seed=iss)
                partner.reset(seed=iss)
                info: dict[str, Any] = {}
                actions: list[NDArray[np.floating]] = []
                for _ in range(COCARRY_DEFAULT_EPISODE_LENGTH):
                    member_action = partner.act(obs)
                    actions.append(member_action)
                    action = {ego_uid: ego.act(obs), partner_uid: member_action}
                    obs, _, terminated, truncated, info = env.step(action)
                    if _to_float(terminated) or _to_float(truncated):
                        break
                tel = env.get_telemetry()  # type: ignore[attr-defined]
                records.append(
                    EpisodeResult(
                        seed=int(cluster_seed),
                        episode_idx=episode,
                        initial_state_seed=iss,
                        success=bool(_to_float(info["success"])),
                        force_peak=_to_float(tel["max_stress_proxy"]),
                        metadata={
                            "member": member.member_name,
                            "probe_variant": variant,
                            "max_tilt_deg": _to_float(tel["max_tilt_deg"]),
                            **_action_statistics(actions),
                        },
                    )
                )
            finally:
                env.close()
        episodes_by_seed[int(cluster_seed)] = records
    return CellRun(
        episodes_by_seed=episodes_by_seed,
        partner_material=material,
        partner_hashes=hashes,
        substream_labels=list(PROBE_SUBSTREAM_LABELS["cocarry"]),
    )


def _run_pickplace_probe(
    set_spec: PartnerSetSpec,
    member: PartnerMemberSpec,
    params: Mapping[str, str],
    *,
    variant: ProbeVariant,
    render_backend: str | None,
) -> CellRun:
    """Pick-place probe: REF-SCRIPT oracle ego + the member's heuristic seat (ADR-011)."""
    del render_backend  # single-env CPU-side stepping; the env owns its P6 streams
    if variant != "fingerprint":
        msg = f"pick-place probe supports only the fingerprint variant; got {variant!r}"
        raise ValueError(msg)
    from chamber.agents.pickplace_ego_scripted import (
        PICKPLACE_EGO_UID,
        ScriptedPickPlaceEgo,
    )
    from chamber.envs.stage1_pickplace import make_stage1_pickplace_env

    condition_id = "stage1_pickplace_panda_plus_fetch_ego_aht_happo_per_agent"
    max_steps = 100
    episodes_by_seed: dict[int, list[EpisodeResult]] = {}
    material: list[dict[str, object]] = []
    hashes: dict[str, str] = {}
    for cluster_seed in set_spec.probe_seeds:
        env = make_stage1_pickplace_env(condition_id=condition_id, root_seed=int(cluster_seed))
        try:
            uids = list(env.action_space.spaces)  # type: ignore[attr-defined]
            ego_uid = PICKPLACE_EGO_UID
            (partner_uid,) = [u for u in uids if u != ego_uid]
            partner_dim = int(env.action_space.spaces[partner_uid].shape[0])  # type: ignore[attr-defined,index]
            ego = ScriptedPickPlaceEgo(mask_partner_obs=False, partner_uid=partner_uid)
            member_spec = member.partner_spec(
                params=params,
                seat_extra={"uid": partner_uid, "action_dim": str(partner_dim)},
            )
            partner = load_partner(member_spec)
            if not material:
                material = [member_material(member, member_spec, redact=member.split == "private")]
                hashes = {member.member_name: member_spec.partner_id}
            records: list[EpisodeResult] = []
            for episode in range(set_spec.probe_episodes_per_seed):
                obs, _ = env.reset(seed=episode)
                ego.reset(seed=episode)
                partner.reset(seed=episode)
                info: dict[str, Any] = {}
                actions: list[NDArray[np.floating]] = []
                for _ in range(max_steps):
                    member_action = partner.act(obs)
                    actions.append(member_action)
                    action = {ego_uid: ego.act(obs), partner_uid: member_action}
                    obs, _, terminated, truncated, info = env.step(action)
                    if _to_float(terminated) or _to_float(truncated):
                        break
                records.append(
                    EpisodeResult(
                        seed=int(cluster_seed),
                        episode_idx=episode,
                        initial_state_seed=episode,
                        success=bool(_to_float(info.get("success", False))),
                        metadata={
                            "member": member.member_name,
                            "probe_variant": variant,
                            **_action_statistics(actions),
                        },
                    )
                )
            episodes_by_seed[int(cluster_seed)] = records
        finally:
            env.close()
    return CellRun(
        episodes_by_seed=episodes_by_seed,
        partner_material=material,
        partner_hashes=hashes,
        substream_labels=list(PROBE_SUBSTREAM_LABELS["stage1_pickplace_as"]),
    )


def _run_handover_probe(
    set_spec: PartnerSetSpec,
    member: PartnerMemberSpec,
    params: Mapping[str, str],
    *,
    variant: ProbeVariant,
    render_backend: str | None,
) -> CellRun:
    """Handover probe: scripted corrector ego + the member's presenter seat (ADR-026).

    ``variant="fingerprint"`` probes at the canonical anchor cell (the
    member-to-member success spread there is the measured coupling);
    ``variant="free_regrasp"`` removes the re-grasp budget for the
    competence floor (only a wall scores low with budget pressure off).
    """
    del render_backend  # pure-Python kinematic env
    from chamber.agents.handover_ego_scripted import ScriptedHandoverEgo
    from chamber.envs.handover_place import make_handover_place_env

    p = HANDOVER_PROBE_ENV_PARAMS
    env = make_handover_place_env(
        free_regrasp=variant == "free_regrasp",
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
    ego = ScriptedHandoverEgo(
        translation_range_m=env.translation_range_m,
        wrist_correction_deg=env.wrist_correction_deg,
    )
    presenter_variant = (
        "matched" if float(params.get("grasp_pose_bias_deg", "0")) == 0.0 else "mismatched"
    )
    member_spec = member.partner_spec(params=params, seat_extra={"variant": presenter_variant})
    presenter = load_partner(member_spec)
    material = [member_material(member, member_spec, redact=member.split == "private")]
    hashes = {member.member_name: member_spec.partner_id}
    episodes_by_seed: dict[int, list[EpisodeResult]] = {}
    for cluster_seed in set_spec.probe_seeds:
        records: list[EpisodeResult] = []
        for episode in range(set_spec.probe_episodes_per_seed):
            iss = int(cluster_seed) * 1000 + episode  # the Gate-0 pairing convention
            obs, _ = env.reset(seed=iss)
            presenter.reset(seed=iss)
            ego.reset(seed=iss)
            presentation = presenter.act(obs)
            obs, _, _, _, _ = env.step(presentation)
            _, _, _, _, info = env.step(ego.act(obs))
            records.append(
                EpisodeResult(
                    seed=int(cluster_seed),
                    episode_idx=episode,
                    initial_state_seed=iss,
                    success=bool(info["success"]),
                    force_peak=float(info["seating_force_proxy_n"]),
                    metadata={
                        "member": member.member_name,
                        "probe_variant": variant,
                        "residual_angular_deg": float(info["residual_angular_deg"]),
                        "failure_mode": str(info["failure_mode"]),
                        **_action_statistics([presentation]),
                    },
                )
            )
        episodes_by_seed[int(cluster_seed)] = records
    return CellRun(
        episodes_by_seed=episodes_by_seed,
        partner_material=material,
        partner_hashes=hashes,
        substream_labels=list(PROBE_SUBSTREAM_LABELS["handover_place"]),
    )


#: Probe-runner registry, keyed by ADR-027 ``task_id`` (ADR-009 §Decision style).
PROBE_RUNNERS: dict[str, Callable[..., CellRun]] = {
    "cocarry": _run_cocarry_probe,
    "stage1_pickplace_as": _run_pickplace_probe,
    "handover_place": _run_handover_probe,
}


def run_member_probe(
    set_spec: PartnerSetSpec,
    member: PartnerMemberSpec,
    params: Mapping[str, str],
    *,
    variant: ProbeVariant = "fingerprint",
    render_backend: str | None = None,
) -> CellRun:
    """Run one member through the set's committed probe suite (ADR-009 as amended).

    Args:
        set_spec: The registered set (probe schedule + task binding).
        member: The member under probe.
        params: Digest-verified exact parameters
            (:func:`chamber.partners.sets.resolve_member_params`).
        variant: ``"fingerprint"`` or the handover ``"free_regrasp"``
            floor variant.
        render_backend: Forwarded to SAPIEN env factories (``"none"``
            on headless hosts).

    Returns:
        The raw probe records + partner custody material, ADR-028
        bundle-ready.

    Raises:
        KeyError: No probe runner for the set's task (lists known keys).
    """
    runner = PROBE_RUNNERS.get(set_spec.task_id)
    if runner is None:
        known = ", ".join(sorted(PROBE_RUNNERS)) or "<none>"
        msg = f"no partner-probe runner for task {set_spec.task_id!r}; known: {known}"
        raise KeyError(msg)
    return runner(set_spec, member, params, variant=variant, render_backend=render_backend)


def fingerprint_statistics(episodes: list[EpisodeResult]) -> dict[str, float]:
    """Aggregate a member's behavioural fingerprint (ADR-009 as amended; ADR-018/I3).

    Success rate, the dyad's stress-channel distribution (mean/p50/p90/
    max of ``force_peak``, the task's canonical instrument — ADR-027
    §Versioning), and the action-distribution summary averaged over the
    probe episodes. Purpose: a reader can tell members apart without
    policy access; parameters stay withheld for private members.
    """
    stats: dict[str, float] = {
        "n_episodes": float(len(episodes)),
        "success_rate": (
            float(np.mean([1.0 if ep.success else 0.0 for ep in episodes])) if episodes else 0.0
        ),
    }
    forces = [float(ep.force_peak) for ep in episodes if ep.force_peak is not None]
    if forces:
        arr = np.asarray(forces, dtype=np.float64)
        stats.update(
            {
                "stress_mean": round(float(np.mean(arr)), 4),
                "stress_p50": round(float(np.percentile(arr, 50)), 4),
                "stress_p90": round(float(np.percentile(arr, 90)), 4),
                "stress_max": round(float(np.max(arr)), 4),
            }
        )
    for key in ("action_abs_mean", "action_std", "action_abs_p90"):
        values = [
            float(ep.metadata[key])
            for ep in episodes
            if isinstance(ep.metadata.get(key), (int, float))
        ]
        if values:
            stats[key] = round(float(np.mean(values)), 6)
    return stats


__all__ = [
    "HANDOVER_PROBE_ENV_PARAMS",
    "PROBE_RUNNERS",
    "PROBE_SUBSTREAM_LABELS",
    "REFERENCE_EGO_IDS",
    "ProbeVariant",
    "fingerprint_statistics",
    "member_material",
    "run_member_probe",
]
