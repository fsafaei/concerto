# SPDX-License-Identifier: Apache-2.0
r"""Co-carry episode driver for ``chamber-eval run`` (ADR-011 as amended; ADR-028 §Decision 3).

Makes ``cocarry`` runnable by the bundle runner
(:mod:`chamber.benchmarks.bundle_runner`) so every CB-06 leaderboard row
is a ``chamber-eval verify``-passing v3 bundle. One module owns the
row-to-policy mapping (the ADR-011 v1.0 baseline set):

===================================  =============================================
``--policy`` id                      Row / ego seat
===================================  =============================================
``random``                           B-RND — the uniform floor.
``static``                           B-STAT — the stationary (zero-action) seat.
``ref_script_cocarry_impedance``     REF-SCRIPT — the matched impedance oracle
                                     (reported *oracle-reference*, never a
                                     baseline).
``happo:<local://uri>``              B-AHT — a frozen HAPPO ego checkpoint.
``happo_blind:<local://uri>``        B-BLIND — a frozen checkpoint evaluated on
                                     the SAME masked interface it trained on
                                     (:mod:`chamber.envs.cocarry_blind_mask`).
``joint_ego:<local://uri>``          B-JOINT ego side of a pair checkpoint
                                     (evaluated as the pair it trained as — pair
                                     the partner seat with
                                     ``--partner frozen_cocarry_joint
                                     --partner-weights <same uri>``).
===================================  =============================================

Rig conventions are the measured ones from
:func:`chamber.benchmarks.admission_cells.run_cocarry_cell`: **one fresh
env per episode** (``initial_state_seed = seed * 1000 + episode`` — the
Gate-0 pairing convention; re-using a stepped co-carry env across resets
is a measured rig artifact), matched initial-state draws across partner
cells, ``force_peak`` = the canonical wrist-stress instrument's episode
maximum (ADR-027 §Versioning), tilt + centroid telemetry in metadata.

Learned egos read the synthesised 46-D ``state``
(:mod:`chamber.envs.cocarry_obs`), so their episodes run inside
:func:`chamber.envs.cocarry_obs.make_cocarry_training_env`'s wrapper
chain (+ the blind mask for ``happo_blind:``); scripted/trivial egos
run on the raw env, byte-matching the admission cells. SAPIEN imports
stay inside function bodies (P2 wrapper-only; ADR-001 §Risks).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from chamber.evaluation.results import EpisodeResult
from chamber.partners.api import PartnerSpec
from chamber.partners.registry import load_partner

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping

    import gymnasium as gym
    from numpy.typing import NDArray

    from chamber.partners.api import FrozenPartner
    from chamber.partners.sets import PartnerMemberSpec, PartnerSetSpec

# Imports for the @register_partner side effects so every baseline seat
# resolves when this module is the entry point (ADR-009 registry style).
import chamber.partners.ablation
import chamber.partners.cocarry_impedance
import chamber.partners.frozen_cocarry_joint
import chamber.partners.frozen_harl  # noqa: F401

#: Substream labels recorded in co-carry bundle seed schedules (ADR-002 P6).
COCARRY_EVAL_SUBSTREAM_LABELS: tuple[str, ...] = ("env.cocarry", "evaluation.bundle_bootstrap")

#: Exact (non-prefixed) co-carry policy ids (ADR-011 §Decision as amended).
COCARRY_POLICY_IDS: tuple[str, ...] = ("random", "static", "ref_script_cocarry_impedance")

#: Checkpoint-loading policy-id prefixes (``<prefix>:<local://uri>``).
COCARRY_POLICY_PREFIXES: tuple[str, ...] = ("happo", "happo_blind", "joint_ego")

#: Per-seed checkpoint-manifest prefixes (``<prefix>:<manifest.json>``).
#: The manifest maps each cluster seed to that training seed's SELECTED
#: checkpoint URI (``{"0": "local://artifacts/...", ...}`` — the
#: ADR-027 checkpoint-selection outputs), so one learned row ships as
#: ONE bundle whose per-seed cells each drive their own seed's policy.
COCARRY_MANIFEST_PREFIXES: dict[str, str] = {
    "happo_manifest": "happo",
    "happo_blind_manifest": "happo_blind",
    "joint_manifest": "joint_ego",
}


def _to_float(value: Any) -> float:  # noqa: ANN401 - torch/np scalar
    """Coerce a torch / numpy scalar-or-(1,) value to a Python float (env 0)."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return float(np.asarray(value).reshape(-1)[0])


def parse_cocarry_policy(policy_id: str) -> tuple[str, str | None]:
    """Split a co-carry ``--policy`` id into ``(kind, checkpoint_uri | None)``.

    ADR-011 §Decision as amended (the row table in the module
    docstring). Loud ``KeyError`` listing the known forms on an unknown
    id (the ADR-009 registry error style).
    """
    if policy_id in COCARRY_POLICY_IDS:
        return policy_id, None
    kind, sep, uri = policy_id.partition(":")
    if sep and uri and (kind in COCARRY_POLICY_PREFIXES or kind in COCARRY_MANIFEST_PREFIXES):
        return kind, uri
    known = ", ".join(
        [
            *COCARRY_POLICY_IDS,
            *(f"{p}:<uri>" for p in COCARRY_POLICY_PREFIXES),
            *(f"{p}:<manifest.json>" for p in COCARRY_MANIFEST_PREFIXES),
        ]
    )
    msg = f"unknown co-carry policy id {policy_id!r}; known: {known}"
    raise KeyError(msg)


def resolve_seed_policy(policy_id: str, seed: int) -> str:
    """Resolve the effective per-seed policy id (ADR-027 §Reporting rules).

    Non-manifest ids pass through unchanged. A
    ``<prefix>_manifest:<manifest.json>`` id loads the JSON mapping of
    cluster seed → selected checkpoint URI (the per-seed ADR-027
    checkpoint-selection outputs) and returns
    ``"<base>:<manifest[seed]>"``. Loud ``KeyError`` when the manifest
    lacks the seed — a missing selection is a campaign bug, never a
    silently skipped cell.

    Raises:
        KeyError: Unknown policy id, or seed absent from the manifest.
        ValueError: Manifest file is not a string-to-string mapping.
    """
    kind, ref = parse_cocarry_policy(policy_id)
    base = COCARRY_MANIFEST_PREFIXES.get(kind)
    if base is None or ref is None:
        return policy_id
    import json
    from pathlib import Path

    payload = json.loads(Path(ref).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in payload.items()
    ):
        msg = f"{ref}: a checkpoint manifest must map seed strings to checkpoint URIs"
        raise ValueError(msg)
    uri = payload.get(str(seed))
    if uri is None:
        msg = (
            f"{ref}: no selected checkpoint for seed {seed} "
            f"(manifest seeds: {sorted(payload)}); every campaign seed must "
            "have a committed ADR-027 selection before eval cells run"
        )
        raise KeyError(msg)
    return f"{base}:{uri}"


class _ZeroEgo:
    """B-STAT: the stationary zero-action ego seat (ADR-011 §Decision as amended)."""

    def __init__(self, *, action_dim: int) -> None:
        self._action_dim = action_dim

    def reset(self, *, seed: int | None = None) -> None:
        """No per-episode state — deterministic by construction (ADR-011 B-STAT)."""
        del seed

    def act(self, obs: Mapping[str, object], *, deterministic: bool = True) -> NDArray[np.float32]:
        """The zero action, regardless of observation (ADR-011 B-STAT)."""
        del obs, deterministic
        return np.zeros(self._action_dim, dtype=np.float32)


class _RandomEgo:
    """B-RND: uniform ``[-1, 1]^action_dim`` ego seat (ADR-011; ADR-002 P6).

    The rng is re-derived per episode from the
    ``benchmarks.cocarry_eval.ego_random.{iss}`` substream so every
    episode is independently re-derivable from the bundle's seed
    schedule.
    """

    _SUBSTREAM: str = "benchmarks.cocarry_eval.ego_random.{iss}"

    def __init__(self, *, action_dim: int, root_seed: int) -> None:
        from concerto.training.seeding import derive_substream

        self._derive = derive_substream
        self._action_dim = action_dim
        self._root_seed = root_seed
        self._rng = derive_substream(
            self._SUBSTREAM.format(iss=0), root_seed=root_seed
        ).default_rng()

    def reset(self, *, seed: int | None = None) -> None:
        """Re-derive the per-episode substream (ADR-002 P6; ADR-011 B-RND)."""
        self._rng = self._derive(
            self._SUBSTREAM.format(iss=int(seed or 0)), root_seed=self._root_seed
        ).default_rng()

    def act(self, obs: Mapping[str, object], *, deterministic: bool = True) -> NDArray[np.float32]:
        """Uniform action on ``[-1, 1]^action_dim`` (ADR-011 B-RND)."""
        del obs, deterministic
        return self._rng.uniform(-1.0, 1.0, size=self._action_dim).astype(np.float32)


def build_cocarry_ego(
    policy_id: str,
    *,
    ego_uid: str,
    partner_uid: str,
    ego_controller_extra: Mapping[str, str],
    action_dim: int,
    root_seed: int,
) -> tuple[FrozenPartner | _ZeroEgo | _RandomEgo, PartnerSpec | None]:
    """Construct the ego seat for a co-carry row (ADR-011 §Decision as amended).

    Args:
        policy_id: A row id per :func:`parse_cocarry_policy`.
        ego_uid: The env-side ego seat uid.
        partner_uid: The opposite seat (needed by the joint ego's
            symmetric state assembly).
        ego_controller_extra: The env-derived matched controller
            geometry for the ego seat
            (:func:`chamber.envs.cocarry.cocarry_matched_controller_specs`).
        action_dim: Ego action width (8 for the Panda seat).
        root_seed: Run-level root seed (B-RND's substream root).

    Returns:
        ``(ego, ego_spec)`` — ``ego_spec`` is the identity-bearing
        :class:`PartnerSpec` for bundle custody when the ego is a
        registry policy, ``None`` for the trivial B-RND/B-STAT seats.

    Raises:
        KeyError: Unknown policy id (lists the known forms).
    """
    kind, uri = parse_cocarry_policy(policy_id)
    if kind == "random":
        return _RandomEgo(action_dim=action_dim, root_seed=root_seed), None
    if kind == "static":
        return _ZeroEgo(action_dim=action_dim), None
    if kind == "ref_script_cocarry_impedance":
        spec = PartnerSpec("cocarry_impedance", 0, None, None, dict(ego_controller_extra))
        return load_partner(spec), spec
    if kind in ("happo", "happo_blind"):
        spec = PartnerSpec("frozen_harl", 0, None, uri, {"uid": ego_uid})
        return load_partner(spec), spec
    # kind == "joint_ego" (parse_cocarry_policy already rejected the rest)
    spec = PartnerSpec(
        "frozen_cocarry_joint",
        0,
        None,
        uri,
        {"uid": ego_uid, "other_uid": partner_uid, "actor_key": "actor_ego"},
    )
    return load_partner(spec), spec


def _build_episode_env(
    *,
    policy_kind: str,
    condition_id: str,
    episode_length: int,
    iss: int,
    render_backend: str | None,
) -> gym.Env[Any, Any]:
    """One fresh env per episode, wrapped per the ego's obs interface (ADR-026; ADR-011).

    Learned HAPPO egos need the synthesised ego ``state``
    (:func:`chamber.envs.cocarry_obs.make_cocarry_training_env`);
    ``happo_blind`` additionally gets the B-BLIND mask — evaluation on
    the trained interface, never the unmasked one. Scripted/trivial
    egos and the joint pair (raw-leaf readers) run on the raw env,
    byte-matching the admission cells.
    """
    if policy_kind in ("happo", "happo_blind"):
        from chamber.envs.cocarry_obs import make_cocarry_training_env

        env = make_cocarry_training_env(
            condition_id=condition_id,
            episode_length=episode_length,
            root_seed=iss,
            render_backend=render_backend,
        )
        if policy_kind == "happo_blind":
            from chamber.envs.cocarry_blind_mask import CoCarryEgoBlindMask

            env = CoCarryEgoBlindMask(env)
        return env
    from chamber.envs.cocarry import make_cocarry_env

    return make_cocarry_env(
        condition_id=condition_id,
        episode_length=episode_length,
        root_seed=iss,
        render_backend=render_backend,
    )


def _run_episode(
    *,
    env: gym.Env[Any, Any],
    ego: Any,  # noqa: ANN401 - FrozenPartner | trivial seat (shared reset/act surface)
    partner: FrozenPartner,
    ego_uid: str,
    partner_uid: str,
    episode_length: int,
    iss: int,
) -> tuple[bool, dict[str, float]]:
    """Roll one episode; return ``(success, telemetry)`` (ADR-026 §Decision 1-2)."""
    obs, _ = env.reset(seed=iss)
    ego.reset(seed=iss)
    partner.reset(seed=iss)
    info: dict[str, Any] = {}
    n_steps = 0
    for _ in range(episode_length):
        action = {ego_uid: ego.act(obs), partner_uid: partner.act(obs)}
        obs, _, terminated, truncated, info = env.step(action)
        n_steps += 1
        if _to_float(terminated) or _to_float(truncated):
            break
    tel = env.get_telemetry()  # type: ignore[attr-defined]
    telemetry = {
        "force_peak": _to_float(tel["max_stress_proxy"]),
        "max_tilt_deg": _to_float(tel["max_tilt_deg"]),
        "centroid_to_goal": _to_float(tel["centroid_to_goal"]),
        # Time-to-success in env ticks (the campaign's secondary metric;
        # equals the truncation horizon on non-terminating episodes).
        "n_steps": float(n_steps),
    }
    return bool(_to_float(info["success"])), telemetry


def run_cocarry_episodes_for_set(
    *,
    policy_id: str,
    set_spec: PartnerSetSpec,
    members: list[tuple[PartnerMemberSpec, dict[str, str]]],
    seeds: list[int],
    episodes_per_seed: int,
    root_seed: int = 0,
    condition_id: str = "cocarry_matched_panda_pair",
    render_backend: str | None = None,
) -> tuple[dict[int, list[EpisodeResult]], list[dict[str, object]], dict[str, str]]:
    """The CB-06 leaderboard grid: one row policy x a partner set (ADR-011; ADR-028 §Decision 3).

    Return contract mirrors
    :func:`chamber.benchmarks.bundle_runner.run_task_episodes_for_set`:
    per-seed records with member-unique ``episode_idx``
    (``member_index * episodes_per_seed + episode``), matched
    initial-state draws across members (every member sees the same
    ``iss = seed * 1000 + episode`` grid — the paired-comparison
    precondition), redacted partner material + custody hashes.

    Raises:
        KeyError: Unknown policy id.
        ValueError: Empty ``seeds`` / ``members``.
    """
    from chamber.benchmarks.partner_probe import member_material
    from chamber.envs.cocarry import (
        COCARRY_DEFAULT_EPISODE_LENGTH,
        cocarry_matched_controller_specs,
    )

    if not seeds or not members:
        msg = "seeds and members must be non-empty"
        raise ValueError(msg)
    parse_cocarry_policy(policy_id)  # loud-fail on an unknown id before any env builds
    controller_specs = cocarry_matched_controller_specs()
    episode_length = COCARRY_DEFAULT_EPISODE_LENGTH

    episodes_by_seed: dict[int, list[EpisodeResult]] = {}
    material: list[dict[str, object]] = []
    hashes: dict[str, str] = {}
    for cluster_seed in seeds:
        # Per-seed policy resolution (ADR-027 checkpoint selection): a
        # manifest id maps each cluster seed to that training seed's
        # selected checkpoint, so one row = one bundle with per-seed cells.
        seed_policy = resolve_seed_policy(policy_id, int(cluster_seed))
        seed_kind, _ = parse_cocarry_policy(seed_policy)
        records: list[EpisodeResult] = []
        for member_index, (member, params) in enumerate(members):
            for episode in range(episodes_per_seed):
                unique_idx = member_index * episodes_per_seed + episode
                iss = int(cluster_seed) * 1000 + episode
                env = _build_episode_env(
                    policy_kind=seed_kind,
                    condition_id=condition_id,
                    episode_length=episode_length,
                    iss=iss,
                    render_backend=render_backend,
                )
                try:
                    ego_uid = str(env.get_wrapper_attr("ego_uid"))
                    partner_uid = str(env.get_wrapper_attr("partner_uid"))
                    action_dim = int(env.action_space.spaces[ego_uid].shape[0])  # type: ignore[attr-defined,index]
                    ego, ego_spec = build_cocarry_ego(
                        seed_policy,
                        ego_uid=ego_uid,
                        partner_uid=partner_uid,
                        ego_controller_extra=controller_specs[ego_uid],
                        action_dim=action_dim,
                        root_seed=root_seed,
                    )
                    seat_extra = dict(controller_specs[partner_uid])
                    if member.registry_class == "frozen_cocarry_joint":
                        seat_extra["other_uid"] = ego_uid
                    member_spec = member.partner_spec(params=params, seat_extra=seat_extra)
                    partner = load_partner(member_spec)
                    ego_key = f"ego:{seed_policy}"
                    if ego_spec is not None and ego_key not in hashes:
                        material.append(_ego_material(ego_key, ego_spec))
                        hashes[ego_key] = ego_spec.partner_id
                    if member.member_name not in hashes:
                        material.append(
                            member_material(member, member_spec, redact=member.split == "private")
                        )
                        hashes[member.member_name] = member_spec.partner_id
                    success, telemetry = _run_episode(
                        env=env,
                        ego=ego,
                        partner=partner,
                        ego_uid=ego_uid,
                        partner_uid=partner_uid,
                        episode_length=episode_length,
                        iss=iss,
                    )
                    metadata: dict[str, Any] = {
                        "condition": condition_id,
                        "policy": policy_id,
                        "member": member.member_name,
                        "partner_set": set_spec.slug,
                        "max_tilt_deg": telemetry["max_tilt_deg"],
                        "centroid_to_goal": telemetry["centroid_to_goal"],
                        "n_steps": telemetry["n_steps"],
                    }
                    if seed_policy != policy_id:
                        metadata["seed_policy"] = seed_policy
                    records.append(
                        EpisodeResult(
                            seed=int(cluster_seed),
                            episode_idx=unique_idx,
                            initial_state_seed=iss,
                            success=success,
                            force_peak=telemetry["force_peak"],
                            metadata=metadata,
                        )
                    )
                finally:
                    env.close()
        episodes_by_seed[int(cluster_seed)] = records
    return episodes_by_seed, material, hashes


def run_cocarry_episodes_adhoc(
    *,
    policy_id: str,
    partner_name: str,
    partner_weights: str | None = None,
    seeds: list[int],
    episodes_per_seed: int,
    root_seed: int = 0,
    condition_id: str = "cocarry_matched_panda_pair",
    render_backend: str | None = None,
) -> tuple[dict[int, list[EpisodeResult]], PartnerSpec]:
    """Ad-hoc single-partner co-carry grid (ADR-028 §Decision 3) — the B-JOINT pair path.

    The B-JOINT row evaluates the pair it trained as (ADR-011 §Decision
    as amended): ``policy_id="joint_ego:<uri>"`` with
    ``partner_name="frozen_cocarry_joint"`` and ``partner_weights`` the
    same pair-checkpoint URI puts the trained partner-side actor on the
    partner seat. Scripted partner classes get the env's matched
    controller geometry as their seat extra.

    Return contract mirrors
    :func:`chamber.benchmarks.bundle_runner.run_task_episodes`.

    Raises:
        KeyError: Unknown policy id / partner class.
        ValueError: Empty ``seeds``.
    """
    from chamber.envs.cocarry import (
        COCARRY_DEFAULT_EPISODE_LENGTH,
        cocarry_matched_controller_specs,
    )

    if not seeds:
        msg = "seeds must be non-empty"
        raise ValueError(msg)
    policy_kind, _ = parse_cocarry_policy(policy_id)
    if policy_kind in COCARRY_MANIFEST_PREFIXES:
        msg = (
            f"policy {policy_id!r}: per-seed checkpoint manifests are not "
            "supported on the ad-hoc single-partner path — the v3 bundle "
            "carries one partner identity, and a per-seed pair means one "
            "identity per seed. Run the B-JOINT row as one bundle per seed "
            "(--seeds <s> --policy joint_ego:<that seed's pair URI>)."
        )
        raise ValueError(msg)
    controller_specs = cocarry_matched_controller_specs()
    episode_length = COCARRY_DEFAULT_EPISODE_LENGTH

    episodes_by_seed: dict[int, list[EpisodeResult]] = {}
    partner_spec: PartnerSpec | None = None
    for cluster_seed in seeds:
        records: list[EpisodeResult] = []
        for episode in range(episodes_per_seed):
            iss = int(cluster_seed) * 1000 + episode
            env = _build_episode_env(
                policy_kind=policy_kind,
                condition_id=condition_id,
                episode_length=episode_length,
                iss=iss,
                render_backend=render_backend,
            )
            try:
                ego_uid = str(env.get_wrapper_attr("ego_uid"))
                partner_uid = str(env.get_wrapper_attr("partner_uid"))
                action_dim = int(env.action_space.spaces[ego_uid].shape[0])  # type: ignore[attr-defined,index]
                ego, _ = build_cocarry_ego(
                    policy_id,
                    ego_uid=ego_uid,
                    partner_uid=partner_uid,
                    ego_controller_extra=controller_specs[ego_uid],
                    action_dim=action_dim,
                    root_seed=root_seed,
                )
                if partner_spec is None:
                    partner_dim = int(env.action_space.spaces[partner_uid].shape[0])  # type: ignore[attr-defined,index]
                    extra = dict(controller_specs[partner_uid])
                    extra["action_dim"] = str(partner_dim)
                    if partner_name == "frozen_cocarry_joint":
                        extra["other_uid"] = ego_uid
                    partner_spec = PartnerSpec(partner_name, 0, None, partner_weights, extra)
                partner = load_partner(partner_spec)
                success, telemetry = _run_episode(
                    env=env,
                    ego=ego,
                    partner=partner,
                    ego_uid=ego_uid,
                    partner_uid=partner_uid,
                    episode_length=episode_length,
                    iss=iss,
                )
                records.append(
                    EpisodeResult(
                        seed=int(cluster_seed),
                        episode_idx=episode,
                        initial_state_seed=iss,
                        success=success,
                        force_peak=telemetry["force_peak"],
                        metadata={
                            "condition": condition_id,
                            "policy": policy_id,
                            "partner": partner_name,
                            "max_tilt_deg": telemetry["max_tilt_deg"],
                            "centroid_to_goal": telemetry["centroid_to_goal"],
                            "n_steps": telemetry["n_steps"],
                        },
                    )
                )
            finally:
                env.close()
        episodes_by_seed[int(cluster_seed)] = records
    if partner_spec is None:  # pragma: no cover - seeds non-empty is checked above
        msg = "no partner spec was built"
        raise ValueError(msg)
    return episodes_by_seed, partner_spec


def _ego_material(name: str, spec: PartnerSpec) -> dict[str, object]:
    """Serialize the ego seat for ``partners.json`` (ADR-028 §Decision 1 custody)."""
    return {
        "name": name,
        "class_name": spec.class_name,
        "seed": spec.seed,
        "checkpoint_step": spec.checkpoint_step,
        "weights_uri": spec.weights_uri,
        "extra": dict(spec.extra),
    }


__all__ = [
    "COCARRY_EVAL_SUBSTREAM_LABELS",
    "COCARRY_MANIFEST_PREFIXES",
    "COCARRY_POLICY_IDS",
    "COCARRY_POLICY_PREFIXES",
    "build_cocarry_ego",
    "parse_cocarry_policy",
    "resolve_seed_policy",
    "run_cocarry_episodes_adhoc",
    "run_cocarry_episodes_for_set",
]
