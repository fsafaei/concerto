# SPDX-License-Identifier: Apache-2.0
"""Episode runner behind ``chamber-eval run`` (ADR-028 §Decision 3).

Resolves a task from the ADR-027 registry (:mod:`chamber.tasks`),
drives it with a named ego policy against a registry partner, and
returns the raw :class:`chamber.evaluation.results.EpisodeResult`
records a v3 bundle is written from. Pure delegation: the runner adds
no env behaviour (P2 wrapper-only, ADR-001 §Decision) and routes every
random draw through :func:`concerto.training.seeding.derive_substream`
(ADR-002 P6 — a bundle is byte-reproducible from its seed schedule).

Scope (v1): CPU-only tasks with the dict-action Gymnasium surface and
a per-task success predicate registered in
:data:`SUCCESS_PREDICATES` — today ``mpe_cooperative_push``, the
Tier-0 rig diagnostic the ADR-028 smoke evaluation pins. SAPIEN-tier
tasks join as their admission campaigns land (ADR-027 §Admission
protocol); an unsupported task loud-fails at dispatch, never
half-runs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

import chamber.tasks
from chamber.evaluation.results import EpisodeResult
from chamber.partners.api import PartnerSpec
from chamber.partners.registry import load_partner
from concerto.training.seeding import derive_substream

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

#: Ego policy ids ``chamber-eval run --policy`` accepts (ADR-011 as
#: amended: ``random`` is the B-RND floor baseline).
POLICY_IDS: tuple[str, ...] = ("random",)

#: Substream label pattern for the random ego (ADR-002 P6). The
#: ``{seed}.{episode}`` leaf scopes each episode's stream.
EGO_SUBSTREAM_PATTERN: str = "benchmarks.bundle_runner.ego_random.{seed}.{episode}"

#: Tier-0 diagnostic success predicate threshold for
#: ``mpe_cooperative_push``: final-step shared coverage reward at or
#: above this value. Calibrated so the B-RND floor lands mid-band
#: (~0.4 success over seeds 0-3) — a *rig diagnostic* bar that gives
#: the summary statistics variance to verify against, NOT a science
#: bar (the task is Tier 0 and carries no cooperation claim,
#: ADR-027 §Tier ladder).
MPE_DIAGNOSTIC_SUCCESS_THRESHOLD: float = -1.2


def _mpe_success(final_reward: float) -> bool:
    return final_reward >= MPE_DIAGNOSTIC_SUCCESS_THRESHOLD


#: Per-task success predicates over the final-step shared reward
#: (ADR-028 §Decision 3). A task absent from this table is not
#: runnable by the bundle runner yet.
SUCCESS_PREDICATES: dict[str, Callable[[float], bool]] = {
    "mpe_cooperative_push": _mpe_success,
}


class EgoPolicy(Protocol):
    """Minimal ego surface the runner drives (ADR-011 baseline contract)."""

    def reset(self, *, seed: int, episode: int) -> None:
        """Re-derive the per-episode stream (ADR-002 P6)."""
        ...

    def act(self, obs: Any) -> NDArray[np.float32]:  # noqa: ANN401 - env-specific nested obs dict
        """Return the ego action for the current observation (ADR-011 baseline contract)."""
        ...


class RandomEgoPolicy:
    """The B-RND floor baseline (ADR-011 §Decision as amended).

    Uniform actions on ``[-1, 1]^action_dim``; rng derived per
    ``(seed, episode)`` from :data:`EGO_SUBSTREAM_PATTERN` so every
    episode is independently re-derivable (ADR-002 P6).
    """

    def __init__(self, *, action_dim: int, root_seed: int) -> None:
        """Bind the action dimensionality and run-level root seed (ADR-002 P6)."""
        self._action_dim = action_dim
        self._root_seed = root_seed
        self._rng = derive_substream(
            EGO_SUBSTREAM_PATTERN.format(seed=0, episode=0), root_seed=root_seed
        ).default_rng()

    def reset(self, *, seed: int, episode: int) -> None:
        """Re-derive the per-episode substream (ADR-002 P6)."""
        label = EGO_SUBSTREAM_PATTERN.format(seed=seed, episode=episode)
        self._rng = derive_substream(label, root_seed=self._root_seed).default_rng()

    def act(self, obs: Any) -> NDArray[np.float32]:  # noqa: ANN401, ARG002 — env-specific obs; the floor baseline ignores it by definition
        """Uniform action on ``[-1, 1]^action_dim`` (ADR-011 B-RND)."""
        return self._rng.uniform(-1.0, 1.0, size=self._action_dim).astype(np.float32)


def build_ego_policy(policy_id: str, *, action_dim: int, root_seed: int) -> EgoPolicy:
    """Construct the named ego policy (ADR-011 §Decision as amended).

    Raises ``KeyError`` listing the known ids on an unknown
    ``policy_id`` (the ADR-009 registry error style).
    """
    if policy_id == "random":
        return RandomEgoPolicy(action_dim=action_dim, root_seed=root_seed)
    known = ", ".join(POLICY_IDS)
    msg = f"unknown policy id {policy_id!r}; known policy ids: {known}"
    raise KeyError(msg)


def build_partner_spec(partner_name: str, *, partner_uid: str, action_dim: int) -> PartnerSpec:
    """Build the ad-hoc single-partner spec for a bundle run (ADR-009 §Decision).

    The spec's identity hash (``PartnerSpec.partner_id``) is what the
    bundle records per member (ADR-028 §Decision 1; ADR-018 custody).
    ``seed=0`` pins one spec (and one hash) per run; per-episode
    behaviour is reseeded through ``partner.reset(seed=...)``.
    """
    return PartnerSpec(
        class_name=partner_name,
        seed=0,
        checkpoint_step=None,
        weights_uri=None,
        extra={"uid": partner_uid, "target_xy": "0.0,0.0", "action_dim": str(action_dim)},
    )


def run_task_episodes(
    *,
    task_id: str,
    task_version: int | None = None,
    policy_id: str,
    partner_name: str,
    seeds: list[int],
    episodes_per_seed: int,
    root_seed: int = 0,
) -> tuple[dict[int, list[EpisodeResult]], PartnerSpec]:
    """Run the task's episode grid and return raw records (ADR-028 §Decision 3).

    One env per cluster seed (``chamber.tasks.make(task_id,
    root_seed=seed)``), ``episodes_per_seed`` episodes each;
    ``EpisodeResult.initial_state_seed`` carries the episode index the
    env folds into its own substream. Success comes from the task's
    :data:`SUCCESS_PREDICATES` entry over the final-step shared
    reward; the raw final reward is kept in ``metadata`` so the
    predicate is re-auditable from the bundle alone.

    Raises:
        KeyError: Unknown task (from :func:`chamber.tasks.get`) or
            unknown policy id.
        NotImplementedError: Task registered but not runnable by the
            bundle runner (no success predicate / no env factory).
    """
    spec = chamber.tasks.get(task_id, version=task_version)
    if spec.task_id not in SUCCESS_PREDICATES:
        supported = ", ".join(sorted(SUCCESS_PREDICATES))
        msg = (
            f"task {spec.slug} is not runnable by the bundle runner yet "
            f"(no success predicate); supported tasks: {supported}"
        )
        raise NotImplementedError(msg)
    success_fn = SUCCESS_PREDICATES[spec.task_id]

    episodes_by_seed: dict[int, list[EpisodeResult]] = {}
    partner_spec: PartnerSpec | None = None
    for seed in seeds:
        env = chamber.tasks.make(task_id, version=task_version, root_seed=seed)
        uids = list(env.action_space.spaces)
        ego_uid, partner_uid = uids[0], uids[1]
        action_dim = int(env.action_space[ego_uid].shape[0])
        if partner_spec is None:
            partner_spec = build_partner_spec(
                partner_name, partner_uid=partner_uid, action_dim=action_dim
            )
        partner = load_partner(partner_spec)
        ego = build_ego_policy(policy_id, action_dim=action_dim, root_seed=root_seed)

        records: list[EpisodeResult] = []
        for episode in range(episodes_per_seed):
            obs, _ = env.reset(seed=episode)
            partner.reset(seed=episode)
            ego.reset(seed=seed, episode=episode)
            reward = 0.0
            done = False
            while not done:
                action = {ego_uid: ego.act(obs), partner_uid: partner.act(obs)}
                obs, reward, terminated, truncated, _ = env.step(action)
                done = bool(terminated or truncated)
            final_reward = float(reward)
            records.append(
                EpisodeResult(
                    seed=seed,
                    episode_idx=episode,
                    initial_state_seed=episode,
                    success=success_fn(final_reward),
                    metadata={"condition": spec.task_id, "final_reward": final_reward},
                )
            )
        episodes_by_seed[seed] = records
    if partner_spec is None:  # empty seed list — nothing ran, nothing to bundle
        msg = "seeds must be non-empty"
        raise ValueError(msg)
    return episodes_by_seed, partner_spec


__all__ = [
    "EGO_SUBSTREAM_PATTERN",
    "MPE_DIAGNOSTIC_SUCCESS_THRESHOLD",
    "POLICY_IDS",
    "SUCCESS_PREDICATES",
    "EgoPolicy",
    "RandomEgoPolicy",
    "build_ego_policy",
    "build_partner_spec",
    "run_task_episodes",
]
