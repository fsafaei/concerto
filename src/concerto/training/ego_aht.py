# SPDX-License-Identifier: Apache-2.0
"""Ego-AHT training loop entry point (T4b.11; ADR-002 §Decisions; plan/05 §3.5).

The single user-facing API for an ego-AHT training run is :func:`train`.
It wires up logging, seeding, per-step rollout collection, checkpoint
emission, and the trainer-factory seam, and returns a :class:`RewardCurve`
for T4b.13's empirical-guarantee assertion.

Dependency-direction discipline (project plan/10 §2 dependency-direction rule):
``concerto.*`` MUST NOT import from ``chamber.*``. The training loop
therefore takes the env and partner via dependency injection — the
chamber-side runner (``chamber.benchmarks.training_runner``) builds the
concrete env (:class:`chamber.envs.mpe_cooperative_push.MPECooperativePushEnv`)
and partner (via :func:`chamber.partners.registry.load_partner`) and
hands them to :func:`train`. That keeps every ``concerto.*`` arrow
pointing only at lower layers.

Trainer-factory injection: this loop lives in ``concerto.*`` and so cannot
import from ``chamber.*`` (project plan/10 §2). The real ego-PPO trainer
(:class:`chamber.benchmarks.ego_ppo_trainer.EgoPPOTrainer`) wraps HARL's
HAPPO and lives on the chamber side; the chamber-side
:func:`chamber.benchmarks.training_runner.run_training` selects it as the
default. :func:`train` therefore continues to accept ``trainer_factory``
and falls back to :class:`RandomEgoTrainer` when called directly without
an injected factory — which keeps the loop unit-testable inside
``concerto.*`` without violating the dependency direction.

ADR-002 risk-mitigation #1: this loop is the substrate the empirical-
guarantee assertion runs against. Do not silently drop steps, mutate
the rollout collection mid-loop, or change the partner-frozen contract
without a new ADR.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

from concerto.training.checkpoints import CheckpointMetadata, save_checkpoint
from concerto.training.logging import bind_run_logger, compute_run_metadata
from concerto.training.seeding import derive_substream

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping

    from numpy.typing import NDArray

    from concerto.training.config import EgoAHTConfig

#: Fallback ego-action dim used by the default :class:`RandomEgoTrainer`
#: when :func:`train` is called without an explicit ``trainer_factory``.
#:
#: Matches :class:`chamber.envs.mpe_cooperative_push.MPECooperativePushEnv`'s
#: action shape. The real :class:`EgoPPOTrainer
#: <chamber.benchmarks.ego_ppo_trainer.EgoPPOTrainer>` reads the action
#: dim from the env directly, so this constant is only the Phase-0
#: ``RandomEgoTrainer`` fallback used when :func:`train` is called
#: without an injected ``trainer_factory``.
_RANDOM_TRAINER_FALLBACK_ACTION_DIM: int = 2


@runtime_checkable
class EnvLike(Protocol):
    """Structural Gymnasium-multi-agent env contract the loop needs (T4b.11; ADR-002 §Decisions).

    Defined locally so :func:`train` does not import a concrete class
    from ``chamber.*`` (project plan/10 §2 dependency-direction rule). The
    chamber-side :class:`chamber.envs.mpe_cooperative_push.MPECooperativePushEnv`
    satisfies this Protocol structurally.
    """

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        """Reset to the start of an episode (T4b.11; ADR-002 §Decisions)."""
        ...  # pragma: no cover

    def step(
        self,
        action: dict[str, NDArray[np.floating]],
    ) -> tuple[Mapping[str, Any], float, bool, bool, Mapping[str, Any]]:
        """Advance one tick (T4b.11; ADR-002 §Decisions).

        Takes a concrete ``dict`` rather than a ``Mapping`` because the
        chamber-side :class:`MPECooperativePushEnv` and the future
        ManiSkill v3 wrappers all require dict-keyed access (mutation
        of the action dict is intentional in some wrappers' rate-limit
        / collision-injection paths).
        """
        ...  # pragma: no cover


@runtime_checkable
class PartnerLike(Protocol):
    """Structural ``FrozenPartner`` contract the loop needs (T4b.11; ADR-009 §Decision).

    Defined locally to keep ``concerto.*`` free of ``chamber.*`` imports.
    The M4a :class:`chamber.partners.api.FrozenPartner` satisfies this
    Protocol structurally — both the heuristic + the future frozen-RL
    adapters drop in unchanged.

    The ``spec`` attribute is declared as :class:`Any` so concrete
    implementations can attach any spec type (M4a's
    :class:`chamber.partners.api.PartnerSpec`); the loop only uses it
    for diagnostic logging, never structurally. ``Any`` is required
    here rather than ``object`` because Protocol attributes are
    invariant and concrete subtypes (PartnerSpec) are not assignable to
    a mutable ``object`` annotation.
    """

    spec: Any

    def reset(self, *, seed: int | None = None) -> None:
        """Reset internal episode state (ADR-009 §Decision)."""
        ...  # pragma: no cover

    def act(
        self,
        obs: Mapping[str, Any],
        *,
        deterministic: bool = True,
    ) -> NDArray[np.floating]:
        """Pick an action for the partner's own uid (ADR-009 §Decision)."""
        ...  # pragma: no cover


class EgoTrainer(Protocol):
    """Minimal contract the training loop expects of an ego trainer (T4b.11; ADR-002 §Decisions).

    Two concrete implementations satisfy this Protocol:

    - :class:`chamber.benchmarks.ego_ppo_trainer.EgoPPOTrainer` — the
      real HARL-HAPPO-backed ego-PPO trainer (M4b-8a / plan/05 §3.5),
      and the default selected by
      :func:`chamber.benchmarks.training_runner.run_training`.
    - :class:`RandomEgoTrainer` — the Phase-0 reference fallback, used
      by :func:`train` when no ``trainer_factory`` is injected.

    The Protocol is intentionally tiny (:meth:`act` + :meth:`observe` +
    :meth:`update` + :meth:`state_dict`) so the seam stays small enough
    that an external user can plug in their own algorithm without
    changing :func:`train`.
    """

    def act(
        self,
        obs: Mapping[str, Any],
        *,
        deterministic: bool = False,
    ) -> NDArray[np.floating]:
        """Sample (or pick deterministically) the ego action (ADR-002 §Decisions)."""
        ...  # pragma: no cover

    def observe(
        self,
        obs: Mapping[str, Any],
        reward: float,
        done: bool,
        *,
        truncated: bool = False,
    ) -> None:
        """Record a (s, a, r, s', done, truncated) tuple (ADR-002 §Decisions; Pardo 2017).

        ``done`` keeps the historical "this step ended the episode"
        meaning (terminated OR truncated). ``truncated`` is a
        keyword-only flag that distinguishes time-limit truncation from
        true termination — needed by PPO-style trainers to bootstrap the
        GAE target correctly (see :func:`chamber.benchmarks.ego_ppo_trainer.compute_gae`
        and project issue #62 for the root-cause writeup). Default
        ``truncated=False`` keeps legacy callers' behavior unchanged.
        """
        ...  # pragma: no cover

    def update(self) -> None:
        """Run one optimisation epoch on the buffered rollout (ADR-002 §Decisions)."""
        ...  # pragma: no cover

    def state_dict(self) -> dict[str, Any]:
        """Return the trainer's torch state-dict for checkpointing (ADR-002 §Decisions)."""
        ...  # pragma: no cover


class TrainerFactory(Protocol):
    """Constructor that builds an :class:`EgoTrainer` (T4b.11; ADR-002 §Decisions).

    The seam through which :func:`train` plugs in the algorithm.
    :func:`chamber.benchmarks.training_runner.run_training` selects
    :meth:`EgoPPOTrainer.from_config <chamber.benchmarks.ego_ppo_trainer.EgoPPOTrainer.from_config>`
    by default (M4b-8a / plan/05 §3.5).
    """

    def __call__(
        self,
        cfg: EgoAHTConfig,
        *,
        env: EnvLike,
        ego_uid: str,
    ) -> EgoTrainer:
        """Build a trainer (ADR-002 §Decisions; plan/05 §3.5)."""
        ...  # pragma: no cover


@dataclass(frozen=True)
class RewardCurve:
    """Result of one ego-AHT training run (T4b.11; T4b.13 consumer; ADR-002 §Decisions).

    Attributes:
        run_id: 16-hex run identifier (from
            :class:`concerto.training.logging.RunContext`).
        per_step_ego_rewards: One entry per env step. T4b.13's empirical-
            guarantee assertion runs over the moving-window-of-10 mean
            of this series.
        per_episode_ego_rewards: Sum-of-step reward per episode. Coarser
            than ``per_step``; the canonical "epoch reward" the
            empirical-guarantee plot embeds in
            ``docs/explanation/why-aht.md``.
        checkpoint_paths: Absolute paths of every ``.pt`` artefact saved
            during the run. T4b.14 reads the canonical 50%-step file
            from this list.
    """

    run_id: str
    per_step_ego_rewards: list[float] = field(default_factory=list)
    per_episode_ego_rewards: list[float] = field(default_factory=list)
    checkpoint_paths: list[Path] = field(default_factory=list)


class RandomEgoTrainer:
    """Reference :class:`EgoTrainer` that samples uniformly from the action space.

    Exists so :func:`train` can be exercised end-to-end before the HARL
    fork lands (T4b.4-T4b.7). Not a real learner — :meth:`update` is a
    no-op, :meth:`state_dict` returns an empty dict — but the loop's
    plumbing (env stepping, partner act, checkpoint emission, JSONL
    logging) is fully exercised. T4b.13 will swap to ``EgoAHTHAPPO``
    via the trainer-factory seam (ADR-002 §Decisions; plan/05 §3.5).
    """

    def __init__(self, *, ego_uid: str, action_dim: int, root_seed: int) -> None:
        """Build a uniform-random reference trainer (T4b.11; ADR-002 §Decisions).

        Args:
            ego_uid: Stored for downstream identity propagation (the
                trainer doesn't read it at action time but the loop
                uses it as the dict-action key).
            action_dim: Length of the ego's action vector.
            root_seed: Project seed; the trainer's RNG is derived via
                :func:`concerto.training.seeding.derive_substream` for
                P6 reproducibility.
        """
        self._ego_uid = ego_uid
        self._action_dim = action_dim
        self._rng: np.random.Generator = derive_substream(
            "training.ego_random", root_seed=root_seed
        ).default_rng()

    def act(
        self,
        obs: Mapping[str, Any],
        *,
        deterministic: bool = False,
    ) -> NDArray[np.floating]:
        """Sample uniformly from [-1, 1]^action_dim (ADR-002 §Decisions; plan/05 §3.5)."""
        del obs
        if deterministic:
            return np.zeros(self._action_dim, dtype=np.float32)
        return self._rng.uniform(-1.0, 1.0, size=self._action_dim).astype(np.float32)

    def observe(
        self,
        obs: Mapping[str, Any],
        reward: float,
        done: bool,
        *,
        truncated: bool = False,
    ) -> None:
        """No-op (ADR-002 §Decisions): reference trainer has no rollout buffer."""
        del obs, reward, done, truncated

    def update(self) -> None:
        """No-op (ADR-002 §Decisions): reference trainer has no learnable params."""

    def state_dict(self) -> dict[str, Any]:
        """Return an empty dict (ADR-002 §Decisions: trainer is parameter-free)."""
        return {}


def train(
    cfg: EgoAHTConfig,
    *,
    env: EnvLike,
    partner: PartnerLike,
    trainer_factory: TrainerFactory | None = None,
    repo_root: Path | None = None,
) -> RewardCurve:
    """Run one ego-AHT training loop (T4b.11; ADR-002 §Decisions; plan/05 §3.5).

    Wires up logging, seeding, per-step rollout, checkpoint emission, and
    the trainer-factory seam. Returns a :class:`RewardCurve` for the
    empirical-guarantee test (T4b.13) and for the user-side T4b.14
    zoo-seed run.

    Args:
        cfg: Validated :class:`~concerto.training.config.EgoAHTConfig`.
        env: Concrete env instance satisfying the :class:`EnvLike`
            structural Protocol. Built upstream by
            :func:`chamber.benchmarks.training_runner.build_env`.
        partner: Concrete frozen partner satisfying :class:`PartnerLike`.
            Built upstream by
            :func:`chamber.benchmarks.training_runner.build_partner`.
        trainer_factory: Constructor that builds the
            :class:`EgoTrainer`. When ``None`` (default), uses
            :class:`RandomEgoTrainer` — the Phase-0 in-``concerto.*``
            fallback. The chamber-side wrapper
            :func:`chamber.benchmarks.training_runner.run_training`
            injects the M4b-8a
            :class:`~chamber.benchmarks.ego_ppo_trainer.EgoPPOTrainer`
            (the real HARL-HAPPO trainer; plan/05 §3.5) here.
        repo_root: Working-tree root for the run-metadata bundle
            (defaults to :func:`pathlib.Path.cwd`).

    Returns:
        :class:`RewardCurve` carrying per-step + per-episode ego rewards
        and the list of saved checkpoint paths.
    """
    repo_root = repo_root or Path.cwd()
    ctx = compute_run_metadata(
        seed=cfg.seed,
        run_kind=cfg.algo,
        repo_root=repo_root,
        extra={
            "task": cfg.env.task,
            "partner_class": cfg.partner.class_name,
        },
    )
    jsonl_path = cfg.log_dir / f"{ctx.run_id}.jsonl"
    logger = bind_run_logger(ctx, jsonl_path=jsonl_path)
    logger.info(
        "training_start",
        total_frames=cfg.total_frames,
        checkpoint_every=cfg.checkpoint_every,
    )

    ego_uid, partner_uid = cfg.env.agent_uids
    if trainer_factory is None:
        trainer: EgoTrainer = RandomEgoTrainer(
            ego_uid=ego_uid,
            action_dim=_RANDOM_TRAINER_FALLBACK_ACTION_DIM,
            root_seed=cfg.seed,
        )
    else:
        trainer = trainer_factory(cfg, env=env, ego_uid=ego_uid)

    obs, _ = env.reset(seed=cfg.seed)
    partner.reset(seed=cfg.seed)
    curve = RewardCurve(run_id=ctx.run_id)
    episode_reward_acc = 0.0
    episode_in_progress = True
    # Episode index is incremented on each terminated/truncated boundary,
    # NOT inferred from step // episode_length — the latter would silently
    # misnumber for envs that early-terminate (P6 anti-pattern flagged in
    # local pre-PR review).
    episode_index = 0

    for step in range(cfg.total_frames):
        ego_action = trainer.act(obs)
        partner_action = partner.act(obs)
        action = {ego_uid: ego_action, partner_uid: partner_action}
        # The trainer.observe(s') contract: this loop passes the *next*
        # observation (the post-step obs), reward, and done flag. M4b-7's
        # HARL trainer is responsible for buffering pre-step obs internally
        # by tracking the last act() input. Phase-0 simplification per
        # plan/05 §3.5; revisit if a future on-policy trainer needs the
        # explicit (s, a, r, s', done) tuple.
        obs, reward, terminated, truncated, _ = env.step(action)
        curve.per_step_ego_rewards.append(reward)
        episode_reward_acc += reward
        # Pardo 2017 / issue #62: thread terminated AND truncated
        # separately. ``done`` keeps the historical episode-boundary
        # meaning (terminated OR truncated) so the loop's reset logic is
        # unchanged; ``truncated`` is the kwarg-only flag PPO-style
        # trainers use to bootstrap GAE correctly at time-limit
        # boundaries.
        trainer.observe(obs, reward, terminated or truncated, truncated=truncated)

        if (step + 1) % cfg.happo.rollout_length == 0:
            trainer.update()
            logger.info(
                "rollout_update",
                step=step + 1,
                last_reward=reward,
            )

        if (step + 1) % cfg.checkpoint_every == 0:
            ckpt_path = _save_run_checkpoint(
                cfg=cfg,
                ctx_run_id=ctx.run_id,
                seed=cfg.seed,
                step=step + 1,
                git_sha=ctx.git_sha,
                pyproject_hash=ctx.pyproject_hash,
                state_dict=trainer.state_dict(),
            )
            curve.checkpoint_paths.append(ckpt_path)
            logger.info("checkpoint_saved", step=step + 1, path=str(ckpt_path))

        if terminated or truncated:
            curve.per_episode_ego_rewards.append(episode_reward_acc)
            episode_reward_acc = 0.0
            episode_in_progress = False
            episode_index += 1
            # P6 reproducibility: derive a per-episode seed via a stable
            # substream rather than additive int mixing (cfg.seed +
            # episode_index would collide across runs). The substream
            # name pins the episode index uniquely under cfg.seed.
            episode_seed_int = int(
                derive_substream(f"training.episode.{episode_index}", root_seed=cfg.seed)
                .default_rng()
                .integers(0, 2**31 - 1)
            )
            obs, _ = env.reset(seed=episode_seed_int)
            partner.reset(seed=episode_seed_int)
            episode_in_progress = True

    # Tail-end episode (final partial chunk) — capture so the curve has
    # full coverage for the empirical-guarantee assertion. Use the
    # episode_in_progress sentinel rather than `episode_reward_acc != 0.0`
    # so a legitimately-zero accumulated reward isn't dropped.
    if episode_in_progress:
        curve.per_episode_ego_rewards.append(episode_reward_acc)

    logger.info(
        "training_end",
        n_episodes=len(curve.per_episode_ego_rewards),
        n_checkpoints=len(curve.checkpoint_paths),
    )
    return curve


def _save_run_checkpoint(
    *,
    cfg: EgoAHTConfig,
    ctx_run_id: str,
    seed: int,
    step: int,
    git_sha: str,
    pyproject_hash: str,
    state_dict: dict[str, Any],
) -> Path:
    """Write one checkpoint .pt + sidecar (T4b.11 + T4b.12)."""
    uri = f"local://artifacts/{ctx_run_id}_step{step}.pt"
    metadata = CheckpointMetadata(
        run_id=ctx_run_id,
        seed=seed,
        step=step,
        git_sha=git_sha,
        pyproject_hash=pyproject_hash,
        sha256="",  # save_checkpoint recomputes from the .pt bytes.
    )
    return save_checkpoint(
        state_dict=state_dict,
        uri=uri,
        metadata=metadata,
        artifacts_root=cfg.artifacts_root,
    )


__all__ = [
    "EgoTrainer",
    "EnvLike",
    "PartnerLike",
    "RandomEgoTrainer",
    "RewardCurve",
    "TrainerFactory",
    "train",
]
