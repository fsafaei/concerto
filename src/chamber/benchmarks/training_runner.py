# SPDX-License-Identifier: Apache-2.0
"""Chamber-side ego-AHT training-run launcher (T4b.11; ADR-002 §Decisions).

Bridges the method-side :func:`concerto.training.ego_aht.train` (which
intentionally does not import ``chamber.*``) to the concrete env +
partner instances that live on the benchmark side. The dependency
direction stays clean: ``chamber.benchmarks.training_runner`` imports
both ``concerto.*`` (the algorithm-agnostic loop + config) and
``chamber.*`` (the env + partner registry), and orchestrates them.

Public surface:

- :func:`build_env` — turns an :class:`~concerto.training.config.EnvConfig`
  into a concrete env (Phase-0 dispatch table:
  ``"mpe_cooperative_push"`` → :class:`MPECooperativePushEnv`;
  ``"stage0_smoke"`` is reserved for T4b.3 and raises
  :class:`NotImplementedError` until the HARL fork's
  ``concerto_env_adapter`` lands).
- :func:`build_partner` — turns a
  :class:`~concerto.training.config.PartnerConfig` into a frozen partner
  via :func:`chamber.partners.registry.load_partner` (ADR-009 §Decision).
- :func:`run_training` — the end-to-end entry point: builds env, builds
  partner, calls :func:`concerto.training.ego_aht.train`, returns the
  :class:`~concerto.training.ego_aht.RewardCurve`.

ADR-002 §Decisions / plan/05 §3.5: this module is the natural home for
the env-task → env-class dispatch table; concrete env classes live
here. M4b-7 will extend the dispatch with the HARL fork's
``concerto_env_adapter`` wrapper for ``stage0_smoke``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import gymnasium as gym

from chamber.benchmarks.ego_ppo_trainer import EgoPPOTrainer
from chamber.benchmarks.stage0_smoke_adapter import make_stage0_training_env
from chamber.envs.mpe_cooperative_push import MPECooperativePushEnv
from chamber.partners.api import PartnerSpec
from chamber.partners.registry import load_partner
from concerto.safety.api import AgentControlModel, DoubleIntegratorControlModel
from concerto.training.ego_aht import EnvLike, PartnerLike, RewardCurve, train

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping
    from pathlib import Path

    from concerto.training.config import EgoAHTConfig, EnvConfig, PartnerConfig
    from concerto.training.ego_aht import TrainerFactory


def build_env(env_cfg: EnvConfig, *, root_seed: int) -> EnvLike:
    """Construct the env from :class:`EnvConfig` (T4b.11; ADR-002 §Decisions; plan/05 §3.5).

    Phase-0 dispatch table:

    - ``"mpe_cooperative_push"`` → :class:`MPECooperativePushEnv` (T4b.13's
      empirical-guarantee env).
    - ``"stage0_smoke"`` → :func:`chamber.benchmarks.stage0_smoke_adapter.make_stage0_training_env`
      (T4b.3 / plan/05 §3.4). Constructs the rig-validated ADR-001
      3-robot env and adapts it to the
      :class:`~concerto.training.ego_aht.EnvLike` shape. Requires a
      Vulkan-capable GPU; raises
      :class:`chamber.envs.errors.ChamberEnvCompatibilityError` on
      CPU-only hosts.

    Args:
        env_cfg: The validated :class:`~concerto.training.config.EnvConfig`.
        root_seed: Project root seed routed to the env's
            :func:`concerto.training.seeding.derive_substream` substream
            for P6 reproducibility.

    Returns:
        Concrete env instance satisfying the
        :class:`~concerto.training.ego_aht.EnvLike` Protocol.

    Raises:
        ValueError: If ``env_cfg.task`` is not in the dispatch table.
        ChamberEnvCompatibilityError: If ``env_cfg.task == "stage0_smoke"``
            and SAPIEN / Vulkan is unavailable (see ADR-001 §Risks).
    """
    if env_cfg.task == "mpe_cooperative_push":
        return MPECooperativePushEnv(
            agent_uids=env_cfg.agent_uids,
            episode_length=env_cfg.episode_length,
            root_seed=root_seed,
        )
    if env_cfg.task == "stage0_smoke":
        return make_stage0_training_env(
            agent_uids=env_cfg.agent_uids,
            episode_length=env_cfg.episode_length,
            root_seed=root_seed,
        )
    raise ValueError(
        f"Unknown env task {env_cfg.task!r}; supported: 'mpe_cooperative_push', 'stage0_smoke'."
    )


def build_partner(partner_cfg: PartnerConfig) -> PartnerLike:
    """Construct the frozen partner from :class:`PartnerConfig` (ADR-009 §Decision).

    Builds a :class:`~chamber.partners.api.PartnerSpec` from the config
    fields and routes through :func:`chamber.partners.registry.load_partner`
    so the M4a registry's loud-fail contract (KeyError listing known
    classes) applies uniformly.

    Args:
        partner_cfg: The validated
            :class:`~concerto.training.config.PartnerConfig`.

    Returns:
        Concrete frozen partner satisfying the
        :class:`~concerto.training.ego_aht.PartnerLike` Protocol.

    Raises:
        KeyError: If ``partner_cfg.class_name`` is not registered in
            :data:`chamber.partners.registry._REGISTRY`.
    """
    spec = PartnerSpec(
        class_name=partner_cfg.class_name,
        seed=partner_cfg.seed,
        checkpoint_step=partner_cfg.checkpoint_step,
        weights_uri=partner_cfg.weights_uri,
        extra=dict(partner_cfg.extra),
    )
    return load_partner(spec)


def run_training(
    cfg: EgoAHTConfig,
    *,
    trainer_factory: TrainerFactory | None = None,
    repo_root: Path | None = None,
) -> RewardCurve:
    """Build env + partner and run the ego-AHT training loop (T4b.11; ADR-002 §Decisions).

    Drives :func:`concerto.training.ego_aht.train` with the chamber-side
    env + partner instances. The default ``trainer_factory`` is
    :meth:`~chamber.benchmarks.ego_ppo_trainer.EgoPPOTrainer.from_config`
    (M4b-8a / plan/05 §3.5): the HARL-HAPPO-backed ego-PPO trainer that
    the empirical-guarantee experiment (M4b-8b / T4b.13) measures.
    Pass ``trainer_factory=None`` is therefore a no-op fallback to the
    learning trainer; callers that explicitly want
    :class:`~concerto.training.ego_aht.RandomEgoTrainer` (test fixtures,
    smoke runs) must construct it and pass it in.

    Args:
        cfg: Validated :class:`~concerto.training.config.EgoAHTConfig`.
        trainer_factory: Optional :class:`~concerto.training.ego_aht.TrainerFactory`.
            ``None`` (default) selects the M4b-8a
            :class:`~chamber.benchmarks.ego_ppo_trainer.EgoPPOTrainer`
            via its ``from_config`` classmethod.
        repo_root: Working-tree root for run-metadata provenance.

    Returns:
        :class:`~concerto.training.ego_aht.RewardCurve`.

    Raises:
        ValueError: If the env task is unknown.
        NotImplementedError: If the env task is ``stage0_smoke`` (deferred).
        KeyError: If the partner class is not registered.
    """
    env = build_env(cfg.env, root_seed=cfg.seed)
    partner = build_partner(cfg.partner)
    factory: TrainerFactory = (
        trainer_factory if trainer_factory is not None else EgoPPOTrainer.from_config
    )
    return train(
        cfg,
        env=env,
        partner=partner,
        trainer_factory=factory,
        repo_root=repo_root,
    )


def build_control_models(env: gym.Env[Any, Any]) -> Mapping[str, AgentControlModel]:
    """Build a per-uid :class:`AgentControlModel` map from an env (ADR-004 §Decision; spike_004A).

    Reads each uid's ``action_space[uid]`` Box shape and returns a
    :class:`concerto.safety.api.DoubleIntegratorControlModel` per uid
    sized to ``shape[0]``. Phase-0 envs in CHAMBER expose their
    actions as Cartesian velocities or accelerations (the safety
    filter's CBF assumes Cartesian acceleration), so the
    double-integrator identity model is the correct default for every
    env currently in the dispatch table.

    Stage-1 AS spike will introduce embodiments whose action space
    is not Cartesian (e.g. a 7-DOF arm whose actions are joint
    torques); those callers construct a :class:`JacobianControlModel`
    directly rather than going through this helper. The helper's
    contract is "every uid I produce uses :class:`DoubleIntegratorControlModel`";
    if that no longer fits the env, the caller MUST supply
    ``control_models`` itself rather than silently relying on the
    helper to do the wrong thing.

    Args:
        env: A multi-agent Gymnasium-conformant env exposing
            ``action_space`` as a :class:`gym.spaces.Dict` of
            :class:`gym.spaces.Box` per uid (the contract every
            Phase-0 CHAMBER env satisfies).

    Returns:
        ``{uid: DoubleIntegratorControlModel(uid, action_dim=shape[0])}``
        for every uid in ``env.action_space.spaces``.

    Raises:
        TypeError: If ``env.action_space`` is not a
            :class:`gymnasium.spaces.Dict` or any ``action_space[uid]``
            is not a :class:`gymnasium.spaces.Box`.
        ValueError: If any uid's action Box is not 1-D (multi-D action
            shapes would need a custom control model rather than the
            double-integrator identity).
    """
    action_space = env.action_space
    if not isinstance(action_space, gym.spaces.Dict):
        msg = (
            "build_control_models requires env.action_space to be a "
            f"gym.spaces.Dict; got {type(action_space).__name__}."
        )
        raise TypeError(msg)
    control_models: dict[str, AgentControlModel] = {}
    for uid, sub in action_space.spaces.items():
        if not isinstance(sub, gym.spaces.Box):
            msg = f"action_space[{uid!r}] must be a gym.spaces.Box; got {type(sub).__name__}."
            raise TypeError(msg)
        if len(sub.shape) != 1:
            msg = (
                f"action_space[{uid!r}].shape={sub.shape} is not 1-D; "
                "non-1-D actions require a custom AgentControlModel rather "
                "than the DoubleIntegratorControlModel default."
            )
            raise ValueError(msg)
        model: AgentControlModel = DoubleIntegratorControlModel(
            uid=uid, action_dim=int(sub.shape[0])
        )
        control_models[uid] = model
    return control_models


__all__ = ["build_control_models", "build_env", "build_partner", "run_training"]
