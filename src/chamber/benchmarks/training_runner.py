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

from typing import TYPE_CHECKING

from chamber.envs.mpe_cooperative_push import MPECooperativePushEnv
from chamber.partners.api import PartnerSpec
from chamber.partners.registry import load_partner
from concerto.training.ego_aht import EnvLike, PartnerLike, RewardCurve, train

if TYPE_CHECKING:  # pragma: no cover
    from pathlib import Path

    from concerto.training.config import EgoAHTConfig, EnvConfig, PartnerConfig
    from concerto.training.ego_aht import TrainerFactory


def build_env(env_cfg: EnvConfig, *, root_seed: int) -> EnvLike:
    """Construct the env from :class:`EnvConfig` (T4b.11; ADR-002 §Decisions; plan/05 §3.5).

    Phase-0 dispatch table:

    - ``"mpe_cooperative_push"`` → :class:`MPECooperativePushEnv` (T4b.13's
      empirical-guarantee env).
    - ``"stage0_smoke"`` → reserved for T4b.3 (HARL fork's
      ``concerto_env_adapter``); raises :class:`NotImplementedError`
      until that lands.

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
        NotImplementedError: If ``env_cfg.task == "stage0_smoke"``.
    """
    if env_cfg.task == "mpe_cooperative_push":
        return MPECooperativePushEnv(
            agent_uids=env_cfg.agent_uids,
            episode_length=env_cfg.episode_length,
            root_seed=root_seed,
        )
    if env_cfg.task == "stage0_smoke":
        raise NotImplementedError(
            "stage0_smoke env is wired up via the HARL fork's "
            "concerto_env_adapter (T4b.3); see ADR-002 §Decisions and "
            "phase0_reading_kit/plan/05-training-stack.md §3.4."
        )
    raise ValueError(
        f"Unknown env task {env_cfg.task!r}; supported: 'mpe_cooperative_push', "
        "'stage0_smoke' (deferred to T4b.3)."
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
    """Build env + partner and run :func:`concerto.training.ego_aht.train` (T4b.11; ADR-002).

    Args:
        cfg: Validated :class:`~concerto.training.config.EgoAHTConfig`.
        trainer_factory: Optional :class:`~concerto.training.ego_aht.TrainerFactory`
            (default: ``None`` → :class:`~concerto.training.ego_aht.RandomEgoTrainer`).
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
    return train(
        cfg,
        env=env,
        partner=partner,
        trainer_factory=trainer_factory,
        repo_root=repo_root,
    )


__all__ = ["build_env", "build_partner", "run_training"]
