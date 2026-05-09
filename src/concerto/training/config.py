# SPDX-License-Identifier: Apache-2.0
"""Hydra-driven config for ego-AHT training runs (T4b.11; ADR-002 §Decisions; plan/05 §2).

The training stack is configured via Hydra YAML files under
``configs/training/<algo>/<task>.yaml``. This module ships the
type-safe Pydantic v2 models that those YAMLs validate against:
``EgoAHTConfig`` is the root, with sub-models for the env / partner /
W&B / hyperparameter blocks.

Why two layers (Hydra + Pydantic)?

- Hydra owns the composition + CLI override surface (defaults, group
  selection, ``+key=value`` overrides).
- Pydantic owns the validation + Python-side type-safety (Path coercion,
  range checks, frozen-by-default to keep call sites from mutating
  config mid-run).

:func:`load_config` is the canonical entry point: it locates the project's
``configs/`` directory by walking up from the caller's CWD, composes the
named config via Hydra, and validates it through :class:`EgoAHTConfig`.

ADR-002 §Decisions ("Hydra config root"): the YAML schema is part of the
public reproducibility contract. Adding a required field is a breaking
change to the config; deprecate-then-remove via a Pydantic
:class:`Field` default and a ``DeprecationWarning`` rather than removing
in place.
"""

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field, field_validator


class _FrozenModel(BaseModel):
    """Base for every config model — frozen + extra='forbid' (loud-fail YAML typos)."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class WandbConfig(_FrozenModel):
    """W&B sink toggle + project name (ADR-002 §Decisions; plan/05 §2).

    Attributes:
        enabled: When ``False`` (default), :func:`bind_run_logger` is
            built without a W&B sink and only the JSONL fallback fires.
            CI runs and unit tests should leave this off.
        project: W&B project name. Used only when ``enabled`` is ``True``.
    """

    enabled: bool = False
    project: str = "concerto-m4b"


class EnvConfig(_FrozenModel):
    """Env-side configuration (T4b.13; ADR-002 §Decisions; plan/05 §3.5).

    Attributes:
        task: Registered task name. Phase-0 supports ``"mpe_cooperative_push"``
            (T4b.13's empirical-guarantee env) and ``"stage0_smoke"``
            (T4b.14's zoo-seed env, deferred to user-side GPU run).
        episode_length: Truncation horizon in env ticks. The empirical-
            guarantee experiment defaults to 50 (matches PettingZoo
            simple_spread).
        agent_uids: 2-element list of uids the env exposes; the first is
            the ego, the second is the frozen partner.
    """

    task: str
    episode_length: int = Field(default=50, gt=0)
    agent_uids: tuple[str, str] = ("ego", "partner")

    @field_validator("agent_uids", mode="before")
    @classmethod
    def _coerce_agent_uids(cls, v: object) -> object:
        # Hydra/OmegaConf parses YAML lists as ListConfig (subclass of list),
        # not tuple — coerce to keep the tuple-typed contract intact.
        if isinstance(v, list):
            return tuple(v)
        return v

    @field_validator("agent_uids")
    @classmethod
    def _check_distinct_uids(cls, v: tuple[str, str]) -> tuple[str, str]:
        if v[0] == v[1]:
            raise ValueError(f"agent_uids must be distinct; got {v!r}")
        return v


class PartnerConfig(_FrozenModel):
    """Frozen partner spec (ADR-009 §Decision; plan/05 §3.5).

    Mirrors the M4a :class:`chamber.partners.api.PartnerSpec` fields so
    the training loop can build a ``PartnerSpec`` directly from the
    config without re-validating.

    Attributes:
        class_name: Registry key passed to
            :func:`chamber.partners.registry.load_partner`. Phase-0
            empirical-guarantee runs use ``"scripted_heuristic"``.
        seed: Per-partner training seed; for scripted partners use ``0``.
        checkpoint_step: Training step at which the checkpoint was
            taken; ``None`` for scripted partners.
        weights_uri: ``local://...`` URI for frozen-RL partners; ``None``
            for scripted partners.
        extra: Free-form string-string metadata routed to
            ``PartnerSpec.extra``.
    """

    class_name: str = "scripted_heuristic"
    seed: int = 0
    checkpoint_step: int | None = None
    weights_uri: str | None = None
    extra: dict[str, str] = Field(default_factory=dict)


class HAPPOHyperparams(_FrozenModel):
    """On-policy ego-AHT HAPPO hyperparameters (ADR-002 §Decisions; plan/05 §3.2).

    Defaults match the published HARL Bi-DexHands-style values where
    they translate; tuned conservatively for the small MPE task to keep
    the empirical-guarantee experiment within its 30-minute CPU budget
    (plan/05 §8).

    Attributes:
        lr: Adam learning rate for the actor + critic.
        gamma: Discount factor.
        gae_lambda: GAE-λ for advantage estimation.
        clip_eps: PPO clipping epsilon.
        n_epochs: Number of optimisation epochs per rollout.
        rollout_length: Frames collected per epoch before the update step.
        batch_size: SGD minibatch size; rollout_length must be a multiple.
        hidden_dim: MLP hidden width for actor + critic.
    """

    lr: float = Field(default=3.0e-4, gt=0.0)
    gamma: float = Field(default=0.99, ge=0.0, le=1.0)
    gae_lambda: float = Field(default=0.95, ge=0.0, le=1.0)
    clip_eps: float = Field(default=0.2, gt=0.0)
    n_epochs: int = Field(default=4, gt=0)
    rollout_length: int = Field(default=1024, gt=0)
    batch_size: int = Field(default=256, gt=0)
    hidden_dim: int = Field(default=64, gt=0)


class EgoAHTConfig(_FrozenModel):
    """Root config for an ego-AHT training run (T4b.11; ADR-002 §Decisions).

    Validates the composed Hydra config and exposes a frozen
    Python-side handle for :func:`concerto.training.ego_aht.train`.

    Attributes:
        algo: Trainer-class registry key. Phase-0: ``"ego_aht_happo"``;
            ``"ego_aht_hatd3"`` is the Phase-1 stub.
        seed: Project root seed routed to
            :func:`concerto.training.seeding.derive_substream`. Two
            runs with the same ``seed`` produce byte-identical CPU
            reward curves (plan/05 §6 criterion 7).
        total_frames: Training budget in env frames. T4b.13 = 100_000.
        checkpoint_every: Save a ``.pt`` artefact every K frames.
        artifacts_root: Directory the ``local://artifacts/...`` URIs
            resolve against (plan/04 §3.8).
        log_dir: Directory where the JSONL logs are written.
        wandb: :class:`WandbConfig`.
        env: :class:`EnvConfig`.
        partner: :class:`PartnerConfig`.
        happo: :class:`HAPPOHyperparams`.
    """

    algo: str = "ego_aht_happo"
    seed: int = Field(default=0, ge=0)
    total_frames: int = Field(default=100_000, gt=0)
    checkpoint_every: int = Field(default=10_000, gt=0)
    artifacts_root: Path = Path("./artifacts")
    log_dir: Path = Path("./logs")
    wandb: WandbConfig = Field(default_factory=WandbConfig)
    env: EnvConfig
    partner: PartnerConfig = Field(default_factory=PartnerConfig)
    happo: HAPPOHyperparams = Field(default_factory=HAPPOHyperparams)


def load_config(
    *,
    config_path: Path,
    overrides: list[str] | None = None,
) -> EgoAHTConfig:
    """Compose a Hydra config + validate via :class:`EgoAHTConfig` (T4b.11; ADR-002 §Decisions).

    The function does NOT use Hydra's ``@hydra.main`` decorator (which
    takes over the process's CWD + argv); it uses the lower-level
    :class:`hydra.compose` API so the call is testable and re-entrant.

    Args:
        config_path: Absolute path to the YAML config file (e.g.
            ``configs/training/ego_aht_happo/mpe_cooperative_push.yaml``).
            The directory becomes Hydra's ``config_dir`` and the
            stem becomes the ``config_name`` it loads. Splitting via the
            file path keeps the YAML's root-level keys at the root of
            the composed config (rather than Hydra's path-folding).
        overrides: Optional list of Hydra CLI-style overrides
            (e.g. ``["seed=7", "happo.lr=1e-4"]``).

    Returns:
        A frozen :class:`EgoAHTConfig` ready for
        :func:`concerto.training.ego_aht.train`.

    Raises:
        pydantic.ValidationError: If the composed config violates the
            schema (missing required field, out-of-range hyperparam,
            duplicate agent uids, etc.).
    """
    config_dir = config_path.parent.resolve()
    config_name = config_path.stem
    # Re-entrancy: another caller (or a leftover pytest fixture) may have
    # left ``GlobalHydra`` initialised. Clear it so initialize_config_dir's
    # ``with`` block can re-bind to the new directory cleanly.
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = compose(config_name=config_name, overrides=overrides or [])
    raw = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(raw, dict):
        raise TypeError(
            f"Composed Hydra config must be a mapping at root; got {type(raw).__name__}"
        )
    return EgoAHTConfig.model_validate(raw)


__all__ = [
    "EgoAHTConfig",
    "EnvConfig",
    "HAPPOHyperparams",
    "PartnerConfig",
    "WandbConfig",
    "load_config",
]
