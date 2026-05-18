# SPDX-License-Identifier: Apache-2.0
"""Stage-1 spike-adapter common surface (ADR-007 §Stage 1a / §Stage 1b; plan/07 §T5b.2).

Single source of truth for the seam that Stage-1 AS and OM adapters
(:mod:`chamber.benchmarks.stage1_as`, :mod:`chamber.benchmarks.stage1_om`)
share — the :class:`EgoActionFactory` Protocol the Phase-1 trained-ego
path plugs into, and the Stage-1a production-default
:func:`_zero_ego_action_factory`.

Why a Protocol (not a per-step ``Callable``):

- The factory contract is *called once per ``(seed, condition)`` pair*.
  That matches the natural shape of a trained-policy injection point —
  Phase 1 (ADR-007 §Stage 1b) supplies a factory that calls
  :func:`concerto.training.ego_aht.train` for the homogeneous condition's
  env at the given seed, then returns a closure that wraps the trained
  policy's ``act`` (the closure is then re-used across the 20 evaluation
  episodes within that ``(seed, condition)`` cell).
- The Phase-0 per-step ``ego_action_fn`` was unable to express this
  contract — there was no place to hang a "this happens once per
  ``(seed, condition)``, not once per step" idea. The Protocol seam
  pins the lifecycle explicitly.

Phase-1 wiring contract (verbatim, do not rewrite without an ADR-007
amendment):

  *Phase-1 supplies a factory that calls
  ``concerto.training.ego_aht.train`` for the homogeneous condition's
  env at the given seed, returns ``lambda obs: trained_policy.act(obs)``.
  The factory is called once per ``(seed, condition)`` pair so the
  trained policy is reused across the 20 evaluation episodes within a
  seed.*

Stage-1a production default: :func:`_zero_ego_action_factory` returns a
closure that emits a zero action vector of the env's ego-action shape.
Stage-1a's purpose is to exercise the rig (env steps, partner acts,
episode terminations, success thresholding, SpikeRun aggregation,
chamber-eval pipeline) without confounding the rig signal with a
trainer signal; a learning ego is sequenced to Stage-1b
(ADR-007 §Stage 1b).

Stage-1b production default (P1.04): :class:`TrainedPolicyFactory` —
wires :func:`chamber.benchmarks.training_runner.run_training` for each
``(seed, condition)`` cell and returns a closure that wraps the trained
:class:`~concerto.training.ego_aht.EgoTrainer`'s ``act`` method
(deterministic mode). One ego-AHT training run per cell (5 GPU-h per
axis on A100 at the 100k-frame budget per ADR-007 §Stage 1b); the
trainer is recovered from the
:class:`~concerto.training.ego_aht.TrainingResult` NamedTuple rather
than reloaded from disk, so no checkpoint round-trip per cell. The
Phase-0 ``_zero_ego_action_factory`` is unchanged — Stage-1a runs keep
the rig-validation contract; the Stage-1b dispatch swap is sequenced
to P1.05 (separate slice).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Protocol, TypeAlias

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:  # pragma: no cover
    from concerto.training.config import EgoAHTConfig

#: Per-step ego-action callable returned by an :class:`EgoActionFactory`.
#:
#: Signature: ``(obs) -> action``. Takes the env's nested observation
#: dict, returns a length-``ego_act_dim`` float32 vector ready to be
#: packed into ``{ego_uid: ego_action, partner_uid: partner_action}``
#: at the env-step boundary by the spike-adapter loop.
EgoActionCallable: TypeAlias = Callable[[Mapping[str, Any]], NDArray[np.float32]]


class EgoActionFactory(Protocol):
    """Stage-1 ego-action factory contract (ADR-007 §Stage 1a / §Stage 1b).

    Called once per ``(seed, condition)`` pair by the Stage-1 spike
    adapter's :func:`_run_axis_with_factories`. The returned callable
    is invoked once per env step for the 20 evaluation episodes within
    that ``(seed, condition)`` cell.

    Two production implementations satisfy this Protocol:

    - :func:`_zero_ego_action_factory` — the Stage-1a default (rig
      validation). Returns a closure that emits a zero action vector
      of the env's ego-action shape. No randomness; no training.
    - :class:`TrainedPolicyFactory` — the Stage-1b default (science
      evaluation; ADR-007 §Stage 1b). Wires
      :func:`concerto.training.ego_aht.train` via
      :func:`chamber.benchmarks.training_runner.run_training` for each
      ``(seed, condition)`` cell, then returns a closure that wraps
      the trained :class:`~concerto.training.ego_aht.EgoTrainer`'s
      ``act`` method (deterministic mode for evaluation).

    The Protocol is intentionally tiny — ``(env, seed)`` → callable —
    so an external user can plug in their own algorithm without
    changing :func:`_run_axis_with_factories`. Project test fixtures
    can build a counting factory (see
    ``tests.integration.test_stage1_*_fake``) to assert the lifecycle
    contract.
    """

    def __call__(
        self,
        env: gym.Env[Any, Any],
        seed: int,
    ) -> Callable[[Mapping[str, Any]], NDArray[np.float32]]:
        """Build the per-step ego-action callable for one ``(seed, condition)`` cell."""
        ...  # pragma: no cover


def _zero_ego_action_factory(
    env: gym.Env[Any, Any],
    seed: int,
) -> Callable[[Mapping[str, Any]], NDArray[np.float32]]:
    """Stage-1a production default: always-zero ego action (ADR-007 §Stage 1a).

    Inspects ``env.action_space`` to discover the ego's action shape
    (every Phase-0 chamber env exposes ``action_space`` as a
    :class:`gymnasium.spaces.Dict` keyed by ``uid``, each value a
    :class:`gymnasium.spaces.Box`). Returns a closure that emits a
    fresh zero vector per call so the spike adapter never observes a
    mutated shared array.

    Stage-1a's purpose is rig validation, not science evaluation. A
    learning ego would confound the rig's signal — env step contract,
    partner act stream, episode terminations, success thresholding —
    with a trainer signal, defeating the rig-validation purpose. The
    trained-ego path is Stage-1b (Phase-1, ADR-007 §Stage 1b).

    Args:
        env: The per-condition env. ``env.action_space`` must be a
            :class:`gymnasium.spaces.Dict` of :class:`gymnasium.spaces.Box`
            (the Phase-0 chamber env contract).
        seed: Per-``(seed, condition)`` cell seed. Unused by the
            zero-action default (no randomness); kept in the signature
            for Protocol conformance.

    Returns:
        A ``(obs) -> action`` callable that returns a length-N float32
        zero vector, where N is the ego's action-space dim.
    """
    del seed  # zero action is deterministic; no per-cell seed needed.
    action_space = env.action_space
    if not isinstance(action_space, gym.spaces.Dict):
        msg = (
            "_zero_ego_action_factory expects env.action_space to be a "
            f"gym.spaces.Dict; got {type(action_space).__name__}."
        )
        raise TypeError(msg)
    # Stage-1a stand-in: every uid shares the same Box action shape
    # (MPE Cooperative-Push), so picking any uid's Box is well-defined.
    # Stage-1b's real ManiSkill v3 env will need explicit ego_uid
    # plumbing — that change goes with the trained-policy factory, not
    # here.
    sample_uid = next(iter(action_space.spaces.keys()))
    ego_box = action_space.spaces[sample_uid]
    if not isinstance(ego_box, gym.spaces.Box):
        msg = (
            f"_zero_ego_action_factory expects action_space[{sample_uid!r}] "
            f"to be a gym.spaces.Box; got {type(ego_box).__name__}."
        )
        raise TypeError(msg)
    n_dims = int(ego_box.shape[0])

    def _act(obs: Mapping[str, Any]) -> NDArray[np.float32]:
        """Emit a fresh length-``n_dims`` zero vector (ADR-007 §Stage 1a)."""
        del obs
        return np.zeros(n_dims, dtype=np.float32)

    return _act


class TrainedPolicyFactory:
    """Phase-1 :class:`EgoActionFactory` — wires the ego-AHT training loop per cell.

    ADR-007 §Stage 1b production factory. Implements the
    :class:`EgoActionFactory` Protocol (this module) by calling
    :func:`chamber.benchmarks.training_runner.run_training` for each
    ``(seed, condition)`` cell and returning a closure that wraps the
    trained :class:`~concerto.training.ego_aht.EgoTrainer`'s ``act``
    method (deterministic mode for evaluation).

    Lifecycle (ADR-007 §Stage 1b lifecycle contract):

    The factory is called *once per ``(seed, condition)`` pair* by the
    Stage-1 spike adapter's :func:`_run_axis_with_factories` (5 seeds x
    2 conditions = 10 calls per Stage-1 spike). Each call drives one
    full training run (5 x 100k frames per axis = 1M frames per axis
    on A100; ~5 GPU-h per axis per the compute plan); the returned
    closure is re-used across the 20 evaluation episodes within the
    cell so the trained policy is not retrained per-episode.

    Determinism (ADR-007 §Discipline + ADR-002 §Decisions; P6):

    The trained-policy seed is set via Pydantic ``model_copy`` on
    ``cfg.seed``; the underlying training run derives all its
    randomness from
    :func:`concerto.training.seeding.derive_substream`. Two invocations
    at the same ``(seed, condition)`` pair produce byte-identical
    trained policies on CPU; on GPU, the cuDNN / cuBLAS non-bit-
    determinism caveat (plan/08 §4) applies.

    Caching policy (intentionally NO cache):

    The factory does NOT cache trained policies across ``__call__``
    invocations — each call trains a fresh policy. Caching would
    silently violate the per-``(seed, condition)`` determinism contract
    if the cache key were not exhaustive (env identity, partner
    identity, prereg tag, code state). If the operator needs to resume
    a partial run, the right tool is the
    :class:`~concerto.training.checkpoints.CheckpointMetadata` sidecar
    serialised under ``cfg.artifacts_root`` (the training loop emits
    one per ``cfg.checkpoint_every`` steps); not an in-memory cache.

    Partner-freeze gate (ADR-009 §Consequences; plan/05 §6 #3):

    The partner-freeze gate is honoured by
    :meth:`EgoPPOTrainer.__init__`'s :func:`_assert_partner_is_frozen`
    call (run first, before any tensor allocation). The factory does
    not bypass it.

    Per-condition partner-uid routing (P1.04 / Q-C refinement):

    The factory reads the env's :attr:`partner_uid` and :attr:`ego_uid`
    properties at ``__call__`` time and overrides
    ``cfg.env.agent_uids`` accordingly — single source of truth at the
    env layer; the factory carries no per-condition dispatch table.
    For AS-homo (partner=``panda_partner``) vs AS-hetero / OM-*
    (partner=``fetch``), the env's ``condition_id`` drives the routing
    via the
    :data:`chamber.envs.stage1_pickplace._CONDITION_TABLE`. Envs that
    don't expose the properties (Tier-1 fake envs, MPE stand-in) fall
    back to ``cfg.env.agent_uids[0]`` / ``[1]`` — the Phase-0 contract
    that pre-dates the property.

    Action-dim handoff to partner (P1.04 / Q-C refinement):

    The :class:`~chamber.partners.heuristic.ScriptedHeuristicPartner`
    reads its action dimension from ``spec.extra["action_dim"]`` and
    zero-pads beyond the planar xy. The factory reads the actual
    ``env.action_space.spaces[partner_uid].shape[0]`` at call time and
    overrides ``cfg.partner.extra["action_dim"]`` so the partner's
    action vector matches the env's expectation per-condition. This
    avoids a side-table; the action_dim is sourced from the env that
    owns the contract.

    Args:
        cfg: A fully-resolved :class:`~concerto.training.config.EgoAHTConfig`.
            The factory rewrites ``cfg.seed`` per call via
            ``cfg.model_copy(update={"seed": call_seed})`` and applies
            the per-condition env+partner overrides described above;
            all other fields are honoured as-is.

    Example wiring at the Stage-1b adapter (sequenced to P1.05):

    .. code-block:: python

        from chamber.benchmarks.stage1_common import TrainedPolicyFactory
        from concerto.training.config import load_config

        factory = TrainedPolicyFactory(
            cfg=load_config(
                config_path=Path("configs/training/ego_aht_happo/stage1_pickplace.yaml")
            )
        )
        spike_run = _run_axis_with_factories(
            prereg=spec,
            prereg_sha=prereg_sha,
            env_factory=_default_env_factory,
            ego_action_factory=factory,
        )
    """

    def __init__(self, *, cfg: EgoAHTConfig) -> None:
        """Build the factory; capture the base config.

        Args:
            cfg: Validated :class:`~concerto.training.config.EgoAHTConfig`
                — the Stage-1b config (typically loaded from
                ``configs/training/ego_aht_happo/stage1_pickplace.yaml``).
                The original instance is never mutated; per-call
                overrides are applied via ``model_copy``.
        """
        self._cfg = cfg

    def __call__(
        self,
        env: gym.Env[Any, Any],
        seed: int,
    ) -> EgoActionCallable:
        """Train a per-cell policy and return the per-step closure (ADR-007 §Stage 1b).

        Lifecycle: invoked once per ``(seed, condition)`` cell by the
        Stage-1 adapter's :func:`_run_axis_with_factories`. The returned
        callable is reused across the 20 evaluation episodes within
        the cell.

        Args:
            env: The per-condition evaluation env. ``env.ego_uid`` /
                ``env.partner_uid`` (P1.04 properties on
                :class:`chamber.envs.stage1_pickplace.Stage1PickPlaceEnv`)
                drive the per-condition overrides applied to the base
                ``cfg``. ``env`` itself is NOT stepped by the factory;
                the training run inside :func:`run_training` builds
                its own env from the (overridden) ``cfg.env``. The
                eval env is what the spike adapter's loop steps after
                this factory returns.
            seed: Per-cell seed; the factory overrides ``cfg.seed`` via
                ``model_copy``.

        Returns:
            ``(obs) -> action`` callable that wraps
            ``trained_policy.act(obs, deterministic=True)`` and casts
            the output to ``float32``.

        Raises:
            ValueError: Propagated from :meth:`EgoPPOTrainer.__init__`
                if the partner is non-frozen (ADR-009 §Consequences).
                Any other exception raised by
                :func:`chamber.benchmarks.training_runner.run_training`
                also propagates unchanged (per the
                ``test_run_training_failure_propagates`` contract in
                ``tests/integration/test_trained_policy_factory_fake.py``).
        """
        # Imported lazily so chamber.benchmarks.stage1_common stays
        # cheap to import for Stage-1a-only consumers (the Phase-0 path
        # that does not need the HARL HAPPO trainer in scope).
        from chamber.benchmarks.training_runner import run_training

        # 1. Resolve the per-condition ego + partner uids from the env
        #    (P1.04 / Q-C: single source of truth at the env layer).
        #    Fall back to the cfg's agent_uids for envs that don't
        #    expose the properties (Tier-1 fake envs, MPE stand-in).
        ego_uid = self._resolve_ego_uid(env)
        partner_uid = self._resolve_partner_uid(env)

        # 2. Resolve the partner's action_dim from the env's
        #    action_space — avoids a per-condition side-table.
        partner_action_dim = self._resolve_partner_action_dim(env, partner_uid)

        # 3. Resolve the env's condition_id for Stage-1b disambiguation
        #    (OM-homo vs OM-hetero share the same agent_uids tuple, so
        #    the condition_id is the explicit signal that drives the
        #    Stage-1b env build inside run_training → build_env). None
        #    for non-Stage-1b envs; build_env ignores the field when
        #    task != "stage1_pickplace".
        condition_id = self._resolve_condition_id(env)

        # 4. Compose the per-call cfg via Pydantic model_copy so the
        #    base config passed at __init__ is never mutated.
        cfg_for_call = self._compose_cfg_for_call(
            seed=seed,
            ego_uid=ego_uid,
            partner_uid=partner_uid,
            partner_action_dim=partner_action_dim,
            condition_id=condition_id,
        )

        # 5. Drive the chamber-side training entry point. Returns a
        #    TrainingResult NamedTuple (P1.04) carrying the curve +
        #    the trained EgoTrainer instance — no checkpoint
        #    round-trip needed.
        result = run_training(cfg_for_call, repo_root=None)
        trainer = result.trainer

        # 6. Build the per-step closure. Deterministic mode for eval;
        #    cast to float32 so the adapter's step-loop dict can be
        #    handed straight to the env's step boundary.
        def _act(obs: Mapping[str, Any]) -> NDArray[np.float32]:
            """ADR-007 §Stage 1b per-step ego closure (deterministic)."""
            action = trainer.act(obs, deterministic=True)
            return np.asarray(action, dtype=np.float32)

        return _act

    # ----- Internal composition helpers -----

    def _resolve_ego_uid(self, env: gym.Env[Any, Any]) -> str:
        """Read ``env.ego_uid`` (P1.04 property); fall back to ``cfg.env.agent_uids[0]``."""
        uid = getattr(env, "ego_uid", None)
        if isinstance(uid, str):
            return uid
        return self._cfg.env.agent_uids[0]

    def _resolve_partner_uid(self, env: gym.Env[Any, Any]) -> str:
        """Read ``env.partner_uid`` (P1.04 property); fall back to ``cfg.env.agent_uids[1]``."""
        uid = getattr(env, "partner_uid", None)
        if isinstance(uid, str):
            return uid
        return self._cfg.env.agent_uids[1]

    def _resolve_partner_action_dim(
        self,
        env: gym.Env[Any, Any],
        partner_uid: str,
    ) -> int:
        """Read partner action_dim from ``env.action_space.spaces[partner_uid].shape[0]``.

        Falls back to the cfg's existing ``partner.extra["action_dim"]`` if
        the env's action_space doesn't carry the partner uid (Tier-1 fake
        envs that don't index by the same uid).
        """
        action_space = env.action_space
        if isinstance(action_space, gym.spaces.Dict):
            sub = action_space.spaces.get(partner_uid)
            if isinstance(sub, gym.spaces.Box) and sub.shape is not None:
                return int(sub.shape[0])
        # Fallback: trust the cfg.
        existing = self._cfg.partner.extra.get("action_dim", "2")
        try:
            return int(existing)
        except (TypeError, ValueError):
            return 2

    def _resolve_condition_id(self, env: gym.Env[Any, Any]) -> str | None:
        """Read ``env.condition_id`` (P1.03 property); ``None`` for non-Stage-1b envs.

        The condition_id disambiguates OM-homo vs OM-hetero at the
        chamber-side build_env dispatch (both conditions share the
        ``("panda_wristcam", "fetch")`` agent_uids tuple, so the
        condition_id is the only signal that drives the right Stage-1b
        env build).
        """
        cid = getattr(env, "condition_id", None)
        return cid if isinstance(cid, str) else None

    def _compose_cfg_for_call(
        self,
        *,
        seed: int,
        ego_uid: str,
        partner_uid: str,
        partner_action_dim: int,
        condition_id: str | None,
    ) -> EgoAHTConfig:
        """Apply seed + env + partner overrides via Pydantic ``model_copy`` (P6)."""
        env_update: dict[str, object] = {"agent_uids": (ego_uid, partner_uid)}
        if condition_id is not None:
            env_update["condition_id"] = condition_id
        env_override = self._cfg.env.model_copy(update=env_update)
        partner_extra = dict(self._cfg.partner.extra)
        partner_extra["uid"] = partner_uid
        partner_extra["action_dim"] = str(partner_action_dim)
        partner_override = self._cfg.partner.model_copy(update={"extra": partner_extra})
        return self._cfg.model_copy(
            update={
                "seed": seed,
                "env": env_override,
                "partner": partner_override,
            }
        )


__all__ = [
    "EgoActionCallable",
    "EgoActionFactory",
    "TrainedPolicyFactory",
]
