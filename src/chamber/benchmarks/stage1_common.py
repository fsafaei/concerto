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
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Protocol, TypeAlias

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

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

    Two concrete implementations satisfy this Protocol in the project:

    - :func:`_zero_ego_action_factory` — the Stage-1a production
      default. Returns a closure that emits a zero action vector of
      the env's ego-action shape. No randomness; rig validation only.
    - Phase-1 trained-policy factory (Stage 1b; not yet implemented
      in this repo). Wires :func:`concerto.training.ego_aht.train` to
      train for ``total_frames=cfg.total_frames`` against the
      ``(env, seed)`` cell, then returns a closure that wraps the
      trained policy's ``act`` method.

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


__all__ = [
    "EgoActionCallable",
    "EgoActionFactory",
]
