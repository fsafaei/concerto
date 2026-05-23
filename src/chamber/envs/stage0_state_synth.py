# SPDX-License-Identifier: Apache-2.0
"""Stage-0 per-uid ``state`` synthesiser (ADR-001 §Validation criteria; closes #184).

ManiSkill v3's ``obs_mode="state_dict"`` (used by
:func:`chamber.benchmarks.stage0_smoke.make_stage0_env`) emits per-uid
``Dict(joint_pos: Box, joint_vel: Box, ...)`` for most robots, with no
top-level ``state`` key. Both the training adapter
(:class:`chamber.benchmarks.stage0_smoke_adapter._Stage0TrainingAdapter`)
and the M4-gate draft-zoo probe in
``tests/integration/test_draft_zoo.py`` expect a per-uid ``state`` Box
on the env's observation space.

This wrapper is the Stage-0 parallel to
:class:`chamber.envs.stage1_obs_filter.Stage1ASStateSynthesizer`: it
concatenates each uid's 1-D :class:`gym.spaces.Box` channels in
alphabetical key order to synthesise a flat ``state`` Box, leaving any
uid that already exposes ``state`` untouched. The synthesis policy
mirrors the existing helper inside
:mod:`chamber.benchmarks.stage0_smoke_adapter` so the adapter remains a
correct pass-through once the env layer already supplies ``state``.

Composing the wrapper at the env-factory layer makes ``make_stage0_env``'s
public observation-space contract self-describing, which is the
contract the Stage-1 factory already follows (ADR-007 §Stage 1b Rev 12).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import gymnasium as gym
import numpy as np

from chamber.envs.errors import ChamberEnvCompatibilityError

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping

#: Per-agent state-channel name the trainer and the draft-zoo probe read
#: (ADR-002 §Decisions). Matches
#: :data:`chamber.benchmarks.stage0_smoke_adapter._STATE_KEY`.
_STATE_KEY: str = "state"

#: Rank of the vectorised-by-num_envs obs shapes ManiSkill emits when
#: ``num_envs=1`` adds a leading batch dimension — ``(1, n)`` is rank 2.
_VECTORISED_RANK: int = 2


class Stage0StateSynthesizer(gym.ObservationWrapper):  # type: ignore[type-arg]
    """Inject per-uid ``state`` into Stage-0 observations (ADR-001 §Validation; #184).

    Stage-0 parallel to
    :class:`chamber.envs.stage1_obs_filter.Stage1ASStateSynthesizer`. For
    every uid in ``inner.action_space`` whose
    ``observation_space["agent"][uid]`` is a :class:`gym.spaces.Dict`
    without a ``"state"`` key, this wrapper concatenates the uid's 1-D
    Box channels (alphabetical order) into a flat ``state`` Box and
    injects it at every step. Uids that already expose ``state`` and
    non-Dict per-uid spaces are passed through unchanged.

    Args:
        env: A multi-agent env with a Dict ``observation_space`` whose
            ``"agent"`` entry is itself a Dict keyed by uid, and whose
            ``action_space`` is a Dict keyed by the same uids.

    Raises:
        TypeError: If ``observation_space`` is not a Dict, or
            ``observation_space["agent"]`` is not a Dict, or
            ``action_space`` is not a Dict.
        ChamberEnvCompatibilityError: If a uid that needs synthesis has
            no 1-D Box channels available — there is nothing to
            concatenate into a flat state vector.
    """

    def __init__(self, env: gym.Env[Any, Any]) -> None:
        """Detect which uids need synthesis at construction (ADR-001 §Validation criteria)."""
        super().__init__(env)
        inner_obs_space = env.observation_space
        if not isinstance(inner_obs_space, gym.spaces.Dict):
            raise TypeError(
                "Stage0StateSynthesizer requires a gym.spaces.Dict observation "
                f"space; got {type(inner_obs_space).__name__}. ADR-001 §Validation criteria."
            )
        inner_action_space = env.action_space
        if not isinstance(inner_action_space, gym.spaces.Dict):
            raise TypeError(
                "Stage0StateSynthesizer requires a gym.spaces.Dict action space; "
                f"got {type(inner_action_space).__name__}. ADR-001 §Validation criteria."
            )
        inner_uids: tuple[str, ...] = tuple(inner_action_space.spaces.keys())
        (
            self.observation_space,
            self._synthesise_state_for,
            self._state_synthesis_keys,
        ) = _build_synthesised_obs_space(inner_obs_space, inner_uids)

    def observation(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        """Inject synthesised ``state`` keys per uid (ADR-001 §Validation criteria)."""
        if not self._synthesise_state_for:
            return dict(observation)
        out: dict[str, Any] = dict(observation)
        agent_in = observation.get("agent", {})
        agent_out: dict[str, Any] = dict(agent_in)
        for uid in self._synthesise_state_for:
            sub_in = agent_in.get(uid, {})
            if not isinstance(sub_in, dict):
                continue
            keys = self._state_synthesis_keys[uid]
            pieces = [np.asarray(sub_in[k], dtype=np.float32).ravel() for k in keys]
            sub_out = dict(sub_in)
            sub_out[_STATE_KEY] = (
                np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
            )
            agent_out[uid] = sub_out
        out["agent"] = agent_out
        return out


def _build_synthesised_obs_space(
    inner_obs_space: gym.spaces.Space[Any],
    inner_uids: tuple[str, ...],
) -> tuple[gym.spaces.Dict, set[str], dict[str, tuple[str, ...]]]:
    """Per-uid obs adaptation: pass-through ``state`` or synthesise it.

    Returns ``(adapted_space, synthesise_for_uids, synthesis_keys_per_uid)``.
    Mirrors the policy in
    :func:`chamber.benchmarks.stage0_smoke_adapter._build_adapted_obs_space`
    so the adapter remains a correct pass-through once this wrapper has
    already supplied ``state`` at the env layer.
    """
    if not isinstance(inner_obs_space, gym.spaces.Dict):
        raise TypeError(
            "Stage0StateSynthesizer requires inner.observation_space to be a "
            f"gym.spaces.Dict; got {type(inner_obs_space).__name__}."
        )
    inner_agent_space = inner_obs_space.spaces["agent"]
    if not isinstance(inner_agent_space, gym.spaces.Dict):
        raise TypeError(
            "Stage0StateSynthesizer requires inner.observation_space['agent'] to be a "
            f"gym.spaces.Dict; got {type(inner_agent_space).__name__}."
        )
    synthesise_for: set[str] = set()
    synthesis_keys: dict[str, tuple[str, ...]] = {}
    adapted_agent: dict[str, gym.spaces.Space[Any]] = {}
    for uid in inner_uids:
        sub_space = inner_agent_space.spaces[uid]
        if not isinstance(sub_space, gym.spaces.Dict):
            adapted_agent[uid] = sub_space
            continue
        if _STATE_KEY in sub_space.spaces:
            adapted_agent[uid] = sub_space
            continue
        synth_box, synth_keys = _synthesised_state_space(sub_space)
        synthesise_for.add(uid)
        synthesis_keys[uid] = synth_keys
        adapted_sub_dict: dict[str, gym.spaces.Space[Any]] = dict(sub_space.spaces)
        adapted_sub_dict[_STATE_KEY] = synth_box
        adapted_agent[uid] = gym.spaces.Dict(adapted_sub_dict)
    adapted_top: dict[str, gym.spaces.Space[Any]] = dict(inner_obs_space.spaces)
    adapted_top["agent"] = gym.spaces.Dict(adapted_agent)
    return gym.spaces.Dict(adapted_top), synthesise_for, synthesis_keys


def _synthesised_state_space(
    sub_space: gym.spaces.Dict,
) -> tuple[gym.spaces.Box, tuple[str, ...]]:
    """Flat state Box from 1-D Box channels of ``sub_space`` (ADR-001 §Validation criteria).

    Concatenates every 1-D :class:`gym.spaces.Box` channel of
    ``sub_space`` in alphabetical key order. Image-shaped channels
    (rank > 1) are deliberately ignored — the trainer's
    :func:`chamber.benchmarks.ego_ppo_trainer._flat_ego_obs` consumes a
    flat vector, and the right place to fold images into a flat
    representation is an explicit encoder, not this wrapper.

    Returns:
        ``(box, keys)`` where ``box`` is the synthesised state space
        and ``keys`` is the alphabetised tuple of source channel names.

    Raises:
        ChamberEnvCompatibilityError: If no 1-D Box channels are
            available — the consumer (trainer or M4 probe) cannot
            proceed without a flat state vector.
    """
    pieces: list[gym.spaces.Box] = []
    keys: list[str] = []
    for key in sorted(sub_space.spaces.keys()):
        sub = sub_space.spaces[key]
        if not isinstance(sub, gym.spaces.Box) or sub.shape is None:
            continue
        if len(sub.shape) == 1:
            pieces.append(sub)
            keys.append(key)
        elif len(sub.shape) == _VECTORISED_RANK and sub.shape[0] == 1:
            squeezed = gym.spaces.Box(
                low=np.asarray(sub.low[0], dtype=np.float32),
                high=np.asarray(sub.high[0], dtype=np.float32),
                dtype=np.float32,
            )
            pieces.append(squeezed)
            keys.append(key)
    if not pieces:
        raise ChamberEnvCompatibilityError(
            "Stage0StateSynthesizer: inner env exposes no 1-D Box channels under "
            "obs['agent'][uid]; cannot synthesise a flat state vector. Adjust the "
            "chamber.benchmarks.stage0_smoke wrapper chain (e.g. include 'state' "
            "or 'joint_pos' in _SMOKE_KEEP for the uid) or pick a different ego "
            "uid (ADR-001 §Validation criteria)."
        )
    low = np.concatenate([np.asarray(p.low, dtype=np.float32).ravel() for p in pieces])
    high = np.concatenate([np.asarray(p.high, dtype=np.float32).ravel() for p in pieces])
    return gym.spaces.Box(low=low, high=high, dtype=np.float32), tuple(keys)


__all__ = ["Stage0StateSynthesizer"]
