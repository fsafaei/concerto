# SPDX-License-Identifier: Apache-2.0
"""Stage-1 OM-axis observation-channel filter (ADR-007 §Stage 1b).

Thin wrapper that reads the inner env's :attr:`condition_id` and applies
the per-condition channel keep-set to the observation dict. The Stage-1
AS conditions don't filter at all (they use ``obs_mode="state_dict"``
which already excludes the camera data the OM axis differentiates on);
the wrapper passes those conditions through untouched.

For the two OM conditions:

- ``stage1_pickplace_vision_only`` — keep ``obs["sensor_data"]`` (the
  RGB-D camera channels) plus the task-extra fields the policy needs
  to be functional at the Stage-1b episode budget (``tcp_pose``,
  ``goal_pos``). **Zero-mask** the agent-proprio sub-dicts under
  ``obs["agent"][uid]`` and the synthesised ``obs["extra"]["force_torque"]``
  channel.
- ``stage1_pickplace_vision_plus_force_torque_plus_proprio`` —
  pass-through (no filtering); the policy sees everything.

ManiSkill v3 obs layout reminder (``mani_skill/envs/sapien_env.py:546-630``):
``obs["agent"][uid]`` carries proprio (state, joint_pos, joint_vel);
``obs["extra"]`` carries task-level fields from
:meth:`_get_obs_extra`; ``obs["sensor_data"]`` carries camera channels
when the env's ``obs_mode`` is image-bearing.

Shape preservation: zero-masked channels become zero arrays of the same
shape and dtype (mirrors
:class:`chamber.envs.texture_filter.TextureFilterObsWrapper`'s pattern).
This keeps downstream policy networks oblivious to which modalities a
given uid actually receives at runtime — only the *values* change
across conditions, not the *shapes*.

References:
- ADR-007 §Stage 1b (OM axis spec); ``spikes/preregistration/OM.yaml``
  (the condition_id strings this wrapper resolves).
- :class:`chamber.envs.texture_filter.TextureFilterObsWrapper` (the
  parallel pattern for the Stage-0 smoke env's per-uid keep-set).
- :class:`chamber.envs.stage1_pickplace.Stage1PickPlaceEnv` (the
  inner env that supplies the synthesised FT channel and the
  ``condition_id`` attribute).
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

#: Channels under ``obs["extra"]`` the OM-vision-only condition keeps.
#:
#: The policy needs at least *some* task-relevant signal beyond the
#: RGB-D camera to be solvable inside the Stage-1b episode budget
#: (100 steps); ``tcp_pose`` + ``goal_pos`` is the minimum "where is
#: the gripper" + "where is the target" pair. Cube pose / cube-relative
#: deltas / FT are zero-masked. If the AS-hetero pilot shows the gap
#: vs vision_only is artefact-of-this-keep-set, tighten or loosen here
#: in a follow-up patch — the keep-set is design-tunable inside the
#: condition_id's spec since the prereg only names the *modality
#: families* (vision vs vision+ft+proprio), not the per-field keep-set
#: (ADR-007 §Discipline).
_OM_VISION_ONLY_EXTRA_KEEP: frozenset[str] = frozenset({"tcp_pose", "goal_pos"})

#: Condition_ids the wrapper actually mutates obs for. Conditions
#: outside this set pass through untouched.
_FILTERED_CONDITIONS: frozenset[str] = frozenset({"stage1_pickplace_vision_only"})


class Stage1OMChannelFilter(gym.ObservationWrapper):  # type: ignore[type-arg]
    """OM-axis per-condition channel filter (ADR-007 §Stage 1b).

    Reads the inner env's ``condition_id`` at construction time and
    captures the resulting filter policy. The wrapper does not re-read
    ``condition_id`` on every step — Stage-1b envs are built once per
    ``(seed, condition)`` cell (see ``stage1_as.py:253-275``) and the
    condition is fixed for the env's lifetime.

    Args:
        env: Inner env exposing a ``condition_id`` attribute. The
            :class:`chamber.envs.stage1_pickplace.Stage1PickPlaceEnv`
            class satisfies this; the wrapper falls back to a
            pass-through identity if ``condition_id`` is missing or
            not one of the two OM strings.

    Raises:
        TypeError: If the inner env's observation space is not a
            :class:`gym.spaces.Dict` (the wrapper relies on dict-shaped
            obs).
    """

    def __init__(self, env: gym.Env[Any, Any]) -> None:
        """Store the per-condition filter policy (ADR-007 §Stage 1b)."""
        super().__init__(env)
        if not isinstance(env.observation_space, gym.spaces.Dict):
            msg = (
                "Stage1OMChannelFilter requires a gym.spaces.Dict observation "
                f"space; got {type(env.observation_space).__name__}."
            )
            raise TypeError(msg)
        # Read condition_id once at construction; the env-per-cell
        # cadence makes this safe and saves a hot-path attr lookup.
        condition_id = getattr(env, "condition_id", None)
        if condition_id is None and hasattr(env, "get_wrapper_attr"):
            try:
                condition_id = env.get_wrapper_attr("condition_id")
            except AttributeError:
                condition_id = None
        self._condition_id: str | None = condition_id
        self._filter_active: bool = condition_id in _FILTERED_CONDITIONS

    @property
    def condition_id(self) -> str | None:
        """ADR-007 §Stage 1b OM axis: the condition_id resolved at construction."""
        return self._condition_id

    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:  # type: ignore[override]
        """Apply the per-condition channel mask (ADR-007 §Stage 1b).

        OM-vision-only zero-masks the per-uid agent-proprio sub-dicts
        and the ``obs["extra"]["force_torque"]`` channel; keeps
        ``obs["extra"]["tcp_pose"]`` + ``obs["extra"]["goal_pos"]``
        (the minimal task-relevant fields) plus ``obs["sensor_data"]``
        (RGB-D camera) untouched. Other conditions pass through.
        """
        if not self._filter_active:
            return observation
        out = dict(observation)
        # Zero-mask per-uid agent proprio (the "no proprioception" half
        # of the vision-only condition_id semantics).
        agent = observation.get("agent")
        if isinstance(agent, dict):
            zeroed_agent: dict[str, Any] = {}
            for uid, sub in agent.items():
                if isinstance(sub, dict):
                    zeroed_agent[uid] = {
                        ch: np.zeros_like(val) if hasattr(val, "shape") else val
                        for ch, val in sub.items()
                    }
                else:
                    zeroed_agent[uid] = sub
            out["agent"] = zeroed_agent
        # Zero-mask task extras NOT in the vision-only keep-set (the
        # "no force-torque" half of the vision-only semantics).
        extra = observation.get("extra")
        if isinstance(extra, dict):
            filtered_extra: dict[str, Any] = {}
            for ch, val in extra.items():
                if ch in _OM_VISION_ONLY_EXTRA_KEEP:
                    filtered_extra[ch] = val
                elif hasattr(val, "shape"):
                    filtered_extra[ch] = np.zeros_like(val)
                else:
                    filtered_extra[ch] = val
            out["extra"] = filtered_extra
        return out


__all__ = ["Stage1OMChannelFilter"]
