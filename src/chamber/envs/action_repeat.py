# SPDX-License-Identifier: Apache-2.0
"""Per-agent action-repeat wrapper.

ADR-001 §Decision item 2; ADR-007 Stage 2 CR axis (rev 3 §Stage 2).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import gymnasium as gym

if TYPE_CHECKING:
    from collections.abc import Mapping

from chamber.envs.errors import ChamberEnvCompatibilityError


class PerAgentActionRepeatWrapper(gym.Wrapper):  # type: ignore[type-arg]
    """Apply different action-repeat counts per agent uid.

    Effective control frequency for agent ``a`` is::

        base_freq / action_repeat[a]

    At each ``env.step`` call, an agent whose repeat counter has not elapsed
    re-applies its most recent action instead of the newly submitted one.

    ADR-001 §Decision item 2; ADR-007 Stage 2 CR axis (rev 3 §Stage 2).

    Args:
        env: Env exposing a ``gym.spaces.Dict`` action space keyed by agent uid.
        action_repeat: Mapping from agent uid to a positive integer repeat count.
            Keys absent from this mapping default to repeat=1 (pass-through).

    Raises:
        ChamberEnvCompatibilityError: if the action space is not a
            ``gym.spaces.Dict`` keyed by agent uid (ADR-001 §Risks).
        ValueError: if any repeat count is less than 1.
    """

    def __init__(self, env: gym.Env, action_repeat: Mapping[str, int]) -> None:  # type: ignore[type-arg]
        """Validate the action space and store per-agent repeat counts."""
        super().__init__(env)
        if not isinstance(env.action_space, gym.spaces.Dict):
            raise ChamberEnvCompatibilityError(
                f"PerAgentActionRepeatWrapper requires a gym.spaces.Dict action "
                f"space keyed by agent uid; got {type(env.action_space).__name__}. "
                f"See ADR-001 §Risks."
            )
        for uid, count in action_repeat.items():
            if count < 1:
                raise ValueError(f"action_repeat[{uid!r}] must be ≥ 1, got {count}")
        self._action_repeat: dict[str, int] = dict(action_repeat)
        self._counters: dict[str, int] = dict.fromkeys(action_repeat, 0)
        self._last_action: dict[str, object] = {}

    def step(self, action: dict) -> tuple:  # type: ignore[override]
        """Step with per-agent action repeat applied.

        ADR-001 §Validation criteria condition (c): the slow agent's effective
        action update interval matches 1/rate within one env tick.
        """
        resolved: dict[str, object] = {}
        for uid, act in action.items():
            repeat = self._action_repeat.get(uid, 1)
            counter = self._counters.get(uid, 0)
            if counter == 0:
                resolved[uid] = act
                self._last_action[uid] = act
                self._counters[uid] = repeat - 1
            else:
                resolved[uid] = self._last_action[uid]
                self._counters[uid] = counter - 1
        return self.env.step(resolved)

    def reset(self, **kwargs: object) -> tuple:  # type: ignore[override]
        """Reset the env and clear all per-agent repeat state.

        ADR-001 §Decision item 2: no state leaks across episodes.
        """
        obs, info = self.env.reset(**kwargs)  # type: ignore[arg-type]
        self._counters = dict.fromkeys(self._action_repeat, 0)
        self._last_action = {}
        return obs, info
