# SPDX-License-Identifier: Apache-2.0
"""Communication-channel observation injector (M1 carrier; M2 fills logic).

ADR-001 §Decision item 3 (comm shaping is the outermost wrapper);
ADR-003 §Decision (fixed-format + optional learned overlay via CommChannel);
ADR-006 §Decision (URLLC-anchored numeric bounds enforced by the channel impl).

The actual comm logic (fixed-format encoding, AoI accounting, optional learned
overlay, URLLC-anchored degradation) lives in :mod:`chamber.comm` (M2).
This wrapper is a thin adapter that places the comm output into ``obs["comm"]``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import gymnasium as gym

from chamber.envs.errors import ChamberEnvCompatibilityError

if TYPE_CHECKING:
    from chamber.comm.api import CommChannel


class CommShapingWrapper(gym.Wrapper):  # type: ignore[type-arg]
    """Inject comm-channel output into the observation dict as ``obs["comm"]``.

    Wraps any env whose observation space is a ``gym.spaces.Dict``. On each
    reset and step, calls ``channel.encode(obs)`` and stores the result under
    the ``"comm"`` key. The channel itself controls all degradation semantics
    (latency, drop, jitter); this wrapper is a pure carrier.

    ADR-001 §Decision item 3; ADR-003 §Decision; ADR-006 §Decision.

    Args:
        env: Env with a ``gym.spaces.Dict`` observation space.
        channel: A :class:`chamber.comm.api.CommChannel` instance. In M1 this
            is a stub :class:`chamber.comm.FixedFormatCommChannel` that emits
            an empty packet shell. M2 fills the real fixed-format encoding.

    Raises:
        ChamberEnvCompatibilityError: if the inner observation space is not a
            ``gym.spaces.Dict`` (ADR-001 §Risks).
    """

    def __init__(self, env: gym.Env, channel: CommChannel) -> None:  # type: ignore[type-arg]
        """Validate the obs space and extend it with a ``comm`` sub-space."""
        super().__init__(env)
        if not isinstance(env.observation_space, gym.spaces.Dict):
            raise ChamberEnvCompatibilityError(
                f"CommShapingWrapper requires a gym.spaces.Dict observation space; "
                f"got {type(env.observation_space).__name__}. See ADR-001 §Risks."
            )
        self._channel = channel
        self._extend_observation_space()

    def _extend_observation_space(self) -> None:
        self.observation_space = gym.spaces.Dict(
            {
                **self.env.observation_space.spaces,  # type: ignore[attr-defined]
                "comm": gym.spaces.Dict({}),  # M2 fills the real comm obs space.
            }
        )

    def reset(self, **kwargs: object) -> tuple[dict, dict]:  # type: ignore[override]
        """Reset the env and channel, injecting comm into the initial obs.

        ADR-001 §Decision item 3; ADR-003 §Decision.
        """
        obs, info = self.env.reset(**kwargs)  # type: ignore[arg-type]
        self._channel.reset()
        obs = dict(obs)
        obs["comm"] = self._channel.encode(obs)
        return obs, info

    def step(self, action: object) -> tuple[dict, object, bool, bool, dict]:  # type: ignore[override]
        """Step the env and inject updated comm into obs.

        ADR-001 §Decision item 3; ADR-003 §Decision.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = dict(obs)
        obs["comm"] = self._channel.encode(obs)
        return obs, reward, terminated, truncated, info
