# SPDX-License-Identifier: Apache-2.0
"""Per-agent observation-channel filter.

ADR-001 §Decision item 1 (obs modality heterogeneity axis);
ADR-007 §Stage 1 OM axis (rev 3 §Stage 1).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import gymnasium as gym
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping


class TextureFilterObsWrapper(gym.ObservationWrapper):  # type: ignore[type-arg]
    """Filter observation channels per agent uid, zero-masking unlisted channels.

    Tensor shapes are preserved — masked channels become zero arrays of the
    same shape and dtype. This keeps downstream policy networks oblivious to
    which modalities a given partner actually receives.

    ADR-001 §Decision item 1; ADR-007 §Stage 1 OM axis.

    Args:
        env: Env whose observation has shape ``{"agent": {uid: {channel: array}}}``.
        keep_per_agent: Mapping from agent uid to the iterable of channel keys
            to keep. Channels not listed are zero-masked. Uids absent from this
            mapping are passed through unchanged.
    """

    def __init__(self, env: gym.Env, keep_per_agent: Mapping[str, Iterable[str]]) -> None:  # type: ignore[type-arg]
        """Store per-agent channel keep-sets as frozensets for O(1) lookup."""
        super().__init__(env)
        self._keep: dict[str, frozenset[str]] = {
            uid: frozenset(keys) for uid, keys in keep_per_agent.items()
        }

    def observation(self, observation: dict) -> dict:  # type: ignore[override]
        """Apply per-agent channel mask, zero-filling unlisted channels.

        ADR-001 §Decision item 1: obs modality is a controllable heterogeneity
        axis; shapes are invariant so vector rollout is unaffected.
        """
        agent_obs = observation.get("agent")
        if not isinstance(agent_obs, dict):
            return observation
        filtered_agents: dict[str, dict] = {}
        for uid, per_agent in agent_obs.items():
            if uid not in self._keep or not isinstance(per_agent, dict):
                filtered_agents[uid] = per_agent
                continue
            keep = self._keep[uid]
            filtered_agents[uid] = {
                ch: val if ch in keep else np.zeros_like(val) for ch, val in per_agent.items()
            }
        return {**observation, "agent": filtered_agents}
