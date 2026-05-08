# SPDX-License-Identifier: Apache-2.0
"""Lightweight fake environments for unit and property tests (no ManiSkill/Vulkan needed)."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np


class FakeMultiAgentEnv(gym.Env):  # type: ignore[type-arg]
    """Minimal multi-agent env for wrapper unit/property tests.

    Action space : ``gym.spaces.Dict`` keyed by agent uid with small Box spaces.
    Observation  : Dict with ``"agent"`` sub-dict; each agent has channels
                   ``"state"``, ``"rgb"``, ``"depth"`` as small Box arrays.
    """

    def __init__(self, agent_uids: tuple[str, ...] = ("a", "b")) -> None:
        super().__init__()
        self._uids = agent_uids
        self.action_space = gym.spaces.Dict(
            {
                uid: gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
                for uid in agent_uids
            }
        )
        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Dict(
                    {
                        uid: gym.spaces.Dict(
                            {
                                "state": gym.spaces.Box(0.0, 1.0, (3,), dtype=np.float32),
                                "rgb": gym.spaces.Box(0.0, 1.0, (4, 4, 3), dtype=np.float32),
                                "depth": gym.spaces.Box(0.0, 1.0, (4, 4, 1), dtype=np.float32),
                            }
                        )
                        for uid in agent_uids
                    }
                )
            }
        )
        self._actions_received: list[dict[str, Any]] = []

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]:
        super().reset(seed=seed)
        self._actions_received = []
        obs = {
            "agent": {
                uid: {
                    "state": np.zeros(3, dtype=np.float32),
                    "rgb": np.zeros((4, 4, 3), dtype=np.float32),
                    "depth": np.zeros((4, 4, 1), dtype=np.float32),
                }
                for uid in self._uids
            }
        }
        return obs, {}

    def step(self, action: dict) -> tuple[dict, float, bool, bool, dict]:
        self._actions_received.append(dict(action))
        obs = {
            "agent": {
                uid: {
                    "state": np.ones(3, dtype=np.float32),
                    "rgb": np.ones((4, 4, 3), dtype=np.float32),
                    "depth": np.ones((4, 4, 1), dtype=np.float32),
                }
                for uid in self._uids
            }
        }
        return obs, 0.0, False, False, {}
