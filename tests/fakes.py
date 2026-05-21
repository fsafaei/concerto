# SPDX-License-Identifier: Apache-2.0
"""Lightweight fake environments for unit and property tests (no ManiSkill/Vulkan needed)."""

from __future__ import annotations

from typing import Any, ClassVar

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


class FakeStage1PickPlaceObs(gym.Env):  # type: ignore[type-arg]
    """Tier-1 fake mirroring ``Stage1PickPlaceEnv``'s AS-hetero obs/action contract.

    Used by ``tests/integration/test_trainer_obs_reader_contract.py`` to
    pin the trainer-obs-reader contract identified by PR #182 Surface 6
    without requiring SAPIEN. The fake reproduces the structural facts
    the audit recorded at
    ``spikes/results/stage1-failure-investigation/2026-05-20/surface_6_obs_contract_audit.txt``:

    - ``obs["agent"]["panda_wristcam"]`` carries ``qpos`` / ``qvel``
      Boxes of shape ``(1, 9)``.
    - ``obs["agent"]["fetch"]`` carries ``qpos`` / ``qvel`` Boxes of
      shape ``(1, 15)``.
    - ``obs["extra"]`` carries ``cube_pose`` ``(1, 7)``, ``goal_pos``
      ``(1, 3)``, and ``tcp_pose`` ``(1, 7)``.
    - ``condition_id`` is the AS-hetero pre-registration string so
      :class:`chamber.envs.stage1_obs_filter.Stage1ASStateSynthesizer`
      activates and injects a ``state`` key under each agent uid.

    The fake is deliberately minimal — no reward, no termination logic,
    no SAPIEN-ish kinematics. The contract under test is the static
    obs-space shape that the trainer reads at construction time
    (``EgoPPOTrainer.from_config`` reads
    ``env.observation_space["agent"][ego_uid]["state"].shape[0]``), so a
    static fake suffices.

    The default ``condition_id`` is the AS-hetero entry from
    ``chamber.envs.stage1_pickplace._CONDITION_TABLE`` because the
    Surface 6 audit captured the hetero case (the homo case has
    structurally identical wire shape for the ego; hetero is the
    higher-bar pin).
    """

    metadata: ClassVar[dict[str, object]] = {"render_modes": []}  # type: ignore[misc]

    AS_HETERO_CONDITION_ID = "stage1_pickplace_panda_plus_fetch_ego_aht_happo_per_agent"

    def __init__(
        self,
        condition_id: str = AS_HETERO_CONDITION_ID,
        ego_uid: str = "panda_wristcam",
        partner_uid: str = "fetch",
        ego_qpos_dim: int = 9,
        partner_qpos_dim: int = 15,
    ) -> None:
        super().__init__()
        self.condition_id = condition_id
        self._ego_uid = ego_uid
        self._partner_uid = partner_uid
        self._ego_qpos_dim = ego_qpos_dim
        self._partner_qpos_dim = partner_qpos_dim
        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Dict(
                    {
                        ego_uid: gym.spaces.Dict(
                            {
                                "qpos": gym.spaces.Box(
                                    -np.inf, np.inf, (1, ego_qpos_dim), np.float32
                                ),
                                "qvel": gym.spaces.Box(
                                    -np.inf, np.inf, (1, ego_qpos_dim), np.float32
                                ),
                            }
                        ),
                        partner_uid: gym.spaces.Dict(
                            {
                                "qpos": gym.spaces.Box(
                                    -np.inf, np.inf, (1, partner_qpos_dim), np.float32
                                ),
                                "qvel": gym.spaces.Box(
                                    -np.inf, np.inf, (1, partner_qpos_dim), np.float32
                                ),
                            }
                        ),
                    }
                ),
                "extra": gym.spaces.Dict(
                    {
                        "cube_pose": gym.spaces.Box(-np.inf, np.inf, (1, 7), np.float32),
                        "goal_pos": gym.spaces.Box(-np.inf, np.inf, (1, 3), np.float32),
                        "tcp_pose": gym.spaces.Box(-np.inf, np.inf, (1, 7), np.float32),
                    }
                ),
            }
        )
        # AS-hetero action dims at the SAPIEN layer: panda_wristcam 8-D
        # (7 arm + 1 mimic-gripper) under pd_joint_delta_pos; fetch 13-D.
        self.action_space = gym.spaces.Dict(
            {
                ego_uid: gym.spaces.Box(-1.0, 1.0, (8,), np.float32),
                partner_uid: gym.spaces.Box(-1.0, 1.0, (13,), np.float32),
            }
        )

    def _sample_obs(self) -> dict[str, Any]:
        return {
            "agent": {
                self._ego_uid: {
                    "qpos": np.zeros((1, self._ego_qpos_dim), dtype=np.float32),
                    "qvel": np.zeros((1, self._ego_qpos_dim), dtype=np.float32),
                },
                self._partner_uid: {
                    "qpos": np.zeros((1, self._partner_qpos_dim), dtype=np.float32),
                    "qvel": np.zeros((1, self._partner_qpos_dim), dtype=np.float32),
                },
            },
            "extra": {
                "cube_pose": np.zeros((1, 7), dtype=np.float32),
                "goal_pos": np.zeros((1, 3), dtype=np.float32),
                "tcp_pose": np.zeros((1, 7), dtype=np.float32),
            },
        }

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]:
        super().reset(seed=seed)
        return self._sample_obs(), {}

    def step(self, action: dict) -> tuple[dict, float, bool, bool, dict]:
        return self._sample_obs(), 0.0, False, False, {}
