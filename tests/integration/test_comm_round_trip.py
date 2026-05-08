# SPDX-License-Identifier: Apache-2.0
"""End-to-end comm round-trip (T2.8).

Builds a 3-robot env, wraps it with the full M2 stack
(``CommShapingWrapper(CommDegradationWrapper(FixedFormatCommChannel,
URLLC_3GPP_R17["factory"]))``), steps 1 000 times, and asserts the
schema-versioned packet invariants and the AoI envelope from
ADR-003 §Decision and ADR-006 §Decision.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from chamber.comm import (
    SCHEMA_VERSION,
    URLLC_3GPP_R17,
    CommDegradationWrapper,
    FixedFormatCommChannel,
    Pose,
)
from chamber.envs.comm_shaping import CommShapingWrapper

_ROBOT_UIDS: tuple[str, ...] = ("panda_wristcam", "fetch", "allegro_hand_right")
_N_STEPS = 1000


class _ThreeRobotPoseEnv(gym.Env):  # type: ignore[type-arg]
    """In-test env emitting per-uid pose for the comm channel (T2.8).

    ADR-003 §Decision: the channel reads ``obs["pose"][uid]`` as the fresh
    sample. This env synthesises a tiny, deterministic pose sequence so the
    integration test can exercise the encoder + degradation pipeline end-to-end
    without depending on Vulkan/SAPIEN.
    """

    def __init__(self) -> None:
        super().__init__()
        self._tick: int = 0
        self.action_space = gym.spaces.Dict(
            {
                uid: gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
                for uid in _ROBOT_UIDS
            }
        )
        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Dict(
                    {
                        uid: gym.spaces.Dict(
                            {"state": gym.spaces.Box(0.0, 1.0, (3,), dtype=np.float32)}
                        )
                        for uid in _ROBOT_UIDS
                    }
                )
            }
        )

    def _emit(self) -> dict[str, Any]:
        agents = {
            uid: {"state": np.full(3, fill_value=float(self._tick), dtype=np.float32)}
            for uid in _ROBOT_UIDS
        }
        poses = {
            uid: Pose(
                xyz=(float(self._tick), 0.0, 0.0),
                quat_wxyz=(1.0, 0.0, 0.0, 0.0),
            )
            for uid in _ROBOT_UIDS
        }
        return {"agent": agents, "pose": poses}

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        del options
        super().reset(seed=seed)
        self._tick = 0
        return self._emit(), {}

    def step(
        self, action: dict[str, Any]
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        del action
        self._tick += 1
        return self._emit(), 0.0, False, False, {}


def _make_pipeline() -> CommShapingWrapper:
    """Compose the canonical M2 wrapper stack used by Stage-0 smoke (T2.9)."""
    inner = _ThreeRobotPoseEnv()
    channel = CommDegradationWrapper(
        FixedFormatCommChannel(),
        URLLC_3GPP_R17["factory"],
        tick_period_ms=1.0,
        root_seed=0,
    )
    return CommShapingWrapper(inner, channel=channel)


class TestSchemaInvariants:
    def test_packet_keys_stable_for_1000_steps(self) -> None:
        """ADR-003 §Decision: every packet carries the five required fields."""
        wrapper = _make_pipeline()
        wrapper.reset(seed=0)
        for _ in range(_N_STEPS):
            obs, *_ = wrapper.step({uid: np.zeros(2, dtype=np.float32) for uid in _ROBOT_UIDS})
            packet = obs["comm"]
            assert set(packet.keys()) == {
                "schema_version",
                "pose",
                "task_state",
                "aoi",
                "learned_overlay",
            }
            assert packet["schema_version"] == SCHEMA_VERSION

    def test_pose_uids_match_env_uids(self) -> None:
        """ADR-003 §Decision: the channel forwards every env-supplied uid."""
        wrapper = _make_pipeline()
        wrapper.reset(seed=0)
        observed_pose_uids: set[str] = set()
        for _ in range(_N_STEPS):
            obs, *_ = wrapper.step({uid: np.zeros(2, dtype=np.float32) for uid in _ROBOT_UIDS})
            observed_pose_uids.update(obs["comm"]["pose"].keys())
        assert observed_pose_uids == set(_ROBOT_UIDS)


class TestAoIEnvelope:
    def test_aoi_bounded_by_latency_envelope(self) -> None:
        """T2.8 acceptance: AoI <= latency + 1/freq across the 1 000-step trace.

        The ``factory`` profile has latency_mean_ms=5.0, latency_std_ms=0.1,
        clipped to [0, 10]. With drop_rate=1e-4 there are essentially no drops
        in 1 000 ticks. ``1/freq`` is one env tick (1 ms here), so the AoI of
        any uid in the visible packet should never exceed 10 + 1 = 11 ticks
        worth (we add headroom for the very first warm-up cycles).
        """
        wrapper = _make_pipeline()
        wrapper.reset(seed=0)
        max_aoi: float = 0.0
        for _ in range(_N_STEPS):
            obs, *_ = wrapper.step({uid: np.zeros(2, dtype=np.float32) for uid in _ROBOT_UIDS})
            for value in obs["comm"]["aoi"].values():
                max_aoi = max(max_aoi, value)
        # factory clip-upper = 2 * 5 = 10 ms; +1 tick env-step margin; +1 tick warm-up.
        envelope = 10.0 + 1.0 + 1.0
        assert max_aoi <= envelope, f"AoI exceeded envelope: max={max_aoi}, bound={envelope}"

    def test_aoi_non_negative(self) -> None:
        """Plan/02 §5: AoI never goes negative under any profile (ADR-003 §Decision)."""
        wrapper = _make_pipeline()
        wrapper.reset(seed=0)
        for _ in range(_N_STEPS):
            obs, *_ = wrapper.step({uid: np.zeros(2, dtype=np.float32) for uid in _ROBOT_UIDS})
            for value in obs["comm"]["aoi"].values():
                assert value >= 0.0


class TestDeterminism:
    def test_same_seed_reproduces_packet_trace(self) -> None:
        """P6: two pipelines reset with the same seed produce identical packet traces."""
        action = {uid: np.zeros(2, dtype=np.float32) for uid in _ROBOT_UIDS}
        a = _make_pipeline()
        b = _make_pipeline()
        a.reset(seed=42)
        b.reset(seed=42)
        for _ in range(200):
            obs_a, *_ = a.step(action)
            obs_b, *_ = b.step(action)
            assert obs_a["comm"] == obs_b["comm"]
