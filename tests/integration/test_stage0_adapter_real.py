# SPDX-License-Identifier: Apache-2.0
"""Tier-2 real-SAPIEN smoke for the Stage-0 training adapter (T4b.3).

Mirrors :class:`tests.integration.test_stage0_smoke.TestRealManiSkillSmoke`'s
two-tier pattern. The whole module is gated on
:func:`chamber.utils.device.sapien_gpu_available` and skips on
Vulkan-less hosts; on a GPU host it drives 100 env steps through the
training adapter and pins the contract that ``obs["agent"]`` +
``obs["comm"]`` keys survive the wrapping (ADR-001 conditions a + b).

The adapter's structural / pure-Python paths are covered by the Tier-1
tests in ``test_stage0_adapter_fake.py`` (always runs on CPU). This
file's only job is to verify the *real* Stage-0 env composes with the
training adapter end-to-end on a GPU host.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

from chamber.benchmarks.stage0_smoke_adapter import make_stage0_training_env
from chamber.utils.device import sapien_gpu_available

_N_STEPS: int = 100
_TRAINING_AGENT_UIDS: tuple[str, str] = ("panda_wristcam", "fetch")


@pytest.mark.smoke
@pytest.mark.gpu
@pytest.mark.skipif(
    not sapien_gpu_available(),
    reason="Requires Vulkan/GPU (SAPIEN); skipped on CPU-only machines",
)
class TestRealStage0Adapter:
    def test_returns_envlike_and_steps_for_100_ticks(self) -> None:
        """ADR-001 conds a+b: 100 steps through the adapter preserve agent + comm keys."""
        env = make_stage0_training_env(
            agent_uids=_TRAINING_AGENT_UIDS,
            episode_length=_N_STEPS,
            root_seed=0,
        )
        obs, _ = env.reset(seed=0)
        # ADR-001 cond.a: agent obs is namespaced; the adapter exposes
        # the training-side ego + partner uids.
        assert "agent" in obs
        assert _TRAINING_AGENT_UIDS[0] in obs["agent"]
        # ADR-001 cond.b: comm channel survives the wrapping.
        assert "comm" in obs
        # Sample actions from the adapter's exposed action_space (only
        # ego + partner) and step; the adapter injects zero actions for
        # the third Stage-0 robot.
        action_space = env.action_space  # type: ignore[attr-defined]
        assert isinstance(action_space, gym.spaces.Dict)
        action_template = {uid: action_space.spaces[uid].sample() for uid in _TRAINING_AGENT_UIDS}
        last_obs = obs
        for _ in range(_N_STEPS):
            last_obs, _r, _term, _trunc, _info = env.step(action_template)
        assert "agent" in last_obs
        assert "comm" in last_obs

    def test_ego_state_is_a_1d_float_vector(self) -> None:
        """ADR-002 §Decisions: the trainer reads obs['agent'][ego_uid]['state'] as 1-D float."""
        env = make_stage0_training_env(
            agent_uids=_TRAINING_AGENT_UIDS,
            episode_length=10,
            root_seed=0,
        )
        obs, _ = env.reset(seed=0)
        ego_state = obs["agent"][_TRAINING_AGENT_UIDS[0]]["state"]
        arr = np.asarray(ego_state)
        assert arr.ndim == 1
        assert np.issubdtype(arr.dtype, np.floating)
