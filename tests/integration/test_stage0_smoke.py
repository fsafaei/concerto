# SPDX-License-Identifier: Apache-2.0
"""Integration tests for the Stage-0 smoke scenario (ADR-001, ADR-007 §Stage 0).

Two test tiers:

1. **Wrapper-structure tests** (no ManiSkill/Vulkan needed) — use a
   :class:`~tests.fakes.FakeMultiAgentEnv` with the exact same 3-robot uid
   tuple as ADR-001 and assert conditions (a), (b), (c) against the wrapper
   chain. These always run in CI.

2. **Real ManiSkill smoke tests** (``@pytest.mark.smoke``) — call
   :func:`~chamber.benchmarks.stage0_smoke.make_stage0_env` which requires a
   Vulkan-capable GPU. Skipped automatically when SAPIEN cannot initialise.
"""

from __future__ import annotations

import numpy as np
import pytest

from chamber.benchmarks.stage0_smoke import (
    _SMOKE_ACTION_REPEAT,
    _SMOKE_KEEP,
    _SMOKE_ROBOT_UIDS,
    make_stage0_env,
)
from chamber.comm import (
    URLLC_3GPP_R17,
    CommDegradationWrapper,
    FixedFormatCommChannel,
)
from chamber.envs import (
    CommShapingWrapper,
    PerAgentActionRepeatWrapper,
    TextureFilterObsWrapper,
)
from chamber.envs.errors import ChamberEnvCompatibilityError
from chamber.utils.device import sapien_gpu_available
from tests.fakes import FakeMultiAgentEnv

_N_STEPS = 100


# ---------------------------------------------------------------------------
# Tier 1: wrapper-structure tests (FakeMultiAgentEnv, no Vulkan)
# ---------------------------------------------------------------------------


def _make_wrapped_fake() -> tuple[CommShapingWrapper, FakeMultiAgentEnv]:
    inner = FakeMultiAgentEnv(agent_uids=_SMOKE_ROBOT_UIDS)
    env = PerAgentActionRepeatWrapper(inner, action_repeat=_SMOKE_ACTION_REPEAT)
    env = TextureFilterObsWrapper(env, keep_per_agent=_SMOKE_KEEP)
    channel = CommDegradationWrapper(
        FixedFormatCommChannel(),
        URLLC_3GPP_R17["factory"],
        tick_period_ms=1.0,
        root_seed=0,
    )
    wrapped = CommShapingWrapper(env, channel=channel)
    return wrapped, inner


class TestADR001CondAFake:
    def test_pass_a_three_agent_namespaced_obs(self) -> None:
        """PASS - ADR-001 cond.a: obs['agent'] has three independently-namespaced agent dicts."""
        wrapped, _ = _make_wrapped_fake()
        obs, _ = wrapped.reset(seed=0)
        assert "agent" in obs, "obs missing 'agent' key"
        agent_obs = obs["agent"]
        assert isinstance(agent_obs, dict), "'agent' obs is not a dict"
        for uid in _SMOKE_ROBOT_UIDS:
            assert uid in agent_obs, f"uid {uid!r} missing from obs['agent']"
        # Namespacing: each uid maps to its own sub-dict, not a shared one.
        ids = [id(agent_obs[uid]) for uid in _SMOKE_ROBOT_UIDS]
        assert len(set(ids)) == len(ids), "agent obs dicts are not independent"


class TestADR001CondBFake:
    def test_pass_b_comm_channel_present(self) -> None:
        """PASS — ADR-001 cond.b: obs['comm'] contains the shaped channel."""
        wrapped, _ = _make_wrapped_fake()
        obs, _ = wrapped.reset(seed=0)
        assert "comm" in obs, "obs missing 'comm' key after reset()"
        obs2, *_ = wrapped.step({uid: np.zeros(2, dtype=np.float32) for uid in _SMOKE_ROBOT_UIDS})
        assert "comm" in obs2, "obs missing 'comm' key after step()"


class TestADR001CondCFake:
    def test_pass_c_slow_agent_action_update_interval(self) -> None:
        """PASS — ADR-001 cond.c: slow agent's effective action update interval matches 1/rate."""
        wrapped, inner = _make_wrapped_fake()
        wrapped.reset(seed=0)
        # panda repeat=5: at 100 Hz, panda updates every 5 env ticks.
        panda_uid = "panda_wristcam"
        for i in range(_N_STEPS):
            action = {uid: np.array([float(i), 0.0], dtype=np.float32) for uid in _SMOKE_ROBOT_UIDS}
            wrapped.step(action)
        received = [r[panda_uid][0] for r in inner._actions_received]
        transitions = 1 + sum(received[i] != received[i - 1] for i in range(1, _N_STEPS))
        expected = _N_STEPS // _SMOKE_ACTION_REPEAT[panda_uid]
        assert transitions == expected, (
            f"FAIL — ADR-001 cond.c: panda action updated {transitions}x, "
            f"expected {expected}x (repeat={_SMOKE_ACTION_REPEAT[panda_uid]})"
        )


# ---------------------------------------------------------------------------
# Tier 2: real ManiSkill smoke (requires Vulkan/GPU)
# ---------------------------------------------------------------------------


@pytest.mark.smoke
@pytest.mark.skipif(
    not sapien_gpu_available(), reason="Requires Vulkan/GPU (SAPIEN); skipped on CPU-only machines"
)
class TestRealManiSkillSmoke:
    def test_make_stage0_env_returns_gymnasium_env(self) -> None:
        """make_stage0_env() returns a valid Gymnasium env (ADR-001 §Validation criteria)."""
        import gymnasium as gym

        env = make_stage0_env(render_mode=None)
        assert isinstance(env, gym.Env)
        env.close()

    def test_100_steps_no_error(self) -> None:
        """Running 100 steps on the real env produces no runtime error (ADR-001 cond.a/b/c)."""
        env = make_stage0_env(render_mode=None)
        obs, _ = env.reset(seed=0)
        stub = {uid: env.action_space[uid].sample() for uid in env.action_space.spaces}  # type: ignore[index, attr-defined]
        for _ in range(_N_STEPS):
            obs, *_ = env.step(stub)
        assert "agent" in obs
        assert "comm" in obs
        env.close()


@pytest.mark.smoke
def test_make_stage0_env_raises_compat_error_without_gpu() -> None:
    """make_stage0_env() raises ChamberEnvCompatibilityError when Vulkan is absent."""
    if sapien_gpu_available():
        pytest.skip("Vulkan is available; error path not reachable")
    with pytest.raises(ChamberEnvCompatibilityError, match="SAPIEN/Vulkan"):
        make_stage0_env()
