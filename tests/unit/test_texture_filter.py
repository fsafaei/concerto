# SPDX-License-Identifier: Apache-2.0
"""Unit tests for TextureFilterObsWrapper.

Covers: masked channels are zeros, kept channels are unchanged, tensor shapes
are invariant, and uid absent from keep_per_agent passes through unchanged.
"""

from __future__ import annotations

import numpy as np

from chamber.envs.texture_filter import TextureFilterObsWrapper
from tests.fakes import FakeMultiAgentEnv


def _make_wrapper(
    keep: dict[str, list[str]], uids: tuple[str, ...] = ("a", "b")
) -> tuple[TextureFilterObsWrapper, FakeMultiAgentEnv]:
    inner = FakeMultiAgentEnv(agent_uids=uids)
    wrapper = TextureFilterObsWrapper(inner, keep_per_agent=keep)
    return wrapper, inner


class TestMasking:
    def test_masked_channels_are_zeros(self) -> None:
        """Channels not in keep set are zero-masked (ADR-001 §Decision item 1)."""
        wrapper, _ = _make_wrapper({"a": ["state"], "b": ["state"]})
        wrapper.reset(seed=0)
        # Step once; inner env returns all-ones arrays.
        obs, *_ = wrapper.step({"a": np.array([0.0, 0.0]), "b": np.array([0.0, 0.0])})
        # 'rgb' and 'depth' are not in keep → must be zero.
        assert np.all(obs["agent"]["a"]["rgb"] == 0.0)
        assert np.all(obs["agent"]["a"]["depth"] == 0.0)
        assert np.all(obs["agent"]["b"]["rgb"] == 0.0)

    def test_kept_channels_are_unchanged(self) -> None:
        """Channels in keep set are returned as-is (ADR-001 §Decision item 1)."""
        wrapper, _ = _make_wrapper({"a": ["state", "rgb"], "b": ["depth"]})
        wrapper.reset(seed=0)
        obs, *_ = wrapper.step({"a": np.array([0.0, 0.0]), "b": np.array([0.0, 0.0])})
        # Inner env returns all-ones; kept channels should be non-zero.
        assert np.all(obs["agent"]["a"]["state"] == 1.0)
        assert np.all(obs["agent"]["a"]["rgb"] == 1.0)
        assert np.all(obs["agent"]["b"]["depth"] == 1.0)


class TestShapeInvariance:
    def test_tensor_shapes_unchanged(self) -> None:
        """Wrapping must not alter any tensor shape (vectorised rollout safety)."""
        inner = FakeMultiAgentEnv(agent_uids=("a",))
        wrapper = TextureFilterObsWrapper(inner, keep_per_agent={"a": ["state"]})
        obs_inner, _ = inner.reset(seed=0)
        obs_wrapped, _ = wrapper.reset(seed=0)
        for channel in ("state", "rgb", "depth"):
            assert (
                obs_wrapped["agent"]["a"][channel].shape == obs_inner["agent"]["a"][channel].shape
            )

    def test_dtype_preserved(self) -> None:
        """Zeroing must not change array dtype."""
        inner = FakeMultiAgentEnv(agent_uids=("a",))
        wrapper = TextureFilterObsWrapper(inner, keep_per_agent={"a": []})
        obs, _ = wrapper.reset(seed=0)
        assert obs["agent"]["a"]["rgb"].dtype == np.float32


class TestPassthrough:
    def test_uid_absent_from_keep_passes_through(self) -> None:
        """Agent uid absent from keep_per_agent receives unmodified obs."""
        # Only 'a' is listed; 'b' should pass through.
        wrapper, _ = _make_wrapper({"a": []})
        wrapper.reset(seed=0)
        obs, *_ = wrapper.step({"a": np.array([0.0, 0.0]), "b": np.array([0.0, 0.0])})
        # 'b' not in keep → all channels pass through (inner returns all-ones).
        assert np.all(obs["agent"]["b"]["rgb"] == 1.0)
        assert np.all(obs["agent"]["b"]["state"] == 1.0)
