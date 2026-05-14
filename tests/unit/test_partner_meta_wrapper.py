# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``chamber.envs.partner_meta`` (plan/04 §3.1; ADR-006 risk #3).

The :class:`PartnerIdAnnotationWrapper` is the producer side of the
M3 conformal filter's partner-swap contract (ADR-004 §risk-mitigation #2).
The tests pin: obs-space extension, reset + step annotation,
:meth:`set_partner_id` rebind, non-Dict-obs rejection, and the property
that the wrapper does not clobber other obs keys.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

from chamber.envs.errors import ChamberEnvCompatibilityError
from chamber.envs.partner_meta import PartnerIdAnnotationWrapper
from tests.fakes import FakeMultiAgentEnv


def _dict_subspaces(space: gym.spaces.Space) -> dict[str, gym.spaces.Space]:
    """Pyright-friendly accessor for a Dict-typed space's sub-spaces."""
    assert isinstance(space, gym.spaces.Dict)
    return space.spaces


def _dict_keys(space: gym.spaces.Space) -> list[str]:
    """Pyright-friendly accessor for a Dict-typed space's keys."""
    return list(_dict_subspaces(space).keys())


class TestObservationSpace:
    def test_obs_space_has_meta_subspace_after_wrap(self) -> None:
        """plan/04 §3.1: the wrapper extends obs space with a 'meta' Dict sub-space."""
        inner = FakeMultiAgentEnv()
        wrapped = PartnerIdAnnotationWrapper(inner, partner_id="abc123")
        assert "meta" in _dict_subspaces(wrapped.observation_space)
        assert isinstance(_dict_subspaces(wrapped.observation_space)["meta"], gym.spaces.Dict)

    def test_existing_keys_preserved(self) -> None:
        """plan/04 §3.1: the wrapper does not strip 'agent' (or other) sub-spaces."""
        inner = FakeMultiAgentEnv()
        wrapped = PartnerIdAnnotationWrapper(inner, partner_id="abc123")
        assert "agent" in _dict_subspaces(wrapped.observation_space)

    def test_non_dict_obs_space_raises(self) -> None:
        """ADR-001 §Risks: the wrapper requires a Dict observation space."""

        class _BoxEnv(gym.Env):  # type: ignore[type-arg]
            observation_space = gym.spaces.Box(0.0, 1.0, (3,), dtype=np.float32)
            action_space = gym.spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)

            def reset(self, *, seed=None, options=None):  # type: ignore[no-untyped-def]
                return np.zeros(3, dtype=np.float32), {}

            def step(self, action):  # type: ignore[no-untyped-def]
                return np.zeros(3, dtype=np.float32), 0.0, False, False, {}

        with pytest.raises(ChamberEnvCompatibilityError, match=r"gym\.spaces\.Dict"):
            PartnerIdAnnotationWrapper(_BoxEnv(), partner_id="abc123")


class TestAnnotation:
    def test_reset_injects_partner_id(self) -> None:
        """ADR-006 risk #3: the initial obs carries partner_id from step 0."""
        inner = FakeMultiAgentEnv()
        wrapped = PartnerIdAnnotationWrapper(inner, partner_id="abc123")
        obs, _ = wrapped.reset(seed=0)
        assert obs["meta"]["partner_id"] == "abc123"

    def test_step_injects_partner_id(self) -> None:
        """ADR-006 risk #3: every step's obs carries partner_id."""
        inner = FakeMultiAgentEnv()
        wrapped = PartnerIdAnnotationWrapper(inner, partner_id="abc123")
        wrapped.reset(seed=0)
        action = {uid: np.zeros(2, dtype=np.float32) for uid in _dict_keys(inner.action_space)}
        obs, _, _, _, _ = wrapped.step(action)
        assert obs["meta"]["partner_id"] == "abc123"

    def test_other_obs_keys_pass_through(self) -> None:
        """plan/04 §3.1: the wrapper does not clobber 'agent' or other obs entries."""
        inner = FakeMultiAgentEnv()
        wrapped = PartnerIdAnnotationWrapper(inner, partner_id="abc123")
        obs, _ = wrapped.reset(seed=0)
        # FakeMultiAgentEnv emits one entry per uid under "agent".
        for uid in _dict_keys(inner.action_space):
            assert uid in obs["agent"]
            assert "state" in obs["agent"][uid]

    def test_pre_existing_meta_entries_preserved(self) -> None:
        """plan/04 §3.1: a future inner wrapper may already populate meta; merge, don't replace.

        Verified via a thin gym.Wrapper that injects ``obs["meta"] = {"foo": "bar"}``
        ahead of :class:`PartnerIdAnnotationWrapper`. The wrapper must
        merge ``partner_id`` into the existing dict rather than overwrite.
        """

        class _MetaInjector(gym.Wrapper):  # type: ignore[type-arg]
            def __init__(self, env: gym.Env) -> None:  # type: ignore[type-arg]
                super().__init__(env)
                self.observation_space = gym.spaces.Dict(
                    {**_dict_subspaces(env.observation_space), "meta": gym.spaces.Dict({})}
                )

            def reset(self, **kwargs):  # type: ignore[no-untyped-def]
                obs, info = self.env.reset(**kwargs)
                out = dict(obs)
                out["meta"] = {"foo": "bar"}
                return out, info

            def step(self, action):  # type: ignore[no-untyped-def]
                obs, reward, terminated, truncated, info = self.env.step(action)
                out = dict(obs)
                out["meta"] = {"foo": "bar"}
                return out, reward, terminated, truncated, info

        inner = _MetaInjector(FakeMultiAgentEnv())
        wrapped = PartnerIdAnnotationWrapper(inner, partner_id="abc123")
        obs, _ = wrapped.reset(seed=0)
        assert obs["meta"]["partner_id"] == "abc123"
        assert obs["meta"]["foo"] == "bar"


class TestPartnerIdBinding:
    def test_partner_id_property_returns_construction_value(self) -> None:
        """plan/04 §3.1: the property exposes the currently-injected partner_id."""
        wrapped = PartnerIdAnnotationWrapper(FakeMultiAgentEnv(), partner_id="abc123")
        assert wrapped.partner_id == "abc123"

    def test_set_partner_id_rebinds_for_subsequent_obs(self) -> None:
        """ADR-004 §risk-mitigation #2: mid-episode swap surface."""
        inner = FakeMultiAgentEnv()
        wrapped = PartnerIdAnnotationWrapper(inner, partner_id="old_hash")
        obs, _ = wrapped.reset(seed=0)
        assert obs["meta"]["partner_id"] == "old_hash"
        wrapped.set_partner_id("new_hash")
        action = {uid: np.zeros(2, dtype=np.float32) for uid in _dict_keys(inner.action_space)}
        obs, _, _, _, _ = wrapped.step(action)
        assert obs["meta"]["partner_id"] == "new_hash"
        assert wrapped.partner_id == "new_hash"

    def test_set_partner_id_does_not_trigger_implicit_reset(self) -> None:
        """ADR-004 §risk-mitigation #2: the conformal filter resets lambda, not the env.

        The wrapper rebinds the annotation only; episode state stays intact
        so the safety filter's downstream reset stays the single authority.
        """
        inner = FakeMultiAgentEnv()
        wrapped = PartnerIdAnnotationWrapper(inner, partner_id="a")
        wrapped.reset(seed=0)
        action = {uid: np.zeros(2, dtype=np.float32) for uid in _dict_keys(inner.action_space)}
        wrapped.step(action)
        n_actions_before = len(inner._actions_received)
        wrapped.set_partner_id("b")
        # No additional inner.step / inner.reset should have been triggered.
        assert len(inner._actions_received) == n_actions_before
