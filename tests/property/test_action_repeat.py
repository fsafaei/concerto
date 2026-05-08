# SPDX-License-Identifier: Apache-2.0
"""Hypothesis property tests for PerAgentActionRepeatWrapper.

Property 1 (count invariant): for any repeat count r and n steps, the number
of distinct actions dispatched to the inner env equals ceil(n / r).

Property 2 (post-reset state equivalence): wrapper state after reset is
identical regardless of the prior trajectory.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from chamber.envs.action_repeat import PerAgentActionRepeatWrapper
from tests.fakes import FakeMultiAgentEnv

_N_STEPS = 100


@given(
    repeat_a=st.integers(min_value=1, max_value=50), repeat_b=st.integers(min_value=1, max_value=50)
)
@settings(max_examples=200)
def test_action_count_invariant(repeat_a: int, repeat_b: int) -> None:
    """For repeat=r and N steps, the inner env receives exactly ceil(N/r) distinct actions.

    Validates ADR-001 §Validation criteria condition (c) for arbitrary repeat counts.
    """
    inner = FakeMultiAgentEnv(agent_uids=("a", "b"))
    wrapper = PerAgentActionRepeatWrapper(inner, action_repeat={"a": repeat_a, "b": repeat_b})
    wrapper.reset(seed=0)

    for i in range(_N_STEPS):
        action = {
            "a": np.array([float(i), 0.0], dtype=np.float32),
            "b": np.array([float(i * 2), 0.0], dtype=np.float32),
        }
        wrapper.step(action)

    received_a = [r["a"][0] for r in inner._actions_received]
    received_b = [r["b"][0] for r in inner._actions_received]

    transitions_a = 1 + sum(received_a[i] != received_a[i - 1] for i in range(1, _N_STEPS))
    transitions_b = 1 + sum(received_b[i] != received_b[i - 1] for i in range(1, _N_STEPS))

    assert transitions_a == math.ceil(_N_STEPS / repeat_a)
    assert transitions_b == math.ceil(_N_STEPS / repeat_b)


@given(
    repeat=st.integers(min_value=1, max_value=20),
    steps_before_reset=st.integers(min_value=0, max_value=50),
)
@settings(max_examples=100)
def test_post_reset_state_equivalence(repeat: int, steps_before_reset: int) -> None:
    """After reset(), wrapper state is independent of prior trajectory.

    ADR-001 §Decision item 2: no state leaks across episodes.
    """
    inner = FakeMultiAgentEnv(agent_uids=("a",))
    wrapper = PerAgentActionRepeatWrapper(inner, action_repeat={"a": repeat})
    wrapper.reset(seed=0)

    # Run some steps with a distinctive action to potentially corrupt state.
    for i in range(steps_before_reset):
        wrapper.step({"a": np.array([999.0, 0.0], dtype=np.float32)})

    # Reset and then take one step with a known action.
    wrapper.reset(seed=1)
    inner._actions_received.clear()
    wrapper.step({"a": np.array([1.0, 0.0], dtype=np.float32)})

    # First step after reset must always use the new action (counter=0).
    assert inner._actions_received[0]["a"][0] == pytest.approx(1.0)
