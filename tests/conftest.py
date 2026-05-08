# SPDX-License-Identifier: Apache-2.0
"""Shared pytest fixtures and configuration for the CONCERTO test suite."""

from __future__ import annotations

import pytest

from tests.fakes import FakeMultiAgentEnv


@pytest.fixture
def fake_env() -> FakeMultiAgentEnv:
    """Two-agent fake env fixture for wrapper tests."""
    return FakeMultiAgentEnv(agent_uids=("a", "b"))


@pytest.fixture
def three_agent_fake_env() -> FakeMultiAgentEnv:
    """Three-agent fake env mirroring ADR-001's smoke robot tuple."""
    return FakeMultiAgentEnv(agent_uids=("panda_wristcam", "fetch", "allegro_hand_right"))
