# SPDX-License-Identifier: Apache-2.0
"""Tier-1 fake tests for the Stage-1b dispatch path on the AS adapter (P1.05).

The Stage-1a path (`--sub-stage 1a`) is exhaustively pinned in
:mod:`tests.integration.test_stage1_as_fake`. This module pins only
the *new* dispatch behaviour P1.05 ships:

1. ``--sub-stage 1b`` routes to ``_stage1b_env_factory`` +
   :class:`TrainedPolicyFactory` (verified via mocks, no SAPIEN).
2. ``--sub-stage 1a`` still routes to ``_default_env_factory`` +
   :func:`_zero_ego_action_factory` (regression pin).
3. The returned :class:`SpikeRun.sub_stage` field matches the
   ``--sub-stage`` flag verbatim (no fallback / drift).
4. ``run_axis`` raises a clear error when ``args.sub_stage`` is
   neither ``"1a"`` nor ``"1b"`` (defensive; argparse choices=...
   makes this path normally unreachable).

The Stage-1b real-env smoke (SAPIEN-gated) lives in
:mod:`tests.integration.test_stage1_as_real_stage1b`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from chamber.benchmarks.stage1_as import (
    _run_axis_with_factories,
    run_axis,
)
from chamber.evaluation.prereg import load_prereg
from tests.fakes import FakeMultiAgentEnv

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    import gymnasium as gym
    from numpy.typing import NDArray

_REPO_ROOT = Path(__file__).resolve().parents[2]
_AS_PREREG = _REPO_ROOT / "spikes" / "preregistration" / "AS.yaml"
_STUB_SHA: str = "0" * 40


def _fake_env_factory(
    condition_id: str, agent_uids: tuple[str, str], root_seed: int
) -> gym.Env[Any, Any]:
    """Drop FakeMultiAgentEnv in for the per-condition env."""
    del condition_id, root_seed
    return FakeMultiAgentEnv(agent_uids=agent_uids)


def _zero_factory(
    env: gym.Env[Any, Any], seed: int
) -> Callable[[Mapping[str, Any]], NDArray[np.float32]]:
    """Tiny EgoActionFactory stub returning a zero-action closure."""
    del env, seed

    def _act(obs: Mapping[str, Any]) -> NDArray[np.float32]:
        del obs
        return np.zeros(2, dtype=np.float32)

    return _act


@pytest.fixture
def prereg():
    return load_prereg(_AS_PREREG)


class TestStage1ASSubStageFieldStamping:
    """The ``sub_stage`` kwarg propagates verbatim to ``SpikeRun.sub_stage``."""

    def test_sub_stage_1a_stamps_field(self, prereg) -> None:  # type: ignore[no-untyped-def]
        run = _run_axis_with_factories(
            prereg=prereg,
            prereg_sha=_STUB_SHA,
            env_factory=_fake_env_factory,
            ego_action_factory=_zero_factory,
            sub_stage="1a",
        )
        assert run.sub_stage == "1a"

    def test_sub_stage_1b_stamps_field(self, prereg) -> None:  # type: ignore[no-untyped-def]
        run = _run_axis_with_factories(
            prereg=prereg,
            prereg_sha=_STUB_SHA,
            env_factory=_fake_env_factory,
            ego_action_factory=_zero_factory,
            sub_stage="1b",
        )
        assert run.sub_stage == "1b"

    def test_sub_stage_defaults_to_1a_when_kwarg_omitted(self, prereg) -> None:  # type: ignore[no-untyped-def]
        """Backward-compat: callers that don't pass sub_stage still get 1a."""
        run = _run_axis_with_factories(
            prereg=prereg,
            prereg_sha=_STUB_SHA,
            env_factory=_fake_env_factory,
            ego_action_factory=_zero_factory,
        )
        assert run.sub_stage == "1a"


class TestStage1ASRunAxisDispatch:
    """``run_axis(args)`` reads ``args.sub_stage`` and routes accordingly."""

    def test_sub_stage_1a_routes_to_default_factories(self) -> None:
        """``args.sub_stage='1a'`` ⇒ ``_default_env_factory`` + ``_zero_ego_action_factory``."""
        args = argparse.Namespace(axis="AS", sub_stage="1a")
        run = run_axis(args)
        assert run.sub_stage == "1a"
        # ADR-007 §Discipline: Stage-1a runs still record the verified
        # prereg blob SHA — the audit chain closes on both sub-stages.
        assert len(run.prereg_sha) == 40

    def test_sub_stage_default_is_1a(self) -> None:
        """No ``sub_stage`` attribute on args ⇒ defaults to 1a (Phase-0 invocation contract)."""
        args = argparse.Namespace(axis="AS")
        run = run_axis(args)
        assert run.sub_stage == "1a"

    def test_sub_stage_1b_routes_to_trained_policy_factory(self) -> None:
        """``args.sub_stage='1b'`` ⇒ calls ``TrainedPolicyFactory`` + ``_stage1b_env_factory``.

        Patches both so the test doesn't touch SAPIEN. Asserts:
        (a) TrainedPolicyFactory is constructed exactly once;
        (b) the stage1b env factory is invoked for each (seed,
            condition) cell;
        (c) the returned SpikeRun carries ``sub_stage='1b'``.
        """
        args = argparse.Namespace(axis="AS", sub_stage="1b")

        # Mock the TrainedPolicyFactory class to return a zero-action factory.
        mock_factory_instance = _zero_factory

        with (
            patch(
                "chamber.benchmarks.stage1_common.TrainedPolicyFactory",
                MagicMock(return_value=mock_factory_instance),
            ) as mock_factory_cls,
            patch(
                "chamber.benchmarks.stage1_as._stage1b_env_factory",
                side_effect=_fake_env_factory,
            ) as mock_env_factory,
            patch(
                "concerto.training.config.load_config",
                MagicMock(return_value=MagicMock()),
            ) as mock_load_config,
        ):
            run = run_axis(args)

        # The factory class is constructed exactly once per run_axis call.
        assert mock_factory_cls.call_count == 1
        # The config loader is called exactly once with the Stage-1b yaml path.
        mock_load_config.assert_called_once()
        # The stage1b env factory is called once per (seed, condition) cell —
        # 5 seeds x 2 conditions = 10 invocations.
        assert mock_env_factory.call_count == 10
        # SpikeRun carries sub_stage='1b'.
        assert run.sub_stage == "1b"
        # Sample size unchanged from Stage-1a: 5x20x2 = 200 episodes.
        assert len(run.episode_results) == 200


class TestStage1ASInvalidSubStageRaises:
    """Defensive: an unknown ``args.sub_stage`` raises a clear error."""

    def test_unknown_sub_stage_raises_value_error(self) -> None:
        args = argparse.Namespace(axis="AS", sub_stage="2")  # Stage-2 doesn't apply
        with pytest.raises(ValueError, match="unknown sub_stage"):
            run_axis(args)
