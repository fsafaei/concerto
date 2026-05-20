# SPDX-License-Identifier: Apache-2.0
"""Tier-1 fake tests for the Stage-1b dispatch path on the OM adapter (P1.05).

Mirror of :mod:`tests.integration.test_stage1_as_fake_stage1b_dispatch`
for the OM axis. The Stage-1a path is pinned in
:mod:`tests.integration.test_stage1_om_fake`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from chamber.benchmarks.stage1_om import (
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
_OM_PREREG = _REPO_ROOT / "spikes" / "preregistration" / "OM.yaml"
_STUB_SHA: str = "0" * 40


def _fake_env_factory(
    condition_id: str, agent_uids: tuple[str, str], root_seed: int
) -> gym.Env[Any, Any]:
    del condition_id, root_seed
    return FakeMultiAgentEnv(agent_uids=agent_uids)


def _zero_factory(
    env: gym.Env[Any, Any], seed: int
) -> Callable[[Mapping[str, Any]], NDArray[np.float32]]:
    del env, seed

    def _act(obs: Mapping[str, Any]) -> NDArray[np.float32]:
        del obs
        return np.zeros(2, dtype=np.float32)

    return _act


@pytest.fixture
def prereg():
    return load_prereg(_OM_PREREG)


class TestStage1OMSubStageFieldStamping:
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


class TestStage1OMRunAxisDispatch:
    """``run_axis(args)`` reads ``args.sub_stage`` and routes accordingly."""

    def test_sub_stage_1a_routes_to_default_factories(self) -> None:
        args = argparse.Namespace(axis="OM", sub_stage="1a")
        run = run_axis(args)
        assert run.sub_stage == "1a"
        assert len(run.prereg_sha) == 40

    def test_sub_stage_default_is_1a(self) -> None:
        args = argparse.Namespace(axis="OM")
        run = run_axis(args)
        assert run.sub_stage == "1a"

    def test_sub_stage_1b_routes_to_trained_policy_factory(self) -> None:
        """Mock the factory + env + config loader so the test stays Tier-1."""
        args = argparse.Namespace(axis="OM", sub_stage="1b")
        mock_factory_instance = _zero_factory

        with (
            patch(
                "chamber.benchmarks.stage1_common.TrainedPolicyFactory",
                MagicMock(return_value=mock_factory_instance),
            ) as mock_factory_cls,
            patch(
                "chamber.benchmarks.stage1_om._stage1b_env_factory",
                side_effect=_fake_env_factory,
            ) as mock_env_factory,
            patch(
                "concerto.training.config.load_config",
                MagicMock(return_value=MagicMock()),
            ) as mock_load_config,
        ):
            run = run_axis(args)

        assert mock_factory_cls.call_count == 1
        mock_load_config.assert_called_once()
        assert mock_env_factory.call_count == 10
        assert run.sub_stage == "1b"
        assert len(run.episode_results) == 200


class TestStage1OMInvalidSubStageRaises:
    def test_unknown_sub_stage_raises_value_error(self) -> None:
        args = argparse.Namespace(axis="OM", sub_stage="2")
        with pytest.raises(ValueError, match="unknown sub_stage"):
            run_axis(args)
