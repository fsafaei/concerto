# SPDX-License-Identifier: Apache-2.0
"""Tier-1 fake-env tests for the Stage-1 AS adapter (T5b.2; plan/07 §3).

Exercises ``chamber.benchmarks.stage1_as.run_axis`` end-to-end against
:class:`tests.fakes.FakeMultiAgentEnv` (CPU-only, no SAPIEN, no
ManiSkill). Asserts:

- The adapter loads ``spikes/preregistration/AS.yaml`` and respects
  its sample-size contract (5 seeds x 20 episodes per (seed,
  condition) = 200 episodes total per spike).
- The SpikeRun's ``condition_pair`` matches the prereg verbatim.
- Each episode's ``metadata["condition"]`` is one of the two
  prereg ids and the homo/hetero split is 50/50.
- Determinism: two runs of ``_run_axis_with_factories`` with the
  same fake-env factory + same ego-action callable produce byte-
  identical SpikeRuns (ADR-002 P6).

Mirrors the Tier-1 pattern from
:mod:`tests.integration.test_stage0_adapter_fake`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from chamber.benchmarks.stage1_as import (
    _CONDITION_UIDS,
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


def _make_args(axis: str) -> argparse.Namespace:
    """Build a minimal argparse Namespace for run_axis. Removes duck-type ``_Args`` repetition."""
    return argparse.Namespace(axis=axis)


def _fake_env_factory(
    condition_id: str, agent_uids: tuple[str, str], root_seed: int
) -> gym.Env[Any, Any]:
    """Drop FakeMultiAgentEnv in for the per-condition env (T5b.2 Tier-1)."""
    del condition_id, root_seed  # FakeMultiAgentEnv is parametrised by uids only.
    return FakeMultiAgentEnv(agent_uids=agent_uids)


def _scripted_ego_action_factory(
    env: gym.Env[Any, Any],
    seed: int,
) -> Callable[[Mapping[str, Any]], NDArray[np.float32]]:
    """Deterministic always-zero EgoActionFactory for the fake-env test.

    Mirrors the production Stage-1a default (`_zero_ego_action_factory`)
    but kept locally in the test module so the test exercises the
    Protocol-shaped seam end-to-end.
    """
    del env, seed  # zero action is deterministic; no env / seed needed.

    def _act(obs: Mapping[str, Any]) -> NDArray[np.float32]:
        del obs
        return np.zeros(2, dtype=np.float32)

    return _act


@pytest.fixture
def prereg():
    """Load the shipped Stage-1 AS pre-registration."""
    return load_prereg(_AS_PREREG)


class TestStage1ASShapesMatchPrereg:
    """plan/07 §3 + plan/07 §2: SpikeRun shape mirrors the prereg's sample-size contract."""

    def test_spike_run_axis_is_as(self, prereg) -> None:
        run = _run_axis_with_factories(
            prereg=prereg,
            env_factory=_fake_env_factory,
            ego_action_factory=_scripted_ego_action_factory,
        )
        assert run.axis == "AS"

    def test_condition_pair_round_trips(self, prereg) -> None:
        run = _run_axis_with_factories(
            prereg=prereg,
            env_factory=_fake_env_factory,
            ego_action_factory=_scripted_ego_action_factory,
        )
        assert run.condition_pair.homogeneous_id == prereg.condition_pair.homogeneous_id
        assert run.condition_pair.heterogeneous_id == prereg.condition_pair.heterogeneous_id

    def test_seeds_round_trip(self, prereg) -> None:
        run = _run_axis_with_factories(
            prereg=prereg,
            env_factory=_fake_env_factory,
            ego_action_factory=_scripted_ego_action_factory,
        )
        assert run.seeds == list(prereg.seeds)

    def test_episode_count_matches_sample_size_contract(self, prereg) -> None:
        """plan/07 §2: 5 seeds x 20 episodes x 2 conditions = 200 episodes per spike."""
        run = _run_axis_with_factories(
            prereg=prereg,
            env_factory=_fake_env_factory,
            ego_action_factory=_scripted_ego_action_factory,
        )
        expected = len(prereg.seeds) * prereg.episodes_per_seed * 2
        assert len(run.episode_results) == expected

    def test_git_tag_passes_through_to_spike_id(self, prereg) -> None:
        run = _run_axis_with_factories(
            prereg=prereg,
            env_factory=_fake_env_factory,
            ego_action_factory=_scripted_ego_action_factory,
        )
        assert run.git_tag == prereg.git_tag
        assert prereg.git_tag in run.spike_id


class TestStage1ASEpisodeMetadata:
    """Per-episode metadata carries the condition + reward summary (ADR-014 §Decision)."""

    def test_condition_metadata_is_homo_or_hetero(self, prereg) -> None:
        run = _run_axis_with_factories(
            prereg=prereg,
            env_factory=_fake_env_factory,
            ego_action_factory=_scripted_ego_action_factory,
        )
        homo = prereg.condition_pair.homogeneous_id
        hetero = prereg.condition_pair.heterogeneous_id
        for ep in run.episode_results:
            assert ep.metadata.get("condition") in {homo, hetero}

    def test_split_is_fifty_fifty(self, prereg) -> None:
        run = _run_axis_with_factories(
            prereg=prereg,
            env_factory=_fake_env_factory,
            ego_action_factory=_scripted_ego_action_factory,
        )
        homo = prereg.condition_pair.homogeneous_id
        hetero = prereg.condition_pair.heterogeneous_id
        n_homo = sum(1 for ep in run.episode_results if ep.metadata.get("condition") == homo)
        n_hetero = sum(1 for ep in run.episode_results if ep.metadata.get("condition") == hetero)
        assert n_homo == n_hetero

    def test_mean_reward_and_step_count_recorded(self, prereg) -> None:
        run = _run_axis_with_factories(
            prereg=prereg,
            env_factory=_fake_env_factory,
            ego_action_factory=_scripted_ego_action_factory,
        )
        for ep in run.episode_results:
            assert "mean_reward" in ep.metadata
            assert "n_steps" in ep.metadata
            float(ep.metadata["mean_reward"])  # parseable
            int(ep.metadata["n_steps"])

    def test_initial_state_seed_unique_per_episode(self, prereg) -> None:
        """ADR-007 §Validation criteria: pairing on (seed, episode_idx, initial_state_seed)."""
        run = _run_axis_with_factories(
            prereg=prereg,
            env_factory=_fake_env_factory,
            ego_action_factory=_scripted_ego_action_factory,
        )
        # The initial_state_seed is derived from (seed, episode_idx); within
        # each seed the per-episode seeds must be pairwise distinct.
        homo_id = prereg.condition_pair.homogeneous_id
        for s in prereg.seeds:
            seeds_for_s = {
                ep.initial_state_seed
                for ep in run.episode_results
                if ep.seed == s and ep.metadata.get("condition") == homo_id
            }
            assert len(seeds_for_s) == prereg.episodes_per_seed


class TestStage1ASDeterminism:
    """ADR-002 P6: two runs with identical inputs produce byte-identical SpikeRuns."""

    def test_two_runs_byte_identical(self, prereg) -> None:
        run_a = _run_axis_with_factories(
            prereg=prereg,
            env_factory=_fake_env_factory,
            ego_action_factory=_scripted_ego_action_factory,
        )
        run_b = _run_axis_with_factories(
            prereg=prereg,
            env_factory=_fake_env_factory,
            ego_action_factory=_scripted_ego_action_factory,
        )
        assert run_a.model_dump_json() == run_b.model_dump_json()


class TestStage1ASConditionMapping:
    """The Phase-0 _CONDITION_UIDS table maps every prereg condition_id (plan/07 §3)."""

    def test_both_prereg_conditions_are_in_the_uid_map(self, prereg) -> None:
        assert prereg.condition_pair.homogeneous_id in _CONDITION_UIDS
        assert prereg.condition_pair.heterogeneous_id in _CONDITION_UIDS

    def test_unknown_condition_raises_value_error(self, prereg) -> None:
        """Defensive: a prereg drift in the condition_id strings is loud, not silent."""
        # Pydantic frozen=True forbids field mutation; build a fresh
        # spec with a bogus condition_id to exercise the failure path.
        from chamber.evaluation.results import ConditionPair

        bogus = prereg.model_copy(
            update={
                "condition_pair": ConditionPair(
                    homogeneous_id="not_in_the_uid_map",
                    heterogeneous_id="also_not_in_the_uid_map",
                )
            }
        )
        with pytest.raises(ValueError, match="condition_id"):
            _run_axis_with_factories(
                prereg=bogus,
                env_factory=_fake_env_factory,
                ego_action_factory=_scripted_ego_action_factory,
            )


class TestStage1ASEgoActionFactoryContract:
    """The EgoActionFactory seam is called once per ``(seed, condition)`` pair.

    Pins the lifecycle contract named in
    :mod:`chamber.benchmarks.stage1_common` — Phase-1's trained-ego
    factory amortises its 100k-frame training across the 20
    evaluation episodes within each cell, so any change that
    silently shifts the factory call cadence (e.g. once-per-episode
    or once-per-spike) is a contract regression that must trip CI.
    """

    def test_factory_is_called_once_per_seed_condition_pair(self, prereg) -> None:
        """Factory invocation count = ``n_seeds x n_conditions`` (5 x 2 = 10)."""
        call_log: list[tuple[int, int]] = []  # (seed, call_index_within_seed)

        def _counting_factory(
            env: gym.Env[Any, Any], seed: int
        ) -> Callable[[Mapping[str, Any]], NDArray[np.float32]]:
            call_log.append((seed, len(call_log)))
            # Honour the user's spec: lambda obs: env.action_space.sample()
            # with a fixed seed, so the factory is exercised end-to-end.
            action_space = env.action_space
            sample_uid = next(iter(action_space.spaces.keys()))  # type: ignore[union-attr]
            ego_box = action_space.spaces[sample_uid]  # type: ignore[union-attr]
            rng = np.random.default_rng(seed)

            def _act(obs: Mapping[str, Any]) -> NDArray[np.float32]:
                del obs
                return rng.uniform(
                    low=float(ego_box.low.min()),
                    high=float(ego_box.high.max()),
                    size=ego_box.shape,
                ).astype(np.float32)

            return _act

        _run_axis_with_factories(
            prereg=prereg,
            env_factory=_fake_env_factory,
            ego_action_factory=_counting_factory,
        )
        expected = len(prereg.seeds) * 2  # 2 conditions
        assert len(call_log) == expected, (
            f"EgoActionFactory contract regression: expected "
            f"{expected} factory calls (n_seeds x n_conditions), got "
            f"{len(call_log)}. Seam may have shifted to "
            "once-per-episode or once-per-spike."
        )
        # Every seed should appear exactly ``n_conditions`` times.
        per_seed_counts = {s: sum(1 for entry in call_log if entry[0] == s) for s in prereg.seeds}
        assert all(c == 2 for c in per_seed_counts.values()), (
            f"per-seed factory call counts uneven: {per_seed_counts}"
        )


class TestStage1ASEntryPointResolution:
    """plan/07 §T5b.1: chamber-spike run --axis AS dispatches to this adapter."""

    def test_run_axis_loads_canonical_prereg(self, tmp_path: Path, monkeypatch) -> None:
        """Smoke: run_axis(args) without --dry-run reads spikes/preregistration/AS.yaml."""
        from chamber.benchmarks import stage1_as

        monkeypatch.setattr(stage1_as, "_default_env_factory", _fake_env_factory, raising=True)
        monkeypatch.setattr(
            stage1_as, "_zero_ego_action_factory", _scripted_ego_action_factory, raising=True
        )
        run = run_axis(_make_args("AS"))
        assert run.axis == "AS"
        # The shipped prereg has 5 seeds x 20 episodes per (seed, condition) = 200 episodes.
        assert len(run.episode_results) == 200

    def test_wrong_axis_raises_value_error(self) -> None:
        """Defensive: a dispatch routing mistake surfaces loudly."""
        with pytest.raises(ValueError, match="AS"):
            run_axis(_make_args("OM"))
