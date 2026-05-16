# SPDX-License-Identifier: Apache-2.0
"""Tier-1 fake-env tests for the Stage-1 OM adapter (T5b.2; plan/07 §3).

Mirrors :mod:`tests.integration.test_stage1_as_fake` for the OM axis.
Exercises ``chamber.benchmarks.stage1_om.run_axis`` end-to-end against
:class:`tests.fakes.FakeMultiAgentEnv` (CPU-only, no SAPIEN, no
ManiSkill). Pins the SpikeRun shape against the OM prereg's
sample-size contract, the homo/hetero metadata split, determinism
under repeated runs (ADR-002 P6), the condition-mapping contract,
and the entry-point dispatch via the canonical
``spikes/preregistration/OM.yaml``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from chamber.benchmarks.stage1_om import (
    _CONDITION_UIDS,
    _run_axis_with_factories,
    run_axis,
)
from chamber.evaluation.prereg import load_prereg
from tests.fakes import FakeMultiAgentEnv

if TYPE_CHECKING:
    from collections.abc import Mapping

    import gymnasium as gym
    from numpy.typing import NDArray

_REPO_ROOT = Path(__file__).resolve().parents[2]
_OM_PREREG = _REPO_ROOT / "spikes" / "preregistration" / "OM.yaml"


def _make_args(axis: str) -> argparse.Namespace:
    """Build a minimal argparse Namespace for run_axis."""
    return argparse.Namespace(axis=axis)


def _fake_env_factory(
    condition_id: str, agent_uids: tuple[str, str], root_seed: int
) -> gym.Env[Any, Any]:
    """Drop FakeMultiAgentEnv in for the per-condition env (T5b.2 Tier-1)."""
    del condition_id, root_seed
    return FakeMultiAgentEnv(agent_uids=agent_uids)


def _scripted_ego_action(
    ego_uid: str,
    obs: Mapping[str, Any],
    partner_action: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Deterministic ego policy for the fake-env test (always-zero, like prod stub)."""
    del ego_uid, obs
    return np.zeros_like(partner_action, dtype=np.float32)


@pytest.fixture
def prereg():
    """Load the shipped Stage-1 OM pre-registration."""
    return load_prereg(_OM_PREREG)


class TestStage1OMShapesMatchPrereg:
    """plan/07 §3 + plan/07 §2: SpikeRun shape mirrors the prereg's sample-size contract."""

    def test_spike_run_axis_is_om(self, prereg) -> None:
        run = _run_axis_with_factories(
            prereg=prereg, env_factory=_fake_env_factory, ego_action_fn=_scripted_ego_action
        )
        assert run.axis == "OM"

    def test_condition_pair_round_trips(self, prereg) -> None:
        run = _run_axis_with_factories(
            prereg=prereg, env_factory=_fake_env_factory, ego_action_fn=_scripted_ego_action
        )
        assert run.condition_pair.homogeneous_id == prereg.condition_pair.homogeneous_id
        assert run.condition_pair.heterogeneous_id == prereg.condition_pair.heterogeneous_id

    def test_seeds_round_trip(self, prereg) -> None:
        run = _run_axis_with_factories(
            prereg=prereg, env_factory=_fake_env_factory, ego_action_fn=_scripted_ego_action
        )
        assert run.seeds == list(prereg.seeds)

    def test_episode_count_matches_sample_size_contract(self, prereg) -> None:
        """plan/07 §2: 5 seeds x 20 episodes x 2 conditions = 200 episodes per spike."""
        run = _run_axis_with_factories(
            prereg=prereg, env_factory=_fake_env_factory, ego_action_fn=_scripted_ego_action
        )
        expected = len(prereg.seeds) * prereg.episodes_per_seed * 2
        assert len(run.episode_results) == expected

    def test_git_tag_passes_through_to_spike_id(self, prereg) -> None:
        run = _run_axis_with_factories(
            prereg=prereg, env_factory=_fake_env_factory, ego_action_fn=_scripted_ego_action
        )
        assert run.git_tag == prereg.git_tag
        assert prereg.git_tag in run.spike_id
        # Distinct from the AS adapter's spike_id prefix.
        assert run.spike_id.startswith("stage1_om_")


class TestStage1OMEpisodeMetadata:
    """Per-episode metadata carries the condition + reward summary (ADR-014 §Decision)."""

    def test_condition_metadata_is_homo_or_hetero(self, prereg) -> None:
        run = _run_axis_with_factories(
            prereg=prereg, env_factory=_fake_env_factory, ego_action_fn=_scripted_ego_action
        )
        homo = prereg.condition_pair.homogeneous_id
        hetero = prereg.condition_pair.heterogeneous_id
        for ep in run.episode_results:
            assert ep.metadata.get("condition") in {homo, hetero}

    def test_split_is_fifty_fifty(self, prereg) -> None:
        run = _run_axis_with_factories(
            prereg=prereg, env_factory=_fake_env_factory, ego_action_fn=_scripted_ego_action
        )
        homo = prereg.condition_pair.homogeneous_id
        hetero = prereg.condition_pair.heterogeneous_id
        n_homo = sum(1 for ep in run.episode_results if ep.metadata.get("condition") == homo)
        n_hetero = sum(1 for ep in run.episode_results if ep.metadata.get("condition") == hetero)
        assert n_homo == n_hetero

    def test_mean_reward_and_step_count_recorded(self, prereg) -> None:
        run = _run_axis_with_factories(
            prereg=prereg, env_factory=_fake_env_factory, ego_action_fn=_scripted_ego_action
        )
        for ep in run.episode_results:
            assert "mean_reward" in ep.metadata
            assert "n_steps" in ep.metadata
            float(ep.metadata["mean_reward"])
            int(ep.metadata["n_steps"])

    def test_initial_state_seed_unique_per_episode(self, prereg) -> None:
        """ADR-007 §Validation criteria: pairing on (seed, episode_idx, initial_state_seed)."""
        run = _run_axis_with_factories(
            prereg=prereg, env_factory=_fake_env_factory, ego_action_fn=_scripted_ego_action
        )
        homo_id = prereg.condition_pair.homogeneous_id
        for s in prereg.seeds:
            seeds_for_s = {
                ep.initial_state_seed
                for ep in run.episode_results
                if ep.seed == s and ep.metadata.get("condition") == homo_id
            }
            assert len(seeds_for_s) == prereg.episodes_per_seed


class TestStage1OMDeterminism:
    """ADR-002 P6: two runs with identical inputs produce byte-identical SpikeRuns."""

    def test_two_runs_byte_identical(self, prereg) -> None:
        run_a = _run_axis_with_factories(
            prereg=prereg, env_factory=_fake_env_factory, ego_action_fn=_scripted_ego_action
        )
        run_b = _run_axis_with_factories(
            prereg=prereg, env_factory=_fake_env_factory, ego_action_fn=_scripted_ego_action
        )
        assert run_a.model_dump_json() == run_b.model_dump_json()


class TestStage1OMConditionMapping:
    """The Phase-0 _CONDITION_UIDS table maps every prereg condition_id (plan/07 §3)."""

    def test_both_prereg_conditions_are_in_the_uid_map(self, prereg) -> None:
        assert prereg.condition_pair.homogeneous_id in _CONDITION_UIDS
        assert prereg.condition_pair.heterogeneous_id in _CONDITION_UIDS

    def test_om_homo_and_hetero_share_the_same_uid_tuple(self, prereg) -> None:
        """plan/07 §3 OM: same panda+fetch agent pair across conditions.

        OM is an observation-modality axis, not an action-space axis,
        so both ``condition_id`` strings map to the same ``agent_uids``
        tuple. The per-condition *obs* divergence is supplied by
        :data:`_OBS_CHANNELS_BY_CONDITION` (Stage 1a stand-in) — see
        :class:`TestStage1OMConditionDivergence` for the
        obs-channel-shape and SpikeRun-divergence assertions.
        """
        homo_uids = _CONDITION_UIDS[prereg.condition_pair.homogeneous_id]
        hetero_uids = _CONDITION_UIDS[prereg.condition_pair.heterogeneous_id]
        assert homo_uids == hetero_uids

    def test_unknown_condition_raises_value_error(self, prereg) -> None:
        """Defensive: a prereg drift in the condition_id strings is loud, not silent."""
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
                prereg=bogus, env_factory=_fake_env_factory, ego_action_fn=_scripted_ego_action
            )


class TestStage1OMConditionDivergence:
    """OM conditions must resolve to distinct env builds (ADR-007 §Stage 1a; plan/07 §3).

    Regression guard for the Phase-0 tuple-collision defect:
    previously both ``condition_id`` strings mapped to identical
    ``agent_uids`` tuples *and* the production ``_default_env_factory``
    ignored ``condition_id`` entirely, so the OM SpikeRun was byte-
    identical across homo and hetero. The fix (PR-B for Gap F) wraps
    :class:`MPECooperativePushEnv` in
    :class:`chamber.benchmarks.stage1_om._ObsChannelFilterEnv` and
    exposes a distinct channel slice per condition.
    """

    def test_each_condition_resolves_to_a_distinct_obs_shape(self, prereg) -> None:
        """``_default_env_factory`` builds envs whose obs shapes differ per condition."""
        from chamber.benchmarks.stage1_om import _default_env_factory

        homo_id = prereg.condition_pair.homogeneous_id
        hetero_id = prereg.condition_pair.heterogeneous_id
        homo_env = _default_env_factory(homo_id, _CONDITION_UIDS[homo_id], root_seed=0)
        hetero_env = _default_env_factory(hetero_id, _CONDITION_UIDS[hetero_id], root_seed=0)
        homo_state = homo_env.observation_space["agent"]["panda_wristcam"]["state"]  # type: ignore[index]
        hetero_state = hetero_env.observation_space["agent"]["panda_wristcam"]["state"]  # type: ignore[index]
        assert homo_state.shape != hetero_state.shape, (
            f"OM tuple-collision regression: both conditions resolved to "
            f"obs shape {homo_state.shape}; the channel-filter wrapper is "
            "not active."
        )

    def test_spike_run_diverges_across_conditions(self, prereg) -> None:
        """End-to-end SpikeRun on ``_default_env_factory`` diverges across conditions.

        Runs the OM adapter end-to-end with the production env
        factory (the MPE stand-in + the per-condition channel filter).
        For at least one ``(seed, episode_idx)`` pair, the
        ``mean_reward`` recorded under the homogeneous condition
        must differ from the same pair under the heterogeneous
        condition — i.e. the partner's actions, and therefore the
        env's reward stream, depend on which obs slice is exposed.
        """
        from chamber.benchmarks.stage1_om import (
            _default_env_factory,
            _zero_ego_action,
        )

        run = _run_axis_with_factories(
            prereg=prereg,
            env_factory=_default_env_factory,
            ego_action_fn=_zero_ego_action,
        )
        homo_id = prereg.condition_pair.homogeneous_id
        hetero_id = prereg.condition_pair.heterogeneous_id
        homo_rewards: dict[tuple[int, int], str] = {}
        hetero_rewards: dict[tuple[int, int], str] = {}
        for ep in run.episode_results:
            key = (ep.seed, ep.episode_idx)
            if ep.metadata["condition"] == homo_id:
                homo_rewards[key] = ep.metadata["mean_reward"]
            elif ep.metadata["condition"] == hetero_id:
                hetero_rewards[key] = ep.metadata["mean_reward"]
        # At least one paired (seed, episode_idx) tuple's mean_reward
        # differs across conditions.
        diverging = [
            k for k in homo_rewards if k in hetero_rewards and homo_rewards[k] != hetero_rewards[k]
        ]
        assert diverging, (
            "OM tuple-collision regression: homo and hetero produced "
            "identical mean_reward across every (seed, episode_idx); "
            "the channel-filter wrapper is not active or has no effect."
        )


class TestStage1OMEntryPointResolution:
    """plan/07 §T5b.1: chamber-spike run --axis OM dispatches to this adapter."""

    def test_run_axis_loads_canonical_prereg(self, tmp_path: Path, monkeypatch) -> None:
        """Smoke: run_axis(args) without --dry-run reads spikes/preregistration/OM.yaml."""
        from chamber.benchmarks import stage1_om

        monkeypatch.setattr(stage1_om, "_default_env_factory", _fake_env_factory, raising=True)
        monkeypatch.setattr(stage1_om, "_zero_ego_action", _scripted_ego_action, raising=True)
        run = run_axis(_make_args("OM"))
        assert run.axis == "OM"
        # The shipped prereg has 5 seeds x 20 episodes per (seed, condition) = 200 episodes.
        assert len(run.episode_results) == 200

    def test_wrong_axis_raises_value_error(self) -> None:
        """Defensive: a dispatch routing mistake surfaces loudly."""
        with pytest.raises(ValueError, match="OM"):
            run_axis(_make_args("AS"))
