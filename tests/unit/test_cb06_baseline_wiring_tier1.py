# SPDX-License-Identifier: Apache-2.0
"""Tier-1 contract pins for the CB-06 baseline wiring (ADR-011 §Decision as amended; ADR-027).

Pure-Python, no SAPIEN: the B-BLIND observation mask
(:mod:`chamber.envs.cocarry_blind_mask`), the B-RND/B-STAT trivial
seats + co-carry policy-id dispatch
(:mod:`chamber.benchmarks.bundle_runner` /
:mod:`chamber.benchmarks.cocarry_eval`), the B-JOINT trainer + pair
checkpoint + frozen-seat roundtrip
(:mod:`chamber.benchmarks.joint_mappo_trainer` /
:mod:`chamber.partners.frozen_cocarry_joint`), and the ADR-027
checkpoint-selection rule (:mod:`chamber.benchmarks.checkpoint_selection`).

The fake co-carry env mirrors ``tests/unit/test_cocarry_rung2_tier1.py``'s
``_FakeCoCarryEnv`` (the real env's key layout: ``qpos``/``qvel``, NOT
``joint_pos`` — that drift already caused one bug).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import gymnasium as gym
import numpy as np
import pytest

from chamber.benchmarks.bundle_runner import StaticEgoPolicy, build_ego_policy
from chamber.benchmarks.checkpoint_selection import (
    SELECTION_SCHEMA,
    CandidateScore,
    CheckpointCandidate,
    CheckpointSelectionResult,
    score_episodes,
    select_by_rule,
    select_checkpoint,
    write_selection_artifact,
)
from chamber.benchmarks.joint_mappo_trainer import (
    JointMAPPOConfig,
    JointMAPPOTrainer,
    load_joint_config,
)
from chamber.envs.cocarry_blind_mask import (
    COCARRY_BLIND_MASK_START,
    COCARRY_BLIND_MASK_STOP,
    CoCarryEgoBlindMask,
    mask_ego_state,
)
from chamber.envs.cocarry_obs import CoCarryEgoStateSynthesizer
from chamber.evaluation.results import EpisodeResult
from chamber.partners.frozen_cocarry_joint import (
    FrozenCoCarryJointPartner,
    joint_partner_full_state,
)
from concerto.training.checkpoints import CheckpointMetadata, save_checkpoint

if TYPE_CHECKING:
    from pathlib import Path

_PANDA_DOF = 9
_EGO_UID = "panda_wristcam"
_PARTNER_UID = "panda_partner"
_EGO_STATE_DIM = 4 * _PANDA_DOF + 7 + 3  # 46


class _FakeCoCarryEnv(gym.Env):  # type: ignore[type-arg]
    """Minimal Tier-1 fake exposing the co-carry obs/action shapes (no SAPIEN)."""

    def __init__(self) -> None:
        super().__init__()
        agent_box = gym.spaces.Dict(
            {
                "qpos": gym.spaces.Box(-np.inf, np.inf, shape=(_PANDA_DOF,), dtype=np.float32),
                "qvel": gym.spaces.Box(-np.inf, np.inf, shape=(_PANDA_DOF,), dtype=np.float32),
            }
        )
        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Dict({_EGO_UID: agent_box, _PARTNER_UID: agent_box}),
                "extra": gym.spaces.Dict(
                    {
                        "bar_pose": gym.spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),
                        "goal_pos": gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                    }
                ),
            }
        )
        self.action_space = gym.spaces.Dict(
            {
                _EGO_UID: gym.spaces.Box(-1.0, 1.0, shape=(8,), dtype=np.float32),
                _PARTNER_UID: gym.spaces.Box(-1.0, 1.0, shape=(8,), dtype=np.float32),
            }
        )
        self.ego_uid = _EGO_UID
        self.partner_uid = _PARTNER_UID

    @staticmethod
    def sample_obs() -> dict[str, Any]:
        def _agent(offset: float) -> dict[str, Any]:
            return {
                "qpos": np.arange(_PANDA_DOF, dtype=np.float32) + offset,
                "qvel": (np.arange(_PANDA_DOF, dtype=np.float32) + offset) * 0.1,
            }

        return {
            "agent": {_EGO_UID: _agent(0.0), _PARTNER_UID: _agent(100.0)},
            "extra": {
                "bar_pose": np.array([0.0, 0.1, 0.2, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                "goal_pos": np.array([0.0, 0.12, 0.28], dtype=np.float32),
            },
        }


class TestBlindMask:
    """The B-BLIND masked slice — the committed definition (ADR-011 §Decision as amended)."""

    def test_mask_indices_pin(self) -> None:
        # partner_qpos(18:27)+partner_qvel(27:36)+bar_pose(36:43); the
        # campaign prereg quotes these numbers — renumbering is a break.
        assert COCARRY_BLIND_MASK_START == 2 * _PANDA_DOF
        assert COCARRY_BLIND_MASK_STOP == 4 * _PANDA_DOF + 7

    def test_masks_partner_and_bar_pose_leaves_ego_and_goal(self) -> None:
        env = _FakeCoCarryEnv()
        synth = CoCarryEgoStateSynthesizer(env)
        mask = CoCarryEgoBlindMask(synth)
        out = mask.observation(synth.observation(env.sample_obs()))
        state = out["agent"][_EGO_UID]["state"]
        assert state.shape == (_EGO_STATE_DIM,)
        # Ego proprioception intact.
        np.testing.assert_array_equal(state[:_PANDA_DOF], np.arange(_PANDA_DOF))
        # Partner block + bar_pose zeroed.
        np.testing.assert_array_equal(
            state[COCARRY_BLIND_MASK_START:COCARRY_BLIND_MASK_STOP],
            np.zeros(COCARRY_BLIND_MASK_STOP - COCARRY_BLIND_MASK_START, dtype=np.float32),
        )
        # Goal intact.
        np.testing.assert_array_equal(
            state[COCARRY_BLIND_MASK_STOP:], np.array([0.0, 0.12, 0.28], dtype=np.float32)
        )

    def test_raw_leaves_and_partner_state_untouched(self) -> None:
        env = _FakeCoCarryEnv()
        synth = CoCarryEgoStateSynthesizer(env)
        mask = CoCarryEgoBlindMask(synth)
        out = mask.observation(synth.observation(env.sample_obs()))
        np.testing.assert_array_equal(
            out["agent"][_PARTNER_UID]["qpos"], np.arange(_PANDA_DOF) + 100.0
        )
        # The partner's own synthesised state is NOT masked (only the ego
        # seat is blind).
        assert np.any(out["agent"][_PARTNER_UID]["state"] != 0.0)
        np.testing.assert_array_equal(
            out["extra"]["bar_pose"],
            np.array([0.0, 0.1, 0.2, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        )

    def test_observation_space_unchanged(self) -> None:
        synth = CoCarryEgoStateSynthesizer(_FakeCoCarryEnv())
        mask = CoCarryEgoBlindMask(synth)
        assert mask.observation_space == synth.observation_space

    def test_requires_synthesizer_first(self) -> None:
        with pytest.raises(TypeError, match="wrap CoCarryEgoStateSynthesizer first"):
            CoCarryEgoBlindMask(_FakeCoCarryEnv())

    def test_mask_ego_state_batched_and_pure(self) -> None:
        batched = np.ones((3, _EGO_STATE_DIM), dtype=np.float32)
        masked = mask_ego_state(batched)
        assert masked.shape == (3, _EGO_STATE_DIM)
        assert np.all(masked[:, COCARRY_BLIND_MASK_START:COCARRY_BLIND_MASK_STOP] == 0.0)
        assert np.all(masked[:, :COCARRY_BLIND_MASK_START] == 1.0)
        # The input is not mutated (raw leaves are shared with the partner).
        assert np.all(batched == 1.0)


class TestTrivialSeats:
    """B-RND / B-STAT construction + dispatch (ADR-011 §Decision as amended)."""

    def test_static_ego_policy_zeros(self) -> None:
        policy = build_ego_policy("static", action_dim=8, root_seed=0)
        assert isinstance(policy, StaticEgoPolicy)
        policy.reset(seed=3, episode=1)
        action = policy.act({})
        assert action.shape == (8,)
        assert np.all(action == 0.0)

    def test_unknown_policy_lists_known_ids(self) -> None:
        with pytest.raises(KeyError, match="random"):
            build_ego_policy("bogus", action_dim=8, root_seed=0)

    def test_cocarry_policy_id_dispatch(self) -> None:
        from chamber.benchmarks.cocarry_eval import parse_cocarry_policy

        assert parse_cocarry_policy("random") == ("random", None)
        assert parse_cocarry_policy("static") == ("static", None)
        assert parse_cocarry_policy("ref_script_cocarry_impedance") == (
            "ref_script_cocarry_impedance",
            None,
        )
        assert parse_cocarry_policy("happo:local://artifacts/a.pt") == (
            "happo",
            "local://artifacts/a.pt",
        )
        assert parse_cocarry_policy("happo_blind:local://artifacts/b.pt") == (
            "happo_blind",
            "local://artifacts/b.pt",
        )
        assert parse_cocarry_policy("joint_ego:local://artifacts/c.pt") == (
            "joint_ego",
            "local://artifacts/c.pt",
        )
        with pytest.raises(KeyError, match="known"):
            parse_cocarry_policy("nope")
        with pytest.raises(KeyError, match="known"):
            parse_cocarry_policy("happo:")  # empty URI

    def test_seed_policy_manifest_resolution(self, tmp_path: Path) -> None:
        from chamber.benchmarks.cocarry_eval import resolve_seed_policy

        manifest = tmp_path / "b_aht_selected.json"
        manifest.write_text(
            json.dumps({"0": "local://artifacts/s0.pt", "1": "local://artifacts/s1.pt"}),
            encoding="utf-8",
        )
        assert (
            resolve_seed_policy(f"happo_manifest:{manifest}", 1) == "happo:local://artifacts/s1.pt"
        )
        assert (
            resolve_seed_policy(f"happo_blind_manifest:{manifest}", 0)
            == "happo_blind:local://artifacts/s0.pt"
        )
        # Non-manifest ids pass through untouched.
        assert resolve_seed_policy("static", 3) == "static"
        assert (
            resolve_seed_policy("happo:local://artifacts/x.pt", 3) == "happo:local://artifacts/x.pt"
        )
        # A missing seed is a campaign bug, never a silently skipped cell.
        with pytest.raises(KeyError, match="seed 7"):
            resolve_seed_policy(f"happo_manifest:{manifest}", 7)

    def test_adhoc_rejects_manifest_ids(self) -> None:
        from chamber.benchmarks.cocarry_eval import run_cocarry_episodes_adhoc

        with pytest.raises(ValueError, match="one bundle per seed"):
            run_cocarry_episodes_adhoc(
                policy_id="joint_manifest:whatever.json",
                partner_name="frozen_cocarry_joint",
                seeds=[0],
                episodes_per_seed=1,
            )

    def test_build_env_rejects_mask_on_non_cocarry(self) -> None:
        from chamber.benchmarks.training_runner import build_env
        from concerto.training.config import EnvConfig

        cfg = EnvConfig(task="mpe_cooperative_push", mask_partner_obs=True)
        with pytest.raises(ValueError, match="mask_partner_obs"):
            build_env(cfg, root_seed=0)


class TestJointPartnerFullState:
    """The symmetric full-state mirror (ADR-011 §Decision as amended)."""

    def test_mirror_of_ego_concat(self) -> None:
        obs = _FakeCoCarryEnv.sample_obs()
        vec = joint_partner_full_state(obs, own_uid=_PARTNER_UID, other_uid=_EGO_UID)
        assert vec.shape == (_EGO_STATE_DIM,)
        np.testing.assert_array_equal(vec[:_PANDA_DOF], np.arange(_PANDA_DOF) + 100.0)
        np.testing.assert_array_equal(vec[2 * _PANDA_DOF : 3 * _PANDA_DOF], np.arange(_PANDA_DOF))
        np.testing.assert_array_equal(
            vec[4 * _PANDA_DOF : 4 * _PANDA_DOF + 7],
            np.array([0.0, 0.1, 0.2, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        )

    def test_missing_leaf_is_loud(self) -> None:
        obs = _FakeCoCarryEnv.sample_obs()
        del obs["extra"]["bar_pose"]
        with pytest.raises(ValueError, match="bar_pose"):
            joint_partner_full_state(obs, own_uid=_PARTNER_UID, other_uid=_EGO_UID)


def _tiny_joint_cfg(artifacts_root: Path) -> JointMAPPOConfig:
    return JointMAPPOConfig.model_validate(
        {
            "seed": 7,
            "total_frames": 16,
            "checkpoint_every": 8,
            "artifacts_root": str(artifacts_root),
            "env": {
                "task": "cocarry",
                "episode_length": 8,
                "agent_uids": [_EGO_UID, _PARTNER_UID],
                "condition_id": "cocarry_matched_panda_pair",
                "num_envs": 1,
            },
            "mappo": {
                "lr": 1e-3,
                "gamma": 0.9,
                "gae_lambda": 0.95,
                "clip_eps": 0.1,
                "n_epochs": 1,
                "rollout_length": 4,
                "batch_size": 4,
                "hidden_dim": 16,
            },
            "runtime": {"device": "cpu", "deterministic_torch": False},
        }
    )


class TestJointMAPPOTrainer:
    """B-JOINT trainer + pair-checkpoint + frozen-seat roundtrip (ADR-011 as amended)."""

    def _make_trainer(self, tmp_path: Path) -> JointMAPPOTrainer:
        cfg = _tiny_joint_cfg(tmp_path / "artifacts")
        return JointMAPPOTrainer(
            cfg=cfg,
            ego_uid=_EGO_UID,
            partner_uid=_PARTNER_UID,
            state_dim=_EGO_STATE_DIM,
            act_dims={_EGO_UID: 8, _PARTNER_UID: 8},
        )

    def _synth_obs(self) -> dict[str, Any]:
        env = _FakeCoCarryEnv()
        return CoCarryEgoStateSynthesizer(env).observation(env.sample_obs())

    def test_act_returns_both_seats(self, tmp_path: Path) -> None:
        trainer = self._make_trainer(tmp_path)
        actions = trainer.act(self._synth_obs())
        assert set(actions) == {_EGO_UID, _PARTNER_UID}
        assert actions[_EGO_UID].shape == (8,)
        assert actions[_PARTNER_UID].shape == (8,)

    def test_rollout_update_runs_and_clears_buffer(self, tmp_path: Path) -> None:
        trainer = self._make_trainer(tmp_path)
        obs = self._synth_obs()
        for step in range(4):
            trainer.act(obs)
            trainer.observe(obs, reward=0.5, done=(step == 3), truncated=(step == 3))
        trainer.update()
        assert not trainer._buf_rewards  # buffer-clear contract pin

    def test_param_sharing_recorded_and_pair_layout(self, tmp_path: Path) -> None:
        cfg = _tiny_joint_cfg(tmp_path / "artifacts")
        assert cfg.param_sharing == "per_agent"
        trainer = self._make_trainer(tmp_path)
        sd = trainer.state_dict()
        assert set(sd) == {
            "actor_ego",
            "actor_ego_optim",
            "actor_partner",
            "actor_partner_optim",
            "critic",
            "critic_optim",
        }
        # Per-agent parameters: the two actors are independent modules.
        ego_first = next(iter(sd["actor_ego"].values()))
        partner_first = next(iter(sd["actor_partner"].values()))
        assert ego_first.data_ptr() != partner_first.data_ptr()

    def test_pair_checkpoint_frozen_seat_roundtrip(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chamber.partners.api import PartnerSpec

        trainer = self._make_trainer(tmp_path)
        artifacts = tmp_path / "artifacts"
        save_checkpoint(
            state_dict=trainer.state_dict(),
            uri="local://artifacts/pair_step8.pt",
            metadata=CheckpointMetadata(
                run_id="deadbeefdeadbeef",
                seed=7,
                step=8,
                git_sha="0" * 40,
                pyproject_hash="0" * 64,
                sha256="",
            ),
            artifacts_root=artifacts,
        )
        monkeypatch.setenv("CONCERTO_ARTIFACTS_ROOT", str(artifacts))
        seat = FrozenCoCarryJointPartner(
            PartnerSpec(
                "frozen_cocarry_joint",
                0,
                None,
                "local://artifacts/pair_step8.pt",
                {"uid": _PARTNER_UID, "other_uid": _EGO_UID},
            )
        )
        seat.reset(seed=0)
        action = seat.act(_FakeCoCarryEnv.sample_obs())
        assert action.shape == (8,)
        # Frozen: repeated calls on the same obs are identical (mode, no RNG).
        np.testing.assert_array_equal(action, seat.act(_FakeCoCarryEnv.sample_obs()))

    def test_frozen_seat_spec_validation(self) -> None:
        from chamber.partners.api import PartnerSpec

        with pytest.raises(ValueError, match="weights_uri"):
            FrozenCoCarryJointPartner(
                PartnerSpec("frozen_cocarry_joint", 0, None, None, {"uid": "a", "other_uid": "b"})
            )
        with pytest.raises(ValueError, match="other_uid"):
            FrozenCoCarryJointPartner(
                PartnerSpec("frozen_cocarry_joint", 0, None, "local://artifacts/x.pt", {"uid": "a"})
            )

    def test_load_joint_config_rejects_bad_override(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text("seed: 0\n", encoding="utf-8")
        with pytest.raises(ValueError, match=r"dotted\.key=value"):
            load_joint_config(cfg_path, overrides=["no-equals-sign"])


def _episode(*, success: bool, stress: float, tilt: float, idx: int = 0) -> EpisodeResult:
    return EpisodeResult(
        seed=0,
        episode_idx=idx,
        initial_state_seed=idx,
        success=success,
        force_peak=stress,
        metadata={"max_tilt_deg": tilt},
    )


class TestCheckpointSelection:
    """The preregistered selection rule (ADR-027 §Reporting rules; Rung-2 precedent)."""

    def test_stress_compliant_conjunction(self) -> None:
        episodes = [
            _episode(success=True, stress=70.0, tilt=5.0, idx=0),  # compliant
            _episode(success=True, stress=200.0, tilt=5.0, idx=1),  # over stress limit
            _episode(success=True, stress=70.0, tilt=20.0, idx=2),  # over tilt limit
            _episode(success=False, stress=10.0, tilt=1.0, idx=3),  # clean failure
        ]
        row = score_episodes(50_000, "local://artifacts/a.pt", episodes)
        assert row.successes == 3
        assert row.compliant_successes == 1
        assert row.n_episodes == 4

    def test_rule_max_then_earliest(self) -> None:
        rows = [
            CandidateScore(100_000, "u2", 10, 9, 9, 70.0, 80.0, 5.0),
            CandidateScore(50_000, "u1", 10, 9, 9, 70.0, 80.0, 5.0),
            CandidateScore(150_000, "u3", 10, 8, 8, 70.0, 80.0, 5.0),
        ]
        best = select_by_rule(rows)
        assert best.step == 50_000  # tie at 9 breaks to the earlier step

    def test_rule_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="no candidate scores"):
            select_by_rule([])

    def test_pair_mode_validation_member_mismatch(self) -> None:
        from chamber.partners.sets import get_partner_set, resolve_set_members

        set_spec = get_partner_set("cocarry_partners", version=1)
        members = resolve_set_members(set_spec, include_private=False)
        with pytest.raises(ValueError, match="joint_ego"):
            select_checkpoint(
                candidates=[CheckpointCandidate(1, "local://artifacts/x.pt")],
                policy_prefix="joint_ego",
                set_spec=set_spec,
                validation_member=members[0],
                selection_seeds=[0],
                episodes_per_seed=1,
            )
        with pytest.raises(ValueError, match="validation_member"):
            select_checkpoint(
                candidates=[CheckpointCandidate(1, "local://artifacts/x.pt")],
                policy_prefix="happo",
                set_spec=set_spec,
                validation_member=None,
                selection_seeds=[0],
                episodes_per_seed=1,
            )

    def test_artifact_roundtrip(self, tmp_path: Path) -> None:
        result = CheckpointSelectionResult(
            schema=SELECTION_SCHEMA,
            selection_rule="rule",
            policy_prefix="happo",
            validation_partner="imp_nominal",
            selection_seeds=[0, 1],
            episodes_per_seed=2,
            scores=[CandidateScore(50_000, "u1", 4, 4, 4, 70.0, 80.0, 5.0)],
            selected_step=50_000,
            selected_uri="u1",
        )
        out = tmp_path / "sel" / "selection.json"
        write_selection_artifact(result, out)
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["schema"] == SELECTION_SCHEMA
        assert payload["selected_step"] == 50_000
        assert payload["scores"][0]["compliant_successes"] == 4
