# SPDX-License-Identifier: Apache-2.0
# pyright: reportPrivateImportUsage=false, reportPrivateUsage=false
"""Tier-1 (no-SAPIEN) tests for the Rung-2 frozen co-carry incumbent slice.

Covers the pure-Python surface that needs no SAPIEN scene (ADR-026
§Decision 4; R-2026-06-B §15 Rung 2):

- the co-carry training config (``cocarry_matched.yaml``) resolves and the
  frozen-partner ``extra`` matches the env's single-source-of-truth spec;
- the freeze manifest is **complete** — every public ``COCARRY_*`` constant
  in :mod:`chamber.envs.cocarry` appears in ``env_constants`` (the
  missing-coefficient guard that closes the Stage-1 AS gap), and a write of
  an incomplete manifest is refused;
- ``f_max`` re-derivation + the pre-stated consistency-band classification;
- the obs synthesizer's ego/partner ``state`` dimension arithmetic and
  raw-leaf preservation, against a Tier-1 fake observation;
- the trainer's partner-freeze gate refuses a non-frozen partner stub
  (HARL-gated — skips without the ``train`` group, per project convention).

Tier-2 SAPIEN/CUDA-gated coverage (training-smoke slope, deterministic
frozen-incumbent reload, matched-reference eval) is in
``tests/integration/test_cocarry_rung2_real.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pytest

import chamber.benchmarks.cocarry_freeze as freeze
import chamber.partners.cocarry_impedance  # noqa: F401 - register "cocarry_impedance"
from chamber.envs.cocarry import cocarry_matched_controller_specs
from chamber.envs.cocarry_obs import CoCarryEgoStateSynthesizer

_CONFIG = Path("configs/training/ego_aht_happo/cocarry_matched.yaml")
_ENV_MODULE = Path("src/chamber/envs/cocarry.py")

# Panda dof under obs_mode="state_dict": 7 arm + 2 gripper fingers.
_PANDA_DOF = 9
_EGO_UID = "panda_wristcam"
_PARTNER_UID = "panda_partner"


class TestTrainingConfigResolves:
    """The co-carry matched training config resolves + matches the env spec (ADR-026 §D4)."""

    def test_config_resolves_to_cocarry_matched(self) -> None:
        from concerto.training.config import load_config

        cfg = load_config(config_path=_CONFIG)
        assert cfg.env.task == "cocarry"
        assert cfg.env.condition_id == "cocarry_matched_panda_pair"
        assert cfg.env.agent_uids == (_EGO_UID, _PARTNER_UID)
        # Single-env: the matched CoCarryImpedancePartner reads env 0 only.
        assert cfg.env.num_envs == 1
        assert cfg.partner.class_name == "cocarry_impedance"
        # Partner-freeze contract is enforceable: cocarry_impedance subclasses
        # PartnerBase (no train/learn/update); safety stack off.
        assert cfg.safety.enabled is False

    def test_partner_extra_matches_env_single_source_of_truth(self) -> None:
        """The yaml's partner.extra must equal the env's partner-seat spec (no drift)."""
        from concerto.training.config import load_config

        cfg = load_config(config_path=_CONFIG)
        assert dict(cfg.partner.extra) == cocarry_matched_controller_specs()[_PARTNER_UID]


class TestFreezeManifestCompleteness:
    """The freeze manifest captures every COCARRY_* constant (R-2026-06-B §15; the AS-gap guard)."""

    def _manifest(self) -> freeze.FreezeManifest:
        return freeze.build_manifest(
            matched_reference_success_rate=1.0,
            n_seed_clusters=12,
            matched_success_stress_p99_n=104.0,
            config_path=_CONFIG,
            env_module_path=_ENV_MODULE,
        )

    def test_every_cocarry_constant_is_frozen(self) -> None:
        # The live introspection IS the source of truth: any COCARRY_*
        # numeric constant added to cocarry.py must appear in the manifest.
        manifest = self._manifest()
        assert freeze.missing_constants(manifest) == []
        # Spot-check the load-bearing reward / limit / geometry constants
        # are present by name (regression against a silent enumeration bug).
        for name in (
            "COCARRY_REWARD_TRANSPORT_COEFF",
            "COCARRY_REWARD_LEVEL_COEFF",
            "COCARRY_REWARD_SETTLE_COEFF",
            "COCARRY_REWARD_SUCCESS_BONUS",
            "COCARRY_REWARD_NORMALIZER",
            "COCARRY_REWARD_TANH_SCALE",
            "COCARRY_REWARD_LEVEL_TANH_SCALE",
            "COCARRY_TILT_MAX_DEG",
            "COCARRY_STRESS_MAX_PROXY_N",
            "COCARRY_BAR_MASS_KG",
            "COCARRY_SETTLE_WINDOW_STEPS",
            "COCARRY_GOAL_CENTROID_XYZ",
        ):
            assert name in manifest.env_constants

    def test_dropping_a_coefficient_is_detected(self) -> None:
        manifest = self._manifest()
        crippled = manifest.to_dict()
        del crippled["env_constants"]["COCARRY_REWARD_SETTLE_COEFF"]
        assert freeze.missing_constants(crippled) == ["COCARRY_REWARD_SETTLE_COEFF"]
        with pytest.raises(freeze.CoCarryFreezeError, match="incomplete"):
            freeze.assert_manifest_complete(crippled)

    def test_write_refuses_incomplete_manifest(self, tmp_path: Path) -> None:
        manifest = self._manifest()
        # Surgically remove a constant from the frozen dataclass copy.
        crippled_constants = {
            k: v for k, v in manifest.env_constants.items() if k != "COCARRY_TILT_MAX_DEG"
        }
        incomplete = freeze.FreezeManifest(
            schema=manifest.schema,
            rung=manifest.rung,
            env_constants=crippled_constants,
            fmax=manifest.fmax,
            matched_reference=manifest.matched_reference,
            training_config_sha256=manifest.training_config_sha256,
            env_module_sha256=manifest.env_module_sha256,
        )
        with pytest.raises(freeze.CoCarryFreezeError):
            freeze.write_manifest(incomplete, tmp_path / "freeze.json")
        assert not (tmp_path / "freeze.json").exists()

    def test_complete_manifest_round_trips(self, tmp_path: Path) -> None:
        manifest = self._manifest()
        out = freeze.write_manifest(manifest, tmp_path / "freeze.json")
        loaded = freeze.load_manifest(out)
        assert loaded["schema"] == freeze.MANIFEST_SCHEMA
        assert loaded["rung"] == 2
        assert freeze.missing_constants(loaded) == []
        # Content hashes are non-empty (the config + env module were hashed).
        assert len(loaded["training_config_sha256"]) == 64
        assert len(loaded["env_module_sha256"]) == 64


class TestFMaxDerivationAndBands:
    """f_max re-derivation + the pre-stated consistency bands (R-2026-06-B §15)."""

    def test_derivation_is_1_25x_p99(self) -> None:
        assert freeze.derive_fmax_from_p99(104.0) == pytest.approx(130.0)
        assert pytest.approx(1.25) == freeze.FMAX_P99_MULTIPLIER

    def test_bands_are_pre_stated(self) -> None:
        # The bands are fixed in code (no forking path): ±20% of 130, and a
        # p99 window covering the #245 15-seed spread.
        assert freeze.FMAX_CONSISTENCY_BAND_N == (104.0, 156.0)
        assert freeze.MATCHED_SUCCESS_P99_BAND_N == (85.0, 120.0)

    def test_consistent_when_both_in_band(self) -> None:
        assert (
            freeze.classify_fmax(fmax_value=130.0, matched_success_stress_p99=104.0)
            == freeze.FMAX_CONSISTENT
        )

    def test_divergent_when_fmax_out_of_band(self) -> None:
        # p99 in band but derived f_max above 156 N.
        assert (
            freeze.classify_fmax(fmax_value=160.0, matched_success_stress_p99=110.0)
            == freeze.FMAX_DIVERGENT
        )

    def test_divergent_when_p99_out_of_band(self) -> None:
        assert (
            freeze.classify_fmax(fmax_value=130.0, matched_success_stress_p99=70.0)
            == freeze.FMAX_DIVERGENT
        )


class _FakeCoCarryEnv(gym.Env):  # type: ignore[type-arg]
    """Minimal Tier-1 fake exposing the co-carry obs/action shapes (no SAPIEN).

    Emits the real env's key layout: per-agent ``Dict(qpos, qvel)`` and
    ``extra`` with ``bar_pose`` (7) + ``goal_pos`` (3) plus the auxiliary
    ``bar_tilt_deg`` / ``wrist_stress_proxy`` channels the synthesizer must
    leave untouched. Mirrors the contract-shape-pin pattern in
    ``tests/unit/test_stage1_pickplace_tier1.py``.
    """

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
                        "bar_tilt_deg": gym.spaces.Box(
                            -np.inf, np.inf, shape=(1,), dtype=np.float32
                        ),
                        "wrist_stress_proxy": gym.spaces.Box(
                            0.0, np.inf, shape=(1,), dtype=np.float32
                        ),
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
        def _agent() -> dict[str, np.ndarray]:  # type: ignore[type-arg]
            return {
                "qpos": np.arange(_PANDA_DOF, dtype=np.float32),
                "qvel": np.arange(_PANDA_DOF, dtype=np.float32) * 0.1,
            }

        return {
            "agent": {_EGO_UID: _agent(), _PARTNER_UID: _agent()},
            "extra": {
                "bar_pose": np.array([0.0, 0.1, 0.2, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                "goal_pos": np.array([0.0, 0.12, 0.28], dtype=np.float32),
                "bar_tilt_deg": np.array([3.0], dtype=np.float32),
                "wrist_stress_proxy": np.array([90.0], dtype=np.float32),
            },
        }


class TestObsSynthesizer:
    """The ego/partner state synthesis arithmetic + raw-leaf preservation (ADR-026 §Decision 1)."""

    # ego: ego_qpos(9)+ego_qvel(9)+partner_qpos(9)+partner_qvel(9)+bar_pose(7)+goal_pos(3) = 46
    _EGO_STATE_DIM = 2 * _PANDA_DOF + 2 * _PANDA_DOF + 7 + 3
    _PARTNER_STATE_DIM = 2 * _PANDA_DOF

    def test_observation_space_state_dims(self) -> None:
        wrapped = CoCarryEgoStateSynthesizer(_FakeCoCarryEnv())
        space = wrapped.observation_space
        assert isinstance(space, gym.spaces.Dict)
        agent_space = space["agent"]
        assert isinstance(agent_space, gym.spaces.Dict)
        ego = agent_space[_EGO_UID]
        partner = agent_space[_PARTNER_UID]
        assert isinstance(ego, gym.spaces.Dict)
        assert isinstance(partner, gym.spaces.Dict)
        assert ego["state"].shape == (self._EGO_STATE_DIM,)
        assert partner["state"].shape == (self._PARTNER_STATE_DIM,)

    def test_runtime_state_dims_and_order(self) -> None:
        env = _FakeCoCarryEnv()
        wrapped = CoCarryEgoStateSynthesizer(env)
        out = wrapped.observation(env.sample_obs())
        ego_state = out["agent"][_EGO_UID]["state"]
        assert ego_state.shape == (self._EGO_STATE_DIM,)
        assert ego_state.dtype == np.float32
        # Positional contract: first 9 entries are ego qpos (0..8), next 9
        # are ego qvel (0..0.8) — the load-bearing concat order.
        np.testing.assert_allclose(ego_state[:_PANDA_DOF], np.arange(_PANDA_DOF))
        np.testing.assert_allclose(
            ego_state[_PANDA_DOF : 2 * _PANDA_DOF], np.arange(_PANDA_DOF) * 0.1, atol=1e-6
        )
        # Last 3 entries are the goal centroid.
        np.testing.assert_allclose(ego_state[-3:], [0.0, 0.12, 0.28], atol=1e-6)
        assert out["agent"][_PARTNER_UID]["state"].shape == (self._PARTNER_STATE_DIM,)

    def test_raw_leaves_preserved_for_frozen_partner(self) -> None:
        """The partner reads raw qpos + extra.goal_pos — those must survive untouched."""
        env = _FakeCoCarryEnv()
        wrapped = CoCarryEgoStateSynthesizer(env)
        obs = env.sample_obs()
        out = wrapped.observation(obs)
        # Raw qpos/qvel still present on both agents (state added alongside).
        for uid in (_EGO_UID, _PARTNER_UID):
            np.testing.assert_array_equal(out["agent"][uid]["qpos"], np.arange(_PANDA_DOF))
        # obs["extra"] is preserved verbatim (goal_pos + the aux channels);
        # the raw leaf is the same float32 array the env emitted.
        np.testing.assert_allclose(out["extra"]["goal_pos"], [0.0, 0.12, 0.28], atol=1e-6)
        assert out["extra"]["goal_pos"] is obs["extra"]["goal_pos"]
        assert "bar_tilt_deg" in out["extra"]
        assert "wrist_stress_proxy" in out["extra"]

    def test_missing_extra_raises(self) -> None:
        env = _FakeCoCarryEnv()
        wrapped = CoCarryEgoStateSynthesizer(env)
        obs = env.sample_obs()
        del obs["extra"]["bar_pose"]
        with pytest.raises(TypeError, match="bar_pose"):
            wrapped.observation(obs)


class TestPartnerFreezeGate:
    """EgoPPOTrainer's partner-freeze gate refuses a non-frozen partner (ADR-009; I3)."""

    def test_non_frozen_partner_stub_raises(self) -> None:
        # HARL-touching surface — skip without the train dependency group
        # (project convention). torch is a hard dep so it is always present.
        pytest.importorskip("harl")
        import torch

        from chamber.benchmarks.ego_ppo_trainer import _assert_partner_is_frozen

        class _UnfrozenPartner:
            """A partner adapter that bypasses PartnerBase and exposes a trainable param."""

            def named_parameters(self) -> list[tuple[str, torch.Tensor]]:
                return [("w", torch.nn.Parameter(torch.zeros(1), requires_grad=True))]

        with pytest.raises(ValueError, match="frozen partner"):
            _assert_partner_is_frozen(_UnfrozenPartner())

    def test_frozen_cocarry_partner_passes_gate(self) -> None:
        pytest.importorskip("harl")
        from chamber.benchmarks.ego_ppo_trainer import _assert_partner_is_frozen
        from chamber.partners.api import PartnerSpec
        from chamber.partners.registry import load_partner

        partner = load_partner(
            PartnerSpec(
                "cocarry_impedance",
                0,
                None,
                None,
                cocarry_matched_controller_specs()[_PARTNER_UID],
            )
        )
        # The PartnerBase shield makes named_parameters lookup raise
        # AttributeError, which the gate treats as "contract enforced".
        _assert_partner_is_frozen(partner)  # must not raise
