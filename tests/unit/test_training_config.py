# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``concerto.training.config`` (T4b.11 + plan/05 §2 Hydra root).

Covers ADR-002 §Decisions (Hydra config root + Pydantic-validated
schema) and plan/05 §6 acceptance criteria 7 (determinism: ``seed`` is
the unique reproducibility pin) + 9 (config schema is the public
reproducibility surface).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from concerto.training.config import (
    EgoAHTConfig,
    EnvConfig,
    HAPPOHyperparams,
    PartnerConfig,
    WandbConfig,
    load_config,
)

# Path to the project's configs/ directory (siblings of src/ and tests/).
_CONFIGS_DIR: Path = Path(__file__).resolve().parents[2] / "configs"


def _minimal_env() -> EnvConfig:
    return EnvConfig(task="mpe_cooperative_push")


class TestWandbConfig:
    def test_defaults(self) -> None:
        """plan/05 §2: W&B is opt-in; default disabled (CI / unit tests)."""
        cfg = WandbConfig()
        assert cfg.enabled is False
        assert cfg.project == "concerto-m4b"

    def test_enabled_round_trip(self) -> None:
        cfg = WandbConfig(enabled=True, project="prod-runs")
        assert cfg.enabled is True
        assert cfg.project == "prod-runs"


class TestEnvConfig:
    def test_defaults(self) -> None:
        """plan/05 §3.5: 50-tick episodes, ('ego', 'partner') uids."""
        cfg = EnvConfig(task="mpe_cooperative_push")
        assert cfg.episode_length == 50
        assert cfg.agent_uids == ("ego", "partner")

    def test_episode_length_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            EnvConfig(task="x", episode_length=0)

    def test_duplicate_agent_uids_raises(self) -> None:
        """Plan/05 §3.5: agent uids must be distinct (matches env contract)."""
        with pytest.raises(ValidationError, match="distinct"):
            EnvConfig(task="x", agent_uids=("ego", "ego"))

    def test_list_input_coerced_to_tuple(self) -> None:
        """OmegaConf parses YAML lists; coerce so the tuple type is preserved."""
        cfg = EnvConfig.model_validate({"task": "x", "agent_uids": ["a", "b"]})
        assert cfg.agent_uids == ("a", "b")

    def test_extra_field_forbidden(self) -> None:
        """ADR-002 §Decisions: extra='forbid' so typos in YAML fail loud."""
        with pytest.raises(ValidationError):
            EnvConfig.model_validate({"task": "x", "agent_uids": ["a", "b"], "typo_key": "value"})


class TestPartnerConfig:
    def test_defaults(self) -> None:
        """plan/05 §3.5: scripted_heuristic is the empirical-guarantee partner."""
        cfg = PartnerConfig()
        assert cfg.class_name == "scripted_heuristic"
        assert cfg.seed == 0
        assert cfg.checkpoint_step is None
        assert cfg.weights_uri is None
        assert cfg.extra == {}

    def test_frozen_rl_partner_round_trip(self) -> None:
        """ADR-009 §Consequences: frozen MAPPO / HARL partner spec round-trips."""
        cfg = PartnerConfig(
            class_name="frozen_harl",
            seed=7,
            checkpoint_step=50_000,
            weights_uri="local://artifacts/happo_seed7_step50k.pt",
            extra={"uid": "allegro_hand_right"},
        )
        assert cfg.weights_uri == "local://artifacts/happo_seed7_step50k.pt"
        assert cfg.extra["uid"] == "allegro_hand_right"


class TestHAPPOHyperparams:
    def test_defaults_match_documented_values(self) -> None:
        """Plan/05 §3.2: defaults are conservative enough for 30-min CPU budget."""
        cfg = HAPPOHyperparams()
        assert cfg.lr == 3.0e-4
        assert cfg.gamma == 0.99
        assert cfg.gae_lambda == 0.95
        assert cfg.clip_eps == 0.2
        assert cfg.n_epochs == 4
        assert cfg.rollout_length == 1024
        assert cfg.batch_size == 256
        assert cfg.hidden_dim == 64

    def test_lr_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            HAPPOHyperparams(lr=0.0)

    def test_gamma_in_range(self) -> None:
        with pytest.raises(ValidationError):
            HAPPOHyperparams(gamma=-0.1)
        with pytest.raises(ValidationError):
            HAPPOHyperparams(gamma=1.1)

    def test_gae_lambda_in_range(self) -> None:
        with pytest.raises(ValidationError):
            HAPPOHyperparams(gae_lambda=2.0)


class TestEgoAHTConfig:
    def test_minimal_construction(self) -> None:
        """Plan/05 §2: only ``env`` is required; everything else has a default."""
        cfg = EgoAHTConfig(env=_minimal_env())
        assert cfg.algo == "ego_aht_happo"
        assert cfg.seed == 0
        assert cfg.total_frames == 100_000
        assert cfg.checkpoint_every == 10_000

    def test_path_string_coerced(self, tmp_path: Path) -> None:
        """OmegaConf passes Path values as strings; coerce so the type is right."""
        artifacts = str(tmp_path / "artifacts")
        logs = str(tmp_path / "logs")
        cfg = EgoAHTConfig(
            env=_minimal_env(),
            artifacts_root=artifacts,  # type: ignore[arg-type]
            log_dir=logs,  # type: ignore[arg-type]
        )
        assert isinstance(cfg.artifacts_root, Path)
        assert isinstance(cfg.log_dir, Path)
        assert cfg.artifacts_root == Path(artifacts)

    def test_negative_seed_raises(self) -> None:
        with pytest.raises(ValidationError):
            EgoAHTConfig(env=_minimal_env(), seed=-1)

    def test_zero_total_frames_raises(self) -> None:
        with pytest.raises(ValidationError):
            EgoAHTConfig(env=_minimal_env(), total_frames=0)

    def test_frozen(self) -> None:
        """ADR-002 §Decisions: config is immutable so call sites cannot mutate mid-run."""
        cfg = EgoAHTConfig(env=_minimal_env())
        with pytest.raises(ValidationError):
            cfg.seed = 7  # type: ignore[misc]

    def test_extra_top_level_field_forbidden(self) -> None:
        """ADR-002 §Decisions: typos in YAML root fail loud."""
        with pytest.raises(ValidationError):
            EgoAHTConfig.model_validate({"env": {"task": "x"}, "typo_key": "value"})


class TestLoadConfig:
    def test_load_mpe_cooperative_push(self) -> None:
        """T4b.13: the empirical-guarantee Hydra config validates."""
        cfg = load_config(
            config_path=_CONFIGS_DIR / "training" / "ego_aht_happo" / "mpe_cooperative_push.yaml",
        )
        assert cfg.algo == "ego_aht_happo"
        assert cfg.seed == 0
        assert cfg.total_frames == 100_000
        assert cfg.env.task == "mpe_cooperative_push"
        assert cfg.env.agent_uids == ("ego", "partner")
        assert cfg.partner.class_name == "scripted_heuristic"
        assert cfg.partner.extra["target_xy"] == "0.0,0.0"

    def test_load_stage0_smoke(self) -> None:
        """T4b.14: the zoo-seed Hydra config validates (user-side GPU run)."""
        cfg = load_config(
            config_path=_CONFIGS_DIR / "training" / "ego_aht_happo" / "stage0_smoke.yaml",
        )
        assert cfg.seed == 7
        assert cfg.total_frames == 100_000
        assert cfg.env.task == "stage0_smoke"
        assert cfg.wandb.enabled is True

    def test_load_with_cli_overrides(self) -> None:
        """ADR-002 §Decisions: CLI overrides shape every YAML field."""
        cfg = load_config(
            config_path=_CONFIGS_DIR / "training" / "ego_aht_happo" / "mpe_cooperative_push.yaml",
            overrides=["seed=42", "happo.lr=1.5e-4", "total_frames=1000"],
        )
        assert cfg.seed == 42
        assert cfg.happo.lr == pytest.approx(1.5e-4)
        assert cfg.total_frames == 1000

    def test_load_invalid_override_raises(self) -> None:
        """Pydantic catches a bad type even when Hydra accepts the override."""
        with pytest.raises(ValidationError):
            load_config(
                config_path=_CONFIGS_DIR
                / "training"
                / "ego_aht_happo"
                / "mpe_cooperative_push.yaml",
                overrides=["happo.gamma=2.0"],
            )


class TestPublicSurface:
    def test_module_exports(self) -> None:
        from concerto.training import config as cfg_mod

        for name in (
            "EgoAHTConfig",
            "EnvConfig",
            "HAPPOHyperparams",
            "PartnerConfig",
            "WandbConfig",
            "load_config",
        ):
            assert hasattr(cfg_mod, name)
