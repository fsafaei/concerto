# SPDX-License-Identifier: Apache-2.0
"""Meta-tests pinning the HARL fork recipe structure (T4b.1, T4b.4-T4b.7).

The recipe under ``scripts/harl-fork-patches/`` is the user-side
deliverable that creates ``concerto-org/harl-fork`` from
``PKU-MARL/HARL`` per ADR-002 §Decisions and plan/05 §2. This test
file pins the structural contract so an accidental rename / drop of
a patch file is caught at unit-test time.

The patches themselves are not Python modules CONCERTO imports — they
live in the fork. These tests therefore verify file presence + minimum
content, not import-time correctness.
"""

from __future__ import annotations

from pathlib import Path

import pytest

#: Project root resolved relative to this test file (concerto/ at the top).
_PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
_RECIPE_ROOT: Path = _PROJECT_ROOT / "scripts" / "harl-fork-patches"
_V010_ROOT: Path = _RECIPE_ROOT / "v0.1.0-aht"


class TestRecipePresence:
    def test_recipe_directory_exists(self) -> None:
        """ADR-002 §Decisions: the recipe directory ships in CONCERTO."""
        assert _RECIPE_ROOT.is_dir()

    def test_readme_exists(self) -> None:
        """plan/05 §2: README.md documents the fork creation steps."""
        readme = _RECIPE_ROOT / "README.md"
        assert readme.is_file()
        text = readme.read_text(encoding="utf-8")
        # Minimum content: the fork creation steps must mention both tags.
        assert "v0.0.0-vendored" in text
        assert "v0.1.0-aht" in text
        assert "concerto-org/harl-fork" in text

    def test_v010_directory_exists(self) -> None:
        """plan/05 §2: the v0.1.0-aht patch directory ships."""
        assert _V010_ROOT.is_dir()


class TestV010PatchFiles:
    """Each patch file documented in the README must exist at the right path."""

    @pytest.mark.parametrize(
        "rel_path",
        [
            "harl/algorithms/actors/ego_aht_happo.py",
            "harl/runners/ego_aht_runner.py",
            "harl/envs/concerto_env_adapter.py",
            "tests/test_ego_aht_happo.py",
            "COMMIT_MSG.txt",
        ],
    )
    def test_patch_file_present(self, rel_path: str) -> None:
        """plan/05 §2 + §3.1: every documented patch file ships in the recipe."""
        patch_file = _V010_ROOT / rel_path
        assert patch_file.is_file(), (
            f"Missing harl-fork patch file at {rel_path}; the README.md recipe references it."
        )


class TestEgoAHTHAPPOPatchContent:
    """Sanity checks on the substantive patch (the EgoAHTHAPPO subclass)."""

    def test_subclasses_happo(self) -> None:
        """plan/05 §3.2 + ADR-002 §Decisions: EgoAHTHAPPO IS-A HAPPO."""
        text = (_V010_ROOT / "harl" / "algorithms" / "actors" / "ego_aht_happo.py").read_text(
            encoding="utf-8"
        )
        assert "class EgoAHTHAPPO(HAPPO):" in text

    def test_validate_partner_is_frozen_method_present(self) -> None:
        """ADR-009 §Consequences: the runtime backstop method ships."""
        text = (_V010_ROOT / "harl" / "algorithms" / "actors" / "ego_aht_happo.py").read_text(
            encoding="utf-8"
        )
        assert "_validate_partner_is_frozen" in text

    def test_from_config_classmethod_present(self) -> None:
        """plan/05 §3.5: TrainerFactory seam binds via from_config."""
        text = (_V010_ROOT / "harl" / "algorithms" / "actors" / "ego_aht_happo.py").read_text(
            encoding="utf-8"
        )
        assert "def from_config(" in text

    def test_adr_009_message_references_in_validation(self) -> None:
        """ADR-009 §Consequences: validation error cites ADR for traceability."""
        text = (_V010_ROOT / "harl" / "algorithms" / "actors" / "ego_aht_happo.py").read_text(
            encoding="utf-8"
        )
        assert "ADR-009" in text


class TestRunnerPatchContent:
    """Sanity checks on the Hydra runner."""

    def test_hydra_main_decorator_present(self) -> None:
        """T4b.7: Hydra-driven runner uses @hydra.main."""
        text = (_V010_ROOT / "harl" / "runners" / "ego_aht_runner.py").read_text(encoding="utf-8")
        assert "@hydra.main" in text

    def test_delegates_to_chamber_runner(self) -> None:
        """T4b.7: the runner delegates to the chamber-side run_training."""
        text = (_V010_ROOT / "harl" / "runners" / "ego_aht_runner.py").read_text(encoding="utf-8")
        assert "run_training" in text
        assert "EgoAHTHAPPO.from_config" in text


class TestEnvAdapterPatchContent:
    def test_env_adapter_class_present(self) -> None:
        """T4b.3: ConcertoEnvAdapter wraps the CONCERTO env."""
        text = (_V010_ROOT / "harl" / "envs" / "concerto_env_adapter.py").read_text(
            encoding="utf-8"
        )
        assert "class ConcertoEnvAdapter:" in text


class TestCommitMessage:
    def test_commit_msg_follows_conventional_commits(self) -> None:
        """plan/00 T0.8: every commit message uses Conventional Commits format."""
        text = (_V010_ROOT / "COMMIT_MSG.txt").read_text(encoding="utf-8")
        first_line = text.splitlines()[0]
        # feat(<scope>): <subject>
        assert first_line.startswith("feat(")

    def test_commit_msg_mentions_v010_tag(self) -> None:
        """plan/05 §2: the commit message names the tag it ships under."""
        text = (_V010_ROOT / "COMMIT_MSG.txt").read_text(encoding="utf-8")
        assert "v0.1.0-aht" in text

    def test_commit_msg_cites_adrs(self) -> None:
        """ADR traceability: commit cites the ADRs the patches implement."""
        text = (_V010_ROOT / "COMMIT_MSG.txt").read_text(encoding="utf-8")
        assert "ADR-002" in text
        assert "ADR-009" in text
