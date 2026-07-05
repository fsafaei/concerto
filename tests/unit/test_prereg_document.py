# SPDX-License-Identifier: Apache-2.0
"""Tier-1 tests for the document-form pre-registration (ADR-028 §Decision 2).

Pins: PREREG_SCHEMA_VERSION exists and equals 1 (making the
``cocarry_freeze`` forward-reference real, ADR-028 §Validation
criteria 4); the document-form model validates loudly; and
``verify_git_tag`` runs the same tag-blob lock for both prereg forms
against a throwaway git repo (the ``test_evaluation_spine`` fixture
pattern).
"""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

if TYPE_CHECKING:
    from pathlib import Path

from chamber.evaluation.prereg import (
    PREREG_SCHEMA_VERSION,
    PreregDocument,
    PreregistrationError,
    load_prereg_document,
    verify_git_tag,
)

_DOC_YAML = """\
schema_version: 1
task_id: handover_place
git_tag: {tag}
revision: rev2
parameters:
  clearance_set: [0.2, 0.35, 0.5]
  grasp_pose_mismatch_deg: [15, 30, 45]
  delta_min_pp: 20.0
decision_rules: >-
  COUPLING_VALID iff gap CI_lower >= delta_min on >= 1 measured cell
  within the realistic takt band; WASHOUT otherwise.
notes: unit-test document
"""


def _git(repo: Path, *args: str) -> None:
    """Fixed git argv against a throwaway test repo (spine-test pattern)."""
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)  # noqa: S603,S607


def _make_repo(tmp_path: Path, *, tag: str) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)  # noqa: S603,S607
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    _git(repo, "config", "commit.gpgsign", "false")
    _git(repo, "config", "tag.gpgsign", "false")
    prereg = repo / "prereg_doc.yaml"
    prereg.write_text(_DOC_YAML.format(tag=tag), encoding="utf-8")
    _git(repo, "add", "prereg_doc.yaml")
    _git(repo, "commit", "-q", "-m", "lock prereg document")
    _git(repo, "tag", tag)
    return repo, prereg


class TestPreregSchemaVersion:
    def test_constant_exists_and_is_one(self) -> None:
        """ADR-028 §Validation criteria 4: the constant is defined, value 1."""
        assert PREREG_SCHEMA_VERSION == 1

    def test_document_defaults_to_current_version(self) -> None:
        doc = PreregDocument(task_id="t", git_tag="tag", parameters={}, decision_rules="rule")
        assert doc.schema_version == PREREG_SCHEMA_VERSION


class TestPreregDocument:
    def test_loads_document_form_yaml(self, tmp_path: Path) -> None:
        path = tmp_path / "doc.yaml"
        path.write_text(_DOC_YAML.format(tag="prereg/doc-v0"), encoding="utf-8")
        doc = load_prereg_document(path)
        assert doc.task_id == "handover_place"
        assert doc.parameters["delta_min_pp"] == 20.0
        assert "COUPLING_VALID" in doc.decision_rules

    def test_unknown_keys_fail_loudly(self, tmp_path: Path) -> None:
        path = tmp_path / "doc.yaml"
        path.write_text(
            _DOC_YAML.format(tag="prereg/doc-v0") + "smuggled_key: 1\n", encoding="utf-8"
        )
        with pytest.raises(ValidationError, match="smuggled_key"):
            load_prereg_document(path)

    def test_round_trip(self) -> None:
        doc = PreregDocument(
            task_id="t", git_tag="tag", parameters={"a": [1, 2]}, decision_rules="rule"
        )
        assert PreregDocument.model_validate_json(doc.model_dump_json()) == doc


class TestVerifyGitTagBothForms:
    def test_document_form_tag_verification_passes(self, tmp_path: Path) -> None:
        repo, prereg = _make_repo(tmp_path, tag="prereg/doc-v0")
        doc = load_prereg_document(prereg)
        sha = verify_git_tag(doc, prereg, repo_path=repo)
        assert len(sha) == 40

    def test_document_form_detects_post_tag_edit(self, tmp_path: Path) -> None:
        """The edit-after-launch anti-pattern trips for the document form too."""
        repo, prereg = _make_repo(tmp_path, tag="prereg/doc-v0")
        doc = load_prereg_document(prereg)
        prereg.write_text(prereg.read_text(encoding="utf-8") + "# sneaky edit\n", encoding="utf-8")
        with pytest.raises(PreregistrationError, match="blob SHA mismatch"):
            verify_git_tag(doc, prereg, repo_path=repo)

    def test_document_form_missing_tag_fails(self, tmp_path: Path) -> None:
        repo, prereg = _make_repo(tmp_path, tag="prereg/doc-v0")
        doc = PreregDocument(
            task_id="t", git_tag="prereg/never-cut", parameters={}, decision_rules="r"
        )
        with pytest.raises(PreregistrationError, match="does not exist"):
            verify_git_tag(doc, prereg, repo_path=repo)
