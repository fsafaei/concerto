# SPDX-License-Identifier: Apache-2.0
"""Integration: the ADR-028 smoke-eval flow through the chamber-eval CLI.

The in-process mirror of ``scripts/smoke_eval.sh`` / the smoke-eval CI
job (ADR-028 §Validation criteria 1): ``chamber-eval run`` on the
Tier-0 CPU task → ``chamber-eval verify`` passes → one-byte tamper →
``chamber-eval verify`` fails. Also exercises the prereg-gated run
path (exit 4 before any episode runs) against a throwaway git repo.
Everything here is CPU-only and Tier-1-safe.
"""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

import pytest

from chamber.cli import _eval_run as eval_run_module
from chamber.cli import eval as eval_cli
from chamber.evaluation.bundles import GitProvenance
from chamber.evaluation.prereg import PREREG_MISMATCH_EXIT_CODE
from chamber.evaluation.results import ResultBundle, load_run_archive

if TYPE_CHECKING:
    from pathlib import Path


def _git(repo: Path, *args: str) -> None:
    """Fixed git argv against a throwaway test repo (spine-test pattern)."""
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)  # noqa: S603,S607


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)  # noqa: S603,S607
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    _git(repo, "config", "commit.gpgsign", "false")
    _git(repo, "config", "tag.gpgsign", "false")
    return repo


@pytest.fixture
def clean_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin a clean-tree provenance so the flow is host-state independent."""
    monkeypatch.setattr(
        eval_run_module, "git_provenance", lambda _repo: GitProvenance(sha="a" * 40, dirty=False)
    )


def _run_bundle(out: Path, *extra: str) -> int:
    return eval_cli.main(
        [
            "run",
            "--task",
            "mpe_cooperative_push",
            "--policy",
            "random",
            "--partner",
            "scripted_heuristic",
            "--seeds",
            "2",
            "--episodes",
            "5",
            "--out",
            str(out),
            *extra,
        ]
    )


@pytest.mark.usefixtures("clean_provenance")
def test_run_verify_tamper_flow(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """ADR-028 §Validation criteria 1, end to end in one process."""
    out = tmp_path / "bundle"
    assert _run_bundle(out) == 0

    loaded = load_run_archive(out / "bundle.json")
    assert isinstance(loaded, ResultBundle)
    assert loaded.dirty is False
    assert loaded.task_id == "mpe_cooperative_push"
    assert loaded.summary.n_episodes == 10

    assert eval_cli.main(["verify", str(out)]) == 0
    table = capsys.readouterr().out
    assert "verify: PASS" in table

    target = out / "episodes_seed0.jsonl"
    raw = bytearray(target.read_bytes())
    raw[10] ^= 0xFF
    target.write_bytes(bytes(raw))

    assert eval_cli.main(["verify", str(out)]) == 1
    table = capsys.readouterr().out
    assert "verify: FAIL" in table
    assert "FAIL  manifest:episodes_seed0.jsonl" in table


@pytest.mark.usefixtures("clean_provenance")
def test_run_is_seed_deterministic(tmp_path: Path) -> None:
    """ADR-002 P6: two runs with identical seed schedules produce identical episodes."""
    assert _run_bundle(tmp_path / "b1") == 0
    assert _run_bundle(tmp_path / "b2") == 0
    for name in ("episodes_seed0.jsonl", "episodes_seed1.jsonl"):
        assert (tmp_path / "b1" / name).read_bytes() == (tmp_path / "b2" / name).read_bytes()


@pytest.mark.usefixtures("clean_provenance")
def test_prereg_gate_blocks_before_any_episode(tmp_path: Path) -> None:
    """A failing tag-blob check exits PREREG_MISMATCH_EXIT_CODE with no bundle written."""
    repo = _init_repo(tmp_path)
    prereg = repo / "doc.yaml"
    prereg.write_text(
        "schema_version: 1\n"
        "task_id: mpe_cooperative_push\n"
        "git_tag: prereg/smoke-v0\n"
        "parameters: {seeds: 2}\n"
        "decision_rules: diagnostic run\n",
        encoding="utf-8",
    )
    _git(repo, "add", "doc.yaml")
    _git(repo, "commit", "-q", "-m", "lock")
    _git(repo, "tag", "prereg/smoke-v0")
    # Post-tag edit: the gate must trip before any episode runs.
    prereg.write_text(prereg.read_text(encoding="utf-8") + "# edited\n", encoding="utf-8")

    out = tmp_path / "bundle"
    monkey_cwd = pytest.MonkeyPatch()
    monkey_cwd.chdir(repo)
    try:
        rc = _run_bundle(out, "--prereg", str(prereg))
    finally:
        monkey_cwd.undo()
    assert rc == PREREG_MISMATCH_EXIT_CODE
    assert not out.exists()


@pytest.mark.usefixtures("clean_provenance")
def test_prereg_gated_run_records_tag_and_blob(tmp_path: Path) -> None:
    """A passing gate stamps prereg_git_tag + blob SHA and copies prereg.yaml in."""
    repo = _init_repo(tmp_path)
    prereg = repo / "doc.yaml"
    prereg.write_text(
        "schema_version: 1\n"
        "task_id: mpe_cooperative_push\n"
        "git_tag: prereg/smoke-v1\n"
        "parameters: {seeds: 2}\n"
        "decision_rules: diagnostic run\n",
        encoding="utf-8",
    )
    _git(repo, "add", "doc.yaml")
    _git(repo, "commit", "-q", "-m", "lock")
    _git(repo, "tag", "prereg/smoke-v1")

    out = tmp_path / "bundle"
    monkey_cwd = pytest.MonkeyPatch()
    monkey_cwd.chdir(repo)
    try:
        rc = _run_bundle(out, "--prereg", str(prereg))
        assert rc == 0
        assert eval_cli.main(["verify", str(out)]) == 0
    finally:
        monkey_cwd.undo()
    loaded = load_run_archive(out / "bundle.json")
    assert isinstance(loaded, ResultBundle)
    assert loaded.prereg_git_tag == "prereg/smoke-v1"
    assert loaded.prereg_blob_sha is not None
    assert (out / "prereg.yaml").is_file()
