# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``chamber-eval admission`` (ADR-027 §Admission protocol CLI).

Pins the gate ordering the acceptance criteria require: the
pre-registration tag check trips with exit ``PREREG_MISMATCH_EXIT_CODE``
(4) **before any cell runs** (ADR-007 §Discipline), the dirty-tree
refusal exits ``DIRTY_TREE_EXIT_CODE`` (7), and a clean run drives the
protocol end to end through the ``chamber-eval`` dispatcher.
"""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

import pytest
import yaml

import chamber.benchmarks.admission_cells as cells_module
from chamber.cli import eval as eval_cli
from chamber.evaluation.admission import AdmissionCellSpec, CellRun
from chamber.evaluation.bundles import DIRTY_TREE_EXIT_CODE
from chamber.evaluation.prereg import PREREG_MISMATCH_EXIT_CODE
from chamber.evaluation.results import EpisodeResult
from chamber.partners.api import PartnerSpec

if TYPE_CHECKING:
    from pathlib import Path

_TAG = "prereg/adm-cli-test"


def _git(repo: Path, *args: str) -> None:
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


def _payload() -> dict[str, object]:
    cell = {
        "cell_id": "a1_reference",
        "runner": "clitest",
        "policy_id": "ref_script",
        "partner_name": "scripted_heuristic",
        "params": {},
    }
    return {
        "schema_version": 1,
        "task_id": "faketask",
        "git_tag": _TAG,
        "parameters": {
            "admission": {
                "task_version": 1,
                "tau_solv": 0.9,
                "stress_limit": None,
                "tau_infeasible": 0.05,
                "delta_min": 0.2,
                "seeds": [0, 1],
                "episodes_per_seed": 3,
                "extension_seeds": [2],
                "n_resamples": 100,
                "root_seed": 0,
                "a1": cell,
                "a2": {**cell, "cell_id": "a2_ablated"},
                "a3": {**cell, "cell_id": "a3_blind", "policy_id": "b_blind"},
            }
        },
        "decision_rules": "A1/A2/A3 per ADR-027 §Admission protocol.",
    }


def _write_tagged_prereg(repo: Path) -> Path:
    prereg = repo / "prereg.yaml"
    prereg.write_text(yaml.safe_dump(_payload(), sort_keys=True), encoding="utf-8")
    _git(repo, "add", "prereg.yaml")
    _git(repo, "commit", "-q", "-m", "lock prereg")
    _git(repo, "tag", _TAG)
    return prereg


@pytest.fixture
def fake_runner(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Register a counting fake cell runner under the 'clitest' key."""
    calls = {"n": 0}
    spec = PartnerSpec("scripted_heuristic", 0, None, None, {"action_dim": "2"})

    def _runner(
        *,
        cell: AdmissionCellSpec,
        seeds: list[int],
        episodes_per_seed: int,
        root_seed: int,
        render_backend: str | None = None,
    ) -> CellRun:
        del root_seed, render_backend
        calls["n"] += 1
        # a1 + a3 succeed everywhere, a2 fails everywhere → the blind ego
        # matches the reference → the A3 demotion path (CONTROL).
        success = not cell.cell_id.startswith("a2")
        return CellRun(
            episodes_by_seed={
                s: [
                    EpisodeResult(
                        seed=s,
                        episode_idx=e,
                        initial_state_seed=e,
                        success=success,
                        metadata={"condition": "clitest"},
                    )
                    for e in range(episodes_per_seed)
                ]
                for s in seeds
            },
            partner_material=[
                {
                    "name": "partner:scripted_heuristic",
                    "class_name": spec.class_name,
                    "seed": spec.seed,
                    "checkpoint_step": None,
                    "weights_uri": None,
                    "extra": dict(spec.extra),
                }
            ],
            partner_hashes={"partner:scripted_heuristic": spec.partner_id},
            substream_labels=["clitest.substream"],
        )

    monkeypatch.setitem(cells_module.CELL_RUNNERS, "clitest", _runner)
    return calls


@pytest.mark.usefixtures("fake_runner")
def test_prereg_gate_exits_4_before_any_cell(tmp_path: Path, fake_runner: dict[str, int]) -> None:
    """The exit-4 path of the acceptance criteria: post-tag edit trips the gate."""
    repo = _init_repo(tmp_path)
    prereg = _write_tagged_prereg(repo)
    prereg.write_text(prereg.read_text(encoding="utf-8") + "# edited\n", encoding="utf-8")
    monkey = pytest.MonkeyPatch()
    monkey.chdir(repo)
    try:
        rc = eval_cli.main(
            [
                "admission",
                "--prereg",
                str(prereg),
                "--out",
                str(repo / "out"),
                "--date",
                "2026-07-05",
            ]
        )
    finally:
        monkey.undo()
    assert rc == PREREG_MISMATCH_EXIT_CODE
    assert fake_runner["n"] == 0
    assert not (repo / "out").exists()


@pytest.mark.usefixtures("fake_runner")
def test_dirty_tree_exits_7(tmp_path: Path, fake_runner: dict[str, int]) -> None:
    repo = _init_repo(tmp_path)
    prereg = _write_tagged_prereg(repo)
    (repo / "scratch.txt").write_text("dirty", encoding="utf-8")
    monkey = pytest.MonkeyPatch()
    monkey.chdir(repo)
    try:
        rc = eval_cli.main(
            [
                "admission",
                "--prereg",
                str(prereg),
                "--out",
                str(repo / "out"),
                "--date",
                "2026-07-05",
            ]
        )
    finally:
        monkey.undo()
    assert rc == DIRTY_TREE_EXIT_CODE
    assert fake_runner["n"] == 0


def test_happy_path_reports_verdict(
    tmp_path: Path,
    fake_runner: dict[str, int],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A clean run prints the verdict line and writes the archive."""
    repo = _init_repo(tmp_path)
    prereg = _write_tagged_prereg(repo)
    out = repo / "spikes" / "admission-out"
    monkey = pytest.MonkeyPatch()
    monkey.chdir(repo)
    try:
        rc = eval_cli.main(
            ["admission", "--prereg", str(prereg), "--out", str(out), "--date", "2026-07-05"]
        )
    finally:
        monkey.undo()
    assert rc == 0
    assert fake_runner["n"] == 3
    printed = capsys.readouterr().out
    assert "faketask@v1 -> CONTROL" in printed  # blind matches reference → demotion
    assert (out / "admission_report.json").is_file()
