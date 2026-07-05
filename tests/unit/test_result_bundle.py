# SPDX-License-Identifier: Apache-2.0
"""Tier-1 tests for the v3 result-bundle schema + IO (ADR-028 §Decision 1, 3, 4).

Covers: the SCHEMA_VERSION 2 → 3 bump, ResultBundle round-trip, the
version-dispatching loader with the v2 read-only compatibility path
(against a committed fixture copied from a real archive — the original
under ``spikes/`` untouched, invariant I8), bundle write → verify, the
dirty-tree refusal in ``chamber-eval run``, and the deterministic
summary recompute.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from chamber.cli import _eval_run as eval_run_module
from chamber.evaluation.bundles import (
    DIRTY_TREE_EXIT_CODE,
    GitProvenance,
    compute_summary,
    verify_bundle_dir,
    write_bundle_dir,
)
from chamber.evaluation.results import (
    SCHEMA_VERSION,
    BundleSummary,
    EpisodeResult,
    PlatformFingerprint,
    ResultBundle,
    SeedSchedule,
    SpikeRun,
    load_run_archive,
)

if TYPE_CHECKING:
    from collections.abc import Callable

from chamber.partners.api import PartnerSpec

_PARTNER_SPEC = PartnerSpec(
    class_name="scripted_heuristic",
    seed=0,
    checkpoint_step=None,
    weights_uri=None,
    extra={"uid": "partner", "target_xy": "0.0,0.0", "action_dim": "2"},
)

_V2_FIXTURE = Path(__file__).parents[1] / "fixtures" / "spike_run_v2_post_widening.json"


def _episodes(seed: int, *, n: int = 4) -> list[EpisodeResult]:
    return [
        EpisodeResult(
            seed=seed,
            episode_idx=idx,
            initial_state_seed=idx,
            success=(idx % 2 == 0),
            metadata={"condition": "unit_test", "final_reward": -1.0 - idx},
        )
        for idx in range(n)
    ]


def _bundle(**overrides: object) -> ResultBundle:
    base: dict[str, object] = {
        "task_id": "mpe_cooperative_push",
        "task_version": 1,
        "policy_id": "random",
        "partner_set_id": "adhoc:scripted_heuristic",
        "partner_hashes": {"scripted_heuristic": _PARTNER_SPEC.partner_id},
        "git_sha": "f" * 40,
        "dirty": False,
        "package_version": "0.0.0-test",
        "seed_schedule": SeedSchedule(
            root_seed=0, seeds=[0, 1], episodes_per_seed=4, substream_labels=["a.{seed}"]
        ),
        "repro_command": "chamber-eval run --task mpe_cooperative_push",
        "platform": PlatformFingerprint(
            os="test-os", python="3.11", numpy="0", torch=None, device="cpu"
        ),
        "manifest": {},
        "summary": compute_summary(_episodes(0) + _episodes(1)),
    }
    base.update(overrides)
    return ResultBundle(**base)  # type: ignore[arg-type]


def _write_test_bundle(out_dir: Path, *, bundle: ResultBundle | None = None) -> ResultBundle:
    return write_bundle_dir(
        out_dir,
        bundle_without_manifest=bundle if bundle is not None else _bundle(),
        episodes_by_seed={0: _episodes(0), 1: _episodes(1)},
        partner_specs=[
            {
                "name": "scripted_heuristic",
                "class_name": "scripted_heuristic",
                "seed": 0,
                "checkpoint_step": None,
                "weights_uri": None,
                "extra": {"uid": "partner", "target_xy": "0.0,0.0", "action_dim": "2"},
            }
        ],
        repro_command="chamber-eval run --task mpe_cooperative_push",
    )


class TestSchemaVersion:
    def test_schema_version_is_three(self) -> None:
        """ADR-028 §Decision 1: SCHEMA_VERSION bumped 2 → 3."""
        assert SCHEMA_VERSION == 3

    def test_result_bundle_defaults_to_current_version(self) -> None:
        assert _bundle().schema_version == 3

    def test_round_trip(self) -> None:
        bundle = _bundle()
        assert ResultBundle.model_validate(bundle.model_dump()) == bundle

    def test_extra_forbidden(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="bogus"):
            ResultBundle(**{**_bundle().model_dump(), "bogus": 1})  # type: ignore[arg-type]


class TestLoadRunArchive:
    def test_v2_fixture_loads_as_spike_run(self) -> None:
        """ADR-028 §Decision 4: v2 archives readable forever, read-only."""
        loaded = load_run_archive(_V2_FIXTURE)
        assert isinstance(loaded, SpikeRun)
        assert loaded.schema_version == 2
        assert loaded.sub_stage == "1b"
        assert len(loaded.episode_results) == 20

    def test_v2_fixture_is_a_verbatim_archive_copy(self) -> None:
        """The fixture mirrors the immutable original byte for byte (I8)."""
        original = (
            Path(__file__).parents[2]
            / "spikes/results/stage1-failure-investigation/2026-05-21-post-widening"
            / "spike_as_hetero_widened_FAILED.json"
        )
        assert _V2_FIXTURE.read_bytes() == original.read_bytes()

    def test_v3_bundle_loads_as_result_bundle(self, tmp_path: Path) -> None:
        path = tmp_path / "bundle.json"
        path.write_text(_bundle().model_dump_json(), encoding="utf-8")
        assert isinstance(load_run_archive(path), ResultBundle)

    def test_versionless_payload_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "junk.json"
        path.write_text(json.dumps({"hello": 1}), encoding="utf-8")
        with pytest.raises(ValueError, match="schema_version"):
            load_run_archive(path)


class TestWriteVerify:
    def test_fresh_bundle_verifies(self, tmp_path: Path) -> None:
        _write_test_bundle(tmp_path / "b")
        rows = verify_bundle_dir(tmp_path / "b", repo_path=Path.cwd())
        failing = [r for r in rows if not r.ok]
        assert not failing, f"unexpected FAIL rows: {failing}"

    def test_refuses_non_empty_directory(self, tmp_path: Path) -> None:
        """Bundles are immutable evidence (I8) — no in-place overwrites."""
        target = tmp_path / "b"
        target.mkdir()
        (target / "stale.txt").write_text("x", encoding="utf-8")
        with pytest.raises(FileExistsError, match="non-empty"):
            _write_test_bundle(target)

    def test_dirty_bundle_fails_eligibility(self, tmp_path: Path) -> None:
        _write_test_bundle(tmp_path / "b", bundle=_bundle(dirty=True))
        rows = verify_bundle_dir(tmp_path / "b", repo_path=Path.cwd())
        eligibility = next(r for r in rows if r.name == "bundle:eligibility")
        assert not eligibility.ok
        relaxed = verify_bundle_dir(tmp_path / "b", repo_path=Path.cwd(), allow_dirty=True)
        assert all(r.ok for r in relaxed)

    def test_v2_archive_is_not_a_verifiable_bundle(self, tmp_path: Path) -> None:
        """ADR-028 §Decision 4: v2 = historical input, not an error — but not admissible."""
        target = tmp_path / "b"
        target.mkdir()
        (target / "bundle.json").write_text(_V2_FIXTURE.read_text(encoding="utf-8"))
        rows = verify_bundle_dir(target, repo_path=Path.cwd())
        assert len(rows) == 1
        assert not rows[0].ok
        assert "historical" in rows[0].detail

    def test_unmanifested_file_fails_membership(self, tmp_path: Path) -> None:
        _write_test_bundle(tmp_path / "b")
        (tmp_path / "b" / "smuggled.txt").write_text("x", encoding="utf-8")
        rows = verify_bundle_dir(tmp_path / "b", repo_path=Path.cwd())
        membership = next(r for r in rows if r.name == "manifest:membership")
        assert not membership.ok


class TestSummary:
    def test_recompute_is_deterministic(self) -> None:
        """ADR-002 P6: same episodes + pinned bootstrap → identical numbers."""
        episodes = _episodes(0) + _episodes(1)
        first = compute_summary(episodes, n_resamples=200, bootstrap_root_seed=7)
        second = compute_summary(episodes, n_resamples=200, bootstrap_root_seed=7)
        assert first == second

    def test_summary_is_a_pure_function_of_its_pins(self) -> None:
        episodes = _episodes(0) + _episodes(1)
        assert compute_summary(episodes, n_resamples=200, bootstrap_root_seed=7) != compute_summary(
            episodes, n_resamples=200, bootstrap_root_seed=8
        )

    def test_summary_round_trip(self) -> None:
        summary = compute_summary(_episodes(0))
        assert BundleSummary.model_validate_json(summary.model_dump_json()) == summary


class TestDirtyTreeRefusal:
    @pytest.fixture
    def force_provenance(self, monkeypatch: pytest.MonkeyPatch) -> Callable[[GitProvenance], None]:
        def _set(provenance: GitProvenance) -> None:
            monkeypatch.setattr(eval_run_module, "git_provenance", lambda _repo: provenance)

        return _set

    def test_run_refuses_dirty_tree(
        self,
        tmp_path: Path,
        force_provenance: Callable[[GitProvenance], None],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from chamber.cli import eval as eval_cli

        force_provenance(GitProvenance(sha="a" * 40, dirty=True))
        rc = eval_cli.main(
            [
                "run",
                "--task",
                "mpe_cooperative_push",
                "--policy",
                "random",
                "--partner",
                "scripted_heuristic",
                "--seeds",
                "1",
                "--episodes",
                "1",
                "--out",
                str(tmp_path / "b"),
            ]
        )
        assert rc == DIRTY_TREE_EXIT_CODE
        assert "dirty" in capsys.readouterr().err
        assert not (tmp_path / "b" / "bundle.json").exists()

    def test_allow_dirty_marks_bundle_ineligible(
        self,
        tmp_path: Path,
        force_provenance: Callable[[GitProvenance], None],
    ) -> None:
        from chamber.cli import eval as eval_cli

        force_provenance(GitProvenance(sha="a" * 40, dirty=True))
        rc = eval_cli.main(
            [
                "run",
                "--task",
                "mpe_cooperative_push",
                "--policy",
                "random",
                "--partner",
                "scripted_heuristic",
                "--seeds",
                "1",
                "--episodes",
                "1",
                "--out",
                str(tmp_path / "b"),
                "--allow-dirty",
            ]
        )
        assert rc == 0
        loaded = load_run_archive(tmp_path / "b" / "bundle.json")
        assert isinstance(loaded, ResultBundle)
        assert loaded.dirty is True
