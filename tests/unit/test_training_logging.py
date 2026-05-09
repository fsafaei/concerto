# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``concerto.training.logging`` (T4b.10).

Covers ADR-002 §Decisions (structured logs + W&B + JSONL fallback) and
plan/05 §2 (every line carries run_id / seed / git_sha / pyproject_hash / step).
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from concerto.training.logging import (
    GIT_SHA_UNKNOWN,
    RunContext,
    bind_run_logger,
    compute_run_metadata,
)


class _FakeWandb:
    """In-memory :class:`WandbSink` for offline tests (plan/05 §2)."""

    def __init__(self) -> None:
        self.calls: list[tuple[dict[str, object], int | None]] = []
        self.config: dict[str, object] = {}

    def log(self, data, *, step=None):  # type: ignore[no-untyped-def]
        # Snapshot the dict so later mutations don't bleed in.
        self.calls.append((dict(data), step))

    def set_config(self, config):  # type: ignore[no-untyped-def]
        self.config = dict(config)


@pytest.fixture
def repo_root(tmp_path: Path) -> Path:
    """A throwaway "project root" with a synthetic pyproject.toml."""
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'fake'\n", encoding="utf-8")
    return tmp_path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


class TestComputeRunMetadata:
    def test_run_id_is_16_hex_chars(self, repo_root: Path) -> None:
        """plan/05 §2: run_id matches PartnerSpec.partner_id keyspace (16 hex)."""
        ctx = compute_run_metadata(seed=0, run_kind="empirical_guarantee", repo_root=repo_root)
        assert re.fullmatch(r"[0-9a-f]{16}", ctx.run_id)

    def test_run_id_is_stable_across_calls(self, repo_root: Path) -> None:
        """ADR-002 §Decisions; P6: identical inputs → identical run_id."""
        a = compute_run_metadata(seed=7, run_kind="zoo_seed", repo_root=repo_root)
        b = compute_run_metadata(seed=7, run_kind="zoo_seed", repo_root=repo_root)
        assert a.run_id == b.run_id

    def test_run_id_changes_with_seed(self, repo_root: Path) -> None:
        a = compute_run_metadata(seed=0, run_kind="x", repo_root=repo_root)
        b = compute_run_metadata(seed=1, run_kind="x", repo_root=repo_root)
        assert a.run_id != b.run_id

    def test_run_id_changes_with_run_kind(self, repo_root: Path) -> None:
        """plan/05 §2: distinct kinds at the same seed don't collide."""
        a = compute_run_metadata(seed=0, run_kind="empirical_guarantee", repo_root=repo_root)
        b = compute_run_metadata(seed=0, run_kind="zoo_seed", repo_root=repo_root)
        assert a.run_id != b.run_id

    def test_run_id_changes_with_pyproject(self, repo_root: Path) -> None:
        """ADR-002 §Decisions: silent dependency drift produces a fresh run_id."""
        a = compute_run_metadata(seed=0, run_kind="x", repo_root=repo_root)
        (repo_root / "pyproject.toml").write_text(
            "[project]\nname = 'fake'\nversion = '0.0.1'\n", encoding="utf-8"
        )
        b = compute_run_metadata(seed=0, run_kind="x", repo_root=repo_root)
        assert a.run_id != b.run_id

    def test_pyproject_hash_is_full_sha256(self, repo_root: Path) -> None:
        """plan/05 §2: pyproject_hash is the *full* SHA-256 hex digest."""
        ctx = compute_run_metadata(seed=0, run_kind="x", repo_root=repo_root)
        assert re.fullmatch(r"[0-9a-f]{64}", ctx.pyproject_hash)

    def test_missing_pyproject_falls_back_to_empty_sha(self, tmp_path: Path) -> None:
        """plan/05 §2: missing pyproject.toml is sentinel, not crash."""
        ctx = compute_run_metadata(seed=0, run_kind="x", repo_root=tmp_path)
        # SHA-256 of empty bytestring.
        assert (
            ctx.pyproject_hash == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        )

    def test_git_sha_unknown_when_not_a_repo(self, tmp_path: Path) -> None:
        """plan/05 §2: outside a git repo, git_sha sentinel is loud, not None."""
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'fake'\n", encoding="utf-8")
        ctx = compute_run_metadata(seed=0, run_kind="x", repo_root=tmp_path)
        assert ctx.git_sha == GIT_SHA_UNKNOWN

    def test_extra_metadata_round_trips(self, repo_root: Path) -> None:
        """plan/05 §2: free-form metadata attaches to every line via the ctx."""
        ctx = compute_run_metadata(
            seed=0,
            run_kind="x",
            repo_root=repo_root,
            extra={"task": "stage0_smoke", "partner": "scripted_heuristic"},
        )
        assert ctx.extra == {"task": "stage0_smoke", "partner": "scripted_heuristic"}

    def test_run_context_is_frozen(self) -> None:
        """Immutability: bound logger context cannot drift mid-run."""
        import dataclasses

        ctx = RunContext(
            run_id="0" * 16,
            seed=0,
            git_sha="abc",
            pyproject_hash="0" * 64,
            run_kind="x",
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            ctx.seed = 1  # type: ignore[misc]


class TestJSONLFallback:
    def test_jsonl_emits_one_object_per_line(self, repo_root: Path, tmp_path: Path) -> None:
        """plan/05 §2: every JSONL line is a self-contained valid JSON object."""
        ctx = compute_run_metadata(seed=0, run_kind="x", repo_root=repo_root)
        jsonl = tmp_path / "logs" / "run.jsonl"
        logger = bind_run_logger(ctx, jsonl_path=jsonl)
        logger.info("epoch_done", step=0, ego_reward=-1.5)
        logger.info("epoch_done", step=1, ego_reward=-1.2)

        lines = _read_jsonl(jsonl)
        assert len(lines) == 2
        for line in lines:
            assert isinstance(line, dict)

    def test_jsonl_each_line_carries_run_context(self, repo_root: Path, tmp_path: Path) -> None:
        """plan/05 §2: every line carries run_id / seed / git_sha / pyproject_hash."""
        ctx = compute_run_metadata(seed=0, run_kind="x", repo_root=repo_root)
        jsonl = tmp_path / "logs" / "run.jsonl"
        logger = bind_run_logger(ctx, jsonl_path=jsonl)
        logger.info("epoch_done", step=0)

        line = _read_jsonl(jsonl)[0]
        for key in ("run_id", "seed", "git_sha", "pyproject_hash", "run_kind", "step"):
            assert key in line
        assert line["run_id"] == ctx.run_id
        assert line["seed"] == ctx.seed
        assert line["git_sha"] == ctx.git_sha
        assert line["pyproject_hash"] == ctx.pyproject_hash

    def test_jsonl_extra_metadata_is_present_on_every_line(
        self, repo_root: Path, tmp_path: Path
    ) -> None:
        """plan/05 §2: ctx.extra fields propagate to every emitted line."""
        ctx = compute_run_metadata(
            seed=0,
            run_kind="x",
            repo_root=repo_root,
            extra={"task": "mpe_cooperative_push"},
        )
        jsonl = tmp_path / "run.jsonl"
        logger = bind_run_logger(ctx, jsonl_path=jsonl)
        logger.info("epoch_done", step=0)
        assert _read_jsonl(jsonl)[0]["task"] == "mpe_cooperative_push"

    def test_jsonl_parent_directory_is_created(self, repo_root: Path, tmp_path: Path) -> None:
        """plan/05 §2: missing parent dir does not crash the run."""
        ctx = compute_run_metadata(seed=0, run_kind="x", repo_root=repo_root)
        jsonl = tmp_path / "deeply" / "nested" / "path" / "run.jsonl"
        logger = bind_run_logger(ctx, jsonl_path=jsonl)
        logger.info("event", step=0)
        assert jsonl.exists()

    def test_jsonl_handles_non_json_serialisable_values(
        self, repo_root: Path, tmp_path: Path
    ) -> None:
        """plan/05 §2: an unserialisable value falls back to str(...) — does NOT crash."""
        ctx = compute_run_metadata(seed=0, run_kind="x", repo_root=repo_root)
        jsonl = tmp_path / "run.jsonl"
        logger = bind_run_logger(ctx, jsonl_path=jsonl)

        class _Custom:
            def __repr__(self) -> str:
                return "<Custom>"

        logger.info("event", step=0, weird=_Custom())
        line = _read_jsonl(jsonl)[0]
        assert line["weird"] == "<Custom>"


class TestWandbSink:
    def test_wandb_sink_receives_event_with_step(self, repo_root: Path, tmp_path: Path) -> None:
        """plan/05 §2: W&B sink receives the metric event, step extracted."""
        ctx = compute_run_metadata(seed=0, run_kind="x", repo_root=repo_root)
        sink = _FakeWandb()
        logger = bind_run_logger(ctx, jsonl_path=tmp_path / "run.jsonl", wandb_sink=sink)
        logger.info("epoch_done", step=42, ego_reward=-1.0)
        assert len(sink.calls) == 1
        data, step = sink.calls[0]
        assert step == 42
        assert data["ego_reward"] == -1.0

    def test_wandb_sink_step_is_none_when_not_provided(
        self, repo_root: Path, tmp_path: Path
    ) -> None:
        """plan/05 §2: events without ``step`` (e.g. setup events) pass step=None."""
        ctx = compute_run_metadata(seed=0, run_kind="x", repo_root=repo_root)
        sink = _FakeWandb()
        logger = bind_run_logger(ctx, jsonl_path=tmp_path / "run.jsonl", wandb_sink=sink)
        logger.info("setup_complete")
        assert sink.calls[0][1] is None

    def test_wandb_sink_does_not_swallow_jsonl_writes(
        self, repo_root: Path, tmp_path: Path
    ) -> None:
        """ADR-002 §Decisions: chaining preserves the JSONL fallback."""
        ctx = compute_run_metadata(seed=0, run_kind="x", repo_root=repo_root)
        sink = _FakeWandb()
        jsonl = tmp_path / "run.jsonl"
        logger = bind_run_logger(ctx, jsonl_path=jsonl, wandb_sink=sink)
        logger.info("event", step=0)
        # Both sinks fire; the JSONL line is still well-formed.
        assert len(sink.calls) == 1
        assert _read_jsonl(jsonl)[0]["event"] == "event"

    def test_no_wandb_sink_means_jsonl_only(self, repo_root: Path, tmp_path: Path) -> None:
        """plan/05 §2: W&B is opt-in; default is JSONL-only (offline-friendly)."""
        ctx = compute_run_metadata(seed=0, run_kind="x", repo_root=repo_root)
        jsonl = tmp_path / "run.jsonl"
        logger = bind_run_logger(ctx, jsonl_path=jsonl)
        logger.info("event", step=0)
        assert _read_jsonl(jsonl)[0]["event"] == "event"


class TestPublicSurface:
    def test_module_exports(self) -> None:
        """The public surface includes the four symbols the trainer touches."""
        from concerto.training import logging as logging_mod

        for name in (
            "GIT_SHA_UNKNOWN",
            "RunContext",
            "WandbSink",
            "bind_run_logger",
            "compute_run_metadata",
        ):
            assert hasattr(logging_mod, name)
