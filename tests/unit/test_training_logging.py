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

from concerto.training.config import WandbConfig
from concerto.training.logging import (
    GIT_SHA_UNKNOWN,
    RunContext,
    bind_run_logger,
    compute_run_metadata,
    log_eval,
    log_scalars,
    make_wandb_run_sink,
)


class _FakeWandb:
    """In-memory :class:`WandbSink` for offline tests (plan/05 §2; ADR-017 §Decisions)."""

    def __init__(self) -> None:
        self.calls: list[tuple[dict[str, object], int | None]] = []
        self.config: dict[str, object] = {}
        self.tags: list[str] = []
        self.closed: bool = False

    def log(self, data, *, step=None):  # type: ignore[no-untyped-def]
        # Snapshot the dict so later mutations don't bleed in.
        self.calls.append((dict(data), step))

    def set_config(self, config):  # type: ignore[no-untyped-def]
        self.config = dict(config)

    def add_tags(self, tags):  # type: ignore[no-untyped-def]
        # Idempotent: preserve order, dedupe.
        for tag in tags:
            if tag not in self.tags:
                self.tags.append(tag)

    def close(self):  # type: ignore[no-untyped-def]
        self.closed = True


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

    def test_run_id_changes_with_config_fingerprint(self, repo_root: Path) -> None:
        """RUNID-COLLISION fix (issue #214; ADR-002 §Revision history 2026-06-10).

        Same-seed same-commit runs differing only in config must get
        distinct run_ids — the 2026-06-10 regime-alignment chain's A1/A2
        seed-0 cells (differing only in gamma) collided without this.
        """
        a = compute_run_metadata(
            seed=0, run_kind="x", repo_root=repo_root, config_fingerprint="aaaa"
        )
        b = compute_run_metadata(
            seed=0, run_kind="x", repo_root=repo_root, config_fingerprint="bbbb"
        )
        assert a.run_id != b.run_id

    def test_run_id_stable_for_same_config_fingerprint(self, repo_root: Path) -> None:
        """P6-adjacent: identical (seed, commit, kind, config) → identical run_id."""
        a = compute_run_metadata(
            seed=0, run_kind="x", repo_root=repo_root, config_fingerprint="aaaa"
        )
        b = compute_run_metadata(
            seed=0, run_kind="x", repo_root=repo_root, config_fingerprint="aaaa"
        )
        assert a.run_id == b.run_id

    def test_none_fingerprint_preserves_legacy_run_id(self, repo_root: Path) -> None:
        """ADR-002 §Revision history 2026-06-10: ``None`` keeps the legacy hash material.

        Config-less callers' historical run_ids must remain reproducible
        — the fingerprint is folded in only when supplied.
        """
        legacy = compute_run_metadata(seed=3, run_kind="zoo_seed", repo_root=repo_root)
        explicit_none = compute_run_metadata(
            seed=3, run_kind="zoo_seed", repo_root=repo_root, config_fingerprint=None
        )
        fingerprinted = compute_run_metadata(
            seed=3, run_kind="zoo_seed", repo_root=repo_root, config_fingerprint="cccc"
        )
        assert legacy.run_id == explicit_none.run_id
        assert legacy.run_id != fingerprinted.run_id

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


class TestLogScalars:
    """P1.05.11 / ADR-017 §Schema: namespaced scalar emission helper."""

    def test_emits_event_scalar_with_namespace(self, repo_root: Path, tmp_path: Path) -> None:
        """ADR-017 §Schema: log_scalars pins event=scalar + metric_namespace."""
        ctx = compute_run_metadata(seed=0, run_kind="x", repo_root=repo_root)
        jsonl = tmp_path / "run.jsonl"
        logger = bind_run_logger(ctx, jsonl_path=jsonl)

        log_scalars(logger, step=100, namespace="train", policy_loss=0.5, dist_entropy=1.2)

        line = _read_jsonl(jsonl)[0]
        assert line["event"] == "scalar"
        assert line["metric_namespace"] == "train"
        assert line["step"] == 100
        assert line["policy_loss"] == 0.5
        assert line["dist_entropy"] == 1.2
        assert "wall_time" in line  # ISO-8601 stamp

    def test_namespace_outside_allow_list_raises(self, repo_root: Path, tmp_path: Path) -> None:
        """ADR-017 §Schema: typo in namespace fails loud — silent typos break chamber-analyze."""
        ctx = compute_run_metadata(seed=0, run_kind="x", repo_root=repo_root)
        logger = bind_run_logger(ctx, jsonl_path=tmp_path / "run.jsonl")

        with pytest.raises(ValueError, match=r"namespace=.* not in allow-list"):
            log_scalars(logger, step=0, namespace="trian", policy_loss=0.5)  # type: ignore[arg-type]

    def test_each_namespace_in_allow_list_accepted(self, repo_root: Path, tmp_path: Path) -> None:
        """ADR-017 §Schema: all five namespaces fire without error."""
        ctx = compute_run_metadata(seed=0, run_kind="x", repo_root=repo_root)
        logger = bind_run_logger(ctx, jsonl_path=tmp_path / "run.jsonl")

        for ns in ("train", "eval", "safety", "hardware", "rollout"):
            log_scalars(logger, step=0, namespace=ns, x=1.0)  # type: ignore[arg-type]

    def test_wandb_sink_receives_scalar(self, repo_root: Path, tmp_path: Path) -> None:
        """ADR-017 §Decisions: scalars flow to both JSONL and W&B sink."""
        ctx = compute_run_metadata(seed=0, run_kind="x", repo_root=repo_root)
        sink = _FakeWandb()
        logger = bind_run_logger(ctx, jsonl_path=tmp_path / "run.jsonl", wandb_sink=sink)

        log_scalars(logger, step=42, namespace="train", policy_loss=0.3)

        assert len(sink.calls) == 1
        data, step = sink.calls[0]
        assert step == 42
        assert data["policy_loss"] == 0.3
        assert data["metric_namespace"] == "train"


class TestLogEval:
    """P1.05.11 / ADR-017 §Schema: eval-cell emission helper."""

    def test_emits_event_eval_with_condition(self, repo_root: Path, tmp_path: Path) -> None:
        ctx = compute_run_metadata(seed=0, run_kind="x", repo_root=repo_root)
        jsonl = tmp_path / "run.jsonl"
        logger = bind_run_logger(ctx, jsonl_path=jsonl)

        log_eval(
            logger,
            step=50000,
            condition="as-hetero",
            success_rate=0.0,
            mean_episode_length=50.0,
            mean_episode_reward=0.012,
            n_terminated=0,
            n_truncated=5,
        )

        line = _read_jsonl(jsonl)[0]
        assert line["event"] == "eval"
        assert line["metric_namespace"] == "eval"
        assert line["condition"] == "as-hetero"
        assert line["step"] == 50000
        assert line["success_rate"] == 0.0
        assert line["n_terminated"] == 0
        assert line["n_truncated"] == 5
        assert "wall_time" in line


class TestMakeWandbRunSink:
    """P1.05.11 / ADR-017 §Decisions: degrade-to-no-op factory paths."""

    def test_disabled_cfg_returns_none(self, repo_root: Path) -> None:
        """ADR-017 §Decisions: cfg.enabled=false → None, silent (operator opt-out)."""
        cfg = WandbConfig(enabled=False)
        ctx = compute_run_metadata(seed=0, run_kind="x", repo_root=repo_root)
        # Silent path — no warning, just None.
        assert make_wandb_run_sink(cfg, ctx) is None

    def test_missing_api_key_returns_none_with_warning(
        self, repo_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ADR-017 §Decisions: missing WANDB_API_KEY (online mode) → None + UserWarning."""
        cfg = WandbConfig(enabled=True, project="concerto-chamber")
        ctx = compute_run_metadata(seed=0, run_kind="x", repo_root=repo_root)
        monkeypatch.delenv("WANDB_API_KEY", raising=False)
        monkeypatch.delenv("WANDB_MODE", raising=False)

        with pytest.warns(UserWarning, match="WANDB_API_KEY missing"):
            result = make_wandb_run_sink(cfg, ctx)
        assert result is None

    def test_offline_mode_does_not_need_api_key(
        self, repo_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ADR-017 §Decisions: offline mode (WANDB_MODE=offline) bypasses the auth check.

        We don't actually call wandb.init in this test — the sink construction is
        verified by a fake `wandb.init` patched in via monkeypatch on the imported
        module, but the key insight is the auth check does not short-circuit.
        """
        import sys
        import types

        cfg = WandbConfig(enabled=True, project="concerto-chamber")
        ctx = compute_run_metadata(seed=0, run_kind="x", repo_root=repo_root)
        monkeypatch.delenv("WANDB_API_KEY", raising=False)
        monkeypatch.setenv("WANDB_MODE", "offline")

        # Patch a minimal wandb module with the methods the factory calls.
        fake_run = _FakeWandb()

        def _fake_init(**kwargs: object) -> _FakeWandb:
            del kwargs  # we only need to return the sink; args validated elsewhere
            return fake_run

        fake_wandb = types.SimpleNamespace(init=_fake_init)
        monkeypatch.setitem(sys.modules, "wandb", fake_wandb)  # type: ignore[arg-type]

        sink = make_wandb_run_sink(
            cfg,
            ctx,
            tags=("stage:1", "sub_stage:1b"),
            config_extras={"prereg_sha": "deadbeef"},
        )
        assert sink is not None  # offline path constructs the sink

    def test_init_raise_returns_none_with_warning(
        self,
        repo_root: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """ADR-017 §Decisions: wandb.init(...) raising degrades gracefully → None + warning."""
        import sys
        import types

        cfg = WandbConfig(enabled=True, project="concerto-chamber")
        ctx = compute_run_metadata(seed=0, run_kind="x", repo_root=repo_root)
        monkeypatch.setenv("WANDB_API_KEY", "fake")
        monkeypatch.delenv("WANDB_MODE", raising=False)

        def _raising_init(**kwargs: object) -> object:
            del kwargs
            raise RuntimeError("simulated wandb auth failure")

        monkeypatch.setitem(
            sys.modules,
            "wandb",
            types.SimpleNamespace(init=_raising_init),  # type: ignore[arg-type]
        )

        with pytest.warns(UserWarning, match="wandb.init"):
            result = make_wandb_run_sink(cfg, ctx)
        assert result is None


class TestPublicSurface:
    def test_module_exports(self) -> None:
        """Public surface: pre-P1.05.11 symbols + four new ones (ADR-017)."""
        from concerto.training import logging as logging_mod

        for name in (
            "GIT_SHA_UNKNOWN",
            "LogNamespace",
            "RunContext",
            "WandbSink",
            "bind_run_logger",
            "compute_run_metadata",
            "log_eval",
            "log_scalars",
            "make_wandb_run_sink",
        ):
            assert hasattr(logging_mod, name)
