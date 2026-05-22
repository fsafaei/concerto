# SPDX-License-Identifier: Apache-2.0
"""Unit tests for :mod:`scripts.wandb_backfill` (P1.05.11; ADR-017 §Decisions D7).

Tests cover the dry-run target discovery and tag-building logic against
synthetic and real (committed) archives. The actual ``wandb.init`` /
``wandb.log`` calls are NOT exercised here — those are covered by the
integration test in ``tests/integration/test_ppo_observability.py``
(commit 9) which uses ``WANDB_MODE=offline`` end-to-end.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_REAL_ARCHIVE_ROOT = Path("spikes/results/stage1-failure-investigation/2026-05-20")
_BACKFILL_SCRIPT = Path("scripts/wandb_backfill.py")


@pytest.fixture(scope="module")
def backfill_module():  # type: ignore[no-untyped-def]
    """Load wandb_backfill.py as a module from its path on disk."""
    import sys

    spec = importlib.util.spec_from_file_location("wandb_backfill", _BACKFILL_SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before exec so dataclass module-resolution sees it.
    sys.modules["wandb_backfill"] = module
    spec.loader.exec_module(module)
    return module


class TestDryRunRealArchive:
    """Founder commit-6 acceptance smoke (ADR-017 §Decisions D7)."""

    def test_dry_run_returns_zero_against_committed_archive(
        self,
        capsys: pytest.CaptureFixture[str],
        backfill_module,  # type: ignore[no-untyped-def]
    ) -> None:
        rc = backfill_module.main(
            [
                "--archive-root",
                str(_REAL_ARCHIVE_ROOT),
                "--dry-run",
                "--project",
                "concerto-chamber",
            ]
        )
        assert rc == 0
        out = capsys.readouterr().out
        # 5 [would replay] lines — one per committed .jsonl cell.
        lines = [ln for ln in out.splitlines() if ln.startswith("[would replay]")]
        assert len(lines) == 5, f"expected 5 dry-run targets, got {len(lines)}"

    def test_dry_run_tags_include_sub_stage_axis_prereg_backfill(
        self,
        capsys: pytest.CaptureFixture[str],
        backfill_module,  # type: ignore[no-untyped-def]
    ) -> None:
        rc = backfill_module.main(
            [
                "--archive-root",
                str(_REAL_ARCHIVE_ROOT),
                "--dry-run",
                "--project",
                "concerto-chamber",
            ]
        )
        assert rc == 0
        out = capsys.readouterr().out
        # ADR-017 §Decisions: prereg_sha appears as a tag (short-form)
        # AND in wandb.config (full SHA). Tags surface in the dry-run
        # text; config payload is verified by the integration test.
        assert "sub_stage:1b" in out
        assert "stage:1" in out
        assert "axis:AS" in out
        assert "prereg:29e397a4" in out
        assert "backfill:true" in out


class TestDryRunMissingArchive:
    def test_returns_two_when_archive_root_missing(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        backfill_module,  # type: ignore[no-untyped-def]
    ) -> None:
        rc = backfill_module.main(
            [
                "--archive-root",
                str(tmp_path / "does_not_exist"),
                "--dry-run",
                "--project",
                "concerto-chamber",
            ]
        )
        assert rc == 2


class TestEnvelopeOnlyTargets:
    """--include-local exercises envelope-only replay for .local/*.json snapshots."""

    def test_envelope_only_picked_up_with_include_local(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        backfill_module,  # type: ignore[no-untyped-def]
    ) -> None:
        # Build a synthetic envelope-only snapshot under tmp_path.
        snapshot = tmp_path / "synthetic_snapshot.json"
        snapshot.write_text(
            json.dumps(
                {
                    "spike_id": "synthetic_p1058_smoke",
                    "schema_version": 2,
                    "sub_stage": "1b",
                    "axis": "AS",
                    "condition_pair": ["c1"],
                    "prereg_sha": "deadbeef" * 5,
                    "git_tag": "synthetic",
                    "seeds": [0],
                    "episode_results": [
                        {
                            "seed": 0,
                            "episode_idx": 0,
                            "initial_state_seed": 0,
                            "success": False,
                            "constraint_violation_peak": 0.0,
                            "fallback_fired": 0,
                            "force_peak": None,
                            "metadata": {},
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        rc = backfill_module.main(
            [
                "--archive-root",
                str(tmp_path),
                "--include-local",
                "--dry-run",
                "--project",
                "concerto-chamber",
            ]
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert "[would replay envelope-only]" in out
        assert "synthetic_p1058_smoke" in out
