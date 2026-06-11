# SPDX-License-Identifier: Apache-2.0
"""Unit tests for :mod:`chamber.cli.analyze` (P1.05.11; ADR-017 §Decisions).

Includes the **real-archive smoke** the founder named in §3.5 commit 5
acceptance: ``chamber-analyze list-runs`` against
``spikes/results/stage1-failure-investigation/2026-05-20`` must return
five cells, each with a non-null ``terminal_success_rate``. This
exercises the per-cell JSONL walk + envelope join end-to-end against
committed data.

Other subcommands are covered with synthetic fixtures so the tests
don't depend on a working SAPIEN host.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from chamber.cli.analyze import main

if TYPE_CHECKING:  # pragma: no cover
    from pathlib import Path


_REAL_ARCHIVE_ROOT = "spikes/results/stage1-failure-investigation/2026-05-20"


def _make_synthetic_archive(root: Path, *, run_id: str, seed: int, success: bool) -> None:
    """Write a minimal ``<run_id>.jsonl`` + envelope under ``root``."""
    jsonl = root / f"{run_id}.jsonl"
    jsonl.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "event": "training_start",
                        "run_id": run_id,
                        "seed": seed,
                        "git_sha": "deadbeef" * 5,
                        "pyproject_hash": "ab" * 32,
                        "run_kind": "ego_aht_happo",
                        "task": "stage1_pickplace",
                        "level": "info",
                    }
                ),
                json.dumps(
                    {
                        "event": "scalar",
                        "metric_namespace": "train",
                        "run_id": run_id,
                        "step": 1024,
                        "policy_loss": 0.5,
                        "dist_entropy": 1.2,
                    }
                ),
                json.dumps(
                    {
                        "event": "scalar",
                        "metric_namespace": "train",
                        "run_id": run_id,
                        "step": 2048,
                        "policy_loss": 0.4,
                        "dist_entropy": 1.1,
                    }
                ),
                json.dumps(
                    {
                        "event": "rollout_update",
                        "run_id": run_id,
                        "step": 2048,
                        "last_reward": 0.003,
                    }
                ),
                json.dumps(
                    {
                        "event": "safety_telemetry_final",
                        "run_id": run_id,
                        "lambda_steady_state": -7.0,
                        "lambda_mean": -6.5,
                    }
                ),
                json.dumps(
                    {
                        "event": "training_end",
                        "run_id": run_id,
                        "n_episodes": 1,
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    envelope = root / "spike_as.json"
    envelope.write_text(
        json.dumps(
            {
                "spike_id": "synthetic",
                "schema_version": 2,
                "sub_stage": "1b",
                "axis": "AS",
                "condition_pair": ["c1", "c2"],
                "prereg_sha": "abc",
                "git_tag": "synthetic",
                "seeds": [seed],
                "episode_results": [
                    {
                        "seed": seed,
                        "episode_idx": 0,
                        "initial_state_seed": 0,
                        "success": success,
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


class TestListRunsRealArchive:
    """Founder's commit-5 acceptance smoke (ADR-017 §Decisions)."""

    def test_returns_five_cells_with_non_null_success(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """`chamber-analyze list-runs ...stage1-failure-investigation/2026-05-20 --json` → 5 cells.

        Each cell must have a non-null ``terminal_success_rate``. The
        envelope's ``episode_results`` are joined by ``seed`` per
        :func:`chamber.cli.analyze._terminal_success_rate_for_seed`.
        """
        rc = main(["list-runs", "--archive-root", _REAL_ARCHIVE_ROOT, "--json"])
        assert rc == 0
        out = capsys.readouterr().out
        rows = json.loads(out)
        assert isinstance(rows, list)
        assert len(rows) == 5, f"expected 5 cells, got {len(rows)}"
        for row in rows:
            assert row["terminal_success_rate"] is not None, (
                f"run_id={row['run_id']} has null terminal_success_rate; "
                "the envelope-join in chamber-analyze regressed."
            )
            # Sanity: pre-P1.05.8 archives still expose the seed.
            assert row["seed"] is not None
            assert row["sub_stage"] == "1b"


class TestListRunsSynthetic:
    def test_emits_table_human_readable(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _make_synthetic_archive(tmp_path, run_id="0" * 16, seed=0, success=True)
        rc = main(["list-runs", "--archive-root", str(tmp_path)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "run_id" in out
        assert "0000000000000000" in out

    def test_empty_archive_returns_zero(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        rc = main(["list-runs", "--archive-root", str(tmp_path), "--json"])
        assert rc == 0
        out = capsys.readouterr().out
        assert json.loads(out) == []


class TestSummary:
    def test_summary_for_known_run(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _make_synthetic_archive(tmp_path, run_id="a" * 16, seed=7, success=True)
        rc = main(
            [
                "summary",
                "a" * 16,
                "--archive-root",
                str(tmp_path),
                "--json",
            ]
        )
        assert rc == 0
        out = json.loads(capsys.readouterr().out)
        assert out["run_id"] == "a" * 16
        assert out["seed"] == 7
        assert out["terminal_success_rate"] == 1.0

    def test_unknown_run_returns_two(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _make_synthetic_archive(tmp_path, run_id="a" * 16, seed=0, success=True)
        rc = main(
            [
                "summary",
                "b" * 16,
                "--archive-root",
                str(tmp_path),
                "--json",
            ]
        )
        assert rc == 2


class TestMetrics:
    def test_dumps_time_series_for_one_metric(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _make_synthetic_archive(tmp_path, run_id="c" * 16, seed=0, success=True)
        rc = main(
            [
                "metrics",
                "c" * 16,
                "--archive-root",
                str(tmp_path),
                "--metric",
                "policy_loss",
                "--namespace",
                "train",
                "--json",
            ]
        )
        assert rc == 0
        rows = json.loads(capsys.readouterr().out)
        assert len(rows) == 2
        assert rows[0]["step"] == 1024
        assert rows[0]["value"] == pytest.approx(0.5)
        assert rows[1]["step"] == 2048
        assert rows[1]["value"] == pytest.approx(0.4)


class TestCompare:
    def test_side_by_side_default_metrics(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _make_synthetic_archive(tmp_path, run_id="d" * 16, seed=0, success=True)
        # Append a second run_id+envelope by adjusting in-place.
        rc = main(
            [
                "compare",
                "d" * 16,
                "--archive-root",
                str(tmp_path),
                "--json",
            ]
        )
        assert rc == 0
        rows = json.loads(capsys.readouterr().out)
        assert len(rows) == 1
        assert rows[0]["run_id"] == "d" * 16
        assert "terminal_success_rate" in rows[0]


class TestRolloutFrames:
    def test_dumps_per_step_records(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        rollouts = tmp_path / "rollouts" / "as-hetero"
        rollouts.mkdir(parents=True)
        sidecar = rollouts / "step_001000.jsonl"
        sidecar.write_text(
            "\n".join(
                json.dumps(
                    {
                        "event": "rollout_step",
                        "metric_namespace": "rollout",
                        "step_global": 1000 + i,
                        "step_episode": i,
                        "obs_summary": {"cube_pose": [1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0]},
                        "action": [0.0] * 8,
                        "reward": float(i),
                        "terminated": False,
                        "truncated": False,
                        "info": {},
                    }
                )
                for i in range(3)
            ),
            encoding="utf-8",
        )
        rc = main(
            [
                "rollout-frames",
                "ignored",
                "--archive-root",
                str(tmp_path),
            ]
        )
        assert rc == 0
        rows = json.loads(capsys.readouterr().out)
        assert len(rows) == 3
        assert rows[0]["step_global"] == 1000


class TestPlot:
    def test_writes_png(
        self,
        tmp_path: Path,
    ) -> None:
        # Synthetic archive with scalar lines.
        _make_synthetic_archive(tmp_path, run_id="e" * 16, seed=0, success=True)
        out_path = tmp_path / "plot.png"
        rc = main(
            [
                "plot",
                "e" * 16,
                "--archive-root",
                str(tmp_path),
                "--metric",
                "policy_loss",
                "--namespace",
                "train",
                "--out",
                str(out_path),
            ]
        )
        # Plot may degrade if matplotlib isn't installed (returns 2);
        # accept either path. The PNG should exist when matplotlib is
        # present.
        assert rc in (0, 2)
        if rc == 0:
            assert out_path.exists()
            assert out_path.stat().st_size > 0
