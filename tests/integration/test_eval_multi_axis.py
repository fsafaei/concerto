# SPDX-License-Identifier: Apache-2.0
"""Integration test: ``chamber-eval`` multi-axis HRS bundle (ADR-008 §Decision; reviewer P1-3).

Hand-rolls three :class:`SpikeRun` archives covering the AS, OM, and
CM axes, then drives ``chamber-eval`` end-to-end. The rendered
leaderboard is compared against a golden Markdown file; a second
test exercises the ``--allow-duplicate-axes`` suffix-disambiguation
path with two AS-axis spikes. A third test confirms that duplicate
axes without the flag exit with status 2 (reviewer P1-3).

The golden file is generated deterministically from the run
patterns + bootstrap seed; regenerate it by copying the rendered
text from ``tmp_path / "actual.md"`` on intentional schema changes.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from chamber.cli.eval import main as eval_main
from chamber.evaluation import (
    ConditionPair,
    EpisodeResult,
    LeaderboardEntry,
    SpikeRun,
)

if TYPE_CHECKING:
    import pytest

_GOLDEN_DIR = Path(__file__).parent / "golden"
_GOLDEN_MULTI = _GOLDEN_DIR / "leaderboard_multi_axis.md"
_GOLDEN_DUP = _GOLDEN_DIR / "leaderboard_dup_as.md"


def _build_run(
    *,
    axis: str,
    spike_id: str,
    homo_pattern: list[bool],
    hetero_pattern: list[bool],
    seeds: list[int] | None = None,
) -> SpikeRun:
    seeds = seeds if seeds is not None else [11, 22, 33]
    homo_id = f"homo_{axis.lower()}"
    hetero_id = f"hetero_{axis.lower()}"
    episodes_per_seed = len(homo_pattern)
    assert len(hetero_pattern) == episodes_per_seed
    results: list[EpisodeResult] = []
    for seed in seeds:
        for idx in range(episodes_per_seed):
            init_seed = seed * 1000 + idx
            results.append(
                EpisodeResult(
                    seed=seed,
                    episode_idx=idx,
                    initial_state_seed=init_seed,
                    success=homo_pattern[idx],
                    constraint_violation_peak=0.0,
                    fallback_fired=0,
                    metadata={"condition": homo_id},
                )
            )
            results.append(
                EpisodeResult(
                    seed=seed,
                    episode_idx=idx,
                    initial_state_seed=init_seed,
                    success=hetero_pattern[idx],
                    constraint_violation_peak=0.0,
                    fallback_fired=0,
                    metadata={"condition": hetero_id},
                )
            )
    return SpikeRun(
        spike_id=spike_id,
        prereg_sha="0" * 40,
        git_tag=f"prereg/{spike_id}",
        axis=axis,
        condition_pair=ConditionPair(homogeneous_id=homo_id, heterogeneous_id=hetero_id),
        seeds=seeds,
        episode_results=results,
    )


def _write_run(path: Path, run: SpikeRun) -> Path:
    path.write_text(run.model_dump_json(indent=2), encoding="utf-8")
    return path


def _three_runs(tmp_path: Path) -> tuple[Path, Path, Path]:
    as_run = _build_run(
        axis="AS",
        spike_id="stage1_as",
        homo_pattern=[True, True, True, True, True],
        hetero_pattern=[True, True, False, False, False],
    )
    om_run = _build_run(
        axis="OM",
        spike_id="stage1_om",
        homo_pattern=[True, True, True, True, True],
        hetero_pattern=[True, False, False, False, False],
    )
    cm_run = _build_run(
        axis="CM",
        spike_id="stage2_cm",
        homo_pattern=[True, True, True, True, True],
        hetero_pattern=[True, True, True, False, False],
    )
    return (
        _write_run(tmp_path / "stage1_as.json", as_run),
        _write_run(tmp_path / "stage1_om.json", om_run),
        _write_run(tmp_path / "stage2_cm.json", cm_run),
    )


def test_multi_axis_pipeline_emits_full_hrs_vector(tmp_path: Path) -> None:
    as_path, om_path, cm_path = _three_runs(tmp_path)
    out_path = tmp_path / "entry.json"
    rc = eval_main(
        [
            str(as_path),
            str(om_path),
            str(cm_path),
            "--method-id",
            "concerto",
            "--seed",
            "7",
            "--n-resamples",
            "200",
            "--output",
            str(out_path),
        ],
    )
    assert rc == 0
    entry = LeaderboardEntry.model_validate_json(out_path.read_text(encoding="utf-8"))
    axes = [e.axis for e in entry.hrs_vector.entries]
    assert axes == ["CM", "OM", "AS"]
    assert entry.spike_runs == ["stage1_as", "stage1_om", "stage2_cm"]


def test_multi_axis_golden_leaderboard(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    as_path, om_path, cm_path = _three_runs(tmp_path)
    rc = eval_main(
        [
            str(as_path),
            str(om_path),
            str(cm_path),
            "--method-id",
            "concerto",
            "--seed",
            "7",
            "--n-resamples",
            "200",
        ],
    )
    assert rc == 0
    rendered = capsys.readouterr().out.rstrip("\n") + "\n"
    actual_path = tmp_path / "actual.md"
    actual_path.write_text(rendered, encoding="utf-8")

    assert _GOLDEN_MULTI.exists(), (
        f"missing golden file {_GOLDEN_MULTI}; regenerate by copying {actual_path}"
    )
    golden = _GOLDEN_MULTI.read_text(encoding="utf-8")
    assert rendered == golden, (
        "rendered multi-axis leaderboard drifted from golden file. Diff:\n"
        f"--- expected ({_GOLDEN_MULTI})\n{golden}\n"
        f"+++ actual ({actual_path})\n{rendered}"
    )
    assert "[PARTIAL:" not in rendered, "multi-axis row must not carry the PARTIAL marker"


def test_single_spike_marks_partial(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    as_path, _om_path, _cm_path = _three_runs(tmp_path)
    rc = eval_main(
        [
            str(as_path),
            "--method-id",
            "concerto",
            "--seed",
            "7",
            "--n-resamples",
            "200",
        ],
    )
    assert rc == 0
    rendered = capsys.readouterr().out
    assert "[PARTIAL: AS]" in rendered
    assert "`concerto`" in rendered


def test_duplicate_axes_rejected_by_default(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    as_a = _build_run(
        axis="AS",
        spike_id="stage1_as_a",
        homo_pattern=[True, True, True, True, True],
        hetero_pattern=[True, True, False, False, False],
    )
    as_b = _build_run(
        axis="AS",
        spike_id="stage1_as_b",
        homo_pattern=[True, True, True, True, True],
        hetero_pattern=[True, False, False, False, False],
    )
    path_a = _write_run(tmp_path / "as_a.json", as_a)
    path_b = _write_run(tmp_path / "as_b.json", as_b)
    rc = eval_main([str(path_a), str(path_b), "--method-id", "concerto"])
    assert rc == 2
    captured = capsys.readouterr()
    assert "duplicate axis IDs" in captured.err
    assert "AS" in captured.err
    assert "--allow-duplicate-axes" in captured.err


def test_duplicate_axes_allowed_suffixes_with_spike_id(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    as_a = _build_run(
        axis="AS",
        spike_id="stage1_as_a",
        homo_pattern=[True, True, True, True, True],
        hetero_pattern=[True, True, False, False, False],
    )
    as_b = _build_run(
        axis="AS",
        spike_id="stage1_as_b",
        homo_pattern=[True, True, True, True, True],
        hetero_pattern=[True, False, False, False, False],
    )
    path_a = _write_run(tmp_path / "as_a.json", as_a)
    path_b = _write_run(tmp_path / "as_b.json", as_b)
    rc = eval_main(
        [
            str(path_a),
            str(path_b),
            "--method-id",
            "concerto",
            "--allow-duplicate-axes",
            "--seed",
            "7",
            "--n-resamples",
            "200",
        ],
    )
    assert rc == 0
    rendered = capsys.readouterr().out.rstrip("\n") + "\n"
    actual_path = tmp_path / "actual.md"
    actual_path.write_text(rendered, encoding="utf-8")

    assert "AS_stage1_as_a=" in rendered
    assert "AS_stage1_as_b=" in rendered

    assert _GOLDEN_DUP.exists(), (
        f"missing golden file {_GOLDEN_DUP}; regenerate by copying {actual_path}"
    )
    golden = _GOLDEN_DUP.read_text(encoding="utf-8")
    assert rendered == golden, (
        "rendered duplicate-axis leaderboard drifted from golden file. Diff:\n"
        f"--- expected ({_GOLDEN_DUP})\n{golden}\n"
        f"+++ actual ({actual_path})\n{rendered}"
    )


def test_missing_spike_run_returns_2(tmp_path: Path) -> None:
    rc = eval_main([str(tmp_path / "does_not_exist.json")])
    assert rc == 2


def test_no_args_prints_usage(capsys: pytest.CaptureFixture[str]) -> None:
    rc = eval_main([])
    captured = capsys.readouterr()
    assert rc == 0
    assert "chamber-eval" in captured.out
