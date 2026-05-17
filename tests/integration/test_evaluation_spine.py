# SPDX-License-Identifier: Apache-2.0
"""Integration test: evaluation spine end-to-end (ADR-007, ADR-008, ADR-014).

Hand-rolls a 3-seed x 5-episodes SpikeRun, runs cluster + paired
bootstrap, builds the HRS vector + scalar, renders the leaderboard,
and compares against a golden Markdown file. Re-runs with rliable
installed (if present) to verify the performance-profile column
appears.

The seeds in this test are intentionally tiny so the cluster /
paired bootstrap is exercised but the run finishes in milliseconds.
The golden file is generated deterministically — regenerate it via
``--regenerate-golden`` (see test body) when the schema legitimately
changes.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from chamber.evaluation import (
    BootstrapCI,
    ConditionPair,
    ConditionResult,
    EpisodeResult,
    HRSVector,
    LeaderboardEntry,
    PairedEpisode,
    SpikeRun,
    aggregate_metrics,
    cluster_bootstrap,
    compute_hrs_scalar,
    compute_hrs_vector,
    pacluster_bootstrap,
    render_leaderboard,
)

_GOLDEN_DIR = Path(__file__).parent / "golden"
_GOLDEN_LEADERBOARD = _GOLDEN_DIR / "leaderboard_min.md"


def _build_spike_run() -> SpikeRun:
    seeds = [11, 22, 33]
    episodes_per_seed = 5
    homo_pattern = [True, True, True, True, False]
    hetero_pattern = [False, True, False, True, True]
    homo_id = "homo_panda_panda"
    hetero_id = "hetero_panda_fetch"
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
        spike_id="stage2_cm_golden",
        prereg_sha="0" * 40,
        git_tag="prereg/stage2-cm-golden-v0",
        axis="CM",
        # ADR-016 §Decision: SpikeRun.sub_stage is required. CM lives
        # at Stage 2 per ADR-007 §Implementation staging.
        sub_stage="2",
        condition_pair=ConditionPair(
            homogeneous_id=homo_id,
            heterogeneous_id=hetero_id,
        ),
        seeds=seeds,
        episode_results=results,
    )


def _paired_episodes(run: SpikeRun) -> list[PairedEpisode]:
    homo = run.condition_pair.homogeneous_id
    hetero = run.condition_pair.heterogeneous_id
    homo_map: dict[tuple[int, int, int], float] = {}
    hetero_map: dict[tuple[int, int, int], float] = {}
    for ep in run.episode_results:
        key = (ep.seed, int(ep.episode_idx), ep.initial_state_seed)
        score = 1.0 if ep.success else 0.0
        if ep.metadata.get("condition") == homo:
            homo_map[key] = score
        elif ep.metadata.get("condition") == hetero:
            hetero_map[key] = score
    return [
        PairedEpisode(
            seed=k[0],
            episode_idx=k[1],
            initial_state_seed=k[2],
            homogeneous=v,
            heterogeneous=hetero_map[k],
        )
        for k, v in homo_map.items()
        if k in hetero_map
    ]


def _run_pipeline(run: SpikeRun) -> tuple[BootstrapCI, ConditionResult, HRSVector, float]:
    rng = np.random.default_rng(7)
    pair_ci = pacluster_bootstrap(_paired_episodes(run), n_resamples=500, rng=rng)
    homo_by_seed: dict[int, list[float]] = {}
    hetero_by_seed: dict[int, list[float]] = {}
    for ep in run.episode_results:
        score = 1.0 if ep.success else 0.0
        if ep.metadata.get("condition") == run.condition_pair.homogeneous_id:
            homo_by_seed.setdefault(ep.seed, []).append(score)
        elif ep.metadata.get("condition") == run.condition_pair.heterogeneous_id:
            hetero_by_seed.setdefault(ep.seed, []).append(score)
    homo_ci = cluster_bootstrap(homo_by_seed, n_resamples=500, rng=rng)
    hetero_ci = cluster_bootstrap(hetero_by_seed, n_resamples=500, rng=rng)
    condition = ConditionResult(
        axis=run.axis,
        n_episodes=len(run.episode_results),
        homogeneous_success=max(0.0, homo_ci.iqm),
        heterogeneous_success=max(0.0, hetero_ci.iqm),
        gap_pp=pair_ci.iqm * 100.0,
        ci_low_pp=pair_ci.ci_low * 100.0,
        ci_high_pp=pair_ci.ci_high * 100.0,
        violation_rate=0.0,
        fallback_rate=0.0,
    )
    vector = compute_hrs_vector({run.axis: condition})
    scalar = max(0.0, compute_hrs_scalar(vector))
    return pair_ci, condition, vector, scalar


def test_pipeline_matches_golden_leaderboard(tmp_path: Path) -> None:
    run = _build_spike_run()
    _, condition, vector, scalar = _run_pipeline(run)
    entry = LeaderboardEntry(
        method_id="concerto-golden",
        spike_runs=[run.spike_id],
        hrs_vector=vector,
        hrs_scalar=scalar,
        violation_rate=condition.violation_rate,
        fallback_rate=condition.fallback_rate,
    )
    rendered = render_leaderboard([entry])

    actual_path = tmp_path / "leaderboard_min.md"
    actual_path.write_text(rendered, encoding="utf-8")

    assert _GOLDEN_LEADERBOARD.exists(), (
        f"missing golden file {_GOLDEN_LEADERBOARD}; "
        "regenerate by copying the test's tmp_path output"
    )
    golden = _GOLDEN_LEADERBOARD.read_text(encoding="utf-8")
    assert rendered == golden, (
        "rendered leaderboard drifted from golden file. Diff:\n"
        f"--- expected ({_GOLDEN_LEADERBOARD})\n{golden}\n"
        f"+++ actual ({actual_path})\n{rendered}"
    )


def test_pipeline_reproducible_under_same_seed() -> None:
    run = _build_spike_run()
    a = _run_pipeline(run)
    b = _run_pipeline(run)
    assert a[0].iqm == pytest.approx(b[0].iqm)
    assert a[0].ci_low == pytest.approx(b[0].ci_low)
    assert a[0].ci_high == pytest.approx(b[0].ci_high)
    assert a[2].entries[0].score == pytest.approx(b[2].entries[0].score)
    assert a[3] == pytest.approx(b[3])


def test_round_trip_spike_run_json(tmp_path: Path) -> None:
    run = _build_spike_run()
    path = tmp_path / "spike_run.json"
    path.write_text(run.model_dump_json(indent=2), encoding="utf-8")
    parsed = SpikeRun.model_validate_json(path.read_text(encoding="utf-8"))
    assert parsed == run


def test_aggregate_metrics_native_path_when_rliable_absent() -> None:
    values: dict[int, list[float]] = {1: [1.0, 1.0, 0.0, 1.0], 2: [0.5, 0.5, 1.0, 0.0]}
    if importlib.util.find_spec("rliable") is None:
        with pytest.warns(RuntimeWarning, match="rliable"):
            metrics = aggregate_metrics(values)
        assert metrics["performance_profile"] is None
    else:
        metrics = aggregate_metrics(values)
        assert metrics["performance_profile"] is not None
    assert metrics["iqm"] is not None
    assert metrics["optimality_gap"] is not None


def test_leaderboard_refuses_empty_hrs_vector() -> None:
    entry = LeaderboardEntry(
        method_id="empty",
        spike_runs=[],
        hrs_vector=HRSVector(entries=[]),
        hrs_scalar=0.0,
        violation_rate=0.0,
        fallback_rate=0.0,
    )
    with pytest.raises(ValueError, match="empty HRS vector"):
        render_leaderboard([entry])


def test_paired_bootstrap_recovers_known_gap() -> None:
    pairs = [
        PairedEpisode(
            seed=s,
            episode_idx=i,
            initial_state_seed=s * 100 + i,
            homogeneous=1.0,
            heterogeneous=0.5,
        )
        for s in range(4)
        for i in range(20)
    ]
    rng = np.random.default_rng(1)
    ci = pacluster_bootstrap(pairs, n_resamples=400, rng=rng)
    assert ci.iqm == pytest.approx(0.5, abs=1e-9)
    assert ci.ci_low == pytest.approx(0.5, abs=1e-9)
    assert ci.ci_high == pytest.approx(0.5, abs=1e-9)


def _regenerate_golden_if_requested(tmp: str) -> None:
    """Internal hook for the maintainer to overwrite the golden file."""
    Path(_GOLDEN_LEADERBOARD).write_text(tmp, encoding="utf-8")


def _sample_three_table_dict() -> dict[str, list[dict[str, object]]]:
    return {
        "table_1": [
            {
                "assumption": "A1",
                "description": "CBF-feasibility",
                "violations": 0,
                "n_steps": 1000,
            },
            {
                "assumption": "A2",
                "description": "predictor bounded",
                "violations": 1,
                "n_steps": 1000,
            },
        ],
        "table_2": [
            {
                "predictor": "gt",
                "conformal_mode": "noLearn",
                "vendor_compliance": None,
                "n_episodes": 100,
                "violations": 0,
                "fallback_fires": 2,
            },
            {
                "predictor": "pred",
                "conformal_mode": "Learn",
                "vendor_compliance": None,
                "n_episodes": 100,
                "violations": 3,
                "fallback_fires": 5,
            },
        ],
        "table_3": [
            {
                "condition": "gt/noLearn",
                "lambda_mean": 0.0,
                "lambda_var": 0.0,
                "oracle_lambda_mean": 0.0,
            },
            {
                "condition": "pred/Learn",
                "lambda_mean": 0.12,
                "lambda_var": 0.004,
                "oracle_lambda_mean": 0.0,
            },
        ],
    }


def test_three_table_markdown_distinguishes_violation_from_fallback() -> None:
    from chamber.evaluation import render_three_table_safety_report

    md = render_three_table_safety_report(_sample_three_table_dict(), fmt="markdown")
    assert "Constraint violations" in md
    assert "Fallback fires" in md
    assert "ADR-014" in md


def test_three_table_latex_renders() -> None:
    from chamber.evaluation import render_three_table_safety_report

    tex = render_three_table_safety_report(_sample_three_table_dict(), fmt="latex")
    assert "\\begin{table}" in tex
    assert "Constraint violations" in tex


def test_three_table_unknown_format_raises() -> None:
    from chamber.evaluation import render_three_table_safety_report

    with pytest.raises(ValueError, match="fmt must be"):
        render_three_table_safety_report(_sample_three_table_dict(), fmt="csv")


def test_three_table_missing_key_raises() -> None:
    from chamber.evaluation import render_three_table_safety_report

    with pytest.raises(ValueError, match="missing required key"):
        render_three_table_safety_report({"table_1": []})


def _write_prereg_yaml(path: Path, *, git_tag: str) -> None:
    payload = (
        "axis: CM\n"
        "condition_pair:\n"
        "  homogeneous_id: homo_panda_panda\n"
        "  heterogeneous_id: hetero_panda_fetch\n"
        "seeds: [11, 22, 33]\n"
        "episodes_per_seed: 5\n"
        "estimator: iqm_success_rate\n"
        "bootstrap_method: cluster\n"
        "failure_policy: strict\n"
        f"git_tag: {git_tag}\n"
        "notes: integration-test prereg\n"
    )
    path.write_text(payload, encoding="utf-8")


def test_load_prereg_validates_schema(tmp_path: Path) -> None:
    from chamber.evaluation import load_prereg

    path = tmp_path / "prereg.yaml"
    _write_prereg_yaml(path, git_tag="prereg/stage2-cm-golden-v0")
    spec = load_prereg(path)
    assert spec.axis == "CM"
    assert spec.bootstrap_method == "cluster"
    assert spec.condition_pair.homogeneous_id == "homo_panda_panda"


def test_load_prereg_rejects_unknown_axis(tmp_path: Path) -> None:
    from pydantic import ValidationError

    from chamber.evaluation import load_prereg

    path = tmp_path / "prereg.yaml"
    _write_prereg_yaml(path, git_tag="prereg/bad")
    # Corrupt the axis to one off the ADR-007 shortlist.
    text = path.read_text(encoding="utf-8").replace("axis: CM", "axis: XX")
    path.write_text(text, encoding="utf-8")
    # XX is a *string* — the schema accepts it but normalised_axis() rejects it.
    with pytest.raises((ValueError, ValidationError), match="axis"):
        load_prereg(path)


def test_load_prereg_rejects_extra_keys(tmp_path: Path) -> None:
    from pydantic import ValidationError

    from chamber.evaluation import load_prereg

    path = tmp_path / "prereg.yaml"
    _write_prereg_yaml(path, git_tag="prereg/bad")
    text = path.read_text(encoding="utf-8") + "unexpected: 42\n"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValidationError):
        load_prereg(path)


def test_verify_git_tag_detects_mismatch(tmp_path: Path) -> None:
    import subprocess

    from chamber.evaluation import (
        PreregistrationError,
        load_prereg,
        verify_git_tag,
    )

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)  # noqa: S603,S607
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "t@t"], check=True)  # noqa: S603,S607
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "t"], check=True)  # noqa: S603,S607
    subprocess.run(["git", "-C", str(repo), "config", "commit.gpgsign", "false"], check=True)  # noqa: S603,S607

    prereg = repo / "prereg.yaml"
    _write_prereg_yaml(prereg, git_tag="prereg/v0")
    subprocess.run(["git", "-C", str(repo), "add", "prereg.yaml"], check=True)  # noqa: S603,S607
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "lock prereg"], check=True)  # noqa: S603,S607
    subprocess.run(["git", "-C", str(repo), "tag", "prereg/v0"], check=True)  # noqa: S603,S607

    spec = load_prereg(prereg)
    sha = verify_git_tag(spec, prereg, repo_path=repo)
    assert isinstance(sha, str)
    assert len(sha) == 40

    # Mutate the file: the on-disk blob SHA shifts but the tagged blob stays put.
    prereg.write_text(prereg.read_text(encoding="utf-8") + "# tampered\n", encoding="utf-8")
    with pytest.raises(PreregistrationError, match="blob SHA mismatch"):
        verify_git_tag(spec, prereg, repo_path=repo)


def test_verify_git_tag_missing_tag(tmp_path: Path) -> None:
    import subprocess

    from chamber.evaluation import (
        PreregistrationError,
        load_prereg,
        verify_git_tag,
    )

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)  # noqa: S603,S607
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "t@t"], check=True)  # noqa: S603,S607
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "t"], check=True)  # noqa: S603,S607
    subprocess.run(["git", "-C", str(repo), "config", "commit.gpgsign", "false"], check=True)  # noqa: S603,S607
    prereg = repo / "prereg.yaml"
    _write_prereg_yaml(prereg, git_tag="prereg/missing")
    subprocess.run(["git", "-C", str(repo), "add", "prereg.yaml"], check=True)  # noqa: S603,S607
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "lock"], check=True)  # noqa: S603,S607
    spec = load_prereg(prereg)
    with pytest.raises(PreregistrationError, match="tag"):
        verify_git_tag(spec, prereg, repo_path=repo)


def test_cluster_bootstrap_invalid_n_resamples() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="n_resamples"):
        cluster_bootstrap({1: [1.0]}, n_resamples=0, rng=rng)


def test_pacluster_bootstrap_invalid_n_resamples() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="n_resamples"):
        pacluster_bootstrap([], n_resamples=0, rng=rng)


def test_pacluster_bootstrap_empty_input() -> None:
    rng = np.random.default_rng(0)
    ci = pacluster_bootstrap([], n_resamples=10, rng=rng)
    assert np.isnan(ci.iqm)
    assert np.isnan(ci.ci_low)
    assert np.isnan(ci.ci_high)


def test_cluster_bootstrap_empty_input() -> None:
    rng = np.random.default_rng(0)
    ci = cluster_bootstrap({}, n_resamples=10, rng=rng)
    assert np.isnan(ci.iqm)
    assert np.isnan(ci.ci_low)


def test_hrs_vector_filters_unmeasured_axes() -> None:
    cond = ConditionResult(
        axis="CM",
        n_episodes=10,
        homogeneous_success=0.8,
        heterogeneous_success=0.6,
        gap_pp=20.0,
        ci_low_pp=10.0,
        ci_high_pp=30.0,
        violation_rate=0.0,
        fallback_rate=0.0,
    )
    vector = compute_hrs_vector({"CM": cond})
    assert [e.axis for e in vector.entries] == ["CM"]


def test_hrs_scalar_empty_vector_is_zero() -> None:
    assert compute_hrs_scalar(HRSVector(entries=[])) == 0.0


def test_hrs_scalar_overridden_weights() -> None:
    cond = ConditionResult(
        axis="CM",
        n_episodes=10,
        homogeneous_success=0.8,
        heterogeneous_success=0.6,
        gap_pp=20.0,
        ci_low_pp=10.0,
        ci_high_pp=30.0,
        violation_rate=0.0,
        fallback_rate=0.0,
    )
    vector = compute_hrs_vector({"CM": cond})
    assert compute_hrs_scalar(vector, weights={"CM": 2.0}) == pytest.approx(0.6)


def test_cli_render_tables_smoke(tmp_path: Path) -> None:
    from chamber.cli.render_tables import main as render_main

    report_path = tmp_path / "three_tables.json"
    report_path.write_text(json.dumps(_sample_three_table_dict()), encoding="utf-8")
    out_path = tmp_path / "out.md"
    rc = render_main(
        ["--safety-report", str(report_path), "--output", str(out_path)],
    )
    assert rc == 0
    assert "Constraint violations" in out_path.read_text(encoding="utf-8")


def test_cli_render_tables_no_args(capsys: pytest.CaptureFixture[str]) -> None:
    from chamber.cli.render_tables import main as render_main

    rc = render_main([])
    captured = capsys.readouterr()
    assert rc == 0
    assert "chamber-render-tables" in captured.out


def test_cli_render_tables_missing_file(tmp_path: Path) -> None:
    from chamber.cli.render_tables import main as render_main

    rc = render_main(["--safety-report", str(tmp_path / "missing.json")])
    assert rc == 2


def test_cli_eval_pipeline(tmp_path: Path) -> None:
    from chamber.cli.eval import main as eval_main

    run = _build_spike_run()
    run_path = tmp_path / "spike_run.json"
    run_path.write_text(run.model_dump_json(indent=2), encoding="utf-8")
    out_path = tmp_path / "entry.json"
    rc = eval_main(
        [
            str(run_path),
            "--method-id",
            "concerto-cli",
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
    assert entry.method_id == "concerto-cli"
    assert entry.hrs_vector.entries
