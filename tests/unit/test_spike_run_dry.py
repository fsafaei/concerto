# SPDX-License-Identifier: Apache-2.0
"""``chamber-spike run --dry-run`` tests (T5b.1; plan/07 §5).

Exercises the synthetic-SpikeRun harness path: the dispatcher emits a
deterministic :class:`~chamber.evaluation.results.SpikeRun` JSON that
matches the plan/07 §2 sample-size contract (5 seeds x 20 episodes
per (seed, condition) = 100 per condition). Downstream the file is
consumed by ``chamber-spike next-stage`` (B7) and / or
``chamber-spike eval`` to produce a leaderboard entry without a GPU
ever being touched.

The real-run path (sans ``--dry-run``) is tested by asserting
``ADAPTER_NOT_WIRED_EXIT_CODE`` on every axis until B8 / B9 ship
``chamber.benchmarks.stage1_{as,om}``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from chamber.cli._spike_run import ADAPTER_NOT_WIRED_EXIT_CODE
from chamber.cli.spike import main
from chamber.evaluation.results import SpikeRun

if TYPE_CHECKING:
    from pathlib import Path

_EXPECTED_EPISODES = 200  # 5 seeds x 20 episodes x 2 conditions


# ---------------------------------------------------------------------------
# Dry-run happy path
# ---------------------------------------------------------------------------


class TestDryRunEmitsValidSpikeRun:
    """plan/07 §5: --dry-run emits a SpikeRun JSON the harness can consume."""

    def test_default_dry_run_writes_a_valid_spike_run(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out = tmp_path / "spike_dry.json"
        rc = main(["run", "--axis", "AS", "--dry-run", "--output", str(out)])
        assert rc == 0
        captured = capsys.readouterr()
        assert "PASS (dry-run)" in captured.out
        assert "axis=AS" in captured.out
        assert out.exists()
        run = SpikeRun.model_validate_json(out.read_text(encoding="utf-8"))
        assert run.axis == "AS"
        assert len(run.seeds) == 5
        assert len(run.episode_results) == _EXPECTED_EPISODES

    def test_dry_run_episodes_split_evenly_across_conditions(self, tmp_path: Path) -> None:
        out = tmp_path / "spike_dry.json"
        main(["run", "--axis", "OM", "--dry-run", "--output", str(out)])
        run = SpikeRun.model_validate_json(out.read_text(encoding="utf-8"))
        homo = run.condition_pair.homogeneous_id
        hetero = run.condition_pair.heterogeneous_id
        n_homo = sum(1 for ep in run.episode_results if ep.metadata.get("condition") == homo)
        n_hetero = sum(1 for ep in run.episode_results if ep.metadata.get("condition") == hetero)
        assert n_homo == 100
        assert n_hetero == 100

    def test_homo_succeeds_every_episode(self, tmp_path: Path) -> None:
        out = tmp_path / "spike_dry.json"
        main(["run", "--axis", "AS", "--dry-run", "--output", str(out)])
        run = SpikeRun.model_validate_json(out.read_text(encoding="utf-8"))
        homo = run.condition_pair.homogeneous_id
        homo_eps = [ep for ep in run.episode_results if ep.metadata.get("condition") == homo]
        assert all(ep.success for ep in homo_eps)

    def test_hetero_count_default_is_ten_of_twenty(self, tmp_path: Path) -> None:
        """Default --dry-run-hetero-success-count = 10 yields 10/20 successes per seed."""
        out = tmp_path / "spike_dry.json"
        main(["run", "--axis", "AS", "--dry-run", "--output", str(out)])
        run = SpikeRun.model_validate_json(out.read_text(encoding="utf-8"))
        hetero = run.condition_pair.heterogeneous_id
        hetero_by_seed: dict[int, list[bool]] = {}
        for ep in run.episode_results:
            if ep.metadata.get("condition") == hetero:
                hetero_by_seed.setdefault(ep.seed, []).append(ep.success)
        for seed, successes in hetero_by_seed.items():
            assert sum(successes) == 10, (
                f"seed {seed} hetero success count: expected 10, got {sum(successes)}"
            )

    def test_hetero_count_override_propagates(self, tmp_path: Path) -> None:
        out = tmp_path / "spike_dry.json"
        rc = main(
            [
                "run",
                "--axis",
                "AS",
                "--dry-run",
                "--dry-run-hetero-success-count",
                "18",
                "--output",
                str(out),
            ]
        )
        assert rc == 0
        run = SpikeRun.model_validate_json(out.read_text(encoding="utf-8"))
        hetero = run.condition_pair.heterogeneous_id
        per_seed_success = [
            sum(
                1
                for ep in run.episode_results
                if ep.metadata.get("condition") == hetero and ep.seed == seed and ep.success
            )
            for seed in (0, 1, 2, 3, 4)
        ]
        assert per_seed_success == [18, 18, 18, 18, 18]


# ---------------------------------------------------------------------------
# Dry-run pairs cleanly through chamber.evaluation.bootstrap
# ---------------------------------------------------------------------------


class TestDryRunFeedsBootstrap:
    """The synthetic SpikeRun pairs cleanly through pacluster_bootstrap."""

    def test_default_dry_run_passes_the_twenty_pp_gate(self, tmp_path: Path) -> None:
        """Default 10/20 hetero success rate → IQM gap ≈ 50pp; gate-PP=20 passes."""
        from chamber.evaluation.bootstrap import build_paired_episodes, pacluster_bootstrap
        from concerto.training.seeding import derive_substream

        out = tmp_path / "spike_dry.json"
        main(["run", "--axis", "AS", "--dry-run", "--output", str(out)])
        run = SpikeRun.model_validate_json(out.read_text(encoding="utf-8"))
        pairs = build_paired_episodes(run)
        assert pairs, "dry-run SpikeRun produces empty paired episodes"
        rng = derive_substream("test", root_seed=0).default_rng()
        ci = pacluster_bootstrap(pairs, n_resamples=500, rng=rng)
        assert ci.ci_low * 100.0 >= 20.0, (
            f"default dry-run should pass the 20pp gate; got ci_low_pp={ci.ci_low * 100.0:.2f}"
        )

    def test_high_hetero_count_falls_below_the_gate(self, tmp_path: Path) -> None:
        """--dry-run-hetero-success-count=18 → IQM gap = 0; gate-PP=20 fails."""
        from chamber.evaluation.bootstrap import build_paired_episodes, pacluster_bootstrap
        from concerto.training.seeding import derive_substream

        out = tmp_path / "spike_dry.json"
        main(
            [
                "run",
                "--axis",
                "AS",
                "--dry-run",
                "--dry-run-hetero-success-count",
                "18",
                "--output",
                str(out),
            ]
        )
        run = SpikeRun.model_validate_json(out.read_text(encoding="utf-8"))
        pairs = build_paired_episodes(run)
        rng = derive_substream("test", root_seed=0).default_rng()
        ci = pacluster_bootstrap(pairs, n_resamples=500, rng=rng)
        assert ci.ci_low * 100.0 < 20.0, (
            f"high hetero-success dry-run should fail the 20pp gate; "
            f"got ci_low_pp={ci.ci_low * 100.0:.2f}"
        )


# ---------------------------------------------------------------------------
# Real-run path: adapter not yet wired (B8 / B9 / Stage-2/3 future)
# ---------------------------------------------------------------------------


class TestRealRunWithoutAdapter:
    """plan/07 §T5b.2: axes without a registered adapter exit with code 6.

    The AS adapter ships in B8 (``chamber.benchmarks.stage1_as``); the
    OM adapter is B9. Stage-2/3 axes (CR/CM/PF/SA) stay deferred per
    the prompt §2 explicit out-of-scope list, so they continue to exit
    with :data:`ADAPTER_NOT_WIRED_EXIT_CODE` until plan/07 §T5b.3 /
    §T5b.4 work lands. Each newly-shipped adapter flips one entry of
    this parametrisation.
    """

    @pytest.mark.parametrize("axis", ["OM", "CR", "CM", "PF", "SA"])
    def test_real_run_exits_adapter_not_wired(
        self, axis: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out = tmp_path / "spike.json"
        rc = main(["run", "--axis", axis, "--output", str(out)])
        assert rc == ADAPTER_NOT_WIRED_EXIT_CODE
        captured = capsys.readouterr()
        assert "FAIL" in captured.err
        assert axis in captured.err
        assert not out.exists()


# ---------------------------------------------------------------------------
# Argparse surface
# ---------------------------------------------------------------------------


class TestArgparseSurface:
    """plan/07 §T5b.1: the `run` subparser is well-formed."""

    def test_help_exits_zero(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as excinfo:
            main(["run", "--help"])
        assert excinfo.value.code == 0
        captured = capsys.readouterr()
        for flag in ("--axis", "--output", "--dry-run", "--dry-run-hetero-success-count"):
            assert flag in captured.out

    def test_bad_axis_choice_errors(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit) as excinfo:
            main(["run", "--axis", "XX", "--output", str(tmp_path / "x.json")])
        assert excinfo.value.code == 2  # argparse "bad usage"

    def test_invalid_hetero_count_raises(self, tmp_path: Path) -> None:
        """--dry-run-hetero-success-count out of [0, 20] raises ValueError."""
        out = tmp_path / "spike_dry.json"
        with pytest.raises(ValueError, match="hetero-success-count"):
            main(
                [
                    "run",
                    "--axis",
                    "AS",
                    "--dry-run",
                    "--dry-run-hetero-success-count",
                    "21",
                    "--output",
                    str(out),
                ]
            )
