# SPDX-License-Identifier: Apache-2.0
"""``chamber-spike next-stage`` gate tests (T5b.1; plan/07 §5; ADR-007 §Implementation staging).

Each test stages a synthetic SpikeRun (via the B7 dry-run path) and
then invokes ``chamber-spike next-stage`` against it. The dry-run's
deterministic episode layout makes the per-axis gap CI lower bound
predictable: with the default 10/20 hetero success rate, IQM gap is
~50pp and the gate passes; with 18/20 hetero success the gap is 0
and the gate fails.

Stage groupings (ADR-007 §Implementation staging):

- Stage 1: AS + OM
- Stage 2: CR + CM
- Stage 3: PF + SA
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from chamber.cli._spike_next_stage import NEXT_STAGE_GATE_EXIT_CODE
from chamber.cli.spike import main

if TYPE_CHECKING:
    from pathlib import Path


def _emit_dry_run(
    *,
    tmp_path: Path,
    axis: str,
    hetero_success_count: int = 10,
    filename: str | None = None,
) -> Path:
    """Stage a dry-run SpikeRun JSON via ``chamber-spike run`` and return its path."""
    out = tmp_path / (filename or f"spike_{axis.lower()}.json")
    rc = main(
        [
            "run",
            "--axis",
            axis,
            "--dry-run",
            "--dry-run-hetero-success-count",
            str(hetero_success_count),
            "--output",
            str(out),
        ]
    )
    assert rc == 0, f"failed to stage dry-run for axis {axis}"
    return out


# ---------------------------------------------------------------------------
# Happy path: at least one prior-stage axis passes
# ---------------------------------------------------------------------------


class TestGatePasses:
    """plan/07 §5: gate passes when >=1 prior-stage axis has ci_low_pp >= 20."""

    def test_stage1_single_axis_passes(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        spike = _emit_dry_run(tmp_path=tmp_path, axis="AS", hetero_success_count=10)
        rc = main(
            [
                "next-stage",
                "--prior-stage",
                "1",
                "--spike-runs",
                str(spike),
            ]
        )
        assert rc == 0
        captured = capsys.readouterr()
        assert "PASS" in captured.out
        assert "axis=AS" in captured.out

    def test_stage1_both_axes_passing(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        as_spike = _emit_dry_run(tmp_path=tmp_path, axis="AS")
        om_spike = _emit_dry_run(tmp_path=tmp_path, axis="OM")
        rc = main(
            [
                "next-stage",
                "--prior-stage",
                "1",
                "--spike-runs",
                str(as_spike),
                str(om_spike),
            ]
        )
        assert rc == 0
        captured = capsys.readouterr()
        assert "2/2 axis(es) pass" in captured.out

    def test_one_pass_and_one_fail_still_clears_gate(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """ADR-007 §Implementation staging: at least one axis clearing >=20pp clears the stage."""
        as_pass = _emit_dry_run(tmp_path=tmp_path, axis="AS", hetero_success_count=10)
        om_fail = _emit_dry_run(tmp_path=tmp_path, axis="OM", hetero_success_count=18)
        rc = main(
            [
                "next-stage",
                "--prior-stage",
                "1",
                "--spike-runs",
                str(as_pass),
                str(om_fail),
            ]
        )
        assert rc == 0
        captured = capsys.readouterr()
        assert "PASS" in captured.out
        assert "1/2 axis(es) pass" in captured.out


# ---------------------------------------------------------------------------
# Failure path: no prior-stage axis passes
# ---------------------------------------------------------------------------


class TestGateFails:
    """plan/07 §5: gate fails when no prior-stage axis meets the threshold."""

    def test_stage1_both_axes_failing(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        as_fail = _emit_dry_run(tmp_path=tmp_path, axis="AS", hetero_success_count=18)
        om_fail = _emit_dry_run(tmp_path=tmp_path, axis="OM", hetero_success_count=18)
        rc = main(
            [
                "next-stage",
                "--prior-stage",
                "1",
                "--spike-runs",
                str(as_fail),
                str(om_fail),
            ]
        )
        assert rc == NEXT_STAGE_GATE_EXIT_CODE
        captured = capsys.readouterr()
        assert "FAIL" in captured.err
        assert "0/2 axis(es) pass" in captured.err

    def test_single_failing_axis(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        spike = _emit_dry_run(tmp_path=tmp_path, axis="AS", hetero_success_count=20)
        rc = main(
            [
                "next-stage",
                "--prior-stage",
                "1",
                "--spike-runs",
                str(spike),
            ]
        )
        assert rc == NEXT_STAGE_GATE_EXIT_CODE


# ---------------------------------------------------------------------------
# Wrong-stage / missing inputs
# ---------------------------------------------------------------------------


class TestInputValidation:
    """Failure modes for malformed or wrong-stage inputs."""

    def test_no_prior_stage_axis_in_runs(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A Stage-2 SpikeRun handed to --prior-stage 1 trips the gate."""
        cr_spike = _emit_dry_run(tmp_path=tmp_path, axis="CR", hetero_success_count=10)
        rc = main(
            [
                "next-stage",
                "--prior-stage",
                "1",
                "--spike-runs",
                str(cr_spike),
            ]
        )
        assert rc == NEXT_STAGE_GATE_EXIT_CODE
        captured = capsys.readouterr()
        assert "no SpikeRun on the prior stage" in captured.err

    def test_missing_spike_run_file(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        missing = tmp_path / "does_not_exist.json"
        rc = main(
            [
                "next-stage",
                "--prior-stage",
                "1",
                "--spike-runs",
                str(missing),
            ]
        )
        assert rc == NEXT_STAGE_GATE_EXIT_CODE
        captured = capsys.readouterr()
        assert "not found" in captured.err

    def test_bad_prior_stage_choice_errors(self) -> None:
        with pytest.raises(SystemExit) as excinfo:
            main(["next-stage", "--prior-stage", "4", "--spike-runs", "x.json"])
        assert excinfo.value.code == 2  # argparse "bad usage"


# ---------------------------------------------------------------------------
# Determinism (ADR-002 P6) + tunable threshold
# ---------------------------------------------------------------------------


class TestDeterminismAndThreshold:
    """Bootstrap RNG is seeded; --gate-pp is tunable."""

    def test_two_invocations_same_seed_same_verdict(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        spike = _emit_dry_run(tmp_path=tmp_path, axis="AS")
        # Drop the dry-run staging output so the captures below contain
        # only the next-stage CLI lines.
        capsys.readouterr()
        first = main(
            [
                "next-stage",
                "--prior-stage",
                "1",
                "--spike-runs",
                str(spike),
                "--seed",
                "7",
            ]
        )
        out_first = capsys.readouterr().out
        second = main(
            [
                "next-stage",
                "--prior-stage",
                "1",
                "--spike-runs",
                str(spike),
                "--seed",
                "7",
            ]
        )
        out_second = capsys.readouterr().out
        assert first == second == 0
        # ci_low_pp column should appear and be byte-identical across runs.
        assert "ci_low_pp" in out_first
        assert out_first == out_second

    def test_tighter_gate_can_fail_a_pass(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A high --gate-pp threshold can override a default-pass dry-run."""
        spike = _emit_dry_run(tmp_path=tmp_path, axis="AS", hetero_success_count=10)
        # Default (gate-pp=20) → passes; gate-pp=80 should fail (gap is ~50pp).
        rc = main(
            [
                "next-stage",
                "--prior-stage",
                "1",
                "--spike-runs",
                str(spike),
                "--gate-pp",
                "80.0",
            ]
        )
        assert rc == NEXT_STAGE_GATE_EXIT_CODE
