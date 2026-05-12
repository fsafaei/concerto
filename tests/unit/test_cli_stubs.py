# SPDX-License-Identifier: Apache-2.0
"""Smoke + sub-command tests for the Phase-0 chamber/concerto CLI entry points.

Covers:

- All four entry points (``concerto``, ``chamber-spike``,
  ``chamber-eval``, ``chamber-render-tables``) load and accept the
  no-argument banner-print path. ``chamber-spike`` gained sub-commands
  in M4b-9b, so its banner path is the bare ``chamber-spike`` call
  with empty argv.
- ``chamber-spike train`` argparse surface: ``--help``, bad path,
  valid path + 1k-frame override (asserts exit-0 + summary line on
  stdout).
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest


def _call_no_arg_main(module_path: str) -> None:
    """Invoke a stub ``main()`` with no arguments — exercises the banner path."""
    mod = importlib.import_module(module_path)
    # The Phase-0 stubs and the M4b-9b chamber-spike both accept either
    # no-argument main() (legacy) or main(argv) (M4b-9b). Try the
    # explicit-empty-argv form first; if the stub's main() doesn't
    # accept that, fall back to the no-argument form.
    try:
        result = mod.main([])
    except TypeError:
        result = mod.main()
    # M4b-9b chamber-spike returns an int exit code; legacy stubs
    # return None. Both 0 and None are acceptable success values.
    assert result in (None, 0)


def test_concerto_cli_main() -> None:
    _call_no_arg_main("concerto.cli")


def test_chamber_spike_main_banner() -> None:
    """M4b-9b: bare ``chamber-spike`` (no sub-command) still prints the banner + help."""
    _call_no_arg_main("chamber.cli.spike")


def test_chamber_eval_main() -> None:
    _call_no_arg_main("chamber.cli.eval")


def test_chamber_render_tables_main() -> None:
    _call_no_arg_main("chamber.cli.render_tables")


# ---------------------------------------------------------------------------
# chamber-spike train sub-command (M4b-9b; ADR-002 §Decisions; plan/05 §3.5)
# ---------------------------------------------------------------------------


class TestChamberSpikeTrain:
    """argparse surface for ``chamber-spike train`` (M4b-9b)."""

    def test_help_exits_zero(self, capsys: pytest.CaptureFixture[str]) -> None:
        """``chamber-spike train --help`` exits 0 and prints usage to stdout."""
        from chamber.cli.spike import main

        with pytest.raises(SystemExit) as excinfo:
            main(["train", "--help"])
        # argparse exits 0 on --help (vs. 2 on bad usage).
        assert excinfo.value.code == 0
        captured = capsys.readouterr()
        # The help text mentions both required and key optional flags.
        assert "--config" in captured.out
        assert "--override" in captured.out
        assert "--check-guarantee" in captured.out

    def test_bad_config_path_errors_loudly(self, tmp_path: Path) -> None:
        """A non-existent --config path must fail at config-load time, not silently."""
        from chamber.cli.spike import main

        bogus = tmp_path / "does_not_exist.yaml"
        # Hydra's compose path raises a (subclass of) Exception when the
        # named config can't be located. The CLI does NOT swallow this —
        # the user must see the failure.
        with pytest.raises(Exception):  # noqa: B017,PT011  # any subclass; Hydra's exact type varies.
            main(["train", "--config", str(bogus)])

    def test_valid_config_returns_zero_and_prints_summary(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """1k-frame override on the production YAML: exit 0 + summary line on stdout."""
        from chamber.cli.spike import main

        # Resolve the production YAML relative to the repo root via the
        # test file's location (mirrors the pattern in
        # test_empirical_guarantee.py).
        repo_root = Path(__file__).resolve().parents[2]
        config_path = (
            repo_root / "configs" / "training" / "ego_aht_happo" / "mpe_cooperative_push.yaml"
        )
        artifacts = tmp_path / "artifacts"
        logs = tmp_path / "logs"
        exit_code = main(
            [
                "train",
                "--config",
                str(config_path),
                "--override",
                "total_frames=1000",
                "--override",
                "checkpoint_every=500",
                "--override",
                "happo.rollout_length=250",
                "--override",
                f"artifacts_root={artifacts}",
                "--override",
                f"log_dir={logs}",
            ]
        )
        assert exit_code == 0
        captured = capsys.readouterr()
        # Summary line: pin the prefix + each field key the script
        # produces. Avoids brittle exact-string matching but still
        # catches a regression that drops a field.
        assert "run_id=" in captured.out
        assert "n_episodes=" in captured.out
        assert "n_checkpoints=" in captured.out
        assert "mean_reward_last_" in captured.out


def test_chamber_spike_main_no_argv_uses_sys_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    """M4b-9b: ``main()`` with no argv falls back to sys.argv per argparse convention.

    Smoke check that the entry-point wiring works when the console
    script is invoked without arguments (the user typing
    ``chamber-spike`` at a terminal): sys.argv is patched to be the
    bare program name, argparse parses no command, the function
    returns 0.
    """
    from chamber.cli.spike import main

    monkeypatch.setattr(sys, "argv", ["chamber-spike"])
    assert main() == 0
