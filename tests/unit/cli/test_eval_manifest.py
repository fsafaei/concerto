# SPDX-License-Identifier: Apache-2.0
"""Tests for the ``chamber-eval manifest`` subcommand (ADR-027 §Versioning).

The manifest subcommand rides alongside the ADR-008 evaluation
pipeline; these tests pin (a) the emitted JSON shape and determinism,
(b) the ``--output`` file mode, and (c) that the historical flat
``chamber-eval`` invocation is untouched by the dispatch.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from chamber.cli import eval as eval_cli

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


class TestManifestSubcommand:
    def test_emits_the_pinned_suite(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert eval_cli.main(["manifest"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["suite"] == "CHAMBER-Bench"
        assert payload["suite_version"] == "1.0"
        assert [t["task_id"] for t in payload["tasks"]] == [
            "mpe_cooperative_push",
            "stage0_smoke",
            "stage1_pickplace_as",
            "stage1_pickplace_om",
            "cocarry",
            "handover_place",
            "amr_handover_dynamic",
            "co_hold_secure",
            "coinsert",
        ]

    def test_two_runs_are_byte_identical(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert eval_cli.main(["manifest"]) == 0
        first = capsys.readouterr().out
        assert eval_cli.main(["manifest"]) == 0
        second = capsys.readouterr().out
        assert first == second

    def test_output_flag_writes_the_file(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out = tmp_path / "manifest.json"
        assert eval_cli.main(["manifest", "--output", str(out)]) == 0
        assert capsys.readouterr().out == ""
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert len(payload["tasks"]) == 9

    def test_legacy_bare_invocation_still_prints_the_banner(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert eval_cli.main([]) == 0
        out = capsys.readouterr().out
        assert "Usage: chamber-eval SPIKE_RUN.json" in out
        assert "chamber-eval manifest" in out
