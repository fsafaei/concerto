# SPDX-License-Identifier: Apache-2.0
"""Smoke-import and call-once tests for Phase-0 CLI stubs."""

from __future__ import annotations

import importlib


def _call_main(module_path: str) -> None:
    mod = importlib.import_module(module_path)
    mod.main()


def test_concerto_cli_main() -> None:
    _call_main("concerto.cli")


def test_chamber_spike_main() -> None:
    _call_main("chamber.cli.spike")


def test_chamber_eval_main() -> None:
    _call_main("chamber.cli.eval")


def test_chamber_render_tables_main() -> None:
    _call_main("chamber.cli.render_tables")
