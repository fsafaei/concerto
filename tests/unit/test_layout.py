# SPDX-License-Identifier: Apache-2.0
"""Verify that every package under src/ has an __init__.py with a docstring."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_SRC = Path(__file__).parents[2] / "src"
_PACKAGES = ["concerto", "chamber"]


def _collect_packages() -> list[Path]:
    return [init for top in _PACKAGES for init in (_SRC / top).rglob("__init__.py")]


@pytest.mark.parametrize("init", _collect_packages(), ids=lambda p: str(p.relative_to(_SRC)))
def test_init_has_docstring(init: Path) -> None:
    source = init.read_text(encoding="utf-8")
    tree = ast.parse(source)
    docstring = ast.get_docstring(tree)
    assert docstring, f"{init.relative_to(_SRC)} is missing a module docstring"
