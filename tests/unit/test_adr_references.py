# SPDX-License-Identifier: Apache-2.0
"""Enforce P1: every public symbol in ADR-bearing modules has an ADR reference."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

_SRC = Path(__file__).parents[2] / "src"

# Modules whose public symbols must carry ADR references.
# Keyed by package-relative path prefix (PurePosixPath string).
_ADR_BEARING_PREFIXES = (
    "concerto/safety",
    "concerto/training",
    "concerto/api",
    "concerto/policies",
    "chamber/envs",
    "chamber/comm",
    "chamber/partners",
    "chamber/evaluation",
    "chamber/benchmarks",
)

_ADR_RE = re.compile(r"ADR-\d{3}", re.IGNORECASE)


def _is_adr_bearing(path: Path) -> bool:
    rel = path.relative_to(_SRC).as_posix()
    return any(rel.startswith(prefix) for prefix in _ADR_BEARING_PREFIXES)


def _public_symbols_without_adr(path: Path) -> list[str]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    missing: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if node.name.startswith("_"):
            continue
        docstring = ast.get_docstring(node) or ""
        if not _ADR_RE.search(docstring):
            missing.append(node.name)
    return missing


def _adr_bearing_sources() -> list[Path]:
    return [p for p in _SRC.rglob("*.py") if _is_adr_bearing(p)]


@pytest.mark.parametrize("path", _adr_bearing_sources(), ids=lambda p: str(p.relative_to(_SRC)))
def test_public_symbols_have_adr_references(path: Path) -> None:
    missing = _public_symbols_without_adr(path)
    assert not missing, (
        f"{path.relative_to(_SRC)}: public symbols missing ADR reference in docstring: "
        + ", ".join(missing)
    )
