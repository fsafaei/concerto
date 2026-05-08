# SPDX-License-Identifier: Apache-2.0
"""Enforce dependency-direction rule: concerto.* must not import chamber.*."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_CONCERTO_SRC = Path(__file__).parents[2] / "src" / "concerto"
_IMPORT_RE = re.compile(
    r"^\s*(?:from\s+chamber|import\s+chamber)",
    re.MULTILINE,
)


def _concerto_sources() -> list[Path]:
    return list(_CONCERTO_SRC.rglob("*.py"))


@pytest.mark.parametrize(
    "path", _concerto_sources(), ids=lambda p: str(p.relative_to(_CONCERTO_SRC))
)
def test_no_chamber_import_in_concerto(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    rel = path.relative_to(_CONCERTO_SRC)
    assert not _IMPORT_RE.search(source), (
        f"concerto/{rel}: imports chamber (violates dependency-direction rule)"
    )
