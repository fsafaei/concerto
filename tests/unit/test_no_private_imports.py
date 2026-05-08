# SPDX-License-Identifier: Apache-2.0
"""Enforce P2: no private ManiSkill imports anywhere in src/."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_SRC = Path(__file__).parents[2] / "src"
_PRIVATE_RE = re.compile(
    r"^\s*(?:from|import)\s+mani_skill(?:\.\w+)*\s+import\s+_",
    re.MULTILINE,
)
_PRIVATE_MODULE_RE = re.compile(
    r"^\s*(?:from|import)\s+mani_skill\._",
    re.MULTILINE,
)


def _python_sources() -> list[Path]:
    return list(_SRC.rglob("*.py"))


@pytest.mark.parametrize("path", _python_sources(), ids=lambda p: str(p.relative_to(_SRC)))
def test_no_private_mani_skill_imports(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    assert not _PRIVATE_RE.search(source), (
        f"{path.relative_to(_SRC)}: imports a private ManiSkill symbol (violates P2)"
    )
    assert not _PRIVATE_MODULE_RE.search(source), (
        f"{path.relative_to(_SRC)}: imports a private ManiSkill sub-module (violates P2)"
    )
