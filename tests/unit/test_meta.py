# SPDX-License-Identifier: Apache-2.0
"""Verify package versions match pyproject.toml."""

from __future__ import annotations

import tomllib
from pathlib import Path

import chamber
import concerto


def _pyproject_version() -> str:
    root = Path(__file__).parents[2]
    with open(root / "pyproject.toml", "rb") as f:
        data = tomllib.load(f)
    return data["project"]["version"]


def test_concerto_version_matches_pyproject() -> None:
    assert concerto.__version__ == _pyproject_version()


def test_chamber_version_matches_pyproject() -> None:
    assert chamber.__version__ == _pyproject_version()
