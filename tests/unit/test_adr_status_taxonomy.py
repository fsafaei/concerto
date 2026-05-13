# SPDX-License-Identifier: Apache-2.0
"""Enforce the ADR status taxonomy and ADR ↔ INDEX consistency.

See `adr/ADR-INDEX.md` §Status taxonomy for the canonical set:
RFC, Provisional, Accepted, Validated, Superseded.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_ADR_DIR = Path(__file__).parents[2] / "adr"
_INDEX = _ADR_DIR / "ADR-INDEX.md"

_ADR_FILE_RE = re.compile(r"^ADR-(\d{3})-.+\.md$")
_STATUS_LINE_RE = re.compile(r"^\*\*Status\.\*\*\s+(.+?)\s*$", re.MULTILINE)
_STATUS_TOKEN_RE = re.compile(
    r"^(RFC|Provisional|Accepted|Validated|Superseded)\b",
    re.IGNORECASE,
)
_INDEX_ROW_RE = re.compile(
    r"^\|\s*(\d{3})\s*\|.*\|\s*([^|]+?)\s*\|\s*$",
    re.MULTILINE,
)

_VALID = ("RFC", "Provisional", "Accepted", "Validated", "Superseded")


def _adr_files() -> list[Path]:
    return sorted(p for p in _ADR_DIR.iterdir() if _ADR_FILE_RE.match(p.name))


def _normalise_status_cell(cell: str) -> str:
    """Strip Markdown footnote refs (<sup>...</sup>) from an INDEX status cell."""
    return re.sub(r"<sup>.*?</sup>", "", cell).strip()


def _status_keyword(raw: str) -> str:
    """Return the first taxonomy keyword from a Status string, or empty."""
    match = _STATUS_TOKEN_RE.match(raw.strip())
    return match.group(1) if match else ""


def _extract_file_status(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    match = _STATUS_LINE_RE.search(text)
    if not match:
        return ""
    return _status_keyword(match.group(1))


def _extract_index_statuses() -> dict[str, str]:
    text = _INDEX.read_text(encoding="utf-8")
    out: dict[str, str] = {}
    for row in _INDEX_ROW_RE.finditer(text):
        num, last_cell = row.group(1), row.group(2)
        out[num] = _status_keyword(_normalise_status_cell(last_cell))
    return out


@pytest.mark.parametrize("path", _adr_files(), ids=lambda p: p.name)
def test_adr_status_is_in_taxonomy(path: Path) -> None:
    status = _extract_file_status(path)
    assert status, f"{path.name}: missing or unparsable **Status.** line"
    assert status in _VALID, f"{path.name}: status {status!r} not in taxonomy {_VALID}"


def test_index_statuses_match_per_file() -> None:
    index_statuses = _extract_index_statuses()
    file_statuses = {
        _ADR_FILE_RE.match(p.name).group(1): _extract_file_status(p)  # type: ignore[union-attr]
        for p in _adr_files()
    }
    missing_in_index = sorted(file_statuses.keys() - index_statuses.keys())
    assert not missing_in_index, (
        f"ADR files not represented in ADR-INDEX.md table: {missing_in_index}"
    )
    extra_in_index = sorted(index_statuses.keys() - file_statuses.keys())
    assert not extra_in_index, (
        f"ADR-INDEX.md table rows without a matching ADR file: {extra_in_index}"
    )
    mismatches = {
        num: (file_statuses[num], index_statuses[num])
        for num in sorted(file_statuses)
        if file_statuses[num] != index_statuses[num]
    }
    assert not mismatches, (
        "ADR file status disagrees with ADR-INDEX.md row "
        "(num: (file_status, index_status)): " + repr(mismatches)
    )


def test_index_statuses_are_all_in_taxonomy() -> None:
    for num, status in sorted(_extract_index_statuses().items()):
        assert status in _VALID, (
            f"ADR-INDEX.md row {num}: status {status!r} not in taxonomy {_VALID}"
        )
