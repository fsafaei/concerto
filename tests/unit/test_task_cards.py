# SPDX-License-Identifier: Apache-2.0
"""Tier-1 tests for the generated task-card / README-table surfaces (ADR-027 §Consequences).

Pins the drift contract: the committed ``docs/reference/tasks/`` cards
and the marker-delimited README task table must match a fresh render
from the ``chamber.tasks`` registry, byte for byte. These tests are the
in-``make test`` mirror of ``make verify-readme-tables`` so the drift
gate cannot be skipped by running only the test suite.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

import chamber.tasks
from chamber.tasks.cards import CARD_PROSE, card_prose
from chamber.tasks.render import (
    README_TABLE_BEGIN,
    README_TABLE_END,
    render_all_cards,
    render_readme_table,
    render_task_card,
)

if TYPE_CHECKING:
    from types import ModuleType

_REPO_ROOT = Path(__file__).parents[2]
_CARDS_DIR = _REPO_ROOT / "docs" / "reference" / "tasks"


def _registered_task_ids() -> set[str]:
    return {slug.partition("@v")[0] for slug in chamber.tasks.list_registered()}


def _load_script(name: str) -> ModuleType:
    path = _REPO_ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(f"_{path.stem}_under_test", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestCardProse:
    def test_every_registered_task_has_prose(self) -> None:
        assert set(CARD_PROSE) == _registered_task_ids()

    def test_unknown_id_lists_known(self) -> None:
        with pytest.raises(KeyError, match="cocarry"):
            card_prose("nope")

    def test_tier3_candidates_state_their_falsifier(self) -> None:
        for task_id in ("co_hold_secure", "amr_handover_dynamic"):
            spec = chamber.tasks.get(task_id)
            assert spec.tier == 3
            assert card_prose(task_id).falsifier, f"{task_id} card must state its falsifier"

    def test_coinsert_states_the_open_challenge(self) -> None:
        prose = card_prose("coinsert")
        assert prose.open_challenge is not None
        assert "open challenge" in prose.open_challenge

    def test_descriptions_are_three_sentences_of_plain_language(self) -> None:
        for task_id, prose in CARD_PROSE.items():
            sentences = [s for s in prose.description.split(". ") if s.strip()]
            assert len(sentences) >= 3, f"{task_id}: card description under three sentences"


class TestRenderedSurfaces:
    def test_rendered_card_carries_the_required_sections(self) -> None:
        card = render_task_card(chamber.tasks.get("cocarry"))
        for section in (
            "## Spaces",
            "## Stress channel",
            "## Axis validity",
            "## Industrial analogue",
            "## Evidence",
            "## How to run one episode",
        ):
            assert section in card

    def test_committed_cards_match_the_registry_render(self) -> None:
        expected = render_all_cards()
        committed = {p.name: p.read_text(encoding="utf-8") for p in _CARDS_DIR.glob("*.md")}
        assert committed == expected, (
            "docs/reference/tasks/ drifts from chamber.tasks; run "
            "`uv run python scripts/render_task_cards.py` and commit"
        )

    def test_committed_readme_block_matches_the_registry_render(self) -> None:
        readme = (_REPO_ROOT / "README.md").read_text(encoding="utf-8")
        begin = readme.find(README_TABLE_BEGIN)
        end = readme.find(README_TABLE_END) + len(README_TABLE_END)
        assert begin != -1, "README BEGIN marker missing"
        assert end != -1, "README END marker missing"
        assert readme[begin:end] + "\n" == render_readme_table(), (
            "README task table drifts from chamber.tasks; run "
            "`uv run python scripts/render_task_table.py` and commit"
        )

    def test_every_card_has_a_mkdocs_nav_entry(self) -> None:
        nav_text = (_REPO_ROOT / "mkdocs.yml").read_text(encoding="utf-8")
        for name in render_all_cards():
            assert f"reference/tasks/{name}" in nav_text, (
                f"docs/reference/tasks/{name} missing from mkdocs.yml nav "
                "(mkdocs --strict would fail on the orphan page)"
            )


class TestRenderScripts:
    def test_render_task_table_check_passes_on_committed_tree(self) -> None:
        module = _load_script("render_task_table.py")
        assert module.main(["--check"]) == 0

    def test_render_task_cards_check_passes_on_committed_tree(self) -> None:
        module = _load_script("render_task_cards.py")
        assert module.main(["--check"]) == 0

    def test_render_task_cards_check_flags_membership_drift(self, tmp_path: Path) -> None:
        module = _load_script("render_task_cards.py")
        for name, content in render_all_cards().items():
            (tmp_path / name).write_text(content, encoding="utf-8")
        (tmp_path / "stray.md").write_text("stale\n", encoding="utf-8")
        findings = module.diff_cards(tmp_path)
        assert findings == ["stale card not in the registry: stray.md"]

    def test_render_task_table_splice_requires_markers(self) -> None:
        module = _load_script("render_task_table.py")
        with pytest.raises(ValueError, match="markers"):
            module.splice_table("no markers here", render_readme_table())
