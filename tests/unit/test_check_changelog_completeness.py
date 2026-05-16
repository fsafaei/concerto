# SPDX-License-Identifier: Apache-2.0
"""``scripts/check_changelog_completeness.py`` unit tests (closes #126).

Pins the script's pure helpers against three classes of input:

- Conventional-Commits subject parsing covers the empirical shapes observed
  in this repo's history (scope-less, simple scope, comma-scope, ``!``
  marker, plain non-CC subject, release-please's own ``chore(main): release``
  headline).
- CHANGELOG top-section extraction handles a normal multi-section file, a
  zero-section file, and a single-section file.
- ``find_missing`` reproduces the exact scenario from issue #126: PR #123's
  ``feat(benchmarks): …`` is silently dropped while ``feat(cli): …`` and
  ``fix(benchmarks): …`` make it through.

The script is loaded by file path (``scripts/`` is not a Python package); the
loader pattern mirrors :mod:`tests.unit.test_coverage_floors`.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "check_changelog_completeness.py"


@pytest.fixture(scope="module")
def script_module() -> ModuleType:
    """Load ``scripts/check_changelog_completeness.py`` as a private module."""
    spec = importlib.util.spec_from_file_location(
        "_check_changelog_completeness_under_test", _SCRIPT_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestParseType:
    """Conventional-Commits ``<type>`` extraction."""

    @pytest.mark.parametrize(
        ("subject", "expected"),
        [
            ("feat: add x", "feat"),
            ("fix(safety): handle nan", "fix"),
            ("docs(adr): ADR-007 revision 4", "docs"),
            ("feat(training,adr): reject non-frozen partner", "feat"),
            ("feat(api)!: breaking rename", "feat"),
            ("chore(main): release 0.4.0", "chore"),
            ("Merge branch 'main' into feat/x", None),
            ("", None),
            ("WIP: random words", None),
            # Empirical: PR #123's headline shape (the one release-please dropped).
            (
                "feat(benchmarks): EgoActionFactory Protocol + Tier-1 contract"
                " tests for the Stage-1 seam (plan/07 §T5b.2) (#123)",
                "feat",
            ),
        ],
    )
    def test_parses_conventional_commits_subjects(
        self, script_module: ModuleType, subject: str, expected: str | None
    ) -> None:
        assert script_module.parse_type(subject) == expected


class TestIsReleaseWorthy:
    """The three release-worthy types match the project's empirical CHANGELOG."""

    @pytest.mark.parametrize(
        "subject",
        [
            "feat: x",
            "fix: y",
            "docs(adr): z",
            "feat(scope)!: breaking",
        ],
    )
    def test_release_worthy_types(self, script_module: ModuleType, subject: str) -> None:
        assert script_module.is_release_worthy(subject) is True

    @pytest.mark.parametrize(
        "subject",
        [
            "chore(deps): bump x",
            "chore(main): release 0.4.0",
            "test(unit): add y",
            "ci: tweak workflow",
            "style: black",
            "refactor(api): rename",
            "perf: micro-opt",
            "build: bump action",
            "revert: oops",
            "Merge pull request #1",
            "",
        ],
    )
    def test_filtered_types(self, script_module: ModuleType, subject: str) -> None:
        assert script_module.is_release_worthy(subject) is False


class TestSplitTopSection:
    """Top-section extraction from a CHANGELOG.md body."""

    def test_multi_section_returns_top(self, script_module: ModuleType) -> None:
        text = (
            "# Changelog\n"
            "\n"
            "Intro paragraph.\n"
            "\n"
            "## [0.4.0](https://example.com) (2026-05-16)\n"
            "\n"
            "### Features\n"
            "\n"
            "* **packaging:** move HARL ([ac9ef22](https://example.com))\n"
            "\n"
            "## [0.3.1](https://example.com) (2026-05-16)\n"
            "\n"
            "### Bug Fixes\n"
            "\n"
            "* **ci:** previous-release content\n"
        )
        result = script_module.split_top_section(text)
        assert result is not None
        version, body = result
        assert version == "0.4.0"
        assert "ac9ef22" in body
        assert "previous-release content" not in body

    def test_no_version_headers_returns_none(self, script_module: ModuleType) -> None:
        text = "# Changelog\n\nNothing here yet.\n"
        assert script_module.split_top_section(text) is None

    def test_single_section_includes_full_tail(self, script_module: ModuleType) -> None:
        text = (
            "# Changelog\n"
            "\n"
            "## [0.1.0](https://example.com) (2026-05-01)\n"
            "\n"
            "### Features\n"
            "\n"
            "* initial release ([deadbee](https://example.com))\n"
        )
        result = script_module.split_top_section(text)
        assert result is not None
        version, body = result
        assert version == "0.1.0"
        assert "deadbee" in body


class TestFindMissing:
    """End-to-end: reproduce the issue #126 skip scenario."""

    def test_reproduces_issue_126(self, script_module: ModuleType) -> None:
        # Three release-worthy commits: feat(cli) + feat(benchmarks) + fix(benchmarks).
        # The CHANGELOG body embeds only two of them; feat(benchmarks)'s short
        # SHA (the PR #123 analog) is the one release-please silently dropped.
        Commit = script_module.Commit
        commits = [
            Commit(
                sha="e06bda93588bbc7db89f28fb28c0dff0ddd746fb",
                subject="feat(cli): chamber-spike summarize-month3 (#124)",
            ),
            Commit(
                sha="04be156d05ef71ab3104c0ec42521020c6290709",
                subject="feat(benchmarks): EgoActionFactory Protocol (#123)",
            ),
            Commit(
                sha="73070a1d2fa2023c22f0818dd5c55427b2594255",
                subject="fix(benchmarks): OM tuple-collision in stage1_om (#122)",
            ),
            Commit(
                sha="63c6885ffffffffffffffffffffffffffffffff0",
                subject="chore(repro): stage1_as.sh + stage1_om.sh (#118)",
            ),
        ]
        body = (
            "### Features\n"
            "\n"
            "* **cli:** chamber-spike summarize-month3 ([e06bda9](https://x/commit/e06bda93588bbc7db89f28fb28c0dff0ddd746fb))\n"
            "\n"
            "### Bug Fixes\n"
            "\n"
            "* **benchmarks:** OM tuple-collision ([73070a1](https://x/commit/73070a1d2fa2023c22f0818dd5c55427b2594255))\n"
        )
        missing = script_module.find_missing(commits, body)
        # Only the feat(benchmarks) commit is release-worthy *and* absent.
        # chore(repro) is filtered out before the SHA check.
        assert [c.short for c in missing] == ["04be156"]
        assert "benchmarks" in missing[0].subject

    def test_empty_when_all_present(self, script_module: ModuleType) -> None:
        Commit = script_module.Commit
        commits = [
            Commit(sha="aaaaaaa1234567890" + "0" * 24, subject="feat: a"),
            Commit(sha="bbbbbbb1234567890" + "0" * 24, subject="fix: b"),
        ]
        body = "feat link ([aaaaaaa](..)) and fix link ([bbbbbbb](..))"
        assert script_module.find_missing(commits, body) == []

    def test_chore_commits_never_flagged(self, script_module: ModuleType) -> None:
        # chore commits are intentionally filtered by release-please. Even if
        # their SHA is absent from the CHANGELOG, the script must not flag them.
        Commit = script_module.Commit
        commits = [
            Commit(sha="c" * 40, subject="chore(deps): bump ubuntu"),
            Commit(sha="d" * 40, subject="chore(main): release 0.4.0"),
        ]
        assert script_module.find_missing(commits, "") == []
