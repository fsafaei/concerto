#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Verify ``CHANGELOG.md`` completeness against ``git log`` (closes #126).

``release-please``'s Conventional-Commits parser has been observed silently
dropping release-worthy commits from the generated CHANGELOG (PR #123 dropped
from the v0.3.0 draft; see issue #126). The upstream root cause is unresolved
and may take multiple iterations to pin down; this script closes the door from
the *project* side by re-running a minimal Conventional-Commits filter over
``git log <prev-tag>..HEAD`` and asserting every release-worthy commit's short
SHA appears in the top ``CHANGELOG.md`` section.

The check is silent on non-release branches: if the top CHANGELOG section's
version already matches the latest git tag, ``release-please`` has not yet
proposed a new section, so there is nothing to verify. The check is only
load-bearing on the ``release-please--branches--main`` PR (and on ``main``
between the merge of that PR and the new tag).

Release-worthy types (matches ``release-type: python``'s defaults + the
project's empirical CHANGELOG sections):

- ``feat`` → ``Features``
- ``fix``  → ``Bug Fixes``
- ``docs`` → ``Documentation``

Excluded types: ``chore`` (filtered by release-please), ``style``, ``refactor``,
``test``, ``ci``, ``perf``, ``build``, ``revert``. None of these have ever
produced an entry in this project's CHANGELOG.

Exit codes:

- 0 — completeness verified, or no release section to verify.
- 1 — at least one release-worthy commit's SHA is missing from the top
  CHANGELOG section.
- 2 — usage / I/O error (no previous tag found, CHANGELOG unreadable, ...).

Usage::

    uv run python scripts/check_changelog_completeness.py
    uv run python scripts/check_changelog_completeness.py --prev-tag v0.3.1
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CHANGELOG_PATH = _REPO_ROOT / "CHANGELOG.md"

RELEASE_WORTHY_TYPES: frozenset[str] = frozenset({"feat", "fix", "docs"})

# Conventional-Commits subject: ``<type>(<scope>)?!?: <description>``.
# ``<scope>`` may contain commas/slashes (``feat(training,adr): …``);
# ``!`` marks a breaking change. We only care about ``<type>`` here.
_TYPE_RE = re.compile(r"^(?P<type>[a-z]+)(?:\([^)]+\))?!?:")

# release-please's top section header, e.g. ``## [0.4.0](...) (2026-05-16)``.
_VERSION_HEADER_RE = re.compile(r"^## \[(?P<version>\d+\.\d+\.\d+)\]")

# A semver-shaped git tag, ``vX.Y.Z`` (no pre-release suffix; the project's
# release-please config does not emit those).
_RELEASE_TAG_RE = re.compile(r"^v\d+\.\d+\.\d+$")


@dataclass(frozen=True)
class Commit:
    """One commit returned by ``git log --pretty=format:%H%x09%s``."""

    sha: str
    subject: str

    @property
    def short(self) -> str:
        """Seven-character abbreviated SHA (release-please's CHANGELOG form)."""
        return self.sha[:7]


def parse_type(subject: str) -> str | None:
    """Return the Conventional-Commits ``<type>`` of ``subject``, or ``None``.

    Subjects that do not match the ``<type>(<scope>)?!?:`` shape (e.g. plain
    ``Merge branch …`` or release-please's own scope-less headlines) return
    ``None`` so the caller can decide how to treat them.
    """
    m = _TYPE_RE.match(subject)
    return m["type"] if m else None


def is_release_worthy(subject: str) -> bool:
    """Return ``True`` iff ``subject``'s type is in :data:`RELEASE_WORTHY_TYPES`."""
    return parse_type(subject) in RELEASE_WORTHY_TYPES


def find_missing(commits: list[Commit], changelog_body: str) -> list[Commit]:
    """Return release-worthy ``commits`` whose short SHA is absent from ``changelog_body``.

    The match is on the 7-char short SHA. release-please embeds the SHA twice
    in each entry (short as the link text, long as the link target), so a
    substring search on the short form is sufficient and matches both forms.
    """
    return [c for c in commits if is_release_worthy(c.subject) and c.short not in changelog_body]


def split_top_section(changelog_text: str) -> tuple[str, str] | None:
    """Return ``(version, body)`` of the first ``## [X.Y.Z]`` section.

    ``body`` is everything between the top version header and the next ``## ``
    line (exclusive on both ends). Returns ``None`` if the file has no version
    headers (vacuous: nothing to verify).
    """
    lines = changelog_text.splitlines()
    start: int | None = None
    for i, line in enumerate(lines):
        if _VERSION_HEADER_RE.match(line):
            start = i
            break
    if start is None:
        return None
    m = _VERSION_HEADER_RE.match(lines[start])
    if m is None:  # unreachable: the loop only stops on a match
        raise RuntimeError("unreachable: version header re-match failed")
    version = m["version"]
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if lines[j].startswith("## "):
            end = j
            break
    body = "\n".join(lines[start + 1 : end])
    return version, body


def _git(*args: str) -> str:
    """Run ``git`` and return stripped stdout.

    Raises :class:`subprocess.CalledProcessError` on non-zero exit so the
    caller can decide whether to recover.
    """
    # S603/S607: ``git`` is a trusted, hard-coded executable name; argv is built
    # from a tuple literal + maintainer-controlled strings, not user input.
    return subprocess.run(  # noqa: S603
        ("git", *args),  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
        cwd=_REPO_ROOT,
    ).stdout.strip()


def latest_release_tag() -> str | None:
    """Return the latest ``vX.Y.Z`` tag, or ``None`` if no such tag exists."""
    tags = _git("tag", "--sort=-v:refname").splitlines()
    for tag in tags:
        if _RELEASE_TAG_RE.match(tag):
            return tag
    return None


def commits_between(prev: str, head: str) -> list[Commit]:
    """Return commits in ``prev..head`` (exclusive..inclusive), newest first."""
    out = _git("log", "--pretty=format:%H%x09%s", f"{prev}..{head}")
    if not out:
        return []
    pairs: list[Commit] = []
    for line in out.splitlines():
        sha, _, subject = line.partition("\t")
        pairs.append(Commit(sha=sha, subject=subject))
    return pairs


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. See module docstring for exit-code semantics."""
    parser = argparse.ArgumentParser(
        description="Verify CHANGELOG.md completeness against git log "
        "since the previous release tag (closes #126).",
    )
    parser.add_argument(
        "--prev-tag",
        help="Previous release tag (default: auto-detected from `git tag`).",
    )
    parser.add_argument(
        "--head",
        default="HEAD",
        help="Head ref to check up to (default: HEAD).",
    )
    args = parser.parse_args(argv)

    if not _CHANGELOG_PATH.exists():
        print(
            f"check_changelog_completeness: {_CHANGELOG_PATH} not found.",
            file=sys.stderr,
        )
        return 2

    top = split_top_section(_CHANGELOG_PATH.read_text(encoding="utf-8"))
    if top is None:
        print("check_changelog_completeness: CHANGELOG.md has no ## [X.Y.Z] sections; skipping.")
        return 0
    top_version, top_body = top

    latest_tag = latest_release_tag()

    # Non-release branch: top CHANGELOG section equals the latest tag, so
    # release-please has not proposed a new section yet. Nothing to verify.
    # An explicit --prev-tag overrides the auto-skip — a maintainer running
    # the check manually (e.g. backfilling against an older release) wants the
    # verification to run regardless.
    if args.prev_tag is None and latest_tag == f"v{top_version}":
        print(
            f"check_changelog_completeness: CHANGELOG top section "
            f"[{top_version}] matches latest tag {latest_tag}; "
            "no pending release to verify."
        )
        return 0

    prev_tag = args.prev_tag or latest_tag
    if prev_tag is None:
        print(
            "check_changelog_completeness: no previous release tag found; "
            "cannot determine commit range.",
            file=sys.stderr,
        )
        return 2

    print(
        f"check_changelog_completeness: verifying CHANGELOG section "
        f"[{top_version}] against commits in {prev_tag}..{args.head}"
    )

    commits = commits_between(prev_tag, args.head)
    missing = find_missing(commits, top_body)

    if missing:
        print(
            f"\nCHANGELOG section [{top_version}] is missing "
            f"{len(missing)} release-worthy commit(s):",
            file=sys.stderr,
        )
        for c in missing:
            print(f"  {c.short}  {c.subject}", file=sys.stderr)
        print(
            "\nThis is the same class of skip that issue #126 tracks. "
            "Either (a) re-run release-please-action on a fresh branch to "
            "regenerate the CHANGELOG, or (b) manually patch the missing "
            "entries onto the release-please PR before merging.",
            file=sys.stderr,
        )
        return 1

    n_release_worthy = sum(1 for c in commits if is_release_worthy(c.subject))
    print(
        f"check_changelog_completeness: all {n_release_worthy} "
        f"release-worthy commit(s) present in [{top_version}]."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
