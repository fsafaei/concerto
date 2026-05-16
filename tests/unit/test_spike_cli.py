# SPDX-License-Identifier: Apache-2.0
"""``chamber-spike`` CLI tests (T5b.1; plan/07 §5 ``test_spike_cli.py``).

Covers the three subcommands B6 ships:

- ``verify-prereg`` — the load + git-tag SHA check (ADR-007
  §Discipline). The happy-path test builds a tiny tmp_path git repo,
  commits a real prereg YAML, cuts the tag, and asserts the CLI exits
  0 with a ``PASS — ... blob_sha=...`` line. The unhappy-path tests
  pin the three documented failure modes: missing tag, on-disk
  tamper, and missing file. ``--all`` (plan/07 §6 #5) walks the six
  canonical axes in their staged order (AS → OM → CR → CM → PF → SA)
  and aggregates per-axis pass/fail without short-circuiting.
- ``list-axes`` — the six ADR-007 §3.4 axis labels appear in stdout
  along with their Phase-0 stage assignment.
- ``list-profiles`` — every key in :data:`chamber.comm.URLLC_3GPP_R17`
  appears in stdout along with its numeric anchors.

The pre-existing ``train`` subcommand tests live in
:mod:`tests.unit.test_cli_stubs` (M4b-9b) and are untouched by this
PR; the top-level banner-path test there continues to exercise the
no-subcommand path.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from typing import TYPE_CHECKING

import pytest

from chamber.cli.spike import main
from chamber.evaluation.prereg import PREREG_MISMATCH_EXIT_CODE

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Top-level surface
# ---------------------------------------------------------------------------


class TestTopLevelSurface:
    """The dispatcher exposes every subcommand B6 adds (plan/07 §T5b.1)."""

    def test_help_lists_every_subcommand(self, capsys: pytest.CaptureFixture[str]) -> None:
        """`--help` enumerates train, verify-prereg, list-axes, list-profiles."""
        with pytest.raises(SystemExit) as excinfo:
            main(["--help"])
        assert excinfo.value.code == 0
        captured = capsys.readouterr()
        for name in ("train", "verify-prereg", "list-axes", "list-profiles"):
            assert name in captured.out, (
                f"top-level --help missing subcommand {name!r}; got: {captured.out}"
            )


# ---------------------------------------------------------------------------
# verify-prereg
# ---------------------------------------------------------------------------


_REQUIRED_GIT_ENV: dict[str, str] = {
    "GIT_AUTHOR_NAME": "test",
    "GIT_AUTHOR_EMAIL": "test@example.com",
    "GIT_COMMITTER_NAME": "test",
    "GIT_COMMITTER_EMAIL": "test@example.com",
    # Disable any host-level GPG-signing config; the test repo is throwaway.
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_SYSTEM": "/dev/null",
    # Disable any host-level commit-msg / pre-commit hooks the user's
    # template-dir might inject into ``git init``; a CI host with a
    # site-wide template-dir would otherwise pull in hooks here.
    "GIT_TEMPLATE_DIR": "/dev/null",
}

# The fixture deliberately does not pin GIT_AUTHOR_DATE / GIT_COMMITTER_DATE.
# No test asserts a specific commit / blob SHA against a fixed value (the
# happy-path test only checks the SHA shape); if a future test ever needs
# byte-identical SHAs, pin the dates here too.


def _git(*args: str, repo: Path) -> None:
    """Run a git command in ``repo``, raising on non-zero exit."""
    git_bin = shutil.which("git")
    assert git_bin is not None, "git not on PATH; the test cannot proceed"
    env = {**os.environ, **_REQUIRED_GIT_ENV}
    subprocess.run(  # noqa: S603 — args fully resolved
        [git_bin, *args],
        cwd=repo,
        env=env,
        check=True,
        capture_output=True,
    )


# Test-only fixture. The `git_tag` deliberately omits the YYYY-MM-DD
# date suffix the canonical prereg files in ``spikes/preregistration/``
# use (``prereg-stage1-AS-2026-05-15``) so the test does not need to
# be updated annually. The schema does not enforce the date suffix;
# it is a maintainer convention pinned by plan/08 §9.
_VALID_YAML = """\
axis: AS
condition_pair:
  homogeneous_id: stage1_homo_test
  heterogeneous_id: stage1_hetero_test
seeds: [0, 1, 2, 3, 4]
episodes_per_seed: 20
estimator: iqm_success_rate
bootstrap_method: cluster
failure_policy: strict
run_purpose: leaderboard
git_tag: prereg-stage1-AS-test
notes: |
  Throwaway fixture for the chamber-spike verify-prereg test.
"""


@pytest.fixture
def tagged_prereg_repo(tmp_path: Path) -> tuple[Path, Path]:
    """Build a tmp git repo with a committed + tagged prereg YAML.

    Returns ``(repo_root, prereg_path)`` where ``prereg_path`` is
    absolute and ``repo_root`` contains a ``.git`` directory plus
    one commit whose tag matches the YAML's ``git_tag`` field.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    _git("init", "--initial-branch=main", "--quiet", str(repo), repo=tmp_path)
    prereg_dir = repo / "spikes" / "preregistration"
    prereg_dir.mkdir(parents=True)
    prereg_path = prereg_dir / "AS.yaml"
    prereg_path.write_text(_VALID_YAML, encoding="utf-8")
    _git("add", "spikes/preregistration/AS.yaml", repo=repo)
    _git("commit", "--no-gpg-sign", "-m", "add AS prereg", repo=repo)
    _git("tag", "-a", "prereg-stage1-AS-test", "-m", "tag AS prereg", repo=repo)
    return repo, prereg_path


class TestVerifyPreregHappyPath:
    """The happy path: on-disk YAML matches the tagged blob SHA."""

    def test_pass_exits_zero(
        self,
        tagged_prereg_repo: tuple[Path, Path],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """verify-prereg returns 0 + prints PASS + blob_sha on success."""
        repo, prereg_path = tagged_prereg_repo
        rc = main(
            [
                "verify-prereg",
                "--spike",
                str(prereg_path),
                "--repo-root",
                str(repo),
            ]
        )
        assert rc == 0, f"expected exit 0, got {rc}"
        captured = capsys.readouterr()
        assert "PASS" in captured.out
        assert "axis=AS" in captured.out
        assert "tag=prereg-stage1-AS-test" in captured.out
        assert "blob_sha=" in captured.out


class TestVerifyPreregFailureModes:
    """Pin the three documented failure paths (ADR-007 §Discipline)."""

    def test_missing_file_exits_nonzero(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A non-existent --spike path exits non-zero with a stderr message."""
        rc = main(
            [
                "verify-prereg",
                "--spike",
                str(tmp_path / "does_not_exist.yaml"),
                "--repo-root",
                str(tmp_path),
            ]
        )
        assert rc != 0
        captured = capsys.readouterr()
        assert "not found" in captured.err

    def test_missing_tag_exits_nonzero(
        self,
        tagged_prereg_repo: tuple[Path, Path],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A YAML whose tag does not exist exits non-zero."""
        repo, prereg_path = tagged_prereg_repo
        _git("tag", "-d", "prereg-stage1-AS-test", repo=repo)
        rc = main(
            [
                "verify-prereg",
                "--spike",
                str(prereg_path),
                "--repo-root",
                str(repo),
            ]
        )
        assert rc != 0
        captured = capsys.readouterr()
        assert "FAIL" in captured.err
        assert "does not exist" in captured.err

    def test_tampered_yaml_exits_nonzero_with_sha_mismatch_message(
        self,
        tagged_prereg_repo: tuple[Path, Path],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Modifying the YAML's bytes post-tag must surface a SHA mismatch."""
        repo, prereg_path = tagged_prereg_repo
        # Tamper: change the homogeneous_id by one character so the
        # bytes (and therefore the blob SHA) shift but the schema still
        # validates.
        original = prereg_path.read_text(encoding="utf-8")
        tampered = original.replace("stage1_homo_test", "stage1_homo_TAMP")
        assert tampered != original
        prereg_path.write_text(tampered, encoding="utf-8")
        rc = main(
            [
                "verify-prereg",
                "--spike",
                str(prereg_path),
                "--repo-root",
                str(repo),
            ]
        )
        assert rc != 0
        captured = capsys.readouterr()
        assert "FAIL" in captured.err
        assert "blob SHA mismatch" in captured.err


class TestVerifyPreregRepoRootInference:
    """--repo-root is optional: the subcommand walks up for a .git ancestor."""

    def test_repo_root_inferred_from_spike_path(
        self,
        tagged_prereg_repo: tuple[Path, Path],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Calling verify-prereg without --repo-root finds the enclosing repo."""
        _repo, prereg_path = tagged_prereg_repo
        rc = main(["verify-prereg", "--spike", str(prereg_path)])
        assert rc == 0, f"expected exit 0, got {rc}"
        captured = capsys.readouterr()
        assert "PASS" in captured.out


# ---------------------------------------------------------------------------
# verify-prereg --all
# ---------------------------------------------------------------------------


def _make_axis_yaml(axis: str) -> str:
    """Build a minimal-but-valid prereg YAML body for a given axis label."""
    return (
        f"axis: {axis}\n"
        "condition_pair:\n"
        f"  homogeneous_id: stage_homo_{axis.lower()}\n"
        f"  heterogeneous_id: stage_hetero_{axis.lower()}\n"
        "seeds: [0, 1, 2, 3, 4]\n"
        "episodes_per_seed: 20\n"
        "estimator: iqm_success_rate\n"
        "bootstrap_method: cluster\n"
        "failure_policy: strict\n"
        "run_purpose: leaderboard\n"
        f"git_tag: prereg-{axis}-test\n"
        "notes: Throwaway fixture for verify-prereg --all test.\n"
    )


@pytest.fixture
def six_axis_tagged_repo(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    """Build a tmp git repo with all six canonical prereg YAMLs committed + tagged.

    Returns ``(repo_root, {axis: prereg_path})``.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    _git("init", "--initial-branch=main", "--quiet", str(repo), repo=tmp_path)
    prereg_dir = repo / "spikes" / "preregistration"
    prereg_dir.mkdir(parents=True)
    paths: dict[str, Path] = {}
    for axis in ("AS", "OM", "CR", "CM", "PF", "SA"):
        path = prereg_dir / f"{axis}.yaml"
        path.write_text(_make_axis_yaml(axis), encoding="utf-8")
        _git("add", f"spikes/preregistration/{axis}.yaml", repo=repo)
        _git("commit", "--no-gpg-sign", "-m", f"add {axis} prereg", repo=repo)
        _git("tag", "-a", f"prereg-{axis}-test", "-m", f"tag {axis} prereg", repo=repo)
        paths[axis] = path
    return repo, paths


_AXIS_LINE_RE = re.compile(r"axis=([A-Z]{2})")


class TestVerifyPreregAll:
    """``chamber-spike verify-prereg --all`` (plan/07 §6 #5)."""

    def test_passes_on_clean_corpus(
        self,
        six_axis_tagged_repo: tuple[Path, dict[str, Path]],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Six clean YAMLs ⇒ exit 0; one PASS line per axis in canonical order."""
        repo, _paths = six_axis_tagged_repo
        rc = main(["verify-prereg", "--all", "--repo-root", str(repo)])
        assert rc == 0, f"expected exit 0, got {rc}"
        captured = capsys.readouterr()
        axis_lines = [ln for ln in captured.out.splitlines() if "axis=" in ln]
        assert len(axis_lines) == 6, f"expected 6 axis lines, got {axis_lines!r}"
        seen = [_AXIS_LINE_RE.search(ln).group(1) for ln in axis_lines]  # type: ignore[union-attr]
        assert seen == ["AS", "OM", "CR", "CM", "PF", "SA"], f"axes printed out of order: {seen!r}"
        for ln in axis_lines:
            assert "PASS" in ln, f"expected PASS on line {ln!r}"
            assert "blob_sha=" in ln, f"expected blob_sha on line {ln!r}"

    def test_fails_loud_on_one_tamper(
        self,
        six_axis_tagged_repo: tuple[Path, dict[str, Path]],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """One tampered YAML ⇒ exit 4; offending axis on stderr; other five PASS."""
        repo, paths = six_axis_tagged_repo
        tampered_path = paths["OM"]
        original = tampered_path.read_text(encoding="utf-8")
        tampered = original.replace("stage_homo_om", "stage_homo_TAMP")
        assert tampered != original
        tampered_path.write_text(tampered, encoding="utf-8")

        rc = main(["verify-prereg", "--all", "--repo-root", str(repo)])
        assert rc == PREREG_MISMATCH_EXIT_CODE, (
            f"expected exit {PREREG_MISMATCH_EXIT_CODE}, got {rc}"
        )
        captured = capsys.readouterr()
        # Offending axis named explicitly on stderr.
        assert "axis=OM" in captured.err, (
            f"expected offending axis OM on stderr; got: {captured.err!r}"
        )
        assert "FAIL" in captured.err
        # The other five still report PASS on stdout — user sees the full picture.
        for axis in ("AS", "CR", "CM", "PF", "SA"):
            assert f"axis={axis}" in captured.out, (
                f"expected non-failing axis {axis} on stdout; got: {captured.out!r}"
            )

    def test_mutually_exclusive_with_spike(
        self,
        six_axis_tagged_repo: tuple[Path, dict[str, Path]],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Passing both --spike and --all is rejected by argparse (exit 2)."""
        repo, paths = six_axis_tagged_repo
        with pytest.raises(SystemExit) as excinfo:
            main(
                [
                    "verify-prereg",
                    "--spike",
                    str(paths["AS"]),
                    "--all",
                    "--repo-root",
                    str(repo),
                ]
            )
        assert excinfo.value.code == 2

    def test_requires_either_spike_or_all(self, tmp_path: Path) -> None:
        """``verify-prereg`` with neither flag is rejected by argparse (exit 2)."""
        del tmp_path
        with pytest.raises(SystemExit) as excinfo:
            main(["verify-prereg"])
        assert excinfo.value.code == 2

    def test_fails_loud_on_missing_axis_yaml(
        self,
        six_axis_tagged_repo: tuple[Path, dict[str, Path]],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A canonical axis YAML deleted post-corpus ⇒ exit 4; the gap is named loudly."""
        repo, paths = six_axis_tagged_repo
        # Pretend the PF YAML was never checked in.
        paths["PF"].unlink()
        rc = main(["verify-prereg", "--all", "--repo-root", str(repo)])
        assert rc == PREREG_MISMATCH_EXIT_CODE, (
            f"expected exit {PREREG_MISMATCH_EXIT_CODE}, got {rc}"
        )
        captured = capsys.readouterr()
        assert "axis=PF" in captured.err
        assert "not found" in captured.err
        for axis in ("AS", "OM", "CR", "CM", "SA"):
            assert f"axis={axis}" in captured.out, (
                f"expected non-failing axis {axis} on stdout; got: {captured.out!r}"
            )

    def test_skips_non_axis_yamls(
        self,
        six_axis_tagged_repo: tuple[Path, dict[str, Path]],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A stray ``_template.yaml`` / ``README.md`` in the prereg dir is ignored."""
        repo, _paths = six_axis_tagged_repo
        prereg_dir = repo / "spikes" / "preregistration"
        (prereg_dir / "_template.yaml").write_text(
            "axis: GARBAGE\nthis is intentionally invalid\n",
            encoding="utf-8",
        )
        (prereg_dir / "README.md").write_text("# decoy\n", encoding="utf-8")
        rc = main(["verify-prereg", "--all", "--repo-root", str(repo)])
        assert rc == 0, f"expected exit 0 (strays ignored), got {rc}"
        captured = capsys.readouterr()
        assert "_template" not in captured.out
        assert "_template" not in captured.err
        assert "README" not in captured.out
        assert "README" not in captured.err


# ---------------------------------------------------------------------------
# list-axes
# ---------------------------------------------------------------------------


class TestListAxes:
    """All six ADR-007 §3.4 axes appear in stdout (T5b.1)."""

    def test_lists_all_six_axes(self, capsys: pytest.CaptureFixture[str]) -> None:
        rc = main(["list-axes"])
        assert rc == 0
        captured = capsys.readouterr()
        for axis in ("AS", "OM", "CR", "CM", "PF", "SA"):
            assert axis in captured.out, f"missing axis {axis!r}"
        # Stage column should distinguish AS+OM (1) / CR+CM (2) / PF+SA (3).
        assert "1" in captured.out
        assert "2" in captured.out
        assert "3" in captured.out


# ---------------------------------------------------------------------------
# list-profiles
# ---------------------------------------------------------------------------


class TestListProfiles:
    """Every chamber.comm.URLLC_3GPP_R17 profile name appears in stdout (T5b.1)."""

    def test_lists_all_six_profiles(self, capsys: pytest.CaptureFixture[str]) -> None:
        from chamber.comm import URLLC_3GPP_R17

        rc = main(["list-profiles"])
        assert rc == 0
        captured = capsys.readouterr()
        for name in URLLC_3GPP_R17:
            assert name in captured.out, f"missing profile {name!r}"

    def test_includes_numeric_columns(self, capsys: pytest.CaptureFixture[str]) -> None:
        rc = main(["list-profiles"])
        assert rc == 0
        captured = capsys.readouterr()
        for column in ("latency_mean_ms", "latency_std_ms", "drop_rate"):
            assert column in captured.out, f"missing column header {column!r}"


# Pytest-skip guard: every test in this file invokes ``git`` via
# subprocess. The ``git`` binary is required by the project's existing
# pre-commit hooks (and by ``chamber.evaluation.prereg.verify_git_tag``
# itself), so this guard is defensive — if it ever fires on a CI host
# the project's other tests would also be broken.
if shutil.which("git") is None:  # pragma: no cover
    pytest.skip("git binary not available", allow_module_level=True)
