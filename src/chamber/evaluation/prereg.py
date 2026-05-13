# SPDX-License-Identifier: Apache-2.0
"""Pre-registration loader + git-tag SHA verification (ADR-007 §Discipline).

ADR-007 §Discipline requires every Phase-0 spike run to be locked
against a pre-registration YAML *before* the run launches: the YAML
is committed, a git tag is created on the commit, and the YAML's
blob SHA at the tag must remain content-identical to the file on
disk at launch time. Editing the YAML after a spike has launched is
the project anti-pattern that this module's
:func:`verify_git_tag` refuses to permit.

The loader uses Pydantic v2 for the YAML schema so unknown keys or
missing required fields fail loudly; the verifier shells out to
``git`` for blob-SHA comparison (we don't depend on libgit2 to keep
the wheel light).
"""

from __future__ import annotations

import shutil
import subprocess
from typing import TYPE_CHECKING, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator

from chamber.evaluation.results import (
    ConditionPair,  # noqa: TC001 — Pydantic field type, needs runtime visibility
)

if TYPE_CHECKING:
    from pathlib import Path

#: Permitted ADR-007 §3.4 axis labels (Option D shortlist).
_AXIS_LABELS = frozenset({"CR", "AS", "OM", "CM", "PF", "SA"})

#: Permitted bootstrap-method labels (ADR-008 §Decision + reviewer P1-9).
BootstrapMethod = Literal["cluster", "hierarchical", "iid"]

#: Permitted failure-policy labels.
FailurePolicy = Literal["strict", "best_effort"]

#: Permitted run-purpose labels (reviewer P1-5).
#:
#: ``leaderboard`` is the default and forbids the pooled IID bootstrap;
#: ``power`` runs are exempt because power calculations frequently want
#: the IID variance baseline against which to size cluster-aware
#: bootstraps; ``debug`` runs are exempt because they are not admitted
#: to the leaderboard at all.
RunPurpose = Literal["power", "leaderboard", "debug"]


class PreregistrationSpec(BaseModel):
    """Schema for an ADR-007 pre-registration YAML (ADR-007 §Discipline).

    Required fields mirror the operational contract in
    ``docs/reference/evaluation.md`` §3: the homogeneous-vs-
    heterogeneous pair, the seed list, the episodes-per-seed budget,
    the estimator, and the bootstrap method. The default bootstrap is
    ``cluster`` per reviewer P1-9 — pooled IID bootstrap on
    seed-clustered data understates the CI width.

    Attributes:
        axis: One of the ADR-007 §3.4 axes
            (``"CR"`` / ``"AS"`` / ``"OM"`` / ``"CM"`` / ``"PF"`` /
            ``"SA"``).
        condition_pair: Homogeneous-vs-heterogeneous baseline pair.
        seeds: List of root seeds for ADR-002 P6 determinism.
        episodes_per_seed: Episode budget per ``(seed, condition)``
            cell.
        estimator: Identifier of the headline estimator (e.g.
            ``"iqm_success_rate"``).
        bootstrap_method: Default ``"cluster"`` (reviewer P1-9);
            ``"hierarchical"`` is an alias accepted for back-compat;
            ``"iid"`` is allowed only for power calculations, not
            leaderboard entries (reviewer P1-5, enforced by
            :meth:`_check_bootstrap_policy`).
        failure_policy: ``"strict"`` (the spike fails if any seed
            errors) or ``"best_effort"`` (errored seeds are dropped
            and reported).
        git_tag: Pre-registration git tag this YAML is locked to
            (ADR-007 §Discipline; verified by
            :func:`verify_git_tag`).
        notes: Free-form notes carried into the leaderboard entry.
        run_purpose: ``"leaderboard"`` (default), ``"power"``, or
            ``"debug"`` (reviewer P1-5). Only ``"leaderboard"`` runs
            are admitted to the public ranking; the other two values
            relax the iid-bootstrap ban so power calculations and
            local debug runs can still use a pooled iid baseline.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    axis: str
    condition_pair: ConditionPair
    seeds: list[int]
    episodes_per_seed: PositiveInt
    estimator: str
    bootstrap_method: BootstrapMethod = "cluster"
    failure_policy: FailurePolicy = "strict"
    git_tag: str
    notes: str = Field(default="")
    run_purpose: RunPurpose = "leaderboard"

    @model_validator(mode="after")
    def _check_bootstrap_policy(self) -> PreregistrationSpec:
        """Enforce the iid-not-allowed-for-leaderboard rule (reviewer P1-5).

        The pooled IID bootstrap on seed-clustered episode data
        understates CI width; admitting it to the leaderboard would
        let entries claim tighter intervals than the data supports.
        Power-analysis and debug runs are exempt because they are not
        published.
        """
        if self.run_purpose == "leaderboard" and self.bootstrap_method == "iid":
            msg = (
                "iid bootstrap is not permitted for leaderboard "
                "entries; use cluster (default) or hierarchical."
            )
            raise ValueError(msg)
        return self

    def normalised_axis(self) -> str:
        """Return the validated ADR-007 axis label (ADR-007 §Decision).

        Raises:
            ValueError: When :attr:`axis` is not one of the ADR-007
                §3.4 Option D shortlist.
        """
        if self.axis not in _AXIS_LABELS:
            msg = (
                f"axis {self.axis!r} is not one of the ADR-007 §3.4 shortlist "
                f"{sorted(_AXIS_LABELS)!r}"
            )
            raise ValueError(msg)
        return self.axis


class PreregistrationError(RuntimeError):
    """Raised when a pre-registration fails the ADR-007 §Discipline check (ADR-007 §Discipline)."""


def load_prereg(path: Path) -> PreregistrationSpec:
    """Load and validate a pre-registration YAML (ADR-007 §Discipline).

    Args:
        path: Absolute path to the YAML file.

    Returns:
        The validated :class:`PreregistrationSpec`.

    Raises:
        FileNotFoundError: When ``path`` does not exist.
        pydantic.ValidationError: When the YAML payload does not
            match the schema.
        ValueError: When :attr:`PreregistrationSpec.axis` is not on
            the ADR-007 §3.4 shortlist.
    """
    raw = path.read_text(encoding="utf-8")
    data = yaml.safe_load(raw) or {}
    spec = PreregistrationSpec.model_validate(data)
    spec.normalised_axis()
    return spec


def _git(*args: str, repo_path: Path) -> str:
    git = shutil.which("git")
    if git is None:
        msg = "git executable not found on PATH; cannot verify ADR-007 pre-registration"
        raise PreregistrationError(msg)
    try:
        result = subprocess.run(  # noqa: S603 — git binary resolved via shutil.which
            [git, *args],
            cwd=repo_path,
            capture_output=True,
            check=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        msg = f"git {' '.join(args)} failed: {exc.stderr.strip()}"
        raise PreregistrationError(msg) from exc
    return result.stdout.strip()


def _blob_sha_at_tag(*, tag: str, file_relpath: str, repo_path: Path) -> str:
    """Return the blob SHA of ``file_relpath`` as stored at ``tag``."""
    spec = f"{tag}:{file_relpath}"
    return _git("rev-parse", spec, repo_path=repo_path)


def _blob_sha_on_disk(*, path: Path, repo_path: Path) -> str:
    """Return the blob SHA of the file on disk via ``git hash-object``."""
    return _git("hash-object", "--", str(path), repo_path=repo_path)


def verify_git_tag(
    spec: PreregistrationSpec,
    prereg_path: Path,
    *,
    repo_path: Path,
) -> str:
    """Verify the pre-registration matches its git tag (ADR-007 §Discipline).

    Confirms (i) the tag exists in the repository and (ii) the blob
    SHA of ``prereg_path`` on disk matches the blob SHA of the same
    file as stored at the tag. The second check is the lock that
    catches the "edit-the-YAML-after-launch" anti-pattern: any change
    to the YAML's bytes after the tag was cut shifts the on-disk
    blob SHA but not the tagged blob SHA.

    Args:
        spec: The loaded :class:`PreregistrationSpec`.
        prereg_path: Path to the YAML file on disk (absolute or
            repo-relative; will be resolved against ``repo_path``).
        repo_path: Root of the git working tree.

    Returns:
        The verified blob SHA (hex digest, 40-char SHA-1).

    Raises:
        PreregistrationError: When the tag is missing, the file is
            outside the repo, or the blob SHAs disagree.
    """
    abs_path = prereg_path if prereg_path.is_absolute() else (repo_path / prereg_path)
    try:
        rel = abs_path.resolve().relative_to(repo_path.resolve())
    except ValueError as exc:
        msg = f"pre-registration path {abs_path} is outside repo {repo_path}"
        raise PreregistrationError(msg) from exc

    try:
        _git("rev-parse", "--verify", f"refs/tags/{spec.git_tag}", repo_path=repo_path)
    except PreregistrationError as exc:
        msg = f"git tag {spec.git_tag!r} does not exist in {repo_path}"
        raise PreregistrationError(msg) from exc

    on_disk = _blob_sha_on_disk(path=abs_path, repo_path=repo_path)
    at_tag = _blob_sha_at_tag(tag=spec.git_tag, file_relpath=str(rel), repo_path=repo_path)
    if on_disk != at_tag:
        msg = (
            f"pre-registration blob SHA mismatch: on-disk {on_disk} != tagged {at_tag} "
            f"(tag={spec.git_tag}, file={rel})"
        )
        raise PreregistrationError(msg)
    return on_disk


__all__ = [
    "BootstrapMethod",
    "FailurePolicy",
    "PreregistrationError",
    "PreregistrationSpec",
    "RunPurpose",
    "load_prereg",
    "verify_git_tag",
]
