# SPDX-License-Identifier: Apache-2.0
"""``chamber-spike verify-prereg`` subcommand (T5b.1; ADR-007 §Discipline).

Verifies that a pre-registration YAML's on-disk blob SHA matches the
blob SHA stored at its committed git tag. The check is the launch-time
guard rail ADR-007 §Discipline requires for every Phase-0 axis spike:
editing the YAML between push and tag-cut shifts the on-disk blob SHA
while leaving the tagged blob SHA unchanged, and this subcommand
refuses to launch in that state.

Delegates to :func:`chamber.evaluation.prereg.verify_git_tag` (PR #94
ship; reviewer P1-9 hardened). On success the verified blob SHA is
printed to stdout; on :class:`PreregistrationError` the message lands
on stderr and the process exits with :data:`PREREG_MISMATCH_EXIT_CODE`
— distinct from argparse's exit-2 ("bad usage") and the training
trip-wire's exit-3 so reproduction scripts can grep the failure mode
unambiguously.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

from chamber.evaluation.prereg import (
    PREREG_MISMATCH_EXIT_CODE,
    PreregistrationError,
    load_prereg,
    verify_git_tag,
)

if TYPE_CHECKING:
    import argparse


def add_parser(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register the ``verify-prereg`` subparser (T5b.1; ADR-007 §Discipline)."""
    parser = sub.add_parser(
        "verify-prereg",
        help="Verify a prereg YAML against its committed git tag (ADR-007 §Discipline).",
        description=(
            "Loads the pre-registration YAML at --spike, then checks that its "
            "on-disk blob SHA matches the blob SHA stored at the YAML's "
            "``git_tag`` field. Exits with code "
            f"{PREREG_MISMATCH_EXIT_CODE} on mismatch or missing tag."
        ),
    )
    parser.add_argument(
        "--spike",
        type=Path,
        required=True,
        help="Path to a pre-registration YAML (e.g. spikes/preregistration/AS.yaml).",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help=(
            "Root of the git working tree (default: directory containing the "
            "--spike file's parent; falls back to the current working directory)."
        ),
    )


def run(args: argparse.Namespace) -> int:
    """Implementation of ``chamber-spike verify-prereg`` (T5b.1; ADR-007 §Discipline).

    Args:
        args: argparse namespace carrying ``spike`` (the YAML path) and
            ``repo_root`` (optional override; resolved from the YAML
            location when unset).

    Returns:
        ``0`` when the on-disk YAML's blob SHA matches the tagged blob
        SHA; :data:`PREREG_MISMATCH_EXIT_CODE` on
        :class:`PreregistrationError` (missing tag, file outside repo,
        or SHA mismatch).
    """
    spike_path: Path = args.spike
    if not spike_path.exists():
        print(
            f"verify-prereg: pre-registration YAML not found at {spike_path}",
            file=sys.stderr,
        )
        return PREREG_MISMATCH_EXIT_CODE

    repo_root: Path = args.repo_root if args.repo_root is not None else _infer_repo_root(spike_path)

    try:
        spec = load_prereg(spike_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"verify-prereg: failed to load {spike_path}: {exc}", file=sys.stderr)
        return PREREG_MISMATCH_EXIT_CODE

    try:
        blob_sha = verify_git_tag(spec, spike_path, repo_path=repo_root)
    except PreregistrationError as exc:
        print(f"verify-prereg: FAIL — {exc}", file=sys.stderr)
        return PREREG_MISMATCH_EXIT_CODE

    print(f"verify-prereg: PASS — axis={spec.axis} tag={spec.git_tag} blob_sha={blob_sha}")
    return 0


def _infer_repo_root(spike_path: Path) -> Path:
    """Resolve a sensible default repo root from the YAML's location.

    Walks up from ``spike_path``'s parent until a ``.git`` entry is
    found (a directory for normal repos, a gitlink file for submodules)
    and forwards the result to :func:`verify_git_tag`, which delegates
    to ``git rev-parse`` — both forms are resolved by git itself. Falls
    back to ``Path.cwd()`` when no enclosing repo is detected, so the
    subcommand stays usable from any working directory.
    """
    candidate = spike_path.resolve().parent
    while candidate != candidate.parent:
        if (candidate / ".git").exists():
            return candidate
        candidate = candidate.parent
    return Path.cwd()


__all__ = ["add_parser", "run"]
