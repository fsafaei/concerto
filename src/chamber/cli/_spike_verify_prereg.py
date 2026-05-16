# SPDX-License-Identifier: Apache-2.0
"""``chamber-spike verify-prereg`` subcommand (T5b.1; ADR-007 §Discipline).

Verifies that a pre-registration YAML's on-disk blob SHA matches the
blob SHA stored at its committed git tag. The check is the launch-time
guard rail ADR-007 §Discipline requires for every Phase-0 axis spike:
editing the YAML between push and tag-cut shifts the on-disk blob SHA
while leaving the tagged blob SHA unchanged, and this subcommand
refuses to launch in that state.

Two modes (plan/07 §6 #5):

- ``--spike <path>`` — verify a single YAML.
- ``--all`` — walk all six canonical axis YAMLs under
  ``spikes/preregistration/`` in :data:`CANONICAL_AXIS_ORDER` (AS → OM
  → CR → CM → PF → SA, ADR-007 §Implementation staging order),
  aggregate per-axis pass/fail without short-circuiting, and exit
  :data:`PREREG_MISMATCH_EXIT_CODE` if any axis failed. The user sees
  the full per-axis state in one pass.

Delegates the actual SHA comparison to
:func:`chamber.evaluation.prereg.verify_git_tag` (PR #94 ship;
reviewer P1-9 hardened). On success the verified blob SHA is printed
to stdout; on :class:`PreregistrationError` the message lands on
stderr and the process exits with :data:`PREREG_MISMATCH_EXIT_CODE` —
distinct from argparse's exit-2 ("bad usage") and the training
trip-wire's exit-3 so reproduction scripts can grep the failure mode
unambiguously.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from chamber.evaluation.prereg import (
    CANONICAL_AXIS_ORDER,
    PREREG_MISMATCH_EXIT_CODE,
    PreregistrationError,
    PreregistrationSpec,
    load_prereg,
    verify_git_tag,
)

if TYPE_CHECKING:
    import argparse


#: Directory inside the repo that holds the six canonical axis YAMLs
#: (plan/07 §6 #5). The ``--all`` walker resolves
#: ``<repo_root> / _PREREG_SUBDIR / {axis}.yaml`` per axis.
_PREREG_SUBDIR: Path = Path("spikes") / "preregistration"


def add_parser(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register the ``verify-prereg`` subparser (T5b.1; ADR-007 §Discipline)."""
    parser = sub.add_parser(
        "verify-prereg",
        help="Verify a prereg YAML against its committed git tag (ADR-007 §Discipline).",
        description=(
            "Verify pre-registration YAML(s) against their committed git tags. "
            "With --spike <path>, checks one YAML; with --all, walks the six "
            "canonical axis YAMLs under spikes/preregistration/ in "
            "AS→OM→CR→CM→PF→SA order and aggregates per-axis results without "
            "short-circuiting. Exits with code "
            f"{PREREG_MISMATCH_EXIT_CODE} on any blob-SHA mismatch or missing tag."
        ),
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--spike",
        type=Path,
        default=None,
        help="Path to a single pre-registration YAML (e.g. spikes/preregistration/AS.yaml).",
    )
    mode.add_argument(
        "--all",
        dest="check_all",
        action="store_true",
        help=(
            "Verify all six canonical axis YAMLs (AS, OM, CR, CM, PF, SA) under "
            "spikes/preregistration/ in their ADR-007 §Implementation staging "
            "order. Aggregates per-axis pass/fail without short-circuiting "
            "(plan/07 §6 #5)."
        ),
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help=(
            "Root of the git working tree (default: with --spike, the directory "
            "containing the YAML's parent; with --all, the first ancestor of the "
            "current working directory that contains a .git entry, falling back "
            "to the current working directory)."
        ),
    )


def run(args: argparse.Namespace) -> int:
    """Implementation of ``chamber-spike verify-prereg`` (T5b.1; ADR-007 §Discipline).

    Args:
        args: argparse namespace. Either ``args.spike`` (single-YAML mode)
            or ``args.check_all`` is set, never both (enforced by the
            argparse mutually-exclusive group; passing neither raises an
            argparse exit-2). ``args.repo_root`` is optional and resolved
            from ``--spike``'s parent or the current working directory
            when unset.

    Returns:
        ``0`` when every checked YAML's on-disk blob SHA matches its
        tagged blob SHA; :data:`PREREG_MISMATCH_EXIT_CODE` on any
        failure (missing tag, missing file, load error, or SHA
        mismatch). Under ``--all`` the return code is the OR over the
        six per-axis outcomes.
    """
    if args.check_all:
        return _run_all(args)
    return _run_one(args)


def _run_one(args: argparse.Namespace) -> int:
    """Single-YAML mode: ``--spike <path>`` (T5b.1; ADR-007 §Discipline)."""
    spike_path: Path = args.spike
    if not spike_path.exists():
        print(
            f"verify-prereg: pre-registration YAML not found at {spike_path}",
            file=sys.stderr,
        )
        return PREREG_MISMATCH_EXIT_CODE

    repo_root: Path = args.repo_root if args.repo_root is not None else _infer_repo_root(spike_path)

    result = _verify_one_axis(prereg_path=spike_path, repo_root=repo_root)
    if result.passed:
        print(result.message)
        return 0
    print(result.message, file=sys.stderr)
    return PREREG_MISMATCH_EXIT_CODE


def _run_all(args: argparse.Namespace) -> int:
    """Six-axis mode: ``--all`` (T5b.1; plan/07 §6 #5).

    Walks :data:`CANONICAL_AXIS_ORDER` (AS → OM → CR → CM → PF → SA),
    verifies each ``<repo_root>/spikes/preregistration/{axis}.yaml`` in
    turn, prints one line per axis (PASS to stdout, FAIL to stderr),
    and exits :data:`PREREG_MISMATCH_EXIT_CODE` if any axis failed.

    Aggregates without short-circuiting per plan/07 §6 #5: a single
    tampered YAML must not hide the state of the other five from the
    user.
    """
    repo_root: Path = args.repo_root if args.repo_root is not None else _infer_repo_root(Path.cwd())
    prereg_dir = repo_root / _PREREG_SUBDIR
    results: list[_AxisResult] = []
    for axis in CANONICAL_AXIS_ORDER:
        prereg_path = prereg_dir / f"{axis}.yaml"
        results.append(_verify_one_axis(prereg_path=prereg_path, repo_root=repo_root, axis=axis))
    for result in results:
        stream = sys.stdout if result.passed else sys.stderr
        print(result.message, file=stream)
    if all(r.passed for r in results):
        return 0
    return PREREG_MISMATCH_EXIT_CODE


@dataclass(frozen=True)
class _AxisResult:
    """Per-axis verification outcome (ADR-007 §Discipline)."""

    axis: str
    passed: bool
    message: str


def _verify_one_axis(
    *,
    prereg_path: Path,
    repo_root: Path,
    axis: str | None = None,
) -> _AxisResult:
    """Verify a single prereg YAML; never raise (ADR-007 §Discipline; plan/07 §6 #5).

    Folds the four failure modes (missing file / load error / git-tag
    missing / blob-SHA mismatch) into a uniform :class:`_AxisResult`
    so the ``--all`` aggregator can render the full per-axis picture
    in one pass without short-circuiting.

    Args:
        prereg_path: Path to the YAML file on disk.
        repo_root: Root of the git working tree.
        axis: Optional canonical axis label (used when called from
            ``--all`` so the message includes ``axis=…`` even when the
            file is missing and the spec cannot be loaded). When
            ``None`` (single-YAML mode) the axis is taken from the
            successfully-loaded :class:`PreregistrationSpec`, or
            omitted if the load fails before that point.
    """
    label = _axis_label(axis)
    if not prereg_path.exists():
        return _AxisResult(
            axis=axis or "",
            passed=False,
            message=f"verify-prereg: FAIL — {label}reason=prereg YAML not found at {prereg_path}",
        )
    try:
        spec = load_prereg(prereg_path)
    except (FileNotFoundError, ValueError) as exc:
        return _AxisResult(
            axis=axis or "",
            passed=False,
            message=f"verify-prereg: FAIL — {label}reason=failed to load {prereg_path}: {exc}",
        )
    label = _axis_label(axis or spec.axis)
    try:
        blob_sha = verify_git_tag(spec, prereg_path, repo_path=repo_root)
    except PreregistrationError as exc:
        return _AxisResult(
            axis=axis or spec.axis,
            passed=False,
            message=f"verify-prereg: FAIL — {label}reason={exc}",
        )
    return _AxisResult(
        axis=axis or spec.axis,
        passed=True,
        message=_format_pass_line(spec=spec, blob_sha=blob_sha),
    )


def _axis_label(axis: str | None) -> str:
    """Render the ``axis=…`` token used in PASS / FAIL message bodies."""
    return f"axis={axis} " if axis else ""


def _format_pass_line(*, spec: PreregistrationSpec, blob_sha: str) -> str:
    """Format the PASS line; matches the existing single-YAML output (T5b.1)."""
    return f"verify-prereg: PASS — axis={spec.axis} tag={spec.git_tag} blob_sha={blob_sha}"


def _infer_repo_root(start: Path) -> Path:
    """Resolve a sensible default repo root by walking up from ``start``.

    Walks up from ``start.resolve().parent`` (when ``start`` is a file)
    or ``start.resolve()`` (when it is a directory) until a ``.git``
    entry is found (a directory for normal repos, a gitlink file for
    submodules) and forwards the result to :func:`verify_git_tag`,
    which delegates to ``git rev-parse`` — both forms are resolved by
    git itself. Falls back to :func:`Path.cwd` when no enclosing repo
    is detected, so the subcommand stays usable from any working
    directory.
    """
    resolved = start.resolve()
    candidate = resolved.parent if resolved.is_file() else resolved
    while candidate != candidate.parent:
        if (candidate / ".git").exists():
            return candidate
        candidate = candidate.parent
    return Path.cwd()


__all__ = ["add_parser", "run"]
