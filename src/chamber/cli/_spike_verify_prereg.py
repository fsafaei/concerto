# SPDX-License-Identifier: Apache-2.0
"""``chamber-spike verify-prereg`` subcommand (T5b.1; ADR-007 §Discipline).

Verifies that a pre-registration YAML's on-disk blob SHA matches the
blob SHA stored at its committed git tag. The check is the launch-time
guard rail ADR-007 §Discipline requires for every Phase-0 axis spike:
editing the YAML between push and tag-cut shifts the on-disk blob SHA
while leaving the tagged blob SHA unchanged, and this subcommand
refuses to launch in that state.

Two modes (plan/07 §6 #5):

- ``--spike <path>`` — verify a single YAML. Output wording is
  byte-identical to the pre-PR shape (PR #94) so downstream grep on
  the FAIL prefix continues to work.
- ``--all`` — walk all six canonical axis YAMLs under
  ``spikes/preregistration/`` in :data:`CANONICAL_AXIS_ORDER` (AS → OM
  → CR → CM → PF → SA, ADR-007 §Implementation staging order),
  aggregate per-axis pass/fail without short-circuiting, and exit
  :data:`PREREG_MISMATCH_EXIT_CODE` if any axis failed. The user sees
  the full per-axis state in one pass; each line carries an
  ``axis=…`` token and a ``reason=…`` failure body so multi-axis
  output is uniformly greppable.

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
from typing import TYPE_CHECKING, Literal

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

#: Discriminator for the three failure modes :func:`_verify_one_axis`
#: can return. Single-axis mode reconstructs the pre-PR FAIL wording
#: from this; ``--all`` mode prepends ``axis=…`` and renders a uniform
#: ``reason=…`` body. Keeps both contracts intact without two parallel
#: code paths (ADR-007 §Discipline; plan/07 §6 #5).
_FailKind = Literal["missing_file", "load_error", "verify_error"]


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
    """Single-YAML mode: ``--spike <path>`` (T5b.1; ADR-007 §Discipline).

    FAIL wording is byte-identical to the pre-PR shape (PR #94) so
    downstream callers that grep on the message prefix continue to
    work. Multi-axis output (``--all``) carries an ``axis=…`` /
    ``reason=…`` shape instead — see :func:`_run_all`.
    """
    spike_path: Path = args.spike
    repo_root: Path = args.repo_root if args.repo_root is not None else _infer_repo_root(spike_path)
    outcome = _verify_one_axis(prereg_path=spike_path, repo_root=repo_root)
    if outcome.passed:
        print(_format_pass_line(outcome))
        return 0
    print(_format_single_axis_fail(outcome), file=sys.stderr)
    return PREREG_MISMATCH_EXIT_CODE


def _run_all(args: argparse.Namespace) -> int:
    """Six-axis mode: ``--all`` (T5b.1; plan/07 §6 #5).

    Walks :data:`CANONICAL_AXIS_ORDER` (AS → OM → CR → CM → PF → SA),
    verifies each ``<repo_root>/spikes/preregistration/{axis}.yaml`` in
    turn, prints one line per axis (PASS to stdout, FAIL to stderr),
    and exits :data:`PREREG_MISMATCH_EXIT_CODE` if any axis failed.

    Aggregates without short-circuiting per plan/07 §6 #5: a single
    tampered YAML must not hide the state of the other five from the
    user. Each line is uniformly greppable as
    ``verify-prereg: <PASS|FAIL> — axis=<AXIS> …``.
    """
    repo_root: Path = args.repo_root if args.repo_root is not None else _infer_repo_root(Path.cwd())
    prereg_dir = repo_root / _PREREG_SUBDIR
    outcomes: list[_AxisOutcome] = []
    for axis in CANONICAL_AXIS_ORDER:
        prereg_path = prereg_dir / f"{axis}.yaml"
        outcomes.append(_verify_one_axis(prereg_path=prereg_path, repo_root=repo_root, axis=axis))
    for outcome in outcomes:
        if outcome.passed:
            print(_format_pass_line(outcome))
        else:
            print(_format_all_axis_fail(outcome), file=sys.stderr)
    if all(o.passed for o in outcomes):
        return 0
    return PREREG_MISMATCH_EXIT_CODE


@dataclass(frozen=True)
class _AxisOutcome:
    """Per-axis verification outcome (ADR-007 §Discipline; plan/07 §6 #5).

    Carries every datum either caller (``--spike`` or ``--all``)
    might need to render its line, so the two callers can pick the
    appropriate format string without re-running the verification
    or losing precision on the failure mode.
    """

    prereg_path: Path
    axis: str  # resolved axis label, or "" if load failed before spec
    passed: bool
    spec: PreregistrationSpec | None = None
    blob_sha: str = ""
    fail_kind: _FailKind | None = None
    fail_detail: str = ""


def _verify_one_axis(
    *,
    prereg_path: Path,
    repo_root: Path,
    axis: str | None = None,
) -> _AxisOutcome:
    """Verify a single prereg YAML; never raise (ADR-007 §Discipline; plan/07 §6 #5).

    Folds the three failure modes (missing file / load error / git-tag
    or blob-SHA mismatch) into a uniform :class:`_AxisOutcome` so the
    ``--all`` aggregator can render the full per-axis picture in one
    pass without short-circuiting. The single-axis caller reconstructs
    the pre-PR FAIL wording from the same outcome via
    :func:`_format_single_axis_fail`.

    Args:
        prereg_path: Path to the YAML file on disk.
        repo_root: Root of the git working tree.
        axis: Canonical axis label, set by ``--all`` callers so the
            outcome carries an axis even when the file is missing
            (single-axis callers pass ``None`` and the axis is taken
            from the loaded :class:`PreregistrationSpec`, or left
            empty if the load fails earlier).
    """
    if not prereg_path.exists():
        return _AxisOutcome(
            prereg_path=prereg_path,
            axis=axis or "",
            passed=False,
            fail_kind="missing_file",
        )
    try:
        spec = load_prereg(prereg_path)
    except (FileNotFoundError, ValueError) as exc:
        return _AxisOutcome(
            prereg_path=prereg_path,
            axis=axis or "",
            passed=False,
            fail_kind="load_error",
            fail_detail=str(exc),
        )
    resolved_axis = axis or spec.axis
    try:
        blob_sha = verify_git_tag(spec, prereg_path, repo_path=repo_root)
    except PreregistrationError as exc:
        return _AxisOutcome(
            prereg_path=prereg_path,
            axis=resolved_axis,
            passed=False,
            spec=spec,
            fail_kind="verify_error",
            fail_detail=str(exc),
        )
    return _AxisOutcome(
        prereg_path=prereg_path,
        axis=resolved_axis,
        passed=True,
        spec=spec,
        blob_sha=blob_sha,
    )


def _format_pass_line(outcome: _AxisOutcome) -> str:
    """Format a PASS line; identical shape for ``--spike`` and ``--all`` (T5b.1)."""
    assert outcome.spec is not None  # noqa: S101 — narrow types for type-checker.
    return (
        f"verify-prereg: PASS — axis={outcome.spec.axis} "
        f"tag={outcome.spec.git_tag} blob_sha={outcome.blob_sha}"
    )


def _format_single_axis_fail(outcome: _AxisOutcome) -> str:
    """Single-axis FAIL line; byte-identical to the pre-PR shape (PR #94)."""
    if outcome.fail_kind == "missing_file":
        return f"verify-prereg: pre-registration YAML not found at {outcome.prereg_path}"
    if outcome.fail_kind == "load_error":
        return f"verify-prereg: failed to load {outcome.prereg_path}: {outcome.fail_detail}"
    return f"verify-prereg: FAIL — {outcome.fail_detail}"


def _format_all_axis_fail(outcome: _AxisOutcome) -> str:
    """``--all`` FAIL line; uniform ``axis=… reason=…`` shape (plan/07 §6 #5)."""
    axis_part = f"axis={outcome.axis} " if outcome.axis else ""
    if outcome.fail_kind == "missing_file":
        return (
            f"verify-prereg: FAIL — {axis_part}"
            f"reason=prereg YAML not found at {outcome.prereg_path}"
        )
    if outcome.fail_kind == "load_error":
        return (
            f"verify-prereg: FAIL — {axis_part}"
            f"reason=failed to load {outcome.prereg_path}: {outcome.fail_detail}"
        )
    return f"verify-prereg: FAIL — {axis_part}reason={outcome.fail_detail}"


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
