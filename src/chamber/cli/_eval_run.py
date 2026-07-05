# SPDX-License-Identifier: Apache-2.0
"""``chamber-eval run`` — produce a v3 result bundle (ADR-028 §Decision 3).

One command, one reviewer-facing artefact: resolve the task via the
ADR-027 registry, gate on the pre-registration tag when given (exit
``PREREG_MISMATCH_EXIT_CODE`` before any episode runs), refuse a dirty
working tree unless ``--allow-dirty`` (exit ``DIRTY_TREE_EXIT_CODE``;
an allowed-dirty bundle is stamped ``dirty: true`` and is ineligible
for leaderboard use), run the episode grid, and write the immutable
bundle directory ``chamber-eval verify`` admits from.
"""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

from pydantic import ValidationError

import chamber
import chamber.tasks
from chamber.benchmarks.bundle_runner import (
    EGO_SUBSTREAM_PATTERN,
    run_task_episodes,
    run_task_episodes_for_set,
)
from chamber.evaluation.bundles import (
    BOOTSTRAP_SUBSTREAM,
    DEFAULT_N_RESAMPLES,
    DIRTY_TREE_EXIT_CODE,
    compute_summary,
    git_provenance,
    platform_fingerprint,
    write_bundle_dir,
)
from chamber.evaluation.prereg import (
    PREREG_MISMATCH_EXIT_CODE,
    PreregDocument,
    PreregistrationError,
    PreregistrationSpec,
    load_prereg,
    load_prereg_document,
    verify_git_tag,
)
from chamber.evaluation.results import ResultBundle, SeedSchedule
from chamber.partners.sets import (
    PartnerMemberSpec,
    PartnerSetSpec,
    WithheldParametersError,
    get_partner_set,
    parse_set_slug,
    resolve_set_members,
)

_USAGE_EXIT_CODE = 2


class _CliExitError(Exception):
    """Carries an error message + exit code up to the single handler."""

    def __init__(self, message: str, code: int) -> None:
        super().__init__(message)
        self.code = code


def _parse_seeds(raw: str) -> list[int]:
    """``"5"`` → ``[0..4]``; ``"0,3,7"`` → ``[0, 3, 7]`` (ADR-028 §Decision 1 seed schedule)."""
    if "," in raw:
        return [int(tok) for tok in raw.split(",") if tok.strip()]
    count = int(raw)
    if count <= 0:
        msg = f"--seeds must be a positive count or a comma list, got {raw!r}"
        raise ValueError(msg)
    return list(range(count))


def _parse_task(raw: str) -> tuple[str, int | None]:
    """Split ``id[@vN]`` into ``(task_id, version | None)`` (ADR-027 §Versioning)."""
    task_id, sep, version = raw.partition("@v")
    return (task_id, int(version)) if sep else (task_id, None)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chamber-eval run",
        description=(
            "Run a CHAMBER-Bench task and write a v3 result bundle (ADR-028). "
            "The bundle directory is immutable evidence: bundle.json, per-seed "
            "episode JSONL, partners.json, REPRO.txt, SHA256SUMS.txt."
        ),
    )
    parser.add_argument("--task", required=True, help="Task id, optionally id@vN (ADR-027).")
    parser.add_argument("--policy", required=True, help="Ego policy id (e.g. 'random' = B-RND).")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--partner-set",
        help="Partner-set slug set_id[@vN] from the ADR-009 set registry (as amended).",
    )
    group.add_argument("--partner", help="Single partner registry name (ad-hoc set).")
    parser.add_argument(
        "--include-private",
        action="store_true",
        help=(
            "Also run the set's private members. Requires the withheld "
            "parameters to be derivable locally (the maintainer-held "
            "CHAMBER_PRIVATE_PARTNER_SEED; ADR-009 as amended — published "
            "hashes, withheld parameters)."
        ),
    )
    parser.add_argument("--seeds", required=True, help="Seed count N (0..N-1) or comma list.")
    parser.add_argument("--episodes", type=int, required=True, help="Episodes per seed.")
    parser.add_argument("--out", type=Path, required=True, help="Bundle directory to create.")
    parser.add_argument(
        "--prereg", type=Path, default=None, help="Pre-registration YAML to gate on."
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help=(
            "Produce a bundle from a dirty working tree anyway; the bundle is "
            "marked dirty: true and is ineligible for leaderboard use."
        ),
    )
    parser.add_argument(
        "--root-seed",
        type=int,
        default=0,
        help="Run-level root seed for ego + bootstrap substreams (ADR-002 P6).",
    )
    parser.add_argument(
        "--n-resamples",
        type=int,
        default=None,
        help="Bootstrap resamples for the bundle summary (default: 2000).",
    )
    return parser


def _prereg_gate(prereg_path: Path, *, repo_path: Path) -> tuple[str, str]:
    """Verify the prereg before any episode runs (ADR-007 §Discipline; ADR-028 §Decision 3).

    Tries the document-form first, falls back to the legacy axis-form
    (ADR-028 §Decision 2). Returns ``(git_tag, verified blob SHA)``.
    """
    try:
        spec: PreregDocument | PreregistrationSpec
        try:
            spec = load_prereg_document(prereg_path)
        except ValidationError:
            spec = load_prereg(prereg_path)
        blob = verify_git_tag(spec, prereg_path, repo_path=repo_path)
    except (PreregistrationError, ValidationError, OSError, ValueError) as exc:
        msg = f"pre-registration gate failed: {exc}"
        raise _CliExitError(msg, PREREG_MISMATCH_EXIT_CODE) from exc
    return spec.git_tag, blob


def _resolve_partner_set(
    args: argparse.Namespace, task_id: str
) -> tuple[PartnerSetSpec, list[tuple[PartnerMemberSpec, dict[str, str]]]]:
    """Resolve ``--partner-set`` to (set, runnable members) (ADR-009 as amended).

    Public members only unless ``--include-private``, which requires the
    withheld parameters to be derivable locally (the maintainer-held
    seed); every resolution is digest-verified (ADR-018 custody).
    """
    set_id, set_version = parse_set_slug(args.partner_set)
    try:
        set_spec = get_partner_set(set_id, version=set_version)
    except KeyError as exc:
        raise _CliExitError(str(exc), _USAGE_EXIT_CODE) from exc
    if set_spec.task_id != task_id:
        msg = (
            f"partner set {set_spec.slug} serves task {set_spec.task_id!r}, "
            f"not {task_id!r} (ADR-009 as amended: sets are per-task)"
        )
        raise _CliExitError(msg, _USAGE_EXIT_CODE)
    try:
        members = resolve_set_members(set_spec, include_private=args.include_private)
    except WithheldParametersError as exc:
        raise _CliExitError(str(exc), _USAGE_EXIT_CODE) from exc
    return set_spec, members


def _run_impl(args: argparse.Namespace, argv: list[str]) -> int:
    repo_path = Path.cwd()
    if args.include_private and args.partner_set is None:
        msg = "--include-private only applies to --partner-set runs"
        raise _CliExitError(msg, _USAGE_EXIT_CODE)
    try:
        seeds = _parse_seeds(args.seeds)
    except ValueError as exc:
        raise _CliExitError(str(exc), _USAGE_EXIT_CODE) from exc
    task_id, task_version = _parse_task(args.task)

    set_spec: PartnerSetSpec | None = None
    set_members: list[tuple[PartnerMemberSpec, dict[str, str]]] = []
    if args.partner_set is not None:
        set_spec, set_members = _resolve_partner_set(args, task_id)

    provenance = git_provenance(repo_path)
    if provenance.dirty and not args.allow_dirty:
        msg = (
            f"working tree at {repo_path} is dirty (or git state is "
            "unprovable); a result bundle must be traceable to a commit. "
            "Commit first, or pass --allow-dirty to produce a "
            "leaderboard-ineligible bundle marked dirty: true."
        )
        raise _CliExitError(msg, DIRTY_TREE_EXIT_CODE)

    prereg_tag: str | None = None
    prereg_blob: str | None = None
    if args.prereg is not None:
        prereg_tag, prereg_blob = _prereg_gate(args.prereg, repo_path=repo_path)

    try:
        if set_spec is not None:
            episodes_by_seed, partner_material, partner_hashes = run_task_episodes_for_set(
                task_id=task_id,
                task_version=task_version,
                policy_id=args.policy,
                set_spec=set_spec,
                members=set_members,
                seeds=seeds,
                episodes_per_seed=args.episodes,
                root_seed=args.root_seed,
            )
            partner_set_id = set_spec.slug
            # The schedule records the per-seed record count so verify's
            # episode arithmetic holds across the member grid (ADR-028).
            schedule_episodes = args.episodes * len(set_members)
        else:
            episodes_by_seed, partner_spec = run_task_episodes(
                task_id=task_id,
                task_version=task_version,
                policy_id=args.policy,
                partner_name=args.partner,
                seeds=seeds,
                episodes_per_seed=args.episodes,
                root_seed=args.root_seed,
            )
            partner_set_id = f"adhoc:{args.partner}"
            schedule_episodes = args.episodes
            partner_hashes = {args.partner: partner_spec.partner_id}
            partner_material = [
                {
                    "name": args.partner,
                    "class_name": partner_spec.class_name,
                    "seed": partner_spec.seed,
                    "checkpoint_step": partner_spec.checkpoint_step,
                    "weights_uri": partner_spec.weights_uri,
                    "extra": dict(partner_spec.extra),
                }
            ]
    except (KeyError, NotImplementedError, ValueError) as exc:
        raise _CliExitError(str(exc), _USAGE_EXIT_CODE) from exc

    spec_registered = chamber.tasks.get(task_id, version=task_version)
    all_episodes = [ep for records in episodes_by_seed.values() for ep in records]
    n_resamples = args.n_resamples if args.n_resamples is not None else DEFAULT_N_RESAMPLES
    repro_command = "chamber-eval run " + shlex.join(argv)
    bundle = ResultBundle(
        task_id=spec_registered.task_id,
        task_version=spec_registered.version,
        policy_id=args.policy,
        partner_set_id=partner_set_id,
        partner_hashes=partner_hashes,
        git_sha=provenance.sha,
        dirty=provenance.dirty,
        package_version=chamber.__version__,
        seed_schedule=SeedSchedule(
            root_seed=args.root_seed,
            seeds=seeds,
            episodes_per_seed=schedule_episodes,
            substream_labels=[EGO_SUBSTREAM_PATTERN, BOOTSTRAP_SUBSTREAM],
        ),
        repro_command=repro_command,
        platform=platform_fingerprint(),
        manifest={},
        summary=compute_summary(
            all_episodes, n_resamples=n_resamples, bootstrap_root_seed=args.root_seed
        ),
        prereg_git_tag=prereg_tag,
        prereg_blob_sha=prereg_blob,
    )
    try:
        final = write_bundle_dir(
            args.out,
            bundle_without_manifest=bundle,
            episodes_by_seed=episodes_by_seed,
            partner_specs=partner_material,
            repro_command=repro_command,
            prereg_source=args.prereg,
        )
    except FileExistsError as exc:
        raise _CliExitError(str(exc), _USAGE_EXIT_CODE) from exc

    dirty_note = "  [DIRTY — leaderboard-ineligible]" if final.dirty else ""
    print(
        f"chamber-eval run: wrote {final.task_id}@v{final.task_version} bundle to "
        f"{args.out} ({final.summary.n_episodes} episodes; success IQM "
        f"{final.summary.success_iqm:.3f} "
        f"[{final.summary.success_ci_low:.3f}, {final.summary.success_ci_high:.3f}])"
        f"{dirty_note}"
    )
    return 0


def run(argv: list[str]) -> int:
    """Entry point for ``chamber-eval run`` (ADR-028 §Decision 3).

    Returns ``0`` on success; ``2`` on usage errors / unknown ids /
    unsupported tasks; ``PREREG_MISMATCH_EXIT_CODE`` (4) when the
    prereg fails schema validation or the tag-blob check;
    ``DIRTY_TREE_EXIT_CODE`` (7) on a dirty tree without
    ``--allow-dirty``.
    """
    args = _build_parser().parse_args(argv)
    try:
        return _run_impl(args, argv)
    except _CliExitError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return exc.code


__all__ = ["run"]
