# SPDX-License-Identifier: Apache-2.0
"""``chamber-spike run`` subcommand (T5b.1; plan/07 §3 / §5).

Launches an ADR-007 axis spike — either a real run against the
chamber-side benchmark adapter (B8 / B9 / future Stage-2/3 PRs) or a
synthetic dry-run that emits a deterministic
:class:`~chamber.evaluation.results.SpikeRun` JSON the harness can
exercise without GPU.

Plan/07 §T5b.2 splits the per-axis adapters into separate modules
(``chamber.benchmarks.stage1_as`` / ``stage1_om`` / ...). This
dispatcher resolves the adapter lazily via ``importlib`` so:

- Stage-1 axes (AS / OM) without their adapter yet shipped surface a
  friendly ``NotImplementedError`` exit on the real path, but still
  work under ``--dry-run``.
- Stage-2/3 axes (CR / CM / PF / SA) are deferred for this session
  per the prompt §2 explicit out-of-scope list; the real path raises
  with a pointer to the relevant plan/07 §T5b.3/§T5b.4 stub.

The dry-run path generates a deterministic SpikeRun matching the
prereg's seed x episodes contract (5 seeds x 20 episodes per
(seed, condition) = 100 episodes per condition) per plan/07 §2. The
homo side always succeeds; the hetero side succeeds for exactly
``--dry-run-hetero-success-count`` episodes per seed (default 10
→ IQM gap = 50pp; pick 18+ to fall below the 20pp gate).
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from chamber.evaluation.results import (
    ConditionPair,
    EpisodeResult,
    SpikeRun,
)

if TYPE_CHECKING:
    import argparse
    from collections.abc import Callable

#: Exit code emitted when ``chamber-spike run`` is invoked on a real
#: (non-dry-run) axis whose adapter is not yet shipped. Distinct from
#: argparse's 2, the train trip-wire's 3, the verify-prereg mismatch's
#: 4, and the next-stage gate's 5 (B7).
ADAPTER_NOT_WIRED_EXIT_CODE: int = 6

#: ADR-007 §3.4 Option D axes + their Phase-0 stage assignment
#: (ADR-007 §Implementation staging). Tuple-of-tuples (rather than a
#: dict) so iteration order is canonical AS → OM → CR → CM → PF → SA.
_STAGE_BY_AXIS: dict[str, int] = {
    "AS": 1,
    "OM": 1,
    "CR": 2,
    "CM": 2,
    "PF": 3,
    "SA": 3,
}

#: Canonical adapter-module names per axis (plan/07 §T5b.2 naming).
#: Stage-1 entries match the B8 / B9 module-name contract; Stage-2/3
#: names are placeholders for the future plan/07 §T5b.3 / §T5b.4 work.
_ADAPTER_MODULE: dict[str, str] = {
    "AS": "chamber.benchmarks.stage1_as",
    "OM": "chamber.benchmarks.stage1_om",
    "CR": "chamber.benchmarks.stage2_cr",
    "CM": "chamber.benchmarks.stage2_cm",
    "PF": "chamber.benchmarks.stage3_pf",
    "SA": "chamber.benchmarks.stage3_sa",
}

#: Conventional adapter entry-point name. Per-axis module exports
#: ``def run_axis(args) -> SpikeRun`` (or compatible) that takes the
#: argparse Namespace and returns a SpikeRun ready for serialisation.
_ADAPTER_ENTRY_NAME: str = "run_axis"

#: Plan/07 §2 sample-size contract: 5 seeds x 20 episodes per (seed,
#: condition) = 100 episodes per condition. The dry-run mirrors the
#: prereg YAML's seed list + episodes_per_seed budget exactly.
_DRY_RUN_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
_DRY_RUN_EPISODES_PER_SEED: int = 20


def add_parser(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register the ``run`` subparser (T5b.1; plan/07 §3 / §5)."""
    parser = sub.add_parser(
        "run",
        help="Launch an ADR-007 axis spike (or emit a dry-run SpikeRun).",
        description=(
            "Real run dispatches to the per-axis adapter "
            "(chamber.benchmarks.stage{1,2,3}_{axis}.run_axis). When the "
            "adapter is not yet shipped, exits with code "
            f"{ADAPTER_NOT_WIRED_EXIT_CODE}. Dry-run mode emits a "
            "deterministic SpikeRun JSON the harness can exercise "
            "without GPU (plan/07 §5)."
        ),
    )
    parser.add_argument(
        "--axis",
        required=True,
        choices=sorted(_STAGE_BY_AXIS),
        help="ADR-007 §3.4 axis to spike.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write the SpikeRun JSON archive.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Emit a synthetic SpikeRun matching the prereg's seed x "
            "episodes contract; do NOT call the chamber-side adapter. "
            "Default success rates: homo 100%%, hetero "
            "(dry-run-hetero-success-count / episodes-per-seed) "
            "(default 10/20 = 50%%; IQM gap = 50pp; passes the 20pp gate)."
        ),
    )
    parser.add_argument(
        "--dry-run-hetero-success-count",
        type=int,
        default=10,
        help=(
            "Per-seed hetero success count in the dry-run SpikeRun. "
            "20 episodes per seed; homo always succeeds; hetero succeeds "
            "for this many (0-20). Default 10 (passes the 20pp gate); "
            "pick 18+ to fail the gate."
        ),
    )


def run(args: argparse.Namespace) -> int:
    """Implementation of ``chamber-spike run`` (T5b.1; plan/07 §3 / §5).

    Args:
        args: argparse namespace carrying ``axis``, ``output``,
            ``dry_run``, and ``dry_run_hetero_success_count``.

    Returns:
        ``0`` on success; :data:`ADAPTER_NOT_WIRED_EXIT_CODE` when the
        real adapter for ``args.axis`` is not yet shipped.
    """
    if args.dry_run:
        spike_run = _emit_dry_run_spike_run(
            axis=args.axis,
            hetero_success_count=args.dry_run_hetero_success_count,
        )
        _write_spike_run(spike_run, args.output)
        print(
            f"chamber-spike run: PASS (dry-run) axis={spike_run.axis} "
            f"output={args.output} episodes={len(spike_run.episode_results)}"
        )
        return 0

    adapter = _resolve_adapter(args.axis)
    if adapter is None:
        stage = _STAGE_BY_AXIS[args.axis]
        msg = (
            f"chamber-spike run --axis {args.axis}: Stage-{stage} adapter "
            f"({_ADAPTER_MODULE[args.axis]}) not yet shipped. Stage-1 axes "
            "(AS / OM) land in plan/07 §T5b.2 (B8 / B9); Stage-2/3 axes are "
            "deferred to plan/07 §T5b.3 / §T5b.4. Re-run with --dry-run "
            "to exercise the harness."
        )
        print(f"chamber-spike run: FAIL -- {msg}", file=sys.stderr)
        return ADAPTER_NOT_WIRED_EXIT_CODE

    spike_run = adapter(args)
    _write_spike_run(spike_run, args.output)
    print(
        f"chamber-spike run: PASS axis={spike_run.axis} output={args.output} "
        f"episodes={len(spike_run.episode_results)}"
    )
    return 0


def _resolve_adapter(
    axis: str,
) -> Callable[[argparse.Namespace], SpikeRun] | None:
    """Try to import the canonical per-axis adapter module (plan/07 §T5b.2).

    Returns ``None`` when the module does not exist (Stage-1 not yet
    shipped, or Stage-2/3 deferred). The CLI surfaces a friendly
    "not yet wired" exit on ``None`` rather than letting the
    ``ImportError`` bubble up.
    """
    module_name = _ADAPTER_MODULE[axis]
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return None
    entry = getattr(module, _ADAPTER_ENTRY_NAME, None)
    if not callable(entry):
        return None
    return entry  # type: ignore[no-any-return]


def _write_spike_run(spike_run: SpikeRun, path: Path) -> None:
    """Serialise a :class:`SpikeRun` to ``path`` via Pydantic's JSON encoder."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(spike_run.model_dump_json(indent=2), encoding="utf-8")


def _emit_dry_run_spike_run(
    *,
    axis: str,
    hetero_success_count: int,
) -> SpikeRun:
    """Build a deterministic SpikeRun for harness testing (plan/07 §5).

    Layout:

    - 5 seeds x 20 episodes per (seed, condition) = 100 episodes per
      condition; matches the plan/07 §2 sample-size contract.
    - Paired on ``(seed, episode_idx, initial_state_seed)`` per
      :class:`chamber.evaluation.bootstrap.PairedEpisode` so the
      :mod:`chamber.cli._spike_next_stage` gate sees a well-defined
      paired-bootstrap input.
    - ``metadata["condition"]`` is set to ``dry_run_homo`` or
      ``dry_run_hetero`` so
      :func:`chamber.cli.eval._build_paired_episodes` can split them.
    - Homo side: every episode succeeds.
    - Hetero side: the first ``hetero_success_count`` episodes per
      seed succeed; the rest fail.

    With the default 10/20 hetero success rate and homo at 100%, the
    per-pair delta is 1.0 for half the pairs and 0.0 for the other
    half — IQM (after dropping the bottom/top 25%) lands at 0.5,
    well above the 20pp gate. Pass ``hetero_success_count=18`` (or
    higher) to produce a SpikeRun that fails the gate.

    Args:
        axis: One of the ADR-007 §3.4 axes.
        hetero_success_count: Per-seed hetero success count, 0-20.

    Returns:
        A :class:`SpikeRun` ready to write to JSON.

    Raises:
        ValueError: When ``hetero_success_count`` is out of range.
    """
    if not (0 <= hetero_success_count <= _DRY_RUN_EPISODES_PER_SEED):
        msg = (
            f"--dry-run-hetero-success-count must be in [0, "
            f"{_DRY_RUN_EPISODES_PER_SEED}]; got {hetero_success_count}"
        )
        raise ValueError(msg)

    condition_pair = ConditionPair(
        homogeneous_id="dry_run_homo",
        heterogeneous_id="dry_run_hetero",
    )
    episode_results: list[EpisodeResult] = []
    for seed in _DRY_RUN_SEEDS:
        for episode_idx in range(_DRY_RUN_EPISODES_PER_SEED):
            initial_state_seed = seed * 10_000 + episode_idx
            # Homo: every episode succeeds.
            episode_results.append(
                EpisodeResult(
                    seed=seed,
                    episode_idx=episode_idx,
                    initial_state_seed=initial_state_seed,
                    success=True,
                    metadata={"condition": condition_pair.homogeneous_id},
                )
            )
            # Hetero: first hetero_success_count succeed, rest fail.
            episode_results.append(
                EpisodeResult(
                    seed=seed,
                    episode_idx=episode_idx,
                    initial_state_seed=initial_state_seed,
                    success=episode_idx < hetero_success_count,
                    metadata={"condition": condition_pair.heterogeneous_id},
                )
            )

    return SpikeRun(
        spike_id=f"dry_run_{axis.lower()}",
        prereg_sha="0" * 40,
        git_tag="dry-run",
        axis=axis,
        condition_pair=condition_pair,
        seeds=list(_DRY_RUN_SEEDS),
        episode_results=episode_results,
    )


__all__ = ["ADAPTER_NOT_WIRED_EXIT_CODE", "add_parser", "run"]
