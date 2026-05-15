# SPDX-License-Identifier: Apache-2.0
"""``chamber-spike next-stage`` subcommand (T5b.1; ADR-007 §Implementation staging).

The launch-time gate between Phase-0 stages. Reads one or more
SpikeRun JSON archives produced by ``chamber-spike run``, bootstraps
the paired-cluster gap CI per axis via
:func:`chamber.evaluation.bootstrap.pacluster_bootstrap`, and refuses
to clear the gate unless at least one axis from the prior stage
shows ``ci_low_pp >= 20.0`` (the ADR-007 §Validation criteria
threshold; ADR-008 §Decision; reviewer P1-9).

Stage groupings per ADR-007 §Implementation staging:

- Stage 1: AS + OM
- Stage 2: CR + CM
- Stage 3: PF + SA

Exit codes:

- ``0`` — at least one prior-stage axis meets the gate.
- :data:`NEXT_STAGE_GATE_EXIT_CODE` (5) — no prior-stage axis meets
  the gate; the next stage is forbidden to launch.

The subcommand reads SpikeRun JSONs rather than ``leaderboard.json``
because the LeaderboardEntry schema does not carry CI bounds (only
the HRS scalar + vector). The paired-cluster bootstrap that produces
``ci_low_pp`` runs on the per-episode SpikeRun data; recomputing it
here keeps the gate independent of any leaderboard caching.

Aggregator: the gate is enforced on the **IQM** of the resampled
per-pair deltas (the bootstrap's ``ci_low``), matching ADR-008
§Decision's rliable-style aggregate convention and
:func:`chamber.evaluation.bootstrap.pacluster_bootstrap`'s output. For
bimodal binary delta data — where the per-pair delta is 0/1 rather
than a smooth fraction — IQM can pin to 0 even when the mean is
non-zero because the middle 50% of sorted resampled values collapse
to the majority value. Future work (ADR-008 amendment) may add a
``--gate-aggregator {iqm,mean}`` flag if real-spike borderline data
makes the choice load-bearing; ``BootstrapCI`` already carries both
the IQM and the mean for downstream consumers.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

from chamber.evaluation.bootstrap import build_paired_episodes, pacluster_bootstrap
from chamber.evaluation.results import SpikeRun
from concerto.training.seeding import derive_substream

if TYPE_CHECKING:
    import argparse

#: Exit code emitted when the prior-stage gate fails. Distinct from
#: argparse's 2, the train trip-wire's 3, the verify-prereg mismatch's
#: 4, and the run-adapter-not-wired's 6.
NEXT_STAGE_GATE_EXIT_CODE: int = 5

#: ADR-007 §Implementation staging axis-to-stage map.
_AXES_BY_STAGE: dict[int, frozenset[str]] = {
    1: frozenset({"AS", "OM"}),
    2: frozenset({"CR", "CM"}),
    3: frozenset({"PF", "SA"}),
}

#: ADR-007 §Validation criteria default gate threshold (percentage
#: points on the paired-cluster bootstrap CI lower bound of the
#: homo - hetero gap). Tuned in plan/07 §2.
_DEFAULT_GATE_PP: float = 20.0

#: Default number of bootstrap resamples for ``pacluster_bootstrap``.
#: Matches :mod:`chamber.cli.eval`'s default — large enough for tight
#: 95% bounds on the per-axis gap, small enough to keep the CLI
#: snappy.
_DEFAULT_N_RESAMPLES: int = 2000

#: Default deterministic root seed for the bootstrap RNG (ADR-002 P6).
_DEFAULT_SEED: int = 0


def add_parser(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register the ``next-stage`` subparser (T5b.1; ADR-007 §Implementation staging)."""
    parser = sub.add_parser(
        "next-stage",
        help="Gate the next ADR-007 stage on the prior stage's >=20pp CI lower bound.",
        description=(
            "Reads SpikeRun JSON archives, bootstraps the paired-cluster gap CI "
            "per axis, and exits 0 only when at least one axis from --prior-stage "
            f"meets ci_low_pp >= --gate-pp (default {_DEFAULT_GATE_PP}). "
            f"Exits with code {NEXT_STAGE_GATE_EXIT_CODE} on gate failure."
        ),
    )
    parser.add_argument(
        "--prior-stage",
        type=int,
        choices=[1, 2, 3],
        required=True,
        help="Stage whose axes are gating the next stage (1, 2, or 3).",
    )
    parser.add_argument(
        "--spike-runs",
        type=Path,
        nargs="+",
        required=True,
        help=(
            "One or more SpikeRun JSON archives (produced by `chamber-spike run` "
            "or by the chamber-side adapter). At least one of these must belong "
            "to --prior-stage to be considered for the gate."
        ),
    )
    parser.add_argument(
        "--gate-pp",
        type=float,
        default=_DEFAULT_GATE_PP,
        help=(
            f"Gate threshold in percentage points (default {_DEFAULT_GATE_PP}; "
            "the ADR-007 §Validation criteria value)."
        ),
    )
    parser.add_argument(
        "--n-resamples",
        type=int,
        default=_DEFAULT_N_RESAMPLES,
        help=f"Bootstrap resamples (default {_DEFAULT_N_RESAMPLES}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=_DEFAULT_SEED,
        help=(f"Bootstrap root seed (ADR-002 P6 determinism; default {_DEFAULT_SEED})."),
    )


def run(args: argparse.Namespace) -> int:
    """Implementation of ``chamber-spike next-stage`` (T5b.1; ADR-007).

    Returns:
        ``0`` when at least one prior-stage axis meets the gate;
        :data:`NEXT_STAGE_GATE_EXIT_CODE` otherwise.
    """
    expected_axes = _AXES_BY_STAGE[args.prior_stage]
    runs = _load_spike_runs(args.spike_runs)
    if runs is None:
        return NEXT_STAGE_GATE_EXIT_CODE

    relevant = [r for r in runs if r.axis in expected_axes]
    ignored = [r for r in runs if r.axis not in expected_axes]
    if ignored:
        ignored_axes = sorted({r.axis for r in ignored})
        print(
            f"chamber-spike next-stage: note -- ignoring {len(ignored)} "
            f"SpikeRun(s) not in Stage-{args.prior_stage} (axes "
            f"{ignored_axes}); these are out of scope for this gate.",
            file=sys.stderr,
        )
    if not relevant:
        print(
            f"chamber-spike next-stage: FAIL -- no SpikeRun on the prior stage "
            f"({sorted(expected_axes)}) found among the {len(runs)} archive(s) "
            f"passed via --spike-runs. The gate requires at least one "
            f"Stage-{args.prior_stage} run.",
            file=sys.stderr,
        )
        return NEXT_STAGE_GATE_EXIT_CODE

    rng = derive_substream("chamber.cli.next_stage.bootstrap", root_seed=args.seed).default_rng()
    results: list[tuple[str, float, float]] = []
    for run_ in relevant:
        pairs = build_paired_episodes(run_)
        if not pairs:
            print(
                f"chamber-spike next-stage: skipping {run_.spike_id!r} (axis "
                f"{run_.axis}): no paired episodes after pairing on "
                "(seed, episode_idx, initial_state_seed).",
                file=sys.stderr,
            )
            continue
        ci = pacluster_bootstrap(pairs, n_resamples=args.n_resamples, rng=rng)
        ci_low_pp = ci.ci_low * 100.0
        ci_high_pp = ci.ci_high * 100.0
        results.append((run_.axis, ci_low_pp, ci_high_pp))

    if not results:
        print(
            "chamber-spike next-stage: FAIL -- no SpikeRun on the prior stage "
            "produced a non-empty paired-bootstrap input. Check the "
            "(seed, episode_idx, initial_state_seed) pairing in each archive.",
            file=sys.stderr,
        )
        return NEXT_STAGE_GATE_EXIT_CODE

    passing = [
        (axis, ci_low_pp, ci_high_pp)
        for axis, ci_low_pp, ci_high_pp in results
        if ci_low_pp >= args.gate_pp
    ]

    _print_summary(
        prior_stage=args.prior_stage,
        gate_pp=args.gate_pp,
        results=results,
        passing=passing,
    )

    if not passing:
        return NEXT_STAGE_GATE_EXIT_CODE
    return 0


def _load_spike_runs(paths: list[Path]) -> list[SpikeRun] | None:
    """Load every SpikeRun archive, returning ``None`` on any I/O / schema error."""
    runs: list[SpikeRun] = []
    for path in paths:
        if not path.exists():
            print(
                f"chamber-spike next-stage: FAIL -- SpikeRun archive not found: {path}",
                file=sys.stderr,
            )
            return None
        try:
            runs.append(SpikeRun.model_validate_json(path.read_text(encoding="utf-8")))
        except (ValueError, OSError) as exc:
            print(
                f"chamber-spike next-stage: FAIL -- could not load SpikeRun {path}: {exc}",
                file=sys.stderr,
            )
            return None
    return runs


def _print_summary(
    *,
    prior_stage: int,
    gate_pp: float,
    results: list[tuple[str, float, float]],
    passing: list[tuple[str, float, float]],
) -> None:
    """Print per-axis CI lines + a final PASS/FAIL verdict."""
    verdict = "PASS" if passing else "FAIL"
    out = sys.stdout if passing else sys.stderr
    print(
        f"chamber-spike next-stage: {verdict} -- Stage-{prior_stage} "
        f"gate (gate_pp={gate_pp:.1f}); "
        f"{len(passing)}/{len(results)} axis(es) pass.",
        file=out,
    )
    for axis, ci_low_pp, ci_high_pp in results:
        marker = "PASS" if ci_low_pp >= gate_pp else "fail"
        print(
            f"  axis={axis}  ci_low_pp={ci_low_pp:.2f}  ci_high_pp={ci_high_pp:.2f}  ({marker})",
            file=out,
        )


__all__ = ["NEXT_STAGE_GATE_EXIT_CODE", "add_parser", "run"]
