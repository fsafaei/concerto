# SPDX-License-Identifier: Apache-2.0
"""``chamber-spike train`` subcommand (M4b-9b; ADR-002 §Decisions; plan/05 §3.5).

Runs the ego-AHT training loop end-to-end on a Hydra YAML config and
prints a one-line summary. With ``--check-guarantee`` evaluates the
ADR-002 §Risks #1 trip-wire (the empirical-guarantee slope test) and
exits with :data:`_TRIP_WIRE_EXIT_CODE` on ``passed=False``.

Re-extracted from the previous all-in-one ``chamber.cli.spike`` module
when B6 (T5b.1) added :mod:`chamber.cli._spike_verify_prereg` and
:mod:`chamber.cli._spike_list`. Each subcommand now lives in its own
module so the top-level dispatcher stays thin (plan/07 §T5b.1).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from chamber.benchmarks.training_runner import run_training
from concerto.training.config import load_config
from concerto.training.empirical_guarantee import assert_positive_learning_slope

if TYPE_CHECKING:
    import argparse
    from collections.abc import Sequence

#: Exit code emitted when ``--check-guarantee`` fires the trip-wire
#: (ADR-002 §Risks #1). Distinct from argparse's exit-2 ("bad usage")
#: so reproduction scripts can grep the failure mode unambiguously.
_TRIP_WIRE_EXIT_CODE: int = 3

#: Number of trailing episodes the summary line averages over (M4b-9b).
_SUMMARY_WINDOW: int = 10


def add_parser(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register the ``train`` subparser (M4b-9b; ADR-002 §Decisions).

    Args:
        sub: The :class:`argparse._SubParsersAction` from the top-level
            ``chamber-spike`` parser.
    """
    train = sub.add_parser(
        "train",
        help="Run the ego-AHT training loop on a Hydra config.",
        description=(
            "Loads a Hydra config and runs concerto.training.ego_aht.train via the "
            "chamber-side training_runner. See M4b-9b / plan/05 §3.5."
        ),
    )
    train.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a Hydra YAML (e.g. configs/training/ego_aht_happo/*.yaml).",
    )
    train.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Hydra-style override, repeatable. e.g. --override total_frames=1000",
    )
    train.add_argument(
        "--check-guarantee",
        action="store_true",
        help=(
            "After training, run assert_positive_learning_slope on the curve and exit "
            f"with code {_TRIP_WIRE_EXIT_CODE} if the trip-wire fires "
            "(ADR-002 §Risks #1; issue #62)."
        ),
    )


def run(args: argparse.Namespace) -> int:
    """Implementation of ``chamber-spike train`` (M4b-9b; ADR-002 §Decisions).

    Args:
        args: argparse namespace carrying ``config``, ``override``, and
            ``check_guarantee`` from the top-level dispatcher.

    Returns:
        ``0`` on success; :data:`_TRIP_WIRE_EXIT_CODE` when
        ``--check-guarantee`` fires.
    """
    return _train_command(
        config_path=args.config,
        overrides=_parse_overrides(args.override),
        check_guarantee=args.check_guarantee,
    )


def _parse_overrides(raw: Sequence[str] | None) -> list[str]:
    """Pass-through Hydra-style ``key=value`` overrides (M4b-9b; ADR-002 §Decisions).

    ``argparse`` collects repeated ``--override`` values into a list.
    The CLI forwards the list as-is to
    :func:`concerto.training.config.load_config`, which threads it
    through to ``hydra.compose(overrides=...)``. Validation (the
    ``key=value`` shape) is Hydra's responsibility.
    """
    if raw is None:
        return []
    return list(raw)


def _train_command(
    *,
    config_path: Path,
    overrides: list[str],
    check_guarantee: bool,
) -> int:
    """Run training + optional guarantee check (M4b-9b; ADR-002 §Decisions)."""
    cfg = load_config(config_path=config_path, overrides=overrides)
    curve = run_training(cfg)
    rewards = list(curve.per_episode_ego_rewards)
    tail = rewards[-_SUMMARY_WINDOW:] if rewards else []
    mean_tail = sum(tail) / len(tail) if tail else 0.0
    print(
        f"run_id={curve.run_id} n_episodes={len(rewards)} "
        f"n_checkpoints={len(curve.checkpoint_paths)} "
        f"mean_reward_last_{len(tail)}={mean_tail:.4f}"
    )
    if check_guarantee:
        report = assert_positive_learning_slope(curve)
        print(
            f"empirical_guarantee slope={report.slope:.6f} "
            f"p_value={report.p_value:.2e} alpha={report.alpha} "
            f"n_episodes={report.n_episodes} passed={report.passed}"
        )
        if not report.passed:
            return _TRIP_WIRE_EXIT_CODE
    return 0


__all__ = ["add_parser", "run"]
