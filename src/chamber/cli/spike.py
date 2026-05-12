# SPDX-License-Identifier: Apache-2.0
"""``chamber-spike`` console entry point (M4b-9b; ADR-002 §Decisions).

Two sub-commands in Phase 0:

- ``chamber-spike train`` — runs the ego-AHT training loop end-to-end
  on a Hydra YAML config (M4b-9b; plan/05 §3.5). The reproduction
  scripts (``scripts/repro/zoo_seed.sh``,
  ``scripts/repro/empirical_guarantee.sh``) invoke this rather than
  hand-rolling the ``run_training`` call site, so the CLI is the
  project's canonical training entry point.
- ``chamber-spike verify-prereg`` — placeholder for the M5
  pre-registration enforcement (plan/01 P4). Will verify the prereg
  YAML SHA matches its git tag before a spike launches. Phase-0 stub
  only.

The ``train`` sub-command:

- Loads ``--config`` via :func:`concerto.training.config.load_config`,
  which composes the Hydra config + validates against
  :class:`~concerto.training.config.EgoAHTConfig`.
- Accepts repeatable ``--override key=value`` flags (e.g.
  ``--override total_frames=1000 --override happo.rollout_length=250``)
  threaded through to Hydra.
- Calls :func:`chamber.benchmarks.training_runner.run_training` and
  prints a one-line summary on stdout (run_id, n_episodes,
  n_checkpoints, mean-of-last-10-episodes reward).
- With ``--check-guarantee`` evaluates
  :func:`concerto.training.empirical_guarantee.assert_positive_learning_slope`
  on the resulting curve and exits non-zero on ``passed=False``. This
  is the canonical empirical-guarantee gate after issue #62 / PR #72
  replaced the legacy moving-window-of-K statistic with the slope
  test.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import chamber
from chamber.benchmarks.training_runner import run_training
from concerto.training.config import load_config
from concerto.training.empirical_guarantee import assert_positive_learning_slope

if TYPE_CHECKING:
    from collections.abc import Sequence

#: Exit code emitted when ``--check-guarantee`` fires the trip-wire
#: (ADR-002 §Risks #1). Picked distinct from argparse's exit-2
#: ("bad usage") so a reproduction script can grep the failure mode
#: unambiguously.
_TRIP_WIRE_EXIT_CODE: int = 3

#: Number of trailing episodes the summary line averages over (M4b-9b).
_SUMMARY_WINDOW: int = 10


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


def _build_parser() -> argparse.ArgumentParser:
    """Build the top-level ``chamber-spike`` argparse surface (M4b-9b; ADR-002 §Decisions)."""
    parser = argparse.ArgumentParser(
        prog="chamber-spike",
        description=(
            f"CHAMBER {chamber.__version__} — spike CLI. Sub-commands: train, verify-prereg."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=False)

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

    sub.add_parser(
        "verify-prereg",
        help="Verify a prereg YAML SHA against its git tag (Phase-0 stub; M5).",
    )

    return parser


def _train_command(
    *,
    config_path: Path,
    overrides: list[str],
    check_guarantee: bool,
) -> int:
    """Implementation of ``chamber-spike train`` (M4b-9b; ADR-002 §Decisions).

    Returns the process exit code: ``0`` on success;
    :data:`_TRIP_WIRE_EXIT_CODE` if ``--check-guarantee`` fires.
    """
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


def _verify_prereg_command() -> int:
    """Placeholder for the M5 pre-registration check (ADR-002 §Decisions; plan/01 P4)."""
    print("verify-prereg: not yet implemented — M5 owns this (plan/01 P4).")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``chamber-spike`` console script (M4b-9b; ADR-002 §Decisions).

    Args:
        argv: Optional argv override for tests. ``None`` (default)
            reads from :data:`sys.argv` per argparse's usual contract.

    Returns:
        Process exit code; ``0`` on success.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        # No sub-command — print version banner + help, exit 0. Keeps
        # the previous chamber-spike entry point's friendly behavior
        # for users who type ``chamber-spike`` without arguments (the
        # existing ``test_chamber_spike_main`` smoke test exercises
        # exactly this path).
        print(f"chamber-spike  (CHAMBER {chamber.__version__})")
        parser.print_help()
        return 0
    if args.command == "train":
        return _train_command(
            config_path=args.config,
            overrides=_parse_overrides(args.override),
            check_guarantee=args.check_guarantee,
        )
    if args.command == "verify-prereg":
        return _verify_prereg_command()
    parser.error(f"unknown command {args.command!r}")
    return 2  # pragma: no cover  # parser.error sys.exits before this returns.


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
