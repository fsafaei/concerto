# SPDX-License-Identifier: Apache-2.0
"""``chamber-spike train-joint`` subcommand (ADR-011 §Decision as amended 2026-07-05).

Runs the B-JOINT MAPPO pair-training loop
(:func:`chamber.benchmarks.joint_mappo_trainer.run_joint_training`) on
a YAML config and prints a one-line summary. B-JOINT trains outside the
AHT setting by construction (both co-carry seats update; ADR-018
§Consequences governs its frozen evaluation), so this subcommand is
deliberately separate from ``chamber-spike train`` — the ego-AHT entry
point and its frozen-partner contract stay untouched.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from chamber.benchmarks.joint_mappo_trainer import load_joint_config, run_joint_training

if TYPE_CHECKING:
    import argparse

#: Number of trailing episodes the summary line averages over (mirrors
#: ``chamber-spike train``'s window).
_SUMMARY_WINDOW: int = 10


def add_parser(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register the ``train-joint`` subparser (ADR-011 §Decision as amended).

    Args:
        sub: The :class:`argparse._SubParsersAction` from the top-level
            ``chamber-spike`` parser.
    """
    train_joint = sub.add_parser(
        "train-joint",
        help="Train the B-JOINT MAPPO co-carry pair (non-AHT upper anchor).",
        description=(
            "Loads a joint_mappo YAML config and runs "
            "chamber.benchmarks.joint_mappo_trainer.run_joint_training — both "
            "seats learn (per-agent MAPPO actors + one shared central critic). "
            "See ADR-011 §Decision as amended 2026-07-05."
        ),
    )
    train_joint.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a YAML (e.g. configs/training/joint_mappo/cocarry_matched.yaml).",
    )
    train_joint.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Dotted override, repeatable — e.g. --override total_frames=2000 "
            "--override runtime.device=cpu (values parsed as YAML scalars)."
        ),
    )


def run(args: argparse.Namespace) -> int:
    """Implementation of ``chamber-spike train-joint`` (ADR-011 §Decision as amended).

    Args:
        args: argparse namespace carrying ``config`` and ``override``.

    Returns:
        ``0`` on success.
    """
    cfg = load_joint_config(args.config, overrides=list(args.override))
    result = run_joint_training(cfg)
    rewards = result.per_episode_rewards
    tail = rewards[-_SUMMARY_WINDOW:] if rewards else []
    mean_tail = sum(tail) / len(tail) if tail else 0.0
    print(
        f"run_id={result.run_id} param_sharing={cfg.param_sharing} "
        f"n_episodes={len(rewards)} n_checkpoints={len(result.checkpoint_paths)} "
        f"mean_reward_last_{len(tail)}={mean_tail:.4f}"
    )
    return 0


__all__ = ["add_parser", "run"]
