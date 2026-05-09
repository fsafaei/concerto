# SPDX-License-Identifier: Apache-2.0
"""Hydra-driven ego-AHT runner (CONCERTO T4b.7).

Lives inside the ``concerto-org/harl-fork`` repo at
``harl/runners/ego_aht_runner.py``. Generated from CONCERTO's
``scripts/harl-fork-patches/v0.1.0-aht/`` recipe.

Bridges the Hydra YAML configs under CONCERTO's ``configs/training/``
to the algorithm-agnostic loop in
:func:`concerto.training.ego_aht.train`, with the
:class:`harl.algorithms.actors.ego_aht_happo.EgoAHTHAPPO` factory
plugged into the trainer-factory seam from CONCERTO M4b-5.

ADR-002 §Decisions: this is the canonical fork-side launch path. The
loop body itself lives in ``concerto.training.ego_aht.train`` so the
fork stays small (~150 LOC across three files); the chamber-side
``chamber.benchmarks.training_runner.run_training`` does env / partner
construction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import hydra
from harl.algorithms.actors.ego_aht_happo import EgoAHTHAPPO

if TYPE_CHECKING:  # pragma: no cover
    from omegaconf import DictConfig


@hydra.main(version_base="1.3", config_path=None)
def main(cfg: DictConfig) -> None:
    """Hydra-decorated entry point for the harl-side runner (T4b.7).

    Recommended Phase-0 invocation, run from a CONCERTO checkout with
    the ``concerto-org/harl-fork`` installed:

    .. code-block:: shell

        python -m harl.runners.ego_aht_runner \\
            --config-path "$PWD/configs/training/ego_aht_happo" \\
            --config-name mpe_cooperative_push

    The function:
    1. Validates the composed config through CONCERTO's
       :class:`concerto.training.config.EgoAHTConfig`.
    2. Calls :func:`chamber.benchmarks.training_runner.run_training`
       with :meth:`EgoAHTHAPPO.from_config` as the trainer factory.
    3. Logs a short summary of the returned
       :class:`concerto.training.ego_aht.RewardCurve` to stdout.

    Args:
        cfg: The Hydra-composed :class:`omegaconf.DictConfig`.

    # UPSTREAM-VERIFY: Hydra's @hydra.main decorator expects a
    # config_path. We pass ``None`` so the caller specifies it via
    # ``--config-path`` on the CLI. Verify this is supported on the
    # pinned Hydra version (1.3+); older Hydra requires a string path.
    """
    from omegaconf import OmegaConf

    from chamber.benchmarks.training_runner import run_training
    from concerto.training.config import EgoAHTConfig

    raw = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(raw, dict):
        raise TypeError(
            f"Composed Hydra config must be a mapping at root; got {type(raw).__name__}"
        )
    pydantic_cfg = EgoAHTConfig.model_validate(raw)

    curve = run_training(
        pydantic_cfg,
        trainer_factory=EgoAHTHAPPO.from_config,
    )

    print(  # stdout summary is the runner's documented surface.
        f"[ego_aht_runner] run_id={curve.run_id} "
        f"steps={len(curve.per_step_ego_rewards)} "
        f"episodes={len(curve.per_episode_ego_rewards)} "
        f"checkpoints={len(curve.checkpoint_paths)}"
    )


if __name__ == "__main__":
    main()
