# SPDX-License-Identifier: Apache-2.0
"""Build the ManiSkill env a contract describes (imports SAPIEN; kept out of ``__init__``)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import gymnasium as gym
import mani_skill.envs  # noqa: F401  (registers the ManiSkill env ids with gymnasium)

from concerto.contracts.spec import get

if TYPE_CHECKING:
    from concerto.contracts.spec import TaskContract


def make_sim_env(contract: TaskContract | str, **kwargs: Any) -> gym.Env:  # noqa: ANN401 - forwarded verbatim to gym.make
    """``gym.make`` the contract's env; ``kwargs`` pass through (e.g. ``sim_backend``)."""
    c = get(contract) if isinstance(contract, str) else contract
    return gym.make(
        c.env_id, robot_uids=c.robot_uid, obs_mode=c.obs_mode, control_mode=c.control_mode, **kwargs
    )
