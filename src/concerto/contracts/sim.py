# SPDX-License-Identifier: Apache-2.0
"""Build the ManiSkill env a contract describes, and register it under a Gymnasium id.

Importing this module imports SAPIEN through ``mani_skill.envs``; it is kept out
of the package ``__init__`` on purpose. The import also registers one Gymnasium
id per contract version (:attr:`TaskContract.gym_id`, e.g.
``concerto/pickcube-v2``) whose entry point is :func:`make_gym_env`: a single
CPU env with unbatched numpy observations and actions that any plain Gymnasium
consumer can train on. Gymnasium imports the module named before the colon, so
``gym.make("concerto.contracts.sim:concerto/pickcube-v2")`` works wherever
concerto is installed, for example as roborl's ``--env-id``.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import gymnasium as gym
import mani_skill.envs  # noqa: F401  (registers the ManiSkill env ids with gymnasium)
from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper

from concerto.contracts.actions import ScaleDeltaActions
from concerto.contracts.spec import get, list_contracts, versions

if TYPE_CHECKING:
    from concerto.contracts.spec import TaskContract

RENDER_BACKEND_ENV_VAR = "CHAMBER_RENDER_BACKEND"
"""Set it to ``none`` on a host without Vulkan; only :func:`make_gym_env` reads it."""


def arm_controller_bounds(env: gym.Env) -> tuple[float, float]:  # type: ignore[type-arg]
    """``(pos_upper, rot_lower)`` of the built env's arm ``pd_ee_delta_pose`` controller."""
    base: Any = env.unwrapped
    config = base.agent.controller.controllers["arm"].config
    return float(config.pos_upper), float(config.rot_lower)


def make_sim_env(contract: TaskContract | str, **kwargs: Any) -> gym.Env:  # noqa: ANN401 - forwarded verbatim to gym.make
    """The batched-torch ManiSkill env implementing ``contract`` (a name picks the latest version).

    ``kwargs`` pass through to ``gym.make`` (e.g. ``sim_backend``); the contract's
    horizon applies unless ``max_episode_steps`` is given. The result is wrapped
    in :class:`ScaleDeltaActions`, which realises the contract's per-step bounds
    and carries ``.contract``.
    """
    c = get(contract) if isinstance(contract, str) else contract
    kwargs.setdefault("max_episode_steps", c.max_episode_steps)
    env = gym.make(
        c.env_id, robot_uids=c.robot_uid, obs_mode=c.obs_mode, control_mode=c.control_mode, **kwargs
    )
    pos_upper, rot_lower = arm_controller_bounds(env)
    return ScaleDeltaActions(env, c, pos_upper=pos_upper, rot_lower=rot_lower)


def make_gym_env(
    task_id: str = "pickcube",
    version: int | None = None,
    *,
    sim_backend: str = "physx_cpu",
    render_backend: str | None = None,
    **kwargs: Any,  # noqa: ANN401 - forwarded verbatim to make_sim_env
) -> gym.Env:
    """One CPU env with unbatched numpy observations and actions (``CPUGymWrapper``).

    The entry point of the registered Gymnasium ids. ``render_backend`` falls
    back to ``$CHAMBER_RENDER_BACKEND`` when that is set (``none`` skips the
    Vulkan render system on Linux); otherwise ManiSkill's default applies.
    """
    backend = render_backend
    if backend is None:
        backend = os.environ.get(RENDER_BACKEND_ENV_VAR)
    if backend is not None:
        kwargs["render_backend"] = backend
    env = make_sim_env(get(task_id, version), sim_backend=sim_backend, **kwargs)
    return CPUGymWrapper(env, ignore_terminations=False, record_metrics=False)


def register_gym_ids() -> list[str]:
    """Register :attr:`TaskContract.gym_id` for every contract version; returns the ids added.

    Idempotent: ids already in Gymnasium's registry are left alone.
    """
    added: list[str] = []
    for task_id in list_contracts():
        for version in versions(task_id):
            c = get(task_id, version)
            if c.gym_id in gym.registry:
                continue
            gym.register(
                id=c.gym_id,
                entry_point="concerto.contracts.sim:make_gym_env",
                kwargs={"task_id": task_id, "version": version},
                max_episode_steps=None,
                disable_env_checker=True,
            )
            added.append(c.gym_id)
    return added


register_gym_ids()
