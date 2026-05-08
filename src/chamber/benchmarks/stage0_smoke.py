# SPDX-License-Identifier: Apache-2.0
"""Stage 0 smoke test — ADR-001 §Validation criteria, ADR-007 §Stage 0.

This module exposes :func:`make_stage0_env`, the canonical wrapped env used
by the Stage-0 acceptance test. The shape of the test — 3-robot tuple,
100 steps, pass/fail conditions (a)-(c) - is fixed by ADR-001 and must not
be changed without superseding that ADR.

Env ID note (2026-05-08, ManiSkill v3.0.1 at commit a4a4f92):
  ManiSkill v3.0.1 does not register a multi-agent env that accepts an
  arbitrary 3-robot ``robot_uids`` tuple under a named env ID. A minimal
  custom :class:`_Stage0SmokeEnv` subclassing ``BaseEnv`` is used instead;
  this is a clean override (public hook, no monkey-patching) as confirmed by
  the ManiSkill audit (``maniskill_audit.md`` §3 "Override path").
  ManiSkill's ``_load_agent`` accepts the 3-robot tuple without raising;
  the only constraint is Vulkan/GPU availability (SAPIEN requires it).
  ADR-001 risk #3 (build_separate=True vs. dict-action conflict) is not yet
  verified on a GPU machine - flagged in the PR description.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    import gymnasium as gym

from chamber.comm import FixedFormatCommChannel
from chamber.envs import (
    CommShapingWrapper,
    PerAgentActionRepeatWrapper,
    TextureFilterObsWrapper,
)
from chamber.envs.errors import ChamberEnvCompatibilityError

# Per-agent action-repeat counts for the ADR-001 smoke scenario.
# Base env runs at 100 Hz; rates: panda=20 Hz, fetch=10 Hz, hand=50 Hz.
_SMOKE_ACTION_REPEAT: dict[str, int] = {
    "panda_wristcam": 5,  # 100 Hz / 5 = 20 Hz
    "fetch": 10,  # 100 Hz / 10 = 10 Hz
    "allegro_hand_right": 2,  # 100 Hz / 2 = 50 Hz
}

_SMOKE_KEEP: dict[str, list[str]] = {
    "panda_wristcam": ["rgb", "depth", "joint_pos", "joint_vel"],
    "fetch": ["state", "joint_pos"],
    "allegro_hand_right": ["joint_pos", "joint_vel", "tactile"],
}

_SMOKE_ROBOT_UIDS: tuple[str, ...] = (
    "panda_wristcam",
    "fetch",
    "allegro_hand_right",
)


def make_stage0_env(*, render_mode: str | None = None) -> gym.Env:  # type: ignore[return]
    """Return the ADR-001 Stage-0 acceptance env.

    ADR-001 §Validation criteria; ADR-007 §Stage 0 (rig validation gate).

    Robots: ``panda_wristcam`` + ``fetch`` + ``allegro_hand_right``.
    Per-agent rates: panda 20 Hz / fetch 10 Hz / hand 50 Hz (repeat counts
    from a 100 Hz base tick, per ADR-001 cond. c).
    Comm shaping: latency 50 ms, drop rate 5 %.
    Obs filter: per-agent channel subset (see ``_SMOKE_KEEP``).

    Raises:
        ChamberEnvCompatibilityError: if the SAPIEN/Vulkan runtime is not
            available (GPU required by ManiSkill v3; see ADR-001 §Risks and
            the probe results in this module's docstring).
    """
    try:
        import mani_skill.envs  # noqa: F401
        from mani_skill.envs.sapien_env import BaseEnv
    except ImportError as exc:
        raise ChamberEnvCompatibilityError(
            "mani_skill is not installed. Install it per pyproject.toml. "
            "See ADR-001 §Evidence basis."
        ) from exc

    class _Stage0SmokeEnv(BaseEnv):
        """Minimal 3-robot env for ADR-001 Stage-0 rig validation.

        ADR-001 §Validation criteria: no task logic; the goal is purely to
        confirm wrapper dispatch, observation namespacing, and action routing
        work correctly across heterogeneous robot_uids.
        """

        SUPPORTED_ROBOTS: ClassVar[list[tuple[str, ...]]] = [_SMOKE_ROBOT_UIDS]  # type: ignore[assignment]

        def _load_scene(self, options: dict) -> None:
            """Load an empty scene (no objects needed for rig validation).

            ADR-001 §Validation criteria: Stage-0 smoke tests the rig, not a task.
            """

        def _initialize_episode(self, env_idx: object, options: dict) -> None:
            """No episode initialisation needed for rig validation.

            ADR-001 §Validation criteria.
            """

    try:
        env = _Stage0SmokeEnv(
            robot_uids=_SMOKE_ROBOT_UIDS,  # type: ignore[arg-type]
            num_envs=1,
            obs_mode="state",
            control_mode=None,
            render_mode=render_mode,
        )
    except RuntimeError as exc:
        raise ChamberEnvCompatibilityError(
            f"SAPIEN/Vulkan initialisation failed: {exc}\n"
            "Stage-0 smoke requires a GPU with Vulkan support. "
            "See ADR-001 §Risks."
        ) from exc

    env = PerAgentActionRepeatWrapper(env, action_repeat=_SMOKE_ACTION_REPEAT)
    env = TextureFilterObsWrapper(env, keep_per_agent=_SMOKE_KEEP)
    # T2.9 will compose CommDegradationWrapper(channel, URLLC_3GPP_R17["factory"])
    # around the channel; for now the smoke test runs without degradation.
    return CommShapingWrapper(env, channel=FixedFormatCommChannel())
