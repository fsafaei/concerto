# SPDX-License-Identifier: Apache-2.0
# pyright: reportPrivateImportUsage=false
#
# Same rationale as :mod:`chamber.benchmarks.ego_ppo_trainer`: torch's
# stub files do not export ``zeros`` via ``__all__`` even though it is
# public API per the official docs. Suppressed file-locally so the
# zero-reward override stays free of per-line ``type: ignore`` noise.
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

import os
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    import gymnasium as gym

from chamber.comm import (
    URLLC_3GPP_R17,
    CommDegradationWrapper,
    FixedFormatCommChannel,
)
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

#: Idempotency guard for :func:`_patch_sapien_urdf_no_visual_material`.
_sapien_urdf_patched: bool = False


def _patch_sapien_urdf_no_visual_material() -> None:
    """Strip visual materials from SAPIEN's URDF loader (ADR-001 §Risks).

    SAPIEN's C++ ``RenderMaterial()`` binding calls into the global Vulkan
    render context. On headless hosts (``CHAMBER_RENDER_BACKEND=none``) no
    render context is initialised, but the URDF loader still instantiates
    a ``RenderMaterial()`` for every visual link before any observation
    rendering occurs — raising ``RuntimeError: failed to find a rendering
    device``. Setting ``render_backend="none"`` on the env constructor
    skips ``RenderSystem`` initialisation but does not suppress
    ``RenderMaterial()``.

    Since the Stage-0 smoke env uses ``obs_mode="state_dict"`` (no rendered
    frames), visual materials are purely cosmetic and safe to drop. The
    patch zero-es every visual's material on ``_build_link``, leaving
    physics and state observations untouched.

    Idempotent: a second call is a no-op.
    """
    global _sapien_urdf_patched  # noqa: PLW0603
    if _sapien_urdf_patched:
        return
    try:
        import sapien.wrapper.urdf_loader as _ul

        _orig = _ul.URDFLoader._build_link

        def _build_link_no_material(self: object, link: object, link_builder: object) -> object:
            for v in getattr(link, "visuals", []):
                v.material = None
            return _orig(self, link, link_builder)  # type: ignore[arg-type]

        _ul.URDFLoader._build_link = _build_link_no_material  # type: ignore[method-assign]
        _sapien_urdf_patched = True
    except ImportError:
        # sapien not installed — make_stage0_env will surface this via the
        # mani_skill import guard with a ChamberEnvCompatibilityError.
        return


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

        def _load_agent(  # type: ignore[override]
            self,
            options: dict,
            initial_agent_poses: object = None,
            build_separate: bool = False,
        ) -> None:
            """Multi-robot ``_load_agent`` override (ADR-001 §Validation criteria).

            Patches two pre-existing ManiSkill issues that block the 3-robot rig:

            1. ``BaseEnv._load_agent`` coerces a scalar ``initial_agent_poses=None``
               into ``[None]`` (length 1), then indexes into it with ``i`` for each
               robot. With 3 robots, ``[None][1]`` raises ``IndexError``. Passing
               an explicit per-agent list keeps the index inside bounds.
            2. ``MultiAgent.__init__`` unconditionally keys every agent as
               ``f"{uid}-{i}"`` once there is more than one robot, propagating
               the suffix into ``action_space`` and ``observation_space``. The
               training stack (config, adapter, trainer) all use bare uids. We
               rebuild ``agents_dict`` with stripped keys and re-bind
               ``get_proprioception`` to iterate the bare-uid dict, before the
               first ``_get_obs()`` runs at the end of ``reset()`` (since
               ``_load_agent`` is invoked inside ``_reconfigure()``).
            """
            import types

            if initial_agent_poses is None:
                # ``self.robot_uids`` is the constructor-passed tuple in the
                # multi-robot path; len(...) gives the agent count.
                initial_agent_poses = [None] * len(self.robot_uids)  # type: ignore[arg-type]
            super()._load_agent(
                options,
                initial_agent_poses=initial_agent_poses,  # type: ignore[arg-type]
                build_separate=build_separate,
            )
            multi = self.agent
            if not hasattr(multi, "agents_dict"):
                # Single-agent path: ManiSkill never appends a "-i" suffix here.
                return

            def _strip_suffix(uid: str) -> str:
                head, _, tail = uid.rpartition("-")
                return head if head and tail.isdigit() else uid

            multi.agents_dict = {  # type: ignore[attr-defined]
                _strip_suffix(uid): agent
                for uid, agent in multi.agents_dict.items()  # type: ignore[attr-defined]
            }

            def _bare_proprioception(self: object) -> dict[str, object]:
                return {
                    uid: agent.get_proprioception()
                    for uid, agent in self.agents_dict.items()  # type: ignore[attr-defined]
                }

            multi.get_proprioception = types.MethodType(_bare_proprioception, multi)

        def compute_normalized_dense_reward(
            self,
            obs: object,
            action: object,
            info: dict,
        ) -> object:
            """Zero reward — Stage-0 is a rig-validation env with no task.

            ADR-001 §Validation criteria: Stage-0 tests wrapper dispatch and
            action routing, not task performance. ``BaseEnv.get_reward`` calls
            this method and the default raises ``NotImplementedError``, so a
            no-task env must override it. Returning a per-num_envs zero tensor
            keeps the trainer's reward bookkeeping intact.
            """
            del obs, action, info
            import torch

            return torch.zeros(self.num_envs, device=self.device)

    render_backend = os.environ.get("CHAMBER_RENDER_BACKEND", "gpu")

    if render_backend == "none":
        # Headless hosts (CUDA-only, no Vulkan — e.g. WSL2 + Docker) need the
        # URDF visual-material strip-patch before the env is constructed; the
        # patch is idempotent (ADR-001 §Risks).
        _patch_sapien_urdf_no_visual_material()

    try:
        env = _Stage0SmokeEnv(
            robot_uids=_SMOKE_ROBOT_UIDS,  # type: ignore[arg-type]
            num_envs=1,
            obs_mode="state_dict",
            control_mode=None,
            render_mode=render_mode,
            render_backend=render_backend,
        )
    except RuntimeError as exc:
        raise ChamberEnvCompatibilityError(
            f"SAPIEN/Vulkan initialisation failed: {exc}\n"
            "Stage-0 smoke requires a GPU with Vulkan support, or "
            "CHAMBER_RENDER_BACKEND=none on a CUDA-only host. "
            "See ADR-001 §Risks."
        ) from exc

    env = PerAgentActionRepeatWrapper(env, action_repeat=_SMOKE_ACTION_REPEAT)
    env = TextureFilterObsWrapper(env, keep_per_agent=_SMOKE_KEEP)
    channel = CommDegradationWrapper(
        FixedFormatCommChannel(),
        URLLC_3GPP_R17["factory"],
        tick_period_ms=1.0,
        root_seed=0,
    )
    return CommShapingWrapper(env, channel=channel)
