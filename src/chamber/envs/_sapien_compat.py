# SPDX-License-Identifier: Apache-2.0
"""SAPIEN / ManiSkill v3 compatibility shims shared across CHAMBER envs.

Two helpers consumed by both :mod:`chamber.benchmarks.stage0_smoke` and
:mod:`chamber.envs.stage1_pickplace`:

- :func:`patch_sapien_urdf_no_visual_material` — idempotent strip of
  visual materials from SAPIEN's URDF loader, so headless hosts
  (``CHAMBER_RENDER_BACKEND=none``; no Vulkan render context) can still
  load URDFs that contain ``<visual><material>`` blocks. SAPIEN's
  ``RenderMaterial()`` constructor calls into the global Vulkan render
  context unconditionally during URDF parsing; the patch zero-es every
  visual's material before ``_build_link`` runs. Physics and state
  observations are untouched.
- :func:`load_agent_with_bare_uids` — calls ``BaseEnv._load_agent`` then
  rebuilds ``MultiAgent.agents_dict`` (and re-binds ``get_proprioception``)
  with bare-uid keys. ``MultiAgent.__init__`` mangles every uid as
  ``f"{uid}-{i}"`` once there is more than one robot
  (``mani_skill/agents/multi_agent.py`` constructor; pinned by
  ``mani-skill==3.0.1``); the training stack, the partner-stack
  dispatch, and the chamber wrapper chain all key on the **bare** uid.
  Without the rebind every ``obs["agent"][uid]`` lookup and every
  ``action_space[uid]`` slot would miss.

Wrapper-only discipline (ADR-001 §Decision (wrapper-only)):
the patches go through ManiSkill's public ``BaseEnv._load_agent`` /
``MultiAgent.agents_dict`` surface. Neither helper imports a
``_private`` symbol from ``mani_skill``; the URDF-loader patch reaches
into ``sapien.wrapper.urdf_loader.URDFLoader._build_link``, which is
the documented extension hook for headless-render adaptation
(see ``maniskill_audit.md`` §3 "Override path" for the precedent
established by Stage-0).

References:
- ADR-001 §Risks (Vulkan unavailability path).
- ADR-007 §Stage 1b (P1.03 introduces the second consumer; previously
  inline in :mod:`chamber.benchmarks.stage0_smoke`).
- ``chamber.benchmarks.stage0_smoke._patch_sapien_urdf_no_visual_material``
  (pre-P1.03 inline form; now delegates here).
"""

from __future__ import annotations

import types
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from mani_skill.envs.sapien_env import BaseEnv

#: Idempotency guard for :func:`patch_sapien_urdf_no_visual_material`.
#:
#: SAPIEN's URDFLoader is process-global; patching it twice would chain
#: the wrappers and silently double-strip material info. Module-level
#: flag (not class-level) so the guard survives across env constructions.
_sapien_urdf_patched: bool = False


def patch_sapien_urdf_no_visual_material() -> None:
    """Strip visual materials from SAPIEN's URDF loader (ADR-001 §Risks).

    Idempotent. SAPIEN's C++ ``RenderMaterial()`` binding calls into the
    global Vulkan render context. On headless hosts
    (``CHAMBER_RENDER_BACKEND=none``) no render context is initialised,
    but the URDF loader still instantiates a ``RenderMaterial()`` for
    every visual link before any observation rendering occurs — raising
    ``RuntimeError: failed to find a rendering device``. Setting
    ``render_backend="none"`` on the env constructor skips
    ``RenderSystem`` initialisation but does not suppress
    ``RenderMaterial()``; the URDF loader still has to be muzzled
    upstream.

    Visual materials are purely cosmetic when ``obs_mode`` is
    state-only; the patch zero-es every visual's ``material`` on
    ``_build_link``, leaving physics and state observations untouched.

    Silently returns on ``ImportError`` so the helper can be called
    unconditionally from CPU-only Tier-1 test entry points; the caller
    is expected to surface the missing-SAPIEN condition via
    :class:`chamber.envs.errors.ChamberEnvCompatibilityError` at env
    construction time, not at this helper's call site.
    """
    global _sapien_urdf_patched  # noqa: PLW0603
    if _sapien_urdf_patched:
        return
    try:
        import sapien.wrapper.urdf_loader as _ul  # noqa: PLC0415

        _orig = _ul.URDFLoader._build_link

        def _build_link_no_material(self: object, link: object, link_builder: object) -> object:
            for v in getattr(link, "visuals", []):
                v.material = None
            return _orig(self, link, link_builder)  # type: ignore[arg-type]

        _ul.URDFLoader._build_link = _build_link_no_material  # type: ignore[method-assign]
        _sapien_urdf_patched = True
    except ImportError:
        return


def load_agent_with_bare_uids(
    env: BaseEnv,
    options: dict[str, Any],
    *,
    initial_agent_poses: object | None = None,
    build_separate: bool = False,
) -> None:
    """Multi-robot ``_load_agent`` override with bare-uid rebinding (ADR-001 §Validation criteria).

    Patches two pre-existing ManiSkill issues that block the
    multi-robot rig:

    1. ``BaseEnv._load_agent`` coerces a scalar ``initial_agent_poses=None``
       into ``[None]`` (length 1), then indexes into it with ``i`` for
       each robot. With 2+ robots, ``[None][1]`` raises ``IndexError``.
       Passing an explicit per-agent list keeps the index inside bounds.
    2. ``MultiAgent.__init__`` unconditionally keys every agent as
       ``f"{uid}-{i}"`` once there is more than one robot, propagating
       the suffix into ``action_space`` and ``observation_space``. The
       training stack (config, adapter, trainer) all use bare uids. We
       rebuild ``agents_dict`` with stripped keys and re-bind
       ``get_proprioception`` to iterate the bare-uid dict, before the
       first ``_get_obs()`` runs at the end of ``reset()`` (since
       ``_load_agent`` is invoked inside ``_reconfigure()``).

    Args:
        env: The :class:`BaseEnv` subclass instance owning the
            articulation list. The helper writes to ``env.agent.agents_dict``
            and re-binds ``env.agent.get_proprioception``.
        options: ManiSkill ``_load_agent`` options dict; forwarded
            untouched.
        initial_agent_poses: Per-agent initial pose list. ``None``
            (default) expands to a ``[None] * len(env.robot_uids)`` list
            so SAPIEN uses the agent's URDF-declared root pose. Pass an
            explicit list (each entry a :class:`sapien.Pose` or ``None``)
            to override per-agent placement.
        build_separate: Forwarded to ``BaseEnv._load_agent``; ManiSkill's
            ``build_separate=True`` flag controls whether each agent
            builds its own articulation (the project's wrapper-only
            multi-agent path requires per-agent articulations).
    """
    from mani_skill.envs.sapien_env import BaseEnv as _BaseEnv  # noqa: PLC0415

    if initial_agent_poses is None:
        # ``self.robot_uids`` is the constructor-passed tuple in the
        # multi-robot path; len(...) gives the agent count.
        initial_agent_poses = [None] * len(env.robot_uids)  # type: ignore[arg-type]
    _BaseEnv._load_agent(  # type: ignore[arg-type]
        env,
        options,
        initial_agent_poses=initial_agent_poses,  # type: ignore[arg-type]
        build_separate=build_separate,
    )
    multi = env.agent
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


__all__ = [
    "load_agent_with_bare_uids",
    "patch_sapien_urdf_no_visual_material",
]
