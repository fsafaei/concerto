# SPDX-License-Identifier: Apache-2.0
"""PandaPartner — second-panda agent for the AS-homogeneous condition (ADR-007 §Stage 1b).

The Stage-1 AS axis pre-registration (``spikes/preregistration/AS.yaml``,
git tag ``prereg-stage1-AS-2026-05-15``) names the homogeneous
condition's agent tuple as ``("panda_wristcam", "panda_partner")``.
ManiSkill v3.0.1's built-in agent registry covers ``panda``,
``panda_wristcam``, ``panda_stick``, ``fetch``, and the Allegro hands
(see ``mani_skill.agents.registration.REGISTERED_AGENTS``) — there is
no ``panda_partner``. The strip-suffix path in
:func:`chamber.envs._sapien_compat.load_agent_with_bare_uids` would
collapse two ``panda_wristcam`` entries to a single ``agents_dict``
key, so the AS-homo "two pandas" case needs a second, distinct uid.

This module defines that uid: a thin :class:`Panda` subclass mounted on
the no-wristcam ``panda_v2.urdf`` (the AS-homo partner doesn't need its
own perception — the ego owns the camera mount via ``panda_wristcam``).

Wrapper-only discipline (ADR-001 §Decision (wrapper-only)):
:class:`PandaPartner` is registered via the public
``@register_agent()`` decorator from
``mani_skill.agents.registration``. No ``mani_skill/`` source patching;
no ``_*`` private imports. The base :class:`Panda` class's
``controller_configs``, ``arm_joint_names``, ``gripper_joint_names``,
``tcp_pose`` property, ``is_grasping``, and ``is_static`` are inherited
unchanged — the only difference from ``panda_wristcam`` is the URDF
choice (no on-gripper camera mount).

References:
- ADR-007 §Stage 1b (P1.03 introduces the partner agent for the AS axis).
- ``spikes/preregistration/AS.yaml`` (the pre-registration that names
  the ``("panda_wristcam", "panda_partner")`` tuple).
- ``mani_skill.agents.robots.panda.panda_wristcam.PandaWristCam`` (the
  parallel uid; pattern this class mirrors).
"""

from __future__ import annotations

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.registration import register_agent
from mani_skill.agents.robots.panda.panda import Panda


@register_agent()
class PandaPartner(Panda):
    """Second-panda agent for the AS-homogeneous condition (ADR-007 §Stage 1b).

    Inherits every Panda hook (arm joints, gripper joints, TCP pose,
    grasp / static predicates, controller configs) from
    :class:`mani_skill.agents.robots.panda.panda.Panda`. The only
    difference is the URDF: :class:`PandaPartner` uses ``panda_v2.urdf``
    (no on-gripper camera mount), since the AS-homo partner is driven by
    the scripted-heuristic / trained policy rather than by camera-fed
    perception (the ego owns the camera via ``panda_wristcam``).
    """

    uid = "panda_partner"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/panda/panda_v2.urdf"
