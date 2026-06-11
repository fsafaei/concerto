# SPDX-License-Identifier: Apache-2.0
r"""CHAMBER-side agent extensions (ADR-001 §Decision; ADR-007 §Stage 1b).

Hosts robot-agent classes the project needs in addition to ManiSkill v3's
built-in registry. Currently:

- :class:`PandaPartner` — a 7-DOF Franka with ``uid="panda_partner"``,
  used by the Stage-1 AS-homogeneous condition pair
  ``(panda_wristcam, panda_partner)`` so the two-panda case resolves to
  distinct uid strings under the
  :func:`chamber.envs._sapien_compat.load_agent_with_bare_uids` strip-
  suffix path (otherwise ``("panda_wristcam", "panda_wristcam")`` would
  collapse to a single dict entry post-strip).
- :class:`PandaJacobianProvider` — wraps :mod:`pytorch_kinematics` to
  expose the 3x7 linear end-effector Jacobian
  :math:`J(q) \\in \\mathbb{R}^{3 \\times 7}` that
  :class:`concerto.safety.api.JacobianControlModel` needs (ADR-004
  §Decision; spike_004A §Per-agent control model).

Wrapper-only discipline (ADR-001 §Decision (wrapper-only)):
:class:`PandaPartner` is registered via the public
``mani_skill.agents.registration.register_agent`` decorator. No
``mani_skill/`` source patching; no ``_*`` private imports.

Tier-1 safety: this module's top-level imports tolerate missing
ManiSkill/SAPIEN at module-load time. The :class:`PandaPartner`
registration side-effect is deferred behind a try/except so
``python -c "import chamber.agents"`` succeeds on a Vulkan-less host —
:class:`chamber.envs.errors.ChamberEnvCompatibilityError` only fires at
env-construction time, never at module-import time (mirrors the
stage0_smoke pattern; ADR-001 §Risks).
"""

from __future__ import annotations

from chamber.agents.panda_jacobian import PandaJacobianProvider

try:
    from chamber.agents.panda_partner import PandaPartner

    _PANDA_PARTNER_REGISTERED: bool = True
except ImportError:  # pragma: no cover - exercised on Vulkan-less CI runners
    # ManiSkill / SAPIEN unavailable in this environment. The Tier-1
    # surface (resolve_condition + build_control_models_for_condition)
    # does not need PandaPartner; the Tier-2 env construction will raise
    # ChamberEnvCompatibilityError from chamber.envs.stage1_pickplace
    # the first time the AS-homo condition is requested.
    PandaPartner = None  # type: ignore[assignment]
    _PANDA_PARTNER_REGISTERED = False

__all__ = [
    "PandaJacobianProvider",
    "PandaPartner",
]
