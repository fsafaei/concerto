# SPDX-License-Identifier: Apache-2.0
"""CHAMBER environment wrappers above ManiSkill v3.

ADR-001 §Decision: extend ManiSkill v3 with three thin wrappers covering
the four heterogeneity axes. ADR-005 §Decision: SAPIEN 3 + Warp-MPM is the
physics substrate, inherited from ManiSkill v3.
"""

from chamber.envs.action_repeat import PerAgentActionRepeatWrapper
from chamber.envs.comm_shaping import CommShapingWrapper
from chamber.envs.errors import ChamberEnvCompatibilityError
from chamber.envs.mpe_cooperative_push import MPECooperativePushEnv
from chamber.envs.texture_filter import TextureFilterObsWrapper

__all__ = [
    "ChamberEnvCompatibilityError",
    "CommShapingWrapper",
    "MPECooperativePushEnv",
    "PerAgentActionRepeatWrapper",
    "TextureFilterObsWrapper",
]
