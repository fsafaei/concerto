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
from chamber.envs.partner_meta import PartnerIdAnnotationWrapper
from chamber.envs.stage1_obs_filter import Stage1OMChannelFilter
from chamber.envs.stage1_pickplace import (
    ConditionConfig,
    build_control_models_for_condition,
    make_stage1_pickplace_env,
    resolve_condition,
)
from chamber.envs.texture_filter import TextureFilterObsWrapper

__all__ = [
    "ChamberEnvCompatibilityError",
    "CommShapingWrapper",
    "ConditionConfig",
    "MPECooperativePushEnv",
    "PartnerIdAnnotationWrapper",
    "PerAgentActionRepeatWrapper",
    "Stage1OMChannelFilter",
    "TextureFilterObsWrapper",
    "build_control_models_for_condition",
    "make_stage1_pickplace_env",
    "resolve_condition",
]
