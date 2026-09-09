# SPDX-License-Identifier: Apache-2.0
"""Scale a contract's delta-pose actions in front of ManiSkill's controller.

A contract may state smaller per-step bounds than the ManiSkill controller it
runs on (``pickcube`` v2: 0.005 m and 0.05 rad per step against the Panda
``pd_ee_delta_pose`` controller's 0.1 m and 0.1 rad). :class:`ScaleDeltaActions`
realises that. It clips exactly the way ManiSkill 3.0.1's
``PDEEPoseController._clip_and_scale_action`` does (translation per element to
``[-1, 1]``, rotation to unit norm), then multiplies by the ratio of the
contract's bound to the controller's, so the controller's own clip never binds
and the physical delta equals :meth:`ActionField.to_physical` of the raw
action. The gripper field passes through untouched. This module imports no
ManiSkill module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import gymnasium as gym
import torch

if TYPE_CHECKING:
    import numpy as np

    from concerto.contracts.spec import TaskContract

_UNIT = 1.0
_SCALE_TOLERANCE = 1e-9
_EXPECTED_DIMS = {"delta_pos": 3, "delta_rot": 3, "gripper": 1}


class ScaleDeltaActions(gym.ActionWrapper):  # type: ignore[type-arg]
    """Clip like ManiSkill, then scale ``delta_pos`` and ``delta_rot`` to the contract.

    For a raw action ``a`` in the contract's ``[-1, 1]`` per field the controller
    receives ``pos_scale * clip(a_pos, -1, 1)`` and ``rot_scale * normclip(a_rot)``,
    with ``pos_scale = contract bound / pos_upper`` and
    ``rot_scale = contract bound / rot_lower``. Both are signed ratios, so a
    sign-convention drift between contract and controller shows up as a
    negative scale and is refused. With scales of 1 the wrapper is the identity
    up to dtype. The action space is unchanged; the wrapper exposes
    ``.contract`` so an env can be traced back to the contract it implements.

    Args:
        env: The ManiSkill env (at any wrapper depth) whose flat action is
            ``[delta_pos(3), delta_rot(3), gripper(1)]``.
        contract: The contract whose physical bounds the wrapped env realises.
        pos_upper: The controller's translation bound (``config.pos_upper``).
        rot_lower: The controller's signed rotation multiplier (``config.rot_lower``).
    """

    def __init__(
        self,
        env: gym.Env,  # type: ignore[type-arg]
        contract: TaskContract,
        *,
        pos_upper: float,
        rot_lower: float,
    ) -> None:
        """Derive the two scale factors from the contract and the controller; refuse bad ones."""
        super().__init__(env)
        slices = contract.action_slices()
        for name, dim in _EXPECTED_DIMS.items():
            if name not in slices or slices[name].stop - slices[name].start != dim:
                raise ValueError(f"contract {contract.task_id!r} needs a {dim}-wide {name!r} field")
        pos = contract.action_field("delta_pos")
        rot = contract.action_field("delta_rot")
        for f in (pos, rot):
            if abs(f.to_physical(f.low) + f.to_physical(f.high)) > _SCALE_TOLERANCE:
                raise ValueError(f"action field {f.name!r} must be symmetric about zero")
        if pos_upper <= 0.0 or rot_lower == 0.0:
            raise ValueError(
                f"controller bounds must be positive/non-zero, got pos_upper={pos_upper}, "
                f"rot_lower={rot_lower}"
            )
        self.contract = contract
        self.pos_scale = pos.to_physical(pos.high) / pos_upper
        self.rot_scale = rot.to_physical(rot.high) / rot_lower
        for name, scale in (("pos_scale", self.pos_scale), ("rot_scale", self.rot_scale)):
            if not 0.0 < scale <= _UNIT + _SCALE_TOLERANCE:
                raise ValueError(
                    f"{name} must be in (0, 1], got {scale}: the contract's bound must not "
                    "exceed the controller's and must share its sign"
                )
        self._pos = slices["delta_pos"]
        self._rot = slices["delta_rot"]

    def action(self, action: np.ndarray | torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """The action ManiSkill's controller receives for the raw ``action``.

        Accepts numpy or torch, ``(action_dim,)`` or ``(batch, action_dim)``;
        returns a float32 tensor of the same shape and never mutates the input.
        """
        t = torch.as_tensor(action, dtype=torch.float32).clone()
        if t.shape[-1] != self.contract.action_dim:
            raise ValueError(f"expected {self.contract.action_dim} action values, got {t.shape}")
        flat = t.reshape(-1, t.shape[-1])
        flat[:, self._pos] = flat[:, self._pos].clamp(-_UNIT, _UNIT) * self.pos_scale
        rot = flat[:, self._rot]
        norm = torch.linalg.norm(rot, dim=1, keepdim=True)
        flat[:, self._rot] = rot / norm.clamp(min=_UNIT) * self.rot_scale
        return flat.reshape(t.shape)
