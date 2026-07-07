# SPDX-License-Identifier: Apache-2.0
# pyright: reportPrivateImportUsage=false
#
# Same rationale as :mod:`chamber.agents.panda_jacobian`: torch's stubs
# don't advertise ``tensor`` / ``float64`` in ``__all__`` even though they
# are public per the docs. Suppressed file-locally.
r"""Scripted-competent Stage-1 pick-place ego — REF-SCRIPT (ADR-011 §Decision as amended).

The scripted matched-reference oracle for ``stage1_pickplace_as``
admission cells (ADR-027 §Admission protocol): a deterministic
phase-machine that picks the cube and places it at the goal using only
observation leaves — ``extra.cube_pose`` / ``extra.goal_pos`` /
``extra.tcp_pose`` and the ego arm's ``qpos`` — mapped to
``pd_joint_delta_pos`` joint deltas through a damped-pseudoinverse
Jacobian (pytorch-kinematics chain over the Panda URDF; the ADR-004
§Decision damped-pseudoinverse convention).

The controller reads **no partner leaf in any mode** — that is not an
implementation shortcut but the measured construct fact ADR-026
§Decision 3 recorded about this task (the success predicate is
ego-only; the partner does not participate), and it is exactly what the
A2/A3 admission cells demonstrate with numbers: the best available
reference on this task is already partner-blind, so the task is a
Tier-1 control, not a cooperation measurement (ADR-027 §Tier ladder).
``mask_partner_obs=True`` additionally deletes the partner's agent
subtree from the observation before any read, so the A3 blind cell's
"acts without partner state" label is enforced by construction rather
than by inspection.

Phase machine (all thresholds in metres, world frame):
``reach_above`` (hover over the cube) → ``descend`` (to grasp depth) →
``grasp`` (close the gripper) → ``move`` (carry to the goal) →
``settle`` (hold still so the ``is_robot_static`` conjunct converges).
A drop during ``move``/``settle`` falls back to ``reach_above``.

References:
- ADR-011 §Decision as amended (REF-SCRIPT / B-BLIND baseline roles).
- ADR-027 §Admission protocol (A1/A2/A3 cells this ego drives).
- ADR-026 §Decision 3 (the ego-solvability construct fact).
- ADR-004 §Decision (damped-pseudoinverse joint-step convention).
- ADR-001 §Risks / P2 (lazy heavyweight imports; module import stays
  Tier-1-safe).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping

    from numpy.typing import NDArray

#: The ego arm uid this oracle drives (the panda seat of every
#: ``stage1_pickplace`` condition — ADR-007 §Stage 1b).
PICKPLACE_EGO_UID: str = "panda_wristcam"

#: Ordered phase labels of the pick-place phase machine.
PICKPLACE_PHASES: tuple[str, ...] = ("reach_above", "descend", "grasp", "move", "settle")

#: World-frame base position of the ego panda in the Stage-1 scene
#: (``chamber.envs.stage1_pickplace._AGENT_BASE_POSE_XYZ``; identity
#: base rotation). The Jacobian chain is rooted at the arm base, so
#: world targets are translated by this offset.
PICKPLACE_EGO_BASE_XYZ: tuple[float, float, float] = (-0.615, 0.0, 0.0)

#: Number of Panda arm DOF the Jacobian spans.
_PANDA_ARM_DOF: int = 7

#: ``pd_joint_delta_pos`` per-step joint-delta scale (rad); the emitted
#: normalised action is mapped through this by the env's controller
#: (mani-skill==3.0.1 panda config: lower=-0.1, upper=0.1).
_ACTION_DELTA_SCALE: float = 0.1

#: Hover height above the cube during ``reach_above``, metres.
_HOVER_ABOVE_M: float = 0.08

#: TCP depth offset relative to the cube centre at grasp, metres.
_GRASP_DEPTH_M: float = -0.002

#: Phase-transition position tolerances, metres.
_REACH_TOL_M: float = 0.015
_DESCEND_TOL_M: float = 0.008

#: Cube-at-goal distance below which the carry switches to ``settle``
#: (inside the env's 0.025 m goal threshold), metres.
_PLACE_TOL_M: float = 0.015

#: Cube-to-TCP distance above which a drop is declared during
#: ``move``/``settle`` and the machine falls back to ``reach_above``.
_DROP_TOL_M: float = 0.06

#: Steps the gripper closes for during ``grasp`` before carrying.
_GRASP_STEPS: int = 8

#: Cartesian proportional gain and damped-pseudoinverse damping of the
#: tracking law (ADR-004 §Decision convention).
_KP: float = 2.5
_ROT_KP: float = 1.0
_DAMPING_SQ: float = 1e-2

#: Gripper action values (normalised): open / close.
_GRIP_OPEN: float = 1.0
_GRIP_CLOSE: float = -1.0


def _to_flat(value: object) -> NDArray[np.float64]:
    """Coerce a torch tensor / ndarray obs leaf to a flat float64 array (env 0)."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()  # type: ignore[union-attr]
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim >= 2:  # noqa: PLR2004 - rank-2 is the (num_envs, dim) batched layout
        arr = arr[0]
    return arr.reshape(-1)


class ScriptedPickPlaceEgo:
    """Deterministic scripted pick-place oracle for the Stage-1 ego seat (ADR-011 REF-SCRIPT).

    Args:
        mask_partner_obs: When ``True``, the partner's ``obs["agent"]``
            subtree is deleted before any read — the B-BLIND enforcement
            for A3 cells (ADR-027 §Admission protocol). The control law
            is identical either way; on this task that *is* the finding
            (ADR-026 §Decision 3).
        partner_uid: The partner agent uid to mask.
        base_xyz: World-frame ego arm base position override.
    """

    def __init__(
        self,
        *,
        mask_partner_obs: bool = False,
        partner_uid: str = "fetch",
        base_xyz: tuple[float, float, float] = PICKPLACE_EGO_BASE_XYZ,
    ) -> None:
        """Build the Panda kinematic chain (lazy heavyweight imports; ADR-001 P2)."""
        import pytorch_kinematics as pk
        import torch
        from mani_skill.agents.robots.panda.panda_wristcam import (
            PandaWristCam,
        )

        self._torch = torch
        urdf_path = PandaWristCam.urdf_path
        if urdf_path is None:  # pragma: no cover - upstream always sets it
            msg = "PandaWristCam.urdf_path is unset; cannot build the kinematic chain"
            raise RuntimeError(msg)
        chain = pk.build_serial_chain_from_urdf(Path(urdf_path).read_bytes(), "panda_hand_tcp")
        self._chain = chain.to(dtype=torch.float64)
        self._mask_partner_obs = mask_partner_obs
        self._partner_uid = partner_uid
        self._base_xyz = np.asarray(base_xyz, dtype=np.float64)
        self._phase: str = PICKPLACE_PHASES[0]
        self._t_in_phase: int = 0

    @property
    def phase(self) -> str:
        """Current phase label (one of :data:`PICKPLACE_PHASES`)."""
        return self._phase

    def reset(self, *, seed: int | None = None) -> None:
        """Reset the phase machine (deterministic; the seed is a protocol formality)."""
        del seed
        self._phase = PICKPLACE_PHASES[0]
        self._t_in_phase = 0

    def act(self, obs: Mapping[str, Any]) -> NDArray[np.float32]:
        """Return the ``pd_joint_delta_pos`` action for the ego seat (ADR-011 REF-SCRIPT).

        Args:
            obs: Stage-1 ``state_dict`` observation mapping. Only
                ``extra.{cube_pose,goal_pos,tcp_pose}`` and the ego's
                ``agent.<uid>.qpos`` are read.

        Returns:
            Float32 action of length 8 (7 normalised joint deltas +
            gripper command).
        """
        if self._mask_partner_obs:
            obs = self._masked(obs)
        agent = obs["agent"]
        q = _to_flat(agent[PICKPLACE_EGO_UID]["qpos"])[:_PANDA_ARM_DOF]
        tcp_p = _to_flat(obs["extra"]["tcp_pose"])[:3]
        cube = _to_flat(obs["extra"]["cube_pose"])[:3]
        goal = _to_flat(obs["extra"]["goal_pos"])[:3]

        grip = _GRIP_OPEN
        target = tcp_p
        if self._phase == "reach_above":
            target = cube + np.asarray([0.0, 0.0, _HOVER_ABOVE_M])
            if float(np.linalg.norm(tcp_p - target)) < _REACH_TOL_M:
                self._transition("descend")
        if self._phase == "descend":
            target = cube + np.asarray([0.0, 0.0, _GRASP_DEPTH_M])
            if float(np.linalg.norm(tcp_p - target)) < _DESCEND_TOL_M:
                self._transition("grasp")
        if self._phase == "grasp":
            target = tcp_p
            grip = _GRIP_CLOSE
            if self._t_in_phase >= _GRASP_STEPS:
                self._transition("move")
        if self._phase in ("move", "settle"):
            grip = _GRIP_CLOSE
            if float(np.linalg.norm(cube - tcp_p)) > _DROP_TOL_M:
                self._transition("reach_above")
                target = cube + np.asarray([0.0, 0.0, _HOVER_ABOVE_M])
                grip = _GRIP_OPEN
            else:
                target = goal
                if self._phase == "move" and float(np.linalg.norm(cube - goal)) < _PLACE_TOL_M:
                    self._transition("settle")

        self._t_in_phase += 1
        dq = self._damped_step(q, target)
        action = np.zeros(_PANDA_ARM_DOF + 1, dtype=np.float32)
        action[:_PANDA_ARM_DOF] = np.clip(dq / _ACTION_DELTA_SCALE, -1.0, 1.0).astype(np.float32)
        action[_PANDA_ARM_DOF] = grip
        return action

    def _masked(self, obs: Mapping[str, Any]) -> dict[str, Any]:
        """Delete the partner agent subtree (B-BLIND enforcement, ADR-027 A3)."""
        out = dict(obs)
        agent = dict(out.get("agent", {}))
        agent.pop(self._partner_uid, None)
        out["agent"] = agent
        return out

    def _transition(self, phase: str) -> None:
        self._phase = phase
        self._t_in_phase = 0

    def _damped_step(
        self, q: NDArray[np.float64], target_world: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Damped-pseudoinverse joint step toward ``target_world`` (ADR-004 §Decision convention).

        6-D task: position tracking plus a tool-z-down orientation term
        so the grasp stays top-down through the carry.
        """
        torch = self._torch
        target_base = np.asarray(target_world, dtype=np.float64) - self._base_xyz
        qt = torch.tensor(q, dtype=torch.float64).unsqueeze(0)
        jac = self._chain.jacobian(qt)[0].numpy()  # (6, 7), base frame
        fk = self._chain.forward_kinematics(qt)
        mat = fk.get_matrix()[0].numpy()  # type: ignore[union-attr]
        tcp_base = mat[:3, 3]
        z_axis = mat[:3, 2]
        err = np.concatenate(
            [
                _KP * (target_base - tcp_base),
                _ROT_KP * np.cross(z_axis, np.asarray([0.0, 0.0, -1.0])),
            ]
        )
        jjt = jac @ jac.T + _DAMPING_SQ * np.eye(6)
        dq = jac.T @ np.linalg.solve(jjt, err)
        return np.clip(dq, -_ACTION_DELTA_SCALE, _ACTION_DELTA_SCALE).astype(np.float64, copy=False)


__all__ = [
    "PICKPLACE_EGO_BASE_XYZ",
    "PICKPLACE_EGO_UID",
    "PICKPLACE_PHASES",
    "ScriptedPickPlaceEgo",
]
