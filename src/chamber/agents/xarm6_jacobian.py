# SPDX-License-Identifier: Apache-2.0
# pyright: reportPrivateImportUsage=false
r"""xArm6 6-DOF end-effector Jacobian + FK provider (ADR-005 §Decision; ADR-026 §Decision 4).

The xArm6 analogue of :class:`chamber.agents.panda_jacobian.PandaJacobianProvider`,
built for the Rung-4 embodiment-heterogeneity (EH) slice (R-2026-06-B §15
Rung 4): the frozen Rung-2 Panda incumbent is held fixed and its Panda partner
is swapped for an **xArm6 + Robotiq 2F-85** teammate. The Panda provider's
serial chain (7 joints to ``panda_hand_tcp``) does not describe the xArm6, so
the xArm6 co-carry controller needs its own ``pytorch_kinematics`` chain — 6
revolute joints (``joint1..joint6``) to the ``eef`` tool-center-point link.

Like the Panda provider this wraps :mod:`pytorch_kinematics` (a transitive
``mani-skill==3.0.1`` dep) and exposes the 3xN **linear** end-effector
Jacobian + companion FK in the arm base (``link_base``) frame, so a Cartesian
error and its damped-pseudoinverse joint step compose consistently. The
co-carry weld locks only the linear DOF (a ball-and-socket pin; rotation
free), so position-only FK/Jacobian is sufficient — orientation is out of
scope, exactly as for the Panda matched controller.

Tier-1-safe: parses the xArm6 URDF via ``pytorch_kinematics`` without SAPIEN,
so ``import chamber.agents.xarm6_jacobian`` and the controller's control law
are exercised by Tier-1 tests on a Vulkan-less host (mirrors
``panda_jacobian`` / the co-carry matched-controller Tier-1 pattern).

References:
- ADR-005 §Decision (ManiSkill v3 / SAPIEN 3 robot availability; the xArm6 +
  Robotiq 2F-85 ships in ``mani-skill==3.0.1`` as ``xarm6_robotiq``).
- ADR-026 §Decision 4 (the Phase-2 forward-design ladder; Rung 4 = EH).
- ADR-004 §Decision (the damped-pseudoinverse joint-step convention).
- R-2026-06-B §15 Rung 4 (the embodiment-shift teammate this serves).
- :class:`chamber.agents.panda_jacobian.PandaJacobianProvider` (the Panda
  analogue this mirrors).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray

#: Number of DOF in the xArm6 arm (excludes the Robotiq gripper joints, which
#: are downstream of the serial chain's ``eef`` end link).
XARM6_ARM_DOF: int = 6

#: Cartesian position dimension (the linear slice of the 6xN Jacobian).
CARTESIAN_POSITION_DIM: int = 3

#: Default xArm6 + Robotiq URDF path under the ManiSkill asset root. The
#: ``xarm6_robotiq`` agent (ADR-005) ships this URDF; the asset is fetched by
#: ``mani_skill.utils.download_asset xarm6_robotiq`` if absent.
_DEFAULT_XARM6_URDF: str = "~/.maniskill/data/robots/xarm6/xarm6_robotiq.urdf"


class XArm6JacobianProvider:
    """Frozen xArm6 URDF chain -> linear Jacobian callable (ADR-005 §Decision; ADR-026 §D4).

    Parses the xArm6 URDF once at construction (via
    :func:`pytorch_kinematics.build_serial_chain_from_urdf`) and caches the
    resulting 6-joint :class:`pytorch_kinematics.SerialChain` to ``eef``. Each
    :meth:`__call__` evaluates the 3x6 linear Jacobian :math:`J(q)`; :meth:`fk_tcp_position`
    evaluates the companion forward kinematics — both in the ``link_base``
    frame, mirroring :class:`chamber.agents.panda_jacobian.PandaJacobianProvider`.

    Args:
        urdf_path: Path to the xArm6 URDF; ``None`` selects the default
            ManiSkill asset path (:data:`_DEFAULT_XARM6_URDF`, ``~`` expanded).
        ee_link: End-effector link the chain terminates at; defaults to
            ``"eef"`` (the ``xarm6_robotiq`` tool-center-point between the
            Robotiq fingers).

    Raises:
        FileNotFoundError: If the URDF does not exist on disk.
        ValueError: If the extracted chain does not have exactly
            :data:`XARM6_ARM_DOF` joints.
    """

    def __init__(
        self,
        urdf_path: str | None = None,
        *,
        ee_link: str = "eef",
    ) -> None:
        """Parse the xArm6 URDF and cache the 6-joint serial chain (ADR-026 §D4)."""
        import os.path

        import pytorch_kinematics as pk

        resolved = os.path.expanduser(urdf_path if urdf_path is not None else _DEFAULT_XARM6_URDF)
        with open(resolved, "rb") as fh:
            urdf_blob = fh.read()
        self._chain = pk.build_serial_chain_from_urdf(urdf_blob, end_link_name=ee_link)
        self._ee_link: str = ee_link
        if self._chain.n_joints != XARM6_ARM_DOF:
            msg = (
                f"XArm6JacobianProvider: expected {XARM6_ARM_DOF}-joint serial chain "
                f"to {ee_link!r}; pytorch_kinematics returned {self._chain.n_joints} "
                "joints. Check URDF integrity or pick a different end-effector link."
            )
            raise ValueError(msg)

    @property
    def ee_link(self) -> str:
        """The end-effector link the chain terminates at (read-only; ADR-026 §D4)."""
        return self._ee_link

    def fk_tcp_position(self, qpos_arm: NDArray[np.floating]) -> NDArray[np.float64]:
        """Forward-kinematics ``eef`` position in the xArm6 base frame (ADR-026 §Decision 4).

        Companion to :meth:`__call__`: evaluates the cached chain's FK at the
        6 arm joint angles and returns the 3-D Cartesian position of ``eef`` in
        the ``link_base`` frame — the same frame :meth:`__call__` returns, so a
        Cartesian error and its damped-pseudoinverse step compose. Consumed by
        the xArm6 co-carry controller.

        Args:
            qpos_arm: The 6 xArm6 arm joint angles, shape ``(6,)``.

        Returns:
            ``eef`` position in the base frame, shape ``(3,)``, float64.

        Raises:
            ValueError: If ``qpos_arm`` shape mismatches ``(6,)``.
        """
        import torch

        q = np.asarray(qpos_arm, dtype=np.float32)
        if q.shape != (XARM6_ARM_DOF,):
            msg = (
                f"XArm6JacobianProvider.fk_tcp_position: qpos_arm shape {q.shape} "
                f"mismatches expected ({XARM6_ARM_DOF},) for the 6-DOF xArm6 chain."
            )
            raise ValueError(msg)
        q_torch = torch.from_numpy(q).unsqueeze(0)
        mat = self._chain.forward_kinematics(q_torch).get_matrix()  # type: ignore[union-attr]
        pos = mat[0, :CARTESIAN_POSITION_DIM, 3].detach().cpu().numpy()
        return np.ascontiguousarray(pos, dtype=np.float64)

    def jacobian(self, qpos_arm: NDArray[np.floating]) -> NDArray[np.float64]:
        """The 3x6 linear end-effector Jacobian at ``qpos_arm`` (ADR-026 §Decision 4).

        Args:
            qpos_arm: The 6 xArm6 arm joint angles, shape ``(6,)``.

        Returns:
            Linear Jacobian rows ``d(eef_pos)/d(q)``, shape ``(3, 6)``, float64.

        Raises:
            ValueError: If ``qpos_arm`` shape mismatches ``(6,)``.
        """
        import torch

        q = np.asarray(qpos_arm, dtype=np.float32)
        if q.shape != (XARM6_ARM_DOF,):
            msg = (
                f"XArm6JacobianProvider.jacobian: qpos_arm shape {q.shape} mismatches "
                f"expected ({XARM6_ARM_DOF},) for the 6-DOF xArm6 chain."
            )
            raise ValueError(msg)
        q_torch = torch.from_numpy(q).unsqueeze(0)
        jac_6x6 = self._chain.jacobian(q_torch)[0]
        jac_3x6 = jac_6x6[:CARTESIAN_POSITION_DIM].detach().cpu().numpy()
        return np.ascontiguousarray(jac_3x6, dtype=np.float64)


__all__ = [
    "CARTESIAN_POSITION_DIM",
    "XARM6_ARM_DOF",
    "XArm6JacobianProvider",
]
