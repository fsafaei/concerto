# SPDX-License-Identifier: Apache-2.0
# pyright: reportPrivateImportUsage=false
#
# Same rationale as :mod:`chamber.benchmarks.stage0_smoke`: torch's stubs
# don't advertise ``from_numpy`` in ``__all__`` even though it's public per
# the docs. Suppressed file-locally so the Jacobian helper stays free of
# per-line ``type: ignore`` noise.
r"""Panda 7-DOF end-effector Jacobian provider (ADR-004 §Decision; ADR-007 §Stage 1b).

Wraps :mod:`pytorch_kinematics` (already a transitive dep of
``mani-skill==3.0.1`` — :mod:`mani_skill.agents.controllers.utils.kinematics`
uses it for built-in IK; P1.03 promotes it to a load-bearing direct dep in
``pyproject.toml`` so :class:`PandaJacobianProvider` has a stable surface
under wrapper-only discipline) to compute the analytical 3x7 linear
end-effector Jacobian :math:`J(q) \\in \\mathbb{R}^{3 \\times 7}` the
:class:`concerto.safety.api.JacobianControlModel` needs.

Why the 3x7 linear slice rather than the full 6x7 SE(3) Jacobian:
:class:`concerto.safety.cbf_qp.AgentSnapshot` carries Cartesian
``position`` (shape ``(d,)``, typically ``d == 3``) and ``velocity``
only — no orientation channel. The CBF backbone's pairwise barrier
:math:`h_{ij}` is a Euclidean-distance constraint; angular Jacobian
rows would be dimensionally orthogonal to it. The damped-pseudoinverse
in :meth:`JacobianControlModel.cartesian_accel_to_action`
(Nakamura & Hanafusa 1986) operates on the 3x7 slice; consistency with
the safety-stack contract is the design constraint.

Note on the closure-via-env-reference workaround (Q3 mod 3 in the P1.03
design conversation):
:class:`AgentSnapshot` does not carry ``qpos`` (extension is sequenced
to slice P1.05.5 per ADR-004 §Open questions; the contingent trigger is
Stage-1b λ-telemetry saturation). The
:class:`PandaJacobianProvider.__call__` signature therefore takes
``(snap, qpos_arm)`` — the env's
:meth:`chamber.envs.stage1_pickplace.Stage1PickPlaceEnv.build_control_models`
constructs a closure ``lambda snap: provider(snap, env._latest_qpos["panda_wristcam"])``
that captures a fresh env reference per ``(seed, condition)`` cell
(safe because :func:`chamber.benchmarks.stage1_as._run_axis_with_factories`
builds the env once per cell and reuses it across the 20 evaluation
episodes within the cell — the closure stays valid for the cell's
lifetime; cited at ``stage1_as.py:253-275``).

References:
- ADR-004 §Decision (the JacobianControlModel contract).
- ADR-004 §Open questions (the AgentSnapshot.qpos extension as the
  long-term fix; slice P1.05.5).
- ADR-007 §Stage 1b (the slice introducing this provider).
- :class:`concerto.safety.api.JacobianControlModel` (the consumer).
- ``mani_skill.agents.controllers.utils.kinematics.Kinematics`` (the
  precedent for using ``pytorch_kinematics`` in ManiSkill's own IK).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray

    from concerto.safety.cbf_qp import AgentSnapshot

#: Number of DOF in the Panda arm (excludes the two gripper finger joints
#: which are downstream of the serial chain's end link).
PANDA_ARM_DOF: int = 7

#: Cartesian position dimension exposed by :class:`AgentSnapshot`.
#:
#: ADR-004 §Decision pins the CBF backbone to Cartesian-only safety
#: bodies; orientation is out of scope for the pairwise distance
#: barrier. Used here to slice the 6-row SE(3) Jacobian to the 3-row
#: linear part :class:`JacobianControlModel.cartesian_accel_to_action`
#: consumes.
CARTESIAN_POSITION_DIM: int = 3


class PandaJacobianProvider:
    """Frozen URDF chain -> linear Jacobian callable (ADR-004 §Decision; §Stage 1b).

    Parses the panda URDF once at construction (via
    :func:`pytorch_kinematics.build_serial_chain_from_urdf`) and caches
    the resulting :class:`pytorch_kinematics.SerialChain`. Each
    :meth:`__call__` evaluates :math:`J(q)` on the cached chain.

    Args:
        urdf_path: Absolute path to the panda URDF on disk. The default
            (``None``) selects the wristcam URDF
            (``{PACKAGE_ASSET_DIR}/robots/panda/panda_v3.urdf``) since
            the Stage-1b panda is by convention the wristcam variant;
            pass ``panda_v2.urdf`` to target the no-camera URDF used by
            :class:`PandaPartner`.
        ee_link: Name of the end-effector link the chain terminates at.
            Defaults to ``"panda_hand_tcp"`` — the tool-center-point
            link the ManiSkill PickCube task uses for
            ``self.agent.tcp_pose`` (see
            ``mani_skill/envs/tasks/tabletop/pick_cube.py:136``). The
            serial chain extracted with this end-link has exactly 7
            joints (the 7 arm joints; the two gripper-finger joints are
            siblings of ``panda_hand``, not in the chain).

    Raises:
        FileNotFoundError: If ``urdf_path`` does not exist on disk.
        ValueError: If :func:`pytorch_kinematics.build_serial_chain_from_urdf`
            fails to extract a chain ending at ``ee_link``.
    """

    def __init__(
        self,
        urdf_path: str | None = None,
        *,
        ee_link: str = "panda_hand_tcp",
    ) -> None:
        """Parse the URDF and cache the serial chain (ADR-007 §Stage 1b)."""
        import pytorch_kinematics as pk

        if urdf_path is None:
            from mani_skill import PACKAGE_ASSET_DIR

            urdf_path = f"{PACKAGE_ASSET_DIR}/robots/panda/panda_v3.urdf"
        with open(urdf_path, "rb") as fh:
            urdf_blob = fh.read()
        self._chain = pk.build_serial_chain_from_urdf(urdf_blob, end_link_name=ee_link)
        self._ee_link: str = ee_link
        if self._chain.n_joints != PANDA_ARM_DOF:
            msg = (
                f"PandaJacobianProvider: expected {PANDA_ARM_DOF}-joint serial chain "
                f"to {ee_link!r}; pytorch_kinematics returned {self._chain.n_joints} "
                "joints. Check URDF integrity or pick a different end-effector link."
            )
            raise ValueError(msg)

    @property
    def ee_link(self) -> str:
        """The end-effector link the chain terminates at (read-only)."""
        return self._ee_link

    def __call__(
        self,
        snap: AgentSnapshot,
        qpos_arm: NDArray[np.floating],
    ) -> NDArray[np.float64]:
        """Compute the 3x7 linear end-effector Jacobian at ``qpos_arm`` (ADR-004 §Decision).

        Args:
            snap: Current :class:`AgentSnapshot` — unused at the
                provider level (the Jacobian is a function of joint
                state, not Cartesian state); accepted for Protocol
                conformance with
                :class:`concerto.safety.api.JacobianControlModel.jacobian_fn`.
                See module docstring for the closure-via-env-reference
                rationale and the contingent P1.05.5 follow-up that
                folds qpos into :class:`AgentSnapshot` directly.
            qpos_arm: Joint-angle vector for the 7 panda arm joints, in
                the same order as ``mani_skill.agents.robots.panda.Panda.arm_joint_names``.
                Shape ``(7,)``; dtype float-coercible to ``float32``.

        Returns:
            Linear Jacobian rows of the 6x7 SE(3) Jacobian at
            ``qpos_arm``, shape ``(3, 7)``, dtype ``float64`` (matches
            the :class:`JacobianControlModel` arithmetic contract at
            ``concerto/safety/api.py:560-561``).

        Raises:
            ValueError: If ``qpos_arm`` shape mismatches ``(7,)``.
        """
        del snap  # closure consumer reads qpos from env; see module docstring.
        import torch

        q = np.asarray(qpos_arm, dtype=np.float32)
        if q.shape != (PANDA_ARM_DOF,):
            msg = (
                f"PandaJacobianProvider: qpos_arm shape {q.shape} mismatches expected "
                f"({PANDA_ARM_DOF},) for the 7-DOF Panda chain."
            )
            raise ValueError(msg)
        q_torch = torch.from_numpy(q).unsqueeze(0)
        # pk.SerialChain.jacobian returns (N_batch, 6, n_dof); rows 0:3
        # are the linear part (d_pos / d_q), rows 3:6 are angular.
        # The CBF backbone consumes the linear part only — orientation
        # is out of scope for the pairwise distance barrier
        # (ADR-004 §Decision). See module docstring.
        jac_6x7 = self._chain.jacobian(q_torch)[0]
        jac_3x7 = jac_6x7[:CARTESIAN_POSITION_DIM].detach().cpu().numpy()
        return np.ascontiguousarray(jac_3x7, dtype=np.float64)


__all__ = [
    "CARTESIAN_POSITION_DIM",
    "PANDA_ARM_DOF",
    "PandaJacobianProvider",
]
