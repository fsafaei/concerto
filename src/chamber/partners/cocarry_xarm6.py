# SPDX-License-Identifier: Apache-2.0
r"""Compliant xArm6 co-carry teammate for the Rung-4 EH measurement (ADR-026 §D4; ADR-005; ADR-009).

Rung 4 (R-2026-06-B §15) swaps the Panda partner for a genuinely different
**body** — an xArm6 + Robotiq 2F-85 (6-DOF) — against the frozen Rung-2 Panda
incumbent, to measure whether *embodiment* heterogeneity degrades cooperation.
A different body forces a different controller (the Panda's 7-DOF
:class:`chamber.partners.cocarry_impedance.CoCarryImpedancePartner` cannot run
on a 6-DOF xArm6), which would confound "different body" with "different
control style". This controller defuses that confound **by construction**: it
is the *same compliant / matched-timing control-style family* the Panda matched
controller uses — the family Rung 3 showed costs ≈ 0 against the incumbent
(admitted PH teammates Δ ≈ 0; only the *stiff* teammate dropped). It is
deliberately **not** stiff (the EH measurement attributes a drop to embodiment
only relative to the Rung-3 PH control-style floor; making the xArm6 stiff
would manufacture a control-style drop, defeating the separation).

Control law (mirrors the matched Panda impedance, on the xArm6 chain):

1. Target ``eef`` (world): ``goal_centroid + end_sign * (bar_half_len, 0, 0)``
   — the arm's bar end when the bar centroid is at the goal.
2. Transform into the xArm6 base (``link_base``) frame via the fixed base pose
   (the env mounts the partner yawed pi about z).
3. Cartesian error ``e = target_base - fk_eef(qpos6)``; a deadzone-gated,
   anti-windup-clamped PI command ``v = clip(Kp e + Ki integral(e), +-step_max)``
   — the same compliant gains as the Panda matched controller.
4. Joint step via the damped (Nakamura & Hanafusa 1986) pseudo-inverse of the
   3x6 linear Jacobian: ``dq = J^T (J J^T + lambda^2 I)^-1 v``.
5. Action: 6 ``pd_joint_delta_pos`` arm deltas (rescaled to the +-0.1 rad
   normalised range) + 1 gripper delta held at 0 (the bar is welded; the
   Robotiq fingers are inert).

Realised on the xArm6 via
:class:`chamber.agents.xarm6_jacobian.XArm6JacobianProvider` (the 3x6 linear
Jacobian + FK to ``eef``). Routed through the partner interface (ADR-009): a
:class:`chamber.partners.interface.PartnerBase` subclass (black-box; the
``_FORBIDDEN_ATTRS`` shield is inherited), reading task leaves only
(``obs["extra"]["goal_pos"]`` + ``obs["agent"][uid]["qpos"]``), never the
ego's identity or internals (I3). Deterministic (a pure function of obs + a
within-episode Cartesian integral cleared in :meth:`reset`), so two loads emit
byte-identical actions on identical obs (P6 / ADR-002).

References:
- ADR-026 §Decision 4 (the Phase-2 ladder; Rung 4 = embodiment heterogeneity).
- ADR-005 §Decision (the xArm6 + Robotiq 2F-85 ManiSkill agent).
- ADR-009 §Decision (frozen black-box partner; task-leaf-only obs).
- ADR-004 §Decision (the damped-pseudoinverse joint step this mirrors).
- R-2026-06-B §15 Rung 4 (the embodiment-shift teammate + the EH-vs-control-style
  separation this control-style choice implements).
- :class:`chamber.partners.cocarry_impedance.CoCarryImpedancePartner` (the
  Panda matched controller this is the xArm6 analogue of, same gains/family).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np

from chamber.partners.interface import PartnerBase
from chamber.partners.registry import register_partner

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray

    from chamber.partners.api import PartnerSpec

#: Number of xArm6 arm DOF (the Jacobian/FK chain span; excludes the gripper).
_XARM6_ARM_DOF: int = 6

#: xArm6 action width: 6 arm joint deltas + 1 Robotiq mimic-gripper delta.
_XARM6_ACTION_DIM: int = 7

#: Number of xyz components.
_XYZ: int = 3

#: ``pd_joint_delta_pos`` per-step arm joint-delta scale (rad); the xArm6 arm
#: sub-controller maps a normalised action ``a in [-1, 1]`` to a delta
#: ``a * _ACTION_DELTA_SCALE`` (mani-skill==3.0.1 xarm6 ``lower=-0.1, upper=0.1``).
_ACTION_DELTA_SCALE: float = 0.1

#: Gripper action (normalised) — 0 = hold (no delta). The bar is welded to the
#: Robotiq gripper by the env's dual-hold attach, so the fingers are inert.
_GRIPPER_ACTION_HOLD: float = 0.0

# Compliant / matched-timing gains — identical to the Panda matched controller
# (chamber.partners.cocarry_impedance), so the xArm6 teammate's control style
# sits in the Rung-3 zero-drop band (R-2026-06-B §15 Rung 4). NOT stiff.

#: Cartesian proportional gain (matched-family compliant value).
_DEFAULT_KP: float = 2.5
#: Per-step Cartesian command clip, metres (matched-family).
_DEFAULT_STEP_MAX_M: float = 0.03
#: Damped pseudo-inverse damping (lambda), Jacobian length units (matched-family).
_DEFAULT_DAMPING: float = 0.04
#: Cartesian integral gain per second (closes the delta-controller load offset).
_DEFAULT_KI: float = 0.6
#: Control timestep (s) integrating the Cartesian error (20 Hz control rate).
_CONTROL_DT: float = 0.05
#: Anti-windup clamp on the accumulated Cartesian integral (metre*s per axis).
_INTEGRAL_CLAMP_MS: float = 0.6
#: Cartesian-error magnitude (m) below which the integral engages (deadzone).
_INTEGRAL_ENGAGE_M: float = 0.12

#: Leveling gain on the bar-end height gap (dimensionless). The xArm6 drives
#: its z target to goal_z PLUS this times the height difference to the other
#: bar end, so it slows when ahead of / catches up to the ego's end — nulling
#: the bar tilt the embodiment asymmetry creates while still placing the bar.
#: 1.0 fully mirrors the gap; the matched Panda pair needs none (symmetry).
_LEVEL_GAIN: float = 1.0


def _parse_vec3(raw: str) -> NDArray[np.float64]:
    """Parse an ``"x,y,z"`` string into a float64 ``(3,)`` array."""
    parts = raw.split(",")
    if len(parts) != _XYZ:
        raise ValueError(f"expected 'x,y,z' with three floats; got {raw!r}")
    try:
        return np.asarray([float(p) for p in parts], dtype=np.float64)
    except ValueError as exc:
        raise ValueError(f"vec3 components must be floats; got {raw!r}") from exc


def _rot_z(yaw_deg: float) -> NDArray[np.float64]:
    """3x3 rotation matrix about world z for ``yaw_deg`` degrees."""
    a = np.radians(yaw_deg)
    c, s = np.cos(a), np.sin(a)
    return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _quat_wxyz_to_matrix(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """3x3 rotation matrix from a ``(w, x, y, z)`` unit quaternion (defensive renormalise)."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    n = float(np.sqrt(w * w + x * x + y * y + z * z))
    if n <= 0.0:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _to_numpy_flat(value: object) -> NDArray[np.float64]:
    """Coerce a torch tensor / ndarray / sequence to a flat float64 array (env 0)."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()  # type: ignore[union-attr]
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim >= 2:  # noqa: PLR2004 - rank-2 is the (num_envs, dim) batched layout
        arr = arr[0]
    return arr.reshape(-1)


@register_partner("cocarry_xarm6_impedance")
class CoCarryXArm6Partner(PartnerBase):
    """Compliant EE-space xArm6 co-carry teammate (ADR-026 §Decision 4; ADR-005; ADR-009).

    Reads ``obs["extra"]["goal_pos"]`` and ``obs["agent"][uid]["qpos"]`` (the
    xArm6's 6 arm joints), computes the cooperative ``eef`` target, and returns
    the 7-D ``pd_joint_delta_pos`` action stepping the eef toward it via the
    damped-pseudoinverse 3x6 Jacobian. Compliant / matched-timing gains (the
    Rung-3 zero-drop control-style family) — NOT stiff.

    ``spec.extra`` keys (mirroring the matched controller's geometry):

    - ``uid`` — the arm uid (default ``"xarm6_robotiq"``).
    - ``base_xyz`` — the arm base position ``"x,y,z"`` (world).
    - ``base_yaw_deg`` — base yaw about z, degrees (``"180"`` partner seat).
    - ``end_sign`` — ``"-1"`` (the xArm6 holds the bar's ``-x`` end).
    - ``bar_half_len`` — half the bar length, metres.
    - ``urdf_path`` / ``kp`` / ``ki`` / ``step_max`` / ``damping`` — optional overrides.
    """

    def __init__(self, spec: PartnerSpec) -> None:
        """Bind the spec and build the xArm6 Jacobian/FK provider (ADR-026 §Decision 4)."""
        super().__init__(spec)
        from chamber.agents.xarm6_jacobian import XArm6JacobianProvider

        extra = spec.extra
        self._uid: str = extra.get("uid", "xarm6_robotiq")
        self._base_xyz: NDArray[np.float64] = _parse_vec3(extra.get("base_xyz", "0.5,0,0"))
        self._base_rot_t: NDArray[np.float64] = _rot_z(float(extra.get("base_yaw_deg", "180"))).T
        self._end_sign: float = float(extra.get("end_sign", "-1"))
        self._bar_half_len: float = float(extra.get("bar_half_len", "0.115"))
        self._kp: float = float(extra.get("kp", str(_DEFAULT_KP)))
        self._ki: float = float(extra.get("ki", str(_DEFAULT_KI)))
        self._step_max: float = float(extra.get("step_max", str(_DEFAULT_STEP_MAX_M)))
        self._damping: float = float(extra.get("damping", str(_DEFAULT_DAMPING)))
        self._level_gain: float = float(extra.get("level_gain", str(_LEVEL_GAIN)))
        # Admittance x,y bias in [0, 1]: 1.0 = drive the full goal end (impedance);
        # <1 = aim only a fraction of the way from the current held end toward
        # the goal (a compliant follower that yields to the ego and lets it lead
        # the transport, minimising the internal fight). Rung-4 tuning lever.
        self._admit_bias: float = float(extra.get("admit_bias", "1.0"))
        self._provider = XArm6JacobianProvider(extra.get("urdf_path"))
        self._integral: NDArray[np.float64] = np.zeros(_XYZ, dtype=np.float64)

    def reset(self, *, seed: int | None = None) -> None:
        """Clear the within-episode Cartesian integral (ADR-009 §Decision; stateless across eps)."""
        del seed
        self._integral = np.zeros(_XYZ, dtype=np.float64)

    def act(
        self,
        obs: Mapping[str, object],
        *,
        deterministic: bool = True,
    ) -> NDArray[np.floating]:
        """Return the 7-D ``pd_joint_delta_pos`` xArm6 action toward the target (ADR-026 §D4).

        Args:
            obs: Gymnasium obs with ``obs["extra"]["goal_pos"]`` (world goal
                centroid) and ``obs["agent"][uid]["qpos"]`` (xArm6 joints).
            deterministic: Ignored — the controller is deterministic.

        Returns:
            Float32 action of length :data:`_XARM6_ACTION_DIM` (6 arm deltas +
            gripper-hold); a gripper-only action if the obs lacks the required
            keys (defensive — keeps the rollout shape stable).
        """
        del deterministic
        goal = self._read_goal(obs)
        qpos = self._read_qpos(obs)
        action = np.zeros(_XARM6_ACTION_DIM, dtype=np.float32)
        action[_XARM6_ARM_DOF] = _GRIPPER_ACTION_HOLD
        if goal is None or qpos is None:
            return action
        # Cooperative target: the xArm6's bar end at the goal in (x, y). For z,
        # the controller LEVELS — it tracks the OTHER bar end's current height
        # (read from obs["extra"]["bar_pose"]) instead of commanding goal_z
        # directly. The Panda matched pair stays level by symmetry (identical
        # arms rise in lockstep); a different body (xArm6) breaks that symmetry
        # and would tilt the bar, so it explicitly follows the other end's
        # height — the cooperative leveling the matched pair gets for free
        # (ADR-026 §D4; R-2026-06-B §15 Rung 4). Whoever leads (the cooperative
        # ego in calibration, or the frozen incumbent in measurement), the
        # xArm6 matches its height. Falls back to goal_z if the bar pose is
        # unavailable.
        coop_end = goal + np.asarray([self._end_sign * self._bar_half_len, 0.0, 0.0])
        target_world = coop_end.copy()
        bar = self._read_bar_pose(obs)
        if bar is not None:
            bar_center = bar[:_XYZ]
            bar_rot = _quat_wxyz_to_matrix(bar[_XYZ : _XYZ + 4])
            my_end = bar_center + bar_rot @ np.asarray(
                [self._end_sign * self._bar_half_len, 0.0, 0.0]
            )
            other_end = bar_center + bar_rot @ np.asarray(
                [-self._end_sign * self._bar_half_len, 0.0, 0.0]
            )
            # Admittance x,y: aim a fraction (``admit_bias``) of the way from
            # the current held end toward the cooperative goal end, so the
            # xArm6 yields to the ego and lets it lead the transport (minimising
            # the internal fight that a stiff absolute-target drive creates
            # against a different-bodied partner). With admit_bias=1 this is the
            # full goal end (pure impedance).
            target_world[:2] = my_end[:2] + self._admit_bias * (coop_end[:2] - my_end[:2])
            # Balanced leveling on z: goal_z plus a correction for the height
            # gap to the OTHER end, so the xArm6 slows when ahead of / catches
            # up to the ego's end — nulling the bar tilt the embodiment
            # asymmetry creates while still contributing the lift.
            target_world[2] = goal[2] + self._level_gain * (other_end[2] - my_end[2])
        target_base = self._base_rot_t @ (target_world - self._base_xyz)
        tcp_base = self._provider.fk_tcp_position(qpos)
        error = target_base - tcp_base
        if float(np.linalg.norm(error)) < _INTEGRAL_ENGAGE_M:
            self._integral = np.clip(
                self._integral + error * _CONTROL_DT, -_INTEGRAL_CLAMP_MS, _INTEGRAL_CLAMP_MS
            )
        v_cmd = np.clip(
            self._kp * error + self._ki * self._integral, -self._step_max, self._step_max
        )
        jac = self._provider.jacobian(qpos)  # (3, 6), base frame
        jjt = jac @ jac.T + (self._damping**2) * np.eye(_XYZ)
        dq = jac.T @ np.linalg.solve(jjt, v_cmd)
        action[:_XARM6_ARM_DOF] = np.clip(dq / _ACTION_DELTA_SCALE, -1.0, 1.0).astype(np.float32)
        return action

    def _read_goal(self, obs: Mapping[str, object]) -> NDArray[np.float64] | None:
        """Read the world goal centroid from ``obs["extra"]["goal_pos"]``."""
        extra = obs.get("extra")
        if not isinstance(extra, Mapping) or "goal_pos" not in extra:
            return None
        goal = _to_numpy_flat(extra["goal_pos"])
        if goal.shape[0] < _XYZ:
            return None
        return goal[:_XYZ]

    def _read_bar_pose(self, obs: Mapping[str, object]) -> NDArray[np.float64] | None:
        """Read the 7-D bar pose ``[xyz, wxyz]`` from ``obs["extra"]["bar_pose"]`` (leveling)."""
        extra = obs.get("extra")
        if not isinstance(extra, Mapping) or "bar_pose" not in extra:
            return None
        pose = _to_numpy_flat(extra["bar_pose"])
        if pose.shape[0] < _XYZ + 4:
            return None
        return pose[: _XYZ + 4]

    def _read_qpos(self, obs: Mapping[str, object]) -> NDArray[np.float64] | None:
        """Read the xArm6's 6 arm joint angles from ``obs["agent"][uid]["qpos"]``."""
        agent = obs.get("agent")
        if not isinstance(agent, Mapping) or self._uid not in agent:
            return None
        entry = agent[self._uid]
        if not isinstance(entry, Mapping) or "qpos" not in entry:
            return None
        qpos = _to_numpy_flat(entry["qpos"])
        if qpos.shape[0] < _XARM6_ARM_DOF:
            return None
        return qpos[:_XARM6_ARM_DOF]


__all__ = ["CoCarryXArm6Partner"]
