# SPDX-License-Identifier: Apache-2.0
r"""Policy-shift co-carry teammates for the Rung-3 PH measurement (ADR-026 §Decision 4; ADR-009).

Rung 3 (R-2026-06-B §15) measures whether pairing the **frozen** Rung-2
co-carry incumbent with a *different-but-capability-matched* partner
**policy** degrades cooperation. This module ships the policy-shift
teammates: competent, hand-written controllers on the **same Panda body**
as the matched :class:`chamber.partners.cocarry_impedance.CoCarryImpedancePartner`
the incumbent trained against, but realising **materially different control
policies** — not reseeds of one controller (that would isolate nothing).
Embodiment is held fixed (the xArm6 teammate is Rung 4); this slice isolates
**policy** heterogeneity.

Each teammate drives the partner seat (``panda_partner``, the bar's -x end)
to the *same* cooperative bar-end target the matched controller computes
(``goal_centroid + end_sign * (bar_half_len, 0, 0)``, transformed into the
arm base frame), so they are all genuinely attempting the cooperative task —
they differ in **how** they get there:

- :class:`CoCarryStiffImpedancePartner` (``cocarry_stiff_impedance``) —
  a **stiffness** shift: high-gain, fast, low-damping pure-proportional
  tracking with no Cartesian integral. Drives hard and converges quickly;
  transmits more interaction force and overshoots more than the compliant
  matched controller.
- :class:`CoCarryAdmittancePartner` (``cocarry_admittance``) — a
  **compliance / admittance** shift: a force-follower that mostly holds
  station at its *current* bar end (read from ``obs["extra"]["bar_pose"]``)
  and contributes only a gentle goal-seeking bias, letting the partner arm
  lead the transport. The opposite control philosophy to driving an
  absolute target with high authority.
- :class:`CoCarrySlewImpedancePartner` (``cocarry_slew_impedance``) — a
  **trajectory-timing** shift: the matched impedance law, but the
  commanded target is slewed along a rate-limited ramp from the arm's
  start bar-end to the cooperative target over a fixed lead-in, rather than
  commanding the final target from t=0. Same destination, different timing.
- :class:`CoCarryNullspaceImpedancePartner` (``cocarry_nullspace_impedance``)
  — a **redundancy-resolution** shift: the same Cartesian primary task, but
  the 7-DOF redundancy is resolved with a null-space posture term pulling
  the arm toward a different nominal configuration (``N = I - J^+ J``). Same
  task-space goal, a different joint-space policy realising it.

All four route through the partner interface (ADR-009 §Decision): they
subclass :class:`chamber.partners.interface.PartnerBase`, so the AHT
freeze-shield (``_FORBIDDEN_ATTRS``) is inherited and
:meth:`chamber.benchmarks.ego_ppo_trainer.EgoPPOTrainer._assert_partner_is_frozen`
passes. They are **partner-agnostic / black-box** with respect to the
incumbent: they read only task obs (``obs["extra"]["goal_pos"]`` /
``["bar_pose"]`` and ``obs["agent"][uid]["qpos"]``), never the ego's
identity or internals (I3). Each is deterministic (the control law is a pure
function of obs + within-episode state cleared in :meth:`reset`), so two
loads emit byte-identical actions on identical obs (P6 / ADR-002).

Why a shared base: the four teammates share the matched controller's
geometry (base pose, bar-end target, damped-pseudoinverse joint step), so
the common machinery lives on :class:`_CoCarryShiftedBase` and each subclass
overrides only the policy hook(s) that make it distinct. This module does
**not** import or edit :mod:`chamber.partners.cocarry_impedance` (the matched
reference is part of the frozen Rung-2 manifest and must not move); the small
geometry helpers are re-implemented locally so the two modules stay
independent.

References:
- ADR-026 §Decision 4 (the Phase-2 forward-design ladder; Rung 3 = the
  policy-heterogeneity measurement against the frozen incumbent).
- ADR-009 §Decision (frozen black-box partner; partner reads task leaves
  only — the anti-leakage contract).
- ADR-004 §Decision (the damped-pseudoinverse joint step these mirror).
- R-2026-06-B §15 Rung 3 (the policy-shift teammate set + capability gate).
- :class:`chamber.partners.cocarry_impedance.CoCarryImpedancePartner` (the
  matched cooperative reference; the template these vary from).
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

#: Number of Panda arm DOF the Jacobian / FK chain spans.
_PANDA_ARM_DOF: int = 7

#: Number of xyz components.
_XYZ: int = 3

#: ``pd_joint_delta_pos`` per-step joint-delta scale (rad); a normalised
#: action ``a in [-1, 1]`` maps to a joint delta ``a * _ACTION_DELTA_SCALE``
#: (mani-skill==3.0.1 panda config ``lower=-0.1, upper=0.1``).
_ACTION_DELTA_SCALE: float = 0.1

#: Gripper action (normalised) held constant — open (the bar is welded by
#: the env's dual-hold attach, so the fingers are inert).
_GRIPPER_ACTION_OPEN: float = 1.0


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


class _CoCarryShiftedBase(PartnerBase):
    r"""Shared geometry + joint-step machinery for the policy-shift teammates (ADR-026 §D4).

    Holds the matched controller's spec parsing (base pose, bar-end sign,
    half-length), the Jacobian/FK provider, and the damped-pseudoinverse
    joint step. Subclasses override the policy hooks:

    - :meth:`_target_world` — the Cartesian target (world frame) the arm
      tracks this step (default: the cooperative bar-end target, shared by
      every teammate); a *timing* shift overrides this to slew it.
    - :meth:`_cartesian_command` — error → clipped Cartesian velocity
      command (default: high-clip proportional); a *stiffness* / *compliance*
      shift overrides the gains / the admittance law here.
    - :meth:`_joint_step` — Cartesian velocity → joint delta (default: the
      damped pseudoinverse); a *redundancy* shift overrides this to add a
      null-space posture term.

    ``spec.extra`` keys mirror the matched controller's (``uid`` / ``base_xyz``
    / ``base_yaw_deg`` / ``end_sign`` / ``bar_half_len``) so the Rung-3 runner
    builds every teammate from the env's single-source-of-truth partner-seat
    geometry (:func:`chamber.envs.cocarry.cocarry_matched_controller_specs`).
    """

    #: Subclass-overridable gains. Defaults match the matched controller's
    #: compliant regime so a subclass that changes nothing else is the
    #: matched law (used by the timing/redundancy variants).
    _KP: float = 2.5
    _STEP_MAX_M: float = 0.03
    _DAMPING: float = 0.04

    def __init__(self, spec: PartnerSpec) -> None:
        """Bind the spec and build the Jacobian/FK provider (ADR-026 §Decision 4)."""
        super().__init__(spec)
        from chamber.agents.panda_jacobian import PandaJacobianProvider

        extra = spec.extra
        self._uid: str = extra.get("uid", "panda_partner")
        self._base_xyz: NDArray[np.float64] = _parse_vec3(extra.get("base_xyz", "0,0,0"))
        self._base_rot: NDArray[np.float64] = _rot_z(float(extra.get("base_yaw_deg", "0")))
        self._base_rot_t: NDArray[np.float64] = self._base_rot.T
        self._end_sign: float = float(extra.get("end_sign", "-1"))
        self._bar_half_len: float = float(extra.get("bar_half_len", "0.115"))
        self._provider = PandaJacobianProvider(extra.get("urdf_path"))
        # Within-episode state (cleared in reset): step counter + cached
        # start bar-end (base frame), populated lazily on the first act.
        self._step: int = 0
        self._start_target_base: NDArray[np.float64] | None = None

    def reset(self, *, seed: int | None = None) -> None:
        """Clear within-episode state (ADR-009 §Decision; stateless across episodes)."""
        del seed
        self._step = 0
        self._start_target_base = None

    # ----- policy hooks (subclasses override) -----

    def _cooperative_target_world(self, goal: NDArray[np.float64]) -> NDArray[np.float64]:
        """The cooperative bar-end target (world frame), shared by every teammate."""
        return goal + np.asarray([self._end_sign * self._bar_half_len, 0.0, 0.0])

    def _target_world(
        self, goal: NDArray[np.float64], obs: Mapping[str, object]
    ) -> NDArray[np.float64]:
        """World-frame target the arm tracks this step (default: the cooperative target)."""
        del obs
        return self._cooperative_target_world(goal)

    def _cartesian_command(
        self, error: NDArray[np.float64], obs: Mapping[str, object]
    ) -> NDArray[np.float64]:
        """Cartesian velocity command from the base-frame error (default: clipped proportional)."""
        del obs
        return np.clip(self._KP * error, -self._STEP_MAX_M, self._STEP_MAX_M)

    def _joint_step(
        self, jac: NDArray[np.float64], v_cmd: NDArray[np.float64], qpos: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Joint delta from the Cartesian command (default: damped pseudoinverse)."""
        del qpos
        jjt = jac @ jac.T + (self._DAMPING**2) * np.eye(_XYZ)
        return jac.T @ np.linalg.solve(jjt, v_cmd)

    # ----- driver -----

    def act(
        self,
        obs: Mapping[str, object],
        *,
        deterministic: bool = True,
    ) -> NDArray[np.floating]:
        """Return the ``pd_joint_delta_pos`` action for this teammate (ADR-026 §Decision 4).

        Deterministic: a pure function of ``obs`` + the within-episode step
        counter. Emits a gripper-only (zero-arm) action if the obs lacks the
        required task leaves (defensive — keeps the rollout shape stable).
        """
        del deterministic
        goal = self._read_goal(obs)
        qpos = self._read_qpos(obs)
        action = np.zeros(_PANDA_ARM_DOF + 1, dtype=np.float32)
        action[_PANDA_ARM_DOF] = _GRIPPER_ACTION_OPEN
        if goal is None or qpos is None:
            return action
        target_world = self._target_world(goal, obs)
        target_base = self._base_rot_t @ (target_world - self._base_xyz)
        tcp_base = self._provider.fk_tcp_position(qpos)
        if self._start_target_base is None:
            self._start_target_base = tcp_base.copy()
        error = target_base - tcp_base
        v_cmd = self._cartesian_command(error, obs)
        jac = self._provider(None, qpos)  # type: ignore[arg-type]  # (3, 7), base frame
        dq = self._joint_step(jac, v_cmd, qpos)
        action[:_PANDA_ARM_DOF] = np.clip(dq / _ACTION_DELTA_SCALE, -1.0, 1.0).astype(np.float32)
        self._step += 1
        return action

    # ----- obs readers (task leaves only; ADR-009 anti-leakage) -----

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
        """Read the 7-D bar pose ``[xyz, wxyz]`` from ``obs["extra"]["bar_pose"]``."""
        extra = obs.get("extra")
        if not isinstance(extra, Mapping) or "bar_pose" not in extra:
            return None
        pose = _to_numpy_flat(extra["bar_pose"])
        if pose.shape[0] < _XYZ + 4:
            return None
        return pose[: _XYZ + 4]

    def _read_qpos(self, obs: Mapping[str, object]) -> NDArray[np.float64] | None:
        """Read the arm's 7 joint angles from ``obs["agent"][uid]["qpos"]``."""
        agent = obs.get("agent")
        if not isinstance(agent, Mapping) or self._uid not in agent:
            return None
        entry = agent[self._uid]
        if not isinstance(entry, Mapping) or "qpos" not in entry:
            return None
        qpos = _to_numpy_flat(entry["qpos"])
        if qpos.shape[0] < _PANDA_ARM_DOF:
            return None
        return qpos[:_PANDA_ARM_DOF]


# --------------------------------------------------------------------------
# 1. Stiffness shift.
# --------------------------------------------------------------------------

#: Stiff-impedance proportional gain (vs matched 2.5): drives hard.
_STIFF_KP: float = 6.0
#: Stiff-impedance per-step Cartesian clip, m (vs matched 0.03): fast.
_STIFF_STEP_MAX_M: float = 0.06
#: Stiff-impedance damping (vs matched 0.04): less suppression near the
#: target, so the high-gain command is realised more fully (stiffer).
_STIFF_DAMPING: float = 0.02


@register_partner("cocarry_stiff_impedance")
class CoCarryStiffImpedancePartner(_CoCarryShiftedBase):
    """High-gain, low-damping stiff impedance teammate (ADR-026 §Decision 4; ADR-009).

    A **stiffness** policy shift: pure high-gain proportional tracking of the
    cooperative bar-end target with no Cartesian integral, a larger per-step
    clip, and reduced damped-pseudoinverse damping. Drives to the target
    faster and stiffer than the compliant matched controller — it transmits
    more interaction force through the shared bar and overshoots more, so the
    transport / stress / level profile the frozen incumbent must cooperate
    with differs materially. Still a competent cooperative controller (it
    tracks the correct target), so it is a candidate for the capability gate.
    """

    _KP = _STIFF_KP
    _STEP_MAX_M = _STIFF_STEP_MAX_M
    _DAMPING = _STIFF_DAMPING


# --------------------------------------------------------------------------
# 2. Compliance / admittance shift.
# --------------------------------------------------------------------------

#: Admittance goal-seeking gain (low — the follower is led, not driving).
_ADMIT_KP: float = 1.2
#: Admittance per-step Cartesian clip, m (gentle).
_ADMIT_STEP_MAX_M: float = 0.025
#: Admittance damping (heavier — soft, compliant joint step).
_ADMIT_DAMPING: float = 0.08
#: Fraction of the way from the current bar end toward the cooperative
#: target the follower aims each step. <1 ⇒ it mostly holds station at the
#: bar and contributes a gentle goal bias, letting the partner arm lead.
_ADMIT_GOAL_BIAS: float = 0.35


@register_partner("cocarry_admittance")
class CoCarryAdmittancePartner(_CoCarryShiftedBase):
    r"""Compliant admittance / force-follower teammate (ADR-026 §Decision 4; ADR-009).

    A **compliance** policy shift — the opposite philosophy to driving an
    absolute target with high authority. The follower reads where its held
    bar end currently is (from ``obs["extra"]["bar_pose"]``) and aims for a
    point only a fraction :data:`_ADMIT_GOAL_BIAS` of the way from there
    toward the cooperative target, with a low gain and heavy joint-step
    damping. It therefore mostly *holds station with the bar* and contributes
    a gentle goal-seeking bias, yielding to the other arm and letting it lead
    the transport. With a competent cooperative reference on the other end it
    still reaches the goal (the capability gate checks this); against the
    frozen incumbent it presents a markedly more compliant, less-leading
    partner than the matched controller.

    Reads the bar pose (a task leaf the env exposes under ``obs["extra"]``),
    never the ego's identity or internals (ADR-009 anti-leakage).
    """

    _KP = _ADMIT_KP
    _STEP_MAX_M = _ADMIT_STEP_MAX_M
    _DAMPING = _ADMIT_DAMPING

    def _target_world(
        self, goal: NDArray[np.float64], obs: Mapping[str, object]
    ) -> NDArray[np.float64]:
        """Aim a fraction of the way from the current bar end toward the cooperative target.

        Falls back to the cooperative target if the bar pose is unavailable
        (so the follower stays well-defined on a degraded obs).
        """
        coop = self._cooperative_target_world(goal)
        bar = self._read_bar_pose(obs)
        if bar is None:
            return coop
        bar_center = bar[:_XYZ]
        bar_rot = _quat_wxyz_to_matrix(bar[_XYZ : _XYZ + 4])
        # The held bar end in world: bar center + R_bar @ (end_sign*half, 0, 0).
        held_end = bar_center + bar_rot @ np.asarray(
            [self._end_sign * self._bar_half_len, 0.0, 0.0]
        )
        return held_end + _ADMIT_GOAL_BIAS * (coop - held_end)


# --------------------------------------------------------------------------
# 3. Trajectory-timing shift.
# --------------------------------------------------------------------------

#: Lead-in over which the slewed target ramps from the start bar-end to the
#: cooperative target (env ticks). The matched controller commands the final
#: target from t=0; this teammate reaches the same place on a schedule.
_SLEW_LEAD_IN_STEPS: int = 140


@register_partner("cocarry_slew_impedance")
class CoCarrySlewImpedancePartner(_CoCarryShiftedBase):
    r"""Slew-rate-limited trajectory-timing teammate (ADR-026 §Decision 4; ADR-009).

    A **trajectory-timing** policy shift: the matched impedance gains, but the
    commanded target is slewed along a linear ramp from the arm's start
    bar-end (cached on the first :meth:`act` after reset, in the base frame)
    to the cooperative target over :data:`_SLEW_LEAD_IN_STEPS` ticks, holding
    the final target thereafter. Same destination and same compliant law as
    the matched controller — a different *timing* of the transport, so the
    frozen incumbent meets the goal-ward motion on a different schedule (the
    bar approaches the goal later, then settles). Competent by construction
    (it converges to the correct target), so it is a capability-gate
    candidate. Uses the base class's start-target cache (within-episode state
    cleared in :meth:`reset`).
    """

    def _target_world(
        self, goal: NDArray[np.float64], obs: Mapping[str, object]
    ) -> NDArray[np.float64]:
        """Slew the world target along a rate-limited ramp; hold the final target after lead-in.

        The ramp is computed in the world frame between the start bar-end
        (the arm's first-step TCP, lifted to world from the cached base-frame
        target) and the cooperative target. To keep the ramp purely a
        function of obs + the step counter, interpolate in the world target
        directly: at step 0 aim at the cooperative target scaled toward the
        current bar end, ramping to the full target.
        """
        coop = self._cooperative_target_world(goal)
        bar = self._read_bar_pose(obs)
        if bar is None:
            return coop
        bar_center = bar[:_XYZ]
        bar_rot = _quat_wxyz_to_matrix(bar[_XYZ : _XYZ + 4])
        held_end = bar_center + bar_rot @ np.asarray(
            [self._end_sign * self._bar_half_len, 0.0, 0.0]
        )
        # Cache the start bar-end (world) on the first step so the ramp has a
        # fixed origin even as the bar moves; reuse the base-class state slot.
        if self._start_target_base is None:
            self._start_target_base = held_end.copy()
        frac = min(1.0, (self._step + 1) / float(_SLEW_LEAD_IN_STEPS))
        return self._start_target_base + frac * (coop - self._start_target_base)


# --------------------------------------------------------------------------
# 4. Redundancy-resolution shift.
# --------------------------------------------------------------------------

#: Null-space posture gain — how hard the redundancy is pulled toward the
#: alternate nominal configuration (per step).
_NULLSPACE_GAIN: float = 0.4

#: Alternate nominal arm posture (7 joints) the null-space term pulls toward.
#: Distinct from the env ready pose — a different elbow/wrist configuration,
#: so the arm realises the same Cartesian task through a different joint path.
_NULLSPACE_NOMINAL_QPOS: NDArray[np.float64] = np.array(
    [0.4, np.pi / 6, -0.4, -np.pi * 5 / 8, 0.3, np.pi * 2 / 3, -np.pi / 6],
    dtype=np.float64,
)


@register_partner("cocarry_nullspace_impedance")
class CoCarryNullspaceImpedancePartner(_CoCarryShiftedBase):
    r"""Null-space redundancy-resolution teammate (ADR-026 §Decision 4; ADR-009).

    A **redundancy-resolution** policy shift: the same Cartesian primary task
    (the matched impedance command to the cooperative bar-end target) but the
    7-DOF redundancy is resolved differently. The joint step adds a
    null-space posture term ``dq = J^+ v + N (k_null * (q_nom - q))`` with the
    null-space projector ``N = I - J^+ J`` (from the 3x7 linear Jacobian and
    its damped pseudoinverse) pulling the arm toward an alternate nominal
    posture :data:`_NULLSPACE_NOMINAL_QPOS`. The end-effector follows the same
    cooperative target, but the arm reaches it through a different joint-space
    path, so its inertia / coupling-force signature through the shared bar
    differs from the matched controller's. Competent (the task-space behaviour
    is the matched law), so it is a capability-gate candidate.
    """

    def _joint_step(
        self, jac: NDArray[np.float64], v_cmd: NDArray[np.float64], qpos: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Damped pseudoinverse primary + null-space posture secondary task."""
        jjt = jac @ jac.T + (self._DAMPING**2) * np.eye(_XYZ)
        jac_pinv = jac.T @ np.linalg.inv(jjt)  # (7, 3) damped pseudoinverse
        dq_primary = jac_pinv @ v_cmd
        null_proj = np.eye(_PANDA_ARM_DOF) - jac_pinv @ jac  # (7, 7) null-space projector
        posture = _NULLSPACE_GAIN * (_NULLSPACE_NOMINAL_QPOS - qpos)
        return dq_primary + null_proj @ posture


#: The Rung-3 policy-shift candidate set (registry class names), in the
#: canonical order the calibration roster + pre-registration enumerate them.
COCARRY_POLICY_SHIFT_CANDIDATES: tuple[str, ...] = (
    "cocarry_stiff_impedance",
    "cocarry_admittance",
    "cocarry_slew_impedance",
    "cocarry_nullspace_impedance",
)


__all__ = [
    "COCARRY_POLICY_SHIFT_CANDIDATES",
    "CoCarryAdmittancePartner",
    "CoCarryNullspaceImpedancePartner",
    "CoCarrySlewImpedancePartner",
    "CoCarryStiffImpedancePartner",
]
