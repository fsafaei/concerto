# SPDX-License-Identifier: Apache-2.0
r"""Hand-written matched co-carry controller (ADR-026 §Decision 1; ADR-009 §Decision).

A competent, deterministic end-effector-space impedance / trajectory
controller for one arm of the co-carry task (:mod:`chamber.envs.cocarry`).
It drives its held bar-end TCP along a straight Cartesian path to the
cooperative target — the goal centroid offset by half the bar length along
the bar axis — so that, when **both** arms run a copy, the shared bar's
centroid reaches the goal while the bar stays level (both ends tracked to
the same goal height).

Why this is the *matched* reference (R-2026-06-B Rungs 0-1): both arms run
an identical copy of this controller on the identical Panda body. It is
the honest high reference against which Rung-3 policy-shift / Rung-4
embodiment-shift teammates will later be measured. There is no learning
here.

Routed through the partner interface (ADR-009 §Decision): even though the
matched pair is two hand-written controllers, the partner arm is driven
through :class:`chamber.partners.interface.PartnerBase` /
:class:`chamber.partners.api.FrozenPartner` (``reset`` / ``act``) so the
rig is forward-compatible with the Rung-2 frozen-incumbent design — the
incumbent will swap in at the partner seat with no rig change. The base
class's ``_FORBIDDEN_ATTRS`` shield (no ``train`` / ``update`` / ...) is
inherited unchanged.

Control law (per arm, per control step ``dt``):

1. Target TCP (world): ``goal_centroid + end_sign * (bar_half_len, 0, 0)``
   — the arm's end of the bar when the bar centroid is at the goal,
   oriented along world x.
2. Transform the target into the arm's base (``panda_link0``) frame using
   the arm's fixed base pose (the env mounts the partner yawed pi about
   z), so the Cartesian error and the analytic Jacobian share a frame.
3. Cartesian error ``e = target_base - fk_tcp(qpos)``; a PI command
   ``v = clip(Kp e + Ki integral(e), +-step_max)`` with a deadzone-gated,
   anti-windup-clamped integral.
4. Joint step via the damped (Nakamura & Hanafusa 1986) pseudo-inverse of
   the 3x7 linear Jacobian: ``dq = J^T (J J^T + lambda^2 I)^-1 v``.
5. Action: ``dq`` rescaled to the ``pd_joint_delta_pos`` normalised range
   (``+-0.1`` rad per step => action in ``[-1, 1]``), gripper held open.

Why the *compliant* ``pd_joint_delta_pos`` (joint-delta) controller
rather than a stiff absolute one: the bar is welded to both grippers, so
the two arms are mechanically coupled. A stiff absolute joint-position
controller makes the two arms fight through the bar (large internal
constraint force, the bar tilts/flings); the joint-delta controller is
compliant, so the arms share the load gently — low wrist stress, the bar
stays level. The delta controller's one weakness, a load-induced Cartesian
steady-state offset (the per-step torque balances gravity a few cm short),
is closed by the deadzone-gated Cartesian integral, which engages only
near the target to avoid transient windup. The integral is within-episode
state, cleared in :meth:`reset` (ADR-009 §Decision — stateless across
episodes).

The controller realises the arm on the Panda via
:class:`chamber.agents.panda_jacobian.PandaJacobianProvider` (the 3x7
linear end-effector Jacobian + the companion FK). Single-env (num_envs=1)
is the Rung-0/1 regime; the controller reads env 0.

References:
- ADR-026 §Decision 1 (the coupling-valid co-carry task this controller
  solves with a matched pair).
- ADR-009 §Decision (the frozen black-box partner contract; the partner
  seat is interface-routed for Rung-2 forward-compatibility).
- ADR-004 §Decision (the JacobianControlModel / damped-pseudoinverse
  convention this controller's joint step mirrors).
- ADR-005 §Decision (ManiSkill v3 / SAPIEN 3 substrate).
- R-2026-06-B Rungs 0-1 (the matched-reference + positive-control design).
- :class:`chamber.partners.heuristic.ScriptedHeuristicPartner` (the
  registered-hand-written-partner pattern this mirrors).
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

#: ``pd_joint_delta_pos`` per-step joint-delta scale (rad). The panda arm
#: controller maps a normalised action ``a in [-1, 1]`` to a joint delta
#: ``a * _ACTION_DELTA_SCALE``; the controller inverts this to emit a
#: normalised action from a desired joint step (mani-skill==3.0.1 panda
#: ``pd_joint_delta_pos`` config: ``lower=-0.1, upper=0.1``).
_ACTION_DELTA_SCALE: float = 0.1

#: Default Cartesian proportional gain (per control step, dimensionless on
#: the metre error). With the per-step clip below this gives a smooth
#: straight-line approach that saturates far from target and tapers near it.
_DEFAULT_KP: float = 2.5

#: Default per-step Cartesian command clip, in metres. 0.03 m/step at the
#: 20 Hz control rate keeps the quasi-static carry smooth while letting the
#: lift converge inside the episode horizon; the dual-hold attach stays
#: low-stress at this rate.
_DEFAULT_STEP_MAX_M: float = 0.03

#: Default damping factor (lambda) for the damped pseudo-inverse, in the
#: Jacobian's length units. 0.04 keeps joint steps bounded near the
#: workspace boundary / kinematic singularities (Nakamura & Hanafusa 1986)
#: without over-suppressing the vertical lift.
_DEFAULT_DAMPING: float = 0.04

#: Default Cartesian integral gain (per second). The ``pd_joint_delta_pos``
#: controller commands a delta from the *measured* joint angles, so under
#: the bar load a purely proportional law settles a few cm short (the pd
#: torque balances gravity at a non-zero Cartesian offset). A small,
#: anti-windup-clamped integral term drives that steady-state offset to
#: zero so the matched pair actually places the bar.
_DEFAULT_KI: float = 0.6

#: Control timestep (s) used to integrate the Cartesian error
#: (mani-skill==3.0.1 default ``control_timestep`` at the env's 20 Hz
#: control rate). Constant here to keep the controller env-agnostic.
_CONTROL_DT: float = 0.05

#: Anti-windup clamp on the accumulated Cartesian integral, in metre*s
#: per axis.
_INTEGRAL_CLAMP_MS: float = 0.6

#: Cartesian-error magnitude (m) below which the integral engages. Gating
#: the integral off during the large-error approach prevents transient
#: windup (which otherwise overshoots, desynchronises the two arms, and
#: tilts the shared bar).
_INTEGRAL_ENGAGE_M: float = 0.12

#: Gripper action (normalised) held constant — open. The bar is welded to
#: the gripper by the env's dual-hold attach, so the fingers are inert;
#: holding them open avoids any incidental self-collision.
_GRIPPER_ACTION_OPEN: float = 1.0

#: Upper bound on the bounded-lag buffer depth (control steps). A lag deeper
#: than this is no longer "sluggish but competent" at the 20 Hz control rate
#: (>0.5 s of pure transport delay destabilises the coupled carry), so the
#: parameterized scripted family (ADR-009 §Decision, v1.0 right-sizing
#: amendment 2026-07-05) never requests it and the constructor loud-fails.
_LAG_STEPS_MAX: int = 10


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


def _to_numpy_flat(value: object) -> NDArray[np.float64]:
    """Coerce a torch tensor / ndarray / sequence to a flat float64 array (env 0)."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()  # type: ignore[union-attr]
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim >= 2:  # noqa: PLR2004 - rank-2 is the (num_envs, dim) batched layout
        # (num_envs, dim) batched obs leaf — Rung-0/1 single-env: take row 0.
        arr = arr[0]
    return arr.reshape(-1)


@register_partner("cocarry_impedance")
class CoCarryImpedancePartner(PartnerBase):
    """EE-space impedance controller for one co-carry arm (ADR-026 §Decision 1; ADR-009 §Decision).

    Reads ``obs["extra"]["goal_pos"]`` (the world goal centroid) and
    ``obs["agent"][uid]["qpos"]`` (the arm's joint angles), computes the
    cooperative TCP target, and returns the ``pd_joint_delta_pos`` action
    that steps the TCP toward it via the damped-pseudoinverse Jacobian.

    ``spec.extra`` keys:

    - ``uid`` — the arm this controller drives (``panda_wristcam`` /
      ``panda_partner``).
    - ``base_xyz`` — the arm's base position ``"x,y,z"`` (world).
    - ``base_yaw_deg`` — the arm's base yaw about z, degrees
      (``"0"`` ego, ``"180"`` partner).
    - ``end_sign`` — ``"+1"`` if the arm holds the bar's ``+x`` end,
      ``"-1"`` for the ``-x`` end.
    - ``bar_half_len`` — half the bar length, metres.
    - ``urdf_path`` — optional; defaults to the wristcam URDF
      (``panda_partner`` shares the kinematic chain, so the default is
      correct for both).
    - ``kp`` / ``ki`` / ``step_max`` / ``damping`` — optional gain overrides.
    - ``lag_steps`` — optional bounded transport lag in control steps
      (default ``"0"`` = the historical byte-identical path). A positive
      lag makes the controller emit the arm command it computed
      ``lag_steps`` steps ago (holding still during the warm-up), i.e. a
      "sluggish but competent" member: delayed but convergent tracking,
      since the target is quasi-static and the PI law keeps integrating
      on the fresh error (ADR-009 §Decision, v1.0 right-sizing amendment
      2026-07-05 — the bounded-lag scripted-family member).
    """

    def __init__(self, spec: PartnerSpec) -> None:
        """Bind the spec and build the Jacobian/FK provider (ADR-026 §Decision 1)."""
        super().__init__(spec)
        from chamber.agents.panda_jacobian import PandaJacobianProvider

        extra = spec.extra
        self._uid: str = extra.get("uid", "panda_wristcam")
        self._base_xyz: NDArray[np.float64] = _parse_vec3(extra.get("base_xyz", "0,0,0"))
        self._base_rot_t: NDArray[np.float64] = _rot_z(float(extra.get("base_yaw_deg", "0"))).T
        self._end_sign: float = float(extra.get("end_sign", "1"))
        self._bar_half_len: float = float(extra.get("bar_half_len", "0.115"))
        self._kp: float = float(extra.get("kp", str(_DEFAULT_KP)))
        self._ki: float = float(extra.get("ki", str(_DEFAULT_KI)))
        self._step_max: float = float(extra.get("step_max", str(_DEFAULT_STEP_MAX_M)))
        self._damping: float = float(extra.get("damping", str(_DEFAULT_DAMPING)))
        lag_raw = extra.get("lag_steps", "0")
        try:
            self._lag_steps: int = int(lag_raw)
        except ValueError as exc:
            raise ValueError(f"lag_steps must be an int in [0, {_LAG_STEPS_MAX}]") from exc
        if not 0 <= self._lag_steps <= _LAG_STEPS_MAX:
            raise ValueError(
                f"lag_steps must be an int in [0, {_LAG_STEPS_MAX}]; got {self._lag_steps}"
            )
        urdf_path = extra.get("urdf_path")
        self._provider = PandaJacobianProvider(urdf_path)
        # Within-episode Cartesian integral (base frame); cleared in reset.
        self._integral: NDArray[np.float64] = np.zeros(_XYZ, dtype=np.float64)
        # Within-episode bounded-lag buffer of past arm commands; cleared in
        # reset. Empty list when lag_steps == 0 (the byte-identical path).
        self._lag_buffer: list[NDArray[np.float32]] = []

    def reset(self, *, seed: int | None = None) -> None:
        """Clear the within-episode integral + lag buffer (ADR-009 §Decision; stateless).

        The control law is a deterministic function of ``obs`` + ``spec``
        plus within-episode state (the Cartesian integral and the
        bounded-lag command buffer) that is zeroed here, so the partner
        carries no state across episodes (ADR-009 §Decision; plan/04 §2).
        The seed is accepted only for
        :class:`~chamber.partners.api.FrozenPartner` Protocol conformance.
        """
        del seed
        self._integral = np.zeros(_XYZ, dtype=np.float64)
        self._lag_buffer = []

    def act(
        self,
        obs: Mapping[str, object],
        *,
        deterministic: bool = True,
    ) -> NDArray[np.floating]:
        """Return the ``pd_joint_delta_pos`` action toward the target (ADR-026 §Decision 1).

        Args:
            obs: Gymnasium observation mapping with ``obs["extra"]
                ["goal_pos"]`` (world goal centroid) and
                ``obs["agent"][uid]["qpos"]`` (the arm's joint angles).
            deterministic: Ignored — the controller is deterministic.

        Returns:
            Float32 action of length ``_PANDA_ARM_DOF + 1`` (7 normalised
            joint deltas + gripper-open); a gripper-only action if the obs
            lacks the required keys (defensive — keeps the rollout shape
            stable).
        """
        del deterministic
        goal = self._read_goal(obs)
        qpos = self._read_qpos(obs)
        action = np.zeros(_PANDA_ARM_DOF + 1, dtype=np.float32)
        action[_PANDA_ARM_DOF] = _GRIPPER_ACTION_OPEN
        if goal is None or qpos is None:
            return action
        # Cooperative TCP target (world) -> arm base frame.
        target_world = goal + np.asarray([self._end_sign * self._bar_half_len, 0.0, 0.0])
        target_base = self._base_rot_t @ (target_world - self._base_xyz)
        tcp_base = self._provider.fk_tcp_position(qpos)
        error = target_base - tcp_base
        # PI law: the deadzone-gated, anti-windup-clamped integral closes
        # the load-induced steady-state offset of the delta-from-measured
        # joint controller, engaging only once the TCP is near its target.
        if float(np.linalg.norm(error)) < _INTEGRAL_ENGAGE_M:
            self._integral = np.clip(
                self._integral + error * _CONTROL_DT, -_INTEGRAL_CLAMP_MS, _INTEGRAL_CLAMP_MS
            )
        v_cmd = np.clip(
            self._kp * error + self._ki * self._integral, -self._step_max, self._step_max
        )
        jac = self._provider(None, qpos)  # type: ignore[arg-type]  # (3, 7), base frame
        # Damped pseudo-inverse joint step: dq = J^T (J J^T + l^2 I)^-1 v.
        jjt = jac @ jac.T + (self._damping**2) * np.eye(_XYZ)
        dq = jac.T @ np.linalg.solve(jjt, v_cmd)
        action[:_PANDA_ARM_DOF] = np.clip(dq / _ACTION_DELTA_SCALE, -1.0, 1.0).astype(np.float32)
        if self._lag_steps == 0:
            return action
        # Bounded lag (ADR-009 §Decision, 2026-07-05 amendment): emit the arm
        # command computed lag_steps steps ago, holding still through the
        # warm-up. Convergent because the target is quasi-static and the PI
        # law keeps integrating on the fresh error.
        self._lag_buffer.append(action[:_PANDA_ARM_DOF].copy())
        delayed = np.zeros(_PANDA_ARM_DOF + 1, dtype=np.float32)
        delayed[_PANDA_ARM_DOF] = _GRIPPER_ACTION_OPEN
        if len(self._lag_buffer) > self._lag_steps:
            delayed[:_PANDA_ARM_DOF] = self._lag_buffer.pop(0)
        return delayed

    def _read_goal(self, obs: Mapping[str, object]) -> NDArray[np.float64] | None:
        """Read the world goal centroid from ``obs["extra"]["goal_pos"]``."""
        extra = obs.get("extra")
        if not isinstance(extra, Mapping) or "goal_pos" not in extra:
            return None
        goal = _to_numpy_flat(extra["goal_pos"])
        if goal.shape[0] < _XYZ:
            return None
        return goal[:_XYZ]

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


__all__ = ["CoCarryImpedancePartner"]
