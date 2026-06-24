# SPDX-License-Identifier: Apache-2.0
r"""Hand-written co-insert controllers — structured base inserter + cooperative holder.

The S2 instrument pair for the contact-rich hold-and-insert task
(:mod:`chamber.envs.coinsert`), the contact-rich generalisation of the co-carry
end-effector-space impedance controller
(:class:`chamber.partners.cocarry_impedance.CoCarryImpedancePartner`):

- :class:`CoInsertBaseInserter` — the **structured base** (the ``base_policy`` +
  ``impedance_overlay`` term of the ADR-026 ego decomposition; **no learned
  residual**). It drives the ego-held peg through a vertical top-down insertion:
  an end-effector-space impedance approach that aligns the peg over the socket
  mouth, then a lead-in / spiral-search press that seats the peg into the blind
  socket. This is the Gate-0 instrument (S3) and the robust comparator the
  learned residual (S5) is measured against.
- :class:`CoInsertReferenceHolder` — the **cooperative reference holder** (the
  co-designed calibration anchor; the ``cocarry_impedance`` analogue for the
  holder seat). It tracks the **free** receptacle so the socket mouth stays
  under the descending peg and yields compliantly to the insertion reaction
  force, so that — paired with the base — the matched pair seats the peg with
  high reliability (the honest high reference; the Δ_min dynamic-range anchor).

Both controllers are **closed-loop on the observed workpiece poses**
(``obs["extra"]["peg_pose"]`` / ``obs["extra"]["receptacle_pose"]``), not on a
hard-coded geometry, so they are robust to the exact ready-pose / weld
calibration the env freezes at S3. The peg is rigidly welded to the ego gripper
and the socket to the holder gripper (the deliberate co-carry attach
simplification — failures reflect coordination, not fingertip friction), so a
Cartesian translation error on the held body maps directly to the gripper TCP;
each controller therefore drives its arm's TCP by the **measured** body-pose
error via the damped (Nakamura & Hanafusa 1986) pseudo-inverse of the 3x7 linear
end-effector Jacobian (:class:`chamber.agents.panda_jacobian.PandaJacobianProvider`),
exactly as the co-carry matched controller does.

Routed through the partner interface (ADR-009 §Decision): both controllers
subclass :class:`chamber.partners.interface.PartnerBase`, so the
``_FORBIDDEN_ATTRS`` shield is inherited unchanged and the holder enters the
measurement rig through the same :class:`chamber.partners.api.FrozenPartner`
Protocol every teammate uses (``reset`` / ``act`` only). The base inserter drives
the ego seat through the identical interface (the co-carry
``build_cooperative_ego`` precedent), so the ego and partner seats are wired the
same way.

Per-episode statelessness (ADR-009 §Decision; P6 / ADR-002): the only state
either controller carries is a within-episode Cartesian integral and a
within-episode step counter (the spiral-search phase clock); BOTH are zeroed in
:meth:`reset`. The reset is **asserted** (a stateful-control bug — an integrator
that leaks across episodes — is the co-carry regression this guards against;
``tests/property`` pins it).

References:
- ADR-026 §Decision 1-4 (the coupling-valid co-insert task; the structured-base
  decomposition; the cooperative-reference calibration anchor; non-gating scope).
- ADR-009 §Decision (the frozen black-box partner contract; interface-routed
  seats; stateless across episodes).
- ADR-004 §Decision (the JacobianControlModel / damped-pseudoinverse convention
  the joint step mirrors).
- ADR-005 §Decision (ManiSkill v3 / SAPIEN 3 substrate).
- ADR-002 (determinism — the controllers are deterministic functions of obs +
  spec + the zeroed within-episode integral / counter).
- :class:`chamber.partners.cocarry_impedance.CoCarryImpedancePartner` (the
  co-carry matched controller this generalises to contact-rich insertion).
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

#: ``pd_joint_delta_pos`` per-step joint-delta scale (rad) — the panda arm
#: controller maps a normalised action ``a in [-1, 1]`` to a joint delta
#: ``a * _ACTION_DELTA_SCALE`` (mani-skill==3.0.1 panda config: ``±0.1`` rad).
#: The controllers invert this to emit a normalised action from a desired step.
_ACTION_DELTA_SCALE: float = 0.1

#: Control timestep (s) used to integrate the Cartesian error (the env's 20 Hz
#: control rate; mani-skill==3.0.1 default ``control_timestep``).
_CONTROL_DT: float = 0.05

#: Gripper action (normalised) held constant — the peg / socket is welded to the
#: gripper by the env's dual-hold attach, so the fingers are inert; a constant
#: value avoids incidental self-collision (the co-carry ``_GRIPPER_ACTION_OPEN``
#: convention). Closed (-1) keeps the welded body nestled in the gripper.
_GRIPPER_ACTION: float = -1.0

#: Orientation-hold proportional gain (per control step, on the rotation-vector
#: error, rad/rad). Drives the gripper back to its captured ready orientation so
#: the welded peg / socket cannot tip under the insertion contact torque (the
#: 6-DOF term the 3x7 linear Jacobian cannot express). The co-insert design.
_ORI_HOLD_GAIN: float = 6.0

#: Per-step angular command clip, radians — keeps the orientation correction
#: quasi-static (no snap).
_ORI_STEP_MAX_RAD: float = 0.15

#: Default peg half-length, metres — the welded peg extends ``±_peg_half_len``
#: along its local +z; the inserting tip is the +z end. Default mirrors
#: :data:`chamber.envs.coinsert.COINSERT_DEPTH_TARGET_M` (the welded-rig peg
#: half-length); overridable via ``spec.extra["peg_half_len"]``.
_DEFAULT_PEG_HALF_LEN_M: float = 0.040

# --- Base inserter gains (the structured base; the co-insert design) ----------

#: Cartesian proportional gain (per control step, dimensionless on the metre
#: error). With the per-step clip below this gives a smooth straight-line
#: approach that saturates far from target and tapers near it (co-carry value).
_BASE_KP: float = 2.5

#: Cartesian integral gain (per second) — closes the load-induced steady-state
#: offset of the ``pd_joint_delta_pos`` delta-from-measured controller (the
#: per-step torque balances the insertion reaction a few mm short). Deadzone-
#: gated + anti-windup-clamped, as in co-carry.
_BASE_KI: float = 0.6

#: Per-step Cartesian command clip, metres. 0.015 m/step at 20 Hz keeps the
#: approach brisk while the press itself is a slower axial velocity below.
_BASE_STEP_MAX_M: float = 0.02

#: Damped pseudo-inverse damping (lambda), in the Jacobian's length units
#: (Nakamura & Hanafusa 1986) — keeps joint steps bounded near singularities.
_BASE_DAMPING: float = 0.04

#: Cartesian-error magnitude (m) below which the integral engages — gating it
#: off during the large-error approach prevents transient windup.
_BASE_INTEGRAL_ENGAGE_M: float = 0.05

#: Anti-windup clamp on the accumulated Cartesian integral (metre*s per axis).
_BASE_INTEGRAL_CLAMP_MS: float = 0.4

#: Approach standoff, metres — during the approach phase the base drives the peg
#: tip to this height above the socket mouth (along the socket axis) while
#: nulling the lateral offset, so the descent starts pre-aligned.
_BASE_APPROACH_STANDOFF_M: float = 0.015

#: Lateral centering tolerance, metres — the base presses only once the peg tip
#: is laterally within this of the socket axis (tight, so the chamfer + clearance
#: can capture it; a peg pressed while off-centre just rests on the rim).
_BASE_ALIGN_TOL_M: float = 0.0012

#: Stall window, control steps — if the press makes no depth progress for this
#: many steps the unjam maneuver engages (the peg has friction-wedged on a slight
#: cock and a stiff press only locks it harder / drags the whole assembly down).
_BASE_STALL_STEPS: int = 3

#: Unjam retract distance, metres — on a stall the base briefly backs the peg UP
#: by this much (relieving the wedge normal force) while spiral-searching, then
#: re-presses; repeated, it walks the peg past the wedge depth. The standard
#: peg-in-hole anti-jam maneuver.
_BASE_RETRACT_M: float = 0.009

#: Unjam retract duration, control steps — how long each back-off pulse lasts.
_BASE_RETRACT_STEPS: int = 5

#: Per-step axial descent VELOCITY during the insertion press, metres/step — a
#: steady quasi-static seat — the target depth advances by this each press step
#: (capped at the depth target). Paired with the unjam retract pulses below, it
#: walks the peg past the friction-wedge depth without slamming.
_BASE_DESCEND_STEP_M: float = 0.002

#: Lead-in / spiral-search amplitude, metres — the lateral search radius the
#: base sweeps while pressing, to find the hole when the initial alignment is at
#: the edge of the chamfer capture. Shrinks linearly to zero as the peg seats.
#: With the chamfer capture radius this defines the FROZEN insertion envelope
#: (:data:`chamber.envs.coinsert` gate0.base_controller_holder_dependence).
_BASE_SPIRAL_AMPLITUDE_M: float = 0.003

#: Spiral angular rate, radians per control step — the search sweep frequency.
_BASE_SPIRAL_OMEGA_RAD: float = 0.6

# --- Reference holder gains (the cooperative anchor; the co-insert design) ----

#: Holder lateral (xy) tracking gain — moderately stiff so the socket mouth
#: stays under the descending peg tip (the cooperative track).
_HOLDER_KP_LATERAL: float = 2.0

#: Holder axial (along-insertion) gain — firm enough to HOLD the socket height
#: against the insertion press (so the peg seats INTO the socket rather than
#: pushing the whole free socket down — an over-soft holder yields downward and
#: the relative insertion depth plateaus), but kept below the lateral gain so it
#: still yields slightly / compliantly (the cooperative compromise). Tuned at the
#: S2 pilot: too soft (≈0.8) plateaus depth ~30 mm; this seats.
_HOLDER_KP_AXIAL: float = 2.0

#: Holder integral gain (per second) — closes the steady-state hold-height
#: offset under the static receptacle weight (deadzone-gated, clamped).
_HOLDER_KI: float = 0.4

#: Holder per-step Cartesian command clip, metres.
_HOLDER_STEP_MAX_M: float = 0.02

#: Holder damped pseudo-inverse damping (lambda).
_HOLDER_DAMPING: float = 0.04

#: Holder integral engage radius / clamp (mirrors the base).
_HOLDER_INTEGRAL_ENGAGE_M: float = 0.05
_HOLDER_INTEGRAL_CLAMP_MS: float = 0.4


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


def _quat_wxyz_to_rot(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """Rotation matrix from a wxyz quaternion (the SAPIEN / ManiSkill convention)."""
    w, x, y, z = q / max(float(np.linalg.norm(q)), 1e-12)
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
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


def _rotvec(rot: NDArray[np.float64]) -> NDArray[np.float64]:
    """Axis-angle (rotation-vector) of a rotation matrix — the orientation error.

    Returns ``angle * axis`` (a 3-vector whose norm is the rotation angle), used
    to drive the gripper orientation back to its captured target so the welded
    peg / socket cannot tip. Deterministic; pure NumPy.
    """
    cos = (float(np.trace(rot)) - 1.0) / 2.0
    angle = float(np.arccos(np.clip(cos, -1.0, 1.0)))
    if angle < 1e-6:  # noqa: PLR2004 - near-identity → no rotation
        return np.zeros(3, dtype=np.float64)
    axis = np.array(
        [rot[2, 1] - rot[1, 2], rot[0, 2] - rot[2, 0], rot[1, 0] - rot[0, 1]], dtype=np.float64
    ) / (2.0 * np.sin(angle))
    return angle * axis


def _lateral_basis(axis: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Two orthonormal vectors spanning the plane perpendicular to ``axis``.

    Used to express the spiral-search lateral offset in world coordinates for an
    arbitrary socket-axis orientation (the insertion need not be exactly world
    vertical). Deterministic — the reference vector is chosen by the smaller
    component of ``axis`` so the cross product is well-conditioned.
    """
    a = axis / max(float(np.linalg.norm(axis)), 1e-12)
    ref = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])  # noqa: PLR2004
    e1 = np.cross(a, ref)
    e1 = e1 / max(float(np.linalg.norm(e1)), 1e-12)
    e2 = np.cross(a, e1)
    return e1, e2


class _CoInsertControllerBase(PartnerBase):
    """Shared obs-reading + Jacobian joint-step machinery (ADR-026; ADR-004 §Decision).

    Both the structured base inserter and the cooperative reference holder read
    the same observation leaves, build the same damped-pseudoinverse joint step
    in the arm base frame, and carry the same within-episode integral + step
    counter (both zeroed in :meth:`reset`). Subclasses differ only in the
    Cartesian target they compute in :meth:`_target_and_body`.
    """

    def __init__(self, spec: PartnerSpec) -> None:
        """Bind the spec and build the Jacobian/FK provider (ADR-026 §Decision 1)."""
        super().__init__(spec)
        from chamber.agents.panda_jacobian import PandaJacobianProvider

        extra = spec.extra
        self._uid: str = extra.get("uid", "panda_wristcam")
        self._base_xyz: NDArray[np.float64] = _parse_vec3(extra.get("base_xyz", "0,0,0"))
        self._base_rot_t: NDArray[np.float64] = _rot_z(float(extra.get("base_yaw_deg", "0"))).T
        self._peg_half_len: float = float(extra.get("peg_half_len", str(_DEFAULT_PEG_HALF_LEN_M)))
        self._damping: float = float(extra.get("damping", str(self._default_damping())))
        self._step_max: float = float(extra.get("step_max", str(self._default_step_max())))
        urdf_path = extra.get("urdf_path")
        self._provider = PandaJacobianProvider(urdf_path)
        # Within-episode state (the ONLY state; zeroed in reset — ADR-009).
        self._integral: NDArray[np.float64] = np.zeros(_XYZ, dtype=np.float64)
        self._step: int = 0
        # Captured target gripper orientation (the ready orientation held fixed so
        # the welded body cannot tip). None until the first act of each episode.
        self._target_rot: NDArray[np.float64] | None = None

    # -- gain hooks (subclasses override) --
    def _default_damping(self) -> float:  # pragma: no cover - trivial
        return _BASE_DAMPING

    def _default_step_max(self) -> float:  # pragma: no cover - trivial
        return _BASE_STEP_MAX_M

    def reset(self, *, seed: int | None = None) -> None:
        """Clear the within-episode integral + step counter (ADR-009; stateless across episodes).

        The control law is a deterministic function of ``obs`` + ``spec`` plus a
        within-episode Cartesian integral and a step counter (the spiral-search
        phase clock); BOTH are zeroed here, so the controller carries no state
        across episodes (ADR-009 §Decision). The reset is asserted post-hoc by
        :meth:`assert_episode_state_clear` (the co-carry stateful-control
        regression guard). The seed is accepted only for
        :class:`~chamber.partners.api.FrozenPartner` Protocol conformance.
        """
        del seed
        self._integral = np.zeros(_XYZ, dtype=np.float64)
        self._step = 0
        self._target_rot = None

    def assert_episode_state_clear(self) -> None:
        """Assert the within-episode state was actually cleared (ADR-009; the reset guard).

        Raises :class:`AssertionError` if the integral, the step counter, or the
        captured target orientation is non-empty — the explicit check the
        property test calls right after :meth:`reset` so state that silently
        leaks across episodes (the co-carry stateful-control bug) cannot recur.
        """
        if self._step != 0 or bool(np.any(self._integral != 0.0)) or self._target_rot is not None:
            msg = (
                f"{type(self).__name__}: per-episode state not cleared after reset "
                f"(step={self._step}, integral_nonzero={bool(np.any(self._integral != 0.0))}, "
                f"target_rot_set={self._target_rot is not None}). "
                "ADR-009 §Decision: partners are stateless across episodes."
            )
            raise AssertionError(msg)

    # -- obs helpers --
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

    def _read_pose(
        self, obs: Mapping[str, object], key: str
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
        """Read a ``(p[3], R[3,3])`` world pose from ``obs["extra"][key]`` (xyz + wxyz)."""
        extra = obs.get("extra")
        if not isinstance(extra, Mapping) or key not in extra:
            return None
        raw = _to_numpy_flat(extra[key])
        if raw.shape[0] < 7:  # noqa: PLR2004 - xyz (3) + wxyz quat (4)
            return None
        return raw[:3], _quat_wxyz_to_rot(raw[3:7])

    def _peg_tip_world(
        self, peg_p: NDArray[np.float64], peg_r: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """World position of the peg's inserting tip (the local +z end of the welded peg)."""
        return peg_p + self._peg_half_len * peg_r[:, 2]

    def _joint_step_action(
        self, qpos: NDArray[np.float64], v_world: NDArray[np.float64]
    ) -> NDArray[np.float32]:
        """Map a world Cartesian command + orientation-hold to a ``pd_joint_delta_pos`` action.

        Builds a 6-DOF command in the arm base frame — the translation
        ``base_rot_t @ v_world`` plus an angular term that drives the gripper
        back to its captured ready orientation (so the rigidly-welded peg /
        socket cannot tip under the insertion contact torque; the 3x7 linear
        Jacobian alone leaves orientation free) — and takes the damped
        pseudo-inverse joint step ``dq = J6^T (J6 J6^T + lambda^2 I)^-1 v6``
        (Nakamura & Hanafusa 1986; ADR-004 §Decision; ADR-026 §Decision 1). The
        gripper channel is held constant (the welded body makes the fingers
        inert).
        """
        v_base = self._base_rot_t @ v_world
        rot_cur = self._provider.fk_tcp_rotation(qpos)  # (3,3) base frame
        if self._target_rot is None:
            self._target_rot = rot_cur  # hold the ready orientation
        omega = _ORI_HOLD_GAIN * _rotvec(self._target_rot @ rot_cur.T)
        omega = np.clip(omega, -_ORI_STEP_MAX_RAD, _ORI_STEP_MAX_RAD)
        v6 = np.concatenate([v_base, omega])
        jac6 = self._provider.jacobian_6x7(qpos)  # (6,7) base frame
        jjt = jac6 @ jac6.T + (self._damping**2) * np.eye(6)
        dq = jac6.T @ np.linalg.solve(jjt, v6)
        action = np.zeros(_PANDA_ARM_DOF + 1, dtype=np.float32)
        action[:_PANDA_ARM_DOF] = np.clip(dq / _ACTION_DELTA_SCALE, -1.0, 1.0).astype(np.float32)
        action[_PANDA_ARM_DOF] = _GRIPPER_ACTION
        return action


@register_partner("coinsert_base_inserter")
class CoInsertBaseInserter(_CoInsertControllerBase):
    """Structured base inserter — impedance approach + spiral-search press (ADR-026 §Decision 1).

    The ``base_policy`` + ``impedance_overlay`` term of the ADR-026 ego
    decomposition, with **no learned residual**. Reads the observed peg pose and
    socket (receptacle) pose, computes the vertical insertion target, and drives
    the ego arm's TCP — and hence the rigidly-welded peg — toward it via the
    damped-pseudoinverse Jacobian step.

    Two-phase control:

    1. **Approach.** While the peg tip is above the socket mouth or laterally
       misaligned beyond :data:`_BASE_ALIGN_TOL_M`, drive the tip to
       :data:`_BASE_APPROACH_STANDOFF_M` above the mouth along the socket axis
       while nulling the lateral offset — so the descent begins pre-aligned.
    2. **Insertion press + lead-in search.** Once aligned, advance the target
       depth by :data:`_BASE_DESCEND_STEP_M` per step (a quasi-static seat, not a
       slam), adding a shrinking spiral lateral offset of initial radius
       :data:`_BASE_SPIRAL_AMPLITUDE_M` to find the hole at the chamfer capture
       edge. The spiral radius decays to zero as the peg seats.

    The base's **holder-dependence boundary** is the physical *insertion
    envelope* = chamfer capture radius + :data:`_BASE_SPIRAL_AMPLITUDE_M`. The
    base attempts insertion for any receptacle pose error within that envelope
    and does NOT re-acquire a receptacle the holder has let drift outside it (it
    keeps pressing toward the last-seen socket pose; it has no global re-search).
    The envelope numbers are emitted into the S2 artifact and frozen at S3
    (:mod:`chamber.envs.coinsert` gate0.base_controller_holder_dependence) — a
    property of the controller's physics, not a degree of freedom tuned to make
    Gate 0 resolve a chosen way.

    ``spec.extra`` keys: ``uid`` / ``base_xyz`` / ``base_yaw_deg`` (the ego arm
    geometry), ``peg_half_len``, optional ``kp`` / ``ki`` / ``step_max`` /
    ``damping`` / ``spiral_amp`` / ``descend_step`` overrides.
    """

    def __init__(self, spec: PartnerSpec) -> None:
        """Bind base-inserter gains (ADR-026 §Decision 1)."""
        super().__init__(spec)
        extra = spec.extra
        self._kp: float = float(extra.get("kp", str(_BASE_KP)))
        self._ki: float = float(extra.get("ki", str(_BASE_KI)))
        self._spiral_amp: float = float(extra.get("spiral_amp", str(_BASE_SPIRAL_AMPLITUDE_M)))
        self._descend_step: float = float(extra.get("descend_step", str(_BASE_DESCEND_STEP_M)))
        self._depth_target: float = float(extra.get("depth_target", "0.040"))
        # Within-episode stall / unjam state for the lead-in search (zeroed in reset).
        self._stall_count: int = 0
        self._retract_count: int = 0
        self._last_depth: float = -1.0

    def _default_damping(self) -> float:
        return _BASE_DAMPING

    def _default_step_max(self) -> float:
        return _BASE_STEP_MAX_M

    def reset(self, *, seed: int | None = None) -> None:
        """Clear within-episode state including the lead-in-search stall clock (ADR-009)."""
        super().reset(seed=seed)
        self._stall_count = 0
        self._retract_count = 0
        self._last_depth = -1.0

    def assert_episode_state_clear(self) -> None:
        """Assert the within-episode state — including the stall clock — was cleared."""
        super().assert_episode_state_clear()
        if self._stall_count != 0 or self._retract_count != 0:
            msg = (
                f"{type(self).__name__}: stall clock not cleared after reset. "
                "ADR-009 §Decision: partners are stateless across episodes."
            )
            raise AssertionError(msg)

    @property
    def insertion_envelope_m(self) -> dict[str, float]:
        """The frozen-at-S3 insertion-envelope numbers (ADR-026 gate0 base-controller dependence).

        Returns the two physical numbers that define the base's holder-dependence
        boundary: the declared spiral-search amplitude and the chamfer capture
        radius (the env's :data:`chamber.envs.coinsert.COINSERT_CHAMFER_M`), plus
        their sum (the envelope). Auditable; emitted into the S2 artifact.
        """
        from chamber.envs.coinsert import COINSERT_CHAMFER_M

        return {
            "spiral_search_amplitude_m": self._spiral_amp,
            "chamfer_capture_radius_m": float(COINSERT_CHAMFER_M),
            "insertion_envelope_m": self._spiral_amp + float(COINSERT_CHAMFER_M),
        }

    def act(  # noqa: PLR0915 - the approach/press/seated phase machine + unjam is one cohesive step
        self, obs: Mapping[str, object], *, deterministic: bool = True
    ) -> NDArray[np.floating]:
        """Return the ego ``pd_joint_delta_pos`` insertion action (ADR-026 §Decision 1)."""
        del deterministic
        self._step += 1
        gripper_only = np.zeros(_PANDA_ARM_DOF + 1, dtype=np.float32)
        gripper_only[_PANDA_ARM_DOF] = _GRIPPER_ACTION
        qpos = self._read_qpos(obs)
        peg = self._read_pose(obs, "peg_pose")
        sock = self._read_pose(obs, "receptacle_pose")
        if qpos is None or peg is None or sock is None:
            return gripper_only
        peg_p, peg_r = peg
        sock_p, sock_r = sock
        tip = self._peg_tip_world(peg_p, peg_r)
        mouth = sock_p  # socket opening sits at the receptacle local origin
        axis = sock_r[:, 2]  # socket axis (points out of the mouth)
        axis = axis / max(float(np.linalg.norm(axis)), 1e-12)

        rel = tip - mouth
        depth = -float(rel @ axis)  # >0 once the tip is below the mouth (inserted)
        lateral_vec = rel - (rel @ axis) * axis
        lateral_mag = float(np.linalg.norm(lateral_vec))

        e1, e2 = _lateral_basis(axis)
        if depth >= (self._depth_target - 0.002):
            # Seated: hold (command ~zero velocity) so both arms go static and the
            # receptacle settles — the success ``static`` ∧ ``settled`` conjuncts
            # need the press to stop once depth is reached.
            self._stall_count = 0
            self._integral = np.zeros(_XYZ, dtype=np.float64)
            v_cmd = np.zeros(_XYZ, dtype=np.float64)
        elif depth < -_BASE_APPROACH_STANDOFF_M or lateral_mag > _BASE_ALIGN_TOL_M:
            # Approach / centre: STIFFLY drive the tip to a standoff above the
            # mouth while nulling the lateral offset, so the descent begins tightly
            # centred. The integral closes the steady-state lateral offset.
            self._stall_count = 0
            self._last_depth = depth
            target = mouth + _BASE_APPROACH_STANDOFF_M * axis
            error = target - tip
            if float(np.linalg.norm(error)) < _BASE_INTEGRAL_ENGAGE_M:
                self._integral = np.clip(
                    self._integral + error * _CONTROL_DT,
                    -_BASE_INTEGRAL_CLAMP_MS,
                    _BASE_INTEGRAL_CLAMP_MS,
                )
            v_cmd = np.clip(
                self._kp * error + self._ki * self._integral, -self._step_max, self._step_max
            )
        else:
            # Insertion press toward the target depth (centred on the mouth +
            # advancing depth). If the press makes no depth progress the peg has
            # friction-wedged on a slight cock; a stiff press only locks it harder
            # and drags the whole free assembly down — so run the UNJAM maneuver:
            # briefly RETRACT (back the peg up, relieving the wedge normal force)
            # while spiral-searching, then re-press. Repeated, it walks the peg
            # past the wedge depth (the FROZEN insertion envelope = the spiral
            # amplitude + the chamfer capture).
            if self._retract_count > 0:
                self._retract_count -= 1
                target_depth = max(depth - _BASE_RETRACT_M, -0.005)
            else:
                if depth <= self._last_depth + 1e-4:
                    self._stall_count += 1
                else:
                    self._stall_count = 0
                if self._stall_count >= _BASE_STALL_STEPS:
                    self._retract_count = _BASE_RETRACT_STEPS
                    self._stall_count = 0
                target_depth = min(depth + self._descend_step, self._depth_target + 0.005)
            self._last_depth = depth
            target = mouth - target_depth * axis
            if self._retract_count > 0:
                ang = _BASE_SPIRAL_OMEGA_RAD * self._step
                target = target + self._spiral_amp * (np.cos(ang) * e1 + np.sin(ang) * e2)
            error = target - tip
            if float(np.linalg.norm(error)) < _BASE_INTEGRAL_ENGAGE_M:
                self._integral = np.clip(
                    self._integral + error * _CONTROL_DT,
                    -_BASE_INTEGRAL_CLAMP_MS,
                    _BASE_INTEGRAL_CLAMP_MS,
                )
            v_cmd = np.clip(
                self._kp * error + self._ki * self._integral, -self._step_max, self._step_max
            )

        return self._joint_step_action(qpos, v_cmd)


@register_partner("coinsert_reference_holder")
class CoInsertReferenceHolder(_CoInsertControllerBase):
    """Cooperative reference holder — tracks the socket, yields compliantly (ADR-026 §Decision 1).

    The co-designed calibration anchor (the ``cocarry_impedance`` analogue for
    the holder seat). It reads the observed peg + socket poses and drives the
    holder arm so the **free** receptacle's mouth stays under the descending peg
    tip (a moderately stiff lateral track) while holding the nominal mouth height
    with a deliberately **soft** axial gain — so it yields compliantly to the
    insertion reaction force rather than fighting it (a stiff holder would spike
    the coupling wrench and read as un-cooperative). Paired with
    :class:`CoInsertBaseInserter`, the matched pair seats the peg reliably — the
    honest high reference and the Δ_min dynamic-range anchor (ADR-026 §Decision
    2; the S2 precondition reference_success_min ≥ 0.9).

    The nominal mouth height is captured on the FIRST :meth:`act` of each episode
    (the warm-started socket height), so the holder defends the height the env
    initialised — not a hard-coded constant — keeping it robust to the S3 freeze.

    ``spec.extra`` keys: ``uid`` / ``base_xyz`` / ``base_yaw_deg`` (the holder arm
    geometry; ``base_yaw_deg`` = 180 for the mirrored partner seat),
    ``peg_half_len``, optional gain overrides ``kp_lateral`` / ``kp_axial`` /
    ``ki`` / ``step_max`` / ``damping``.
    """

    def __init__(self, spec: PartnerSpec) -> None:
        """Bind reference-holder gains (ADR-026 §Decision 1)."""
        super().__init__(spec)
        extra = spec.extra
        self._kp_lateral: float = float(extra.get("kp_lateral", str(_HOLDER_KP_LATERAL)))
        self._kp_axial: float = float(extra.get("kp_axial", str(_HOLDER_KP_AXIAL)))
        self._ki: float = float(extra.get("ki", str(_HOLDER_KI)))
        self._nominal_mouth: NDArray[np.float64] | None = None

    def _default_damping(self) -> float:
        return _HOLDER_DAMPING

    def _default_step_max(self) -> float:
        return _HOLDER_STEP_MAX_M

    def reset(self, *, seed: int | None = None) -> None:
        """Clear within-episode state including the captured nominal mouth pose (ADR-009)."""
        super().reset(seed=seed)
        self._nominal_mouth = None

    def assert_episode_state_clear(self) -> None:
        """Assert the within-episode state — including the nominal mouth pose — was cleared."""
        super().assert_episode_state_clear()
        if self._nominal_mouth is not None:
            msg = (
                f"{type(self).__name__}: nominal mouth pose not cleared after reset. "
                "ADR-009 §Decision: partners are stateless across episodes."
            )
            raise AssertionError(msg)

    def act(self, obs: Mapping[str, object], *, deterministic: bool = True) -> NDArray[np.floating]:
        """Return the holder ``pd_joint_delta_pos`` cooperative-hold action (ADR-026 §Decision 1).

        Cooperative reference behaviour: hold the socket steady at the
        warm-started insertion pose (captured on the first act), giving the base
        inserter a fixed, well-posed target to centre on, with a moderately stiff
        LATERAL gain (keep the mouth under the peg) and a deliberately SOFT AXIAL
        gain so the holder yields compliantly to the insertion reaction force
        rather than fighting it. Orientation is held by :meth:`_joint_step_action`
        (the socket cannot tip / spill the part).
        """
        del deterministic
        self._step += 1
        gripper_only = np.zeros(_PANDA_ARM_DOF + 1, dtype=np.float32)
        gripper_only[_PANDA_ARM_DOF] = _GRIPPER_ACTION
        qpos = self._read_qpos(obs)
        sock = self._read_pose(obs, "receptacle_pose")
        if qpos is None or sock is None:
            return gripper_only
        sock_p, sock_r = sock
        mouth = sock_p
        axis = sock_r[:, 2]
        axis = axis / max(float(np.linalg.norm(axis)), 1e-12)

        # Capture the warm-started mouth pose once; hold it.
        if self._nominal_mouth is None:
            self._nominal_mouth = mouth.copy()

        err_vec = self._nominal_mouth - mouth
        axial_err = float(err_vec @ axis) * axis
        lateral_err = err_vec - axial_err
        # Stiff lateral hold + soft axial yield (the cooperative compliance).
        error = self._kp_lateral * lateral_err + self._kp_axial * axial_err
        if float(np.linalg.norm(err_vec)) < _HOLDER_INTEGRAL_ENGAGE_M:
            self._integral = np.clip(
                self._integral + axial_err * _CONTROL_DT,
                -_HOLDER_INTEGRAL_CLAMP_MS,
                _HOLDER_INTEGRAL_CLAMP_MS,
            )
        v_cmd = np.clip(error + self._ki * self._integral, -self._step_max, self._step_max)
        return self._joint_step_action(qpos, v_cmd)


__all__ = ["CoInsertBaseInserter", "CoInsertReferenceHolder"]
