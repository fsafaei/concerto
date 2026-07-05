# SPDX-License-Identifier: Apache-2.0
r"""Partner-blind co-carry ego — B-BLIND for the A3 admission check (ADR-027 §Admission protocol).

The A3 partner-relevance check compares a coupling-aware reference ego
against a partner-blind ego that "acts without partner state or
coupling feedback" (ADR-027 §Admission protocol; B-BLIND, ADR-011 as
amended). On co-carry the matched reference
(:class:`chamber.partners.cocarry_impedance.CoCarryImpedancePartner`)
reads no partner leaf at all — its only coupling channel is
**proprioceptive**: the ``pd_joint_delta_pos`` law commands deltas from
the *measured* joint angles, so the load transmitted through the welded
bar (i.e. the partner's motion) enters through ``qpos``, and the
deadzone-gated Cartesian integral exists precisely to close the
load-induced steady-state offset (see the reference controller's module
docstring).

This class is that reference **stripped of exactly the coupling
channel**, nothing else:

1. **Dead-reckoned joints.** The measured ``qpos`` is read once, at the
   first control step of the episode (the published ready pose — no
   load history has accumulated at t=0), and thereafter the controller
   integrates its *own commanded* joint deltas open-loop. The
   partner's influence on the measured trajectory can no longer be
   sensed.
2. **Integral disabled.** ``ki = 0`` — the term whose sole function is
   to respond to the bar-transmitted load is removed.

Everything else — the cooperative TCP target from the world goal (task
specification, not partner state), the P law, the damped-pseudoinverse
joint step, the gains — is inherited unchanged, so the blind ego is
exactly as competent as the reference in the no-load limit and the A3
gap isolates the value of coupling feedback (ADR-026 §Decision 2's
falsifiability requirement: the demotion outcome is a pre-committed,
reportable result, not a rigged pass).

References:
- ADR-027 §Admission protocol (A3; failure demotes the task to Tier 1).
- ADR-011 §Decision as amended (B-BLIND doubles as admission check A3).
- ADR-026 §Decision 2 (coupling positive-control discipline).
- ADR-009 §Decision (partner interface; deterministic, stateless
  across episodes).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from chamber.partners.cocarry_impedance import CoCarryImpedancePartner
from chamber.partners.registry import register_partner

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping

    from numpy.typing import NDArray

    from chamber.partners.api import PartnerSpec

#: Registry name of the partner-blind co-carry ego (ADR-027 §Admission protocol A3).
COCARRY_BLIND_IMPEDANCE_CLASS: str = "cocarry_blind_impedance"

#: Number of Panda arm DOF (mirrors the reference controller).
_PANDA_ARM_DOF: int = 7

#: Number of xyz components.
_XYZ: int = 3

#: ``pd_joint_delta_pos`` per-step joint-delta scale (rad) — the scale the
#: emitted normalised action is mapped through by the env's controller,
#: and therefore the scale the dead-reckoner integrates its own commands
#: at (mani-skill==3.0.1 panda ``pd_joint_delta_pos``: lower=-0.1, upper=0.1).
_ACTION_DELTA_SCALE: float = 0.1


@register_partner(COCARRY_BLIND_IMPEDANCE_CLASS)
class CoCarryBlindImpedancePartner(CoCarryImpedancePartner):
    """The matched co-carry controller minus the coupling channel (ADR-027 §Admission A3).

    ``spec.extra`` keys are those of
    :class:`chamber.partners.cocarry_impedance.CoCarryImpedancePartner`;
    a ``ki`` override is ignored — the integral is the coupling-feedback
    term and is forced to zero (see module docstring).
    """

    def __init__(self, spec: PartnerSpec) -> None:
        """Bind the reference gains with ``ki`` forced to zero (ADR-027 §Admission protocol A3)."""
        super().__init__(spec)
        self._ki = 0.0
        # Dead-reckoned arm joint estimate; booted on the first act() of
        # each episode from the measured ready pose.
        self._q_hat: NDArray[np.float64] | None = None

    def reset(self, *, seed: int | None = None) -> None:
        """Clear the dead-reckoned joint estimate (ADR-009 §Decision; stateless across episodes)."""
        super().reset(seed=seed)
        self._q_hat = None

    def act(
        self,
        obs: Mapping[str, object],
        *,
        deterministic: bool = True,
    ) -> NDArray[np.floating]:
        """Return the blind ``pd_joint_delta_pos`` action (ADR-027 §Admission protocol A3).

        Reads the world goal (task specification) and — on the first
        step of the episode only — the measured joint angles (the
        published ready pose). Thereafter the joint state is
        dead-reckoned from the controller's own emitted commands, so no
        partner- or load-induced deviation is ever observed.

        Args:
            obs: Gymnasium observation mapping (see the reference
                controller); the ``qpos`` leaf is consumed only at the
                episode's first step.
            deterministic: Ignored — the controller is deterministic.

        Returns:
            Float32 action of length 8 (7 normalised joint deltas +
            gripper-open); a gripper-only action if the obs lacks the
            required keys (defensive, mirrors the reference).
        """
        del deterministic
        goal = self._read_goal(obs)
        action = np.zeros(_PANDA_ARM_DOF + 1, dtype=np.float32)
        action[_PANDA_ARM_DOF] = 1.0  # gripper open, mirrors the reference
        if self._q_hat is None:
            boot = self._read_qpos(obs)
            if boot is None:
                return action
            self._q_hat = np.asarray(boot[:_PANDA_ARM_DOF], dtype=np.float64).copy()
        if goal is None:
            return action
        # Reference control law evaluated at the dead-reckoned joints,
        # P-only (the integral is the coupling channel; forced to 0).
        target_world = goal + np.asarray([self._end_sign * self._bar_half_len, 0.0, 0.0])
        target_base = self._base_rot_t @ (target_world - self._base_xyz)
        tcp_base = self._provider.fk_tcp_position(self._q_hat)
        error = target_base - tcp_base
        v_cmd = np.clip(self._kp * error, -self._step_max, self._step_max)
        jac = self._provider(None, self._q_hat)  # type: ignore[arg-type]  # (3, 7), base frame
        jjt = jac @ jac.T + (self._damping**2) * np.eye(_XYZ)
        dq = jac.T @ np.linalg.solve(jjt, v_cmd)
        normalised = np.clip(dq / _ACTION_DELTA_SCALE, -1.0, 1.0)
        # Dead-reckon forward by the command actually emitted.
        self._q_hat = self._q_hat + normalised * _ACTION_DELTA_SCALE
        action[:_PANDA_ARM_DOF] = normalised.astype(np.float32)
        return action


__all__ = ["COCARRY_BLIND_IMPEDANCE_CLASS", "CoCarryBlindImpedancePartner"]
