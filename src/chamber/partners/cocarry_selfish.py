# SPDX-License-Identifier: Apache-2.0
r"""Self-interested (non-co-designed) co-carry teammates for the Rung-5 CD measurement.

Rung 5 (co-design / objective-sharing axis) measures whether pairing the
**frozen** trained co-carry incumbent with an independently-competent but
**non-co-designed** partner degrades cooperation — the regime the thesis is
about (a partner you cannot co-optimise), and the in-kit Rung-2-STOP regime
(854 N internal stress vs a frozen non-cooperating partner on this task).

The isolation that makes a Rung-5 drop attributable to co-design *by
construction*: every Arm-B variant here is the **same Panda body, the same
matched-impedance class, and the SAME physical/dynamical gains** as the
cooperative reference
:class:`chamber.partners.cocarry_impedance.CoCarryImpedancePartner` and the
Arm-A control-style family
(:mod:`chamber.partners.cocarry_policy_shift`). They differ from the
cooperative reference in **exactly one factor — the coordination objective**:
each pursues a *self-interested own-objective* and is **blind to the joint
internal coupling stress** (it never observes or optimises it). Arm A varies
control *style* within the shared cooperative objective; Arm B varies the
*objective* at matched style. Holding body + class + gains fixed and varying
only the objective is the mediation guarantee (the zeroed-static control is
not needed and is not used).

The three variants (all at the matched gains ``_KP=2.5`` / ``_STEP_MAX_M=0.03``
/ ``_DAMPING=0.04`` inherited from
:class:`chamber.partners.cocarry_policy_shift._CoCarryShiftedBase`):

- :class:`CoCarrySelfishGoalPartner` (``cocarry_selfish_goal``) — drives its
  own grip toward the **goal centroid itself** (a self-referenced "get me to
  the goal" objective), ignoring the complementary bar-end split the
  cooperative geometry depends on. It still transports toward the goal
  (competent), but pulls off the shared load axis, inducing internal stress.
- :class:`CoCarrySelfishEffortPartner` (``cocarry_selfish_effort``) — tracks
  the cooperative bar-end target but **minimises its own actuation**: a
  dead-band zeroes small-error commands and the command authority is
  down-scaled, so it contributes lazily and lets the partner lead. Competent
  (it still progresses on large errors) but under-contributing by its own
  effort-minimising objective.
- :class:`CoCarrySelfishStationPartner` (``cocarry_selfish_station``) — holds
  its **own start bar-end pose** (station-keeping, cached on the first act),
  indifferent to transport. Perfectly competent at its own objective (stay
  put); whether it clears the joint capability gate is the Stage-2 empirical
  question.

All three route through the partner interface (ADR-009 §Decision): they
subclass :class:`chamber.partners.cocarry_policy_shift._CoCarryShiftedBase`
(hence :class:`chamber.partners.interface.PartnerBase`), so the AHT
freeze-shield (``_FORBIDDEN_ATTRS``) is inherited and
:meth:`chamber.benchmarks.ego_ppo_trainer.EgoPPOTrainer._assert_partner_is_frozen`
passes. They are **partner-agnostic / black-box**: they read only task obs
(``obs["extra"]["goal_pos"]`` and ``obs["agent"][uid]["qpos"]``), never the
ego's identity or internals (I3). Each is deterministic (a pure function of
obs + within-episode state cleared in :meth:`reset`), so two loads emit
byte-identical actions on identical obs (P6 / ADR-002).

References:
- ADR-026 §Decision 4 (the Phase-2 forward-design ladder; Rung 5 = the
  co-design / objective-sharing measurement against the frozen incumbent).
- ADR-009 §Decision (frozen black-box partner; partner reads task leaves only).
- :class:`chamber.partners.cocarry_policy_shift._CoCarryShiftedBase` (the
  matched-impedance machinery these reuse unchanged — only the objective hook
  differs, so the variants are matched-except-objective by construction).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from chamber.partners.cocarry_policy_shift import _CoCarryShiftedBase
from chamber.partners.registry import register_partner

#: Number of xyz components (local copy; avoids importing a private constant).
_XYZ: int = 3

#: The Rung-5 Arm-B candidate set (the non-co-designed variants), in
#: registration order. The Stage-2 calibration iterates this; >= 2 must clear
#: the C_min independent-competence gate to proceed to a CD-axis claim
#: (ADR-026 §Decision 4; the prereg's replication requirement).
COCARRY_SELFISH_CANDIDATES: tuple[str, ...] = (
    "cocarry_selfish_goal",
    "cocarry_selfish_effort",
    "cocarry_selfish_station",
)

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping

    from numpy.typing import NDArray

    from chamber.partners.api import PartnerSpec

#: Selfish-effort dead-band (m): below this own-error the variant emits no
#: Cartesian command (minimise actuation), so it acts only on large errors.
_EFFORT_DEADBAND_M: float = 0.04
#: Selfish-effort command-authority scale (vs the matched 1.0): contributes
#: at reduced authority once it does act (a lazy-but-moving partner).
_EFFORT_AUTHORITY: float = 0.5


@register_partner("cocarry_selfish_goal")
class CoCarrySelfishGoalPartner(_CoCarryShiftedBase):
    """Goal-greedy self-interested teammate (ADR-026 §Decision 4; ADR-009).

    Matched-impedance class + matched gains; the ONLY shift is the objective:
    it drives its own grip toward the **goal centroid itself** rather than its
    complementary cooperative bar-end. Competent (it transports toward the
    goal) but non-co-designed (it ignores the shared load-split geometry, so it
    pulls off-axis and loads the coupling). Reads task leaves only (I3).
    """

    def _target_world(
        self, goal: NDArray[np.float64], obs: Mapping[str, object]
    ) -> NDArray[np.float64]:
        """Track the goal centroid directly (the self-referenced own-objective)."""
        del obs
        return goal.astype(np.float64, copy=True)


@register_partner("cocarry_selfish_effort")
class CoCarrySelfishEffortPartner(_CoCarryShiftedBase):
    """Effort-minimising self-interested teammate (ADR-026 §Decision 4; ADR-009).

    Matched-impedance class + matched gains, tracking the SAME cooperative
    bar-end target as the reference; the ONLY shift is the objective: it
    **minimises its own actuation** via a dead-band (no command below
    :data:`_EFFORT_DEADBAND_M` of own error) and a down-scaled command
    authority (:data:`_EFFORT_AUTHORITY`), so it contributes lazily and lets
    the partner lead. Competent on large errors, but under-contributing by its
    own effort-minimising objective. Reads task leaves only (I3).
    """

    def _cartesian_command(
        self, error: NDArray[np.float64], obs: Mapping[str, object]
    ) -> NDArray[np.float64]:
        """Matched proportional law, dead-banded + down-scaled (minimise own effort)."""
        del obs
        if float(np.linalg.norm(error)) < _EFFORT_DEADBAND_M:
            return np.zeros(_XYZ, dtype=np.float64)
        cmd = np.clip(self._KP * error, -self._STEP_MAX_M, self._STEP_MAX_M)
        return _EFFORT_AUTHORITY * cmd


@register_partner("cocarry_selfish_station")
class CoCarrySelfishStationPartner(_CoCarryShiftedBase):
    """Station-keeping self-interested teammate (ADR-026 §Decision 4; ADR-009).

    Matched-impedance class + matched gains; the ONLY shift is the objective:
    it holds its **own start bar-end pose** (cached in world frame on the first
    act), indifferent to transport. Perfectly competent at its own objective
    (stay put); whether the pair clears the joint capability gate is the
    Stage-2 empirical question (a station-keeper that fails C_min is EXCLUDED,
    per the prereg's >= 2-competent-variant requirement). Reads task leaves
    only (I3).
    """

    def __init__(self, spec: PartnerSpec) -> None:
        """Bind the spec and add the cached station target (ADR-026 §Decision 4)."""
        super().__init__(spec)
        self._station_world: NDArray[np.float64] | None = None

    def reset(self, *, seed: int | None = None) -> None:
        """Clear the cached station target + base within-episode state (ADR-009)."""
        super().reset(seed=seed)
        self._station_world = None

    def _target_world(
        self, goal: NDArray[np.float64], obs: Mapping[str, object]
    ) -> NDArray[np.float64]:
        """Hold the own start bar-end pose (cached world-frame TCP on the first act)."""
        if self._station_world is not None:
            return self._station_world
        qpos = self._read_qpos(obs)
        if qpos is None:  # defensive: fall back to the cooperative target until qpos arrives
            return self._cooperative_target_world(goal)
        tcp_base = self._provider.fk_tcp_position(qpos)
        self._station_world = self._base_xyz + self._base_rot @ tcp_base
        return self._station_world
