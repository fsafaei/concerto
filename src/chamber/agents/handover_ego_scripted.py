# SPDX-License-Identifier: Apache-2.0
"""Scripted competent ego corrector for the Gate-0 handover-and-place spike.

Phase-2, non-gating ADR-026 research spike (invariant I1). For Gate-0 the ego is
**scripted, not trained** — a best-effort analytic corrector — so the spike runs on
CPU in hours and needs no GPU campaign. The trained cooperation residual is out of
scope (a later, separately-authorised stage).

What it does (executor-prompt Rev 2 mechanism). It observes the presented part's
*lateral offset* and *grasp-pose / orientation error* (both allowed under ADR-009
§Decision — pose visibility is permitted; presenter-policy access is not) and computes
the best corrective placement it can:

* **Lateral offset -> translation.** A competent six-axis ego cancels the lateral
  offset by *translating* its place motion, within its reach. Cheap, no re-grasp; this
  is why the lateral channel is the success tolerance, not the coupling channel.
* **Grasp-pose error -> in-grasp reorientation, else re-grasp.** It nulls the grasp-pose
  error in-grasp up to its wrist-correction range; if the remainder would still break
  the angular success window it requests a re-grasp. It requests the re-grasp whenever
  the wrist alone cannot resolve it, irrespective of the budget: affordability is a
  *physical* constraint the env enforces (the budget-mediated channel), and surfacing
  the request is what lets the Gate-0 diagnostic label a failure budget-mediated vs
  intrinsic.

The ego reads only the task spec and the presented physical state; it holds no
reference to the presenter and reads no presenter policy (ADR-009). It performs no
learning; if a *learned* ego is ever substituted here, the trainer must gate it via
``chamber.benchmarks.ego_ppo_trainer._assert_partner_is_frozen`` — out of scope for
Gate-0.

Its kinematic ranges (translation reach, wrist-correction range) are six-axis-arm
properties supplied at construction; the defaults mirror the NON-BINDING placeholders
in :mod:`chamber.envs.handover_place` and are overridden by the spike runner with the
Stage-0-derived values.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from chamber.envs.handover_place import (
    HANDOVER_DEFAULT_ANGULAR_WINDOW_DEG,
    HANDOVER_DEFAULT_TRANSLATION_RANGE_M,
    HANDOVER_DEFAULT_WRIST_CORRECTION_DEG,
    HANDOVER_EGO_DIM,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy.typing import NDArray


class ScriptedHandoverEgo:
    """Best-effort analytic ego corrector (ADR-026 §Decision; ADR-009 §Decision).

    Stateless across episodes; ``act`` maps one phase-1 observation to the 4-vector ego
    action ``[translate_x, translate_y, reorient_deg, regrasp_flag]``. Reads only the
    presented lateral offset, the grasp-pose error, and the task spec — never a
    presenter policy.
    """

    def __init__(
        self,
        *,
        translation_range_m: float = HANDOVER_DEFAULT_TRANSLATION_RANGE_M,
        wrist_correction_deg: float = HANDOVER_DEFAULT_WRIST_CORRECTION_DEG,
    ) -> None:
        """Bind the ego's translation reach + wrist-correction range (ADR-026; ADR-009)."""
        self.translation_range_m = float(translation_range_m)
        self.wrist_correction_deg = float(wrist_correction_deg)

    def reset(self, *, seed: int | None = None) -> None:
        """No-op reset (the corrector is stateless; ADR-026). ``seed`` is ignored."""
        del seed

    def act(self, obs: Mapping[str, Any]) -> NDArray[np.float64]:
        """Compute the corrective placement for one presented part (ADR-026; ADR-009).

        Translates the lateral offset (clipped to reach) and reorients the grasp-pose
        error in-grasp (clipped to the wrist-correction range); requests a re-grasp iff
        the wrist alone leaves a residual that would break the angular success window.
        Raises if called before the presentation step has populated the observation.
        """
        lateral = obs.get("lateral_offset")
        grasp_pose_error = obs.get("grasp_pose_error_deg")
        if lateral is None or grasp_pose_error is None:
            raise RuntimeError(
                "ScriptedHandoverEgo.act called before the presentation step "
                "(lateral_offset / grasp_pose_error_deg is None)"
            )
        lateral_offset = np.asarray(lateral, dtype=np.float64).reshape(-1)
        grasp_pose_error_deg = float(grasp_pose_error)
        spec = obs["spec"]
        angular_window_deg = float(
            spec.get("angular_window_deg", HANDOVER_DEFAULT_ANGULAR_WINDOW_DEG)
        )

        # Lateral -> translation, clipped to the ego's reach.
        translation = -lateral_offset
        trans_mag = float(np.linalg.norm(translation))
        if trans_mag > self.translation_range_m and trans_mag > 0.0:
            translation = translation * (self.translation_range_m / trans_mag)

        # Grasp-pose -> in-grasp reorientation, clipped to the wrist-correction range.
        reorient = float(
            np.clip(-grasp_pose_error_deg, -self.wrist_correction_deg, self.wrist_correction_deg)
        )
        # Residual the wrist alone cannot remove; request a re-grasp iff it would break
        # the angular window. Budget affordability is the env's call, not the ego's.
        residual_after_wrist = max(0.0, abs(grasp_pose_error_deg) - self.wrist_correction_deg)
        regrasp_flag = 1.0 if residual_after_wrist >= angular_window_deg else 0.0

        action = np.zeros(HANDOVER_EGO_DIM, dtype=np.float64)
        action[0] = translation[0]
        action[1] = translation[1]
        action[2] = reorient
        action[3] = regrasp_flag
        return action
