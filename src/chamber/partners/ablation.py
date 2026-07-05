# SPDX-License-Identifier: Apache-2.0
"""Partner-ablation instrument — the zero-action partner seat (ADR-027 §Admission protocol).

The A2 two-robot-infeasibility check runs "the best single-robot /
partner-ablated variant" of a task (ADR-027 §Admission protocol; the
coupling positive-control of ADR-026 §Decision 2). Where a task has no
dedicated single-robot condition, the partner seat is *zeroed in the
task's own terms*: the partner remains in the action dict so the env
steps unchanged, but its action stream is identically zero. This module
is that instrument — a registered partner class so the ablated cell's
result bundle records a real, registry-resolvable partner identity
(ADR-028 §Decision 3 partner-hash custody) instead of an ad-hoc lambda.

Distinct from :class:`chamber.partners.heuristic.ScriptedHeuristicPartner`:
the heuristic partner is a *policy* that happens to emit zeros at its
default target (ADR-002 §Revision 2026-05-21); this class is an
*intervention* that emits zeros by construction, so an A2 cell's
"partner ablated" label is guaranteed by code, not by an empirical
property of another policy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from chamber.partners.interface import PartnerBase
from chamber.partners.registry import register_partner

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping

    from numpy.typing import NDArray

    from chamber.partners.api import PartnerSpec

#: Registry name of the ablation instrument (ADR-027 §Admission protocol).
PARTNER_ABLATED_ZERO_CLASS: str = "partner_ablated_zero"


@register_partner(PARTNER_ABLATED_ZERO_CLASS)
class PartnerAblatedZero(PartnerBase):
    """Zero-action partner seat for A2 partner-ablated cells (ADR-027 §Admission protocol).

    Emits an all-zero action of length ``spec.extra["action_dim"]``
    regardless of the observation — the partner is physically removed /
    zeroed in the task's own terms (ADR-026 §Decision 2's single-robot
    side of the coupling positive-control). Deterministic and stateless
    (ADR-009 §Decision).
    """

    def __init__(self, spec: PartnerSpec) -> None:
        """Bind the action dimensionality from ``spec.extra`` (ADR-027 §Admission protocol).

        Raises:
            ValueError: When ``spec.extra["action_dim"]`` is missing or
                not a positive integer.
        """
        super().__init__(spec)
        raw = spec.extra.get("action_dim")
        try:
            action_dim = int(raw) if raw is not None else 0
        except ValueError as exc:
            msg = f"action_dim must be a positive int; got {raw!r}"
            raise ValueError(msg) from exc
        if action_dim < 1:
            msg = f"action_dim must be a positive int; got {raw!r}"
            raise ValueError(msg)
        self._action_dim: int = action_dim

    def reset(self, *, seed: int | None = None) -> None:
        """No-op — the instrument is stateless (ADR-009 §Decision)."""
        del seed

    def act(
        self,
        obs: Mapping[str, object],
        *,
        deterministic: bool = True,
    ) -> NDArray[np.floating]:
        """Return the all-zero action (ADR-027 §Admission protocol A2).

        Args:
            obs: Ignored — the ablated seat observes nothing.
            deterministic: Ignored — zeros are zeros.

        Returns:
            Float32 zeros of the bound ``action_dim``.
        """
        del obs, deterministic
        return np.zeros(self._action_dim, dtype=np.float32)


__all__ = ["PARTNER_ABLATED_ZERO_CLASS", "PartnerAblatedZero"]
