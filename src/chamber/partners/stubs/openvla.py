# SPDX-License-Identifier: Apache-2.0
"""OpenVLA partner stub — Phase-1 entry (ADR-010 §Decision Option B).

Phase-0 ships only the interface so Phase-1 work is mechanical: drop in
the OpenVLA LoRA-adapted inference harness and replace the
:class:`NotImplementedError` raises. The :class:`PartnerSpec.weights_uri`
field carries the LoRA state-dict path; the per-DoF 256-bin action
tokenisation is the CBF interception boundary specified in
ADR-010 §Decision.

Calling :meth:`OpenVLAPartner.act` or :meth:`OpenVLAPartner.reset` raises
:class:`NotImplementedError` referencing the Phase-1 ticket so any code path
that touches the stub fails loudly rather than silently returning zeros.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NoReturn

from chamber.partners.interface import PartnerBase
from chamber.partners.registry import register_partner

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy as np
    from numpy.typing import NDArray

#: Phase-1 ticket reference embedded in :class:`NotImplementedError` messages.
_PHASE1_TICKET = "https://github.com/concerto-org/concerto/issues/<TBD-Phase-1-openvla>"


@register_partner("openvla_lora_specialist")
class OpenVLAPartner(PartnerBase):
    """OpenVLA LoRA-adapted specialist — Phase-1 stub (ADR-010 §Decision Option B).

    LoRA fine-tune protocol (ADR-010 §Decision; ADR-009 §Decision):
    10-150 demos, single GPU at ≤7.0 GB VRAM, 256-bin-per-DoF action
    tokenisation. The 5-15 Hz throughput ceiling is enforced at the env
    wrapper layer (``chamber.envs.action_repeat``), not here.
    """

    def reset(self, *, seed: int | None = None) -> NoReturn:
        """Phase-1 stub — raises (ADR-010 §Decision Option B).

        Args:
            seed: Accepted for Protocol conformance only.

        Raises:
            NotImplementedError: Always; the real LoRA inference harness is
                Phase-1 work.
        """
        del seed
        raise NotImplementedError(
            f"OpenVLAPartner.reset is Phase-1 work; see ADR-010 §Decision and {_PHASE1_TICKET}."
        )

    def act(
        self,
        obs: Mapping[str, object],
        *,
        deterministic: bool = True,
    ) -> NDArray[np.floating]:
        """Phase-1 stub — raises (ADR-010 §Decision Option B).

        Args:
            obs: Gymnasium observation mapping.
            deterministic: Accepted for Protocol conformance only.

        Raises:
            NotImplementedError: Always; the real 256-bin action-token
                inference is Phase-1 work.
        """
        del obs, deterministic
        raise NotImplementedError(
            f"OpenVLAPartner.act is Phase-1 work; see ADR-010 §Decision and {_PHASE1_TICKET}."
        )


__all__ = ["OpenVLAPartner"]
