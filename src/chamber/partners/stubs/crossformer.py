# SPDX-License-Identifier: Apache-2.0
"""CrossFormer partner stub — Phase-1 entry (ADR-010 §Decision Option B).

Phase-0 ships only the interface; Phase-1 wires up CrossFormer's frozen
zero-shot generalist via masked-modality + embodiment-specific readout
tokens (ADR-010 §Rationale, citing notes/tier2/33_crossformer.md). The
action-chunking cadence (100-action @ 20 Hz bimanual; 4-action @ 5-15 Hz
single-arm) sets the worst-case mid-chunk CBF interception latency budget
and lives downstream in :mod:`chamber.envs.action_repeat`.

Calling :meth:`CrossFormerPartner.act` or :meth:`CrossFormerPartner.reset`
raises :class:`NotImplementedError` referencing the Phase-1 ticket.
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
_PHASE1_TICKET = "https://github.com/concerto-org/concerto/issues/<TBD-Phase-1-crossformer>"


@register_partner("crossformer_zero_shot")
class CrossFormerPartner(PartnerBase):
    """CrossFormer frozen zero-shot generalist — Phase-1 stub (ADR-010 §Decision Option B).

    Frozen across all 20 cross-embodiment classes; no per-task LoRA budget
    (ADR-010 §Decision: zero fine-tuning cost). Per ADR-010 §Risks the
    zero-shot deployment must pass a per-task ablation before use as a
    primary partner; the stub does not bypass that gate.
    """

    def reset(self, *, seed: int | None = None) -> NoReturn:
        """Phase-1 stub — raises (ADR-010 §Decision Option B).

        Args:
            seed: Accepted for Protocol conformance only.

        Raises:
            NotImplementedError: Always; the real zero-shot inference
                harness is Phase-1 work.
        """
        del seed
        raise NotImplementedError(
            f"CrossFormerPartner.reset is Phase-1 work; see ADR-010 §Decision and {_PHASE1_TICKET}."
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
            NotImplementedError: Always; the real action-chunk inference
                is Phase-1 work.
        """
        del obs, deterministic
        raise NotImplementedError(
            f"CrossFormerPartner.act is Phase-1 work; see ADR-010 §Decision and {_PHASE1_TICKET}."
        )


__all__ = ["CrossFormerPartner"]
