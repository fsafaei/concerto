# SPDX-License-Identifier: Apache-2.0
"""CommChannel Protocol — method-level contract for inter-agent communication.

ADR-003 §Decision (both fixed-format and learned-overlay channels implement
this Protocol); ADR-006 §Decision (URLLC-anchored numeric bounds are
maintained by implementations, not the Protocol itself).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class CommChannel(Protocol):
    """Contract for a communication channel between agents in a shared env.

    ADR-003 §Decision: every concrete channel (fixed-format in M2,
    optional learned overlay in M2) implements exactly this interface.
    ADR-006 §Decision: implementations are responsible for honouring
    the URLLC-anchored latency / drop-rate numeric bounds.
    """

    def reset(self) -> None:
        """Reset channel state at episode start.

        ADR-003 §Decision: called by CommShapingWrapper on env.reset().
        """
        ...

    def encode(self, state: object) -> dict[str, object]:
        """Encode current env state into a comm observation dict.

        ADR-003 §Decision; ADR-006 §Decision.

        Args:
            state: Env-side state (obs dict or raw physics state); implementation-defined.

        Returns:
            A dict of comm-channel observations to inject into obs["comm"].
        """
        ...
