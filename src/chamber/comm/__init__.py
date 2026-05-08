# SPDX-License-Identifier: Apache-2.0
"""CHAMBER communication stack — fixed-format channel + URLLC degradation.

ADR-003 §Decision (both fixed-format and optional learned overlay);
ADR-006 §Decision (URLLC-anchored numeric bounds for latency, jitter, drop).

Real implementation is M2. This module exposes only the stub carrier needed
for the CommShapingWrapper in M1.
"""

from __future__ import annotations


class FixedFormatCommChannel:
    """Fixed-format inter-agent communication channel (stub for M1).

    ADR-003 §Decision: the canonical channel encoding pose, task-state
    predicates, and AoI timestamps into a fixed-size binary packet.
    ADR-006 §Decision: URLLC-anchored latency / drop-rate bounds applied
    by the degradation layer (implemented in M2).

    Args:
        latency_ms: One-way channel latency in milliseconds (M2 implements).
        drop_rate: Bernoulli drop probability per packet (M2 implements).
    """

    def __init__(self, latency_ms: float = 0.0, drop_rate: float = 0.0) -> None:
        """Initialise the stub channel with M2 latency/drop parameters.

        ADR-003 §Decision; ADR-006 §Decision.
        TODO(M2): initialise ring buffer and Bernoulli drop mask.
        """
        self._latency_ms = latency_ms
        self._drop_rate = drop_rate

    def reset(self) -> None:
        """Reset channel state at episode start.

        ADR-003 §Decision: called by CommShapingWrapper on env.reset().
        TODO(M2): flush ring buffer and AoI timestamps.
        """

    def encode(self, state: object) -> dict[str, object]:  # noqa: ARG002
        """Encode env state into a comm observation dict.

        ADR-003 §Decision; ADR-006 §Decision.
        TODO(M2): encode pose, task-state predicates, and AoI timestamps
        into a fixed-format packet; apply latency and drop degradation.

        Returns:
            Empty dict in M1; M2 fills with real comm data.
        """
        return {}
