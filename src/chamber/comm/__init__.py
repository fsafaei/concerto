# SPDX-License-Identifier: Apache-2.0
"""CHAMBER communication stack — fixed-format channel + URLLC degradation.

ADR-003 §Decision (mandatory fixed-format channel + opt-in learned overlay);
ADR-006 §Decision (URLLC-anchored numeric bounds for latency, jitter, drop).

PR1 of M2 ships the public api (Protocol re-export, packet TypedDicts,
schema version, error/warning types). The fixed-format encoder, AoI clock,
degradation wrapper, learned-overlay slot, and URLLC profile table land in
follow-up PRs (T2.2-T2.6) per ``phase0_reading_kit/plan/02-comm-stack.md``.
"""

from __future__ import annotations

from chamber.comm.aoi import AoIClock
from chamber.comm.api import (
    SCHEMA_VERSION,
    CommChannel,
    CommPacket,
    Pose,
    TaskStatePredicate,
)
from chamber.comm.errors import ChamberCommError, ChamberCommQPSaturationWarning


class FixedFormatCommChannel:
    """Fixed-format inter-agent communication channel (M1 stub; replaced in T2.3).

    ADR-003 §Decision: the canonical channel encoding pose, task-state
    predicates, and AoI timestamps into a fixed-format packet. ADR-006
    §Decision: URLLC-anchored latency / drop-rate bounds applied externally
    by :class:`chamber.comm.degradation.CommDegradationWrapper` (T2.5).

    This class is the M1 carrier so ``chamber.benchmarks.stage0_smoke`` can
    instantiate a channel; PR3 of M2 (T2.3) replaces it with the real encoder
    in ``chamber/comm/fixed_format.py`` and the Stage-0 wiring is updated in
    T2.9 to compose with the degradation wrapper.

    Args:
        latency_ms: Legacy M1 kwarg, retained until T2.9 swaps Stage-0 to
            ``CommDegradationWrapper``. Ignored here; degradation is no longer
            an in-channel concern (ADR-003 §Decision; plan/02 §3.2).
        drop_rate: Legacy M1 kwarg; see ``latency_ms``.
    """

    def __init__(self, latency_ms: float = 0.0, drop_rate: float = 0.0) -> None:
        """Initialise the M1 stub channel (ADR-003 §Decision).

        Legacy ``latency_ms`` / ``drop_rate`` kwargs are retained for the
        Stage-0 smoke wiring until T2.9; they are ignored here because
        degradation is composed externally.
        """
        self._latency_ms = latency_ms
        self._drop_rate = drop_rate

    def reset(self, *, seed: int | None = None) -> None:
        """Reset channel state at episode start (ADR-003 §Decision).

        Args:
            seed: Optional RNG seed for the channel substream. Accepted to
                satisfy :class:`~chamber.comm.api.CommChannel`; the real
                seed wiring lands in T2.3.
        """
        del seed  # honoured by the real channel in T2.3

    def encode(self, state: object) -> CommPacket:
        """Encode env state into a comm-channel packet (ADR-003 §Decision).

        M1 stub returns an empty packet shell with the current schema version
        and no per-uid entries; the real encoder (T2.3) fills pose / task-state /
        AoI per uid and composes with :class:`~chamber.comm.degradation.CommDegradationWrapper`
        for URLLC-anchored degradation (ADR-006 §Decision).
        """
        del state  # consumed by the real encoder in T2.3
        return CommPacket(
            schema_version=SCHEMA_VERSION,
            pose={},
            task_state={},
            aoi={},
            learned_overlay=None,
        )

    def decode(self, packet: CommPacket) -> dict[str, object]:
        """Decode a comm-channel packet back into the typed state subset (ADR-003 §Decision).

        M1 stub returns an empty dict; the real decoder (T2.3) emits the
        pose / task-state / AoI subset of state keyed by uid.
        """
        del packet  # consumed by the real decoder in T2.3
        return {}


__all__ = [
    "SCHEMA_VERSION",
    "AoIClock",
    "ChamberCommError",
    "ChamberCommQPSaturationWarning",
    "CommChannel",
    "CommPacket",
    "FixedFormatCommChannel",
    "Pose",
    "TaskStatePredicate",
]
