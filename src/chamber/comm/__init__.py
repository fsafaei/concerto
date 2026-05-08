# SPDX-License-Identifier: Apache-2.0
"""CHAMBER communication stack — fixed-format channel + URLLC degradation.

ADR-003 §Decision (mandatory fixed-format channel + opt-in learned overlay);
ADR-006 §Decision (URLLC-anchored numeric bounds for latency, jitter, drop).

Public surface:

- :class:`CommChannel` — Protocol (re-exported from :mod:`chamber.comm.api`).
- :class:`CommPacket`, :class:`Pose`, :class:`TaskStatePredicate`,
  :data:`SCHEMA_VERSION` — wire format.
- :class:`FixedFormatCommChannel` — the mandatory fixed-format encoder/decoder.
- :class:`AoIClock` — per-uid Age-of-Information accounting.
- :class:`ChamberCommError`, :class:`ChamberCommQPSaturationWarning` — error
  and warning types.

Subsequent PRs (T2.4-T2.6) add the learned-overlay slot, the
:class:`CommDegradationWrapper`, and the URLLC profile table.
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
from chamber.comm.degradation import (
    CommDegradationStats,
    CommDegradationWrapper,
    DegradationProfile,
)
from chamber.comm.errors import ChamberCommError, ChamberCommQPSaturationWarning
from chamber.comm.fixed_format import FixedFormatCommChannel
from chamber.comm.learned_overlay import LearnedOverlay
from chamber.comm.profiles import URLLC_3GPP_R17

__all__ = [
    "SCHEMA_VERSION",
    "URLLC_3GPP_R17",
    "AoIClock",
    "ChamberCommError",
    "ChamberCommQPSaturationWarning",
    "CommChannel",
    "CommDegradationStats",
    "CommDegradationWrapper",
    "CommPacket",
    "DegradationProfile",
    "FixedFormatCommChannel",
    "LearnedOverlay",
    "Pose",
    "TaskStatePredicate",
]
