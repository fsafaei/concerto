# SPDX-License-Identifier: Apache-2.0
"""URLLC-anchored degradation sweep profiles (T2.6).

ADR-006 §Decision: latency 1-100 ms, jitter μs-10 ms, drop 10⁻⁶ to 10⁻²
(URLLC + 3GPP Release 17 + arXiv 2501.12792 industrial-trial anchors). The
six named profiles below cover the sweep range the ADR-007 Stage-2 CM spike
will reproduce; the ``saturation`` row is held aside so the QP-saturation
property test (T2.7) can prove or refute ADR-006 risk #5 (R5).

Profiles are evaluated in production by composition with
:class:`chamber.comm.degradation.CommDegradationWrapper`; the channel itself
ships no degradation (ADR-003 §Decision).
"""

from __future__ import annotations

from chamber.comm.degradation import DegradationProfile

#: ADR-006 §Decision + ADR-007 Stage-2 CM sweep table.
#:
#: Six profiles ordered by aggressiveness (``ideal`` -> ``saturation``).
#: The exact numeric values are the project-frozen URLLC + 3GPP R17
#: anchors; revising any row requires an ADR amendment.
URLLC_3GPP_R17: dict[str, DegradationProfile] = {
    "ideal": DegradationProfile(latency_mean_ms=0.0, latency_std_ms=0.0, drop_rate=0.0),
    "urllc": DegradationProfile(latency_mean_ms=1.0, latency_std_ms=0.0, drop_rate=1e-6),
    "factory": DegradationProfile(latency_mean_ms=5.0, latency_std_ms=0.1, drop_rate=1e-4),
    "wifi": DegradationProfile(latency_mean_ms=10.0, latency_std_ms=1.0, drop_rate=1e-2),
    "lossy": DegradationProfile(latency_mean_ms=30.0, latency_std_ms=5.0, drop_rate=1e-2),
    "saturation": DegradationProfile(latency_mean_ms=100.0, latency_std_ms=10.0, drop_rate=1e-1),
}


__all__ = ["URLLC_3GPP_R17"]
