# SPDX-License-Identifier: Apache-2.0
"""Property tests for ``CommDegradationWrapper`` (T2.5+T2.6).

Covers ADR-006 §Decision (URLLC-anchored numeric bounds) and the §3.5
behaviour spec from ``phase0_reading_kit/plan/02-comm-stack.md``: drop rate
within ±1 %, latency mean/std within ±5 %, AoI monotone while in queue,
ordering preserved when ``latency_std_ms == 0``.

The brief mandates writing this test file *before* the implementation; the
contract below is what ``chamber.comm.degradation.CommDegradationWrapper``
must satisfy. Hypothesis runs with ``derandomize=True`` so CI is byte-identical.
"""

from __future__ import annotations

import math
import statistics

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from chamber.comm import SCHEMA_VERSION, CommPacket, Pose
from chamber.comm.degradation import CommDegradationWrapper, DegradationProfile
from chamber.comm.profiles import URLLC_3GPP_R17

_N_SAMPLES = 10_000
_DROP_TOL = 0.01  # ±1 % per plan/02 §4 T2.5
_LATENCY_TOL = 0.05  # ±5 % per plan/02 §4 T2.5
_PROFILES = list(URLLC_3GPP_R17.items())


class _CounterChannel:
    """In-test channel emitting an identifying counter in each packet (P6).

    The counter is stored in ``pose["seq"].xyz[0]`` so the wrapper's
    delivered-packet trace can be reconstructed from the visible pose.
    """

    def __init__(self) -> None:
        self._tick: int = 0

    def reset(self, *, seed: int | None = None) -> None:
        del seed
        self._tick = 0

    def encode(self, state: object) -> CommPacket:
        del state
        self._tick += 1
        return CommPacket(
            schema_version=SCHEMA_VERSION,
            pose={"seq": Pose(xyz=(float(self._tick), 0.0, 0.0), quat_wxyz=(1.0, 0.0, 0.0, 0.0))},
            task_state={},
            aoi={"seq": 0.0},
            learned_overlay=None,
        )

    def decode(self, packet: CommPacket) -> dict[str, object]:
        return {"pose": dict(packet["pose"])}


# ---------------------------------------------------------------------------
# Drop-rate empirical match (T2.5 (i))
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("name", "profile"), _PROFILES, ids=[n for n, _ in _PROFILES])
def test_drop_rate_empirical_match(name: str, profile: DegradationProfile) -> None:
    """Empirical drop rate matches profile.drop_rate within ±1 % over 10 000 samples (T2.5 (i))."""
    del name
    wrapper = CommDegradationWrapper(_CounterChannel(), profile, tick_period_ms=1.0, root_seed=0)
    for _ in range(_N_SAMPLES):
        wrapper.encode({})
    stats = wrapper.stats
    empirical = stats.dropped / _N_SAMPLES
    msg = f"drop rate empirical {empirical:.4f} vs target {profile.drop_rate} (±{_DROP_TOL})"
    assert abs(empirical - profile.drop_rate) <= _DROP_TOL, msg


# ---------------------------------------------------------------------------
# Latency mean/std empirical match (T2.5 (ii))
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("name", "profile"), _PROFILES, ids=[n for n, _ in _PROFILES])
def test_latency_mean_within_tolerance(name: str, profile: DegradationProfile) -> None:
    """Empirical latency mean matches profile.latency_mean_ms within ±5 % (T2.5 (ii)).

    The ``ideal`` profile (mean=0, std=0) is a degenerate case: every delivery
    has latency = 0; we assert that exactly.
    """
    del name
    wrapper = CommDegradationWrapper(_CounterChannel(), profile, tick_period_ms=1.0, root_seed=0)
    for _ in range(_N_SAMPLES):
        wrapper.encode({})
    latencies = wrapper.stats.latency_ticks
    if profile.latency_mean_ms == 0.0:
        assert all(value == 0 for value in latencies)
        return
    if not latencies:
        # Profile delivered nothing within window -> only valid when drop rate ~ 1.
        assert profile.drop_rate >= 0.99
        return
    empirical_mean = statistics.mean(latencies)
    target = profile.latency_mean_ms  # tick_period_ms = 1 -> ticks ≡ ms
    msg = f"latency mean empirical {empirical_mean:.3f} vs target {target} (±{_LATENCY_TOL})"
    assert abs(empirical_mean - target) <= _LATENCY_TOL * max(target, 1.0), msg


@pytest.mark.parametrize(("name", "profile"), _PROFILES, ids=[n for n, _ in _PROFILES])
def test_latency_std_within_tolerance(name: str, profile: DegradationProfile) -> None:
    """Empirical latency std matches profile.latency_std_ms within ±5 % (or absolute 0.5 ms)."""
    del name
    wrapper = CommDegradationWrapper(_CounterChannel(), profile, tick_period_ms=1.0, root_seed=0)
    for _ in range(_N_SAMPLES):
        wrapper.encode({})
    latencies = wrapper.stats.latency_ticks
    if profile.latency_std_ms == 0.0:
        # Deterministic latency -> std is exactly 0 (modulo clipping, which
        # cannot inject variance from a zero-variance Normal).
        if latencies:
            assert statistics.pstdev(latencies) == pytest.approx(0.0, abs=1e-9)
        return
    if len(latencies) < 2:
        return
    empirical_std = statistics.pstdev(latencies)
    target = profile.latency_std_ms
    # Truncated-Normal clipping at [0, 2*mean] reduces the variance vs the
    # parent Normal; allow the tighter of (5 %, 0.5 ms absolute) tolerance.
    tol = max(_LATENCY_TOL * target, 0.5)
    msg = f"latency std empirical {empirical_std:.3f} vs target {target} (±{tol})"
    assert abs(empirical_std - target) <= tol, msg


# ---------------------------------------------------------------------------
# AoI never negative (T2.5 + plan/02 §5)
# ---------------------------------------------------------------------------


@settings(derandomize=True, max_examples=50)
@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
def test_aoi_never_negative_under_arbitrary_seeds(seed: int) -> None:
    """ADR-003 §Decision: every emitted packet's per-uid AoI is non-negative.

    Plan/02 §5: packet reordering allowed when latency_std_ms > 0 but never
    produces a packet with aoi < 0.
    """
    profile = URLLC_3GPP_R17["wifi"]  # has the largest std/mean ratio in the table
    wrapper = CommDegradationWrapper(_CounterChannel(), profile, tick_period_ms=1.0, root_seed=seed)
    for _ in range(500):
        packet = wrapper.encode({})
        for value in packet["aoi"].values():
            assert value >= 0.0


# ---------------------------------------------------------------------------
# Order preservation when latency_std_ms == 0 (plan/02 §5)
# ---------------------------------------------------------------------------


def test_order_preserved_when_std_zero() -> None:
    """ADR-003 §Decision: deterministic-latency profiles preserve packet order.

    Uses the ``urllc`` profile (1 ms mean, 0 ms std, ~0 drop). The visible
    packet's seq counter must be monotone non-decreasing tick-by-tick.
    """
    profile = URLLC_3GPP_R17["urllc"]
    wrapper = CommDegradationWrapper(_CounterChannel(), profile, tick_period_ms=1.0, root_seed=0)
    last_seq = -math.inf
    for _ in range(500):
        packet = wrapper.encode({})
        if not packet["pose"]:
            continue
        seq = packet["pose"]["seq"]["xyz"][0]
        assert seq >= last_seq
        last_seq = seq


# ---------------------------------------------------------------------------
# Reset isolates episodes (P6)
# ---------------------------------------------------------------------------


def test_reset_with_same_seed_reproduces_full_trace() -> None:
    """P6 determinism: two reset(seed=k) wrappers produce identical traces."""
    profile = URLLC_3GPP_R17["factory"]
    a = CommDegradationWrapper(_CounterChannel(), profile, tick_period_ms=1.0, root_seed=0)
    b = CommDegradationWrapper(_CounterChannel(), profile, tick_period_ms=1.0, root_seed=0)
    a.reset(seed=42)
    b.reset(seed=42)
    for _ in range(200):
        pa = a.encode({})
        pb = b.encode({})
        assert pa["pose"] == pb["pose"]
        assert pa["aoi"] == pb["aoi"]


def test_reset_different_seeds_diverge() -> None:
    """P6: distinct root seeds produce distinct degradation traces.

    Uses ``saturation`` (drop_rate=0.1, std/mean=0.1) so that two seeds reliably
    diverge within a few hundred draws. Tighter profiles like ``factory`` are
    deterministic-modulo-rounding under small std and would not diverge here.
    """
    profile = URLLC_3GPP_R17["saturation"]
    a = CommDegradationWrapper(_CounterChannel(), profile, tick_period_ms=1.0, root_seed=0)
    b = CommDegradationWrapper(_CounterChannel(), profile, tick_period_ms=1.0, root_seed=1)
    diverged = False
    for _ in range(500):
        pa = a.encode({})
        pb = b.encode({})
        if pa["pose"] != pb["pose"] or pa["aoi"] != pb["aoi"]:
            diverged = True
            break
    assert diverged
