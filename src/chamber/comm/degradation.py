# SPDX-License-Identifier: Apache-2.0
"""URLLC-anchored comm degradation wrapper (T2.5).

ADR-006 §Decision: URLLC-anchored numeric bounds — latency 1-100 ms,
jitter μs-10 ms, drop 10⁻⁶ to 10⁻² — are *applied externally* by this
wrapper rather than by the channel itself, so the encoding logic in
:class:`chamber.comm.fixed_format.FixedFormatCommChannel` stays pure and
testable. ADR-007 Stage-2 CM sweep ranges are exposed via
:data:`chamber.comm.profiles.URLLC_3GPP_R17`.

Behaviour (plan/02 §3.5):

  1. With probability ``profile.drop_rate``, the freshly-encoded packet is
     dropped; the wrapper returns the previous visible packet with all
     per-uid AoI values incremented by one tick.
  2. Otherwise the packet is queued with a delay drawn from
     ``Normal(latency_mean_ms, latency_std_ms)`` clipped to
     ``[0, 2 * latency_mean_ms]``. Packets due at or before the current
     tick are released; the most-recently-due packet becomes the visible
     one, with its per-uid AoI incremented by the time it spent in queue.

Determinism (P6): every random draw is seeded via
``concerto.training.seeding.derive_substream("comm.degrade", ...)`` so two
wrappers reset with the same seed produce byte-identical traces.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

from chamber.comm.api import SCHEMA_VERSION, CommChannel, CommPacket
from concerto.training.seeding import derive_substream

#: Substream name for the wrapper's degradation RNG (P6 determinism).
_SUBSTREAM_NAME = "comm.degrade"


@dataclass(frozen=True)
class DegradationProfile:
    """Latency / jitter / drop bounds for a single experimental condition.

    ADR-006 §Decision: anchor values come from the URLLC + 3GPP R17 industrial
    trial corpus. The named profiles in :mod:`chamber.comm.profiles` cover the
    Phase-0 Stage-2 CM sweep table.

    Args:
        latency_mean_ms: Mean of the Gaussian latency draw (ms).
        latency_std_ms: Std of the Gaussian latency draw (ms); the draw is
            clipped to ``[0, 2 * latency_mean_ms]``.
        drop_rate: Bernoulli drop probability per packet (per encode call).
    """

    latency_mean_ms: float
    latency_std_ms: float
    drop_rate: float


@dataclass(frozen=True)
class CommDegradationStats:
    """Telemetry counters surfaced for the Stage-2 CM spike (ADR-006 §Decision).

    Each delivered packet contributes one entry to ``latency_ticks``
    (the in-queue dwell time, in ticks). Each dropped packet increments
    ``dropped`` without producing a latency entry.
    """

    dropped: int = 0
    delivered: int = 0
    latency_ticks: tuple[int, ...] = field(default_factory=tuple)


@dataclass
class _Queued:
    release_tick: int
    send_tick: int
    packet: CommPacket


def _empty_packet() -> CommPacket:
    return CommPacket(
        schema_version=SCHEMA_VERSION,
        pose={},
        task_state={},
        aoi={},
        learned_overlay=None,
    )


class CommDegradationWrapper:
    """Compose-time degradation around a fixed-format channel (ADR-006 §Decision).

    Wraps any :class:`~chamber.comm.api.CommChannel` and injects
    URLLC-anchored latency / jitter / drop semantics described above. The
    inner channel is unaware of degradation; the wrapper is itself a
    :class:`CommChannel` so it composes transparently with
    :class:`chamber.envs.comm_shaping.CommShapingWrapper`.

    Args:
        channel: The inner channel whose packets will be degraded.
        profile: The degradation envelope (see :class:`DegradationProfile`).
        tick_period_ms: How many milliseconds each ``encode`` call represents.
            Defaults to ``1.0`` so the URLLC ms-denominated profiles map
            tick-for-millisecond.
        root_seed: Root seed for the wrapper's RNG (P6 determinism).
    """

    def __init__(
        self,
        channel: CommChannel,
        profile: DegradationProfile,
        *,
        tick_period_ms: float = 1.0,
        root_seed: int = 0,
    ) -> None:
        """Build the wrapper (ADR-006 §Decision)."""
        if tick_period_ms <= 0.0:
            msg = f"tick_period_ms must be positive; got {tick_period_ms!r}"
            raise ValueError(msg)
        self._channel = channel
        self._profile = profile
        self._tick_period_ms = tick_period_ms
        self._root_seed: int = root_seed
        self._rng = derive_substream(_SUBSTREAM_NAME, root_seed=self._root_seed).default_rng()
        self._tick: int = 0
        self._queue: list[_Queued] = []
        self._visible: CommPacket = _empty_packet()
        self._dropped: int = 0
        self._delivered: int = 0
        self._latency_ticks: list[int] = []

    @property
    def stats(self) -> CommDegradationStats:
        """Return a snapshot of telemetry counters (ADR-006 §Decision; ADR-007 Stage 2 CM)."""
        return CommDegradationStats(
            dropped=self._dropped,
            delivered=self._delivered,
            latency_ticks=tuple(self._latency_ticks),
        )

    def reset(self, *, seed: int | None = None) -> None:
        """Reset the wrapper, the inner channel, and the RNG (ADR-006 §Decision).

        Args:
            seed: Optional new root seed; otherwise the previously stored
                root seed is reused so episode resets stay reproducible.
        """
        if seed is not None:
            self._root_seed = seed
        self._channel.reset(seed=self._root_seed)
        self._rng = derive_substream(_SUBSTREAM_NAME, root_seed=self._root_seed).default_rng()
        self._tick = 0
        self._queue = []
        self._visible = _empty_packet()
        self._dropped = 0
        self._delivered = 0
        self._latency_ticks = []

    def encode(self, state: object) -> CommPacket:
        """Apply URLLC-anchored degradation to the inner channel's packet (ADR-006 §Decision).

        See module docstring for the per-step behaviour spec.
        """
        self._tick += 1
        fresh = self._channel.encode(state)
        if self._draw_drop():
            self._dropped += 1
            self._visible = self._age_packet(self._visible, by_ticks=1)
            return _copy_packet(self._visible)
        delay_ticks = self._draw_delay_ticks()
        self._queue.append(
            _Queued(release_tick=self._tick + delay_ticks, send_tick=self._tick, packet=fresh)
        )
        self._release_due()
        return _copy_packet(self._visible)

    def decode(self, packet: CommPacket) -> dict[str, object]:
        """Delegate decoding to the inner channel (ADR-003 §Decision)."""
        return self._channel.decode(packet)

    def _draw_drop(self) -> bool:
        """True iff the per-tick Bernoulli drop trial fires (P6 deterministic)."""
        if self._profile.drop_rate <= 0.0:
            return False
        return bool(self._rng.random() < self._profile.drop_rate)

    def _draw_delay_ticks(self) -> int:
        """Sample a clipped-Normal latency and convert to whole ticks (P6)."""
        mean = self._profile.latency_mean_ms
        std = self._profile.latency_std_ms
        upper = 2.0 * mean
        if std == 0.0:
            ms = max(0.0, min(mean, upper))
        else:
            rng_sample = float(self._rng.normal(loc=mean, scale=std))
            ms = max(0.0, min(rng_sample, upper))
        return round(ms / self._tick_period_ms)

    def _release_due(self) -> None:
        """Pop every queued packet whose release tick is <= now (ADR-006)."""
        if not self._queue:
            return
        due: list[_Queued] = [q for q in self._queue if q.release_tick <= self._tick]
        if not due:
            return
        self._queue = [q for q in self._queue if q.release_tick > self._tick]
        # The latest send_tick wins as the visible packet (most recent fresh sample).
        winner = max(due, key=lambda q: q.send_tick)
        for q in due:
            self._delivered += 1
            self._latency_ticks.append(q.release_tick - q.send_tick)
        dwell = self._tick - winner.send_tick
        self._visible = self._age_packet(winner.packet, by_ticks=dwell)

    @staticmethod
    def _age_packet(packet: CommPacket, *, by_ticks: int) -> CommPacket:
        """Return a copy of ``packet`` with all per-uid AoI advanced by ``by_ticks`` (ADR-003)."""
        if not packet["aoi"]:
            return packet
        aged_aoi = {uid: value + float(by_ticks) for uid, value in packet["aoi"].items()}
        aged: CommPacket = CommPacket(
            schema_version=packet["schema_version"],
            pose=dict(packet["pose"]),
            task_state=dict(packet["task_state"]),
            aoi=aged_aoi,
            learned_overlay=packet["learned_overlay"],
        )
        return aged


def _copy_packet(packet: CommPacket) -> CommPacket:
    """Return an independent shallow-copy of ``packet`` (caller cannot mutate visible)."""
    return CommPacket(
        schema_version=packet["schema_version"],
        pose=copy.copy(packet["pose"]),
        task_state=copy.copy(packet["task_state"]),
        aoi=copy.copy(packet["aoi"]),
        learned_overlay=packet["learned_overlay"],
    )
