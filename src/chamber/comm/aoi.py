# SPDX-License-Identifier: Apache-2.0
"""Per-uid Age-of-Information clock (T2.2).

ADR-003 §Decision: AoI is the latency proxy carried in every
:class:`~chamber.comm.api.CommPacket`. ADR-006 §Decision: AoI is the
URLLC-anchored latency-axis instrumentation for the ADR-007 Stage-2 CM
spike.

Semantics follow notes/tier2/41 (Ballotta & Talak 2024): for each uid,
AoI is the elapsed simulation time since the last fresh sample arrived.
``mark_fresh(uid)`` resets that uid to zero; ``tick(dt)`` advances every
registered uid by ``dt``. The clock holds no env-side dependency — wall
time is supplied by the caller (typically the env tick) so the class
remains pure-Python and trivially deterministic (P6).
"""

from __future__ import annotations


class AoIClock:
    """Per-uid Age-of-Information accounting (ADR-003 §Decision; ADR-006 §Decision).

    Usage::

        clock = AoIClock()
        clock.mark_fresh("panda")  # registers "panda" with AoI=0
        clock.tick(0.01)  # +10 ms for every registered uid
        clock.aoi("panda")  # -> 0.01

    Uids are auto-registered on first :meth:`mark_fresh`. :meth:`reset`
    clears all registrations so a fresh episode begins from a clean slate.
    """

    def __init__(self) -> None:
        """Create an empty clock with no uids registered (ADR-003 §Decision)."""
        self._aoi: dict[str, float] = {}

    def reset(self) -> None:
        """Clear every registered uid (ADR-003 §Decision; T2.2 acceptance criterion).

        After ``reset``, :meth:`aoi` raises ``KeyError`` for every uid until
        the next :meth:`mark_fresh`. This matches the channel's
        ``CommChannel.reset`` semantics: a new episode starts with no state.
        """
        self._aoi.clear()

    def mark_fresh(self, uid: str) -> None:
        """Mark ``uid`` as having received a fresh sample (AoI -> 0).

        ADR-003 §Decision: every fresh transmission zeroes that uid's AoI.
        Auto-registers ``uid`` on first call.
        """
        self._aoi[uid] = 0.0

    def tick(self, dt: float) -> None:
        """Advance time by ``dt`` for every registered uid (ADR-003 §Decision).

        Args:
            dt: Elapsed simulation time in seconds. Must be non-negative;
                AoI cannot run backwards (notes/tier2/41 §V definition).

        Raises:
            ValueError: when ``dt`` is negative.
        """
        if dt < 0.0:
            msg = f"tick dt must be non-negative; got {dt!r}"
            raise ValueError(msg)
        for uid in self._aoi:
            self._aoi[uid] += dt

    def aoi(self, uid: str) -> float:
        """Return the current AoI for ``uid`` (ADR-003 §Decision).

        Raises:
            KeyError: if ``uid`` is not registered (loud-fail per ADR-003).
        """
        return self._aoi[uid]

    def snapshot(self) -> dict[str, float]:
        """Return an independent copy of the per-uid AoI map (ADR-003 §Decision).

        The returned dict is decoupled from the clock's internal state, so
        callers may freely mutate it (typical use: pack into the next
        ``CommPacket["aoi"]`` field).
        """
        return dict(self._aoi)
