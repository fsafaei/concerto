# SPDX-License-Identifier: Apache-2.0
"""Property tests for AoIClock (T2.2).

ADR-003 §Decision: AoI (Age of Information) is the latency proxy carried in
every CommPacket; the clock implements the freshness semantics from
notes/tier2/41_ballotta_talak_2024.md (AoI = elapsed time since the last
fresh sample for that uid).

The brief mandates writing this test file *before* the implementation; the
properties below are the contract ``chamber.comm.aoi.AoIClock`` must satisfy.
Hypothesis runs with ``derandomize=True`` so CI is byte-identical (P6).
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from chamber.comm.aoi import AoIClock

_UIDS: tuple[str, ...] = ("a", "b", "c")


@dataclass(frozen=True)
class _TickOp:
    dt: float


@dataclass(frozen=True)
class _FreshOp:
    uid: str


@dataclass(frozen=True)
class _ResetOp:
    pass


_uids = st.sampled_from(_UIDS)
_dt = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
_op = st.one_of(
    _dt.map(_TickOp),
    _uids.map(_FreshOp),
    st.just(_ResetOp()),
)


# ---------------------------------------------------------------------------
# Reference-implementation property test (subsumes all individual properties)
# ---------------------------------------------------------------------------


@settings(derandomize=True, max_examples=200)
@given(ops=st.lists(_op, max_size=50))
def test_aoi_matches_reference_implementation(ops: list[object]) -> None:
    """AoIClock matches a dict-of-floats reference under arbitrary operations.

    Reference semantics (notes/tier2/41 Ballotta & Talak):
      - mark_fresh(uid) sets aoi[uid] = 0 (fresh sample arrived);
      - tick(dt) increments every registered uid's aoi by dt;
      - reset() clears all registrations;
      - uids are auto-registered on first mark_fresh.
    """
    clock = AoIClock()
    ref: dict[str, float] = {}

    for op in ops:
        if isinstance(op, _ResetOp):
            clock.reset()
            ref.clear()
        elif isinstance(op, _FreshOp):
            clock.mark_fresh(op.uid)
            ref[op.uid] = 0.0
        elif isinstance(op, _TickOp):
            clock.tick(op.dt)
            for uid in ref:
                ref[uid] += op.dt
        else:  # pragma: no cover — exhaustive
            msg = f"unhandled op: {op!r}"
            raise AssertionError(msg)

        for uid, expected in ref.items():
            actual = clock.aoi(uid)
            mismatch = f"AoI mismatch for {uid!r}: ref={expected}, clock={actual}"
            assert actual == pytest.approx(expected, abs=1e-12), mismatch
            assert actual >= 0.0


# ---------------------------------------------------------------------------
# Targeted properties (named for failure-message clarity)
# ---------------------------------------------------------------------------


@settings(derandomize=True, max_examples=100)
@given(dts=st.lists(_dt, min_size=1, max_size=20))
def test_aoi_monotone_non_decreasing_between_fresh(dts: list[float]) -> None:
    """Between two mark_fresh() calls, AoI is monotone non-decreasing (ADR-003 §Decision)."""
    clock = AoIClock()
    clock.mark_fresh("a")
    last = clock.aoi("a")
    for dt in dts:
        clock.tick(dt)
        current = clock.aoi("a")
        assert current >= last, f"AoI non-monotone: {last} -> {current}"
        last = current


@settings(derandomize=True, max_examples=100)
@given(dts=st.lists(_dt, min_size=1, max_size=20))
def test_aoi_zeros_on_mark_fresh(dts: list[float]) -> None:
    """mark_fresh(uid) drives aoi(uid) to exactly zero (ADR-003 §Decision)."""
    clock = AoIClock()
    clock.mark_fresh("a")
    for dt in dts:
        clock.tick(dt)
    clock.mark_fresh("a")
    assert clock.aoi("a") == 0.0


@settings(derandomize=True, max_examples=50)
@given(dts=st.lists(_dt, min_size=0, max_size=10))
def test_reset_zeros_all_uids(dts: list[float]) -> None:
    """reset() clears every registered uid (T2.2 acceptance criterion)."""
    clock = AoIClock()
    for uid in _UIDS:
        clock.mark_fresh(uid)
    for dt in dts:
        clock.tick(dt)
    clock.reset()
    for uid in _UIDS:
        with pytest.raises(KeyError):
            clock.aoi(uid)


@settings(derandomize=True, max_examples=100)
@given(
    dts=st.lists(_dt, min_size=1, max_size=10),
    target=_uids,
)
def test_per_uid_independence(dts: list[float], target: str) -> None:
    """mark_fresh(target) does not perturb other uids' AoI (ADR-003 §Decision)."""
    clock = AoIClock()
    for uid in _UIDS:
        clock.mark_fresh(uid)
    for dt in dts:
        clock.tick(dt)
    others_before = {uid: clock.aoi(uid) for uid in _UIDS if uid != target}
    clock.mark_fresh(target)
    for uid, expected in others_before.items():
        assert clock.aoi(uid) == pytest.approx(expected, abs=1e-12)


# ---------------------------------------------------------------------------
# Argument-validation properties
# ---------------------------------------------------------------------------


def test_negative_tick_rejected() -> None:
    """Negative dt is rejected: AoI cannot run backwards (ADR-003 §Decision)."""
    clock = AoIClock()
    clock.mark_fresh("a")
    with pytest.raises(ValueError, match="non-negative"):
        clock.tick(-1.0)


def test_aoi_unknown_uid_raises() -> None:
    """aoi(uid) for an unregistered uid raises KeyError (loud-fail per ADR-003)."""
    clock = AoIClock()
    with pytest.raises(KeyError):
        clock.aoi("missing")


def test_snapshot_returns_independent_copy() -> None:
    """snapshot() returns a dict the caller can mutate without affecting the clock."""
    clock = AoIClock()
    clock.mark_fresh("a")
    snap = clock.snapshot()
    snap["a"] = 999.0
    assert clock.aoi("a") == 0.0


def test_snapshot_reflects_current_state() -> None:
    """snapshot() returns the current per-uid AoI map (T2.2)."""
    clock = AoIClock()
    clock.mark_fresh("a")
    clock.mark_fresh("b")
    clock.tick(0.5)
    snap = clock.snapshot()
    assert snap == {"a": 0.5, "b": 0.5}
