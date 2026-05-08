# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the real ``FixedFormatCommChannel`` (T2.3).

Covers ADR-003 §Decision (encode/decode shape, schema-version mismatch loud-fail,
AoI advancement) and exercises the channel's deterministic seeding via
``concerto.training.seeding.derive_substream`` (P6).
"""

from __future__ import annotations

import pytest

from chamber.comm import (
    SCHEMA_VERSION,
    ChamberCommError,
    CommPacket,
    FixedFormatCommChannel,
    Pose,
)


def _pose(*, x: float = 0.0) -> Pose:
    return Pose(xyz=(x, 0.0, 0.0), quat_wxyz=(1.0, 0.0, 0.0, 0.0))


class TestEncodeShape:
    def test_packet_carries_current_schema_version(self) -> None:
        """ADR-003 §Decision: every packet stamps the schema version."""
        channel = FixedFormatCommChannel()
        packet = channel.encode({})
        assert packet["schema_version"] == SCHEMA_VERSION

    def test_packet_passes_through_pose_input(self) -> None:
        """ADR-003 §Decision: pose values arrive unchanged in the packet."""
        channel = FixedFormatCommChannel()
        state = {"pose": {"a": _pose(x=1.5), "b": _pose(x=-0.5)}}
        packet = channel.encode(state)
        assert packet["pose"]["a"] == _pose(x=1.5)
        assert packet["pose"]["b"] == _pose(x=-0.5)

    def test_packet_passes_through_task_state(self) -> None:
        """ADR-003 §Decision: task-state predicates arrive unchanged."""
        channel = FixedFormatCommChannel()
        state = {"task_state": {"a": {"grasp_side": "left"}}}
        packet = channel.encode(state)
        assert packet["task_state"]["a"] == {"grasp_side": "left"}

    def test_packet_learned_overlay_passes_through_none(self) -> None:
        """ADR-003 §Decision risk-mitigation 1: learned_overlay defaults to None."""
        channel = FixedFormatCommChannel()
        packet = channel.encode({})
        assert packet["learned_overlay"] is None

    def test_packet_learned_overlay_passes_through_value(self) -> None:
        """ADR-003 §Decision: learned_overlay is forwarded verbatim."""
        sentinel = object()
        channel = FixedFormatCommChannel()
        packet = channel.encode({"learned_overlay": sentinel})
        assert packet["learned_overlay"] is sentinel

    def test_non_mapping_state_yields_empty_packet(self) -> None:
        """ADR-003 §Decision: a non-mapping state ages prior uids without raising."""
        channel = FixedFormatCommChannel()
        packet = channel.encode(None)
        assert packet["pose"] == {}
        assert packet["task_state"] == {}


class TestEncodeAoI:
    def test_first_encode_marks_uids_fresh(self) -> None:
        """First encode of a uid sets aoi[uid] = 0 (ADR-003 §Decision; T2.2 semantics)."""
        channel = FixedFormatCommChannel()
        packet = channel.encode({"pose": {"a": _pose()}})
        assert packet["aoi"]["a"] == 0.0

    def test_missing_uid_ages_by_one_tick(self) -> None:
        """Encoding a state without uid X ages X's AoI by one tick (ADR-003 §Decision)."""
        channel = FixedFormatCommChannel()
        channel.encode({"pose": {"a": _pose()}})  # mark a fresh
        packet = channel.encode({"pose": {}})  # next tick: a stale by 1
        assert packet["aoi"]["a"] == 1.0

    def test_repeated_encodes_age_uids_linearly(self) -> None:
        """N consecutive empty encodes after a fresh sample give AoI = N (ADR-003)."""
        channel = FixedFormatCommChannel()
        channel.encode({"pose": {"a": _pose()}})
        packet = channel.encode({"pose": {}})
        for _ in range(3):
            packet = channel.encode({"pose": {}})
        assert packet["aoi"]["a"] == 4.0

    def test_re_marking_zeroes_aoi(self) -> None:
        """Re-supplying a uid's pose drives its AoI back to zero (ADR-003)."""
        channel = FixedFormatCommChannel()
        channel.encode({"pose": {"a": _pose()}})
        channel.encode({"pose": {}})  # ages a to 1
        packet = channel.encode({"pose": {"a": _pose()}})  # fresh again
        assert packet["aoi"]["a"] == 0.0


class TestReset:
    def test_reset_zeros_known_uids(self) -> None:
        """reset() clears AoI registrations (ADR-003 §Decision; T2.2)."""
        channel = FixedFormatCommChannel()
        channel.encode({"pose": {"a": _pose()}})
        channel.reset()
        packet = channel.encode({"pose": {}})
        # After reset, "a" was never re-introduced this episode.
        assert "a" not in packet["aoi"]

    def test_reset_with_seed_is_idempotent_on_rng(self) -> None:
        """reset(seed=k) twice gives the same RNG state (P6)."""
        channel = FixedFormatCommChannel()
        channel.reset(seed=42)
        first = channel._rng.random(8)
        channel.reset(seed=42)
        second = channel._rng.random(8)
        assert (first == second).all()

    def test_reset_default_keeps_prior_seed(self) -> None:
        """reset() with no seed re-seeds with the most-recent root_seed (P6)."""
        channel = FixedFormatCommChannel()
        channel.reset(seed=7)
        first = channel._rng.random(4)
        channel.reset()  # no seed -> re-use 7
        second = channel._rng.random(4)
        assert (first == second).all()


class TestDecodeRoundTrip:
    def test_decode_returns_typed_subset(self) -> None:
        """decode(encode(state)) reproduces pose / task_state / learned_overlay (T2.3)."""
        channel = FixedFormatCommChannel()
        state = {
            "pose": {"a": _pose(x=1.0)},
            "task_state": {"a": {"grasp_side": "right"}},
            "learned_overlay": None,
        }
        out = channel.decode(channel.encode(state))
        assert out["pose"] == state["pose"]
        assert out["task_state"] == state["task_state"]
        assert out["learned_overlay"] is None

    def test_decode_includes_aoi(self) -> None:
        """ADR-003 §Decision: decoded state surfaces AoI per uid."""
        channel = FixedFormatCommChannel()
        out = channel.decode(channel.encode({"pose": {"a": _pose()}}))
        assert out["aoi"] == {"a": 0.0}

    def test_decode_returns_independent_copy(self) -> None:
        """decode returns a fresh dict; mutation does not affect the channel state."""
        channel = FixedFormatCommChannel()
        out = channel.decode(channel.encode({"pose": {"a": _pose()}}))
        out["pose"]["a"] = _pose(x=999.0)  # type: ignore[index]
        # Subsequent encode is unaffected.
        next_packet = channel.encode({"pose": {}})
        assert "a" in next_packet["aoi"]


class TestSchemaVersionGuard:
    def test_schema_mismatch_raises_chamber_comm_error(self) -> None:
        """ADR-003 §"Schema evolution": decoder rejects mismatched packets (loud-fail)."""
        channel = FixedFormatCommChannel(schema_version=1)
        bad_packet = CommPacket(
            schema_version=2,
            pose={},
            task_state={},
            aoi={},
            learned_overlay=None,
        )
        with pytest.raises(ChamberCommError, match="schema_version"):
            channel.decode(bad_packet)

    def test_schema_match_does_not_raise(self) -> None:
        """ADR-003 §"Schema evolution": matching schema decodes cleanly."""
        channel = FixedFormatCommChannel(schema_version=1)
        ok_packet = CommPacket(
            schema_version=1,
            pose={},
            task_state={},
            aoi={},
            learned_overlay=None,
        )
        channel.decode(ok_packet)


class TestProtocolConformance:
    def test_satisfies_runtime_protocol(self) -> None:
        """Real channel implements the CommChannel Protocol (ADR-003 §Decision)."""
        from chamber.comm.api import CommChannel

        channel = FixedFormatCommChannel()
        assert isinstance(channel, CommChannel)
