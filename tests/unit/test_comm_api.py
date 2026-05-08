# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``chamber.comm`` public api (T2.1).

Covers ADR-003 §Decision (Protocol shape, packet TypedDict, schema version)
and ADR-006 §Risks #5 (warning class identity).
"""

from __future__ import annotations

from typing import Any

import pytest

from chamber import comm as comm_pkg
from chamber.comm import (
    SCHEMA_VERSION,
    ChamberCommError,
    ChamberCommQPSaturationWarning,
    CommChannel,
    CommPacket,
    FixedFormatCommChannel,
    Pose,
    TaskStatePredicate,
)
from chamber.comm import api as comm_api


class TestSchemaVersion:
    def test_schema_version_is_one(self) -> None:
        """ADR-003 §Decision: ship at v1; bump requires a new ADR."""
        assert SCHEMA_VERSION == 1

    def test_schema_version_is_int(self) -> None:
        """Packets carry an integer schema_version (ADR-003 §Decision)."""
        assert isinstance(SCHEMA_VERSION, int)

    def test_schema_version_re_export_is_identity(self) -> None:
        """The package-level re-export is the same object as the api module's."""
        assert SCHEMA_VERSION is comm_api.SCHEMA_VERSION


class TestCommPacketShape:
    def test_pose_typeddict_constructs_with_xyz_and_quat(self) -> None:
        """Pose carries SE(3) xyz + wxyz quaternion (ADR-003 §Decision)."""
        pose: Pose = Pose(xyz=(0.0, 0.0, 0.0), quat_wxyz=(1.0, 0.0, 0.0, 0.0))
        assert pose["xyz"] == (0.0, 0.0, 0.0)
        assert pose["quat_wxyz"] == (1.0, 0.0, 0.0, 0.0)

    def test_task_state_predicate_total_false_allows_empty(self) -> None:
        """TaskStatePredicate is total=False — every key optional (ADR-003 §Decision)."""
        empty: TaskStatePredicate = TaskStatePredicate()
        assert empty == {}
        populated: TaskStatePredicate = TaskStatePredicate(grasp_side="left")
        assert populated.get("grasp_side") == "left"

    def test_comm_packet_has_required_keys(self) -> None:
        """CommPacket carries schema_version, pose, task_state, aoi, learned_overlay."""
        packet: CommPacket = CommPacket(
            schema_version=SCHEMA_VERSION,
            pose={"a": Pose(xyz=(0.0, 0.0, 0.0), quat_wxyz=(1.0, 0.0, 0.0, 0.0))},
            task_state={"a": TaskStatePredicate(grasp_side="left")},
            aoi={"a": 0.0},
            learned_overlay=None,
        )
        for key in ("schema_version", "pose", "task_state", "aoi", "learned_overlay"):
            assert key in packet

    def test_comm_packet_learned_overlay_is_optional(self) -> None:
        """ADR-003 §Decision risk-mitigation 1: learned_overlay is null-safe."""
        packet: CommPacket = CommPacket(
            schema_version=SCHEMA_VERSION,
            pose={},
            task_state={},
            aoi={},
            learned_overlay=None,
        )
        assert packet["learned_overlay"] is None


class TestCommChannelProtocol:
    def test_protocol_lives_in_chamber_comm_api(self) -> None:
        """Protocol is defined in ``chamber.comm.api`` (ADR-003 §Decision; plan/02 §3.1)."""
        assert CommChannel is comm_api.CommChannel

    def test_fixed_format_stub_satisfies_protocol_runtime_check(self) -> None:
        """The M1 stub satisfies the (runtime_checkable) Protocol (ADR-003 §Decision)."""
        channel: Any = FixedFormatCommChannel()
        assert isinstance(channel, CommChannel)

    def test_protocol_has_reset_encode_decode(self) -> None:
        """ADR-003 §Decision: Protocol exposes reset / encode / decode."""
        for method in ("reset", "encode", "decode"):
            assert hasattr(CommChannel, method)

    def test_reset_accepts_seed_keyword(self) -> None:
        """ADR-003 §Decision: reset takes an optional ``seed`` keyword (P6 determinism)."""
        channel = FixedFormatCommChannel()
        channel.reset(seed=0)
        channel.reset()  # default-None must work too


class TestStubChannelBehaviour:
    def test_encode_returns_packet_with_current_schema_version(self) -> None:
        """ADR-003 §Decision: encode emits a packet carrying SCHEMA_VERSION."""
        packet = FixedFormatCommChannel().encode({})
        assert packet["schema_version"] == SCHEMA_VERSION

    def test_encode_returns_packet_with_all_keys(self) -> None:
        """ADR-003 §Decision: encode emits a CommPacket with all five keys."""
        packet = FixedFormatCommChannel().encode({})
        for key in ("schema_version", "pose", "task_state", "aoi", "learned_overlay"):
            assert key in packet

    def test_decode_round_trip_does_not_raise(self) -> None:
        """ADR-003 §Decision: decode accepts the channel's own packets."""
        channel = FixedFormatCommChannel()
        channel.decode(channel.encode({}))


class TestErrorTypes:
    def test_chamber_comm_error_is_runtime_error(self) -> None:
        """ChamberCommError subclasses RuntimeError (loud-fail per ADR-003 §Decision)."""
        assert issubclass(ChamberCommError, RuntimeError)

    def test_chamber_comm_error_can_be_raised_and_caught(self) -> None:
        """ChamberCommError carries a message and propagates."""
        with pytest.raises(ChamberCommError, match="boom"):
            raise ChamberCommError("boom")

    def test_qp_saturation_warning_is_user_warning(self) -> None:
        """ChamberCommQPSaturationWarning subclasses UserWarning (ADR-006 R5)."""
        assert issubclass(ChamberCommQPSaturationWarning, UserWarning)

    def test_qp_saturation_warning_can_be_emitted(self) -> None:
        """ChamberCommQPSaturationWarning fires through the warnings module."""
        import warnings

        with pytest.warns(ChamberCommQPSaturationWarning, match="saturation"):
            warnings.warn("QP saturation", ChamberCommQPSaturationWarning, stacklevel=1)


class TestPublicSurface:
    def test_all_listed_symbols_resolve(self) -> None:
        """Every name in ``__all__`` is importable from the package."""
        for name in comm_pkg.__all__:
            assert hasattr(comm_pkg, name), f"chamber.comm missing exported symbol {name!r}"

    def test_all_contains_expected_symbols(self) -> None:
        """The public surface includes the api + error + stub exports."""
        expected = {
            "SCHEMA_VERSION",
            "ChamberCommError",
            "ChamberCommQPSaturationWarning",
            "CommChannel",
            "CommPacket",
            "FixedFormatCommChannel",
            "Pose",
            "TaskStatePredicate",
        }
        assert expected.issubset(set(comm_pkg.__all__))
