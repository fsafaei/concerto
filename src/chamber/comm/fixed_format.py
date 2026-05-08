# SPDX-License-Identifier: Apache-2.0
"""Fixed-format communication channel — encoder + decoder (T2.3).

ADR-003 §Decision: this channel is the *mandatory* baseline every black-box
partner can speak. It encodes pose + task-state predicates + AoI per uid into
a schema-versioned :class:`~chamber.comm.api.CommPacket`, with an opt-in
``learned_overlay`` slot that jointly-trained partners may fill.

ADR-006 §Decision: the channel itself ships **no degradation**. URLLC-anchored
latency / jitter / drop is composed externally via
:class:`chamber.comm.degradation.CommDegradationWrapper` (T2.5) so that the
encoding logic stays pure, deterministic, and trivially testable.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from chamber.comm.aoi import AoIClock
from chamber.comm.api import SCHEMA_VERSION, CommPacket, Pose, TaskStatePredicate
from chamber.comm.errors import ChamberCommError
from concerto.training.seeding import derive_substream

#: Default substream name for the fixed-format channel's RNG (P6 determinism).
_SUBSTREAM_NAME = "comm.fixed"


class FixedFormatCommChannel:
    """Mandatory fixed-format channel (ADR-003 §Decision).

    On every :meth:`encode` call:
      1. The internal :class:`AoIClock` advances by one env tick (``+1.0``).
      2. Every uid present in ``state["pose"]`` / ``state["task_state"]`` is
         :meth:`AoIClock.mark_fresh`-ed (its AoI drops to zero).
      3. The result is bundled into a :class:`CommPacket` carrying
         :data:`~chamber.comm.api.SCHEMA_VERSION` and a snapshot of every
         registered uid's AoI.

    :meth:`decode` is the inverse projection: it returns the schema-typed
    subset of the state expressible via the packet (pose / task-state / AoI /
    learned overlay). A schema-version mismatch raises
    :class:`~chamber.comm.errors.ChamberCommError`.

    Determinism: an internal numpy ``Generator`` is seeded via
    :func:`concerto.training.seeding.derive_substream` and re-seeded on
    :meth:`reset`. The encoder is currently fully deterministic from input;
    the RNG is held for future randomised-encoding features (e.g. dropout on
    optional fields) without changing the public surface.

    Args:
        schema_version: Version stamp for emitted packets and the value
            decoders demand. Bumping it requires a new ADR
            (ADR-003 §"Schema evolution").
    """

    def __init__(self, *, schema_version: int = SCHEMA_VERSION) -> None:
        """Build a fixed-format channel (ADR-003 §Decision).

        Args:
            schema_version: Wire-format version stamp; defaults to the
                project-wide :data:`SCHEMA_VERSION`.
        """
        self._schema_version = schema_version
        self._aoi = AoIClock()
        self._root_seed: int = 0
        self._rng = derive_substream(_SUBSTREAM_NAME, root_seed=self._root_seed).default_rng()

    def reset(self, *, seed: int | None = None) -> None:
        """Reset the AoI clock and re-seed the channel RNG (ADR-003 §Decision).

        Args:
            seed: Optional new root seed for the channel substream (P6
                determinism). When provided, replaces the previously stored
                root seed; when omitted, the existing root seed is reused so
                episode resets remain reproducible without a fresh seed
                argument.
        """
        self._aoi.reset()
        if seed is not None:
            self._root_seed = seed
        self._rng = derive_substream(_SUBSTREAM_NAME, root_seed=self._root_seed).default_rng()

    def encode(self, state: object) -> CommPacket:
        """Encode env state into a :class:`CommPacket` (ADR-003 §Decision).

        Accepts any mapping with optional ``"pose"`` / ``"task_state"`` /
        ``"learned_overlay"`` keys; non-mapping inputs are treated as empty
        state (the channel simply ages every previously-registered uid).

        Args:
            state: Source state mapping. Recognised keys:

                - ``"pose"``: ``dict[str, Pose]`` — fresh pose per uid.
                - ``"task_state"``: ``dict[str, TaskStatePredicate]`` —
                  optional fresh predicate per uid.
                - ``"learned_overlay"``: optional ``torch.Tensor`` for
                  jointly-trained B6 partners; passed through unchanged.

        Returns:
            A :class:`CommPacket` carrying the current schema version, a
            shallow copy of pose / task-state, the AoI snapshot, and the
            opt-in learned overlay.
        """
        pose_in, task_state_in, learned = self._extract(state)
        self._aoi.tick(1.0)
        for uid in pose_in:
            self._aoi.mark_fresh(uid)
        for uid in task_state_in:
            self._aoi.mark_fresh(uid)
        return CommPacket(
            schema_version=self._schema_version,
            pose=dict(pose_in),
            task_state=dict(task_state_in),
            aoi=self._aoi.snapshot(),
            learned_overlay=learned,
        )

    def decode(self, packet: CommPacket) -> dict[str, object]:
        """Decode a :class:`CommPacket` back into the typed state subset (ADR-003 §Decision).

        Round-trip guarantee: ``decode(encode(state))`` reproduces every
        schema-typed field of ``state`` (pose / task-state / learned overlay)
        and adds the AoI snapshot the channel computed during encode.

        Args:
            packet: A packet previously emitted by :meth:`encode` (or another
                channel honouring the same :data:`SCHEMA_VERSION`).

        Returns:
            A dict with keys ``"pose"`` / ``"task_state"`` / ``"aoi"`` /
            ``"learned_overlay"`` (each a shallow copy where applicable).

        Raises:
            ChamberCommError: when ``packet["schema_version"]`` differs from
                this channel's schema version (ADR-003 §"Schema evolution").
        """
        if packet["schema_version"] != self._schema_version:
            msg = (
                f"comm packet schema_version mismatch: "
                f"channel expects {self._schema_version}, packet carries "
                f"{packet['schema_version']}"
            )
            raise ChamberCommError(msg)
        return {
            "pose": dict(packet["pose"]),
            "task_state": dict(packet["task_state"]),
            "aoi": dict(packet["aoi"]),
            "learned_overlay": packet["learned_overlay"],
        }

    @staticmethod
    def _extract(
        state: object,
    ) -> tuple[Mapping[str, Pose], Mapping[str, TaskStatePredicate], Any]:
        """Pluck the recognised sub-keys from a state-shaped mapping (ADR-003 §Decision)."""
        if not isinstance(state, Mapping):
            return {}, {}, None
        state_map = cast("Mapping[str, Any]", state)
        pose: Mapping[str, Pose] = state_map.get("pose") or {}
        task_state: Mapping[str, TaskStatePredicate] = state_map.get("task_state") or {}
        learned: Any = state_map.get("learned_overlay")
        return pose, task_state, learned
