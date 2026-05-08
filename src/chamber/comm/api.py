# SPDX-License-Identifier: Apache-2.0
"""Public API for the CHAMBER comm stack — Protocol + packet shape + schema.

ADR-003 §Decision: defines the canonical wire format of the mandatory
fixed-format channel (pose per uid, task-state predicate per uid, AoI per uid)
plus an opt-in learned-overlay tensor field, and the
:class:`CommChannel` Protocol that every concrete channel implements.

ADR-006 §Decision: schema versioning protects the URLLC-anchored numeric
bounds against silent drift; bumping :data:`SCHEMA_VERSION` requires a new ADR.

The Protocol lives on the benchmark side (this module) rather than under
``concerto.api`` because its signatures reference :class:`CommPacket`, which
is the benchmark's wire format. The dependency-direction rule (plan/10 §2)
forbids ``concerto.*`` from importing ``chamber.*``; placing the Protocol
here keeps the rule intact.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypedDict, runtime_checkable

if TYPE_CHECKING:
    import torch

#: Wire-format schema version (ADR-003 §Decision; ADR-006 §Decision).
#:
#: Bumping this value is a breaking change to the public packet shape and
#: requires a new ADR. Decoders MUST refuse packets whose ``schema_version``
#: is not equal to this constant.
SCHEMA_VERSION: int = 1


class Pose(TypedDict):
    """SE(3) pose in the env's base frame (ADR-003 §Decision).

    The fixed-format channel ships pose as the minimum every black-box
    partner can produce. Quaternion convention is wxyz (real part first),
    matching SAPIEN / ManiSkill v3.
    """

    xyz: tuple[float, float, float]
    quat_wxyz: tuple[float, float, float, float]


class TaskStatePredicate(TypedDict, total=False):
    """Typed task-state predicates for richer coordination (ADR-003 §Decision).

    The fixed-format extension field; all keys are optional so partners that
    do not produce them remain interoperable. Adding a new key is a
    schema bump.
    """

    grasp_side: str
    object_class: str
    contact_force: float


class CommPacket(TypedDict):
    """The mandatory fixed-format wire packet (ADR-003 §Decision).

    Every concrete :class:`CommChannel` produces and consumes packets of
    exactly this shape. Downstream:

    - The safety filter (M3) reads ``packet["pose"][uid]`` and
      ``packet["aoi"][uid]`` deterministically.
    - The partner stack (M4) writes ``packet["pose"][uid] = self.last_pose``.
    - The evaluation harness (M5) logs from the same surface.

    ADR-006 §Decision: AoI is the URLLC-anchored latency proxy.
    """

    schema_version: int
    pose: dict[str, Pose]
    task_state: dict[str, TaskStatePredicate]
    aoi: dict[str, float]
    learned_overlay: torch.Tensor | None


@runtime_checkable
class CommChannel(Protocol):
    """Contract for an inter-agent communication channel (ADR-003 §Decision).

    Implementations encode the env-side state into a :class:`CommPacket` and
    decode it back into the dict subset that downstream consumers (safety
    filter M3, partner stack M4, evaluation harness M5) expect under
    ``obs["comm"]``. ADR-006 §Decision: implementations honour the
    URLLC-anchored latency / jitter / drop bounds when composed with
    :class:`chamber.comm.degradation.CommDegradationWrapper`.
    """

    def reset(self, *, seed: int | None = None) -> None:
        """Reset channel state at episode start (ADR-003 §Decision).

        Args:
            seed: Optional root seed for the channel's deterministic RNG
                substream (P6 reproducibility). Implementations route this
                through ``concerto.training.seeding.derive_substream``.
        """
        ...

    def encode(self, state: object) -> CommPacket:
        """Encode env state into a comm-channel packet (ADR-003 §Decision).

        Args:
            state: Env-side observation or raw physics state (impl-defined).

        Returns:
            A :class:`CommPacket` honouring :data:`SCHEMA_VERSION`.
        """
        ...

    def decode(self, packet: CommPacket) -> dict[str, object]:
        """Decode a comm-channel packet back into the typed state subset (ADR-003 §Decision).

        Args:
            packet: A :class:`CommPacket` previously produced by :meth:`encode`.

        Returns:
            The schema-typed subset of the env state expressible via the
            packet (pose / task-state predicate / AoI), keyed by uid.
        """
        ...


__all__ = [
    "SCHEMA_VERSION",
    "CommChannel",
    "CommPacket",
    "Pose",
    "TaskStatePredicate",
]
