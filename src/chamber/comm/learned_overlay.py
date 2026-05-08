# SPDX-License-Identifier: Apache-2.0
"""Opt-in learned-message overlay slot (T2.4).

ADR-003 §Decision (the second of the two channels — opt-in, null-safe);
ADR-003 §Risks risk-mitigation 1: black-box partners leave this slot as
``None`` so the fixed-format channel alone is sufficient for ad-hoc
deployment. Jointly-trained partners (HetGPPO, CommFormer) populate it
with a message tensor.

Phase-0 ships only the slot itself plus a no-op encoder; the real
learned-comm machinery (HetGPPO message GNN) is a Phase-1 concern.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


class LearnedOverlay:
    """Holds an optional learned-message tensor (ADR-003 §Decision).

    The class is a thin slot wrapper: ``encode`` returns whatever tensor a
    jointly-trained partner has set (or ``None``), and ``decode`` is
    null-safe so black-box consumers can call it unconditionally.

    Args:
        tensor: Initial overlay value. Defaults to ``None`` so the slot is
            interoperable with the black-box partner contract out of the box.
    """

    def __init__(self, tensor: torch.Tensor | None = None) -> None:
        """Build an overlay slot, optionally pre-populated (ADR-003 §Decision)."""
        self._tensor: torch.Tensor | None = tensor

    @property
    def tensor(self) -> torch.Tensor | None:
        """Return the currently held tensor, or ``None`` (ADR-003 §Decision)."""
        return self._tensor

    def set(self, tensor: torch.Tensor | None) -> None:
        """Replace the held tensor (ADR-003 §Decision).

        Jointly-trained partners call this from their forward pass; black-box
        partners never invoke it (or invoke it with ``None``), preserving the
        null-safety guarantee.
        """
        self._tensor = tensor

    def reset(self) -> None:
        """Clear the overlay (ADR-003 §Decision; episode boundary).

        Called by the channel on env reset so cross-episode message tensors
        do not leak across episode boundaries.
        """
        self._tensor = None

    def encode(self) -> torch.Tensor | None:
        """No-op encoder: returns the held tensor (ADR-003 §Decision).

        Phase-1 will replace this with the message GNN forward pass; for
        Phase-0 the slot semantics are sufficient for the
        :class:`~chamber.comm.api.CommPacket` ``learned_overlay`` field.
        """
        return self._tensor

    @staticmethod
    def decode(packet_overlay: torch.Tensor | None) -> torch.Tensor | None:
        """Null-safe decoder for the packet's learned overlay (ADR-003 §Decision).

        Black-box partners receive a packet whose ``learned_overlay`` is
        ``None``; jointly-trained partners receive the producer's tensor
        unchanged. Either path returns without raising.
        """
        return packet_overlay
