# SPDX-License-Identifier: Apache-2.0
# pyright: reportPrivateImportUsage=false
"""Unit tests for ``LearnedOverlay`` (T2.4).

Covers ADR-003 §Decision risk-mitigation 1 (the overlay is null-safe; black-box
partners never see it populated) and the no-op encoder semantics that the
fixed-format channel forwards into ``CommPacket["learned_overlay"]``.
"""

from __future__ import annotations

import torch

from chamber.comm import LearnedOverlay


class TestNullSafety:
    def test_default_is_none(self) -> None:
        """ADR-003 §Decision: default overlay is None (black-box-partner safe)."""
        overlay = LearnedOverlay()
        assert overlay.tensor is None

    def test_encode_returns_none_by_default(self) -> None:
        """ADR-003 §Decision: encode() of an unset slot returns None."""
        assert LearnedOverlay().encode() is None

    def test_decode_handles_none(self) -> None:
        """ADR-003 §Decision: decode(None) returns None without raising."""
        assert LearnedOverlay.decode(None) is None


class TestTensorPassThrough:
    def test_init_with_tensor_is_held(self) -> None:
        """ADR-003 §Decision: a jointly-trained partner's tensor is held verbatim."""
        t = torch.zeros(3, 4)
        overlay = LearnedOverlay(tensor=t)
        assert overlay.tensor is t

    def test_encode_returns_held_tensor(self) -> None:
        """ADR-003 §Decision: encode() exposes the held tensor unchanged (no-op encoder)."""
        t = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        overlay = LearnedOverlay(tensor=t)
        assert torch.equal(overlay.encode(), t)  # type: ignore[arg-type]

    def test_decode_returns_input_tensor_verbatim(self) -> None:
        """ADR-003 §Decision: decode(tensor) returns the same tensor object."""
        t = torch.ones(2, 2)
        assert LearnedOverlay.decode(t) is t

    def test_set_replaces_held_tensor(self) -> None:
        """set() updates the slot (ADR-003 §Decision)."""
        overlay = LearnedOverlay()
        new_t = torch.zeros(1, 1)
        overlay.set(new_t)
        assert overlay.tensor is new_t

    def test_set_to_none_clears_slot(self) -> None:
        """set(None) returns the overlay to a black-box-safe state (ADR-003)."""
        overlay = LearnedOverlay(tensor=torch.zeros(1))
        overlay.set(None)
        assert overlay.tensor is None
        assert overlay.encode() is None


class TestEpisodeReset:
    def test_reset_clears_held_tensor(self) -> None:
        """reset() drops the held tensor (ADR-003 §Decision; episode boundary)."""
        overlay = LearnedOverlay(tensor=torch.zeros(2))
        overlay.reset()
        assert overlay.tensor is None

    def test_reset_idempotent(self) -> None:
        """reset() is safe to call repeatedly (ADR-003 §Decision)."""
        overlay = LearnedOverlay()
        overlay.reset()
        overlay.reset()
        assert overlay.tensor is None
