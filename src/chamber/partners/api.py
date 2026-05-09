# SPDX-License-Identifier: Apache-2.0
"""Public partner-stack API (ADR-009 §Decision; ADR-010 §Decision; ADR-003 §Decision).

The :class:`FrozenPartner` Protocol is the contract every concrete partner in
:mod:`chamber.partners` (heuristic / frozen-RL / Phase-1 VLA stubs) implements.
The :class:`PartnerSpec` dataclass is the identity-bearing handle the registry,
loader, and the M3 conformal filter all read from.

ADR-009 §Decision: black-box AHT — partner parameters are frozen during ego
training. The Protocol exposes ``reset`` and ``act`` only; runtime enforcement
of "no joint training" lives on :class:`chamber.partners.interface.PartnerBase`
via the ``_FORBIDDEN_ATTRS`` shield.

ADR-006 risk #3 / ADR-004 §risk-mitigation #2: the ``partner_id`` attribute is
the stable hash that the M3 conformal filter reads on every step to detect
mid-episode partner swap and re-initialise ``lambda``. Surfacing it on
``obs["meta"]["partner_id"]`` is the M4 producer-side commitment.

ADR-010 §Decision: Phase-1 VLA partners (OpenVLA / CrossFormer) plug into the
same Protocol; the LoRA fine-tune output path lives on
:class:`PartnerSpec.weights_uri`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping

    import numpy as np
    from numpy.typing import NDArray


@dataclass(frozen=True)
class PartnerSpec:
    """Identity-bearing spec used to build, load, and address a partner (ADR-009 §Decision).

    The :attr:`partner_id` property is the stable hash that the M3 conformal
    filter reads on every step (ADR-004 §risk-mitigation #2; ADR-006 risk #3).
    Two specs that hash the same partner_id refer to the same logical partner;
    a partner-set change between steps re-initialises the conformal ``lambda``.

    Attributes:
        class_name: Registry key — the string passed to
            :func:`chamber.partners.registry.register_partner`.
        seed: Training-time seed; for scripted partners use ``0``.
        checkpoint_step: Training step at which the checkpoint was taken;
            ``None`` for scripted partners or for stubs.
        weights_uri: ``local://`` or remote URI pointing at the frozen
            checkpoint; ``None`` for scripted partners.
        extra: Free-form string→string dict for task / embodiment / uid
            metadata. Read by the partner implementation, NOT by the hash.
    """

    class_name: str
    seed: int
    checkpoint_step: int | None
    weights_uri: str | None
    extra: dict[str, str] = field(default_factory=dict)

    @property
    def partner_id(self) -> str:
        """Stable 16-hex-char hash of ``(class_name, seed, checkpoint_step, weights_uri)``.

        ADR-004 §risk-mitigation #2 / ADR-006 risk #3: surfaced on
        ``obs["meta"]["partner_id"]`` so the conformal filter can re-initialise
        ``lambda`` on partner identity change. The 16-char (64-bit) keyspace
        is sufficient for the Phase-1 zoo target of 144 partners with vast
        margin (plan/04 §8 Notes).

        The hash material uses :func:`repr` of a tuple of the four identity
        fields (in this fixed order) rather than f-string interpolation so
        ``None`` cannot collide with the literal string ``"None"`` — a real
        risk if a Phase-1 partner ever ships with ``weights_uri="None"``
        (ADR-006 risk #3 swap-detection contract). Reordering the fields
        in :class:`PartnerSpec` would silently change every existing
        partner_id; do not reorder without a new ADR.

        Returns:
            Lowercase 16-character hex digest of the SHA-256 of
            ``repr((class_name, seed, checkpoint_step, weights_uri))``.
            ``extra`` is intentionally excluded so per-task / per-embodiment
            metadata does not collide with the conformal filter's
            swap-detection contract (plan/04 §3.1).
        """
        material = repr((self.class_name, self.seed, self.checkpoint_step, self.weights_uri))
        return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


@runtime_checkable
class FrozenPartner(Protocol):
    """A partner the ego agent treats as a black box (ADR-009 §Decision).

    Implementations MUST NOT expose any update / train / learn methods at
    runtime. The base class :class:`chamber.partners.interface.PartnerBase`
    enforces this via a ``__getattr__`` shield over ``_FORBIDDEN_ATTRS``.
    Type checkers reject Protocol-typed access to ``train`` / ``learn`` /
    ``update_params`` because those names are not part of this Protocol.

    Stateless across episodes: ``reset(*, seed)`` clears every internal
    cache (plan/04 §2 "Partner state" — Phase-0 deliberately excludes
    recurrent partner policies; if Phase 1 needs them, ADR amendment).
    """

    spec: PartnerSpec

    def reset(self, *, seed: int | None = None) -> None:
        """Reset internal episode state (ADR-009 §Decision; P6 reproducibility).

        Args:
            seed: Optional root seed for the partner's deterministic RNG
                substream. Implementations route this through
                :func:`concerto.training.seeding.derive_substream`.
        """
        ...  # pragma: no cover

    def act(
        self,
        obs: Mapping[str, object],
        *,
        deterministic: bool = True,
    ) -> NDArray[np.floating]:
        """Map observation to action (ADR-009 §Decision; ADR-003 §Decision).

        The observation mapping is the Gymnasium-conformant
        ``{"agent": {uid: proprio}, "comm": CommPacket, "meta": {...}}``
        produced by :mod:`chamber.envs` wrappers. Partners read
        ``obs["agent"][uid]`` for proprioception and ``obs["comm"]`` for the
        ADR-003 fixed-format channel; they MUST NOT touch any other key.

        The accepted type is :class:`~collections.abc.Mapping` rather than
        :class:`dict` so that callers can pass concrete nested dicts without
        running into Mapping-vs-dict invariance issues at the call site.

        Args:
            obs: Gymnasium observation dict.
            deterministic: Whether to sample greedily (default ``True``);
                stochastic policies pass ``False``. Scripted partners ignore.

        Returns:
            Action vector for the partner's own uid, shape and dtype defined
            by the env's action space slice for that uid.
        """
        ...  # pragma: no cover


__all__ = ["FrozenPartner", "PartnerSpec"]
