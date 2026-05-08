# SPDX-License-Identifier: Apache-2.0
"""Deterministic seeding harness for CONCERTO (P6 reproducibility).

Every randomised piece of CONCERTO and CHAMBER (env wrappers, comm channel,
partner zoo, training loop) draws from a numpy ``Generator`` seeded via
:func:`derive_substream`. The function maps a ``(name, root_seed)`` pair to a
fresh ``Substream`` whose RNG is independent of every other named substream
under the same root seed; reusing the same pair always returns byte-identical
draws.

Cited as ADR-002 §"deterministic seeding harness" (training stack)
and ADR-003 §Decision (the comm channel uses ``derive_substream("comm.fixed", ...)``
to seed its substream).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


def _name_to_spawn_key(name: str) -> tuple[int, ...]:
    """Hash ``name`` into a stable spawn-key tuple (P6).

    BLAKE2b at 16 bytes gives 128 bits — enough collision-resistance for the
    finite set of named substreams the project uses without dragging in a
    cryptographic dependency.
    """
    digest = hashlib.blake2b(name.encode("utf-8"), digest_size=16).digest()
    return (
        int.from_bytes(digest[:8], "big", signed=False),
        int.from_bytes(digest[8:], "big", signed=False),
    )


@dataclass(frozen=True)
class Substream:
    """Wraps a numpy ``SeedSequence`` with convenience constructors (P6).

    ADR-002 §"deterministic seeding harness". The frozen dataclass makes the
    substream identity hashable; equality is structural over the underlying
    seed-sequence's entropy + spawn key, which is enough for caching.
    """

    seed_sequence: np.random.SeedSequence

    def default_rng(self) -> np.random.Generator:
        """Construct a fresh numpy ``Generator`` from this substream (P6).

        ADR-002 §"deterministic seeding harness". Two calls return independent
        generators that produce identical sequences when freshly built — i.e.,
        the substream is a *seed factory*, not an iterator.
        """
        return np.random.default_rng(self.seed_sequence)

    def spawn(self, n: int) -> list[Substream]:
        """Spawn ``n`` independent child substreams (P6).

        ADR-002 §"deterministic seeding harness". Use when a producer needs
        per-worker / per-episode RNG without colliding with sibling streams.
        """
        return [Substream(seed_sequence=child) for child in self.seed_sequence.spawn(n)]


def derive_substream(name: str, *, root_seed: int = 0) -> Substream:
    """Derive a deterministic named substream from a root seed (P6).

    ADR-002 §"deterministic seeding harness"; ADR-003 §Decision (the comm
    channel uses ``derive_substream("comm.fixed", root_seed=...)``).

    Args:
        name: Stable label for the substream. Distinct names yield independent
            RNG streams under the same ``root_seed``; reusing the pair
            ``(name, root_seed)`` always returns byte-identical draws.
        root_seed: Project-wide root seed. Defaults to 0 so call sites can
            ship working defaults; production runs override per-experiment.

    Returns:
        A :class:`Substream` whose :meth:`Substream.default_rng` is the
        canonical entry-point for downstream randomness.
    """
    seed_seq = np.random.SeedSequence(
        entropy=root_seed,
        spawn_key=_name_to_spawn_key(name),
    )
    return Substream(seed_sequence=seed_seq)


__all__ = ["Substream", "derive_substream"]
