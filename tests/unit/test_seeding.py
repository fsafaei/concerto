# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``concerto.training.seeding`` (P6 determinism harness).

Covers ADR-002 §"deterministic seeding harness" and ADR-003 §Decision (the
comm channel relies on ``derive_substream("comm.fixed", ...)`` for its RNG).
"""

from __future__ import annotations

import numpy as np

from concerto.training.seeding import Substream, derive_substream


class TestDeterminism:
    def test_same_name_and_seed_reproduces_draws(self) -> None:
        """Same (name, root_seed) -> byte-identical draws (P6)."""
        a = derive_substream("comm.fixed", root_seed=42).default_rng().integers(0, 2**32, size=8)
        b = derive_substream("comm.fixed", root_seed=42).default_rng().integers(0, 2**32, size=8)
        assert np.array_equal(a, b)

    def test_distinct_names_diverge(self) -> None:
        """Distinct names under the same root seed give independent RNGs."""
        a = derive_substream("comm.fixed", root_seed=42).default_rng().random(64)
        b = derive_substream("comm.degrade", root_seed=42).default_rng().random(64)
        assert not np.allclose(a, b)

    def test_distinct_root_seeds_diverge(self) -> None:
        """Same name under distinct root seeds gives independent RNGs."""
        a = derive_substream("comm.fixed", root_seed=0).default_rng().random(64)
        b = derive_substream("comm.fixed", root_seed=1).default_rng().random(64)
        assert not np.allclose(a, b)

    def test_default_rng_is_a_fresh_generator_each_call(self) -> None:
        """Substream is a *seed factory*: two default_rng() calls re-seed independently."""
        sub = derive_substream("comm.fixed", root_seed=7)
        first = sub.default_rng().random(8)
        second = sub.default_rng().random(8)
        assert np.array_equal(first, second)


class TestSpawn:
    def test_spawn_yields_independent_streams(self) -> None:
        """Spawned children diverge from each other and from the parent (P6)."""
        parent = derive_substream("comm.fixed", root_seed=0)
        children = parent.spawn(3)
        draws = [c.default_rng().random(32) for c in children]
        assert len(children) == 3
        assert not np.allclose(draws[0], draws[1])
        assert not np.allclose(draws[1], draws[2])

    def test_spawn_returns_substream_instances(self) -> None:
        """``spawn`` returns Substream objects (so callers can recurse)."""
        children = derive_substream("comm.fixed", root_seed=0).spawn(2)
        for child in children:
            assert isinstance(child, Substream)


class TestSubstreamSurface:
    def test_substream_is_frozen_dataclass_value(self) -> None:
        """Substreams compare structurally and are hashable (caching-friendly)."""
        a = derive_substream("comm.fixed", root_seed=0)
        b = derive_substream("comm.fixed", root_seed=0)
        assert hash(a.seed_sequence.entropy) == hash(b.seed_sequence.entropy)

    def test_default_rng_returns_numpy_generator(self) -> None:
        """default_rng() returns ``numpy.random.Generator`` (P6 contract)."""
        rng = derive_substream("comm.fixed", root_seed=0).default_rng()
        assert isinstance(rng, np.random.Generator)
