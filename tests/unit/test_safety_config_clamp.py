# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``SafetyConfig.clamp_floor_ratio`` invariants (P1.05.7 / #180).

Pins the strict `clamp_floor_ratio < saturation_threshold` invariant
that keeps the audit-gate's predicate-A trip boundary separated from
the in-loop clamp's engagement boundary by a non-zero buffer. Without
the strict inequality the clamp would engage at the same boundary the
audit gate trips, producing |λ_ss| == threshold on every clamped step
— a degenerate fixed-trip-on-clamp regime.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from concerto.training.config import SafetyConfig


class TestClampFloorRatioStrictInequality:
    """``clamp_floor_ratio`` must be strictly less than ``saturation_threshold``."""

    def test_default_clamp_below_default_saturation(self) -> None:
        """Defaults satisfy the invariant: 0.7 < 0.9."""
        cfg = SafetyConfig()
        assert cfg.clamp_floor_ratio == 0.7
        assert cfg.saturation_threshold == 0.9
        assert cfg.clamp_floor_ratio < cfg.saturation_threshold

    def test_equal_clamp_and_saturation_raises(self) -> None:
        """clamp_floor_ratio == saturation_threshold ⇒ ValidationError."""
        with pytest.raises(ValidationError, match="strictly less than"):
            SafetyConfig(clamp_floor_ratio=0.9, saturation_threshold=0.9)

    def test_clamp_above_saturation_raises(self) -> None:
        """clamp_floor_ratio > saturation_threshold ⇒ ValidationError."""
        with pytest.raises(ValidationError, match="strictly less than"):
            SafetyConfig(clamp_floor_ratio=0.95, saturation_threshold=0.9)

    def test_clamp_well_below_saturation_validates(self) -> None:
        """clamp_floor_ratio=0.5, saturation_threshold=0.9 ⇒ 0.4 x cap buffer."""
        cfg = SafetyConfig(clamp_floor_ratio=0.5, saturation_threshold=0.9)
        assert cfg.clamp_floor_ratio == 0.5
        assert cfg.saturation_threshold == 0.9

    def test_clamp_non_positive_raises_by_field_validator(self) -> None:
        """``Field(gt=0.0)`` rejects clamp_floor_ratio<=0 at field-level."""
        with pytest.raises(ValidationError):
            SafetyConfig(clamp_floor_ratio=0.0)
        with pytest.raises(ValidationError):
            SafetyConfig(clamp_floor_ratio=-0.1)

    def test_clamp_above_one_raises_by_field_validator(self) -> None:
        """``Field(le=1.0)`` rejects clamp_floor_ratio > 1."""
        with pytest.raises(ValidationError):
            SafetyConfig(clamp_floor_ratio=1.5)


class TestClampBoundaryArithmetic:
    """At production cap=10 m/s², the buffer between clamp + gate is 2.0 m/s²."""

    def test_production_default_buffer_is_two_meters_per_sec_squared(self) -> None:
        """clamp at ±7.0, gate at ±9.0; buffer = 2.0 m/s² (issue #180 design)."""
        cfg = SafetyConfig()
        cap = 10.0  # PANDA_CARTESIAN_ACCEL_CAPACITY_MS2 default
        clamp_floor = cfg.clamp_floor_ratio * cap
        gate_trip = cfg.saturation_threshold * cap
        assert clamp_floor == 7.0
        assert gate_trip == 9.0
        assert gate_trip - clamp_floor == 2.0
