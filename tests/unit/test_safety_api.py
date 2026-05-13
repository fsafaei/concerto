# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``concerto.safety`` public API (T3.1).

Covers ADR-004 §Decision (Bounds, SafetyState, JointSafetyFilter Protocol),
ADR-006 §Decision Option C (per-task numeric bounds), and the
:class:`ConcertoSafetyInfeasible` error contract from ADR-004 §Decision.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import numpy as np
import pytest

from concerto.safety import (
    DEFAULT_EPSILON,
    DEFAULT_ETA,
    DEFAULT_WARMUP_STEPS,
    Bounds,
    ConcertoSafetyInfeasible,
    FilterInfo,
    JointSafetyFilter,
    SafetyState,
)

if TYPE_CHECKING:
    from concerto.safety.api import FloatArray


def test_bounds_is_frozen_dataclass() -> None:
    bounds = Bounds(action_norm=1.0, action_rate=0.5, comm_latency_ms=1.0, force_limit=20.0)
    assert bounds.action_norm == 1.0
    assert bounds.action_rate == 0.5
    assert bounds.comm_latency_ms == 1.0
    assert bounds.force_limit == 20.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        bounds.action_norm = 99.0  # type: ignore[misc]


def test_safety_state_defaults_match_adr_004() -> None:
    state = SafetyState(lambda_=np.zeros(3, dtype=np.float64))
    assert state.epsilon == DEFAULT_EPSILON == -0.05
    assert state.eta == DEFAULT_ETA == 0.01
    assert state.warmup_steps_remaining == 0
    assert DEFAULT_WARMUP_STEPS == 50


def test_safety_state_lambda_is_mutated_in_place() -> None:
    state = SafetyState(lambda_=np.zeros(3, dtype=np.float64))
    state.lambda_[0] = 0.25
    assert state.lambda_[0] == 0.25


def test_concerto_safety_infeasible_is_runtime_error() -> None:
    err = ConcertoSafetyInfeasible("QP infeasible — route to braking")
    assert isinstance(err, RuntimeError)
    with pytest.raises(ConcertoSafetyInfeasible, match="braking"):
        raise err


def test_filter_info_typed_dict_carries_lambda_key() -> None:
    info: FilterInfo = {
        "lambda": np.zeros(2, dtype=np.float64),
        "constraint_violation": np.zeros(2, dtype=np.float64),
        "prediction_gap_loss": None,
        "fallback_fired": False,
        "qp_solve_ms": 0.05,
    }
    assert "lambda" in info
    assert info["fallback_fired"] is False
    assert info["prediction_gap_loss"] is None


def test_filter_info_typed_dict_admits_prediction_gap_loss_array() -> None:
    info: FilterInfo = {
        "lambda": np.zeros(2, dtype=np.float64),
        "constraint_violation": np.zeros(2, dtype=np.float64),
        "prediction_gap_loss": np.array([0.1, 0.0], dtype=np.float64),
        "fallback_fired": False,
        "qp_solve_ms": 0.05,
    }
    gap = info["prediction_gap_loss"]
    assert gap is not None
    assert gap.shape == (2,)


class _StubFilter:
    """Minimal Protocol implementation for the runtime_checkable test."""

    def reset(self, *, seed: int | None = None) -> None:
        del seed

    def filter(
        self,
        proposed_action: dict[str, FloatArray],
        obs: dict[str, object],
        state: SafetyState,
        bounds: Bounds,
    ) -> tuple[dict[str, FloatArray], FilterInfo]:
        del obs, bounds
        info: FilterInfo = {
            "lambda": state.lambda_.copy(),
            "constraint_violation": np.zeros_like(state.lambda_),
            "prediction_gap_loss": None,
            "fallback_fired": False,
            "qp_solve_ms": 0.0,
        }
        return proposed_action, info


def test_safety_filter_protocol_runtime_checkable() -> None:
    assert isinstance(_StubFilter(), JointSafetyFilter)


def test_safety_filter_stub_round_trip_returns_typed_payload() -> None:
    stub = _StubFilter()
    state = SafetyState(lambda_=np.zeros(2, dtype=np.float64))
    bounds = Bounds(action_norm=1.0, action_rate=0.5, comm_latency_ms=1.0, force_limit=20.0)
    proposed = {"agent_a": np.zeros(3, dtype=np.float64)}
    safe, info = stub.filter(
        proposed_action=proposed, obs={"meta": {"partner_id": None}}, state=state, bounds=bounds
    )
    assert safe is proposed
    assert info["fallback_fired"] is False
    assert info["lambda"].shape == (2,)
