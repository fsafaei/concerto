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
    canonical_pair_key,
    canonical_pair_order,
    lambda_from_jsonable,
    lambda_to_jsonable,
    make_lambda_dict,
    make_pair_keys,
)

if TYPE_CHECKING:
    from concerto.safety.api import FloatArray


def test_bounds_is_frozen_dataclass() -> None:
    bounds = Bounds(
        action_linf_component=1.0,
        cartesian_accel_capacity=1.0,
        action_rate=0.5,
        comm_latency_ms=1.0,
        force_limit=20.0,
    )
    assert bounds.action_linf_component == 1.0
    assert bounds.cartesian_accel_capacity == 1.0
    assert bounds.action_rate == 0.5
    assert bounds.comm_latency_ms == 1.0
    assert bounds.force_limit == 20.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        bounds.action_linf_component = 99.0  # type: ignore[misc]


def test_safety_state_defaults_match_adr_004() -> None:
    state = SafetyState(lambda_=make_lambda_dict(("a", "b", "c")))
    assert state.epsilon == DEFAULT_EPSILON == -0.05
    assert state.eta == DEFAULT_ETA == 0.01
    assert state.warmup_steps_remaining == 0
    assert DEFAULT_WARMUP_STEPS == 50


def test_safety_state_lambda_is_mutated_in_place() -> None:
    state = SafetyState(lambda_=make_lambda_dict(("a", "b")))
    state.lambda_[("a", "b")] = 0.25
    assert state.lambda_[("a", "b")] == 0.25


def test_concerto_safety_infeasible_is_runtime_error() -> None:
    err = ConcertoSafetyInfeasible("QP infeasible — route to braking")
    assert isinstance(err, RuntimeError)
    with pytest.raises(ConcertoSafetyInfeasible, match="braking"):
        raise err


def test_filter_info_typed_dict_carries_lambda_key() -> None:
    info: FilterInfo = {
        "lambda": make_lambda_dict(("a", "b", "c")),
        "constraint_violation": np.zeros(3, dtype=np.float64),
        "prediction_gap_loss": None,
        "fallback_fired": False,
        "qp_solve_ms": 0.05,
    }
    assert "lambda" in info
    assert info["fallback_fired"] is False
    assert info["prediction_gap_loss"] is None


def test_filter_info_typed_dict_admits_prediction_gap_loss_dict() -> None:
    info: FilterInfo = {
        "lambda": make_lambda_dict(("a", "b")),
        "constraint_violation": np.zeros(1, dtype=np.float64),
        "prediction_gap_loss": {("a", "b"): 0.1},
        "fallback_fired": False,
        "qp_solve_ms": 0.05,
    }
    gap = info["prediction_gap_loss"]
    assert gap is not None
    assert gap[("a", "b")] == 0.1


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
            "lambda": dict(state.lambda_),
            "constraint_violation": np.zeros(len(state.lambda_), dtype=np.float64),
            "prediction_gap_loss": None,
            "fallback_fired": False,
            "qp_solve_ms": 0.0,
        }
        return proposed_action, info


def test_safety_filter_protocol_runtime_checkable() -> None:
    assert isinstance(_StubFilter(), JointSafetyFilter)


def test_safety_filter_stub_round_trip_returns_typed_payload() -> None:
    stub = _StubFilter()
    state = SafetyState(lambda_=make_lambda_dict(("a", "b")))
    bounds = Bounds(
        action_linf_component=1.0,
        cartesian_accel_capacity=1.0,
        action_rate=0.5,
        comm_latency_ms=1.0,
        force_limit=20.0,
    )
    proposed = {"agent_a": np.zeros(3, dtype=np.float64)}
    safe, info = stub.filter(
        proposed_action=proposed, obs={"meta": {"partner_id": None}}, state=state, bounds=bounds
    )
    assert safe is proposed
    assert info["fallback_fired"] is False
    assert len(info["lambda"]) == 1
    assert ("a", "b") in info["lambda"]


# --- issue #144 / ADR-014 v3 lambda-keying helpers ----------------


def test_canonical_pair_order_sorts_and_rejects_duplicates() -> None:
    """canonical_pair_order returns a lex-sorted list and rejects duplicates."""
    assert canonical_pair_order(("c", "a", "b")) == ["a", "b", "c"]
    with pytest.raises(ValueError, match="duplicate"):
        canonical_pair_order(("a", "b", "a"))


def test_canonical_pair_key_orders_lex_smaller_first() -> None:
    """canonical_pair_key returns ``(min, max)`` regardless of input order."""
    assert canonical_pair_key("p1", "ego") == ("ego", "p1")
    assert canonical_pair_key("ego", "p1") == ("ego", "p1")


def test_canonical_pair_key_rejects_self_pair() -> None:
    with pytest.raises(ValueError, match="distinct UIDs"):
        canonical_pair_key("a", "a")


def test_make_pair_keys_canonical_upper_triangle() -> None:
    """make_pair_keys enumerates the canonical upper-triangular pair set."""
    assert make_pair_keys(("c", "a", "b")) == [
        ("a", "b"),
        ("a", "c"),
        ("b", "c"),
    ]
    # Empty + single-uid sets emit an empty pair list.
    assert make_pair_keys(()) == []
    assert make_pair_keys(("solo",)) == []


def test_make_lambda_dict_fills_canonical_pair_set() -> None:
    """make_lambda_dict produces one entry per canonical pair, filled uniformly."""
    d = make_lambda_dict(("c", "a", "b"), fill=0.25)
    assert d == {("a", "b"): 0.25, ("a", "c"): 0.25, ("b", "c"): 0.25}


def test_lambda_to_and_from_jsonable_round_trip() -> None:
    """Structured-list wire form round-trips through to/from helpers."""
    d = {("a", "b"): 0.1, ("a", "c"): -0.2, ("b", "c"): 0.3}
    payload = lambda_to_jsonable(d)
    # The wire form is a list of {"a": ..., "b": ..., "value": ...} dicts
    # in canonical pair order.
    assert payload == [
        {"a": "a", "b": "b", "value": 0.1},
        {"a": "a", "b": "c", "value": -0.2},
        {"a": "b", "b": "c", "value": 0.3},
    ]
    assert lambda_from_jsonable(payload) == d


def test_lambda_from_jsonable_canonicalises_inverted_pair() -> None:
    """A v3 entry that lists ``(b, a)`` is rebuilt under the canonical key."""
    payload = [{"a": "p1", "b": "ego", "value": 0.5}]
    assert lambda_from_jsonable(payload) == {("ego", "p1"): 0.5}


def test_lambda_from_jsonable_rejects_non_list_payload() -> None:
    with pytest.raises(TypeError, match="expected list"):
        lambda_from_jsonable({"not": "a list"})  # type: ignore[arg-type]


def test_lambda_from_jsonable_legacy_v2_path_warns_and_parses() -> None:
    """v2 (flat float list + uids) parses with a DeprecationWarning."""
    import warnings

    payload = [0.1, 0.2, 0.3]  # canonical pair order over ("a", "b", "c")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = lambda_from_jsonable(payload, uids=("a", "b", "c"))
    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert deprecations, "legacy v2 reader did not surface a DeprecationWarning"
    assert result == {("a", "b"): 0.1, ("a", "c"): 0.2, ("b", "c"): 0.3}


def test_lambda_from_jsonable_legacy_v2_requires_uids() -> None:
    """v2 (flat float list) without ``uids`` is a TypeError, not a silent dict."""
    with pytest.raises(TypeError, match="uids"):
        lambda_from_jsonable([0.1, 0.2, 0.3])


def test_lambda_from_jsonable_legacy_v2_length_mismatch_is_loud() -> None:
    """v2 length mismatch raises ValueError naming the canonical pair count."""
    with pytest.raises(ValueError, match="canonical pair count"):
        lambda_from_jsonable([0.1], uids=("a", "b", "c"))
