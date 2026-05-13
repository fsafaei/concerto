# SPDX-License-Identifier: Apache-2.0
"""Property test: CENTRALIZED mode with heterogeneous action_dim (spike_004A).

Three agents of distinct ``action_dim`` (2, 4, 7) — the
heterogeneity-aware refactor must build a QP whose decision variable
is the *concatenation* of per-uid actions according to each
:attr:`AgentControlModel.action_dim`, with **no** "first uid's shape"
fallback (reviewer P0-2). The Cartesian-space CBF row construction
projects through each agent's control map so the row coefficients
carry the correct per-agent loading.

This is the test the AS axis of ADR-007 §Stage 1 requires before any
spike against a 7-DOF arm + 2-DOF base partner can claim a
≥20pp gap is *real* rather than an artefact of the homogeneous QP
flattener.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from concerto.safety.api import (
    AgentControlModel,
    Bounds,
    DoubleIntegratorControlModel,
    FloatArray,
    SafetyState,
)
from concerto.safety.cbf_qp import AgentSnapshot, ExpCBFQP


class _OverActuatedModel:
    """Toy over-actuated model: first ``position_dim`` action components ↔ Cartesian accel.

    Stand-in for over-actuated embodiments (e.g. the Stage-1 AS spike's
    7-DOF arm whose end-effector position is 3-D). The Cartesian image
    is the first ``position_dim`` components of the action vector;
    redundant DOFs do not load Cartesian acceleration.
    """

    def __init__(self, uid: str, action_dim: int, position_dim: int = 2) -> None:
        if action_dim < position_dim:
            msg = f"action_dim {action_dim} < position_dim {position_dim} not supported"
            raise ValueError(msg)
        self.uid = uid
        self.action_dim = action_dim
        self.position_dim = position_dim

    def action_to_cartesian_accel(
        self, state: AgentSnapshot, action: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        del state
        return action[: self.position_dim].astype(np.float64, copy=False)

    def cartesian_accel_to_action(
        self, state: AgentSnapshot, cartesian_accel: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        del state
        out = np.zeros(self.action_dim, dtype=np.float64)
        out[: self.position_dim] = cartesian_accel
        return out

    def max_cartesian_accel(self, bounds: Bounds) -> float:
        return float(bounds.action_norm)


def _bounds(action_norm: float = 3.0) -> Bounds:
    return Bounds(
        action_norm=action_norm,
        action_rate=0.5,
        comm_latency_ms=1.0,
        force_limit=20.0,
    )


def _models_2_4_7() -> dict[str, AgentControlModel]:
    """spike_004A example: a 2-D base, a 4-D arm-like, and a 7-D arm-like agent."""
    out: dict[str, AgentControlModel] = {}
    out["base"] = DoubleIntegratorControlModel(uid="base", action_dim=2)
    out["arm4"] = _OverActuatedModel(uid="arm4", action_dim=4, position_dim=2)
    out["arm7"] = _OverActuatedModel(uid="arm7", action_dim=7, position_dim=2)
    return out


def _snaps_triangle(rng: np.random.Generator) -> dict[str, AgentSnapshot]:
    """Three agents on a triangle — separated enough that the QP is well-posed."""
    return {
        "base": AgentSnapshot(
            position=np.array([-3.0, 0.0], dtype=np.float64),
            velocity=rng.uniform(-0.3, 0.3, size=2).astype(np.float64),
            radius=0.2,
        ),
        "arm4": AgentSnapshot(
            position=np.array([3.0, 0.0], dtype=np.float64),
            velocity=rng.uniform(-0.3, 0.3, size=2).astype(np.float64),
            radius=0.2,
        ),
        "arm7": AgentSnapshot(
            position=np.array([0.0, 3.0], dtype=np.float64),
            velocity=rng.uniform(-0.3, 0.3, size=2).astype(np.float64),
            radius=0.2,
        ),
    }


@settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(min_value=0, max_value=10_000))
def test_centralized_heterogeneous_qp_builds_and_solves(seed: int) -> None:
    """spike_004A §Three-mode taxonomy: heterogeneous CENTRALIZED QP solves cleanly."""
    rng = np.random.default_rng(seed)
    models = _models_2_4_7()
    snaps = _snaps_triangle(rng)
    proposed = {
        uid: rng.uniform(-0.5, 0.5, size=models[uid].action_dim).astype(np.float64)
        for uid in models
    }
    cbf = ExpCBFQP.centralized(control_models=models, cbf_gamma=2.0)
    raw_safe, info = cbf.filter(
        proposed_action=proposed,
        obs={"agent_states": snaps, "meta": {"partner_id": None}},
        state=SafetyState(lambda_=np.zeros(3, dtype=np.float64)),
        bounds=_bounds(),
    )
    safe = cast("dict[str, FloatArray]", raw_safe)
    # Per-agent shape matches the control model's action_dim.
    for uid in ("base", "arm4", "arm7"):
        assert safe[uid].shape == (models[uid].action_dim,), (
            f"safe[{uid}].shape={safe[uid].shape} mismatches action_dim={models[uid].action_dim}"
        )
        assert np.all(np.isfinite(safe[uid]))
    # Three pairs (3 choose 2) of CBF rows.
    assert info["constraint_violation"].shape == (3,)
    assert info["lambda"].shape == (3,)


def test_centralized_heterogeneous_passes_through_when_well_separated() -> None:
    """When all pairs are well-separated the QP returns u_hat (per-uid)."""
    rng = np.random.default_rng(0)
    models = _models_2_4_7()
    snaps = _snaps_triangle(rng)
    # Make velocities zero so no CBF activation under any reasonable u_hat.
    for uid in snaps:
        snaps[uid] = AgentSnapshot(
            position=snaps[uid].position,
            velocity=np.zeros(2, dtype=np.float64),
            radius=snaps[uid].radius,
        )
    proposed = {
        uid: rng.uniform(-0.3, 0.3, size=models[uid].action_dim).astype(np.float64)
        for uid in models
    }
    cbf = ExpCBFQP.centralized(control_models=models, cbf_gamma=2.0)
    raw_safe, _ = cbf.filter(
        proposed_action=proposed,
        obs={"agent_states": snaps, "meta": {"partner_id": None}},
        state=SafetyState(lambda_=np.zeros(3, dtype=np.float64)),
        bounds=_bounds(),
    )
    safe = cast("dict[str, FloatArray]", raw_safe)
    for uid in ("base", "arm4", "arm7"):
        np.testing.assert_allclose(safe[uid], proposed[uid], atol=1e-5)


def test_centralized_rejects_proposed_action_shape_mismatch() -> None:
    """Per-uid proposed action shape must match the control model's action_dim."""
    rng = np.random.default_rng(0)
    models = _models_2_4_7()
    snaps = _snaps_triangle(rng)
    proposed = {
        "base": np.zeros(2, dtype=np.float64),
        "arm4": np.zeros(5, dtype=np.float64),  # wrong: arm4.action_dim=4
        "arm7": np.zeros(7, dtype=np.float64),
    }
    cbf = ExpCBFQP.centralized(control_models=models)
    with pytest.raises(ValueError, match=r"proposed_action\['arm4'\]"):
        cbf.filter(
            proposed_action=proposed,
            obs={"agent_states": snaps, "meta": {"partner_id": None}},
            state=SafetyState(lambda_=np.zeros(3, dtype=np.float64)),
            bounds=_bounds(),
        )


def test_centralized_rejects_uid_not_in_control_models() -> None:
    """The QP refuses uids that were not declared at construction (reviewer P0-2)."""
    rng = np.random.default_rng(0)
    models: dict[str, AgentControlModel] = {}
    models["base"] = DoubleIntegratorControlModel(uid="base", action_dim=2)
    snaps = _snaps_triangle(rng)
    proposed = {
        "base": np.zeros(2, dtype=np.float64),
        "arm4": np.zeros(4, dtype=np.float64),  # not in control_models
    }
    cbf = ExpCBFQP.centralized(control_models=models)
    with pytest.raises(KeyError, match="'arm4'"):
        cbf.filter(
            proposed_action=proposed,
            obs={"agent_states": snaps, "meta": {"partner_id": None}},
            state=SafetyState(lambda_=np.zeros(1, dtype=np.float64)),
            bounds=_bounds(),
        )
