# SPDX-License-Identifier: Apache-2.0
"""Multi-pair aggregation + embodiment-routed override tests for the braking fallback.

Covers ADR-004 risk-mitigation #1 corrections:

1. Per-uid *aggregation* of pairwise repulsion vectors. The prior
   "write override per pair" path lost superposition: a uid in two
   simultaneous dangerous pairs got the last-processed pair's override
   only. Property test constructs the three-agent case where uid_0
   sits between two simultaneously-dangerous partners and asserts the
   uid_0 override reflects the *sum* of the two pairwise unit vectors
   (then scaled to ``bounds.action_norm``), not the last-iteration
   value.

2. Embodiment dispatch via
   :class:`concerto.safety.emergency.EmergencyController`. Unit test
   wires the :class:`JacobianEmergencyController` placeholder and
   asserts ``maybe_brake`` raises :class:`NotImplementedError` rather
   than silently writing a Cartesian-shaped vector into a 7-vector
   action slot.
"""

from __future__ import annotations

import re

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from concerto.safety.api import Bounds
from concerto.safety.braking import maybe_brake
from concerto.safety.cbf_qp import AgentSnapshot
from concerto.safety.emergency import (
    CartesianAccelEmergencyController,
    JacobianEmergencyController,
)


def _bounds(action_norm: float = 5.0) -> Bounds:
    return Bounds(
        action_norm=action_norm,
        action_rate=0.5,
        comm_latency_ms=1.0,
        force_limit=20.0,
    )


def _snap(x: float, y: float, vx: float, vy: float, r: float = 0.2) -> AgentSnapshot:
    return AgentSnapshot(
        position=np.array([x, y], dtype=np.float64),
        velocity=np.array([vx, vy], dtype=np.float64),
        radius=r,
    )


def test_three_agent_collinear_squeeze_aggregates_to_zero() -> None:
    """uid_0 between uid_1 (left) and uid_2 (right) on the x-axis ⇒ ~zero override.

    The two push-apart unit vectors on uid_0 point in opposite x
    directions and cancel out; the prior last-pair-wins path would
    have left uid_0 with a full-magnitude push along whichever pair
    was iterated last, which is the exact bug this fix targets.
    """
    snaps = {
        "uid_0": _snap(0.0, 0.0, 0.0, 0.0),
        "uid_1": _snap(-0.1, 0.0, 1.0, 0.0),
        "uid_2": _snap(0.1, 0.0, -1.0, 0.0),
    }
    proposed = {
        "uid_0": np.zeros(2, dtype=np.float64),
        "uid_1": np.zeros(2, dtype=np.float64),
        "uid_2": np.zeros(2, dtype=np.float64),
    }
    bounds = _bounds(action_norm=5.0)
    override, fired = maybe_brake(proposed, snaps, bounds=bounds)
    assert fired is True
    assert override is not None
    np.testing.assert_allclose(override["uid_0"], [0.0, 0.0], atol=1e-12)
    # The outer two uids each have one repulsion vector ⇒ full
    # emergency-stop magnitude.
    assert np.linalg.norm(override["uid_1"]) == pytest.approx(bounds.action_norm)
    assert np.linalg.norm(override["uid_2"]) == pytest.approx(bounds.action_norm)


@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    gap_x=st.floats(min_value=0.05, max_value=0.30),
    gap_y=st.floats(min_value=0.05, max_value=0.30),
    closing_speed=st.floats(min_value=1.0, max_value=4.0),
)
def test_uid0_override_is_normalised_sum_of_pair_unit_vectors(
    gap_x: float, gap_y: float, closing_speed: float
) -> None:
    """uid_0's override matches normalise-then-saturate of its pair unit vectors.

    Geometry: uid_0 at origin; uid_1 to the left along -x (n_hat_01 =
    +x); uid_2 below along -y (n_hat_02 = +y). The two unit vectors
    are orthogonal — their sum has norm sqrt(2), which the
    :class:`CartesianAccelEmergencyController` then rescales to
    ``bounds.action_norm`` along the net (1, 1)/sqrt(2) direction.

    A last-pair-wins regression would yield an override aligned with
    only one of the two axes (either +x or +y, depending on dict
    iteration order); this test fails immediately for any such
    regression.
    """
    snap_0 = _snap(0.0, 0.0, 0.0, 0.0)
    snap_1 = _snap(-gap_x, 0.0, closing_speed, 0.0)
    snap_2 = _snap(0.0, -gap_y, 0.0, closing_speed)
    snaps = {"uid_0": snap_0, "uid_1": snap_1, "uid_2": snap_2}
    proposed = {
        "uid_0": np.zeros(2, dtype=np.float64),
        "uid_1": np.zeros(2, dtype=np.float64),
        "uid_2": np.zeros(2, dtype=np.float64),
    }
    bounds = _bounds(action_norm=5.0)
    override, fired = maybe_brake(proposed, snaps, bounds=bounds)
    assert fired is True
    assert override is not None

    # Reconstruct the expected aggregate on uid_0: +n_hat from (uid_0,
    # uid_1) plus +n_hat from (uid_0, uid_2). _push_apart_unit_vector
    # returns Dp_i_minus_j / |Dp_i_minus_j|.
    dp01 = snap_0.position - snap_1.position
    dp02 = snap_0.position - snap_2.position
    n_hat_01 = dp01 / np.linalg.norm(dp01)
    n_hat_02 = dp02 / np.linalg.norm(dp02)
    aggregate = n_hat_01 + n_hat_02
    expected = (aggregate / np.linalg.norm(aggregate)) * bounds.action_norm
    np.testing.assert_allclose(override["uid_0"], expected, atol=1e-10)

    # Last-pair-wins regression check: the override must not be aligned
    # with either single pair's repulsion direction alone. Both x and y
    # components must be strictly positive (the aggregate direction is
    # (1, 1)/sqrt(2)).
    assert override["uid_0"][0] > 0.0
    assert override["uid_0"][1] > 0.0
    # Magnitude is at the emergency-stop cap.
    assert np.linalg.norm(override["uid_0"]) == pytest.approx(bounds.action_norm)


def test_uid_with_zero_dangerous_pairs_keeps_proposed_action_unchanged() -> None:
    """A uid in no dangerous pair is not routed through the emergency hook.

    Locks in the contract: ``override[uid] = proposed_action[uid]``
    (not a zero action) when the uid has zero repulsion vectors.
    """
    snaps = {
        "uid_0": _snap(-0.1, 0.0, 1.0, 0.0),
        "uid_1": _snap(0.1, 0.0, -1.0, 0.0),
        "uid_far": _snap(50.0, 0.0, 0.0, 0.0),
    }
    proposed = {
        "uid_0": np.zeros(2, dtype=np.float64),
        "uid_1": np.zeros(2, dtype=np.float64),
        "uid_far": np.array([0.7, -0.3], dtype=np.float64),
    }
    override, fired = maybe_brake(proposed, snaps, bounds=_bounds())
    assert fired is True
    assert override is not None
    np.testing.assert_array_equal(override["uid_far"], [0.7, -0.3])


def test_explicit_controllers_map_is_honoured() -> None:
    """Passing ``emergency_controllers`` replaces the default per-uid dispatch."""
    snaps = {
        "uid_a": _snap(-0.1, 0.0, 1.0, 0.0),
        "uid_b": _snap(0.1, 0.0, -1.0, 0.0),
    }
    proposed = {
        "uid_a": np.zeros(2, dtype=np.float64),
        "uid_b": np.zeros(2, dtype=np.float64),
    }
    controllers = {
        "uid_a": CartesianAccelEmergencyController(),
        "uid_b": CartesianAccelEmergencyController(),
    }
    override, fired = maybe_brake(
        proposed, snaps, bounds=_bounds(), emergency_controllers=controllers
    )
    assert fired is True
    assert override is not None
    # Single-pair case ⇒ magnitudes match the saturation cap on both
    # uids; directions along the line between centres (±x).
    assert np.linalg.norm(override["uid_a"]) == pytest.approx(_bounds().action_norm)
    assert np.linalg.norm(override["uid_b"]) == pytest.approx(_bounds().action_norm)


def test_jacobian_emergency_controller_raises_with_expected_message() -> None:
    """Placeholder fails loudly so 7-DOF uids do not silently corrupt actions.

    ADR-004 risk-mitigation #1 follow-up: until the Stage-1 AS spike
    delivers the Jacobian-aware controller, a 7-DOF uid routed through
    the braking fallback MUST raise rather than write a Cartesian-
    shaped vector into a 7-vector action slot. This test wires the
    placeholder and asserts the raise + message.
    """
    snaps = {
        "arm": _snap(-0.05, 0.0, 1.0, 0.0),
        "base": _snap(0.5, 0.0, -1.0, 0.0),
    }
    proposed = {
        # 7-DOF joint-torque action for the arm; 2-DOF base velocity.
        "arm": np.zeros(7, dtype=np.float64),
        "base": np.zeros(2, dtype=np.float64),
    }
    controllers = {
        "arm": JacobianEmergencyController(),
        "base": CartesianAccelEmergencyController(),
    }
    with pytest.raises(
        NotImplementedError,
        match=re.escape(
            "Jacobian-aware emergency override is a Stage-1 deliverable; "
            "see ADR-004 risk-mitigation #1 follow-up."
        ),
    ):
        maybe_brake(
            proposed,
            snaps,
            bounds=_bounds(),
            emergency_controllers=controllers,
        )
