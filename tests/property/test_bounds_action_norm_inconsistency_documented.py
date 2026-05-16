# SPDX-License-Identifier: Apache-2.0
"""xfail-pinned regression flag for the ``Bounds.action_norm`` semantic inconsistency.

External-review finding P1-3 (2026-05-16): ``Bounds.action_norm`` is
consumed by two safety layers with mismatched semantics —
:class:`concerto.safety.cbf_qp.ExpCBFQP` enforces it as a
**per-component L-infinity** bound, while
:class:`concerto.safety.emergency.CartesianAccelEmergencyController`
reads it as an **L2** magnitude cap on the emergency override. For a
``d``-dimensional action these can disagree by ``sqrt(d)``: the CBF
filter may authorise an action the emergency fallback cannot deliver.

Tracking issue: https://github.com/fsafaei/concerto/issues/146.

This test is **intentionally marked ``@pytest.mark.xfail(strict=True)``**
so that:

- While the inconsistency is unresolved, the test fails *and the
  failure is expected* — `xfail strict` keeps it visible without
  breaking CI.
- When the field split lands (`action_linf_component` +
  `cartesian_accel_capacity`) and the inconsistency is resolved, the
  test flips from ``xfail`` to ``xpass``. Under ``strict=True``, an
  unexpected pass *fails* CI, forcing a follow-up PR to either:
  (a) remove the ``xfail`` marker (and let this test stand as a
      coverage net for the post-split consistency), or
  (b) delete the test entirely if the post-split contract makes the
      inconsistency unreachable by construction.

The test does NOT itself fix anything. It is a guarantee that the fix
in the tracked issue, when it lands, cannot land *silently*.

The two assertions:

1. ``ExpCBFQP`` permits an action ``u = (1.0, 1.0, 1.0)`` (L-infinity
   norm 1.0) at ``action_norm=1.0`` — the L-infinity envelope. The
   Euclidean norm of that action is ``sqrt(3) ~= 1.732``.
2. ``CartesianAccelEmergencyController`` produces an override whose
   Euclidean norm equals ``action_norm = 1.0`` — the L2 envelope.

The inconsistency: CBF authorised ``||u||_2 ~ 1.732`` for the ego in
nominal operation, but if the same step fires the emergency fallback,
the override magnitude is L2-capped at ``1.0``. The fallback cannot
deliver what the CBF derivation assumed feasible.

References:
- ADR-004 §Open questions (2026-05-16 amendment): names the four
  touch points and the field-split path.
- ``concerto.safety.api.Bounds`` field docstring: names the safe
  operator pattern (``action_norm = capacity / sqrt(d)``).
- ``src/concerto/safety/cbf_qp.py`` lines 824, 964 — the L-infinity
  consumer.
- ``src/concerto/safety/emergency.py`` line 174 — the L2 consumer.
"""

from __future__ import annotations

import numpy as np
import pytest

from concerto.safety.api import (
    Bounds,
    DoubleIntegratorControlModel,
)
from concerto.safety.cbf_qp import AgentSnapshot, ExpCBFQP
from concerto.safety.emergency import CartesianAccelEmergencyController


@pytest.mark.xfail(
    strict=True,
    reason=(
        "ADR-004 §Open questions: Bounds.action_norm has mismatched semantics "
        "between ExpCBFQP (L-infinity per-component) and the emergency "
        "controller (L2 magnitude). When issue #146 closes and the field is "
        "split into action_linf_component + cartesian_accel_capacity, this "
        "test must xpass and the marker must be removed in a follow-up PR."
    ),
)
def test_bounds_action_norm_cbf_and_emergency_agree_on_action_set() -> None:
    """The two safety consumers of ``Bounds.action_norm`` must envelope the same action set.

    Under the post-split contract, the ego action ``u`` that the CBF
    filter accepts as safe at the L-infinity bound MUST also be an
    action the emergency controller is capable of delivering at the
    corresponding L2 magnitude — i.e. the two layers must be
    consistent on what "max permissible action" means.

    This test fails today (xfail) because ``action_norm`` is consumed
    inconsistently. It will xpass once the field-split lands.
    """
    bounds = Bounds(
        action_norm=1.0,
        action_rate=0.5,
        comm_latency_ms=1.0,
        force_limit=20.0,
    )

    # --- Side A: ExpCBFQP authorises ``u = (1, 1)`` (||u||_inf = 1.0). ---
    # Two well-separated agents so the CBF constraint is inactive and
    # the QP reduces to projecting the proposed action onto the
    # action-bound box. Under the current L-infinity-per-component
    # convention the QP returns the proposed action unchanged.
    cbf = ExpCBFQP.centralized(
        control_models={
            "a": DoubleIntegratorControlModel(uid="a", action_dim=2),
            "b": DoubleIntegratorControlModel(uid="b", action_dim=2),
        },
        cbf_gamma=2.0,
    )
    snaps = {
        "a": AgentSnapshot(
            position=np.array([-10.0, 0.0], dtype=np.float64),
            velocity=np.array([0.0, 0.0], dtype=np.float64),
            radius=0.2,
        ),
        "b": AgentSnapshot(
            position=np.array([10.0, 0.0], dtype=np.float64),
            velocity=np.array([0.0, 0.0], dtype=np.float64),
            radius=0.2,
        ),
    }
    proposed = {
        "a": np.array([1.0, 1.0], dtype=np.float64),  # ||u||_inf = 1.0
        "b": np.array([0.0, 0.0], dtype=np.float64),
    }
    from concerto.safety.api import SafetyState

    raw_safe, _ = cbf.filter(
        proposed_action=proposed,
        obs={"agent_states": snaps, "meta": {"partner_id": None}},
        state=SafetyState(lambda_=np.zeros(1, dtype=np.float64)),
        bounds=bounds,
    )
    assert isinstance(raw_safe, dict)
    safe_a = raw_safe["a"]
    cbf_allowed_l2 = float(np.linalg.norm(safe_a))

    # --- Side B: emergency controller L2-caps at ``action_norm = 1.0``. ---
    # Construct a pair-wise repulsion that aggregates to a unit vector
    # along ``+x``; the saturation step then scales it to magnitude
    # ``bounds.action_norm``.
    controller = CartesianAccelEmergencyController()
    override = controller.compute_override(
        agent_state=snaps["a"],
        pairwise_repulsion_vectors=[np.array([1.0, 0.0], dtype=np.float64)],
        bounds=bounds,
    )
    emergency_max_l2 = float(np.linalg.norm(override))

    # The post-split contract is: the two layers envelope the SAME L2
    # magnitude. Today they don't — CBF authorises sqrt(2) ~= 1.414 (a
    # 2-D action with each component at 1.0), emergency caps at 1.0.
    # The assertion below is the post-split target; it fails today (the
    # xfail marker captures the expected failure) and will pass once
    # the field-split lands.
    assert cbf_allowed_l2 == pytest.approx(emergency_max_l2, abs=1e-9), (
        f"Bounds.action_norm semantic inconsistency: ExpCBFQP authorises "
        f"||u||_2 ~ {cbf_allowed_l2:.4f}, emergency caps at "
        f"{emergency_max_l2:.4f}. See issue #146 for the field-split fix."
    )
