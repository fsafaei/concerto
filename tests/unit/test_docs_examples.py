# SPDX-License-Identifier: Apache-2.0
"""Code-block tests for the published docs (T3.12).

Plan/03 §4 T3.12: "Code-block-test the example." Each function in
this module mirrors a code block in ``docs/explanation/*.md``
verbatim (modulo the assertion lines and the surrounding test
scaffolding) so the doc and the runtime contract cannot drift —
when either changes, the test fails or a reviewer notices the
mismatch in the diff.
"""

from __future__ import annotations

import numpy as np


def test_why_conformal_walkthrough_example() -> None:
    """Mirror of ``docs/explanation/why-conformal.md`` "Worked example" code block.

    Validates that the three-layer composition example (braking
    fallback + outer exp CBF-QP + conformal lambda update) runs
    end-to-end on a head-on 2-agent scenario.
    """
    # --- BEGIN: code block from docs/explanation/why-conformal.md ---
    from concerto.safety.api import Bounds, SafetyState
    from concerto.safety.braking import maybe_brake
    from concerto.safety.cbf_qp import AgentSnapshot, ExpCBFQP
    from concerto.safety.conformal import update_lambda

    bounds = Bounds(
        action_norm=5.0,
        action_rate=0.5,
        comm_latency_ms=1.0,
        force_limit=20.0,
    )
    state = SafetyState(
        lambda_=np.zeros(1, dtype=np.float64),
        epsilon=-0.05,
        eta=0.01,
    )
    cbf = ExpCBFQP(cbf_gamma=2.0)

    agents = {
        "a": AgentSnapshot(
            position=np.array([-1.0, 0.0], dtype=np.float64),
            velocity=np.array([1.0, 0.0], dtype=np.float64),
            radius=0.2,
        ),
        "b": AgentSnapshot(
            position=np.array([1.0, 0.0], dtype=np.float64),
            velocity=np.array([-1.0, 0.0], dtype=np.float64),
            radius=0.2,
        ),
    }
    nominal = {
        "a": np.zeros(2, dtype=np.float64),
        "b": np.zeros(2, dtype=np.float64),
    }

    # 1. Per-step braking fallback (kinematic backstop).
    override, fired = maybe_brake(nominal, agents, bounds=bounds)
    if fired and override is not None:
        safe = override
    else:
        # 2. Outer exp CBF-QP (Wang-Ames-Egerstedt 2017).
        safe, info = cbf.filter(
            proposed_action=nominal,
            obs={"agent_states": agents, "meta": {"partner_id": "demo"}},
            state=state,
            bounds=bounds,
        )
        # 3. Conformal lambda update (Huriot & Sibai 2025 §IV).
        update_lambda(state, info["loss_k"], in_warmup=False)
    # --- END: code block from docs/explanation/why-conformal.md ---

    # The example runs without exception; verify the safe action is
    # well-shaped and finite so a reader can trust the doc's claim
    # that the loop produces usable output.
    assert set(safe.keys()) == {"a", "b"}
    for uid in ("a", "b"):
        assert safe[uid].shape == (2,)
        assert np.all(np.isfinite(safe[uid]))


def test_why_conformal_comm_wrapper_example() -> None:
    """Mirror of ``docs/explanation/why-conformal.md`` comm-wrapper code block.

    Validates that the Stage-2 CM spike's composed channel constructs
    cleanly with the documented arguments.
    """
    # --- BEGIN: code block from docs/explanation/why-conformal.md ---
    from chamber.comm import (
        URLLC_3GPP_R17,
        CommDegradationWrapper,
        FixedFormatCommChannel,
    )

    channel = CommDegradationWrapper(
        FixedFormatCommChannel(),
        URLLC_3GPP_R17["factory"],
        tick_period_ms=1.0,
        root_seed=0,
    )
    # --- END: code block from docs/explanation/why-conformal.md ---

    # The published example must produce a usable channel object;
    # verify the public surface the doc implicitly promises.
    assert channel is not None
    assert hasattr(channel, "encode")
    assert hasattr(channel, "decode")
    assert hasattr(channel, "reset")
