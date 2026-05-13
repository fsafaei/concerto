# SPDX-License-Identifier: Apache-2.0
"""Integration test: partner-swap lambda reset + warmup loss decrease (T3.11).

Plan/03 §5 ``test_partner_swap.py``: lambda resets correctly on
partner swap; warmup window is observed; per-pair loss decreases over
the warmup window (the swap-transient mitigation per ADR-004
risk-mitigation #2; ADR-006 risk #3).
"""

from __future__ import annotations

import numpy as np
import pytest

from concerto.safety.api import DEFAULT_WARMUP_STEPS, SafetyState
from concerto.safety.conformal import reset_on_partner_swap, update_lambda


def test_partner_swap_resets_lambda_to_lambda_safe() -> None:
    """ADR-004 risk-mitigation #2: ``lambda`` is reset to ``lambda_safe`` on swap."""
    state = SafetyState(
        lambda_=np.array([0.42, -0.13], dtype=np.float64),  # adapted to partner-1
        epsilon=0.05,
        eta=0.05,
    )
    reset_on_partner_swap(state, n_pairs=2, lambda_safe=0.0, n_warmup_steps=DEFAULT_WARMUP_STEPS)
    np.testing.assert_array_equal(state.lambda_, [0.0, 0.0])
    assert state.warmup_steps_remaining == DEFAULT_WARMUP_STEPS


def test_partner_swap_warmup_decrements_to_zero_then_clamps() -> None:
    """ADR-004 risk-mitigation #2: warmup_steps_remaining decrements + clamps at 0."""
    state = SafetyState(
        lambda_=np.zeros(1, dtype=np.float64),
        epsilon=0.05,
        eta=0.05,
        warmup_steps_remaining=5,
    )
    for expected_remaining in (4, 3, 2, 1, 0):
        update_lambda(state, np.zeros(1, dtype=np.float64), in_warmup=True)
        assert state.warmup_steps_remaining == expected_remaining
    # Further warmup-mode calls clamp at 0 (no underflow).
    update_lambda(state, np.zeros(1, dtype=np.float64), in_warmup=True)
    assert state.warmup_steps_remaining == 0


def test_partner_swap_warmup_widens_step_relative_to_steady_state() -> None:
    """During warmup, the effective epsilon multiplier (1.5x) widens lambda step.

    Same loss stream applied to a warmup-mode and a steady-state-mode
    state should diverge: warmup pushes lambda further per step
    (ADR-004 risk-mitigation #2).
    """
    base = SafetyState(
        lambda_=np.zeros(1, dtype=np.float64),
        epsilon=0.05,
        eta=0.1,
        warmup_steps_remaining=10,
    )
    warm = SafetyState(
        lambda_=np.zeros(1, dtype=np.float64),
        epsilon=0.05,
        eta=0.1,
        warmup_steps_remaining=10,
    )
    losses = np.array([0.0], dtype=np.float64)
    for _ in range(5):
        update_lambda(base, losses, in_warmup=False)
        update_lambda(warm, losses, in_warmup=True)
    assert warm.lambda_[0] > base.lambda_[0]


def test_partner_swap_per_pair_loss_decreases_over_warmup() -> None:
    """plan/03 §5: per-pair loss decreases over the warmup window.

    Closed-loop synthetic mirroring ADR-006 risk #3: lambda was
    adapted high to the *previous* partner; on swap, lambda_safe
    captures the worst-case envelope so the warmup begins with
    elevated loss. The conformal update drives lambda (and therefore
    l_k) toward the steady-state target across the warmup, so the
    last quartile's mean loss is below the first quartile's.
    """
    rng = np.random.default_rng(0)
    state = SafetyState(
        lambda_=np.zeros(1, dtype=np.float64),
        epsilon=0.05,
        eta=0.1,
    )
    # Worst-case bound seeds the swap; warmup window primed.
    reset_on_partner_swap(state, n_pairs=1, lambda_safe=1.0, n_warmup_steps=200)

    losses_first_quartile: list[float] = []
    losses_last_quartile: list[float] = []
    n = 200
    for k in range(n):
        # Stationary prediction-vs-truth gap stream — Theorem 3 setup.
        e_k = float(rng.normal(0.0, 0.3))
        l_k = max(0.0, float(state.lambda_[0]) - e_k)
        if k < n // 4:
            losses_first_quartile.append(l_k)
        elif k >= 3 * n // 4:
            losses_last_quartile.append(l_k)
        update_lambda(
            state,
            np.array([l_k], dtype=np.float64),
            in_warmup=state.warmup_steps_remaining > 0,
        )

    early_mean = float(np.mean(losses_first_quartile))
    late_mean = float(np.mean(losses_last_quartile))
    assert late_mean < early_mean, (
        f"per-pair loss did not decrease over warmup: early {early_mean:.4f}, late {late_mean:.4f}"
    )


def test_partner_swap_lambda_safe_can_be_nonzero() -> None:
    """ADR-006 risk #3: ``lambda_safe`` is configurable, not zero by default."""
    state = SafetyState(
        lambda_=np.zeros(1, dtype=np.float64),
        epsilon=0.05,
        eta=0.05,
    )
    reset_on_partner_swap(state, n_pairs=1, lambda_safe=0.25, n_warmup_steps=10)
    assert state.lambda_[0] == 0.25
    assert state.warmup_steps_remaining == 10


def test_partner_swap_changes_pair_count() -> None:
    """ADR-004 risk-mitigation #2: partner swap can also change the pair count."""
    state = SafetyState(
        lambda_=np.array([0.5, -0.3, 0.1], dtype=np.float64),  # 3 pairs
        epsilon=0.05,
        eta=0.05,
    )
    reset_on_partner_swap(state, n_pairs=1, lambda_safe=0.0, n_warmup_steps=20)
    assert state.lambda_.shape == (1,)
    assert state.warmup_steps_remaining == 20


def test_partner_swap_consumer_gates_on_partner_id_none() -> None:
    """ADR-004 risk-mitigation #2 + plan/03 §8: ``partner_id is None`` ⇒ single-partner mode.

    The consumer side (PR5 ExpCBFQP) reads ``obs["meta"]["partner_id"]``
    and must NOT raise on None — single-partner Phase-0 mode runs
    without M4's producer wired up. Document the contract by importing
    the filter and asserting it tolerates the None marker.
    """
    from concerto.safety.api import Bounds
    from concerto.safety.cbf_qp import AgentSnapshot, ExpCBFQP

    cbf = ExpCBFQP()
    bounds = Bounds(action_norm=2.0, action_rate=0.5, comm_latency_ms=1.0, force_limit=20.0)
    snaps = {
        "a": AgentSnapshot(
            position=np.zeros(2, dtype=np.float64),
            velocity=np.zeros(2, dtype=np.float64),
            radius=0.2,
        ),
        "b": AgentSnapshot(
            position=np.array([10.0, 0.0], dtype=np.float64),
            velocity=np.zeros(2, dtype=np.float64),
            radius=0.2,
        ),
    }
    proposed = {
        "a": np.zeros(2, dtype=np.float64),
        "b": np.zeros(2, dtype=np.float64),
    }
    state = SafetyState(lambda_=np.zeros(1, dtype=np.float64))
    # partner_id is None: must NOT raise.
    safe, _ = cbf.filter(
        proposed_action=proposed,
        obs={"agent_states": snaps, "meta": {"partner_id": None}},
        state=state,
        bounds=bounds,
    )
    assert set(safe.keys()) == {"a", "b"}


def test_partner_swap_update_lambda_mismatch_after_pair_count_change_raises() -> None:
    """ADR-004 risk-mitigation #2: mid-episode pair-count change without reset surfaces loudly.

    If a caller forgets to call :func:`reset_on_partner_swap` after a
    partner-set change, the next ``update_lambda`` call sees a shape
    mismatch between ``state.lambda_`` and the supplied per-pair loss
    vector and raises ValueError so the bug doesn't silently corrupt
    the conformal state.
    """
    state = SafetyState(
        lambda_=np.zeros(2, dtype=np.float64),  # 2 pairs
        epsilon=0.05,
        eta=0.05,
    )
    # Loss vector for 1 pair (post-swap) but state still has 2 — caller forgot to reset.
    with pytest.raises(ValueError, match="shape"):
        update_lambda(state, np.zeros(1, dtype=np.float64), in_warmup=False)
