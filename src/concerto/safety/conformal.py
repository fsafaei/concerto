# SPDX-License-Identifier: Apache-2.0
"""Conformal CBF slack update + partner-trajectory predictor stub (ADR-004 §Decision).

Implements Huriot & Sibai 2025 §IV (ICRA 2025; arXiv:2409.18862v4): the
additive slack ``lambda`` on each pairwise CBF constraint is updated
each control step by Theorem 3's rule

    lambda_{k+1} = lambda_k + eta * (eps - l_k)

where ``l_k`` is the per-pair Huriot-Sibai loss — the worst-case CBF-
constraint gap between the conformal and ground-truth constraints
(§IV.A). The default ``eps = -0.05`` (plan/03 §2; ADR-004 §Decision)
biases ``lambda`` toward tighter constraints during contact-rich
manipulation; positive ``eps`` is the standard Theorem 3 regime where
the empirical average loss tracks ``eps``.

M3 ships a constant-velocity partner-trajectory predictor stub per
plan/03 §3.3; the AoI-conditioned predictor (Ballotta-Talak 2024) is a
Phase-1 upgrade.

Partner-swap (ADR-004 risk-mitigation #2; ADR-006 risk #3): on
``obs["meta"]["partner_id"]`` identity change, ``lambda`` is reset to
``lambda_safe`` (the QP-feasibility-preserving value derived from
:class:`Bounds`) and the next ``warmup_steps_remaining`` steps run with
``eps_warmup = 1.5 * eps`` (narrower target — for the default negative
``eps``, "narrower" means more negative, i.e., tighter still).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from concerto.safety.api import DEFAULT_WARMUP_STEPS
from concerto.safety.cbf_qp import AgentSnapshot

if TYPE_CHECKING:
    from concerto.safety.api import FloatArray, SafetyState

#: Multiplier applied to ``state.epsilon`` during the partner-swap
#: warmup window (ADR-004 risk-mitigation #2). For positive ``eps`` the
#: target average loss is reduced; for negative ``eps`` (conservative
#: manipulation regime) the target becomes more negative — both directions
#: tighten the constraint relative to the steady-state target.
_WARMUP_EPSILON_FACTOR: float = 1.5


def update_lambda(
    state: SafetyState,
    loss_k: FloatArray,
    *,
    in_warmup: bool = False,
) -> None:
    """Conformal slack update (Huriot & Sibai 2025 §IV; ADR-004 §Decision).

    Applies Theorem 3's rule ``lambda_{k+1} = lambda_k + eta * (eps - l_k)``
    in place on ``state.lambda_``. During the partner-swap warmup window
    (``in_warmup=True``), ``eps`` is multiplied by
    :data:`_WARMUP_EPSILON_FACTOR` so the constraint stays tighter than
    the steady-state target (ADR-004 risk-mitigation #2). The
    ``warmup_steps_remaining`` counter decrements each warmup step
    (clamped to zero).

    Args:
        state: Mutable :class:`concerto.safety.api.SafetyState`;
            ``lambda_`` is updated in place.
        loss_k: Per-pair loss ``l_k``, shape ``(N_pairs,)`` matching
            ``state.lambda_``. Computed from the pairwise CBF gap by
            :func:`compute_per_pair_loss` (Huriot & Sibai §IV.A).
        in_warmup: When True, narrow ``eps`` and decrement
            ``warmup_steps_remaining`` (ADR-004 risk-mitigation #2).

    Raises:
        ValueError: If ``loss_k`` shape mismatches ``state.lambda_``.
    """
    if loss_k.shape != state.lambda_.shape:
        msg = f"loss_k shape {loss_k.shape} mismatches state.lambda_ shape {state.lambda_.shape}"
        raise ValueError(msg)
    epsilon = state.epsilon * _WARMUP_EPSILON_FACTOR if in_warmup else state.epsilon
    state.lambda_ = state.lambda_ + state.eta * (epsilon - loss_k)
    if in_warmup:
        state.warmup_steps_remaining = max(0, state.warmup_steps_remaining - 1)


def constant_velocity_predict(snap: AgentSnapshot, dt: float) -> AgentSnapshot:
    """Constant-velocity partner-trajectory predictor stub (ADR-004 §Decision).

    Plan/03 §3.3 specifies the M3 stub: predict the next state assuming
    velocity is constant over the lookahead. The AoI-conditioned
    predictor (Ballotta-Talak 2024) is a Phase-1 upgrade.

    Args:
        snap: Current per-agent kinematic state.
        dt: Lookahead horizon in seconds (positive).

    Returns:
        Predicted :class:`AgentSnapshot` at ``t + dt`` under the constant-
        velocity model. Radius is preserved.
    """
    return AgentSnapshot(
        position=(snap.position + snap.velocity * dt).astype(np.float64, copy=False),
        velocity=snap.velocity.copy(),
        radius=snap.radius,
    )


def compute_per_pair_loss(
    predicted_h: FloatArray,
    actual_h: FloatArray,
) -> FloatArray:
    """Per-pair Huriot-Sibai 2025 §IV.A loss (ADR-004 §Decision).

    ``l_k = max(0, predicted_h - actual_h)`` per pair: positive when the
    conformal barrier *over*estimated the safe set (we believed it was
    safer than reality), zero otherwise. The conformal update consumes
    this signal so ``lambda`` tracks the gap between predicted and
    ground-truth constraints.

    Args:
        predicted_h: Conformal barrier values per pair, shape
            ``(N_pairs,)``, dtype ``float64``.
        actual_h: Ground-truth barrier values per pair (same shape).

    Returns:
        Per-pair loss, shape ``(N_pairs,)``, dtype ``float64``.

    Raises:
        ValueError: If shapes mismatch.
    """
    if predicted_h.shape != actual_h.shape:
        msg = f"predicted_h shape {predicted_h.shape} mismatches actual_h shape {actual_h.shape}"
        raise ValueError(msg)
    diff: FloatArray = (predicted_h - actual_h).astype(np.float64, copy=False)
    return np.maximum(0.0, diff)


def reset_on_partner_swap(
    state: SafetyState,
    *,
    n_pairs: int,
    lambda_safe: float = 0.0,
    n_warmup_steps: int = DEFAULT_WARMUP_STEPS,
) -> None:
    """Reset lambda to lambda_safe and enter warmup (ADR-004 risk-mitigation #2).

    On partner identity change (detected via
    ``obs["meta"]["partner_id"]`` mismatch with the previous step), the
    conformal layer's stationarity assumption breaks (ADR-006 risk #3).
    This helper resets ``state.lambda_`` to a vector of ``lambda_safe``
    values and primes ``warmup_steps_remaining`` so the next
    ``n_warmup_steps`` calls to :func:`update_lambda` use the narrower
    warmup target.

    Args:
        state: Mutable :class:`SafetyState`.
        n_pairs: Number of agent pairs (sets the new ``lambda_`` length).
        lambda_safe: QP-feasibility-preserving value (default ``0.0``;
            ADR-006 §Decision specifies the worst-case bounded-prediction-
            error envelope at ``lambda``).
        n_warmup_steps: Length of the high-caution window (default
            :data:`concerto.safety.api.DEFAULT_WARMUP_STEPS` = 50).
    """
    state.lambda_ = np.full(n_pairs, lambda_safe, dtype=np.float64)
    state.warmup_steps_remaining = n_warmup_steps


__all__ = [
    "compute_per_pair_loss",
    "constant_velocity_predict",
    "reset_on_partner_swap",
    "update_lambda",
]
