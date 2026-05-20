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
``lambda_safe`` and the next ``warmup_steps_remaining`` steps run with
``eps_warmup = 1.5 * eps`` (narrower target — for the default negative
``eps``, "narrower" means more negative, i.e., tighter still). The
default ``lambda_safe=0.0`` is a Phase-0 conservative *placeholder*,
not a derived QP-feasibility-preserving value; the derivation
``lambda_safe(bounds, predictor_error_bound, dt, pair_geometry)`` is
deferred to ADR-004 §Open questions. See :func:`reset_on_partner_swap`
for the per-caller override guidance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from concerto.safety.api import (
    DEFAULT_WARMUP_STEPS,
    canonical_pair_order,
    make_lambda_dict,
    make_pair_keys,
)
from concerto.safety.cbf_qp import AgentSnapshot, pair_h_value

if TYPE_CHECKING:
    from collections.abc import Iterable

    from concerto.safety.api import FloatArray, LambdaDict, SafetyState

#: Multiplier applied to ``state.epsilon`` during the partner-swap
#: warmup window (ADR-004 risk-mitigation #2). For positive ``eps`` the
#: target average loss is reduced; for negative ``eps`` (conservative
#: manipulation regime) the target becomes more negative — both directions
#: tighten the constraint relative to the steady-state target.
_WARMUP_EPSILON_FACTOR: float = 1.5


def update_lambda(
    state: SafetyState,
    loss_k: LambdaDict,
    *,
    in_warmup: bool = False,
    lambda_bound: float | None = None,
) -> None:
    """Conformal slack update (Huriot & Sibai 2025 §IV; ADR-004 §Decision).

    Applies Theorem 3's rule ``lambda_{k+1} = lambda_k + eta * (eps - l_k)``
    in place on every entry of ``state.lambda_``. This is the low-level
    primitive: callers MUST supply ``loss_k`` as the Huriot & Sibai
    §IV.A prediction-gap loss — i.e. the output of
    :func:`compute_per_pair_loss` against a partner-trajectory
    predictor — and NOT the constraint-violation signal ``-h`` from the
    CBF backbone. Theorem 3's risk bound holds for the prediction gap
    only; driving the update from ``-h`` would conflate per-step
    constraint violations with predictor error and break the cited
    guarantee. For the wired-up form that consumes a predictor stub
    directly, use :func:`update_lambda_from_predictor`.

    During the partner-swap warmup window (``in_warmup=True``), ``eps``
    is multiplied by :data:`_WARMUP_EPSILON_FACTOR` so the constraint
    stays tighter than the steady-state target (ADR-004
    risk-mitigation #2). The ``warmup_steps_remaining`` counter
    decrements each warmup step (clamped to zero).

    Symmetric clamp (P1.05.7 / issue #180; ADR-004 §Decision Rev 6):
    when ``lambda_bound`` is not ``None``, each per-pair λ is clamped
    to ``[-lambda_bound, +lambda_bound]`` *after* the Theorem 3 update.
    The clamp is symmetric (both directions) per the #178 lesson that
    asymmetric safety surfaces have credibility blind spots. The
    bound is supplied by callers as ``clamp_floor_ratio x
    cartesian_accel_capacity`` (production wiring lives in
    :func:`concerto.training.ego_aht.train`'s safety block; defaults
    ``0.7 x 10 = 7.0`` so the audit-gate's symmetric predicate A
    (``|λ| >= 0.9 x cap = 9``) reserves a 2.0 m/s² buffer above the
    clamp boundary). When ``lambda_bound`` is ``None`` (default), no
    clamp — preserves the Theorem 3 primitive for analysis tests +
    backwards compat with all pre-P1.05.7 callers.

    Trade-off (ADR-004 §Decision Revision 6): the clamped variant
    trades Theorem 3's exact long-run average-loss bound for
    operational robustness — the bound now holds modulo the clamped
    boundary, which is consistent with the engineering meaning of
    "conformal slack bounded by the conservative manipulation
    envelope". Pre-launch evidence (P1.05 100k-frame AS-hetero probe)
    measured drift exactly ``-η x |ε| = -5e-04`` per step, projecting
    unclamped λ_ss ≈ -49.76 — clamp engages at step ~14000 under
    production defaults and pins λ at ``-lambda_bound`` for the
    remaining ~86k frames.

    Args:
        state: Mutable :class:`concerto.safety.api.SafetyState`;
            ``lambda_`` is updated in place.
        loss_k: Per-pair prediction-gap loss ``l_k = max(0, predicted_h -
            actual_h)`` as a :data:`LambdaDict` keyed by canonical
            UID-pair tuples (identical key-set to ``state.lambda_``).
            Derived from :func:`compute_per_pair_loss` (Huriot &
            Sibai §IV.A), not from ``FilterInfo["constraint_violation"]``.
        in_warmup: When True, narrow ``eps`` and decrement
            ``warmup_steps_remaining`` (ADR-004 risk-mitigation #2).
        lambda_bound: Optional symmetric clamp magnitude. ``None``
            (default) ⇒ Theorem 3 unclamped primitive (pre-P1.05.7
            behaviour). When a strictly positive float, each λ is
            clamped to ``[-lambda_bound, +lambda_bound]`` after the
            Theorem 3 update.

    Raises:
        ValueError: If ``loss_k`` does not share the exact key set of
            ``state.lambda_`` (issue #144 made pair-keying a type
            invariant; cross-uid mismatch is loud-fail at this seam),
            or if ``lambda_bound`` is non-positive.
    """
    if loss_k.keys() != state.lambda_.keys():
        msg = (
            f"loss_k key set {sorted(loss_k.keys())!r} mismatches "
            f"state.lambda_ key set {sorted(state.lambda_.keys())!r}"
        )
        raise ValueError(msg)
    if lambda_bound is not None and lambda_bound <= 0.0:
        msg = (
            f"lambda_bound must be strictly positive when set; got {lambda_bound}. "
            "Pass ``None`` to skip the clamp."
        )
        raise ValueError(msg)
    epsilon = state.epsilon * _WARMUP_EPSILON_FACTOR if in_warmup else state.epsilon
    for key, current in state.lambda_.items():
        updated = current + state.eta * (epsilon - loss_k[key])
        if lambda_bound is not None:
            # Symmetric clamp (#180): both directions. np.clip avoided
            # for the scalar inline-path; the equivalent min/max is
            # cheaper + clearer at this scale.
            if updated > lambda_bound:
                updated = lambda_bound
            elif updated < -lambda_bound:
                updated = -lambda_bound
        state.lambda_[key] = updated
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


def compute_prediction_gap_for_pairs(
    snaps_now: dict[str, AgentSnapshot],
    snaps_predicted: dict[str, AgentSnapshot],
    *,
    alpha_pair: float,
    gamma: float,
) -> LambdaDict:
    """Per-pair Huriot & Sibai §IV.A prediction-gap loss from snapshot pairs (ADR-004 §Decision).

    Evaluates the Wang-Ames-Egerstedt 2017 §III pairwise barrier value
    on both ``snaps_now`` (the actual current state) and
    ``snaps_predicted`` (the prediction made at the previous step under
    a partner-trajectory predictor, e.g. :func:`constant_velocity_predict`)
    and returns the per-pair loss
    ``l_k = max(0, predicted_h - actual_h)`` — positive when the
    predictor over-estimated the safe set, zero otherwise.

    Args:
        snaps_now: Per-agent snapshots at the current step ``k+1``.
        snaps_predicted: Per-agent snapshots predicted for step ``k+1``
            from the previous step's state.
        alpha_pair: Joint braking capacity used in the barrier
            ``h_ij = sqrt(2 * alpha_pair * max(|Dp| - D_s, 0)) +
            (Dp^T/|Dp|) Dv`` (Wang-Ames-Egerstedt 2017 §III). Typically
            ``2 * bounds.cartesian_accel_capacity`` for symmetric
            agents (the barrier formula's ``alpha`` is a Cartesian
            acceleration capacity, P1.02 § ADR-004 amendment).
        gamma: Class-K function gain. Accepted for forward
            compatibility with drift-aware variants; the
            relative-degree-1 barrier value itself does not depend on
            ``gamma``.

    Returns:
        Per-pair loss as a :data:`LambdaDict` keyed by canonical
        UID-pair tuples (issue #144 / ADR-014 v3). Dict insertion
        order does not affect the result; the only invariant the
        caller must satisfy is that the *key sets* of ``snaps_now``
        and ``snaps_predicted`` match.

    Raises:
        ValueError: If the two snapshot dicts do not share identical
            key sets (the pair-key alignment with ``state.lambda_``
            depends on it). Pre-amendment this also rejected differing
            insertion orders; that constraint is lifted in this
            release because both inputs are keyed by canonical UID-pair
            tuples internally (external-review P1, 2026-05-16; issue #144).
    """
    del gamma  # currently unused; kept on the signature for forward compat
    if set(snaps_now.keys()) != set(snaps_predicted.keys()):
        msg = (
            "snaps_now and snaps_predicted must share identical key sets; "
            f"got {sorted(snaps_now.keys())!r} vs {sorted(snaps_predicted.keys())!r}"
        )
        raise ValueError(msg)
    pair_keys = make_pair_keys(snaps_now.keys())
    loss: LambdaDict = {}
    for uid_i, uid_j in pair_keys:
        predicted_h = pair_h_value(
            snaps_predicted[uid_i], snaps_predicted[uid_j], alpha_pair=alpha_pair
        )
        actual_h = pair_h_value(snaps_now[uid_i], snaps_now[uid_j], alpha_pair=alpha_pair)
        loss[(uid_i, uid_j)] = max(0.0, float(predicted_h - actual_h))
    return loss


def update_lambda_from_predictor(
    state: SafetyState,
    snaps_now: dict[str, AgentSnapshot],
    snaps_prev: dict[str, AgentSnapshot],
    *,
    alpha_pair: float,
    gamma: float,
    dt: float,
    in_warmup: bool = False,
    lambda_bound: float | None = None,
) -> LambdaDict:
    """Wire the constant-velocity predictor into the conformal update (ADR-004 §Decision).

    Computes the predictor's forecast for step ``k+1`` from the previous
    step's snapshots ``snaps_prev``, evaluates the per-pair prediction-
    gap loss against the actual ``snaps_now``, and applies the Huriot &
    Sibai 2025 §IV update rule via :func:`update_lambda`. The signal
    driving ``lambda`` is the predictor's error from one step ago — the
    quantity Theorem 3's risk bound is stated against, not the per-step
    CBF gap from the QP backbone.

    The ``lambda_bound`` kwarg (P1.05.7 / issue #180) is forwarded to
    :func:`update_lambda` for the symmetric clamp; see that function's
    docstring for the engineering motivation + the clamp-vs-audit-gate
    boundary buffer story.

    Args:
        state: Mutable :class:`SafetyState`; ``lambda_`` is updated in
            place.
        snaps_now: Per-agent snapshots at the current step ``k+1``.
        snaps_prev: Per-agent snapshots at the previous step ``k``.
        alpha_pair: Joint braking capacity used in the barrier
            evaluation (same value the CBF backbone uses, typically
            ``2 * bounds.cartesian_accel_capacity`` for symmetric
            agents).
        gamma: Class-K function gain. Forwarded for forward compat
            with drift-aware loss forms.
        dt: Control-step duration in seconds. The predictor extrapolates
            ``snaps_prev`` forward by ``dt`` to obtain the prediction
            of ``snaps_now``.
        in_warmup: When True, narrow ``eps`` and decrement
            ``warmup_steps_remaining`` (ADR-004 risk-mitigation #2).
        lambda_bound: Optional symmetric clamp magnitude
            (``[-lambda_bound, +lambda_bound]``). Forwarded to
            :func:`update_lambda`. ``None`` ⇒ Theorem 3 unclamped
            primitive (pre-P1.05.7 behaviour); a positive float ⇒
            clamp engaged. The Phase-1 training rollout supplies
            ``cfg.safety.clamp_floor_ratio x bounds.cartesian_accel_capacity``.

    Returns:
        The per-pair prediction-gap loss :data:`LambdaDict` used to
        drive the update — callers populate
        :class:`concerto.safety.api.FilterInfo` ``"prediction_gap_loss"``
        from this for the ADR-014 three-table report.

    Raises:
        ValueError: If ``snaps_now`` and ``snaps_prev`` do not share
            identical key *sets* (insertion order is tolerated as of
            the 2026-05-16 canonical-pair-keying amendment; see
            ADR-004 §Decision), or if the resulting loss key-set
            mismatches ``state.lambda_``, or if ``lambda_bound`` is
            non-positive when set (forwarded loud-fail from
            :func:`update_lambda`).
    """
    if set(snaps_now.keys()) != set(snaps_prev.keys()):
        msg = (
            "snaps_now and snaps_prev must share identical key sets; "
            f"got {sorted(snaps_now.keys())!r} vs {sorted(snaps_prev.keys())!r}"
        )
        raise ValueError(msg)
    uids = canonical_pair_order(snaps_now.keys())
    snaps_predicted = {uid: constant_velocity_predict(snaps_prev[uid], dt) for uid in uids}
    loss = compute_prediction_gap_for_pairs(
        snaps_now,
        snaps_predicted,
        alpha_pair=alpha_pair,
        gamma=gamma,
    )
    update_lambda(state, loss, in_warmup=in_warmup, lambda_bound=lambda_bound)
    return loss


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
    uids: Iterable[str],
    lambda_safe: float = 0.0,
    n_warmup_steps: int = DEFAULT_WARMUP_STEPS,
) -> None:
    """Reset lambda to lambda_safe and enter warmup (ADR-004 risk-mitigation #2).

    On partner identity change (detected via
    ``obs["meta"]["partner_id"]`` mismatch with the previous step), the
    conformal layer's stationarity assumption breaks (ADR-006 risk #3).
    This helper rebuilds ``state.lambda_`` as a fresh
    :data:`concerto.safety.api.LambdaDict` over the canonical pair set
    for ``uids`` (every entry initialised to ``lambda_safe``) and
    primes ``warmup_steps_remaining`` so the next ``n_warmup_steps``
    calls to :func:`update_lambda` use the narrower warmup target.

    Issue #144 / ADR-014 v3 (2026-05-19) replaced the prior
    ``n_pairs: int`` argument with ``uids: Iterable[str]``: the dict
    representation of ``lambda_`` requires the actual UID set to build
    the canonical pair-key tuples, not just the pair count.

    The default ``lambda_safe=0.0`` is a Phase-0 conservative
    *placeholder*, **not** a derived QP-feasibility-preserving value.
    The derived form ``lambda_safe(bounds, predictor_error_bound, dt,
    pair_geometry)`` — the worst-case bounded-prediction-error envelope
    that would preserve QP feasibility — has not been implemented; its
    derivation is deferred to ADR-004 §"Open questions deferred to a
    later ADR" (see also the ADR-INDEX open-work footnote ``a``).
    Callers who require provable safety should override ``lambda_safe``
    explicitly with a value derived from their own problem's bounds,
    not rely on the default. The Phase-0 partner-swap tests
    (``tests/integration/test_partner_swap.py``) exercise both the
    default and explicit-override paths.

    Args:
        state: Mutable :class:`SafetyState`.
        uids: Iterable of agent UIDs in the (post-swap) partner set;
            determines the canonical pair-key set of the rebuilt
            ``state.lambda_``.
        lambda_safe: Per-pair slack used at reset. **Phase-0 placeholder
            default: ``0.0``.** The derived QP-feasibility-preserving form
            named in ADR-006 §Decision is not implemented; see the
            ADR-004 §Open questions entry above. Callers that need a
            provable safety envelope MUST pass an explicitly-derived
            value.
        n_warmup_steps: Length of the high-caution window (default
            :data:`concerto.safety.api.DEFAULT_WARMUP_STEPS` = 50).
    """
    state.lambda_ = make_lambda_dict(uids, fill=lambda_safe)
    state.warmup_steps_remaining = n_warmup_steps


__all__ = [
    "compute_per_pair_loss",
    "compute_prediction_gap_for_pairs",
    "constant_velocity_predict",
    "reset_on_partner_swap",
    "update_lambda",
    "update_lambda_from_predictor",
]
