# SPDX-License-Identifier: Apache-2.0
"""Reproduction of Huriot & Sibai 2025 Table I gt/Learn row (T3.6).

ADR-004 §Decision (conformal slack overlay); plan/03 §4 T3.6 + §5
"Reproduction". Marked ``@pytest.mark.slow`` per plan/03: skipped on
PR CI, run nightly + before release.

This test validates the Theorem 3 average-loss bound

    1/K' Σ l_k ≤ ε + (λ_1 - λ_safe + η) / (η · K')

(Huriot & Sibai 2025 §IV.D; ICRA 2025 arXiv:2409.18862v4) by
simulating a closed-loop conformal update against a stationary
prediction-vs-truth gap distribution, and asserting the empirical
average loss tracks ε within ±5%.

The Stanford Drone Dataset reproduction (the paper's actual §V toy
robot navigating recorded pedestrian trajectories) is a Phase-1
cross-check requiring the SDD trajectory dataset and the darts
40-frame predictor pipeline. This synthetic Theorem-3 test validates
the update rule's mathematical correctness independently — note 42's
gt/Learn row is the configuration with ground-truth predictions and
λ updated online (η > 0), and Theorem 3's bound is what that row
empirically achieves.

T3.6 is a TDD entry point: this file lands first as a failing test
(``concerto.safety.conformal.update_lambda`` does not yet exist), then
PR6's implementation makes it pass.
"""

from __future__ import annotations

import numpy as np
import pytest

from concerto.safety.api import SafetyState

#: Simulation horizon. Theorem 3's bound term ``(λ_1 - λ_safe + η)/(η·K)``
#: vanishes as ``K → ∞``; ``20_000`` steps gives ``< 5%`` headroom around
#: ``ε`` for any reasonable ``λ_1`` initialisation.
_K_STEPS: int = 20_000

#: Target average loss (Huriot & Sibai 2025 §IV.A). Positive value tests
#: the natural Theorem 3 convergence regime; the conservative-manipulation
#: regime ``ε < 0`` (plan/03 §2 default) drives ``λ`` monotonically toward
#: tighter constraints and the bound becomes one-sided — see
#: ``test_conformal.py`` for that regime.
_EPSILON: float = 0.05

#: Conformal learning rate.
_ETA: float = 0.05

#: Standard deviation of the synthetic prediction-vs-truth gap stream.
#: Picked so the Theorem 3 equilibrium ``λ`` lands at a non-degenerate
#: negative value (analytic equilibrium ``φ(λ) = ε`` ⇒ ``λ ≈ -0.20``).
_GAP_STD: float = 0.3

#: Tolerance: the empirical average loss must be within 5% of ε.
_REL_TOL: float = 0.05


@pytest.mark.slow
def test_huriot_sibai_2025_table_i_gt_learn_average_loss_within_5_percent() -> None:
    """Reproduce Huriot & Sibai 2025 Table I gt/Learn row (Theorem 3, ±5%).

    Synthetic Theorem 3 reproduction — closed-loop conformal update
    against a stationary prediction-vs-truth gap distribution. Asserts
    the empirical average loss is within 5% of ε.
    """
    rng = np.random.default_rng(0)

    state = SafetyState(
        lambda_=np.zeros(1, dtype=np.float64),
        epsilon=_EPSILON,
        eta=_ETA,
    )

    cumulative_loss = 0.0
    for _ in range(_K_STEPS):
        # e_k models the GT-vs-conformal slack residual at the
        # constraint boundary (Huriot & Sibai §IV.A): when ``λ`` exceeds
        # the residual, the conformal CBF overestimates the safe set
        # and ``l_k > 0`` (the loss the rule is designed to control).
        e_k = float(rng.normal(loc=0.0, scale=_GAP_STD))
        l_k_val = max(0.0, float(state.lambda_[0]) - e_k)
        cumulative_loss += l_k_val

        from concerto.safety.conformal import update_lambda

        update_lambda(state, np.array([l_k_val], dtype=np.float64), in_warmup=False)

    avg_loss = cumulative_loss / _K_STEPS
    rel_err = abs(avg_loss - _EPSILON) / _EPSILON
    assert rel_err < _REL_TOL, (
        f"Huriot-Sibai Table I gt/Learn reproduction: avg_loss "
        f"{avg_loss:.5f} differs from epsilon {_EPSILON} by "
        f"{rel_err * 100:.1f}% (>{_REL_TOL * 100:.0f}%)"
    )
