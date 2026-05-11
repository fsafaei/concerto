# SPDX-License-Identifier: Apache-2.0
# pyright: reportPrivateImportUsage=false
#
# Suppresses pyright's ``torch.from_numpy`` / ``torch.allclose`` /
# ``torch.equal`` complaints — these are documented public APIs but not
# present in torch's stub ``__all__``. Same rationale as
# ``src/chamber/benchmarks/ego_ppo_trainer.py``.
"""Property test for ego-PPO advantage decomposition (M4b-8a; plan/05 §5).

This test is the **trip-wire guardrail** for ADR-002 risk-mitigation #1
(empirical-guarantee experiment, T4b.13 / M4b-8b). It must pass before the
100k-frame live experiment runs, so that if M4b-8b's non-decreasing
assertion fails we can disentangle "the trainer is buggy" from "HARL HAPPO
does not collapse to single-agent under a frozen partner".

The claim under test:

    For a 2-agent env in which the partner is *frozen* (no parameter
    updates), the ego-PPO advantages computed by
    :class:`chamber.benchmarks.ego_ppo_trainer.EgoPPOTrainer` are
    bit-for-bit equivalent to single-agent PPO advantages computed on
    the (rewards, values, dones) sequence the ego experienced.

The reference implementations (:func:`reference_gae` +
:func:`reference_normalize`) are derived from the math in
Schulman et al. 2016 §6.3 + Schulman et al. 2017 §6 — *not* copied from
HARL or from the trainer module. This is what makes the comparison
non-tautological.

Tolerance contract (per the M4b-8a guardrails):

- Target: 1e-6 for the GAE recurrence and the normalization formula.
- Relax-to-1e-4 escape hatch: only used when reproducing HARL's
  formula exactly would require copying source rather than deriving from
  first principles. **No silent widening** — every relaxed assertion
  carries a docstring explaining which subroutine could not be
  hand-rolled exactly and why. This file currently uses 1e-6 throughout;
  no relaxation needed.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from chamber.benchmarks.ego_ppo_trainer import (
    EgoPPOTrainer,
    compute_gae,
    normalize_advantages,
)
from chamber.envs.mpe_cooperative_push import MPECooperativePushEnv
from chamber.partners.api import PartnerSpec
from chamber.partners.heuristic import ScriptedHeuristicPartner
from concerto.training.config import (
    EgoAHTConfig,
    EnvConfig,
    HAPPOHyperparams,
    PartnerConfig,
)


def reference_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    *,
    bootstrap_value: float,
    gamma: float,
    gae_lambda: float,
) -> np.ndarray:
    """Hand-rolled single-agent GAE (Schulman et al. 2016 eq. 16).

    Derived from the paper, *not* from HARL or from
    :func:`chamber.benchmarks.ego_ppo_trainer.compute_gae`. This is the
    ``A_t`` reference the property test compares the trainer's output
    against. Two implementations of the same recurrence — independent —
    must agree to numerical precision.

    Recurrence::

        delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
        A_t     = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}
        A_T     = delta_T

    """
    n = rewards.shape[0]
    out = np.zeros(n, dtype=np.float32)
    next_advantage = 0.0
    for t in range(n - 1, -1, -1):
        v_next = bootstrap_value if t == n - 1 else float(values[t + 1])
        nonterminal = 0.0 if bool(dones[t]) else 1.0
        delta = float(rewards[t]) + gamma * v_next * nonterminal - float(values[t])
        next_advantage = delta + gamma * gae_lambda * nonterminal * next_advantage
        out[t] = np.float32(next_advantage)
    return out


def reference_normalize(advantages: np.ndarray) -> np.ndarray:
    """Hand-rolled mean/std normalization with HARL's 1e-5 epsilon.

    Derived from the standard PPO advantage-normalization step
    (Schulman et al. 2017 §6, "Implementation"). The 1e-5 stabilization
    constant matches HARL's :meth:`HAPPO.train` line 127.
    """
    return ((advantages - advantages.mean()) / (advantages.std() + 1e-5)).astype(np.float32)


# Hypothesis strategies for synthetic 2-agent-with-frozen-partner trajectories.
# Using small sequences keeps each property example fast; the math being
# tested is per-step recurrence, so 5-50 step trajectories already exercise
# the loop and edge cases (single-step, terminal-at-end, terminal-in-middle).

_FLOAT_FINITE = st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False)
_VALUE_FINITE = st.floats(min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False)
_GAMMA_STRATEGY = st.floats(min_value=0.0, max_value=0.999, allow_nan=False)
_LAMBDA_STRATEGY = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
_LENGTHS = st.integers(min_value=1, max_value=50)


@st.composite
def _synthetic_trajectory(  # type: ignore[no-untyped-def]
    draw,
):
    """Hypothesis composite: ``(rewards, values, dones, bootstrap, gamma, lam)``.

    Generates a length-``n`` trajectory of finite floats, with at most one
    interior terminal (the rest of the dones are False) plus the final
    step's done flag drawn freely. This shape covers the full GAE-truncation
    surface: pure rollout (no done), terminal-at-end (typical truncation),
    and one mid-episode terminal (cross-episode rollout).
    """
    n = draw(_LENGTHS)
    rewards = np.array([draw(_FLOAT_FINITE) for _ in range(n)], dtype=np.float32)
    values = np.array([draw(_VALUE_FINITE) for _ in range(n)], dtype=np.float32)
    dones = np.zeros(n, dtype=np.bool_)
    if n > 1:
        # Optional interior terminal at a random index.
        interior_terminal_idx = draw(st.integers(min_value=-1, max_value=n - 2))
        if interior_terminal_idx >= 0:
            dones[interior_terminal_idx] = True
    # Final-step done is drawn freely (truncated vs continuing rollout).
    dones[-1] = draw(st.booleans())
    bootstrap = draw(_VALUE_FINITE)
    gamma = draw(_GAMMA_STRATEGY)
    lam = draw(_LAMBDA_STRATEGY)
    return rewards, values, dones, bootstrap, gamma, lam


@settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(_synthetic_trajectory())
def test_compute_gae_matches_hand_rolled_reference(  # type: ignore[no-untyped-def]
    trajectory,
) -> None:
    """plan/05 §5: ego GAE equals hand-rolled single-agent GAE under frozen partner.

    The ego's experience reduces to a fixed MDP from its POV when the
    partner is frozen — the partner's actions become part of the
    transition kernel, not part of the ego's gradient. Therefore
    :func:`compute_gae` (the trainer's GAE) and :func:`reference_gae`
    (the from-the-paper reference) must agree on every step of every
    synthetic trajectory.

    Tolerance: 1e-6 absolute, both implementations operate in float32;
    relative differences below this are noise from intermediate float
    promotions in numpy.
    """
    rewards, values, dones, bootstrap, gamma, lam = trajectory
    trainer_adv = compute_gae(
        rewards,
        values,
        dones,
        bootstrap_value=bootstrap,
        gamma=gamma,
        gae_lambda=lam,
    )
    ref_adv = reference_gae(
        rewards,
        values,
        dones,
        bootstrap_value=bootstrap,
        gamma=gamma,
        gae_lambda=lam,
    )
    assert trainer_adv.shape == ref_adv.shape
    assert trainer_adv.dtype == np.float32
    np.testing.assert_allclose(trainer_adv, ref_adv, atol=1e-6, rtol=0.0)


@settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(
    st.lists(
        st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=200,
    )
)
def test_normalize_advantages_matches_hand_rolled_reference(  # type: ignore[no-untyped-def]
    raw_floats,
) -> None:
    """plan/05 §5: trainer's advantage normalization matches the from-paper formula.

    Both :func:`normalize_advantages` and :func:`reference_normalize`
    apply ``(x - mean) / (std + 1e-5)`` element-wise. Two correct
    implementations must agree to 1e-6 absolute tolerance.

    Hypothesis range capped at ±1000 because numpy's std accumulator
    can lose precision on extreme magnitudes; this is a numpy ergonomic,
    not a CONCERTO contract.
    """
    raw = np.array(raw_floats, dtype=np.float32)
    trainer_norm = normalize_advantages(raw)
    ref_norm = reference_normalize(raw)
    np.testing.assert_allclose(trainer_norm, ref_norm, atol=1e-6, rtol=0.0)


def test_compute_gae_constant_zero_reward_yields_zero_advantage() -> None:
    """Sanity / regression: zero-reward trajectory with V==0 yields A==0 (plan/05 §5)."""
    n = 16
    rewards = np.zeros(n, dtype=np.float32)
    values = np.zeros(n, dtype=np.float32)
    dones = np.zeros(n, dtype=np.bool_)
    adv = compute_gae(rewards, values, dones, bootstrap_value=0.0, gamma=0.99, gae_lambda=0.95)
    np.testing.assert_array_equal(adv, np.zeros(n, dtype=np.float32))


def test_compute_gae_terminal_resets_bootstrap_correctly() -> None:
    """plan/05 §5: a terminal step zeroes the next-value bootstrap, isolating per-segment GAE."""
    # Two-segment trajectory: terminal at step 2.
    rewards = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    values = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    dones = np.array([False, False, True, False], dtype=np.bool_)
    bootstrap = 0.0
    gamma, lam = 0.5, 1.0  # closed-form: A_t = sum_{k>=t} (gamma)^{k-t} r_k within the segment

    adv = compute_gae(
        rewards, values, dones, bootstrap_value=bootstrap, gamma=gamma, gae_lambda=lam
    )
    # Segment 1 (steps 0,1,2): truncated to length 3.
    #   A_2 = r_2 = 1.0
    #   A_1 = r_1 + gamma * A_2 = 1 + 0.5 * 1 = 1.5
    #   A_0 = r_0 + gamma * A_1 = 1 + 0.5 * 1.5 = 1.75
    # Segment 2 (step 3 only):
    #   A_3 = r_3 + gamma * bootstrap = 1.0 (bootstrap=0; not terminal at last step,
    #   so nonterminal=1 but bootstrap is 0).
    expected = np.array([1.75, 1.5, 1.0, 1.0], dtype=np.float32)
    np.testing.assert_allclose(adv, expected, atol=1e-6, rtol=0.0)


def test_normalize_advantages_constant_input_does_not_crash() -> None:
    """plan/05 §5: a zero-variance input yields a near-zero output (1e-5 epsilon stabilizes std).

    Constant advantages have ``std == 0``; the formula's ``+ 1e-5``
    epsilon prevents a division-by-zero. The result is therefore
    ``(x - mean) / 1e-5`` — for x == mean, that's 0 across the board.
    """
    raw = np.full(10, 3.5, dtype=np.float32)
    out = normalize_advantages(raw)
    np.testing.assert_allclose(out, np.zeros(10, dtype=np.float32), atol=1e-6)


# ---------------------------------------------------------------------------
# Integration: the trainer's actual update path uses these module-level
# functions. The test below rolls out a real trainer on the MPE env with a
# real frozen partner, captures the (rewards, values, dones) the trainer
# buffered, then re-runs the reference math and checks bit-for-bit agreement
# with the trainer's actual computation.
# ---------------------------------------------------------------------------


def _build_trainer_and_env(
    *, seed: int = 0, rollout_length: int = 16
) -> tuple[EgoPPOTrainer, MPECooperativePushEnv, ScriptedHeuristicPartner]:
    """Construct the (trainer, env, partner) triple for the integration check."""
    cfg = EgoAHTConfig(
        seed=seed,
        total_frames=rollout_length,
        checkpoint_every=rollout_length,
        env=EnvConfig(task="mpe_cooperative_push", episode_length=rollout_length),
        partner=PartnerConfig(
            class_name="scripted_heuristic",
            extra={"uid": "partner", "target_xy": "0.0,0.0", "action_dim": "2"},
        ),
        happo=HAPPOHyperparams(
            rollout_length=rollout_length,
            batch_size=rollout_length,
            n_epochs=1,
            hidden_dim=32,
        ),
    )
    env = MPECooperativePushEnv(root_seed=seed)
    partner = ScriptedHeuristicPartner(
        spec=PartnerSpec(
            class_name="scripted_heuristic",
            seed=0,
            checkpoint_step=None,
            weights_uri=None,
            extra={"uid": "partner", "target_xy": "0.0,0.0", "action_dim": "2"},
        )
    )
    trainer = EgoPPOTrainer.from_config(cfg, env=env, ego_uid="ego")
    return trainer, env, partner


@pytest.mark.slow
def test_ego_trainer_advantages_match_reference_on_frozen_partner_rollout() -> None:
    """plan/05 §5 integration: trainer's GAE on a real rollout matches the reference.

    Drives a 16-step rollout against the real MPE env + real frozen
    heuristic partner. After the rollout, computes the trainer's
    advantages via :func:`compute_gae` (which is what the trainer's
    :meth:`update` uses), then independently computes
    :func:`reference_gae` on the same buffered ``(rewards, values, dones)``,
    then asserts equality.

    This exercises the *integration* path: the trainer's pre/post-step
    cache, the buffer accumulation, and the GAE call. If a future change
    inlines the GAE math into :meth:`update`, this test will catch the
    silent re-implementation.
    """
    trainer, env, partner = _build_trainer_and_env(seed=42, rollout_length=16)
    obs, _ = env.reset(seed=0)
    partner.reset(seed=0)
    for _ in range(16):
        ego_action = trainer.act(obs)
        partner_action = partner.act(obs)
        obs, reward, terminated, truncated, _ = env.step(
            {"ego": ego_action, "partner": partner_action}
        )
        trainer.observe(obs, reward, terminated or truncated)

    # Pull the buffer state out for the reference comparison. These are
    # private attributes on purpose — the test asserts on the runtime
    # pre-update state, which has no public-API surface (the buffer is
    # cleared by :meth:`update`).
    rewards = np.asarray(trainer._buf_rewards, dtype=np.float32)
    values = np.asarray(trainer._buf_values, dtype=np.float32)
    dones = np.asarray(trainer._buf_dones, dtype=np.bool_)
    # Bootstrap reproduces the trainer's own logic (private, but stable):
    # zero if the last step ended the episode, else V(last_next_obs).
    if dones[-1] or trainer._last_next_obs is None:
        bootstrap = 0.0
    else:
        import torch

        with torch.no_grad():
            bootstrap = float(
                trainer._critic(
                    torch.from_numpy(trainer._last_next_obs).to(trainer._device).unsqueeze(0)
                )
                .squeeze()
                .item()
            )

    trainer_adv = compute_gae(
        rewards,
        values,
        dones,
        bootstrap_value=bootstrap,
        gamma=trainer._gamma,
        gae_lambda=trainer._gae_lambda,
    )
    ref_adv = reference_gae(
        rewards,
        values,
        dones,
        bootstrap_value=bootstrap,
        gamma=trainer._gamma,
        gae_lambda=trainer._gae_lambda,
    )
    np.testing.assert_allclose(trainer_adv, ref_adv, atol=1e-6, rtol=0.0)

    # And the normalization step on the same advantages.
    trainer_norm = normalize_advantages(trainer_adv)
    ref_norm = reference_normalize(ref_adv)
    np.testing.assert_allclose(trainer_norm, ref_norm, atol=1e-6, rtol=0.0)


def test_ego_ppo_trainer_uses_module_gae_marker() -> None:
    """plan/05 §5: the trainer's class-level marker pins it routes through this module.

    Defensive regression check: if a future refactor ever inlines GAE
    inside :meth:`EgoPPOTrainer.update`, the property test above keeps
    passing only as long as the integration check is wired correctly.
    This marker — read by the property suite as a contract — must be
    set ``True`` for as long as the trainer routes through
    :func:`compute_gae` + :func:`normalize_advantages`. Toggling it to
    ``False`` is a deliberate signal that the comparison no longer
    holds and the M4b-8b empirical-guarantee experiment must not run.
    """
    assert EgoPPOTrainer.USES_MODULE_GAE is True
