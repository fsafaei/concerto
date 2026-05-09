# SPDX-License-Identifier: Apache-2.0
"""Integration test: full M3 safety stack in a synthetic multi-agent loop (T3.11).

Plan/03 §4 T3.11 + §5 ``test_safety_in_loop.py``: in a Stage-0-like
synthetic env, run 100 steps with the full filter stack (exp CBF +
conformal lambda + OSCBF + braking) and assert no crashes, no
infeasibility, no per-step time > 1 ms (target; under coverage
instrumentation we relax to a generous bound).

The Stage-0 SAPIEN/ManiSkill smoke env is GPU-skipped on CPU CI; this
synthetic loop exercises the same composition pattern with double-
integrator multi-agent dynamics + a synthetic 7-DOF arm Jacobian, so
the filter stack's wiring is verified end-to-end on every CPU run.
"""

from __future__ import annotations

import sys
import time

import numpy as np
import pytest

from concerto.safety.api import Bounds, SafetyState
from concerto.safety.braking import maybe_brake
from concerto.safety.cbf_qp import AgentSnapshot, ExpCBFQP
from concerto.safety.conformal import update_lambda
from concerto.safety.errors import ConcertoSafetyInfeasible
from concerto.safety.oscbf import (
    OSCBF,
    OSCBFConstraints,
    joint_limit_constraint_row,
)
from concerto.safety.reporting import (
    AssumptionRow,
    ConditionRow,
    GapRow,
    ThreeTableReport,
)


def _bounds(action_norm: float = 5.0) -> Bounds:
    return Bounds(
        action_norm=action_norm,
        action_rate=0.5,
        comm_latency_ms=1.0,
        force_limit=20.0,
    )


def _build_oscbf_arm_constraints(
    n_joints: int, upper: float = 2.0, lower: float = -2.0
) -> OSCBFConstraints:
    rows: list[np.ndarray] = []
    rhs: list[float] = []
    for j in range(n_joints):
        a, b = joint_limit_constraint_row(n_joints=n_joints, joint_index=j, upper=upper)
        rows.append(a)
        rhs.append(b)
        a, b = joint_limit_constraint_row(n_joints=n_joints, joint_index=j, lower=lower)
        rows.append(a)
        rhs.append(b)
    return OSCBFConstraints(a=np.vstack(rows), b=np.asarray(rhs, dtype=np.float64))


def test_full_safety_stack_runs_100_steps_without_crash() -> None:  # noqa: PLR0915
    # PLR0915: integration test for the full M3 stack inherently exercises
    # many call sites (braking + outer CBF + conformal + OSCBF) over a
    # 100-step rollout; statement count is the price of end-to-end coverage.
    """T3.11: full filter stack runs 100 steps; no crash, no infeasibility, bounded per-step time.

    Composition pattern (matches plan/03 §1):

    1. Per-step braking-fallback check (kinematic; bypasses the QP per
       ADR-004 risk-mitigation #1).
    2. Outer exp CBF-QP if the fallback didn't fire (Wang-Ames-Egerstedt
       2017 backbone).
    3. Conformal lambda update consuming the per-pair loss returned in
       :class:`FilterInfo` (Huriot & Sibai 2025 Theorem 3).
    4. OSCBF inner filter on a synthetic 7-DOF arm command (Morton &
       Pavone 2025 §IV).

    The 1 ms ADR-004 §"OSCBF target" timing is asserted for the median
    step under coverage; the strict per-step < 1 ms uncovered bound is
    asserted in the dedicated PR8 timing test
    (``test_oscbf_mean_solve_time_under_1ms_on_7dof_arm``).
    """
    dt = 0.05
    n_steps = 100
    radius = 0.2
    safety_distance = 2 * radius
    rng = np.random.default_rng(0)

    p_a = np.array([-2.0, 0.0], dtype=np.float64)
    p_b = np.array([2.0, 0.0], dtype=np.float64)
    v_a = np.array([1.0, 0.0], dtype=np.float64)
    v_b = np.array([-1.0, 0.0], dtype=np.float64)

    cbf = ExpCBFQP(cbf_gamma=2.0)
    bounds = _bounds(action_norm=5.0)
    state = SafetyState(
        lambda_=np.zeros(1, dtype=np.float64),
        epsilon=-0.05,
        eta=0.01,
    )

    n_joints = 7
    oscbf = OSCBF(n_joints=n_joints)
    j_arm = (rng.standard_normal((6, n_joints)) * 0.3).astype(np.float64)
    arm_constraints = _build_oscbf_arm_constraints(n_joints)

    min_distance = float("inf")
    step_wall_ms: list[float] = []
    fallback_fires = 0
    qp_calls = 0

    for k in range(n_steps):
        snaps = {
            "a": AgentSnapshot(position=p_a.copy(), velocity=v_a.copy(), radius=radius),
            "b": AgentSnapshot(position=p_b.copy(), velocity=v_b.copy(), radius=radius),
        }
        proposed = {
            "a": np.zeros(2, dtype=np.float64),
            "b": np.zeros(2, dtype=np.float64),
        }
        obs = {"agent_states": snaps, "meta": {"partner_id": "fixed"}}

        step_start = time.perf_counter()

        # 1. Braking fallback (per-step backstop; ADR-004 risk-mitigation #1).
        override, fired = maybe_brake(proposed, snaps, bounds=bounds)
        safe: dict[str, np.ndarray]
        if fired and override is not None:
            fallback_fires += 1
            safe = override
        else:
            # 2. Outer CBF-QP (Wang-Ames-Egerstedt 2017).
            try:
                safe, info = cbf.filter(
                    proposed_action=proposed,
                    obs=obs,
                    state=state,
                    bounds=bounds,
                )
                qp_calls += 1
                # 3. Conformal lambda update (Huriot & Sibai 2025 §IV).
                update_lambda(state, info["loss_k"], in_warmup=False)
            except ConcertoSafetyInfeasible as exc:
                pytest.fail(f"step {k}: outer CBF-QP infeasible: {exc}")

        # 4. OSCBF inner filter on a synthetic arm command (Morton-Pavone §IV).
        q_dot_nom = (rng.standard_normal(n_joints) * 0.2).astype(np.float64)
        try:
            _ = oscbf.solve(
                q_dot_nom=q_dot_nom,
                nu_nom=np.zeros(6, dtype=np.float64),
                jacobian=j_arm,
                constraints=arm_constraints,
            )
        except ConcertoSafetyInfeasible as exc:
            pytest.fail(f"step {k}: OSCBF infeasible: {exc}")

        step_wall_ms.append((time.perf_counter() - step_start) * 1000.0)

        v_a += safe["a"] * dt
        v_b += safe["b"] * dt
        p_a += v_a * dt
        p_b += v_b * dt
        min_distance = min(min_distance, float(np.linalg.norm(p_a - p_b)))

    # No crash, no infeasibility (caught above with pytest.fail).
    assert len(step_wall_ms) == n_steps

    # ADR-004 §"OSCBF target" 1 ms: enforced at median, with a generous
    # uncov-vs-cov envelope so we don't false-fail under the coverage
    # tracer's 2-3x inflation. The dedicated 1 ms uncovered assertion
    # lives in tests/property/test_oscbf.py per PR8.
    median_ms = float(np.median(step_wall_ms))
    p95_ms = float(np.percentile(step_wall_ms, 95))
    cov_envelope_ms = 5.0 if sys.gettrace() is not None else 1.5
    assert median_ms < cov_envelope_ms, (
        f"median per-step wall time {median_ms:.3f} ms exceeds "
        f"{cov_envelope_ms:.1f} ms (p95={p95_ms:.3f} ms)"
    )

    # Sanity: collision-avoidance invariant from PR5 holds end-to-end.
    assert min_distance > safety_distance, (
        f"head-on agents collided: min distance {min_distance:.4f} <= D_s {safety_distance:.4f}"
    )

    # End-to-end emission: the FilterInfo telemetry feeds the ADR-014
    # three-table renderer. Build a smoke report from the loop's
    # counters and verify it round-trips.
    report = ThreeTableReport(
        table_1=(
            AssumptionRow(
                assumption="A1",
                description="bounded gradient norm",
                violations=0,
                n_steps=n_steps,
            ),
            AssumptionRow(
                assumption="A2",
                description="bounded prediction error",
                violations=0,
                n_steps=n_steps,
            ),
            AssumptionRow(
                assumption="A3",
                description="QP feasibility with lambda",
                violations=0,
                n_steps=n_steps,
            ),
        ),
        table_2=(
            ConditionRow(
                predictor="gt",
                conformal_mode="Learn",
                vendor_compliance=None,
                n_episodes=1,
                violations=0,
                fallback_fires=fallback_fires,
            ),
        ),
        table_3=(
            GapRow(
                condition="gt/Learn",
                lambda_mean=float(np.mean(state.lambda_)),
                lambda_var=float(np.var(state.lambda_)),
                oracle_lambda_mean=0.0,
            ),
        ),
    )
    parsed = ThreeTableReport.from_jsonable(report.to_jsonable())
    assert parsed == report
