# SPDX-License-Identifier: Apache-2.0
# pyright: reportPrivateImportUsage=false, reportAttributeAccessIssue=false
"""Tier-2 SAPIEN-gated tests for the co-carry rig (ADR-026 §Decision 1-2; R-2026-06-B Rungs 0-1).

Real ManiSkill v3 env construction + Rung-0/1 coverage that needs a
Vulkan/GPU host. Mirrors :mod:`tests.integration.test_stage1_pickplace_real`'s
skipif pattern; the whole module is skipped on CPU-only runners.

Coverage:

1. **Per-condition construction.** The matched + single-arm conditions
   build a SAPIEN scene with the two-Panda ``agents_dict``.
2. **Telemetry plumbing.** The env exposes the bar tilt, the wrist
   constraint-solver stress proxy, and the centroid-to-goal distance.
3. **Rung-0 stability gate.** The dual-hold attach holds the bar under a
   zero-action hold across seeds — telemetry finite, tilt + stress
   bounded (R-2026-06-B "rigid-joint stability check gates Phase A").
4. **Rung-1 matched competence.** The matched controller pair reaches
   high joint success across seeds (the honest high reference).
5. **Rung-1 coupling positive-control.** A single arm reaches ~= 0 success
   on the same task, and the tilt constraint binds on matched successes
   while a single arm blows past it (ADR-026 §Decision 2 — the
   load-bearing falsifiability result).

These tests use a modest seed count to bound runtime; the
``scripts/repro/cocarry_rung{0,1}_*.sh`` scripts run the larger
seed sweeps for the PR evidence.

ADR-026 §Decision 1-2; ADR-009 §Decision; ADR-005 §Decision; R-2026-06-B Rungs 0-1.
"""

from __future__ import annotations

import numpy as np
import pytest

from chamber.benchmarks.cocarry_runner import (
    build_matched_controllers,
    evaluate_condition,
    rollout_hold,
)
from chamber.envs.cocarry import (
    COCARRY_TILT_MAX_DEG,
    make_cocarry_env,
)
from chamber.utils.device import sapien_gpu_available

pytestmark = [
    pytest.mark.slow,
    pytest.mark.gpu,
    pytest.mark.skipif(
        not sapien_gpu_available(),
        reason="Requires Vulkan/GPU (SAPIEN); skipped on CPU-only machines",
    ),
]

_MATCHED = "cocarry_matched_panda_pair"
_SINGLE = "cocarry_single_arm_positive_control"


class TestConstruction:
    """Both conditions construct a SAPIEN scene with the two-Panda tuple."""

    @pytest.mark.parametrize("condition_id", [_MATCHED, _SINGLE])
    def test_two_pandas_load(self, condition_id: str) -> None:
        env = make_cocarry_env(condition_id=condition_id, episode_length=10)
        try:
            assert set(env.agent.agents_dict.keys()) == {"panda_wristcam", "panda_partner"}
            assert env.single_arm is (condition_id == _SINGLE)
        finally:
            env.close()

    def test_telemetry_keys_present_and_finite(self) -> None:
        env = make_cocarry_env(condition_id=_MATCHED, episode_length=10)
        try:
            env.reset(seed=0)
            tel = env.get_telemetry()
            for key in ("tilt_deg", "stress_proxy", "centroid_to_goal"):
                assert key in tel
                assert np.isfinite(np.asarray(tel[key].detach().cpu()).reshape(-1)[0])
        finally:
            env.close()

    def test_matched_controllers_build(self) -> None:
        controllers = build_matched_controllers()
        assert set(controllers) == {"panda_wristcam", "panda_partner"}


class TestRung0Stability:
    """Rung-0: the dual-hold attach holds the bar (R-2026-06-B stability gate)."""

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_attach_stable_under_hold(self, seed: int) -> None:
        m = rollout_hold(seed=seed, n_steps=60)
        assert m.finite, "telemetry went non-finite — solver blow-up"
        # The bar is held roughly level (no collapse) and the constraint
        # force stays bounded (a generous ceiling; the steady hold is far
        # lower — the bound here only catches a blow-up).
        assert m.max_tilt_deg < 30.0, f"bar not held (tilt {m.max_tilt_deg:.1f} deg)"
        assert m.max_stress_proxy < 2000.0, f"constraint force unbounded ({m.max_stress_proxy:.0f})"


class TestRung1MatchedCompetence:
    """Rung-1: the matched pair is the honest high reference (ADR-026 §Decision 1)."""

    def test_matched_pair_reaches_high_success(self) -> None:
        seeds = list(range(6))
        metrics = evaluate_condition(condition_id=_MATCHED, seeds=seeds, episode_length=320)
        rate = float(np.mean([m.success for m in metrics]))
        assert rate >= 0.8, f"matched success {rate:.0%} below the high-reference bar ({metrics})"
        # Bar held level throughout (the level conjunct is satisfied with
        # margin on matched successes).
        for m in metrics:
            if m.success:
                assert m.max_tilt_deg < COCARRY_TILT_MAX_DEG


class TestRung1CouplingPositiveControl:
    """Rung-1: the load-bearing coupling positive-control (ADR-026 §Decision 2)."""

    def test_single_arm_is_approximately_zero(self) -> None:
        seeds = list(range(6))
        metrics = evaluate_condition(condition_id=_SINGLE, seeds=seeds, episode_length=320)
        rate = float(np.mean([m.success for m in metrics]))
        # A single arm cannot keep the cantilevered bar level -> ~= 0.
        assert rate <= 0.1, f"single-arm success {rate:.0%} — task is NOT two-robot-infeasible"

    def test_tilt_constraint_binds_matched_vs_single(self) -> None:
        """The level constraint is load-bearing: single-arm blows past it, matched holds it.

        On matched successes the bar tilt stays under the limit (the
        constraint is satisfied, not slack at zero — it is approached and
        respected); a single arm drives the tilt far past the limit. The
        large separation is the evidence the coupling constraint actually
        binds (R-2026-06-B; ADR-026 §Decision 2).
        """
        matched = evaluate_condition(
            condition_id=_MATCHED, seeds=list(range(4)), episode_length=320
        )
        single = evaluate_condition(condition_id=_SINGLE, seeds=list(range(4)), episode_length=320)
        matched_tilt = float(np.median([m.max_tilt_deg for m in matched]))
        single_tilt = float(np.median([m.max_tilt_deg for m in single]))
        # Matched holds the bar level (well under the limit); single-arm
        # drives it far past — a wide, decisive separation.
        assert matched_tilt < COCARRY_TILT_MAX_DEG
        assert single_tilt > 2.0 * COCARRY_TILT_MAX_DEG
