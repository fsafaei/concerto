# SPDX-License-Identifier: Apache-2.0
# pyright: reportPrivateImportUsage=false, reportAttributeAccessIssue=false
"""Tier-1 (no-SAPIEN-scene) tests for the Rung-3 PH measurement (ADR-026 §Decision 4; ADR-009).

Covers the pure-Python surface that needs no SAPIEN scene:

- :mod:`chamber.partners.cocarry_policy_shift` — the four policy-shift
  teammates are registered, parse their spec, and their ``act`` returns an
  8-D action that is finite, bounded, and deterministic across reloads. Each
  teammate's distinguishing policy hook is exercised (stiffness gains, the
  admittance bar-following branch, the slew ramp over steps, the null-space
  joint step), plus the defensive hold on incomplete obs and the geometry
  helpers. The Jacobian/FK provider parses the panda URDF via
  pytorch_kinematics without SAPIEN — Tier-1-safe (mirrors
  ``test_cocarry_tier1.py::TestImpedanceControllerTier1``).
- :mod:`chamber.benchmarks.cocarry_ph` — the pure statistics the
  pre-registration locks: the capability gate, the paired Δ, the cluster
  bootstrap one-sided CI (deterministic under P6), the IQM, the Δ pooling,
  the pre-committed decision + null rule, and the cluster-robust binomial
  confirmatory — all on synthetic ``{seed: bool}`` outcomes.

Tier-2 SAPIEN/CUDA-gated coverage (the real shifted-eval pipeline) is in
``tests/integration/test_cocarry_rung3_ph_real.py``.

ADR-026 §Decision 4; ADR-009 §Decision; R-2026-06-B §15 Rung 3.
"""

from __future__ import annotations

import numpy as np
import pytest

from chamber.benchmarks import cocarry_ph as ph
from chamber.partners.cocarry_policy_shift import (
    COCARRY_POLICY_SHIFT_CANDIDATES,
    _parse_vec3,
    _quat_wxyz_to_matrix,
)

# --------------------------------------------------------------------------
# Policy-shift teammates (chamber.partners.cocarry_policy_shift).
# --------------------------------------------------------------------------

_IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0])  # wxyz, level bar
_READY_Q7 = np.array([0.0, np.pi / 8, 0.0, -np.pi * 5 / 8, 0.0, np.pi * 3 / 4, -np.pi / 4])


def _partner(class_name: str):
    """Build a policy-shift teammate on the partner seat (the env geometry)."""
    from chamber.partners.api import PartnerSpec
    from chamber.partners.registry import load_partner

    extra = {
        "uid": "panda_partner",
        "base_xyz": "0.5,0,0",
        "base_yaw_deg": "180",
        "end_sign": "-1",
        "bar_half_len": "0.115",
    }
    return load_partner(PartnerSpec(class_name, 0, None, None, extra))


def _obs(q7: np.ndarray, goal: np.ndarray, bar: np.ndarray | None = None) -> dict:
    """Synthetic partner-seat obs: panda_partner qpos + task leaves."""
    extra: dict = {"goal_pos": goal.astype(np.float32)}
    if bar is not None:
        extra["bar_pose"] = bar.astype(np.float32)
    return {
        "agent": {"panda_partner": {"qpos": np.concatenate([q7, [0.04, 0.04]]).astype(np.float32)}},
        "extra": extra,
    }


class TestPolicyShiftRegistration:
    """All four teammates register under their canonical class names (ADR-026 §D4)."""

    def test_candidate_tuple_has_four_distinct(self) -> None:
        assert len(COCARRY_POLICY_SHIFT_CANDIDATES) == 4
        assert len(set(COCARRY_POLICY_SHIFT_CANDIDATES)) == 4

    def test_all_registered(self) -> None:
        from chamber.partners.registry import list_registered

        registered = list_registered()
        for cls in COCARRY_POLICY_SHIFT_CANDIDATES:
            assert cls in registered


class TestPolicyShiftActContract:
    """Every teammate returns a finite, bounded, deterministic 8-D action (ADR-009)."""

    @pytest.mark.parametrize("cls", COCARRY_POLICY_SHIFT_CANDIDATES)
    def test_act_eight_dim_open_gripper_bounded(self, cls: str) -> None:
        ctrl = _partner(cls)
        ctrl.reset(seed=0)
        a = ctrl.act(_obs(_READY_Q7, np.array([0.0, 0.12, 0.28]), bar=_make_bar()))
        assert a.shape == (8,)
        assert a.dtype == np.float32
        assert a[7] == pytest.approx(1.0)  # gripper open
        assert np.all(np.abs(a) <= 1.0 + 1e-6)
        assert np.all(np.isfinite(a))

    @pytest.mark.parametrize("cls", COCARRY_POLICY_SHIFT_CANDIDATES)
    def test_act_deterministic_across_reloads(self, cls: str) -> None:
        obs = _obs(_READY_Q7, np.array([0.0, 0.12, 0.28]), bar=_make_bar())
        a = _partner(cls)
        a.reset(seed=0)
        b = _partner(cls)
        b.reset(seed=0)
        np.testing.assert_array_equal(a.act(obs), b.act(obs))

    @pytest.mark.parametrize("cls", COCARRY_POLICY_SHIFT_CANDIDATES)
    def test_incomplete_obs_holds(self, cls: str) -> None:
        ctrl = _partner(cls)
        ctrl.reset(seed=0)
        a = ctrl.act({"agent": {}, "extra": {}})
        assert a.shape == (8,)
        assert a[7] == pytest.approx(1.0)
        np.testing.assert_array_equal(a[:7], np.zeros(7, dtype=np.float32))

    @pytest.mark.parametrize("cls", COCARRY_POLICY_SHIFT_CANDIDATES)
    def test_missing_goal_holds(self, cls: str) -> None:
        ctrl = _partner(cls)
        ctrl.reset(seed=0)
        obs = {
            "agent": {"panda_partner": {"qpos": np.zeros(9, dtype=np.float32)}},
            "extra": {},
        }
        np.testing.assert_array_equal(ctrl.act(obs)[:7], np.zeros(7, dtype=np.float32))

    @pytest.mark.parametrize("cls", COCARRY_POLICY_SHIFT_CANDIDATES)
    def test_short_qpos_holds(self, cls: str) -> None:
        ctrl = _partner(cls)
        ctrl.reset(seed=0)
        obs = {
            "agent": {"panda_partner": {"qpos": np.zeros(5, dtype=np.float32)}},
            "extra": {"goal_pos": np.zeros(3, dtype=np.float32)},
        }
        np.testing.assert_array_equal(ctrl.act(obs)[:7], np.zeros(7, dtype=np.float32))


class TestStiffImpedanceDistinct:
    """The stiff teammate drives harder than the matched-gain default (ADR-026 §D4)."""

    def test_higher_gain_yields_larger_command(self) -> None:
        stiff = _partner("cocarry_stiff_impedance")
        slew = _partner("cocarry_slew_impedance")  # matched gains, no slew at far error
        stiff.reset(seed=0)
        slew.reset(seed=0)
        # A goal far from the start: the stiff (high-gain, large clip) command
        # saturates to a larger joint step than the matched-gain controller.
        goal = np.array([0.0, 0.3, 0.5])
        a_stiff = stiff.act(_obs(_READY_Q7, goal, bar=_make_bar()))
        a_slew = slew.act(_obs(_READY_Q7, goal, bar=_make_bar()))
        assert np.linalg.norm(a_stiff[:7]) > np.linalg.norm(a_slew[:7])


class TestAdmittanceFollower:
    """The admittance follower reads the bar pose and yields toward it (ADR-026 §D4; ADR-009)."""

    def test_uses_bar_pose_branch(self) -> None:
        ctrl = _partner("cocarry_admittance")
        ctrl.reset(seed=0)
        # With and without the bar pose the follower computes a different
        # target (with bar => aims partway from the held bar end), so the
        # actions differ — proving the bar-following branch is live.
        goal = np.array([0.0, 0.12, 0.28])
        bar = _make_bar(center=np.array([0.05, 0.0, 0.2]))
        a_with = ctrl.act(_obs(_READY_Q7, goal, bar=bar))
        ctrl.reset(seed=0)
        a_without = ctrl.act(_obs(_READY_Q7, goal, bar=None))
        assert not np.array_equal(a_with, a_without)


class TestSlewTiming:
    """The slew teammate ramps its target over the lead-in (ADR-026 §D4)."""

    def test_target_fraction_grows_with_step(self) -> None:
        from chamber.partners.cocarry_policy_shift import _SLEW_LEAD_IN_STEPS

        ctrl = _partner("cocarry_slew_impedance")
        ctrl.reset(seed=0)
        goal = np.array([0.0, 0.3, 0.5])
        bar = _make_bar()
        obs = _obs(_READY_Q7, goal, bar=bar)
        # Internal: the ramp fraction is (step+1)/lead_in. Drive many steps and
        # confirm the commanded joint step grows then the controller keeps
        # acting (no crash, bounded) across the whole lead-in.
        out = ctrl.act(obs)
        first = np.linalg.norm(out[:7])
        for _ in range(_SLEW_LEAD_IN_STEPS):
            out = ctrl.act(obs)
            assert np.all(np.isfinite(out))
            assert np.all(np.abs(out) <= 1.0 + 1e-6)
        last = np.linalg.norm(out[:7])
        # The full-target command (late) exceeds the heavily-attenuated start.
        assert last >= first


class TestNullspaceResolution:
    """The null-space teammate adds a posture secondary task (ADR-026 §D4)."""

    def test_nullspace_changes_joint_step_vs_plain_pinv(self) -> None:
        # The null-space teammate and a matched-gain controller share the
        # primary task but differ in the joint step (the posture term), so for
        # the same obs their arm commands differ.
        ns = _partner("cocarry_nullspace_impedance")
        plain = _partner("cocarry_slew_impedance")  # matched gains, plain pinv
        ns.reset(seed=0)
        plain.reset(seed=0)
        goal = np.array([0.0, 0.12, 0.28])
        bar = _make_bar()
        a_ns = ns.act(_obs(_READY_Q7, goal, bar=bar))
        a_plain = plain.act(_obs(_READY_Q7, goal, bar=bar))
        assert not np.array_equal(a_ns[:7], a_plain[:7])


class TestGeometryHelpers:
    """The local geometry helpers (ADR-026 §D4)."""

    def test_parse_vec3_roundtrip(self) -> None:
        np.testing.assert_allclose(_parse_vec3("0.5,0,-0.5"), np.array([0.5, 0.0, -0.5]))

    def test_parse_vec3_rejects_wrong_arity(self) -> None:
        with pytest.raises(ValueError, match="x,y,z"):
            _parse_vec3("1,2")

    def test_parse_vec3_rejects_non_float(self) -> None:
        with pytest.raises(ValueError, match="floats"):
            _parse_vec3("a,b,c")

    def test_quat_identity_is_identity_matrix(self) -> None:
        np.testing.assert_allclose(_quat_wxyz_to_matrix(_IDENTITY_QUAT), np.eye(3), atol=1e-9)

    def test_quat_zero_norm_guard_returns_identity(self) -> None:
        np.testing.assert_array_equal(_quat_wxyz_to_matrix(np.zeros(4)), np.eye(3))

    def test_quat_90deg_about_z(self) -> None:
        # wxyz for +90 deg about z: w=cos45, z=sin45.
        c = np.cos(np.pi / 4)
        r = _quat_wxyz_to_matrix(np.array([c, 0.0, 0.0, c]))
        np.testing.assert_allclose(r @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-9)


def _make_bar(center: np.ndarray | None = None) -> np.ndarray:
    """A 7-D bar pose [xyz, wxyz] — level bar at ``center`` (default origin-ish)."""
    c = np.array([0.0, 0.0, 0.17]) if center is None else center
    return np.concatenate([c, _IDENTITY_QUAT])


# --------------------------------------------------------------------------
# Pure statistics (chamber.benchmarks.cocarry_ph).
# --------------------------------------------------------------------------

_SEEDS = list(range(12))


def _const_map(value: bool) -> dict[int, bool]:
    return dict.fromkeys(_SEEDS, value)


def _rate_map(n_success: int) -> dict[int, bool]:
    return {s: (s < n_success) for s in _SEEDS}


class TestCapabilityGate:
    """The capability gate (ADR-026 §D4; R-2026-06-B §15 Rung 3)."""

    def test_c_min_is_pre_registered_value(self) -> None:
        assert ph.C_MIN == 0.75
        assert ph.CALIBRATION_BAND == (0.75, 1.0)

    def test_gate_boundary_inclusive(self) -> None:
        assert ph.passes_capability_gate(0.75) is True  # == C_min passes
        assert ph.passes_capability_gate(0.7499) is False
        assert ph.passes_capability_gate(1.0) is True

    def test_success_rate(self) -> None:
        assert ph.success_rate(_const_map(True)) == 1.0
        assert ph.success_rate(_rate_map(6)) == 0.5
        assert np.isnan(ph.success_rate({}))


class TestPairedDelta:
    """The paired within-seed Δ (R-2026-06-B §15 Rung 3)."""

    def test_mean_delta(self) -> None:
        pd = ph.paired_delta(_const_map(True), _rate_map(6))
        assert pd["mean_delta"] == pytest.approx(0.5)
        assert pd["reference_rate"] == 1.0
        assert pd["shifted_rate"] == 0.5

    def test_mismatched_seed_sets_raise(self) -> None:
        with pytest.raises(ValueError, match="paired"):
            ph.paired_delta({0: True}, {1: True})


class TestTrimmedMeanIqm:
    """IQM / trimmed mean (the secondary estimator)."""

    def test_iqm_trims_extremes(self) -> None:
        # 0,0,1,...,1 — IQM (25% trim) drops the two extremes each side.
        vals = [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        assert ph.iqm(vals) == pytest.approx(1.0)

    def test_trimmed_mean_empty_is_nan(self) -> None:
        assert np.isnan(ph.trimmed_mean([]))


class TestClusterBootstrap:
    """The cluster (seed) bootstrap one-sided CI (R-2026-06-B §15; P6 determinism)."""

    def test_pooled_point_and_determinism(self) -> None:
        ref = _const_map(True)
        teammates = {"A": _rate_map(6), "B": _rate_map(9), "C": _rate_map(3)}
        b1 = ph.cluster_bootstrap_delta(ref, teammates, n_boot=3000, root_seed=0)
        b2 = ph.cluster_bootstrap_delta(ref, teammates, n_boot=3000, root_seed=0)
        # Pooled point = 1.0 - mean(0.5, 0.75, 0.25) = 0.5.
        assert b1["pooled_mean_delta"] == pytest.approx(0.5)
        # Byte-reproducible under P6 (same root_seed -> identical CI).
        assert b1["pooled_ci_lower_one_sided"] == b2["pooled_ci_lower_one_sided"]
        assert set(b1["per_teammate"]) == {"A", "B", "C"}

    def test_one_sided_lower_below_point(self) -> None:
        ref = _const_map(True)
        teammates = {"A": _rate_map(6)}
        b = ph.cluster_bootstrap_delta(ref, teammates, n_boot=3000, root_seed=0)
        assert b["pooled_ci_lower_one_sided"] <= b["pooled_mean_delta"]

    def test_mismatched_teammate_seeds_raise(self) -> None:
        with pytest.raises(ValueError, match="paired"):
            ph.cluster_bootstrap_delta(_const_map(True), {"A": {0: True}}, n_boot=10)


class TestDecisionRule:
    """The pre-committed decision + null rule (R-2026-06-B §15 Rung 3)."""

    def test_drop_when_above_delta_min_and_ci_excludes_zero(self) -> None:
        d = ph.decide(
            pooled_mean_delta=0.4,
            pooled_ci_lower_one_sided=0.15,
            pooled_ci_upper_for_null=0.6,
            positive_control_holds=True,
            any_excluded=False,
        )
        assert d["verdict"] == ph.VERDICT_DROP
        assert d["drop_rule_met"] is True

    def test_drop_requires_ci_excludes_zero(self) -> None:
        # Mean above Δ_min but the one-sided CI touches 0 -> not a drop.
        d = ph.decide(
            pooled_mean_delta=0.3,
            pooled_ci_lower_one_sided=-0.01,
            pooled_ci_upper_for_null=0.6,
            positive_control_holds=True,
            any_excluded=False,
        )
        assert d["verdict"] != ph.VERDICT_DROP

    def test_null_when_ci_excludes_delta_min_and_control_holds(self) -> None:
        d = ph.decide(
            pooled_mean_delta=0.02,
            pooled_ci_lower_one_sided=-0.05,
            pooled_ci_upper_for_null=0.10,  # < Δ_min 0.20
            positive_control_holds=True,
            any_excluded=False,
        )
        assert d["verdict"] == ph.VERDICT_NULL
        assert d["null_rule_met"] is True

    def test_null_requires_positive_control(self) -> None:
        d = ph.decide(
            pooled_mean_delta=0.02,
            pooled_ci_lower_one_sided=-0.05,
            pooled_ci_upper_for_null=0.10,
            positive_control_holds=False,
            any_excluded=False,
        )
        assert d["verdict"] == ph.VERDICT_INDETERMINATE

    def test_indeterminate_middle_ground(self) -> None:
        d = ph.decide(
            pooled_mean_delta=0.12,
            pooled_ci_lower_one_sided=0.02,
            pooled_ci_upper_for_null=0.30,  # straddles Δ_min
            positive_control_holds=True,
            any_excluded=False,
        )
        assert d["verdict"] == ph.VERDICT_INDETERMINATE

    def test_null_caveat_present_when_excluded(self) -> None:
        d = ph.decide(
            pooled_mean_delta=0.02,
            pooled_ci_lower_one_sided=-0.05,
            pooled_ci_upper_for_null=0.10,
            positive_control_holds=True,
            any_excluded=True,
        )
        assert "null_caveat" in d
        assert "truncate" in d["null_caveat"]


class TestGlmmConfirmatory:
    """The cluster-robust binomial confirmatory (R-2026-06-B §15 Rung 3)."""

    def test_ok_on_separable_but_nondegenerate(self) -> None:
        # Reference mostly succeeds; teammates mostly fail but with variation
        # on both sides (no perfect separation) -> a fitted negative coef.
        ref = {s: (s != 0) for s in _SEEDS}  # 11/12
        teammate = {s: (s % 3 == 0) for s in _SEEDS}  # 4/12
        out = ph.cluster_robust_glmm(ref, {"A": teammate})
        assert out["status"] == "ok"
        assert out["coef_shifted_logodds"] < 0.0  # shifted lowers success

    def test_degenerate_when_no_outcome_variation(self) -> None:
        out = ph.cluster_robust_glmm(_const_map(True), {"A": _const_map(True)})
        assert out["status"] == "degenerate"


class TestConjunctSummary:
    """The per-conjunct failure summary (R-2026-06-B §15 Rung 3 mechanism reporting)."""

    def test_counts_failed_conjuncts(self) -> None:
        metrics = [
            ph.ConjunctMetrics(
                seed=0,
                success=False,
                is_placed=False,
                is_level=True,
                is_unstressed=True,
                both_static=True,
                centroid_to_goal=0.13,
                max_tilt_deg=10.0,
                max_stress_proxy=70.0,
                n_steps=320,
            ),
            ph.ConjunctMetrics(
                seed=1,
                success=True,
                is_placed=True,
                is_level=True,
                is_unstressed=True,
                both_static=True,
                centroid_to_goal=0.09,
                max_tilt_deg=11.0,
                max_stress_proxy=72.0,
                n_steps=320,
            ),
        ]
        s = ph.conjunct_failure_summary(metrics)
        assert s["n"] == 2
        assert s["success_rate"] == pytest.approx(0.5)
        assert s["fail_placed"] == 1
        assert s["fail_level"] == 0

    def test_empty_summary(self) -> None:
        assert ph.conjunct_failure_summary([]) == {}


class TestPreRegisteredConstants:
    """The pre-registration-grade constants are fixed (R-2026-06-B §15 Rung 3)."""

    def test_seed_sets_disjoint_from_rung2(self) -> None:
        meas = set(ph.MEASUREMENT_SEEDS)
        calib = set(ph.CALIBRATION_SEEDS)
        rung2_s = set(range(10000, 10012))
        rung2_v = set(range(20000, 20024))
        assert len(meas) == 12
        assert len(calib) == 12
        assert meas.isdisjoint(calib)
        assert meas.isdisjoint(rung2_s | rung2_v)
        assert calib.isdisjoint(rung2_s | rung2_v)

    def test_delta_min_is_pre_registered(self) -> None:
        assert ph.DELTA_MIN == 0.20
