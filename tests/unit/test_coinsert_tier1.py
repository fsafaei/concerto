# SPDX-License-Identifier: Apache-2.0
# pyright: reportPrivateImportUsage=false, reportAttributeAccessIssue=false
"""Tier-1 (no-SAPIEN-scene) tests for the co-insert S0 skeleton (ADR-026 §Decision 1-4).

Covers the pure-Python surface of :mod:`chamber.envs.coinsert` that needs no
SAPIEN scene:

- module import on a Vulkan-less host (the Tier-1 contract; ADR-001 §Risks / P2).
- :func:`resolve_coinsert_condition` — the matched reference + single-inserter
  positive control, and the loud-fail on a bogus id.
- :func:`evaluate_coinsert_success` — the joint ``seated ∧ within_force ∧
  static ∧ settled`` conjunction on synthetic inputs (all-pass => success; each
  single violation => failure), and that the S2-derived force limits are
  required (no default).
- :func:`coinsert_capability_gate_floor` — the relative ``C_min = max(0.75,
  M - 0.25)`` rule (the co-insert design).
- :func:`coinsert_realism_compliance_line` — the proposed-ADR-018 §1 audit line
  on a compliant run, and the loud-fail on a realism violation.
- The frozen-able parameter + decision-rule constants present, typed, and at
  their declared co-insert values; the force limits are ``None`` placeholders.

Tier-2 SAPIEN-gated coverage (real :class:`CoInsertEnv` construction, the
S1 contact instrument, the S2 base inserter) is deferred to later slices.

ADR-026 §Decision 1-4; ADR-009 §Consequences; the co-insert design
"""

from __future__ import annotations

import numpy as np
import pytest

from chamber.envs.coinsert import (
    COINSERT_C_MIN_FLOOR,
    COINSERT_C_MIN_MARGIN,
    COINSERT_CLEARANCE_SET_M,
    COINSERT_COUPLE_FORCE_MAX_N,
    COINSERT_DELTA_MIN,
    COINSERT_DEPTH_TARGET_M,
    COINSERT_GATE0_BASE_FAILURE_MAX,
    COINSERT_INSERT_FORCE_MAX_N,
    COINSERT_N_BOOT,
    COINSERT_PEG_DIAMETER_M,
    COINSERT_RECEPTACLE_MASS_KG,
    COINSERT_REFERENCE_SUCCESS_MIN,
    CoInsertCondition,
    coinsert_capability_gate_floor,
    coinsert_realism_compliance_line,
    evaluate_coinsert_success,
    resolve_coinsert_condition,
)

_MATCHED = "coinsert_matched_reference"
_SINGLE = "coinsert_single_inserter_positive_control"

# Synthetic S2-derived force limits for the predicate tests (real values are
# measured at S2; these are test fixtures, not asserted task numbers).
_F_INSERT = 50.0
_F_COUPLE = 80.0


class TestResolveCondition:
    """Co-insert condition resolution (ADR-026 §Decision 1-2)."""

    def test_matched_reference(self) -> None:
        cfg = resolve_coinsert_condition(_MATCHED)
        assert isinstance(cfg, CoInsertCondition)
        assert cfg.agent_uids == ("panda_wristcam", "panda_partner")
        assert cfg.single_inserter is False

    def test_single_inserter_positive_control(self) -> None:
        cfg = resolve_coinsert_condition(_SINGLE)
        assert cfg.agent_uids == ("panda_wristcam", "panda_partner")
        assert cfg.single_inserter is True

    def test_unknown_condition_raises_naming_options(self) -> None:
        with pytest.raises(ValueError, match="not one of the co-insert"):
            resolve_coinsert_condition("coinsert_bogus")
        with pytest.raises(ValueError, match="ADR-026"):
            resolve_coinsert_condition("")


class TestSuccessPredicate:
    """Joint ``seated ∧ within_force ∧ static ∧ settled`` predicate (the co-insert design)."""

    @staticmethod
    def _ok_kwargs() -> dict[str, object]:
        # A clean success: seated to depth, aligned, both forces under budget,
        # static, settled.
        return {
            "seated_depth_m": COINSERT_DEPTH_TARGET_M,
            "axis_align_deg": 1.0,
            "peak_insert_force_n": 10.0,
            "peak_couple_wrench_n": 20.0,
            "both_static": True,
            "settled": True,
            "f_insert_max": _F_INSERT,
            "f_couple_max": _F_COUPLE,
        }

    def test_all_conjuncts_pass_is_success(self) -> None:
        assert bool(evaluate_coinsert_success(**self._ok_kwargs())) is True  # type: ignore[arg-type]

    def test_not_seated_deep_enough_fails(self) -> None:
        kw = self._ok_kwargs()
        kw["seated_depth_m"] = COINSERT_DEPTH_TARGET_M / 2.0
        assert bool(evaluate_coinsert_success(**kw)) is False  # type: ignore[arg-type]

    def test_axis_misaligned_fails(self) -> None:
        kw = self._ok_kwargs()
        kw["axis_align_deg"] = 45.0
        assert bool(evaluate_coinsert_success(**kw)) is False  # type: ignore[arg-type]

    def test_over_insert_force_fails(self) -> None:
        kw = self._ok_kwargs()
        kw["peak_insert_force_n"] = _F_INSERT + 1.0
        assert bool(evaluate_coinsert_success(**kw)) is False  # type: ignore[arg-type]

    def test_over_couple_wrench_fails(self) -> None:
        kw = self._ok_kwargs()
        kw["peak_couple_wrench_n"] = _F_COUPLE + 1.0
        assert bool(evaluate_coinsert_success(**kw)) is False  # type: ignore[arg-type]

    def test_not_static_fails(self) -> None:
        kw = self._ok_kwargs()
        kw["both_static"] = False
        assert bool(evaluate_coinsert_success(**kw)) is False  # type: ignore[arg-type]

    def test_not_settled_fails(self) -> None:
        kw = self._ok_kwargs()
        kw["settled"] = False
        assert bool(evaluate_coinsert_success(**kw)) is False  # type: ignore[arg-type]

    def test_broadcasts_over_batches(self) -> None:
        out = evaluate_coinsert_success(
            seated_depth_m=np.array([COINSERT_DEPTH_TARGET_M, 0.0]),
            axis_align_deg=np.array([1.0, 1.0]),
            peak_insert_force_n=np.array([10.0, 10.0]),
            peak_couple_wrench_n=np.array([20.0, 20.0]),
            both_static=np.array([True, True]),
            settled=np.array([True, True]),
            f_insert_max=_F_INSERT,
            f_couple_max=_F_COUPLE,
        )
        assert out.tolist() == [True, False]

    def test_force_limits_are_required_no_default(self) -> None:
        # The S2-derived limits must be supplied explicitly — S0 asserts no
        # unmeasured number (the co-insert design).
        kw = self._ok_kwargs()
        del kw["f_insert_max"]
        with pytest.raises(TypeError):
            evaluate_coinsert_success(**kw)  # type: ignore[arg-type]


class TestCapabilityGateFloor:
    """Relative ``C_min = max(0.75, M - 0.25)`` (the co-insert design; ADR-026 §Risks)."""

    def test_comfortable_reference_lifts_floor(self) -> None:
        # M = 0.98 => max(0.75, 0.73) = 0.75 (floor binds just below the margin).
        assert coinsert_capability_gate_floor(0.98) == pytest.approx(0.75)
        # M = 1.0 => max(0.75, 0.75) = 0.75.
        assert coinsert_capability_gate_floor(1.0) == pytest.approx(0.75)

    def test_relative_form_exceeds_floor_for_high_reference(self) -> None:
        # M = 1.05 (hypothetical) => max(0.75, 0.80) = 0.80 — the relative form
        # lifts above the bare floor.
        assert coinsert_capability_gate_floor(1.05) == pytest.approx(0.80)

    def test_low_reference_clamps_to_floor(self) -> None:
        assert coinsert_capability_gate_floor(0.4) == pytest.approx(COINSERT_C_MIN_FLOOR)


class TestRealismComplianceLine:
    """Proposed-ADR-018 §1 realism-compliance audit line (the co-insert design)."""

    def test_compliant_run_emits_parseable_line(self) -> None:
        line = coinsert_realism_compliance_line(
            partner_uid="panda_partner",
            partner_class="PandaReferenceHolder",
            frozen_assert_passed=True,
            forbidden_imports_absent=True,
            condition_id=_MATCHED,
        )
        assert line.startswith("realism_compliance ")
        assert "partner_uid=panda_partner" in line
        assert "frozen_assert_passed=true" in line
        assert "forbidden_imports_absent=true" in line
        # The shield set is enumerated for audit (load_state_dict is a member).
        assert "load_state_dict" in line
        assert "ADR-009" in line

    @pytest.mark.parametrize(
        ("frozen", "imports_absent"),
        [(False, True), (True, False), (False, False)],
    )
    def test_violation_fails_loudly(self, frozen: bool, imports_absent: bool) -> None:
        with pytest.raises(ValueError, match="realism-compliance VIOLATION"):
            coinsert_realism_compliance_line(
                partner_uid="panda_partner",
                partner_class="LeakyHolder",
                frozen_assert_passed=frozen,
                forbidden_imports_absent=imports_absent,
                condition_id=_MATCHED,
            )


class TestFrozenableConstants:
    """The declared co-insert frozen-able constants are present, typed, and correct."""

    def test_physical_parameters_match_design(self) -> None:
        assert pytest.approx(0.016) == COINSERT_PEG_DIAMETER_M
        assert COINSERT_CLEARANCE_SET_M == (1.0e-3, 0.5e-3, 0.2e-3)
        assert pytest.approx(0.040) == COINSERT_DEPTH_TARGET_M
        assert pytest.approx(0.5) == COINSERT_RECEPTACLE_MASS_KG

    def test_decision_rule_constants_match_d4(self) -> None:
        assert pytest.approx(0.20) == COINSERT_DELTA_MIN
        assert pytest.approx(0.75) == COINSERT_C_MIN_FLOOR
        assert pytest.approx(0.25) == COINSERT_C_MIN_MARGIN
        assert pytest.approx(0.5) == COINSERT_GATE0_BASE_FAILURE_MAX
        assert pytest.approx(0.9) == COINSERT_REFERENCE_SUCCESS_MIN
        assert COINSERT_N_BOOT == 10000

    def test_force_limits_are_none_placeholders_until_s2(self) -> None:
        # Derived from measured matched-pair distributions at S2; never asserted
        # at S0 (the co-insert design).
        assert COINSERT_INSERT_FORCE_MAX_N is None
        assert COINSERT_COUPLE_FORCE_MAX_N is None
