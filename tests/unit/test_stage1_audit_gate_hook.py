# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the Stage-1b audit-gate hook (P1.04.5; ADR-007 §Stage 1b).

The hook lives at ``scripts/repro/stage1_{as,om}_stage1b.sh``. These
tests invoke the script via :mod:`subprocess` against synthetic JSONL
fixtures that mimic the ``safety_telemetry_final`` events
:func:`concerto.training.ego_aht.train` emits. Pins the predicate
exit codes (8 = saturation, 9 = λ stuck, 0 = pass / vacuous /
safety-disabled, 2 = usage, 3 = JSONL malformed).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_AS_SCRIPT = _REPO_ROOT / "scripts" / "repro" / "stage1_as_stage1b_audit_gate.sh"
_OM_SCRIPT = _REPO_ROOT / "scripts" / "repro" / "stage1_om_stage1b_audit_gate.sh"


def _write_jsonl(tmp_path: Path, *events: dict) -> Path:
    """Write one synthetic JSONL with the given events; return path."""
    path = tmp_path / "run.jsonl"
    with path.open("w") as f:
        for event in events:
            f.write(json.dumps(event) + "\n")
    return path


def _final(**overrides: object) -> dict:
    """Build a safety_telemetry_final event with sane defaults."""
    base = {
        "event": "safety_telemetry_final",
        "safety_enabled": True,
        "predictor_kind": "constant_velocity",
        "lambda_steady_state": 0.5,
        "lambda_mean": 0.5,
        "lambda_var": 0.01,
        "lambda_max_observed": 1.0,
        "lambda_min_observed": 0.0,
        "cartesian_accel_capacity": 10.0,
        "saturation_threshold": 0.9,
        "saturated": False,
        "n_filter_calls": 100,
        "n_fallback_fires": 0,
        "n_qp_infeasible": 0,
    }
    base.update(overrides)
    return base


def _run_hook(script: Path, jsonl_path: Path) -> tuple[int, str, str]:
    """Run the audit-gate hook script; return (exit_code, stdout, stderr)."""
    result = subprocess.run(  # noqa: S603 - test invokes a known repo-local script
        ["bash", str(script), str(jsonl_path)],  # noqa: S607 - bash is repo-required for the audit-gate hook test; hard-coding /bin/bash would break on non-FHS hosts (NixOS, macOS Homebrew, CI runners with /usr/local/bin/bash)
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode, result.stdout, result.stderr


# ----- Pass cases -----


class TestAuditGateHookPassCases:
    """Both predicates pass → exit 0."""

    def test_low_lambda_with_variance_passes(self, tmp_path: Path) -> None:
        """λ adapted to 0.5 with variance 0.01; both predicates pass."""
        jsonl = _write_jsonl(tmp_path, _final())
        code, stdout, _ = _run_hook(_AS_SCRIPT, jsonl)
        assert code == 0
        assert "PASS audit gate" in stdout

    def test_vacuous_case_lambda_stays_at_zero_passes(self, tmp_path: Path) -> None:
        """λ_mean ≈ 0 (never adapted): predicate B is vacuously satisfied."""
        jsonl = _write_jsonl(
            tmp_path,
            _final(lambda_mean=0.0, lambda_var=0.0, lambda_steady_state=0.0),
        )
        code, _, _ = _run_hook(_AS_SCRIPT, jsonl)
        assert code == 0


# ----- Predicate A: saturation (exit 8) -----


class TestAuditGatePredicateA:
    """λ_steady_state >= 0.9 x cartesian_accel_capacity → exit 8."""

    def test_lambda_at_threshold_trips_predicate_a(self, tmp_path: Path) -> None:
        """λ_steady_state = 9.5, threshold = 0.9 x 10 = 9.0."""
        jsonl = _write_jsonl(
            tmp_path,
            _final(lambda_steady_state=9.5, lambda_mean=9.0, lambda_var=0.5),
        )
        code, _, stderr = _run_hook(_AS_SCRIPT, jsonl)
        assert code == 8
        assert "predicate A" in stderr
        assert "saturated" in stderr.lower()

    def test_lambda_just_above_threshold_trips(self, tmp_path: Path) -> None:
        """λ_steady_state = 9.01 (just above 9.0 threshold) → exit 8."""
        jsonl = _write_jsonl(
            tmp_path,
            _final(lambda_steady_state=9.01, lambda_mean=8.0, lambda_var=0.5),
        )
        code, _, _ = _run_hook(_AS_SCRIPT, jsonl)
        assert code == 8


# ----- Predicate A symmetric (issue #178) -----


class TestAuditGatePredicateASymmetric:
    """λ_steady_state <= -saturation_threshold * cap → exit 8 (#178)."""

    def test_negative_lambda_at_threshold_trips_predicate_a(self, tmp_path: Path) -> None:
        """λ_steady_state = -9.5, |λ| >= 0.9 x 10 = 9.0 → exit 8."""
        jsonl = _write_jsonl(
            tmp_path,
            _final(lambda_steady_state=-9.5, lambda_mean=-9.0, lambda_var=0.5),
        )
        code, _, stderr = _run_hook(_AS_SCRIPT, jsonl)
        assert code == 8
        assert "predicate A" in stderr
        assert "symmetric" in stderr.lower()

    def test_negative_lambda_just_below_negative_threshold_passes(self, tmp_path: Path) -> None:
        """λ_steady_state = -8.99 → |λ|=8.99 < 9.0 → exit 0."""
        jsonl = _write_jsonl(
            tmp_path,
            _final(lambda_steady_state=-8.99, lambda_mean=-5.0, lambda_var=0.5),
        )
        code, _, _ = _run_hook(_AS_SCRIPT, jsonl)
        assert code == 0

    def test_negative_lambda_at_production_extrapolation_trips(self, tmp_path: Path) -> None:
        """Production-scale extrapolation (λ_ss ≈ -49) easily trips the symmetric guard."""
        jsonl = _write_jsonl(
            tmp_path,
            _final(lambda_steady_state=-49.0, lambda_mean=-25.0, lambda_var=0.5),
        )
        code, _, _ = _run_hook(_AS_SCRIPT, jsonl)
        assert code == 8


# ----- Predicate B: λ stuck (exit 9) -----


class TestAuditGatePredicateB:
    """λ adapted but didn't vary → exit 9."""

    def test_lambda_adapted_but_stuck_at_constant_trips_predicate_b(self, tmp_path: Path) -> None:
        """λ_mean > 1e-6 (adapted) but λ_var ≈ 0 (stuck) → exit 9."""
        jsonl = _write_jsonl(
            tmp_path,
            _final(
                lambda_steady_state=0.5,
                lambda_mean=0.5,
                lambda_var=0.0,
            ),
        )
        code, _, stderr = _run_hook(_AS_SCRIPT, jsonl)
        assert code == 9
        assert "predicate B" in stderr
        assert "stuck" in stderr.lower()

    def test_lambda_mean_just_below_threshold_does_not_trip_predicate_b(
        self, tmp_path: Path
    ) -> None:
        """λ_mean = 1e-7 (below 1e-6 adaptation threshold) → vacuous, exit 0."""
        jsonl = _write_jsonl(
            tmp_path,
            _final(
                lambda_steady_state=1e-7,
                lambda_mean=1e-7,
                lambda_var=0.0,
            ),
        )
        code, _, _ = _run_hook(_AS_SCRIPT, jsonl)
        assert code == 0

    def test_negative_lambda_mean_with_stuck_var_trips_predicate_b(self, tmp_path: Path) -> None:
        """λ_mean = -0.5 (|λ_mean| > 1e-6, adapted negatively) AND λ_var = 0 → exit 9 (#178).

        Mirrors the positive-direction stuck case under the eps<0 regime: if λ
        drifted negative but the variance is zero, the conformal slack is
        degenerate just as it would be under a positive drift with no variance.
        """
        jsonl = _write_jsonl(
            tmp_path,
            _final(
                lambda_steady_state=-0.5,
                lambda_mean=-0.5,
                lambda_var=0.0,
            ),
        )
        code, _, stderr = _run_hook(_AS_SCRIPT, jsonl)
        assert code == 9
        assert "predicate B" in stderr

    def test_negative_lambda_mean_just_below_adaptation_threshold_passes(
        self, tmp_path: Path
    ) -> None:
        """λ_mean = -1e-7 → |λ_mean|=1e-7 < 1e-6 → vacuous (#178)."""
        jsonl = _write_jsonl(
            tmp_path,
            _final(
                lambda_steady_state=-1e-7,
                lambda_mean=-1e-7,
                lambda_var=0.0,
            ),
        )
        code, _, _ = _run_hook(_AS_SCRIPT, jsonl)
        assert code == 0


# ----- Safety-disabled path -----


class TestAuditGateSafetyDisabledPath:
    """safety_enabled=False → exit 0 with operator-override message."""

    def test_safety_disabled_emits_warning_and_passes(self, tmp_path: Path) -> None:
        jsonl = _write_jsonl(tmp_path, _final(safety_enabled=False))
        code, stdout, _ = _run_hook(_AS_SCRIPT, jsonl)
        assert code == 0
        assert "safety disabled by operator override" in stdout

    def test_safety_disabled_skips_predicate_a_even_when_saturated(self, tmp_path: Path) -> None:
        """safety_enabled=False overrides even when λ would saturate."""
        jsonl = _write_jsonl(
            tmp_path,
            _final(safety_enabled=False, lambda_steady_state=9.99),
        )
        code, _, _ = _run_hook(_AS_SCRIPT, jsonl)
        assert code == 0


# ----- Error paths -----


class TestAuditGateErrorPaths:
    """Usage + malformed-JSONL error codes."""

    def test_missing_argument_exits_2(self, tmp_path: Path) -> None:
        result = subprocess.run(  # noqa: S603 - test invokes a known repo-local script
            ["bash", str(_AS_SCRIPT)],  # noqa: S607 - bash is repo-required; hard-coding /bin/bash would break on non-FHS hosts
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 2
        assert "Usage" in result.stderr

    def test_nonexistent_jsonl_exits_2(self, tmp_path: Path) -> None:
        code, _, stderr = _run_hook(_AS_SCRIPT, tmp_path / "does_not_exist.jsonl")
        assert code == 2
        assert "not found" in stderr

    def test_jsonl_without_final_event_exits_3(self, tmp_path: Path) -> None:
        """No safety_telemetry_final event in JSONL → exit 3."""
        # Synthetic JSONL with only training_start + rollout_update events.
        jsonl = _write_jsonl(
            tmp_path,
            {"event": "training_start", "total_frames": 30},
            {"event": "rollout_update", "step": 10},
        )
        code, _, stderr = _run_hook(_AS_SCRIPT, jsonl)
        assert code == 3
        assert "no safety_telemetry_final" in stderr


# ----- OM script parity -----


class TestOMScriptParity:
    """The OM-axis hook is structurally identical to the AS-axis hook."""

    def test_om_hook_passes_on_low_lambda(self, tmp_path: Path) -> None:
        """Same JSONL → same exit code from the OM-axis script."""
        jsonl = _write_jsonl(tmp_path, _final())
        code, _, _ = _run_hook(_OM_SCRIPT, jsonl)
        assert code == 0

    def test_om_hook_trips_predicate_a_on_saturation(self, tmp_path: Path) -> None:
        jsonl = _write_jsonl(
            tmp_path,
            _final(lambda_steady_state=9.5, lambda_mean=9.0, lambda_var=0.5),
        )
        code, _, _ = _run_hook(_OM_SCRIPT, jsonl)
        assert code == 8
