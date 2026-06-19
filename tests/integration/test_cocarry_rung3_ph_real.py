# SPDX-License-Identifier: Apache-2.0
# pyright: reportPrivateImportUsage=false, reportAttributeAccessIssue=false
"""Tier-2 SAPIEN/CUDA-gated smoke for the Rung-3 PH measurement pipeline (ADR-026 §Decision 4).

Real ManiSkill v3 coverage that needs a Vulkan/GPU host. Mirrors
:mod:`tests.integration.test_cocarry_rung2_real`'s skipif pattern; skipped on
CPU-only runners and (for the HARL-backed frozen incumbent) without the
``train`` dependency group.

Coverage (R-2026-06-B §15 Rung 3) — pipeline well-formedness, NOT the
measurement itself (the headline Δ is the founder's pre-registered run via
``scripts/repro/cocarry_rung3_ph_measure.sh``; these run on throwaway seeds
with a short horizon to bound Tier-2 runtime):

1. **Calibration pipeline.** An admitted teammate paired with the cooperative
   reference ego rolls a well-formed :class:`~chamber.benchmarks.cocarry_ph.ConjunctMetrics`
   with the conjunct breakdown populated.
2. **Reference + shifted pipeline.** The frozen incumbent (ego) against the
   matched partner and against a shifted teammate both roll well-formed
   metrics — the generic two-controller rollout drives either seat.

ADR-026 §Decision 4; ADR-009 §Decision; R-2026-06-B §15 Rung 3.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from chamber.utils.device import sapien_gpu_available

pytestmark = [
    pytest.mark.slow,
    pytest.mark.gpu,
    pytest.mark.skipif(
        not sapien_gpu_available(),
        reason="Requires Vulkan/GPU (SAPIEN); skipped on CPU-only machines",
    ),
]

# The frozen incumbent's EgoPPOTrainer is unavailable without the `train`
# group; skip the whole module rather than error at import.
pytest.importorskip("harl", reason="frozen incumbent needs the HARL fork (--group train)")

_CONFIG = Path("configs/training/ego_aht_happo/cocarry_matched.yaml")
_CKPT_URI = "local://artifacts/f6dad85ec9df4d58_step100000.pt"
_ARTIFACTS_ROOT = Path("artifacts")
# Throwaway seeds + short horizon — pipeline smoke, not the measurement.
_SMOKE_SEEDS = [888000, 888001]
_SMOKE_EPISODE_LEN = 120


def _cfg():
    from concerto.training.config import load_config

    return load_config(config_path=_CONFIG, overrides=[])


def _assert_well_formed(m) -> None:
    from chamber.benchmarks.cocarry_ph import ConjunctMetrics

    assert isinstance(m, ConjunctMetrics)
    assert isinstance(m.success, bool)
    for flag in (m.is_placed, m.is_level, m.is_unstressed, m.both_static):
        assert isinstance(flag, bool)
    assert np.isfinite(m.centroid_to_goal)
    assert np.isfinite(m.max_tilt_deg)
    assert m.max_tilt_deg >= 0.0
    assert np.isfinite(m.max_stress_proxy)
    assert m.max_stress_proxy >= 0.0


class TestCalibrationPipeline:
    """An admitted teammate + cooperative reference rolls well-formed metrics (ADR-026 §D4)."""

    def test_calibration_episode_well_formed(self) -> None:
        from chamber.benchmarks.cocarry_ph import evaluate_calibration

        metrics = evaluate_calibration(
            candidate_class="cocarry_stiff_impedance",
            seeds=_SMOKE_SEEDS,
            episode_length=_SMOKE_EPISODE_LEN,
            render_backend="none",
        )
        assert len(metrics) == len(_SMOKE_SEEDS)
        for m in metrics:
            _assert_well_formed(m)


class TestReferenceAndShiftedPipeline:
    """The frozen incumbent rolls against the matched + a shifted teammate (ADR-026 §D4)."""

    def test_reference_pipeline_well_formed(self) -> None:
        from chamber.benchmarks.cocarry_ph import evaluate_incumbent_vs_partner

        metrics = evaluate_incumbent_vs_partner(
            cfg=_cfg(),
            checkpoint_uri=_CKPT_URI,
            artifacts_root=_ARTIFACTS_ROOT,
            partner_class="cocarry_impedance",
            seeds=_SMOKE_SEEDS,
            episode_length=_SMOKE_EPISODE_LEN,
            render_backend="none",
        )
        assert len(metrics) == len(_SMOKE_SEEDS)
        for m in metrics:
            _assert_well_formed(m)

    def test_shifted_pipeline_well_formed(self) -> None:
        from chamber.benchmarks.cocarry_ph import evaluate_incumbent_vs_partner

        metrics = evaluate_incumbent_vs_partner(
            cfg=_cfg(),
            checkpoint_uri=_CKPT_URI,
            artifacts_root=_ARTIFACTS_ROOT,
            partner_class="cocarry_admittance",
            seeds=_SMOKE_SEEDS,
            episode_length=_SMOKE_EPISODE_LEN,
            render_backend="none",
        )
        assert len(metrics) == len(_SMOKE_SEEDS)
        for m in metrics:
            _assert_well_formed(m)
