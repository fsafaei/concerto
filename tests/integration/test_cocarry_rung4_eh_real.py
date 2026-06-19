# SPDX-License-Identifier: Apache-2.0
# pyright: reportPrivateImportUsage=false, reportAttributeAccessIssue=false
"""Tier-2 SAPIEN/CUDA-gated smoke for the Rung-4 embodiment-shift rig (ADR-026 §D4; ADR-005).

Real ManiSkill v3 coverage that needs a Vulkan/GPU host (and the xArm6 asset).
Mirrors :mod:`tests.integration.test_cocarry_rung3_ph_real`'s skip pattern.

Coverage (R-2026-06-B §15 Rung 4) — rig well-formedness, NOT the measurement
(the EH measurement does not run: the xArm6 fails the capability gate; the
calibration roster is the founder's GPU artifact). Throwaway seeds, short
horizon:

1. **Mixed-embodiment stability + calibration pipeline.** The Panda ego
   (cooperative reference) + the xArm6 partner on one welded bar roll a
   well-formed, finite :class:`~chamber.benchmarks.cocarry_ph.ConjunctMetrics`.
2. **Frozen incumbent loads on the xArm6 env.** The 46-D actor + critic load
   through the partner-observation adapter and the incumbent runs against the
   xArm6 teammate (well-formed metrics) — the embodiment-shift eval path.

ADR-026 §Decision 4; ADR-005 §Decision; ADR-009 §Decision; R-2026-06-B §15 Rung 4.
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

pytest.importorskip("harl", reason="frozen incumbent needs the HARL fork (--group train)")

_CONFIG = Path("configs/training/ego_aht_happo/cocarry_matched.yaml")
_CKPT_URI = "local://artifacts/f6dad85ec9df4d58_step100000.pt"
_ARTIFACTS_ROOT = Path("artifacts")
_SMOKE_SEEDS = [889000, 889001]
_SMOKE_EPISODE_LEN = 120


def _assert_well_formed(m) -> None:
    from chamber.benchmarks.cocarry_ph import ConjunctMetrics

    assert isinstance(m, ConjunctMetrics)
    assert isinstance(m.success, bool)
    assert np.isfinite(m.max_tilt_deg)
    assert np.isfinite(m.max_stress_proxy)
    assert np.isfinite(m.centroid_to_goal)


class TestMixedEmbodimentStabilityPipeline:
    """Panda ego + xArm6 partner roll a finite, well-formed episode (ADR-026 §D4; ADR-005)."""

    def test_calibration_episode_finite(self) -> None:
        from chamber.benchmarks.cocarry_ph import (
            XARM6_CONDITION_ID,
            XARM6_PARTNER_CLASS,
            XARM6_PARTNER_UID,
            evaluate_calibration,
        )

        metrics = evaluate_calibration(
            candidate_class=XARM6_PARTNER_CLASS,
            seeds=_SMOKE_SEEDS,
            condition_id=XARM6_CONDITION_ID,
            partner_uid=XARM6_PARTNER_UID,
            episode_length=_SMOKE_EPISODE_LEN,
            render_backend="none",
        )
        assert len(metrics) == len(_SMOKE_SEEDS)
        for m in metrics:
            _assert_well_formed(m)


class TestFrozenIncumbentOnXArm6:
    """The frozen incumbent loads (adapter) + runs against the xArm6 (ADR-026 §D4; Rung 4)."""

    def test_incumbent_vs_xarm6_well_formed(self) -> None:
        from chamber.benchmarks.cocarry_ph import (
            XARM6_CONDITION_ID,
            XARM6_PARTNER_CLASS,
            XARM6_PARTNER_UID,
            evaluate_incumbent_vs_partner,
        )
        from concerto.training.config import load_config

        cfg = load_config(config_path=_CONFIG, overrides=[])
        metrics = evaluate_incumbent_vs_partner(
            cfg=cfg,
            checkpoint_uri=_CKPT_URI,
            artifacts_root=_ARTIFACTS_ROOT,
            partner_class=XARM6_PARTNER_CLASS,
            seeds=_SMOKE_SEEDS,
            episode_length=_SMOKE_EPISODE_LEN,
            render_backend="none",
            condition_id=XARM6_CONDITION_ID,
            partner_uid=XARM6_PARTNER_UID,
        )
        assert len(metrics) == len(_SMOKE_SEEDS)
        for m in metrics:
            _assert_well_formed(m)
