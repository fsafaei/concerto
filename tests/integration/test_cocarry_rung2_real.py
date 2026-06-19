# SPDX-License-Identifier: Apache-2.0
# pyright: reportPrivateImportUsage=false, reportAttributeAccessIssue=false
"""Tier-2 SAPIEN/CUDA-gated tests for the Rung-2 frozen co-carry incumbent (ADR-026 §Decision 4).

Real ManiSkill v3 training + frozen-incumbent coverage that needs a
Vulkan/GPU host. Mirrors :mod:`tests.integration.test_cocarry_real`'s skipif
pattern; the whole module is skipped on CPU-only runners and (for the
HARL-backed trainer) without the ``train`` dependency group.

Coverage (R-2026-06-B §15 Rung 2):

1. **Training-smoke learning signal.** A short single-env co-carry training
   run wires the env + synthesizer + frozen ``cocarry_impedance`` partner +
   HAPPO trainer end-to-end and produces a learning-signal verdict the
   ADR-002 slope check can compute (finite slope, a real PASSED/FAILED
   verdict, a healthy reward curve). The **significant-positive-slope gate
   is the founder's longer train-to-reference run** (the Step-1 stop
   criterion), not this smoke — a smoke-budget run on the truncated task is
   not expected to clear alpha=0.05 (see the slice plan / build-order note).
2. **Deterministic frozen-incumbent reload (§3.4).** Loading the same
   checkpoint twice and acting on identical obs yields byte-identical
   actions (``deterministic=True`` is the policy mode — no RNG).
3. **Matched-eval pipeline.** The frozen incumbent (ego) + matched frozen
   partner roll an episode and produce well-formed
   :class:`~chamber.benchmarks.cocarry_runner.EpisodeMetrics`. The
   near-the-Rung-1-reference assertion belongs to the founder's full run /
   the repro-script gate, not a smoke-budget incumbent.

ADR-026 §Decision 4; ADR-002 §Risks #1 (slope check) + partner-freeze gate;
ADR-009 §Decision; R-2026-06-B §15 Rung 2.
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

# The HARL-backed EgoPPOTrainer is unavailable without the `train` group;
# skip the whole module rather than error at import (project convention).
pytest.importorskip("harl", reason="EgoPPOTrainer needs the HARL fork (--group train)")

_CONFIG = Path("configs/training/ego_aht_happo/cocarry_matched.yaml")

# Smoke budget: single-env, short truncated episodes, enough episodes
# (>= DEFAULT_MIN_EPISODES=20) for a non-vacuous slope verdict. Kept small
# to bound Tier-2 runtime; the founder's gate run uses the full config.
_SMOKE_OVERRIDES = [
    "total_frames=4000",
    "env.episode_length=80",
    "happo.rollout_length=80",
    "happo.batch_size=80",
    "checkpoint_every=2000",
    "seed=0",
]


def _smoke_cfg(tmp_path: Path):  # EgoAHTConfig — concerto import kept inside to stay light
    from concerto.training.config import load_config

    return load_config(
        config_path=_CONFIG,
        overrides=[
            *_SMOKE_OVERRIDES,
            f"artifacts_root={tmp_path / 'artifacts'}",
            f"log_dir={tmp_path / 'logs'}",
        ],
    )


class TestTrainingSmoke:
    """The co-carry training cell wires end-to-end and yields a slope verdict (ADR-026 §D4)."""

    def test_training_smoke_produces_learning_signal(self, tmp_path: Path) -> None:
        from chamber.benchmarks.cocarry_incumbent import slope_report_from_curve
        from chamber.benchmarks.training_runner import run_training
        from concerto.training.learning_signal_check import CheckStatus

        cfg = _smoke_cfg(tmp_path)
        result = run_training(cfg)
        curve = result.curve

        # The reward curve is healthy: enough episodes for a real verdict,
        # all finite.
        rewards = np.asarray(
            [float(np.asarray(r).reshape(-1)[0]) for r in curve.per_episode_ego_rewards]
        )
        assert rewards.size >= 20
        assert np.all(np.isfinite(rewards))
        assert len(curve.checkpoint_paths) >= 1

        # The ADR-002 slope check is operative: a real verdict (not
        # INVALID / INSUFFICIENT_DATA) with a finite slope. We do NOT assert
        # CheckStatus.PASSED here — clearing alpha=0.05 on a smoke-budget run of
        # the truncated task is the founder's longer train-to-reference gate
        # (the Step-1 stop criterion), per the slice plan.
        report = slope_report_from_curve(curve)
        assert report.status in (CheckStatus.PASSED, CheckStatus.FAILED)
        assert np.isfinite(report.slope)


class TestFrozenIncumbentDeterminism:
    """The frozen incumbent loads + acts deterministically across reloads (§3.4; ADR-026 §D4)."""

    def test_reload_is_byte_identical(self, tmp_path: Path) -> None:
        from chamber.benchmarks.cocarry_incumbent import build_matched_eval, load_frozen_incumbent
        from chamber.benchmarks.training_runner import run_training

        cfg = _smoke_cfg(tmp_path)
        result = run_training(cfg)
        ckpt = result.curve.checkpoint_paths[-1]
        uri = "local://artifacts/" + ckpt.name
        artifacts_root = tmp_path / "artifacts"

        env, partner = build_matched_eval(root_seed=123, episode_length=80)
        try:
            obs, _ = env.reset(seed=123)
            act_a = load_frozen_incumbent(
                cfg=cfg, env=env, partner=partner, checkpoint_uri=uri, artifacts_root=artifacts_root
            )
            act_b = load_frozen_incumbent(
                cfg=cfg, env=env, partner=partner, checkpoint_uri=uri, artifacts_root=artifacts_root
            )
            a1 = act_a(obs)
            a2 = act_b(obs)
            assert a1.shape == (8,)
            assert a1.dtype == np.float32
            assert np.array_equal(a1, a2)
        finally:
            env.close()


class TestMatchedEvalPipeline:
    """The frozen incumbent + matched partner roll a well-formed eval episode (ADR-026 §D4)."""

    def test_eval_episode_is_well_formed(self, tmp_path: Path) -> None:
        from chamber.benchmarks.cocarry_incumbent import (
            build_matched_eval,
            load_frozen_incumbent,
            rollout_incumbent_episode,
        )
        from chamber.benchmarks.training_runner import run_training

        cfg = _smoke_cfg(tmp_path)
        result = run_training(cfg)
        uri = "local://artifacts/" + result.curve.checkpoint_paths[-1].name
        artifacts_root = tmp_path / "artifacts"

        env, partner = build_matched_eval(root_seed=7, episode_length=80)
        try:
            ego_act = load_frozen_incumbent(
                cfg=cfg, env=env, partner=partner, checkpoint_uri=uri, artifacts_root=artifacts_root
            )
            metrics = rollout_incumbent_episode(
                ego_act=ego_act, env=env, partner=partner, seed=7, episode_length=80
            )
        finally:
            env.close()

        # Well-formed metrics — the eval pipeline (ego=incumbent,
        # partner=impedance) executes and the joint predicate is computable.
        # The "near the Rung-1 reference" gate is the founder's full run.
        assert isinstance(metrics.success, bool)
        assert np.isfinite(metrics.max_tilt_deg)
        assert np.isfinite(metrics.max_stress_proxy)
        assert np.isfinite(metrics.centroid_to_goal)
        assert metrics.max_tilt_deg >= 0.0
        assert metrics.max_stress_proxy >= 0.0
