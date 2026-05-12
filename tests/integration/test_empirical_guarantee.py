# SPDX-License-Identifier: Apache-2.0
"""Integration test for the ADR-002 risk-mitigation #1 trip-wire (T4b.13).

Runs the 100k-frame ego-AHT HAPPO experiment end-to-end against the
production ``mpe_cooperative_push.yaml`` config and asserts the
per-episode reward has a statistically significant positive learning
slope (plan/05 §6 criterion 4; one-sided p < :data:`DEFAULT_ALPHA`).

The test wallclock is bounded at 30 minutes (plan/05 §8). If the run
exceeds the budget the test fails loudly — do NOT raise the budget or
relax the alpha to coerce a passing result. The trip-wire's whole
purpose is to surface a violated assumption (Huriot-Sibai 2025 Theorem
7's frozen-partner monotonicity), and silencing it would defeat ADR-002
§Risks #1.

**Canonical statistic (issue #62):** the slope test
:func:`assert_positive_learning_slope` replaced the original moving-
window-of-K non-decreasing-fraction (:func:`assert_non_decreasing_window`)
because the per-episode reward noise (stdev ~20) dominates the
moving-window mean's signal at 100k frames — the moving-window fraction
sat near 0.5 (random walk) even when the trainer was materially
improving. The slope test integrates over the whole curve and rejects
"slope <= 0" with p ≪ 1e-10 across seeds once the truncation fix in
:func:`chamber.benchmarks.ego_ppo_trainer.compute_gae` is in place.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from chamber.benchmarks.training_runner import run_training
from concerto.training.config import load_config
from concerto.training.empirical_guarantee import (
    DEFAULT_ALPHA,
    DEFAULT_MIN_EPISODES,
    assert_positive_learning_slope,
)

#: Wallclock budget for the 100k-frame run (plan/05 §8).
#:
#: 30 minutes is the hard ceiling. If the run goes over, do NOT widen
#: this constant — reduce task complexity instead (smaller hidden_dim,
#: shorter episode length) and document the change in the PR.
_MAX_WALLCLOCK_SECONDS: float = 30 * 60.0


@pytest.mark.slow
def test_empirical_guarantee_holds_on_100k_frames(tmp_path: Path) -> None:
    """Plan/05 §6 #4: 100k ego-AHT HAPPO frames clear the empirical-guarantee slope trip-wire."""
    repo_root = Path(__file__).resolve().parents[2]
    config_path = repo_root / "configs" / "training" / "ego_aht_happo" / "mpe_cooperative_push.yaml"
    cfg = load_config(
        config_path=config_path,
        overrides=[
            f"artifacts_root={tmp_path / 'artifacts'}",
            f"log_dir={tmp_path / 'logs'}",
        ],
    )
    assert cfg.total_frames == 100_000, (
        "Production config should run at 100k frames (plan/05 §6 #4). "
        "If you must override for a local sanity check, do it in a local "
        "script — don't lower the gate."
    )

    t0 = time.perf_counter()
    curve = run_training(cfg, repo_root=repo_root)
    elapsed = time.perf_counter() - t0

    assert elapsed < _MAX_WALLCLOCK_SECONDS, (
        f"Empirical-guarantee run took {elapsed:.1f}s, exceeding the "
        f"{_MAX_WALLCLOCK_SECONDS:.0f}s budget (plan/05 §8). Reduce task "
        "complexity rather than raising the budget."
    )

    report = assert_positive_learning_slope(
        curve, alpha=DEFAULT_ALPHA, min_episodes=DEFAULT_MIN_EPISODES
    )
    assert report.passed, (
        "ADR-002 risk-mitigation #1 trip-wire fired: empirical guarantee "
        "(slope test) failed. "
        f"slope={report.slope:.5f}, p_value={report.p_value:.4e}, "
        f"alpha={report.alpha}, n_episodes={report.n_episodes}. "
        "DO NOT silence by lowering alpha or shortening the budget; open "
        "a #scope-revision issue and hand control back to the maintainer "
        "(plan/05 §6 #4)."
    )
