# SPDX-License-Identifier: Apache-2.0
"""Tier-2 SAPIEN-gated tests for the Stage-1b dispatch path on the AS adapter (P1.05).

End-to-end smoke that ``chamber.benchmarks.stage1_as.run_axis`` with
``args.sub_stage='1b'`` builds the real
:class:`chamber.envs.stage1_pickplace.Stage1PickPlaceEnv` env per
condition and routes through
:class:`chamber.benchmarks.stage1_common.TrainedPolicyFactory`. The
full Stage-1b science launch (5 seeds x 100k frames per axis) is
out-of-scope here — the founder runs that via
``scripts/repro/stage1_as_stage1b.sh`` on the RTX 2080 / A100 box and
captures the archive under
``spikes/results/stage1-AS-stage1b-<UTC-date>/`` (PR 5b operational
artefact).

This Tier-2 test shrinks **both** the prereg sample sizes (1 seed x 1
episode x 2 conditions) AND the cfg's ``total_frames`` (1k instead of
100k) so the smoke completes in a couple of minutes on the RTX 2080
rather than the ~80 min the full-budget cfg would take. The dispatch
contract and the env build are what's exercised, not the science
gate.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from chamber.benchmarks.stage1_as import run_axis
from chamber.evaluation.prereg import load_prereg
from chamber.evaluation.results import SpikeRun
from chamber.utils.device import sapien_gpu_available
from concerto.training.config import load_config as _real_load_config

pytestmark = [
    pytest.mark.slow,
    pytest.mark.gpu,
    pytest.mark.skipif(
        not sapien_gpu_available(),
        reason="Stage-1b dispatch requires SAPIEN/Vulkan (the real ManiSkill pick-place env)",
    ),
]

_REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _tiny_prereg_for_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shrink the prereg + cfg for the Tier-2 smoke.

    Two shrinks happen here:

    1. ``_load_canonical_prereg`` is patched to return a 1-seed x
       1-episode-per-seed x 2-condition spec (so the adapter loop
       runs 2 (seed, condition) cells, each building one env + one
       trained policy).
    2. ``load_config`` is patched to wrap the real loader and rewrite
       ``total_frames``, ``rollout_length``, ``batch_size`` to tiny
       values so each cell's training run completes in ~30 s instead
       of the production ~25 min. The other cfg fields (env.task,
       env.condition_id, partner, safety) are honoured verbatim — the
       smoke is about the dispatch contract, not the gradient signal.
    """
    spec = load_prereg(_REPO_ROOT / "spikes" / "preregistration" / "AS.yaml")
    tiny = spec.model_copy(update={"seeds": [0], "episodes_per_seed": 1})

    def _fake_load_prereg(*args, **kwargs):  # type: ignore[no-untyped-def]
        del args, kwargs
        return tiny, _REPO_ROOT / "spikes" / "preregistration" / "AS.yaml"

    def _shrunken_load_config(*args, **kwargs):  # type: ignore[no-untyped-def]
        cfg = _real_load_config(*args, **kwargs)
        # Tier-2 smoke budget: 1k frames is enough to cover one PPO
        # update at rollout_length=256 + a handful of safety_telemetry
        # window flushes. Production runs at 100k per ADR-007 §Stage 1b.
        return cfg.model_copy(
            update={
                "total_frames": 1000,
                "happo": cfg.happo.model_copy(
                    update={"rollout_length": 256, "batch_size": 64}
                ),
            }
        )

    monkeypatch.setattr(
        "chamber.benchmarks.stage1_as._load_canonical_prereg",
        _fake_load_prereg,
    )
    monkeypatch.setattr(
        "concerto.training.config.load_config",
        _shrunken_load_config,
    )


class TestStage1ASStage1bDispatchSmoke:
    """The dispatch path runs end-to-end on real SAPIEN env + TrainedPolicyFactory."""

    def test_stage1b_dispatch_completes_end_to_end(self) -> None:
        """``run_axis`` with ``args.sub_stage='1b'`` returns a sub_stage=1b SpikeRun."""
        args = argparse.Namespace(axis="AS", sub_stage="1b")
        run = run_axis(args)
        assert isinstance(run, SpikeRun)
        assert run.sub_stage == "1b"
        assert run.axis == "AS"
        # 1 seed x 1 episode x 2 conditions = 2 episodes total in the smoke.
        assert len(run.episode_results) == 2

    def test_stage1b_dispatch_records_prereg_sha(self) -> None:
        """ADR-007 §Discipline: the prereg blob SHA is verified + recorded under sub_stage='1b'."""
        args = argparse.Namespace(axis="AS", sub_stage="1b")
        run = run_axis(args)
        assert len(run.prereg_sha) == 40
