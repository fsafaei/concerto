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

This Tier-2 test shrinks the prereg's sample sizes (1 seed x 1 episode
x 2 conditions) so the smoke completes in a couple of minutes on the
RTX 2080 — the dispatch contract and the env build are what's
exercised, not the science gate.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from chamber.benchmarks.stage1_as import run_axis
from chamber.evaluation.prereg import load_prereg
from chamber.evaluation.results import SpikeRun
from chamber.utils.device import sapien_gpu_available

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
    """Shrink the prereg's seeds/episodes_per_seed for the Tier-2 smoke.

    The real spike runs 5 seeds x 20 episodes x 2 conditions ≈ 1M
    training frames per axis on RTX 2080; that's the founder's
    ``scripts/repro/stage1_as_stage1b.sh`` launch, not a unit test.
    For the Tier-2 dispatch smoke, override the prereg loader to
    return a 1-seed x 1-episode-per-seed x 2-condition spec.
    """
    spec = load_prereg(_REPO_ROOT / "spikes" / "preregistration" / "AS.yaml")
    tiny = spec.model_copy(update={"seeds": [0], "episodes_per_seed": 1})

    def _fake_load(*args, **kwargs):  # type: ignore[no-untyped-def]
        del args, kwargs
        return tiny, _REPO_ROOT / "spikes" / "preregistration" / "AS.yaml"

    monkeypatch.setattr(
        "chamber.benchmarks.stage1_as._load_canonical_prereg",
        _fake_load,
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
