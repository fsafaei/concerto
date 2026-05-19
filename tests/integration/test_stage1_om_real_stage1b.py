# SPDX-License-Identifier: Apache-2.0
"""Tier-2 SAPIEN-gated tests for the Stage-1b dispatch path on the OM adapter (P1.05).

Twin of :mod:`tests.integration.test_stage1_as_real_stage1b`. The full
Stage-1b science launch is out-of-scope here; founder runs the launcher
``scripts/repro/stage1_om_stage1b.sh``. This Tier-2 test pins only the
end-to-end dispatch smoke (1 seed x 1 episode x 2 conditions).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from chamber.benchmarks.stage1_om import run_axis
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
    """Shrink the prereg sample sizes for the Tier-2 dispatch smoke."""
    spec = load_prereg(_REPO_ROOT / "spikes" / "preregistration" / "OM.yaml")
    tiny = spec.model_copy(update={"seeds": [0], "episodes_per_seed": 1})

    def _fake_load(*args, **kwargs):  # type: ignore[no-untyped-def]
        del args, kwargs
        return tiny, _REPO_ROOT / "spikes" / "preregistration" / "OM.yaml"

    monkeypatch.setattr(
        "chamber.benchmarks.stage1_om._load_canonical_prereg",
        _fake_load,
    )


class TestStage1OMStage1bDispatchSmoke:
    """The dispatch path runs end-to-end on real SAPIEN env + TrainedPolicyFactory."""

    def test_stage1b_dispatch_completes_end_to_end(self) -> None:
        args = argparse.Namespace(axis="OM", sub_stage="1b")
        run = run_axis(args)
        assert isinstance(run, SpikeRun)
        assert run.sub_stage == "1b"
        assert run.axis == "OM"
        assert len(run.episode_results) == 2

    def test_stage1b_dispatch_records_prereg_sha(self) -> None:
        args = argparse.Namespace(axis="OM", sub_stage="1b")
        run = run_axis(args)
        assert len(run.prereg_sha) == 40
