# SPDX-License-Identifier: Apache-2.0
"""Tier-2 SAPIEN-gated tests for the Stage-1b dispatch path on the OM adapter (P1.05).

Twin of :mod:`tests.integration.test_stage1_as_real_stage1b`. The full
Stage-1b science launch is out-of-scope here; founder runs the launcher
``scripts/repro/stage1_om_stage1b.sh``. This Tier-2 test shrinks both
the prereg (1 seed x 1 episode x 2 conditions) and the cfg
(``total_frames=1000``, smaller rollout_length/batch_size) so the
smoke completes in ~2 min on the RTX 2080.

**Retired under ADR-026 §Decision 3/5.** OM is non-coupling-valid on the
ego-solvable Stage-1 pick-place task, so AS/OM promotion to Validated is
closed and the vision-head ``EgoPPOTrainer`` extension (formerly P1.05.6 /
issue #177) was never built and will not be. The technical shape of the
retired path is unchanged: the OM dispatch routes through
``EgoPPOTrainer.from_config``, which reads ``obs["agent"][ego_uid]["state"]``
(a flat 1-D Box). The OM wrapper chain
``Stage1OMChannelFilter(Stage1ASStateSynthesizer(inner))`` makes the AS
state synthesizer a pass-through when ``config.is_om_condition=True``
(``chamber.envs.stage1_obs_filter.Stage1ASStateSynthesizer.__init__``),
so OM envs have no ``state`` key — the trainer raises ``KeyError: 'state'``
at construction. A state-synthesis workaround stays wrong for the same
reason as before: silent state exposure under OM-homo would have corrupted
the OM-homo science measurement. Both Tier-2 cases stay
``pytest.mark.skip`` (not xfail); they remain only as retired-path
dispatch coverage, not pending work.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from chamber.benchmarks.stage1_om import run_axis
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
    pytest.mark.skip(
        reason=(
            "OM Stage-1b dispatch path retired under ADR-026 §Decision 3/5: "
            "OM is non-coupling-valid on the ego-solvable Stage-1 pick-place "
            "task and AS/OM promotion to Validated is closed, so the "
            "vision-head EgoPPOTrainer extension (formerly P1.05.6 / issue "
            "#177) was never built and will not be. The EgoPPOTrainer reads "
            "obs['agent'][uid]['state'] (a flat Box) and the OM wrapper chain "
            "intentionally does not synthesize that key, so the trainer raises "
            "KeyError at construction; state-synthesis would have silently "
            "corrupted the OM-homo science measurement. This smoke remains "
            "only as retired-path dispatch coverage, not pending work."
        ),
    ),
]

_REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _tiny_prereg_for_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shrink the prereg + cfg for the Tier-2 smoke (twin of the AS variant)."""
    spec = load_prereg(_REPO_ROOT / "spikes" / "preregistration" / "OM.yaml")
    tiny = spec.model_copy(update={"seeds": [0], "episodes_per_seed": 1})

    def _fake_load_prereg(*args, **kwargs):  # type: ignore[no-untyped-def]
        del args, kwargs
        return tiny, _REPO_ROOT / "spikes" / "preregistration" / "OM.yaml"

    def _shrunken_load_config(*args, **kwargs):  # type: ignore[no-untyped-def]
        cfg = _real_load_config(*args, **kwargs)
        return cfg.model_copy(
            update={
                "total_frames": 1000,
                "happo": cfg.happo.model_copy(update={"rollout_length": 256, "batch_size": 64}),
            }
        )

    monkeypatch.setattr(
        "chamber.benchmarks.stage1_om._load_canonical_prereg",
        _fake_load_prereg,
    )
    monkeypatch.setattr(
        "concerto.training.config.load_config",
        _shrunken_load_config,
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
