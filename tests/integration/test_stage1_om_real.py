# SPDX-License-Identifier: Apache-2.0
"""Tier-2 real-env tests for the Stage-1 OM adapter (T5b.2; plan/07 §3).

Drives ``chamber.benchmarks.stage1_om.run_axis`` through the
production code path — the canonical pre-registration loader + the
default env factory + the zero-ego placeholder. The default env
factory currently returns
:class:`chamber.envs.mpe_cooperative_push.MPECooperativePushEnv`
(CPU-only; SAPIEN-free) per the Phase-0 stand-in scoping documented
in :mod:`chamber.benchmarks.stage1_om`. The Tier-2 test is therefore
marked ``@pytest.mark.slow`` rather than ``@pytest.mark.gpu`` — the
SAPIEN gate moves in when Phase-1 swaps the env factory for the real
Stage-1 obs-modality factory.

Mirrors :mod:`tests.integration.test_stage1_as_real`.
"""

from __future__ import annotations

import argparse

import pytest

from chamber.benchmarks.stage1_om import run_axis
from chamber.evaluation.results import SpikeRun

# TODO(plan/07 §T5b.2 Phase-1): add @pytest.mark.gpu +
# sapien_gpu_available() guards once _default_env_factory returns a
# SAPIEN env. Until then the Phase-0 MPE stand-in is CPU-only and the
# slow marker is the right gate.
pytestmark = pytest.mark.slow


def test_run_axis_smoke_on_real_mpe_factory() -> None:
    """plan/07 §3 + plan/07 §T5b.2: run_axis runs end-to-end on the production env factory.

    Calls into ``run_axis`` with the canonical pre-registration loader
    and the default ``_default_env_factory`` (MPE-backed). Asserts the
    returned :class:`SpikeRun` carries the expected sample size and
    axis label; does NOT assert a specific gap or success rate (the
    Phase-0 stand-in env has no OM-axis signal — the maintainer's
    real-spike launch will use the real Stage-1 obs-modality factory).
    """
    run = run_axis(argparse.Namespace(axis="OM"))
    assert isinstance(run, SpikeRun)
    assert run.axis == "OM"
    # plan/07 §2 sample-size contract: 5 seeds x 20 episodes x 2 conditions.
    assert len(run.episode_results) == 200
    for ep in run.episode_results:
        assert ep.metadata.get("condition") in {
            run.condition_pair.homogeneous_id,
            run.condition_pair.heterogeneous_id,
        }
