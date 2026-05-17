# SPDX-License-Identifier: Apache-2.0
"""Tier-2 real-env tests for the Stage-1 AS adapter (T5b.2; plan/07 §3).

Drives ``chamber.benchmarks.stage1_as.run_axis`` through the
production code path — the canonical pre-registration loader + the
default env factory + the zero-ego placeholder. The default env
factory currently returns
:class:`chamber.envs.mpe_cooperative_push.MPECooperativePushEnv`
(CPU-only; SAPIEN-free) per the Phase-0 stand-in scoping documented
in :mod:`chamber.benchmarks.stage1_as`. The Tier-2 test is therefore
marked ``@pytest.mark.slow`` rather than ``@pytest.mark.gpu`` — the
SAPIEN gate moves in when Phase-1 swaps the env factory for the real
Stage-1 pick-place env.

The slow path runs the full 5 seeds x 20 episodes x 2 conditions x 50
steps = 20_000 env steps; on a Mac that takes a few seconds. The
test is intentionally a "does the whole pipe survive a real
end-to-end run" smoke rather than a science assertion.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from chamber.benchmarks.stage1_as import run_axis
from chamber.evaluation.prereg import load_prereg, verify_git_tag
from chamber.evaluation.results import SpikeRun

# TODO(plan/07 §T5b.2 Phase-1): add @pytest.mark.gpu +
# sapien_gpu_available() guards once _default_env_factory returns a
# SAPIEN env. Until then the Phase-0 MPE stand-in is CPU-only and the
# slow marker is the right gate.
pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def run() -> SpikeRun:
    """Drive ``run_axis`` once per module; share across the assertions below.

    The full 200-episode roll-out takes a few seconds; the class-level
    fixture amortises that across the smoke + prereg-discipline tests.
    """
    return run_axis(argparse.Namespace(axis="AS"))


class TestStage1ASSmoke:
    """Smoke contract: production env factory + canonical prereg loader (plan/07 §T5b.2)."""

    def test_run_axis_smoke_on_real_mpe_factory(self, run: SpikeRun) -> None:
        """plan/07 §3 + plan/07 §T5b.2: run_axis runs end-to-end on the production env factory.

        Calls into ``run_axis`` with the canonical pre-registration
        loader and the default ``_default_env_factory`` (MPE-backed).
        Asserts the returned :class:`SpikeRun` carries the expected
        sample size and axis label; does NOT assert a specific gap or
        success rate (the Phase-0 stand-in env has no AS-axis signal —
        the maintainer's real-spike launch will use the real Stage-1
        pick-place env).
        """
        assert isinstance(run, SpikeRun)
        assert run.axis == "AS"
        # plan/07 §2 sample-size contract: 5 seeds x 20 episodes x 2 conditions.
        assert len(run.episode_results) == 200
        for ep in run.episode_results:
            assert ep.metadata.get("condition") in {
                run.condition_pair.homogeneous_id,
                run.condition_pair.heterogeneous_id,
            }


class TestStage1ASPreregDiscipline:
    """ADR-007 §Discipline: every SpikeRun MUST carry the verified prereg blob SHA.

    Empty ``prereg_sha`` was the 2026-05-17 root cause of the audit-trail
    defect in ``spikes/results/stage1-{AS,OM}-20260517/`` (the Stage-1
    adapter recorded ``prereg_sha=""`` with a "filled in by the
    launch-time chamber-spike verify-prereg step" comment, but that
    step never backfilled the SpikeRun on disk). The audit chain does
    not close until the adapter calls
    :func:`chamber.evaluation.prereg.verify_git_tag` itself and threads
    the returned blob SHA into ``SpikeRun(prereg_sha=...)``.
    """

    def test_run_axis_records_prereg_sha_in_spike_run(self, run: SpikeRun) -> None:
        """Per ADR-007 §Discipline: produced SpikeRun carries the verified blob SHA.

        Resolves the canonical AS pre-registration relative to
        ``Path.cwd()`` (the same convention the adapter uses), calls
        :func:`verify_git_tag` directly to obtain the ground-truth blob
        SHA, and asserts that ``run.prereg_sha`` is non-empty AND
        matches verbatim.
        """
        prereg_path = Path.cwd() / "spikes" / "preregistration" / "AS.yaml"
        spec = load_prereg(prereg_path)
        expected_sha = verify_git_tag(spec, prereg_path, repo_path=Path.cwd())
        # 40-char hex (SHA-1) per ``git hash-object``.
        assert len(expected_sha) == 40
        assert run.prereg_sha == expected_sha, (
            f"ADR-007 §Discipline violation: SpikeRun.prereg_sha "
            f"{run.prereg_sha!r} does not match the verified blob "
            f"SHA {expected_sha!r} for the tagged YAML at "
            f"{spec.git_tag!r}. The audit chain does not close."
        )

    def test_run_axis_records_sub_stage_1a(self, run: SpikeRun) -> None:
        """Per ADR-016 §Decision: Stage-1a adapter stamps ``sub_stage="1a"``.

        The production AS adapter runs against the Phase-0 MPE stand-in
        (no real ≥20 pp gate measurement; see ADR-007 §Stage 1a). The
        produced SpikeRun MUST carry ``sub_stage="1a"`` so the
        :mod:`chamber.cli._spike_summarize_month3` routing short-
        circuits the four-state logic and emits ``Defer — Stage 1b not
        yet measured`` rather than treating the null result as a
        structural failure (the 2026-05-17 incident root cause that
        PR 2 closed).
        """
        assert run.sub_stage == "1a", (
            f"Stage-1a adapter regression: produced SpikeRun.sub_stage "
            f"is {run.sub_stage!r}, expected '1a'. The summarizer "
            "will mis-route to Stop instead of Defer (ADR-007 §Stage 1a)."
        )
