# SPDX-License-Identifier: Apache-2.0
"""Env-integration test for :class:`chamber.observability.RolloutRecorder` (P1.05.11; ADR-017).

This module ships the regression pin that exercises the full recorder
pipeline against a real :class:`chamber.envs.stage1_pickplace.Stage1PickPlaceEnv`
with ``render_mode="rgb_array"`` — the missing piece the synthetic-frame
unit tests at ``tests/unit/observability/test_rollout_recorder.py``
cannot cover (they use a NumPy uint8 RNG instead of
:meth:`env.render`).

xfail-strict gating
-------------------

This test is marked ``@pytest.mark.xfail(strict=True)`` because the
author's host had an out-of-sync ``nvidia.ko`` kernel module / userspace
``libnvidia-ml.so`` version mismatch at slice-write time (Stage 0 probe
2 on 2026-05-22); ManiSkill's SAPIEN renderer fell back from
``sapien_cuda`` to CPU and :meth:`env.render` returned ``None`` (warning
text: "Requested to use render device 'sapien_cuda', but CUDA device
was not found. Falling back to 'cpu' device. Rendering might be
disabled."). The wrapper chain itself is intact — env reset returns
valid obs — so the recorder's structural correctness is exercised by
the synthetic-frame unit tests on every ``make verify``.

When the host driver/library mismatch resolves (typically a reboot),
this test flips from xfail to xpass; the **marker must be removed in
the same commit that flips it**, mirroring the two precedents this
slice's ADR-017 §Decisions cites verbatim:

1. **P1.02 ``Bounds.action_norm`` inconsistency.** ADR-004 §Revision
   history 2026-05-17 added ``tests/property/test_bounds_action_norm_inconsistency_documented.py``
   as ``@pytest.mark.xfail(strict=True)``; the marker was removed in
   the same P1.02 commit (2026-05-17) that split
   ``Bounds.action_norm`` into ``action_linf_component`` and
   ``cartesian_accel_capacity``. The xfail-strict discipline forced
   the remediation PR to close the regression-pin lifecycle cleanly.

2. **P1.05.8 trainer obs reader contract.** ADR-002 §Revision history
   2026-05-21 added ``tests/integration/test_trainer_obs_reader_contract.py``
   as ``@pytest.mark.xfail(strict=True)`` so the contract xfailed on
   pre-P1.05.8 ``main`` and xpassed when the Surface 6 remediation
   landed; PR #185 removed the marker in the same commit as the
   :class:`Stage1ASStateSynthesizer` widening (the contract under
   test became met; the assertion became a positive regression net).

Closure path: either (a) folded into the same P1.05.11 PR if the host
reboot lands before merge; or (b) a one-line follow-up commit
``test(observability): remove env-render xfail post-host-reboot`` on
a tiny PR off ``main``. The grep for ``"P1.05.11"`` + ``"xfail"`` in
this module's docstring locates the marker for the maintainer who
flips it.

This test is gated by ``pytest.mark.gpu`` because real SAPIEN render
requires a working CUDA + Vulkan stack. CI without GPU skips it; the
xfail-strict applies only when the test actually runs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from chamber.observability import RolloutRecorder

if TYPE_CHECKING:  # pragma: no cover
    from pathlib import Path


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.xfail(
        strict=True,
        reason=(
            "P1.05.11 / ADR-017: env-render-through-recorder gated on host nvidia "
            "driver/library version mismatch. See module docstring for the "
            "P1.02 + P1.05.8 xfail-strict precedents and the marker-removal "
            "lifecycle. Resolution path: host reboot. Marker must be removed "
            "in the same commit that flips this from xfail to xpass."
        ),
    ),
]


def test_recorder_against_real_env_render(tmp_path: Path) -> None:
    """Exercise the full recorder pipeline against a real ``render_mode='rgb_array'`` env.

    Constructs ``make_stage1_pickplace_env(..., render_mode='rgb_array')``,
    drives a short eval episode through :class:`RolloutRecorder`, and
    asserts:

    1. ``env.render()`` returns a ``(H, W, 3)`` ``np.uint8`` ndarray
       (rendering is functional on this host).
    2. The recorder emits a non-empty MP4 (encoder produced bytes
       from the captured frames).
    3. The per-step sidecar JSONL has one line per step with the
       expected ``event="rollout_step"`` shape.
    """
    from chamber.envs.stage1_pickplace import make_stage1_pickplace_env

    env = make_stage1_pickplace_env(
        condition_id="stage1_pickplace_panda_plus_fetch_ego_aht_happo_per_agent",
        episode_length=10,
        root_seed=0,
        render_mode="rgb_array",
    )
    obs, _ = env.reset(seed=0)

    # Probe rendering once before instantiating the recorder so a clean
    # AssertionError fires when rendering is the problem (rather than a
    # downstream symptom in the recorder).
    frame_probe = env.render()
    assert isinstance(frame_probe, np.ndarray), (
        f"env.render() returned {type(frame_probe).__name__}, not np.ndarray. "
        "If this assertion fires (xfail-strict), the host driver/library "
        "mismatch is the cause; see module docstring."
    )
    # RGB frame contract: 3 dims, 3 channels in the last axis, uint8.
    assert frame_probe.ndim == 3, frame_probe.shape
    assert frame_probe.shape[-1] == 3, frame_probe.shape
    assert frame_probe.dtype == np.uint8

    recorder = RolloutRecorder(
        archive_dir=tmp_path,
        condition="as-hetero",
        global_step=0,
        fps=20,
        ego_uid="panda_wristcam",
        partner_uid="fetch",
    )
    recorder.begin_episode(seed=0)

    for _ in range(3):
        # Use a deterministic zero action so the test does not depend on
        # a trained policy. The recorder cares about the structural
        # pipeline, not policy quality.
        ego_action = np.zeros(8, dtype=np.float32)
        partner_action = np.zeros(13, dtype=np.float32)
        action = {"panda_wristcam": ego_action, "fetch": partner_action}
        obs, reward, terminated, truncated, info = env.step(action)
        frame = env.render()
        recorder.record_step(
            obs,
            ego_action,
            reward=float(reward),
            info=info,
            terminated=bool(terminated),
            truncated=bool(truncated),
            frame=np.asarray(frame, dtype=np.uint8),
        )

    mp4_path, jsonl_path = recorder.finalise()
    assert mp4_path is not None
    assert mp4_path.exists()
    assert mp4_path.stat().st_size > 0
    assert jsonl_path is not None
    lines = jsonl_path.read_text().splitlines()
    assert len(lines) == 3

    env.close()
