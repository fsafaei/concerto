# SPDX-License-Identifier: Apache-2.0
"""Tier-2 SAPIEN-gated reset probe for the vectorised OM vision-only env (issue #231).

ADR-007 §Stage 1b Rev 17 (vectorised cells) x OM axis: real GPU-parallel
(``physx_cuda``) construction of the vision-only condition at
``num_envs > 1``. Pre-fix, :class:`chamber.envs.stage1_obs_filter.
Stage1OMChannelFilter` zero-masked via ``np.zeros_like``, which raises
``TypeError: can't convert cuda:0 device type tensor to numpy`` on
GPU-sim tensors — the env could not ``reset()`` at ``num_envs > 1``
(reproduced at N=64/128/256; ``OM_UNBLOCK_ASSESSMENT_2026-06-11.md``
§1.2, PR #229).

Subprocess isolation (load-bearing — same constraint as
:mod:`tests.integration.test_stage1_vectorised_real`): SAPIEN's GPU
PhysX backend can only be enabled before any other PhysX use in the
process, and the session's GPU gate constructs a CPU-PhysX probe env at
collection time, so the check runs in a fresh interpreter and asserts
on the JSON it prints.

The Tier-1 counterpart (CPU torch-tensor duck through the filter) is
``tests/unit/test_stage1_pickplace_tier1.py::TestStage1OMChannelFilter::
test_om_vision_only_masks_tensor_leaves_device_aware``.
"""

from __future__ import annotations

import json
import subprocess
import sys

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

_PROBE_TIMEOUT_S = 600


def _run_probe(script: str) -> dict:  # type: ignore[type-arg]
    """Run ``script`` in a fresh interpreter; parse the JSON on its last stdout line."""
    proc = subprocess.run(  # noqa: S603 - fixed interpreter, repo-authored script
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=_PROBE_TIMEOUT_S,
        check=False,
    )
    assert proc.returncode == 0, f"probe failed:\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    last = proc.stdout.strip().splitlines()[-1]
    return json.loads(last)


_OM_VISION_ONLY_RESET_PROBE = r"""
import json
import torch
from chamber.envs.stage1_pickplace import make_stage1_pickplace_env

N = 64
env = make_stage1_pickplace_env(
    condition_id="stage1_pickplace_vision_only",
    episode_length=4,
    root_seed=0,
    num_envs=N,
)
obs, _ = env.reset(seed=0)  # pre-#231-fix: TypeError inside the OM filter

def leaf_report(val):
    return {
        "is_tensor": isinstance(val, torch.Tensor),
        "device": str(val.device.type) if isinstance(val, torch.Tensor) else None,
        "all_zero": bool((val == 0).all()) if hasattr(val, "shape") else None,
        "batch": int(val.shape[0]) if getattr(val, "shape", ()) else None,
    }

out = {"env_device": str(env.device.type), "masked": {}, "kept": {}}
for uid, sub in obs["agent"].items():
    for ch, val in sub.items():
        out["masked"][f"agent.{uid}.{ch}"] = leaf_report(val)
for ch in ("force_torque", "cube_pose"):
    if ch in obs["extra"]:
        out["masked"][f"extra.{ch}"] = leaf_report(obs["extra"][ch])
for ch in ("tcp_pose", "goal_pos"):
    out["kept"][f"extra.{ch}"] = leaf_report(obs["extra"][ch])
env.close()
print(json.dumps(out))
"""


class TestOMVisionOnlyVectorisedReset:
    def test_reset_succeeds_and_masked_keys_are_zero_tensors_on_device(self) -> None:
        """Issue #231 regression net: vision-only OM reset at num_envs=64.

        The probe building + resetting at all is the headline assertion
        (pre-fix it dies with the CUDA->numpy TypeError); the JSON then
        pins that every masked leaf stayed a torch tensor on the env's
        device with the (num_envs, ...) batch intact and value zero, and
        that the keep-set (``tcp_pose`` / ``goal_pos``) was not masked.
        """
        out = _run_probe(_OM_VISION_ONLY_RESET_PROBE)
        assert out["masked"], "probe found no masked leaves to check"
        for key, report in out["masked"].items():
            assert report["all_zero"] is True, f"{key} not zero-masked"
            assert report["batch"] == 64, f"{key} lost the num_envs batch dim"
            if report["is_tensor"]:
                assert report["device"] == out["env_device"], f"{key} left the env device"
        # The agent-proprio leaves are GPU-sim emissions — these MUST
        # have stayed tensors (container follows the input; the
        # synthesised extra.force_torque is CPU-side numpy by
        # construction and correctly masks to an ndarray instead).
        agent_leaves = {k: r for k, r in out["masked"].items() if k.startswith("agent.")}
        assert agent_leaves, "probe found no agent-proprio leaves to check"
        for key, report in agent_leaves.items():
            assert report["is_tensor"], f"{key} lost its tensor container"
        for key, report in out["kept"].items():
            assert report["all_zero"] is False, f"keep-set {key} was wrongly masked"
