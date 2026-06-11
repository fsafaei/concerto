# SPDX-License-Identifier: Apache-2.0
"""Tier-2 SAPIEN-gated tests for the vectorised Stage-1b cell (P1.05.10).

ADR-007 §Stage 1b regime-alignment revision: real GPU-parallel
(``physx_cuda``) construction of :class:`Stage1PickPlaceEnv` at
``num_envs > 1``.

Subprocess isolation (load-bearing): SAPIEN's GPU PhysX backend can only
be enabled **before any other PhysX use in the process**, and this test
session's own GPU gate (:func:`chamber.utils.device.sapien_gpu_available`)
constructs a CPU-PhysX probe env at collection time — so a GPU-parallel
env can never be built inside the pytest process itself. Each check
therefore runs ``sys.executable -c <script>`` in a fresh subprocess and
asserts on the JSON the script prints. The same constraint applies to
production launches: one vectorised cell per process (the
``chamber-spike`` CLI already runs one cell per invocation, so this is
not a new operational constraint).

Covered:

1. **Batched obs/reward/flag shapes** through the full wrapper chain
   (AS synthesizer state ``(N, 65)``; reward / terminated / truncated
   ``(N,)``), auto-reset at the horizon with ``info["final_observation"]``,
   and the ``build_agent_snapshots`` loud-fail at ``num_envs > 1``
   (ADR-004 snapshot contract; the vectorised cell runs with the
   ``safety.enabled=false`` operator override).
2. **Per-env RNG reproducibility (P6 / ADR-002)** — two *separate
   processes* with the same root_seed produce identical per-env cube
   spawns (a stronger cross-process replication of the determinism
   contract), and the per-env streams are mutually independent.

The Tier-1 counterpart (fakes, no SAPIEN) is
:mod:`tests.unit.test_stage1_vectorised_cell`.
"""

from __future__ import annotations

import json
import subprocess
import sys

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


_STRUCTURAL_PROBE = r"""
import json
import numpy as np
import torch
from chamber.envs.stage1_pickplace import make_stage1_pickplace_env

N = 4
HORIZON = 4
env = make_stage1_pickplace_env(
    condition_id="stage1_pickplace_panda_plus_fetch_ego_aht_happo_per_agent",
    episode_length=HORIZON,
    root_seed=0,
    num_envs=N,
)
out = {}
obs, _ = env.reset(seed=0)
state = np.asarray(obs["agent"]["panda_wristcam"]["state"].detach().cpu()
                   if hasattr(obs["agent"]["panda_wristcam"]["state"], "detach")
                   else obs["agent"]["panda_wristcam"]["state"])
out["state_shape"] = list(state.shape)

def zero_actions():
    return {
        uid: torch.zeros((N, int(space.shape[-1])), device=env.device)
        for uid, space in env.action_space.spaces.items()
    }

trunc_steps = []
final_obs_at = []
for step in range(2 * HORIZON):
    obs2, reward, terminated, truncated, info = env.step(zero_actions())
    trunc = np.asarray(truncated.detach().cpu()).reshape(-1)
    if trunc.any():
        trunc_steps.append(step)
        if "final_observation" in info:
            final_obs_at.append(step)
def to_np(x):
    return np.asarray(x.detach().cpu() if hasattr(x, "detach") else x)

out["reward_shape"] = list(to_np(reward).reshape(-1).shape)
out["state2_shape"] = list(to_np(obs2["agent"]["panda_wristcam"]["state"]).shape)
out["trunc_steps"] = trunc_steps
out["final_obs_at"] = final_obs_at
try:
    env.build_agent_snapshots()
    out["snapshots_raised"] = False
except RuntimeError:
    out["snapshots_raised"] = True
env.close()
print(json.dumps(out))
"""

_RNG_PROBE = r"""
import json
import numpy as np
from chamber.envs.stage1_pickplace import make_stage1_pickplace_env

env = make_stage1_pickplace_env(
    condition_id="stage1_pickplace_panda_plus_fetch_ego_aht_happo_per_agent",
    episode_length=8,
    root_seed=7,
    num_envs=4,
)
env.reset(seed=7)
cubes = np.asarray(env.cube.pose.p.detach().cpu(), dtype=np.float64)
env.close()
print(json.dumps({"cubes": cubes.tolist()}))
"""


_DISPATCH_ORDER_PROBE = r"""
import json
from pathlib import Path

import numpy as np

from chamber.benchmarks.stage1_common import TrainedPolicyFactory
from concerto.training.config import load_config

CID = "stage1_pickplace_panda_plus_fetch_ego_aht_happo_per_agent"
cfg = load_config(
    config_path=Path("configs/training/ego_aht_happo/stage1_pickplace.yaml"),
    overrides=[
        "env.num_envs=16",
        "total_frames=512",
        "checkpoint_every=512",
        "happo.rollout_length=16",
        "happo.batch_size=256",
        "safety.enabled=false",
        "artifacts_root=/tmp/dispatch_order_probe/artifacts",
        "log_dir=/tmp/dispatch_order_probe/logs",
    ],
)
# PRODUCTION ADAPTER ORDER (the issue #215 regression surface):
# 1. factory constructed first (the adapter evaluates the
#    ego_action_factory argument before its cells loop) — this is
#    where GPU PhysX must get enabled;
factory = TrainedPolicyFactory(cfg=cfg)
# 2. per-cell CPU-sim eval env built BEFORE the factory call;
from chamber.envs.stage1_pickplace import make_stage1_pickplace_env

eval_env = make_stage1_pickplace_env(
    condition_id=CID, episode_length=10, root_seed=0, num_envs=1
)
# 3. factory invoked with the eval env — trains at num_envs=16 on
#    physx_cuda inside run_training (pre-fix: fatal here).
act = factory(eval_env, seed=0)
obs, _ = eval_env.reset(seed=0)
for _ in range(10):
    obs, _r, _t, _tr, _i = eval_env.step(
        {"panda_wristcam": act(obs), "fetch": np.zeros(13, dtype=np.float32)}
    )
eval_env.close()
print(json.dumps({"dispatch_order_ok": True}))
"""


class TestVectorisedConstruction:
    def test_production_dispatch_order_at_num_envs_gt_1(self) -> None:
        """Issue #215 regression net: eval-env-before-factory-call composes at N>1.

        Replicates the spike adapter's exact per-cell ordering
        (factory constructed first, CPU-sim eval env second, factory
        invoked third — training the vectorised cell inside) in a
        fresh subprocess. Pre-fix this died at the training-env build
        with "GPU PhysX can only be enabled once"; the fix enables GPU
        PhysX at factory construction (ADR-007 §Stage 1b Rev 17).
        """
        out = _run_probe(_DISPATCH_ORDER_PROBE)
        assert out["dispatch_order_ok"] is True

    def test_batched_shapes_autoreset_and_snapshot_guard(self) -> None:
        """One GPU build covers shapes, horizon auto-reset, and the safety guard."""
        out = _run_probe(_STRUCTURAL_PROBE)
        n, horizon = 4, 4
        assert out["state_shape"][0] == n
        assert len(out["state_shape"]) == 2
        assert out["reward_shape"] == [n]
        assert out["state2_shape"] == out["state_shape"]
        # Truncation fires exactly at the horizon (steps are 0-indexed),
        # then again one full horizon later — proving the partial reset
        # actually restarted the episode clock.
        assert out["trunc_steps"] == [horizon - 1, 2 * horizon - 1]
        assert out["final_obs_at"] == out["trunc_steps"]
        assert out["snapshots_raised"] is True

    def test_per_env_rng_reproducible_across_processes_and_independent(self) -> None:
        """P6 / ADR-002: same root_seed in two processes → identical per-env spawns."""
        cubes_a = np.asarray(_run_probe(_RNG_PROBE)["cubes"])
        cubes_b = np.asarray(_run_probe(_RNG_PROBE)["cubes"])
        np.testing.assert_allclose(cubes_a, cubes_b)
        # Independence: the per-env draws are not all identical.
        assert not np.allclose(cubes_a[0], cubes_a[1:])
