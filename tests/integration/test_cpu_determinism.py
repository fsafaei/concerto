# SPDX-License-Identifier: Apache-2.0
"""CPU byte-identicality integration test (plan/05 §6 #7; plan/08 §4).

Trip-wire for the project's determinism harness (P6):

- Two CPU training runs with the same ``cfg.seed`` MUST produce
  byte-identical per-step reward sequences.
- Two CPU training runs with different ``cfg.seed`` values MUST diverge.
- Checkpoints emitted by two same-seed runs MUST carry matching
  SHA-256 digests in their sidecars (the ``.pt`` payload bytes are a
  deterministic function of the trainer's state-dict, which is itself
  a deterministic function of ``cfg.seed`` under the harness).

If this test fails the most likely cause is a raw ``np.random``,
``torch.randn``, or ``torch.manual_seed`` call introduced somewhere in
the trainer or env stack that bypasses
:func:`concerto.training.seeding.derive_substream`. Trace + fix the
bypass; do NOT relax this test with ``pytest.approx`` or per-element
tolerance — the contract is bit-identical, not numerically close.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from chamber.benchmarks.training_runner import run_training
from concerto.training.checkpoints import load_checkpoint
from concerto.training.config import load_config

#: Per-run frame budget. Small enough for fast CI but long enough that
#: the rollout buffer cycles through update() at least twice
#: (``total_frames // happo.rollout_length`` cycles).
_TOTAL_FRAMES: int = 2000

#: Checkpoint cadence in frames. With ``_TOTAL_FRAMES = 2000`` this fires
#: twice per run (at step 1000 and step 2000), giving two checkpoint
#: matches per same-seed pair.
_CHECKPOINT_EVERY: int = 1000

#: Rollout length in frames. Smaller than the default 1024 so the test
#: gets 4 update cycles per run (vs ~2 with the production default at
#: this frame budget); exercises minibatch RNG determinism more
#: thoroughly without bloating wall-time.
_ROLLOUT_LENGTH: int = 500


def _run_once(*, seed: int, run_dir: Path, repo_root: Path):  # type: ignore[no-untyped-def]
    """Run one ego-AHT training pass under the harness; return the curve."""
    config_path = repo_root / "configs" / "training" / "ego_aht_happo" / "mpe_cooperative_push.yaml"
    cfg = load_config(
        config_path=config_path,
        overrides=[
            f"seed={seed}",
            f"total_frames={_TOTAL_FRAMES}",
            f"checkpoint_every={_CHECKPOINT_EVERY}",
            f"happo.rollout_length={_ROLLOUT_LENGTH}",
            f"artifacts_root={run_dir / 'artifacts'}",
            f"log_dir={run_dir / 'logs'}",
        ],
    )
    return cfg, run_training(cfg, repo_root=repo_root)


@pytest.mark.slow
def test_cpu_determinism_same_seed_byte_identical(tmp_path: Path) -> None:
    """plan/05 §6 #7: two CPU runs at the same seed produce byte-identical curves.

    Also asserts that the seed-0 vs seed-1 pair diverges, and that the
    two seed-0 runs' matched checkpoints carry equal SHA-256 digests on
    their sidecars.
    """
    repo_root = Path(__file__).resolve().parents[2]

    cfg_a, curve_a = _run_once(seed=0, run_dir=tmp_path / "seed0_a", repo_root=repo_root)
    cfg_b, curve_b = _run_once(seed=0, run_dir=tmp_path / "seed0_b", repo_root=repo_root)
    _cfg_c, curve_c = _run_once(seed=1, run_dir=tmp_path / "seed1", repo_root=repo_root)

    # Same-seed: bit-identical per-step reward arrays. np.array_equal on
    # float32 is bit-comparison, not tolerance-comparison — exactly what
    # the harness contract requires (P6; plan/08 §4).
    rewards_a = np.asarray(curve_a.per_step_ego_rewards, dtype=np.float32)
    rewards_b = np.asarray(curve_b.per_step_ego_rewards, dtype=np.float32)
    rewards_c = np.asarray(curve_c.per_step_ego_rewards, dtype=np.float32)
    assert rewards_a.shape == (_TOTAL_FRAMES,)
    assert np.array_equal(rewards_a, rewards_b), (
        "Two CPU training runs at seed=0 produced different per-step reward "
        "sequences. This is a determinism-harness regression (P6 / plan/05 §6 #7): "
        "trace the divergence to the first call site that uses np.random / "
        "torch.randn / torch.manual_seed outside derive_substream. DO NOT relax "
        "this assertion with pytest.approx — the contract is bit-identical."
    )

    # Different-seed: the curves must diverge somewhere. Otherwise the
    # trainer is ignoring the seed parameter entirely (a different but
    # equally bad bug).
    assert not np.array_equal(rewards_a, rewards_c), (
        "seed=0 and seed=1 produced byte-identical per-step rewards — the "
        "trainer or env is ignoring the seed parameter (plan/05 §6 #7)."
    )

    # Same-seed runs produce the same run_id (derived from
    # ``(seed, git_sha, pyproject_hash, run_kind)`` in compute_run_metadata),
    # so checkpoint relative URIs match and the test can pair them by step.
    assert curve_a.run_id == curve_b.run_id, (
        "Two seed=0 runs produced different run_ids; run_id derivation "
        "must be a deterministic function of (seed, git_sha, "
        "pyproject_hash, run_kind) per concerto.training.logging."
    )
    assert curve_a.run_id != curve_c.run_id, (
        "seed=0 and seed=1 produced the same run_id; run_id should fold the seed into its hash."
    )

    # Pair checkpoints by step and assert sidecar SHA-256 equality. With
    # ``_TOTAL_FRAMES = 2000`` and ``_CHECKPOINT_EVERY = 1000`` the loop
    # emits checkpoints at step 1000 and step 2000.
    assert len(curve_a.checkpoint_paths) == _TOTAL_FRAMES // _CHECKPOINT_EVERY
    assert len(curve_a.checkpoint_paths) == len(curve_b.checkpoint_paths)
    for path_a, path_b in zip(curve_a.checkpoint_paths, curve_b.checkpoint_paths, strict=True):
        # Resolve back to URIs and load via the same path the M4a partner
        # adapters will use in Phase 3 (plan/04 §3.8).
        uri_a = f"local://{path_a.relative_to(cfg_a.artifacts_root)}"
        uri_b = f"local://{path_b.relative_to(cfg_b.artifacts_root)}"
        _, meta_a = load_checkpoint(uri=uri_a, artifacts_root=cfg_a.artifacts_root)
        _, meta_b = load_checkpoint(uri=uri_b, artifacts_root=cfg_b.artifacts_root)
        assert meta_a.step == meta_b.step
        assert meta_a.sha256 == meta_b.sha256, (
            f"Checkpoint at step {meta_a.step}: seed=0 run A SHA-256 "
            f"{meta_a.sha256!r} does not match seed=0 run B SHA-256 "
            f"{meta_b.sha256!r}. Bit-identical state dicts must hash to "
            "the same digest — a mismatch points to non-determinism in "
            "the trainer's state, not in the integrity check itself."
        )
