# SPDX-License-Identifier: Apache-2.0
"""Unit tests for :class:`chamber.observability.RolloutRecorder` (P1.05.11; ADR-017 §Decisions).

These tests use a **synthetic frame source** (NumPy uint8 RNG) — they
do NOT call :meth:`env.render`. Per the Stage 0 probe 2 outcome
(host NVIDIA driver/library mismatch), the env-integration test that
does real ``env.render(mode="rgb_array")`` lives at
``tests/integration/test_rollout_recorder_env_render.py`` with a single
``@pytest.mark.xfail(strict=True)`` marker. These unit tests are NOT
xfail and validate the encoder + per-step JSONL + finalisation
end-to-end today.

xfail-strict precedents this slice mirrors:

- P1.02 ``Bounds.action_norm`` inconsistency
  (``tests/property/test_bounds_action_norm_inconsistency_documented.py``;
  ADR-004 §Revision history 2026-05-17, resolved 2026-05-17 / P1.02).
- P1.05.8 trainer obs reader contract
  (``tests/integration/test_trainer_obs_reader_contract.py:42-48``;
  ADR-002 §Revision history 2026-05-21, resolved by PR #185).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

from chamber.observability import RolloutRecorder

if TYPE_CHECKING:  # pragma: no cover
    from pathlib import Path


@pytest.fixture
def archive_dir(tmp_path: Path) -> Path:
    """A throwaway archive directory; the recorder creates `rollouts/...` under it."""
    return tmp_path


def _random_frame(rng: np.random.Generator, *, h: int = 84, w: int = 84) -> np.ndarray:
    """Synthetic (H, W, 3) uint8 frame for encoder smoke tests."""
    return rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)


def _minimal_stage1_obs() -> dict[str, object]:
    """A minimal Stage-1-shaped obs dict the recorder's `obs_summary` selector exercises."""
    return {
        "extra": {
            "cube_pose": np.array([1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            "goal_pos": np.array([0.0, 0.5, 0.5], dtype=np.float32),
            "tcp_pose": np.array([0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        },
        "agent": {
            "panda_wristcam": {
                "qpos": np.zeros(8, dtype=np.float32),
                "qvel": np.zeros(8, dtype=np.float32),
            },
            "fetch": {
                "qpos": np.zeros(13, dtype=np.float32),
                "qvel": np.zeros(13, dtype=np.float32),
            },
        },
    }


class TestObsSummary:
    """Curated obs_summary selector — non-Stage-1 obs shapes don't crash."""

    def test_summary_contains_cube_goal_tcp(self, archive_dir: Path) -> None:
        rng = np.random.default_rng(0)
        recorder = RolloutRecorder(
            archive_dir=archive_dir,
            condition="as-hetero",
            global_step=1000,
            ego_uid="panda_wristcam",
            partner_uid="fetch",
        )
        recorder.begin_episode(seed=0)
        recorder.record_step(
            _minimal_stage1_obs(),
            action=np.zeros(8, dtype=np.float32),
            reward=0.0,
            info={},
            terminated=False,
            truncated=False,
            frame=_random_frame(rng),
        )
        _, jsonl_path = recorder.finalise()
        assert jsonl_path is not None
        line = json.loads(jsonl_path.read_text().splitlines()[0])
        assert "cube_pose" in line["obs_summary"]
        assert "goal_pos" in line["obs_summary"]
        assert "tcp_pose" in line["obs_summary"]
        assert "ego_qpos" in line["obs_summary"]
        assert "ego_qvel" in line["obs_summary"]
        assert "partner_qpos" in line["obs_summary"]
        assert "partner_qvel" in line["obs_summary"]

    def test_summary_lifts_is_grasped_from_info(self, archive_dir: Path) -> None:
        recorder = RolloutRecorder(
            archive_dir=archive_dir,
            condition="as-hetero",
            global_step=0,
            ego_uid="panda_wristcam",
            partner_uid="fetch",
        )
        recorder.begin_episode(seed=0)
        recorder.record_step(
            _minimal_stage1_obs(),
            action=[0.0] * 8,
            reward=0.0,
            info={"is_grasped": True, "gripper_width": 0.05},
            terminated=False,
            truncated=False,
            frame=None,
        )
        _, jsonl_path = recorder.finalise()
        assert jsonl_path is not None
        line = json.loads(jsonl_path.read_text().splitlines()[0])
        assert line["obs_summary"]["is_grasped"] is True
        assert line["obs_summary"]["gripper_width"] == pytest.approx(0.05)

    def test_summary_tolerates_missing_obs_keys(self, archive_dir: Path) -> None:
        """ADR-017 §Decisions: non-Stage-1 obs shapes degrade gracefully (no KeyError)."""
        recorder = RolloutRecorder(
            archive_dir=archive_dir,
            condition="smoke",
            global_step=0,
            ego_uid="ego",
            partner_uid=None,
        )
        recorder.begin_episode(seed=0)
        recorder.record_step(
            {"agent": {"ego": {}}},  # no qpos/qvel, no extra
            action=[0.0],
            reward=0.0,
            info=None,
            terminated=False,
            truncated=False,
            frame=None,
        )
        _, jsonl_path = recorder.finalise()
        assert jsonl_path is not None
        line = json.loads(jsonl_path.read_text().splitlines()[0])
        # obs_summary is present but empty — recorder didn't crash.
        assert line["obs_summary"] == {}


class TestJSONLEmission:
    """Per-step JSONL sidecar contract."""

    def test_one_line_per_step(self, archive_dir: Path) -> None:
        rng = np.random.default_rng(0)
        recorder = RolloutRecorder(
            archive_dir=archive_dir,
            condition="as-hetero",
            global_step=1000,
            ego_uid="panda_wristcam",
            partner_uid="fetch",
        )
        recorder.begin_episode(seed=0)
        for i in range(5):
            recorder.record_step(
                _minimal_stage1_obs(),
                action=np.zeros(8, dtype=np.float32),
                reward=float(i),
                info={},
                terminated=False,
                truncated=False,
                frame=_random_frame(rng),
            )
        _, jsonl_path = recorder.finalise()
        assert jsonl_path is not None
        lines = jsonl_path.read_text().splitlines()
        assert len(lines) == 5
        for i, line_raw in enumerate(lines):
            line = json.loads(line_raw)
            assert line["event"] == "rollout_step"
            assert line["metric_namespace"] == "rollout"
            assert line["step_episode"] == i
            assert line["step_global"] == 1000 + i
            assert line["condition"] == "as-hetero"
            assert line["reward"] == pytest.approx(float(i))

    def test_jsonl_path_under_rollouts_condition_step(self, archive_dir: Path) -> None:
        recorder = RolloutRecorder(
            archive_dir=archive_dir,
            condition="as-hetero",
            global_step=25000,
            ego_uid="panda_wristcam",
            partner_uid="fetch",
        )
        recorder.begin_episode(seed=0)
        recorder.record_step(
            _minimal_stage1_obs(),
            action=np.zeros(8, dtype=np.float32),
            reward=0.0,
            info={},
            terminated=False,
            truncated=False,
            frame=None,
        )
        _, jsonl_path = recorder.finalise()
        assert jsonl_path is not None
        # Path shape: <archive_dir>/rollouts/<condition>/step_<NNNNNN>.jsonl
        assert jsonl_path == archive_dir / "rollouts" / "as-hetero" / "step_025000.jsonl"

    def test_per_step_jsonl_false_skips_emission(self, archive_dir: Path) -> None:
        recorder = RolloutRecorder(
            archive_dir=archive_dir,
            condition="smoke",
            global_step=0,
            per_step_jsonl=False,
            ego_uid="ego",
            partner_uid=None,
        )
        recorder.begin_episode(seed=0)
        recorder.record_step(
            {"agent": {"ego": {}}},
            action=[0.0],
            reward=0.0,
            info=None,
            terminated=False,
            truncated=False,
            frame=None,
        )
        _, jsonl_path = recorder.finalise()
        assert jsonl_path is None


class TestMP4Emission:
    """MP4 encode path — synthetic frames; the env-integration test is xfail-strict."""

    def test_mp4_written_when_frames_captured(self, archive_dir: Path) -> None:
        """Synthetic-frame encoder smoke (not xfail per Stage 0 founder direction)."""
        rng = np.random.default_rng(0)
        recorder = RolloutRecorder(
            archive_dir=archive_dir,
            condition="as-hetero",
            global_step=25000,
            fps=20,
            ego_uid="panda_wristcam",
            partner_uid="fetch",
        )
        recorder.begin_episode(seed=0)
        for _ in range(10):
            recorder.record_step(
                _minimal_stage1_obs(),
                action=np.zeros(8, dtype=np.float32),
                reward=0.0,
                info={},
                terminated=False,
                truncated=False,
                frame=_random_frame(rng),
            )
        mp4_path, _ = recorder.finalise()
        assert mp4_path is not None
        assert mp4_path.exists()
        assert mp4_path.stat().st_size > 0
        # Path shape: <archive_dir>/rollouts/<condition>/step_<NNNNNN>.mp4
        assert mp4_path == archive_dir / "rollouts" / "as-hetero" / "step_025000.mp4"

    def test_no_frames_means_no_mp4(self, archive_dir: Path) -> None:
        """ADR-017 §Decisions: degrade gracefully when env.render returns None."""
        recorder = RolloutRecorder(
            archive_dir=archive_dir,
            condition="smoke",
            global_step=0,
            ego_uid="ego",
            partner_uid=None,
        )
        recorder.begin_episode(seed=0)
        # Frame=None throughout — mimics the Stage 0 probe 2 host state.
        recorder.record_step(
            {"agent": {"ego": {}}},
            action=[0.0],
            reward=0.0,
            info=None,
            terminated=False,
            truncated=False,
            frame=None,
        )
        mp4_path, jsonl_path = recorder.finalise()
        assert mp4_path is None
        # JSONL still emitted; that's the agent-side fallback.
        assert jsonl_path is not None


class TestFrameValidation:
    """record_step() loud-fails on wrong-shape frames (defensive)."""

    def test_rejects_non_ndarray_frame(self, archive_dir: Path) -> None:
        recorder = RolloutRecorder(
            archive_dir=archive_dir,
            condition="smoke",
            global_step=0,
            ego_uid="ego",
            partner_uid=None,
        )
        recorder.begin_episode(seed=0)
        with pytest.raises(ValueError, match="numpy ndarray"):
            recorder.record_step(
                {"agent": {"ego": {}}},
                action=[0.0],
                reward=0.0,
                info=None,
                terminated=False,
                truncated=False,
                frame="not_an_array",  # type: ignore[arg-type]
            )

    def test_rejects_wrong_shape_frame(self, archive_dir: Path) -> None:
        recorder = RolloutRecorder(
            archive_dir=archive_dir,
            condition="smoke",
            global_step=0,
            ego_uid="ego",
            partner_uid=None,
        )
        recorder.begin_episode(seed=0)
        with pytest.raises(ValueError, match="must be \\(H, W, 3\\)"):
            recorder.record_step(
                {"agent": {"ego": {}}},
                action=[0.0],
                reward=0.0,
                info=None,
                terminated=False,
                truncated=False,
                frame=np.zeros((84, 84), dtype=np.uint8),  # 2-D, missing channels
            )
