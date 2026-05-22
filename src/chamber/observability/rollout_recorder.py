# SPDX-License-Identifier: Apache-2.0
"""Eval-rollout MP4 + per-step sidecar JSONL recorder (P1.05.11; ADR-017 §Decisions).

The recorder captures one full eval episode every
:attr:`RolloutRecorderConfig.interval_frames` training frames, emitting
two paired artefacts at
``<archive_dir>/rollouts/<condition>/step_<NNNNNN>.{mp4,jsonl}``:

- an MP4 the founder can watch in W&B (uploaded via
  :meth:`chamber.observability.rollout_recorder.RolloutRecorder.finalise`'s
  return value, which the caller hands to
  :func:`concerto.training.logging.WandbSink.log_artifact` or to
  ``wandb.Video`` directly), and
- a per-step JSONL the **agent** can parse — `chamber-analyze
  rollout-frames <run-id>` reads it.

Per ADR-017 §Schema appendix the per-step records are **sidecar files**,
not appended to the main per-cell ``<run_id>.jsonl`` stream. Rollout
data is bulky and tooling that streams the main JSONL must not be
forced to parse rollout-frame payloads.

The per-step JSONL carries a **curated** ``obs_summary`` field — never
the full 65-D flattened ego state. The summary includes ``cube_pose``
(7), ``goal_pos`` (3), ``tcp_pose`` (7), ``ego_qpos`` / ``ego_qvel``,
``partner_qpos`` / ``partner_qvel``, plus ``is_grasped`` and
``gripper_width`` from ``info`` when the env exposes them. This keeps
agent-side reasoning tractable; full-obs logging is deferred to a
follow-up per ADR-017 §Open questions.

Graceful degradation: ``imageio`` is imported lazily inside
:meth:`RolloutRecorder.finalise`. When unavailable the recorder logs a
structured warning and emits the per-step JSONL only (no MP4). The
``[observability]`` optional dep at ``pyproject.toml`` pins
``imageio[ffmpeg]``; callers who skip the extra see the degraded path.
"""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping
    from pathlib import Path

    from numpy.typing import NDArray

_LOGGER = logging.getLogger(__name__)

#: Keys lifted from ``obs["extra"]`` (Stage1ASStateSynthesizer task-extras
#: layout) into the per-step ``obs_summary``. Cube pose (7-D), goal
#: position (3-D), and ego TCP pose (7-D) are the agent-side reasoning
#: anchors the founder named in CONSULTATION_BRIEF §2 (2026-05-21).
#: Missing keys are tolerated (skipped silently) so non-Stage-1 envs
#: that pass through the recorder do not fail loud.
_OBS_EXTRA_KEYS: tuple[str, ...] = ("cube_pose", "goal_pos", "tcp_pose")

#: Keys lifted from ``obs["agent"][<uid>]`` per ego/partner. Synthesizer
#: emits ``qpos`` / ``qvel`` on the inner agent sub-dict (per ADR-007
#: §Stage 1b Rev 12); we capture both for both agents so the agent-side
#: reader can correlate joint trajectories with cube/goal motion.
_AGENT_SUBKEYS: tuple[str, ...] = ("qpos", "qvel")

#: Keys lifted from the env's per-step ``info`` dict. ``is_grasped`` and
#: ``gripper_width`` are the canonical pick-place success-correlates that
#: ManiSkill's Stage-1b env emits; missing keys are tolerated.
_INFO_KEYS: tuple[str, ...] = ("is_grasped", "gripper_width")


def _to_jsonable(val: object) -> object:
    """Convert torch / numpy scalars + arrays to JSON-serialisable shapes.

    The per-step JSONL writer uses ``json.dumps(default=str)`` as the
    backstop, but pre-converting numpy / torch types here keeps the
    output compact (numeric, not stringified) for the keys the agent-
    side reader actually parses.

    Args:
        val: Raw value from an obs / info dict. May be a numpy scalar,
            a numpy ndarray (1-D or higher), a torch tensor (caller is
            expected to ``.cpu()`` first; we don't import torch here),
            a Python primitive, or a nested dict/list of any of these.

    Returns:
        A JSON-serialisable shape: ``int``, ``float``, ``str``, ``bool``,
        ``list``, ``dict``, or ``None``. Numpy arrays become lists of
        floats. Unknown types fall through unchanged (``json.dumps``'s
        ``default=str`` backstop handles them).
    """
    if isinstance(val, np.ndarray):
        return val.astype(float).tolist()
    if isinstance(val, np.generic):  # numpy scalar (e.g. np.float32)
        return val.item()
    if isinstance(val, (list, tuple)):
        return [_to_jsonable(v) for v in val]
    if isinstance(val, dict):
        return {k: _to_jsonable(v) for k, v in val.items()}
    return val


def _build_obs_summary(
    obs: Mapping[str, Any],
    *,
    ego_uid: str,
    partner_uid: str | None,
    info: Mapping[str, Any] | None,
) -> dict[str, object]:
    """Build the small curated ``obs_summary`` for one rollout step (ADR-017 §Decisions).

    Args:
        obs: The per-step env obs dict. Expected shape matches the
            Stage1ASStateSynthesizer / Stage1OMChannelFilter output:
            ``obs["agent"][<uid>]`` with ``qpos`` / ``qvel`` per agent,
            ``obs["extra"]`` with task fields (cube_pose, goal_pos,
            tcp_pose). Any missing key is tolerated (skipped).
        ego_uid: The ego agent's uid (matches
            ``cfg.env.agent_uids[0]``).
        partner_uid: The partner uid; ``None`` for single-agent envs.
        info: Per-step ``info`` dict from the env. ``is_grasped`` /
            ``gripper_width`` are lifted when present.

    Returns:
        A flat-ish dict ready for ``json.dumps``. Numpy arrays are
        converted to ``list[float]`` for compactness.
    """
    summary: dict[str, object] = {}
    # Task extras (cube_pose, goal_pos, tcp_pose).
    extra = obs.get("extra")
    if isinstance(extra, dict):
        for key in _OBS_EXTRA_KEYS:
            if key in extra:
                summary[key] = _to_jsonable(extra[key])
    # Per-agent qpos / qvel.
    agent = obs.get("agent")
    if isinstance(agent, dict):
        for uid_label, uid in (("ego", ego_uid), ("partner", partner_uid)):
            if uid is None:
                continue
            sub = agent.get(uid)
            if not isinstance(sub, dict):
                continue
            for key in _AGENT_SUBKEYS:
                if key in sub:
                    summary[f"{uid_label}_{key}"] = _to_jsonable(sub[key])
    # Info-level signals (is_grasped, gripper_width).
    if info is not None:
        for key in _INFO_KEYS:
            if key in info:
                summary[key] = _to_jsonable(info[key])
    return summary


@dataclass
class _StepRecord:
    """One frame of a recorded eval episode (P1.05.11; ADR-017 §Schema appendix).

    The recorder accumulates these in memory through the episode, then
    serialises them to the per-step sidecar JSONL on :meth:`finalise`.
    Kept as a dataclass so unit tests can introspect captured records
    directly without re-parsing JSON.
    """

    step_global: int
    step_episode: int
    obs_summary: dict[str, object]
    action: list[float]
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, object]
    frame_index: int


@dataclass
class RolloutRecorder:
    """Eval-rollout MP4 + per-step sidecar JSONL recorder (P1.05.11; ADR-017 §Decisions).

    The :func:`chamber.benchmarks.stage1_common`/:func:`stage1_{as,om}`
    eval cell builds one of these per eval boundary when
    :attr:`RolloutRecorderConfig.enabled` is ``True``. The recorder
    captures every step of the episode and finalises into paired
    artefacts on disk.

    Lifecycle:

    .. code-block:: python

        recorder = RolloutRecorder(
            archive_dir=cell_dir,
            condition="as-hetero",
            global_step=global_step,
            fps=20,
            per_step_jsonl=True,
            ego_uid="panda_wristcam",
            partner_uid="fetch",
        )
        recorder.begin_episode(seed=0)
        # … loop … per step …
        recorder.record_step(obs, action, reward, info, terminated, truncated, frame)
        mp4_path, jsonl_path = recorder.finalise()

    Attributes:
        archive_dir: Cell-level archive directory (typically
            ``<spike_archive>/<seed>``). The recorder writes to
            ``archive_dir / "rollouts" / condition / "step_<NNNNNN>.{mp4,jsonl}"``.
        condition: Stage-1 ``condition_id`` (or short slug) for the
            episode. Becomes the sub-directory name under ``rollouts/``.
        global_step: Training-frame counter at the eval boundary.
            Becomes the ``step_<NNNNNN>`` filename prefix (zero-padded
            to 6 digits).
        fps: Video frame rate for the MP4 encode. Default 20 (matches
            ManiSkill's Stage-1b ``control_freq``).
        per_step_jsonl: When ``False``, skip the JSONL emission and
            emit the MP4 only. Default ``True`` per ADR-017.
        ego_uid: Ego agent uid; used by :func:`_build_obs_summary` to
            select the right sub-dict.
        partner_uid: Partner uid; ``None`` for single-agent envs.
    """

    archive_dir: Path
    condition: str
    global_step: int
    fps: int = 20
    per_step_jsonl: bool = True
    ego_uid: str = "ego"
    partner_uid: str | None = "partner"
    _episode_seed: int | None = field(default=None, init=False, repr=False)
    _episode_step: int = field(default=0, init=False, repr=False)
    _records: list[_StepRecord] = field(default_factory=list, init=False, repr=False)
    _frames: list[NDArray[np.uint8]] = field(default_factory=list, init=False, repr=False)

    def begin_episode(self, *, seed: int) -> None:
        """Reset the per-episode step counter (P1.05.11; ADR-017 §Decisions).

        Called once per eval episode at reset time. Clears any prior
        per-step state — the recorder is reusable across episodes within
        a single ``finalise()`` boundary if the caller chooses, though
        the canonical lifecycle is one recorder per episode (matches
        the eval-cell wiring in commit 4).

        Args:
            seed: The env-reset seed for this episode. Recorded in
                the per-step JSONL as ``seed`` on every line.
        """
        self._episode_seed = seed
        self._episode_step = 0
        self._records.clear()
        self._frames.clear()

    def record_step(
        self,
        obs: Mapping[str, Any],
        action: NDArray[np.floating] | list[float],
        reward: float,
        info: Mapping[str, Any] | None,
        *,
        terminated: bool,
        truncated: bool,
        frame: NDArray[np.uint8] | None,
    ) -> None:
        """Accumulate one step's record + optional frame (P1.05.11; ADR-017 §Schema).

        The frame is appended to the internal frame buffer; the obs +
        action + reward + flags + info go into a :class:`_StepRecord`.
        Both buffers are flushed on :meth:`finalise`.

        Args:
            obs: Per-step env obs dict (post-step, i.e. the obs the env
                returned alongside ``reward``).
            action: The ego action that produced ``reward``. ``list``
                or ``np.ndarray`` of floats; serialised as a flat list.
            reward: The per-step ego reward (env's shared scalar; same
                contract as the trainer's :meth:`observe`).
            info: The env's per-step ``info`` dict. ``is_grasped`` /
                ``gripper_width`` are lifted into the obs_summary.
                ``None`` is tolerated.
            terminated: Episode termination flag (true-terminal).
            truncated: Episode truncation flag (time-limit).
            frame: Optional ``(H, W, 3)`` uint8 ndarray from
                ``env.render(mode="rgb_array")``. ``None`` skips the
                frame buffer for this step (the JSONL record still
                lands).

        Raises:
            ValueError: If ``frame`` is not ``None`` and not a 3-channel
                uint8 ndarray.
        """
        if frame is not None:
            if not isinstance(frame, np.ndarray):
                msg = (
                    f"RolloutRecorder.record_step: frame must be a numpy ndarray; "
                    f"got {type(frame).__name__}."
                )
                raise ValueError(msg)
            if frame.ndim != 3 or frame.shape[-1] != 3:  # noqa: PLR2004 - 3 channels (RGB), 3 dims
                msg = (
                    f"RolloutRecorder.record_step: frame must be (H, W, 3); "
                    f"got shape {frame.shape}."
                )
                raise ValueError(msg)
            if frame.dtype != np.uint8:
                # Defensive: imageio's mp4 encoder expects uint8.
                frame = frame.astype(np.uint8)
            self._frames.append(frame)

        action_list = (
            action.astype(float).tolist() if isinstance(action, np.ndarray) else list(action)
        )
        record = _StepRecord(
            step_global=self.global_step + self._episode_step,
            step_episode=self._episode_step,
            obs_summary=_build_obs_summary(
                obs,
                ego_uid=self.ego_uid,
                partner_uid=self.partner_uid,
                info=info,
            ),
            action=action_list,
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            info={k: _to_jsonable(v) for k, v in (info or {}).items()},
            frame_index=len(self._frames) - 1 if frame is not None else -1,
        )
        self._records.append(record)
        self._episode_step += 1

    def finalise(self) -> tuple[Path | None, Path | None]:
        """Write the MP4 + per-step JSONL to disk (P1.05.11; ADR-017 §Decisions).

        Returns the paths so the caller can hand them to W&B as
        artefacts (``WandbSink.log_artifact`` for the MP4;
        ``wandb.Video`` for the MP4; a plain file artefact for the
        JSONL). Either path may be ``None`` if its emission was
        skipped:

        - MP4 is skipped when no frames were captured (e.g. the env
          renderer returned ``None`` — current host-state on the
          author's box per Stage 0 probe 2), OR when ``imageio`` is
          not importable (lazy guard inside this method).
        - JSONL is skipped when ``per_step_jsonl=False``.

        Returns:
            ``(mp4_path | None, jsonl_path | None)`` — paths to the
            written artefacts, or ``None`` for any skipped emission.

        Raises:
            OSError: If the archive directory cannot be created.
        """
        out_dir = self.archive_dir / "rollouts" / self.condition
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = f"step_{self.global_step:06d}"

        mp4_path: Path | None = None
        if self._frames:
            mp4_path = out_dir / f"{stem}.mp4"
            try:
                import imageio.v3 as iio  # noqa: PLC0415 - lazy degrade-gracefully import
            except ImportError as exc:
                _LOGGER.warning(
                    "RolloutRecorder: `imageio` not importable (%s); MP4 emission skipped. "
                    "Install with `uv sync --extra observability`. (ADR-017 §Decisions)",
                    type(exc).__name__,
                )
                warnings.warn(
                    "imageio not importable; RolloutRecorder degrading to JSONL-only.",
                    UserWarning,
                    stacklevel=2,
                )
                mp4_path = None
            else:
                # imageio.v3.imwrite signature for video: pass a list/iter
                # of frames; the ffmpeg plugin handles the encode. fps
                # is passed via the ``fps`` plugin kwarg.
                try:
                    iio.imwrite(
                        str(mp4_path),
                        np.stack(self._frames),
                        plugin="FFMPEG",
                        fps=self.fps,
                    )
                except Exception as exc:  # mp4 encode failures must not crash the run
                    _LOGGER.warning(
                        "RolloutRecorder: MP4 encode raised (%s: %s); video emission skipped. "
                        "Per-step JSONL is unaffected.",
                        type(exc).__name__,
                        str(exc)[:200],
                    )
                    mp4_path = None

        jsonl_path: Path | None = None
        if self.per_step_jsonl and self._records:
            jsonl_path = out_dir / f"{stem}.jsonl"
            wall_time = datetime.now(UTC).isoformat()
            with jsonl_path.open("w", encoding="utf-8") as fh:
                for rec in self._records:
                    line = {
                        "event": "rollout_step",
                        "metric_namespace": "rollout",
                        "step_global": rec.step_global,
                        "step_episode": rec.step_episode,
                        "condition": self.condition,
                        "seed": self._episode_seed,
                        "wall_time": wall_time,
                        "obs_summary": rec.obs_summary,
                        "action": rec.action,
                        "reward": rec.reward,
                        "terminated": rec.terminated,
                        "truncated": rec.truncated,
                        "info": rec.info,
                        "frame_index": rec.frame_index,
                    }
                    fh.write(json.dumps(line, default=str, sort_keys=True) + "\n")

        return mp4_path, jsonl_path


__all__ = [
    "RolloutRecorder",
]
