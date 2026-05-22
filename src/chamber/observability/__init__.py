# SPDX-License-Identifier: Apache-2.0
"""Chamber-side observability surface (P1.05.11; ADR-017 §Decisions).

The ``concerto.training.logging`` module owns the per-cell structured
log surface (RunContext + JSONL + W&B sink Protocol + ``log_scalars`` /
``log_eval`` helpers). This package layers chamber-specific
observability on top: the :class:`RolloutRecorder` captures eval-rollout
videos + per-step sidecar JSONLs paired 1:1 (ADR-017 §Schema appendix).
The recorder is the killer feature of the slice — agents can read the
sidecar JSONL at step granularity to diagnose "what is the policy doing
in this rollout?" without watching video.

Per ADR-017 §Decisions D6, per-step rollout records are written to a
sidecar JSONL at ``<archive_dir>/rollouts/<condition>/step_<NNNNNN>.jsonl``
paired with the MP4 at the same path — they are NOT appended to the
main per-cell ``<run_id>.jsonl`` stream. Tooling that streams the main
JSONL is not forced to parse rollout-frame payloads.

Public surface:

- :class:`RolloutRecorder` — the recorder (P1.05.11).

Re-exports the :class:`~concerto.training.config.RolloutRecorderConfig`
Pydantic block for convenience.
"""

from __future__ import annotations

from chamber.observability.rollout_recorder import RolloutRecorder
from concerto.training.config import RolloutRecorderConfig

__all__ = [
    "RolloutRecorder",
    "RolloutRecorderConfig",
]
