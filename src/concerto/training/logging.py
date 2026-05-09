# SPDX-License-Identifier: Apache-2.0
"""Structured logging for ego-AHT training runs (ADR-002 §Decisions; plan/05 §2).

Every line emitted during a training run carries the ``RunContext`` fields
(``run_id``, ``seed``, ``git_sha``, ``pyproject_hash``) plus a per-event
``step`` so any line in any artefact can be traced back to the exact code
+ config + RNG state that produced it. Phase-0 ships two sinks:

- A local JSONL file (``logs/<run_id>.jsonl``) — the offline-replayable
  fallback that does not depend on W&B being reachable. Every JSONL line
  is a self-contained valid JSON object.
- An opt-in :class:`wandb.sdk.wandb_run.Run` sink — the caller passes a
  pre-constructed run object as the ``wandb_sink`` kwarg of
  :func:`bind_run_logger`. Phase-0 uses W&B's offline mode in tests so
  the sink can be exercised without a network round-trip. The four
  :class:`RunContext` provenance fields are written to W&B's ``config``
  once at bind time (via :meth:`WandbSink.set_config`) rather than to
  every metric event, so they appear as run-level metadata rather than
  scalar metrics.

Determinism: :func:`compute_run_metadata` derives ``run_id`` from
``(seed, git_sha, pyproject_hash, run_kind)`` so two reruns with identical
code + config + seed produce the same ``run_id`` (P6 reproducibility). The
``pyproject_hash`` is the SHA-256 of the project's ``pyproject.toml`` so a
silent dependency drift produces a different ``run_id``.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from typing import IO, TYPE_CHECKING, Any, Protocol

import structlog

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping
    from pathlib import Path

#: Sentinel value used when the working tree is not a git repository or the
#: ``git`` binary is not on ``PATH``. Plan/05 §2: every log line carries
#: ``git_sha`` even when the run is unrooted (e.g. inside a container) so
#: downstream readers can detect missing provenance loudly.
GIT_SHA_UNKNOWN: str = "unknown"


@dataclass(frozen=True)
class RunContext:
    """Per-run provenance bundle bound to every log line (ADR-002 §Decisions; plan/05 §2; P6).

    Attributes:
        run_id: Stable 16-hex-char hash derived from the other four fields
            via :func:`compute_run_metadata`. Used as the JSONL filename
            and as the W&B run name so artefacts collated by ``run_id``
            from different sinks describe the same run.
        seed: Project-wide root seed; ego-AHT runs derive every RNG
            substream from this via :func:`concerto.training.seeding.derive_substream`.
        git_sha: Full SHA of the currently-checked-out commit, or
            :data:`GIT_SHA_UNKNOWN` when the working tree is not a git
            repository or the ``git`` binary is missing.
        pyproject_hash: SHA-256 of ``pyproject.toml`` (full hex digest).
            A silent dependency drift produces a different hash and
            therefore a different ``run_id``.
        run_kind: Free-form label describing what this run is for (e.g.
            ``"empirical_guarantee"``, ``"zoo_seed"``). Folded into the
            ``run_id`` hash so two distinct kinds at the same seed do not
            collide.
        extra: Optional free-form string-string metadata attached to every
            log line (e.g. task name, partner class). Read-only by
            convention; defaults to empty dict.
    """

    run_id: str
    seed: int
    git_sha: str
    pyproject_hash: str
    run_kind: str
    extra: dict[str, str] = field(default_factory=dict)


def _detect_git_sha(repo_root: Path) -> str:
    """Return the current commit SHA of ``repo_root`` or :data:`GIT_SHA_UNKNOWN`.

    Plan/05 §2: the run-context bundle is always populated; missing git
    provenance is a sentinel rather than a None so the downstream JSONL
    schema is uniform.

    Args:
        repo_root: Working-tree directory to query.

    Returns:
        Full SHA string, or :data:`GIT_SHA_UNKNOWN` on any failure
        (missing ``git`` binary, not-a-repo, detached HEAD with no commits).
    """
    try:
        # Inputs are: literal "git" + literal subcommand + a Path-as-str
        # (the working tree the run was launched from). No untrusted shell
        # interpolation is possible because the list form is used.
        result = subprocess.run(  # noqa: S603  # see comment above.
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],  # noqa: S607  # ditto.
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return GIT_SHA_UNKNOWN
    return result.stdout.strip() or GIT_SHA_UNKNOWN


def _compute_pyproject_hash(pyproject_path: Path) -> str:
    """Return the SHA-256 of ``pyproject_path`` (plan/05 §2; P6 reproducibility).

    Args:
        pyproject_path: File to hash. The hash detects silent dependency
            drift between reruns at the same ``(seed, git_sha)`` pair.

    Returns:
        Full lowercase hex digest. If the file does not exist (rare;
        we are inside the project), returns the SHA-256 of the empty
        bytestring so the schema is still uniform.
    """
    if not pyproject_path.exists():
        return hashlib.sha256(b"").hexdigest()
    return hashlib.sha256(pyproject_path.read_bytes()).hexdigest()


def compute_run_metadata(
    *,
    seed: int,
    run_kind: str,
    repo_root: Path,
    extra: Mapping[str, str] | None = None,
) -> RunContext:
    """Build :class:`RunContext` with auto-detected git/pyproject hashes (ADR-002; plan/05 §2).

    The ``run_id`` is the first 16 hex chars of the SHA-256 of
    ``(seed, git_sha, pyproject_hash, run_kind)``, matching the
    16-hex-char convention used for ``PartnerSpec.partner_id`` (plan/04
    §3.1) so artefact filenames are uniform across the project.

    Args:
        seed: Root seed; passed to
            :func:`concerto.training.seeding.derive_substream` upstream.
        run_kind: Free-form label (e.g. ``"empirical_guarantee"``).
        repo_root: Working-tree directory to query for ``git_sha`` and
            ``pyproject.toml``.
        extra: Optional string-string metadata attached to every line.

    Returns:
        A frozen :class:`RunContext` ready to bind via :func:`bind_run_logger`.
    """
    git_sha = _detect_git_sha(repo_root)
    pyproject_hash = _compute_pyproject_hash(repo_root / "pyproject.toml")
    material = repr((seed, git_sha, pyproject_hash, run_kind))
    run_id = hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]
    return RunContext(
        run_id=run_id,
        seed=seed,
        git_sha=git_sha,
        pyproject_hash=pyproject_hash,
        run_kind=run_kind,
        extra=dict(extra or {}),
    )


class _JSONLSink:
    """structlog processor that writes one JSON object per line (plan/05 §2).

    The processor is the *terminal* renderer: it serialises the event_dict
    as JSON, appends a newline, and writes to the open file handle. The
    serialisation is byte-stable for byte-identical inputs; non-JSON-
    serialisable values are passed through ``str(...)`` so an
    :class:`UnsupportedType` does not crash the run.
    """

    def __init__(self, fh: IO[str]) -> None:
        """Bind the open file handle (ADR-002 §Decisions).

        Args:
            fh: Writable text-mode handle (e.g. produced by ``open(path, "w")``).
                The caller owns the handle's lifecycle; the sink does not close it.
        """
        self._fh = fh

    def __call__(
        self,
        logger: object,
        method_name: str,
        event_dict: dict[str, Any],
    ) -> str:
        """Serialise + write one line (structlog processor protocol).

        Args:
            logger: structlog logger reference (unused).
            method_name: log-level method name (unused; the level is in
                ``event_dict["level"]`` if the caller bound it).
            event_dict: structlog event mapping (the bound run context +
                per-call kwargs).

        Returns:
            The serialised line so structlog can return it as a no-op for
            other sinks chained downstream.
        """
        del logger, method_name
        line = json.dumps(event_dict, default=str, sort_keys=True)
        self._fh.write(line + "\n")
        self._fh.flush()
        return line


class WandbSink(Protocol):
    """Minimal subset of :class:`wandb.sdk.wandb_run.Run` we depend on (ADR-002 §Decisions).

    Plan/05 §2: W&B is opt-in. The Protocol lets tests inject an in-memory
    fake without requiring the real ``wandb`` package to be importable.
    """

    def log(self, data: Mapping[str, object], *, step: int | None = None) -> None:
        """Emit one structured event (W&B-side surface; ADR-002 §Decisions)."""
        ...  # pragma: no cover

    def set_config(self, config: Mapping[str, object]) -> None:
        """Store run-level metadata once at bind time (ADR-002 §Decisions; plan/05 §2).

        Called by :func:`bind_run_logger` to pin the four
        :class:`RunContext` provenance fields (``run_id``, ``seed``,
        ``git_sha``, ``pyproject_hash``) on the W&B run as metadata rather
        than as per-step metrics.
        """
        ...  # pragma: no cover


#: Run-level metadata keys that go to ``wandb.run.config`` once at bind
#: time, NOT as per-step metrics on every emitted event (plan/05 §2; ADR-002).
_RUN_CONTEXT_KEYS: frozenset[str] = frozenset(
    {"run_id", "seed", "git_sha", "pyproject_hash", "run_kind"}
)


class _WandbProcessor:
    """structlog processor that forwards events to a :class:`WandbSink` (plan/05 §2).

    Extracts ``step`` from the event_dict (default ``None``), drops the
    :data:`_RUN_CONTEXT_KEYS` (those are pinned via :meth:`WandbSink.set_config`
    once at bind time, not re-emitted on every metric line), and forwards
    the remaining fields as ``data``. It re-emits the original event_dict
    so downstream processors (e.g. :class:`_JSONLSink`) still receive the
    full context.
    """

    def __init__(self, sink: WandbSink) -> None:
        """Bind the W&B sink (ADR-002 §Decisions).

        Args:
            sink: Object satisfying :class:`WandbSink`. Tests pass an
                in-memory fake; production passes a ``wandb.run`` instance.
        """
        self._sink = sink

    def __call__(
        self,
        logger: object,
        method_name: str,
        event_dict: dict[str, Any],
    ) -> dict[str, Any]:
        """Forward to the W&B sink + pass through (structlog protocol).

        Args:
            logger: Unused.
            method_name: Unused.
            event_dict: structlog event mapping. The ``step`` key (if
                present) is mapped to W&B's ``step=`` kwarg; run-context
                fields (:data:`_RUN_CONTEXT_KEYS`) are stripped from the
                forwarded ``data`` since they live on ``run.config``.

        Returns:
            ``event_dict`` unchanged so chained processors see the full
            context.
        """
        del logger, method_name
        step_raw = event_dict.get("step")
        step = int(step_raw) if isinstance(step_raw, int) else None
        metric_data = {k: v for k, v in event_dict.items() if k not in _RUN_CONTEXT_KEYS}
        self._sink.log(metric_data, step=step)
        return event_dict


def bind_run_logger(
    ctx: RunContext,
    *,
    jsonl_path: Path,
    wandb_sink: WandbSink | None = None,
) -> structlog.BoundLogger:
    """Build a structlog logger that emits JSONL + optionally W&B (ADR-002 §Decisions; plan/05 §2).

    The returned logger has every :class:`RunContext` field bound to it so
    every emitted line carries them automatically. Callers add per-event
    fields (e.g. ``step``, ``ego_reward``) via ``logger.info(...)``.

    The function does NOT close the JSONL file; the caller owns its
    lifecycle (typically a ``with open(...) as fh:`` block held open for
    the duration of the run).

    Args:
        ctx: Run-level provenance bundle.
        jsonl_path: Filesystem path where the JSONL fallback is written.
            Parent directory is created if missing.
        wandb_sink: Optional :class:`WandbSink` (production: a ``wandb.run``;
            tests: an in-memory fake). When ``None``, only JSONL is emitted.

    Returns:
        A :class:`structlog.BoundLogger` wrapping the JSONL (and optionally
        W&B) sinks. The logger is not module-global; callers explicitly
        pass it through their training loop so unit tests can substitute
        in-memory sinks freely.
    """
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    fh = jsonl_path.open("w", encoding="utf-8")
    if wandb_sink is not None:
        # Pin run-level provenance once on the W&B run's config so the four
        # context fields don't appear as per-step metrics in charts.
        wandb_sink.set_config(
            {
                "run_id": ctx.run_id,
                "seed": ctx.seed,
                "git_sha": ctx.git_sha,
                "pyproject_hash": ctx.pyproject_hash,
                "run_kind": ctx.run_kind,
                **ctx.extra,
            }
        )
    processors: list[Any] = [structlog.processors.add_log_level]
    if wandb_sink is not None:
        processors.append(_WandbProcessor(wandb_sink))
    processors.append(_JSONLSink(fh))
    base = structlog.wrap_logger(
        logger=None,  # type: ignore[arg-type]  # structlog uses a no-op base when None.
        processors=processors,
    )
    return base.bind(
        run_id=ctx.run_id,
        seed=ctx.seed,
        git_sha=ctx.git_sha,
        pyproject_hash=ctx.pyproject_hash,
        run_kind=ctx.run_kind,
        **ctx.extra,
    )


__all__ = [
    "GIT_SHA_UNKNOWN",
    "RunContext",
    "WandbSink",
    "bind_run_logger",
    "compute_run_metadata",
]
