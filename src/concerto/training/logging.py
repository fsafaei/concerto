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
import logging
import os
import subprocess
import warnings
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import IO, TYPE_CHECKING, Any, Literal, Protocol

import structlog

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterable, Mapping, Sequence
    from pathlib import Path

    from concerto.training.config import WandbConfig

#: Sentinel value used when the working tree is not a git repository or the
#: ``git`` binary is not on ``PATH``. Plan/05 §2: every log line carries
#: ``git_sha`` even when the run is unrooted (e.g. inside a container) so
#: downstream readers can detect missing provenance loudly.
GIT_SHA_UNKNOWN: str = "unknown"

#: Allow-list of metric namespaces (P1.05.11; ADR-017 §Decisions). The
#: ``metric_namespace`` field on ``event="scalar"`` lines must be one
#: of these values; the ``chamber-analyze metrics --namespace`` flag
#: filters on the same set. Extending the allow-list requires editing
#: this constant + ADR-017 §Schema appendix together.
LogNamespace = Literal["train", "eval", "safety", "hardware", "rollout"]

#: Concrete tuple of allowed namespace strings — for runtime validation
#: in :func:`log_scalars` (the ``Literal`` is type-only).
_LOG_NAMESPACES: frozenset[str] = frozenset({"train", "eval", "safety", "hardware", "rollout"})


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
    #: Optional safety-stack telemetry summary (P1.04.5; ADR-007 §Stage 1b).
    #:
    #: ``None`` for pre-P1.04.5 callers; populated by
    #: :func:`concerto.training.ego_aht.train` at end-of-cell when
    #: ``safety_filter`` was wired (the
    #: :class:`concerto.training.safety_telemetry.SafetyAggregator`'s
    #: ``finalise()`` output). Forward-additive widening — existing
    #: JSONL parsers ignore the field; the new audit-gate hook reads
    #: the ``safety_telemetry_final`` JSONL event directly (more
    #: discoverable than reading the in-memory ``RunContext`` post-
    #: training). The ``RunContext`` carrying the field is the
    #: source-of-truth for in-process callers (e.g.
    #: :class:`chamber.benchmarks.stage1_common.TrainedPolicyFactory`)
    #: that need the summary without re-parsing the JSONL.
    safety_telemetry: dict[str, object] | None = None


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
    config_fingerprint: str | None = None,
) -> RunContext:
    """Build :class:`RunContext` with auto-detected git/pyproject hashes (ADR-002; plan/05 §2).

    The ``run_id`` is the first 16 hex chars of the SHA-256 of
    ``(seed, git_sha, pyproject_hash, run_kind)`` — plus
    ``config_fingerprint`` when supplied — matching the 16-hex-char
    convention used for ``PartnerSpec.partner_id`` (plan/04 §3.1) so
    artefact filenames are uniform across the project.

    RUNID-COLLISION fix (issue #214; ADR-002 §Revision history
    2026-06-10): without the config in the hash material, same-seed
    runs at the same commit collide — observed in the 2026-06-10
    regime-alignment chain, where cells differing only in regime knobs
    (gamma, num_envs, frames) reproduced each other's run_ids and their
    plain ``{run_id}.jsonl`` archive copies overwrote. Callers that own
    a run config (the training loop) pass its fingerprint; ``None``
    preserves the legacy hash material byte-for-byte so config-less
    callers' historical run_ids remain reproducible.

    Args:
        seed: Root seed; passed to
            :func:`concerto.training.seeding.derive_substream` upstream.
        run_kind: Free-form label (e.g. ``"empirical_guarantee"``).
        repo_root: Working-tree directory to query for ``git_sha`` and
            ``pyproject.toml``.
        extra: Optional string-string metadata attached to every line.
        config_fingerprint: Optional stable hash of the run's config
            (the training loop passes the SHA-256 of the Pydantic
            ``model_dump_json()`` excluding the operator-side sink
            fields ``artifacts_root`` / ``log_dir`` / ``wandb`` —
            output destinations are not run semantics, and the P6
            byte-identity test pins that two same-seed runs in
            different scratch dirs share one identity; see
            :func:`concerto.training.ego_aht.train`). Folded into the
            run_id material when not ``None``.

    Returns:
        A frozen :class:`RunContext` ready to bind via :func:`bind_run_logger`.
    """
    git_sha = _detect_git_sha(repo_root)
    pyproject_hash = _compute_pyproject_hash(repo_root / "pyproject.toml")
    if config_fingerprint is None:
        material = repr((seed, git_sha, pyproject_hash, run_kind))
    else:
        material = repr((seed, git_sha, pyproject_hash, run_kind, config_fingerprint))
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
    """Minimal :class:`wandb.sdk.wandb_run.Run` surface we depend on (ADR-002, ADR-017 §Decisions).

    Plan/05 §2: W&B is opt-in. The Protocol lets tests inject an in-memory
    fake without requiring the real ``wandb`` package to be importable.
    P1.05.11 widened the surface from two methods (``log`` /
    ``set_config``) to four; the new methods (``add_tags`` and
    ``close``) are part of ADR-017's W&B-run-lifecycle contract.
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

    def add_tags(self, tags: Iterable[str]) -> None:
        """Append tags to the W&B run (P1.05.11; ADR-017 §Decisions).

        ADR-017 §Decisions: ``stage:<n>``, ``sub_stage:<x>``,
        ``condition:<id>``, ``prereg:<sha8>``, ``backfill:<bool>``
        appear here. Implementations must be idempotent: re-adding an
        existing tag is a no-op.
        """
        ...  # pragma: no cover

    def close(self) -> None:
        """Finalise the W&B run (P1.05.11; ADR-017 §Decisions).

        Called by the training driver once the cell completes. The
        :class:`_WandbRunSink` impl calls ``wandb.finish()``; in-memory
        test fakes typically no-op.
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


def _iso_utc_now() -> str:
    """Return current UTC time as a tz-aware ISO-8601 string (ADR-017 §Decisions).

    Used by :func:`log_scalars` and :func:`log_eval` so every namespaced
    line carries a wall-clock timestamp that the ``chamber-analyze``
    CLI's ``metrics`` and ``compare`` subcommands can render as the
    x-axis when ``step`` is unavailable.

    Returns:
        ISO-8601 string with ``+00:00`` offset, e.g.
        ``"2026-05-22T15:54:48.752123+00:00"``.
    """
    return datetime.now(UTC).isoformat()


def log_scalars(
    logger: structlog.BoundLogger,
    *,
    step: int,
    namespace: LogNamespace,
    **scalars: float,
) -> None:
    """Emit one ``event="scalar"`` line with a namespace tag (P1.05.11; ADR-017 §Schema).

    Thin wrapper that pins the ``event``, ``metric_namespace``, and
    ``wall_time`` fields so the ``chamber-analyze`` CLI's namespace-
    based filtering has a stable contract. Other fields flow through
    as scalar metric values.

    The wrapper does not validate the metric *values* — the caller is
    responsible for passing ``float``-shaped data. ``None`` / NaN is
    accepted (downstream readers must handle them).

    Args:
        logger: A bound logger from :func:`bind_run_logger`. Its bound
            ``RunContext`` fields propagate to the emitted line.
        step: Global training step (frames). Routed to W&B's ``step=``
            kwarg via :class:`_WandbProcessor` and appears as a top-
            level ``step`` field in the JSONL line.
        namespace: One of ``"train"``, ``"eval"``, ``"safety"``,
            ``"hardware"``, ``"rollout"``. The ``chamber-analyze
            metrics --namespace`` flag filters on this value.
        **scalars: Metric key → value pairs. Each pair appears as a
            top-level field on the emitted JSONL line.

    Raises:
        ValueError: If ``namespace`` is not in the allow-list (defensive
            backstop against typos that would silently make a
            metric un-findable by ``chamber-analyze``).
    """
    if namespace not in _LOG_NAMESPACES:
        msg = (
            f"log_scalars: namespace={namespace!r} not in allow-list "
            f"{sorted(_LOG_NAMESPACES)!r}. Extending the allow-list requires "
            "editing concerto.training.logging._LOG_NAMESPACES + ADR-017 §Schema."
        )
        raise ValueError(msg)
    logger.info(
        "scalar",
        step=step,
        metric_namespace=namespace,
        wall_time=_iso_utc_now(),
        **scalars,
    )


def log_eval(
    logger: structlog.BoundLogger,
    *,
    step: int,
    condition: str,
    **results: float | int,
) -> None:
    """Emit one ``event="eval"`` line for an eval-cell completion (P1.05.11; ADR-017 §Schema).

    Eval events differ from training scalars: they carry the
    ``condition`` identifier (one of the four Stage-1 ``condition_id``
    strings) and aggregate terminal statistics (``success_rate``,
    ``mean_episode_length``, ``mean_episode_reward``,
    ``n_terminated``, ``n_truncated``). The ``chamber-analyze
    summary`` subcommand surfaces these as the run's terminal metrics.

    Args:
        logger: A bound logger from :func:`bind_run_logger`.
        step: Global training step (frames) at which the eval was
            triggered.
        condition: Stage-1 ``condition_id`` (e.g.
            ``"stage1_pickplace_panda_plus_fetch_ego_aht_happo_per_agent"``)
            or a short slug like ``"as-hetero"``.
        **results: Terminal metric key → value pairs. ``int``-shaped
            counts (``n_terminated``, ``n_truncated``) are accepted
            alongside ``float`` rates.
    """
    logger.info(
        "eval",
        step=step,
        condition=condition,
        metric_namespace="eval",
        wall_time=_iso_utc_now(),
        **results,
    )


class _WandbRunSink:
    """Concrete :class:`WandbSink` wrapping a real ``wandb.sdk.wandb_run.Run`` (P1.05.11; ADR-017).

    Built by :func:`make_wandb_run_sink` when ``wandb`` is importable
    and ``cfg.enabled=True`` and ``wandb.init`` succeeds. Production
    callers do not instantiate directly — they go through the factory
    so the graceful-degrade-to-no-op path is uniform.

    The sink owns the W&B run lifecycle: it calls ``wandb.finish()`` on
    :meth:`close`, which the training driver invokes once the cell is
    done (after the JSONL file handle is closed). The W&B run *must*
    be finished cleanly for the run to appear in the W&B UI as
    "completed" rather than "crashed"; the audit story depends on this.
    """

    def __init__(self, run: Any) -> None:  # noqa: ANN401 - wandb.Run is dynamic by design
        """Bind a ``wandb.sdk.wandb_run.Run`` (ADR-017 §Decisions).

        Args:
            run: The object returned by :func:`wandb.init`. We don't
                import the type here — wandb is a runtime dep but the
                logger module's structlog-only path must keep importing
                cheaply. The ``Any`` annotation is intentional;
                ANN401 is suppressed at the param level.
        """
        self._run = run

    def log(self, data: Mapping[str, object], *, step: int | None = None) -> None:
        """Forward to :meth:`wandb.run.log` (ADR-002 §Decisions)."""
        # wandb's log signature accepts a dict + an optional int step.
        self._run.log(dict(data), step=step)

    def set_config(self, config: Mapping[str, object]) -> None:
        """Pin run-level config once at bind time (ADR-002 §Decisions; plan/05 §2)."""
        # wandb.config is dict-like; update() copies key/value pairs.
        self._run.config.update(dict(config))

    def add_tags(self, tags: Iterable[str]) -> None:
        """Append tags to the W&B run (ADR-017 §Decisions: tag-based filtering).

        Idempotent: tags already on the run are not duplicated. The W&B
        Python client exposes ``run.tags`` as a tuple; we re-bind via
        the explicit setter ``run.tags = (...)``.
        """
        existing = tuple(getattr(self._run, "tags", ()) or ())
        merged = tuple(dict.fromkeys((*existing, *tags)))  # preserves order, dedupes
        self._run.tags = merged

    def close(self) -> None:
        """Finalise the W&B run (ADR-017 §Decisions: clean-completion contract).

        Calls ``wandb.finish()`` so the run is uploaded and marked
        completed. Safe to call once per sink; subsequent calls are
        no-ops at the wandb-client level. Errors during finish are
        caught + logged (the JSONL is the canonical record; a failed
        wandb upload must not break the training driver).
        """
        try:
            self._run.finish()
        except Exception as exc:  # top-level catch by design — degrade gracefully
            logging.getLogger(__name__).warning(
                "wandb.run.finish() raised; the JSONL artefact is unaffected. "
                "exc_type=%s message=%s",
                type(exc).__name__,
                str(exc)[:200],
            )


def make_wandb_run_sink(
    cfg: WandbConfig,
    ctx: RunContext,
    *,
    tags: Sequence[str] = (),
    config_extras: Mapping[str, object] | None = None,
) -> WandbSink | None:
    """Build a W&B sink, or return ``None`` to degrade to JSONL-only (P1.05.11; ADR-017).

    The factory implements the optional-sink discipline: every failure
    path returns ``None`` (with a single warning), never raises. The
    training driver passes the returned value as ``wandb_sink=`` to
    :func:`bind_run_logger`; ``None`` keeps the existing JSONL-only
    behaviour intact.

    Failure paths (each emits a single :class:`UserWarning` and returns
    ``None``):

    - ``cfg.enabled is False`` (the most common path; not really a
      "failure" — the operator opted out).
    - ``WANDB_API_KEY`` is missing AND ``cfg.mode != "offline"``
      (online uploads require auth; offline mode writes to disk).
    - ``import wandb`` raises (defensive — wandb is a runtime dep
      today, but the no-import path stays here for the future where
      the founder may demote it to extras).
    - ``wandb.init(...)`` raises (network, auth, quota, ...).

    On success, the W&B run is created with:

    - ``id = ctx.run_id`` — reuses the 16-hex per-cell run id so the
      W&B run name matches the JSONL filename one-to-one.
    - ``project = cfg.project`` — default ``"concerto-chamber"`` per
      ADR-017 §Decisions.
    - ``tags = (*tags, "stage:<x>", "run_kind:<x>", ...)`` — the
      caller supplies stage/sub_stage/condition/prereg tags; the
      factory does not invent them. ADR-017 §Decisions pins that
      ``prereg_sha`` is also written to ``wandb.config`` so the W&B UI
      can filter both on a tag (cheap match) AND on a config field
      (auditable).
    - ``config`` carries the four :class:`RunContext` provenance
      fields plus any ``config_extras`` (e.g. ``prereg_sha``,
      ``git_sha``, ``command_line``). These appear in the run's
      "Overview" panel — searchable via the W&B UI's filter panel.

    Args:
        cfg: The :class:`concerto.training.config.WandbConfig` block
            from the composed Hydra config.
        ctx: Per-cell :class:`RunContext` (binds ``run_id``, ``seed``,
            ``git_sha``, ``pyproject_hash``, ``run_kind``).
        tags: Optional iterable of pre-built tag strings. ADR-017
            §Decisions: include ``stage:<n>``, ``sub_stage:<x>``,
            ``condition:<id>``, ``prereg:<sha8>``, ``backfill:true|false``
            at minimum.
        config_extras: Optional mapping merged into ``wandb.config``
            on top of the four :class:`RunContext` provenance fields.
            Use for ``prereg_sha`` (full SHA, alongside the
            short-form in ``tags``), ``command_line``, and other
            non-provenance audit metadata.

    Returns:
        A :class:`_WandbRunSink` wrapping the live W&B run, or
        ``None`` if any failure path above fired. The training driver
        passes the returned value (or ``None``) to
        :func:`bind_run_logger`.
    """
    if not cfg.enabled:
        # Operator opt-out; not a failure — silent (no warning).
        return None

    # Auth check before the (expensive) wandb import. ``WANDB_API_KEY`` is
    # required for online mode. Offline mode (``WANDB_MODE=offline``) writes
    # to disk and does not need a key; the offline path is what the
    # deterministic-seed equivalence smoke uses.
    wandb_mode = os.environ.get("WANDB_MODE", "online")
    if wandb_mode != "offline" and not os.environ.get("WANDB_API_KEY"):
        warnings.warn(
            "WANDB_API_KEY missing and WANDB_MODE != 'offline'; W&B sink degrades "
            "to no-op (run continues with JSONL only). Set WANDB_API_KEY or "
            "export WANDB_MODE=offline to silence. (ADR-017 §Decisions)",
            UserWarning,
            stacklevel=2,
        )
        return None

    try:
        import wandb  # noqa: PLC0415 - import inside try is the degrade-gracefully contract
    except ImportError as exc:
        warnings.warn(
            f"`import wandb` failed ({type(exc).__name__}: {exc!s:.200}); "
            "W&B sink degrades to no-op (run continues with JSONL only). "
            "Install with `uv sync` or `pip install wandb`. (ADR-017 §Decisions)",
            UserWarning,
            stacklevel=2,
        )
        return None

    # Build the wandb.config payload — RunContext provenance + caller extras.
    cfg_payload: dict[str, object] = {
        "run_id": ctx.run_id,
        "seed": ctx.seed,
        "git_sha": ctx.git_sha,
        "pyproject_hash": ctx.pyproject_hash,
        "run_kind": ctx.run_kind,
    }
    cfg_payload.update(ctx.extra)
    if config_extras:
        cfg_payload.update(dict(config_extras))

    try:
        run = wandb.init(
            id=ctx.run_id,
            name=ctx.run_id,
            project=cfg.project,
            tags=tuple(tags) if tags else None,
            config=cfg_payload,
            resume="never",
            reinit=True,  # allow multiple per-cell runs in one process
        )
    except Exception as exc:  # top-level catch by design — degrade gracefully
        warnings.warn(
            f"wandb.init(...) raised ({type(exc).__name__}: {exc!s:.200}); "
            "W&B sink degrades to no-op (run continues with JSONL only). "
            "(ADR-017 §Decisions)",
            UserWarning,
            stacklevel=2,
        )
        return None

    return _WandbRunSink(run)


__all__ = [
    "GIT_SHA_UNKNOWN",
    "LogNamespace",
    "RunContext",
    "WandbSink",
    "bind_run_logger",
    "compute_run_metadata",
    "log_eval",
    "log_scalars",
    "make_wandb_run_sink",
]
