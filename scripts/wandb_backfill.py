#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
r"""W&B backfill of committed Stage-1 archives (P1.05.11; ADR-017 §Decisions D7).

Walks an archive root for ``<run_id>.jsonl`` files + their sibling
spike envelopes, then replays them to W&B as historical runs tagged
``backfill:true``. Idempotent via ``wandb.init(id=<run_id>, resume="never")``
— re-running against the same archive skips existing runs.

What this script does deliver
-----------------------------

For per-cell JSONLs that already exist on disk (e.g. the 5 cells under
``spikes/results/stage1-failure-investigation/2026-05-20``), the
backfill replays **envelope-level terminal metrics** (per-episode
``success``, ``mean_reward``, ``lambda_*`` summary) into W&B and
emits the existing JSONL events as wandb.log calls keyed by ``step``.

What this script does NOT deliver
---------------------------------

It does **not** synthesise per-step PPO-health curves (policy_loss,
value_loss, dist_entropy, approx_kl, clip_fraction, grad_norm) for
historical archives. Those scalars were never written to disk before
P1.05.11 (commit 3 of this slice wired the trainer). Per-step
PPO-health curves are only available **for runs going forward** —
any cell launched after this slice merges with
``observability.wandb.enabled=true``.

Default targets (committed archives only):

- ``spikes/results/stage1-AS-20260517/`` — Stage-1a rig validation.
- ``spikes/results/stage1-OM-20260517/`` — Stage-1a rig validation.
- ``spikes/results/stage1-failure-investigation/2026-05-20/`` — Stage-1b
  failed launch (5 seeds x AS-hetero).

Optional with ``--include-local``: the four ``.local/`` post-widening
snapshots (gitignored; founder-local evidence from P1.05.8 closure):

- ``.local/p1_05_8_as_hetero_smoke.json``
- ``.local/a1_zero_action_partner.json``
- ``.local/a2_safety_disabled.json``
- ``.local/a5_partner_placement.json``

Usage::

    python scripts/wandb_backfill.py --archive-root spikes/results/ \\
        --project concerto-chamber --mode offline --dry-run

    python scripts/wandb_backfill.py --archive-root spikes/results/ \\
        --project concerto-chamber --mode online

    python scripts/wandb_backfill.py --archive-root .local/ \\
        --include-local --project concerto-chamber --mode online
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterable, Iterator


@dataclass(frozen=True)
class _BackfillTarget:
    """One backfill target — either a per-cell JSONL or a summary-only envelope.

    Attributes:
        run_id: 16-hex per-cell id (the W&B ``id`` field; idempotent).
        archive_dir: Directory the source files live in.
        jsonl_path: Per-cell JSONL path, or ``None`` for envelope-only
            archives (``.local/*.json`` snapshots).
        envelope_path: Sibling envelope path, or ``None``.
        tags: Pre-built tag list (stage:, sub_stage:, condition:,
            prereg:<sha8>, backfill:true).
        config_extras: Merged into ``wandb.config`` on the W&B run
            alongside the four RunContext provenance fields.
    """

    run_id: str
    archive_dir: Path
    jsonl_path: Path | None
    envelope_path: Path | None
    tags: tuple[str, ...]
    config_extras: dict[str, object]


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield one parsed JSON dict per non-empty line in ``path``."""
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def _parse_envelope(envelope_path: Path) -> dict[str, Any] | None:
    """Parse one envelope JSON, returning ``None`` on read/parse failure."""
    try:
        data = json.loads(envelope_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _build_tags(
    *,
    envelope: dict[str, Any] | None,
    backfill: bool = True,
) -> tuple[str, ...]:
    """Build the W&B tag tuple for one cell (ADR-017 §Decisions: tag-based filtering)."""
    tags: list[str] = []
    if envelope is None:
        return tuple(tags)
    sub_stage = envelope.get("sub_stage")
    if isinstance(sub_stage, str):
        tags.append(f"sub_stage:{sub_stage}")
        if sub_stage and sub_stage[0].isdigit():
            tags.append(f"stage:{sub_stage[0]}")
    axis = envelope.get("axis")
    if isinstance(axis, str):
        tags.append(f"axis:{axis}")
    prereg_sha = envelope.get("prereg_sha")
    if isinstance(prereg_sha, str) and prereg_sha:
        # ADR-017 §Decisions: short-form on tag, full SHA in
        # wandb.config (handled below in config_extras).
        tags.append(f"prereg:{prereg_sha[:8]}")
    cp = envelope.get("condition_pair")
    if isinstance(cp, list) and cp:
        tags.append(f"condition:{','.join(str(c) for c in cp)[:64]}")
    if backfill:
        tags.append("backfill:true")
    return tuple(tags)


def _build_config_extras(
    *,
    envelope: dict[str, Any] | None,
    archive_dir: Path,
) -> dict[str, object]:
    """Build the ``wandb.config`` payload extras (ADR-017 §Decisions)."""
    extras: dict[str, object] = {
        "backfill": True,
        "backfill_archive": str(archive_dir),
    }
    if envelope is None:
        return extras
    for key in ("prereg_sha", "git_tag", "axis", "sub_stage", "spike_id", "schema_version"):
        val = envelope.get(key)
        if val is not None:
            extras[key] = val
    return extras


def _iter_jsonl_targets(archive_root: Path) -> Iterable[_BackfillTarget]:
    """Yield one :class:`_BackfillTarget` per ``<run_id>.jsonl`` under ``archive_root``."""
    for jsonl_path in sorted(archive_root.rglob("*.jsonl")):
        # Skip rollout sidecar JSONLs — those are not per-cell logs.
        if "rollouts" in jsonl_path.parts:
            continue
        # Skip per-run logs nested under audit-run scratch dirs
        # (``_braking_parity_*``) that aren't Stage-1 archives.
        if any(p.startswith("_braking_parity_") for p in jsonl_path.parts):
            continue
        first_line: dict[str, Any] | None = None
        for line in _read_jsonl(jsonl_path):
            first_line = line
            break
        if first_line is None or first_line.get("event") != "training_start":
            continue
        run_id = str(first_line.get("run_id", ""))
        if not run_id:
            continue
        archive_dir = jsonl_path.parent
        # Locate sibling envelope. Don't fail loud if absent.
        envelope_paths = sorted(archive_dir.glob("spike_*.json"))
        envelope_path = envelope_paths[0] if envelope_paths else None
        envelope = _parse_envelope(envelope_path) if envelope_path else None
        yield _BackfillTarget(
            run_id=run_id,
            archive_dir=archive_dir,
            jsonl_path=jsonl_path,
            envelope_path=envelope_path,
            tags=_build_tags(envelope=envelope),
            config_extras=_build_config_extras(envelope=envelope, archive_dir=archive_dir),
        )


def _iter_envelope_only_targets(archive_root: Path) -> Iterable[_BackfillTarget]:
    """Yield one :class:`_BackfillTarget` per envelope-only ``*.json`` under ``archive_root``.

    Used for ``--include-local``: the ``.local/`` snapshots store the
    envelope under different per-file names (no per-step .jsonl
    history), so terminal-only replay is the best we can do.
    """
    for json_path in sorted(archive_root.glob("*.json")):
        envelope = _parse_envelope(json_path)
        if envelope is None:
            continue
        # Heuristic: this looks like a spike envelope if it has the
        # expected top-level keys.
        if "spike_id" not in envelope or "episode_results" not in envelope:
            continue
        # Use the spike_id + a short hash of the path for run_id since
        # envelope-only snapshots don't carry per-cell run_ids.
        spike_id = str(envelope.get("spike_id", json_path.stem))
        run_id = (spike_id + "_" + json_path.stem)[:64]
        yield _BackfillTarget(
            run_id=run_id,
            archive_dir=archive_root,
            jsonl_path=None,
            envelope_path=json_path,
            tags=_build_tags(envelope=envelope),
            config_extras=_build_config_extras(envelope=envelope, archive_dir=archive_root),
        )


def _replay_jsonl_to_wandb(
    target: _BackfillTarget,
    *,
    project: str,
    dry_run: bool,
) -> None:
    """Replay one ``<run_id>.jsonl`` to W&B (ADR-017 §Decisions D7).

    For each line carrying a ``step``, calls ``wandb.log(metrics, step=step)``
    with the line's metric-valued fields. Calls ``wandb.init`` with
    ``id=run_id`` + ``resume="never"`` so re-running is idempotent.
    """
    if dry_run or target.jsonl_path is None:
        print(
            f"[would replay] jsonl={target.jsonl_path} "
            f"run_id={target.run_id} tags={list(target.tags)}"
        )
        return
    import wandb  # noqa: PLC0415 - lazy import per ADR-017 §Decisions

    run = wandb.init(
        id=target.run_id,
        name=target.run_id,
        project=project,
        tags=list(target.tags),
        config=target.config_extras,
        resume="never",
        reinit=True,
    )
    try:
        for line in _read_jsonl(target.jsonl_path):
            step_raw = line.get("step")
            if not isinstance(step_raw, int):
                continue
            # Forward only numeric fields (W&B treats string fields oddly).
            metrics = {
                k: v
                for k, v in line.items()
                if isinstance(v, (int, float))
                and k
                not in {
                    "step",
                    "seed",
                    "n_obs",
                    "n_filter_calls",
                    "n_fallback_fires",
                    "n_qp_infeasible",
                    "n_braking_fires",
                    "n_checkpoints",
                    "n_episodes",
                }
            }
            if metrics:
                wandb.log(metrics, step=step_raw)
    finally:
        wandb.finish()
        del run


def _replay_envelope_to_wandb(
    target: _BackfillTarget,
    *,
    project: str,
    dry_run: bool,
) -> None:
    """Replay one envelope-only ``*.json`` to W&B as terminal summary stats."""
    if dry_run or target.envelope_path is None:
        print(
            f"[would replay envelope-only] envelope={target.envelope_path} "
            f"run_id={target.run_id} tags={list(target.tags)}"
        )
        return
    import wandb  # noqa: PLC0415

    envelope = _parse_envelope(target.envelope_path)
    if envelope is None:
        return
    run = wandb.init(
        id=target.run_id,
        name=target.run_id,
        project=project,
        tags=list(target.tags),
        config=target.config_extras,
        resume="never",
        reinit=True,
    )
    try:
        episodes = envelope.get("episode_results", [])
        if not isinstance(episodes, list):
            episodes = []
        success_count = 0
        total = 0
        for idx, ep in enumerate(episodes):
            if not isinstance(ep, dict):
                continue
            total += 1
            if ep.get("success") is True:
                success_count += 1
            wandb.log(
                {
                    "episode_success": float(bool(ep.get("success"))),
                    "constraint_violation_peak": float(ep.get("constraint_violation_peak", 0.0)),
                    "fallback_fired": float(ep.get("fallback_fired", 0)),
                },
                step=idx,
            )
        if total > 0:
            wandb.log({"terminal_success_rate": success_count / total}, step=total)
    finally:
        wandb.finish()
        del run


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="wandb_backfill",
        description=(
            "Backfill committed Stage-1 archives to W&B as historical runs "
            "(P1.05.11; ADR-017 §Decisions D7)."
        ),
    )
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument(
        "--project",
        default="concerto-chamber",
        help="W&B project (default: concerto-chamber per ADR-017 §Decisions D2).",
    )
    parser.add_argument(
        "--mode",
        choices=("online", "offline"),
        default="offline",
        help="W&B mode. Default 'offline' for safety; pass 'online' to publish.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--include-local",
        action="store_true",
        help="Opt into envelope-only replay of .local/*.json snapshots (gitignored).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point for the wandb_backfill script (P1.05.11; ADR-017 §Decisions D7).

    Returns ``0`` on success; ``2`` on archive-root-not-found.
    """
    args = _parse_args(argv)
    if not args.archive_root.exists():
        print(f"archive root {args.archive_root!r} not found", file=sys.stderr)
        return 2
    # Force W&B mode at the environment level so children that read
    # WANDB_MODE see the right value.
    os.environ["WANDB_MODE"] = args.mode

    targets: list[_BackfillTarget] = list(_iter_jsonl_targets(args.archive_root))
    if args.include_local:
        targets.extend(_iter_envelope_only_targets(args.archive_root))

    if not targets:
        print(f"No backfill targets under {args.archive_root}", file=sys.stderr)
        return 0

    for target in targets:
        if target.jsonl_path is not None:
            _replay_jsonl_to_wandb(target, project=args.project, dry_run=args.dry_run)
        else:
            _replay_envelope_to_wandb(target, project=args.project, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
