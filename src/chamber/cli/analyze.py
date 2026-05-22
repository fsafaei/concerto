# SPDX-License-Identifier: Apache-2.0
"""``chamber-analyze`` — read-only co-analysis CLI (P1.05.11; ADR-017 §Decisions).

The CLI is the **agent's interface** to the canonical JSONL record.
Per ADR-017 §Decisions D5, it reads from local JSONL archives — never
from the W&B API — so it works offline, in CI, and is reproducible
from the canonical record alone.

Six subcommands; every one supports ``--json`` for agent consumption:

- :func:`_cmd_list_runs` — table of runs in an archive.
- :func:`_cmd_summary` — terminal stats for one run.
- :func:`_cmd_metrics` — time-series dump for one metric.
- :func:`_cmd_compare` — side-by-side comparison across N runs.
- :func:`_cmd_plot` — matplotlib PNG comparing one metric across N runs.
- :func:`_cmd_rollout_frames` — dump per-step rollout sidecar JSONL
  rows (ADR-017 §Schema appendix).

Pattern mirrors :mod:`chamber.cli.render_tables`: argparse-based,
JSON-first output, no click/typer dependency.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterable


@dataclass(frozen=True)
class _RunSummary:
    """Per-cell run summary surfaced by ``chamber-analyze list-runs`` (ADR-017 §Decisions).

    Built by :func:`_load_run_summary` from a single ``<run_id>.jsonl``
    file plus the (optional) sibling envelope. Fields without data in
    the archive are populated as ``None`` rather than omitted so the
    JSON shape is stable.

    Attributes:
        run_id: 16-hex per-cell identifier from
            :func:`concerto.training.logging.compute_run_metadata`.
        jsonl_path: Absolute path to the source ``<run_id>.jsonl`` file.
        stage: Stage label (``"1"``, ``"2"``, ``"3"``) lifted from
            ``sub_stage`` on the envelope; ``None`` when no envelope is
            adjacent.
        sub_stage: ``"1a"`` / ``"1b"`` / ``"2"`` / ``"3"``; ``None``
            without envelope.
        condition: Condition_id string; ``None`` without envelope.
        seed: Seed lifted from the JSONL's first ``training_start``
            event; ``None`` if not parseable.
        task: Task name (``"stage1_pickplace"``, ``"mpe_cooperative_push"``,
            …); from ``training_start``.
        run_kind: From ``training_start`` (e.g. ``"ego_aht_happo"``).
        git_sha: From ``training_start``.
        n_steps: Maximum ``step`` value observed in the JSONL; ``None``
            for empty / pre-P1.05.11 archives without scalar events.
        terminal_success_rate: Lifted from a matching envelope's
            ``episode_results`` aggregated by seed/condition. ``None``
            when no envelope is adjacent OR the envelope doesn't
            include a matching seed.
        mean_reward_final: Last ``last_reward`` from any
            ``rollout_update`` event (legacy P1.05.x archives) or the
            last ``mean_episode_reward`` from any ``event=eval`` line
            (post-P1.05.11). ``None`` if neither is present.
        lambda_steady_state: From the ``safety_telemetry_final`` event
            when present.
    """

    run_id: str
    jsonl_path: str
    stage: str | None = None
    sub_stage: str | None = None
    condition: str | None = None
    seed: int | None = None
    task: str | None = None
    run_kind: str | None = None
    git_sha: str | None = None
    n_steps: int | None = None
    terminal_success_rate: float | None = None
    mean_reward_final: float | None = None
    lambda_steady_state: float | None = None


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    """Yield one parsed JSON dict per non-empty line in ``path``."""
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                # ADR-017 §Decisions: malformed lines are skipped with a
                # silent continue — the CLI must be robust against
                # historical archives that pre-date schema discipline.
                continue
            if isinstance(obj, dict):
                yield obj


def _find_envelope(archive_root: Path) -> dict[str, Any] | None:
    """Locate and parse the spike envelope (``spike_*.json``) under ``archive_root``.

    Returns ``None`` when no envelope is present (e.g. an archive of
    bare JSONLs without a SpikeRun bundle). Picks the first envelope
    found alphabetically when multiple exist; matches the convention
    in :mod:`chamber.evaluation` of one envelope per archive directory.

    Args:
        archive_root: Directory to search (non-recursive — envelopes
            live at the archive root, not in subdirectories).

    Returns:
        Parsed envelope dict, or ``None``.
    """
    candidates = sorted(
        p
        for p in archive_root.glob("spike_*.json")
        # Don't accidentally match leaderboard.json or random siblings.
        if p.is_file()
    )
    for path in candidates:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
    return None


def _terminal_success_rate_for_seed(
    envelope: dict[str, Any] | None, *, seed: int | None
) -> float | None:
    """Aggregate ``success`` over ``episode_results`` filtered by ``seed``.

    Pre-P1.05.11 archives have no ``event="eval"`` lines in the
    per-cell JSONL — terminal success comes from the envelope's
    aggregate ``episode_results``. This helper does that join.

    Args:
        envelope: Parsed envelope dict or ``None``.
        seed: Seed to filter on; ``None`` skips the join.

    Returns:
        ``mean(success for ep in envelope.episode_results if ep.seed == seed)``,
        or ``None`` when no matching episodes exist.
    """
    if envelope is None or seed is None:
        return None
    episodes = envelope.get("episode_results", [])
    if not isinstance(episodes, list):
        return None
    successes: list[float] = []
    for ep in episodes:
        if not isinstance(ep, dict):
            continue
        if ep.get("seed") != seed:
            continue
        success = ep.get("success")
        if isinstance(success, bool):
            successes.append(float(success))
    if not successes:
        return None
    return sum(successes) / len(successes)


def _load_run_summary(jsonl_path: Path, *, envelope: dict[str, Any] | None) -> _RunSummary:  # noqa: PLR0912 - field-by-field JSONL→summary join
    """Build a :class:`_RunSummary` from one ``<run_id>.jsonl`` file (ADR-017 §Decisions).

    Args:
        jsonl_path: Path to a per-cell JSONL.
        envelope: Optional parsed sibling envelope (for joining
            terminal success rate by seed).

    Returns:
        Populated :class:`_RunSummary`. Fields without source data are
        ``None``.
    """
    training_start: dict[str, Any] | None = None
    safety_final: dict[str, Any] | None = None
    last_rollout_reward: float | None = None
    last_eval_reward: float | None = None
    max_step: int = 0
    for line in _read_jsonl(jsonl_path):
        event = line.get("event")
        if event == "training_start":
            training_start = line
        elif event == "safety_telemetry_final":
            safety_final = line
        elif event == "rollout_update":
            raw_reward = line.get("last_reward")
            # P1.05.x sometimes wrote ``"tensor([...])"`` as the str
            # repr; tolerate both forms.
            if isinstance(raw_reward, (int, float)):
                last_rollout_reward = float(raw_reward)
        elif event == "eval":
            mer = line.get("mean_episode_reward")
            if isinstance(mer, (int, float)):
                last_eval_reward = float(mer)
        step = line.get("step")
        if isinstance(step, int) and step > max_step:
            max_step = step

    run_id = "unknown"
    seed: int | None = None
    task: str | None = None
    run_kind: str | None = None
    git_sha: str | None = None
    if training_start is not None:
        run_id = str(training_start.get("run_id", "unknown"))
        seed_val = training_start.get("seed")
        seed = int(seed_val) if isinstance(seed_val, int) else None
        task = (
            str(training_start.get("task")) if isinstance(training_start.get("task"), str) else None
        )
        run_kind = (
            str(training_start.get("run_kind"))
            if isinstance(training_start.get("run_kind"), str)
            else None
        )
        git_sha = (
            str(training_start.get("git_sha"))
            if isinstance(training_start.get("git_sha"), str)
            else None
        )

    sub_stage = (
        str(envelope.get("sub_stage"))
        if envelope and isinstance(envelope.get("sub_stage"), str)
        else None
    )
    stage = sub_stage[0] if sub_stage else None
    condition = None
    if envelope is not None:
        # Envelope may carry `condition_pair` (list[str]) or
        # `condition_id` (single str); fall back gracefully.
        cp = envelope.get("condition_pair")
        if isinstance(cp, list) and cp:
            condition = ",".join(str(c) for c in cp)
        elif isinstance(envelope.get("condition_id"), str):
            condition = str(envelope["condition_id"])

    lambda_ss = None
    if safety_final is not None:
        raw_lambda = safety_final.get("lambda_steady_state")
        if isinstance(raw_lambda, (int, float)):
            lambda_ss = float(raw_lambda)

    return _RunSummary(
        run_id=run_id,
        jsonl_path=str(jsonl_path),
        stage=stage,
        sub_stage=sub_stage,
        condition=condition,
        seed=seed,
        task=task,
        run_kind=run_kind,
        git_sha=git_sha,
        n_steps=max_step if max_step > 0 else None,
        terminal_success_rate=_terminal_success_rate_for_seed(envelope, seed=seed),
        mean_reward_final=last_eval_reward if last_eval_reward is not None else last_rollout_reward,
        lambda_steady_state=lambda_ss,
    )


def _walk_runs(archive_root: Path) -> list[_RunSummary]:
    """Enumerate every ``<run_id>.jsonl`` under ``archive_root`` (ADR-017 §Decisions).

    Walks the directory non-recursively (the canonical archive layout
    has one envelope + N per-cell JSONLs at the same level). Falls back
    to recursive walk if no JSONLs are found at the top level.

    Args:
        archive_root: Directory to search.

    Returns:
        List of :class:`_RunSummary`, sorted by ``seed`` then ``run_id``.
    """
    jsonl_paths = sorted(archive_root.glob("*.jsonl"))
    if not jsonl_paths:
        jsonl_paths = sorted(archive_root.rglob("*.jsonl"))
    envelope = _find_envelope(archive_root)
    return sorted(
        (_load_run_summary(p, envelope=envelope) for p in jsonl_paths),
        key=lambda s: (s.seed if s.seed is not None else 0, s.run_id),
    )


def _emit_json(payload: object, output: Path | None) -> None:
    """Write ``payload`` as JSON to ``output`` or stdout."""
    text = json.dumps(payload, indent=2, default=str)
    if output is None:
        print(text)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")


def _cmd_list_runs(args: argparse.Namespace) -> int:
    """``chamber-analyze list-runs`` — enumerate runs under an archive root."""
    summaries = _walk_runs(args.archive_root)
    if args.json:
        _emit_json([asdict(s) for s in summaries], args.output)
        return 0
    # Human-readable table.
    if not summaries:
        print(f"No JSONL runs found under {args.archive_root}", file=sys.stderr)
        return 0
    widths = {
        "run_id": 18,
        "sub_stage": 9,
        "seed": 6,
        "n_steps": 9,
        "success": 10,
        "reward": 10,
        "lambda_ss": 10,
    }
    header = (
        f"{'run_id':<{widths['run_id']}} "
        f"{'sub_stage':<{widths['sub_stage']}} "
        f"{'seed':>{widths['seed']}} "
        f"{'n_steps':>{widths['n_steps']}} "
        f"{'success':>{widths['success']}} "
        f"{'reward':>{widths['reward']}} "
        f"{'lambda_ss':>{widths['lambda_ss']}}"
    )
    print(header)
    print("-" * len(header))
    for s in summaries:
        seed_s = str(s.seed) if s.seed is not None else "-"
        nsteps_s = str(s.n_steps) if s.n_steps is not None else "-"
        success_s = "-" if s.terminal_success_rate is None else f"{s.terminal_success_rate:.3f}"
        reward_s = "-" if s.mean_reward_final is None else f"{s.mean_reward_final:.3e}"
        lambda_s = "-" if s.lambda_steady_state is None else f"{s.lambda_steady_state:.3f}"
        print(
            f"{s.run_id:<{widths['run_id']}} "
            f"{(s.sub_stage or '-'):<{widths['sub_stage']}} "
            f"{seed_s:>{widths['seed']}} "
            f"{nsteps_s:>{widths['n_steps']}} "
            f"{success_s:>{widths['success']}} "
            f"{reward_s:>{widths['reward']}} "
            f"{lambda_s:>{widths['lambda_ss']}}"
        )
    return 0


def _cmd_summary(args: argparse.Namespace) -> int:
    """``chamber-analyze summary <run-id>`` — terminal stats for one run."""
    summaries = _walk_runs(args.archive_root)
    match = next((s for s in summaries if s.run_id == args.run_id), None)
    if match is None:
        print(f"run_id={args.run_id!r} not found under {args.archive_root}", file=sys.stderr)
        return 2
    if args.json:
        _emit_json(asdict(match), args.output)
        return 0
    for k, v in asdict(match).items():
        print(f"  {k}: {v}")
    return 0


def _cmd_metrics(args: argparse.Namespace) -> int:
    """``chamber-analyze metrics <run-id> --metric <name>`` — time-series dump."""
    summaries = _walk_runs(args.archive_root)
    match = next((s for s in summaries if s.run_id == args.run_id), None)
    if match is None:
        print(f"run_id={args.run_id!r} not found under {args.archive_root}", file=sys.stderr)
        return 2
    rows: list[dict[str, object]] = []
    for line in _read_jsonl(Path(match.jsonl_path)):
        if args.namespace is not None and line.get("metric_namespace") != args.namespace:
            continue
        if args.metric not in line:
            continue
        val = line.get(args.metric)
        if not isinstance(val, (int, float)):
            continue
        rows.append(
            {
                "step": line.get("step"),
                "wall_time": line.get("wall_time"),
                "value": float(val),
            }
        )
    if args.json or args.fmt == "json":
        _emit_json(rows, args.output)
        return 0
    # CSV fallback.
    out = args.output.open("w", encoding="utf-8") if args.output else sys.stdout
    try:
        print("step,wall_time,value", file=out)
        for r in rows:
            print(f"{r['step']},{r['wall_time'] or ''},{r['value']}", file=out)
    finally:
        if out is not sys.stdout:
            out.close()
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    """``chamber-analyze compare <run-a> <run-b> [...] --metrics M1,M2,...``."""
    summaries = _walk_runs(args.archive_root)
    by_id = {s.run_id: s for s in summaries}
    rows: list[dict[str, object]] = []
    for run_id in args.run_ids:
        match = by_id.get(run_id)
        if match is None:
            rows.append({"run_id": run_id, "error": "not_found"})
            continue
        d = asdict(match)
        row: dict[str, object] = {"run_id": run_id}
        for m in args.metrics:
            row[m] = d.get(m)
        rows.append(row)
    if args.json:
        _emit_json(rows, args.output)
        return 0
    # Plain stdout table.
    columns = ["run_id", *args.metrics]
    print(" | ".join(columns))
    print(" | ".join("-" * len(c) for c in columns))
    for row in rows:
        print(" | ".join(str(row.get(c, "-")) for c in columns))
    return 0


def _cmd_plot(args: argparse.Namespace) -> int:
    """``chamber-analyze plot <run-id> [...] --metric <name> --out <path.png>``."""
    summaries = _walk_runs(args.archive_root)
    by_id = {s.run_id: s for s in summaries}
    try:
        import matplotlib  # noqa: PLC0415 - lazy import (viz extra)

        matplotlib.use("Agg")  # headless
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ImportError as exc:
        print(
            f"chamber-analyze plot requires matplotlib (install with "
            f"`uv sync --extra viz`). exc={type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2
    fig, ax = plt.subplots(figsize=(8, 5))
    for run_id in args.run_ids:
        match = by_id.get(run_id)
        if match is None:
            continue
        xs: list[int] = []
        ys: list[float] = []
        for line in _read_jsonl(Path(match.jsonl_path)):
            if args.namespace and line.get("metric_namespace") != args.namespace:
                continue
            if args.metric not in line:
                continue
            val = line.get(args.metric)
            step = line.get("step")
            if isinstance(val, (int, float)) and isinstance(step, int):
                xs.append(step)
                ys.append(float(val))
        if xs:
            ax.plot(xs, ys, label=run_id)
    ax.set_xlabel("step")
    ax.set_ylabel(args.metric)
    if args.title:
        ax.set_title(args.title)
    if args.ymin is not None or args.ymax is not None:
        ax.set_ylim(args.ymin, args.ymax)
    if args.logy:
        ax.set_yscale("log")
    ax.legend(loc="best", fontsize="small")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return 0


def _cmd_rollout_frames(args: argparse.Namespace) -> int:
    """``chamber-analyze rollout-frames <run-id>`` — dump rollout sidecar JSONL.

    Rollout sidecar JSONLs live at
    ``<archive_root>/rollouts/<condition>/step_<NNNNNN>.jsonl``
    per ADR-017 §Schema appendix. Each row is one ``event="rollout_step"``
    record. This subcommand surfaces them to the agent.

    Args:
        args: Parsed arguments. ``run_id`` is informational (rollout
            sidecars are per-condition+step, not per-run_id; the
            ``--archive-root`` flag points at the archive directory).
            ``--episode`` filters by global step. ``--field`` filters
            to one obs/info key on each row.
    """
    rollouts_dir = args.archive_root / "rollouts"
    if not rollouts_dir.exists():
        print(f"No rollouts directory under {args.archive_root}", file=sys.stderr)
        return 2
    candidates = sorted(rollouts_dir.rglob("step_*.jsonl"))
    if args.episode is not None:
        target_stem = f"step_{args.episode:06d}"
        candidates = [p for p in candidates if p.stem == target_stem]
    rows: list[dict[str, object]] = []
    for path in candidates:
        for line in _read_jsonl(path):
            if args.field is not None:
                obs_summary = line.get("obs_summary", {})
                info = line.get("info", {})
                payload = {
                    "step_global": line.get("step_global"),
                    args.field: (
                        obs_summary.get(args.field) if isinstance(obs_summary, dict) else None
                    )
                    if (isinstance(obs_summary, dict) and args.field in obs_summary)
                    else (info.get(args.field) if isinstance(info, dict) else None),
                }
                rows.append(payload)
            else:
                rows.append(line)
    _emit_json(rows, args.out)
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="chamber-analyze",
        description=(
            "Read-only co-analysis CLI for the canonical JSONL archive "
            "(P1.05.11; ADR-017 §Decisions)."
        ),
    )
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    # list-runs
    p_list = subparsers.add_parser("list-runs", help="Enumerate per-cell runs in an archive.")
    p_list.add_argument("--archive-root", type=Path, required=True)
    p_list.add_argument("--json", action="store_true")
    p_list.add_argument("--output", type=Path, default=None)
    p_list.set_defaults(func=_cmd_list_runs)

    # summary
    p_sum = subparsers.add_parser("summary", help="Terminal stats for one run_id.")
    p_sum.add_argument("run_id")
    p_sum.add_argument("--archive-root", type=Path, required=True)
    p_sum.add_argument("--json", action="store_true")
    p_sum.add_argument("--output", type=Path, default=None)
    p_sum.set_defaults(func=_cmd_summary)

    # metrics
    p_m = subparsers.add_parser(
        "metrics", help="Time-series dump for one scalar metric on one run."
    )
    p_m.add_argument("run_id")
    p_m.add_argument("--archive-root", type=Path, required=True)
    p_m.add_argument("--metric", required=True)
    p_m.add_argument("--namespace", default=None)
    p_m.add_argument("--fmt", choices=("json", "csv"), default="json")
    p_m.add_argument("--json", action="store_true", help="Force JSON (overrides --fmt).")
    p_m.add_argument("--output", type=Path, default=None)
    p_m.set_defaults(func=_cmd_metrics)

    # compare
    p_c = subparsers.add_parser("compare", help="Side-by-side comparison across N runs.")
    p_c.add_argument("run_ids", nargs="+")
    p_c.add_argument("--archive-root", type=Path, required=True)
    p_c.add_argument(
        "--metrics",
        type=lambda s: [m.strip() for m in s.split(",") if m.strip()],
        default=[
            "mean_reward_final",
            "terminal_success_rate",
            "lambda_steady_state",
        ],
    )
    p_c.add_argument("--json", action="store_true")
    p_c.add_argument("--output", type=Path, default=None)
    p_c.set_defaults(func=_cmd_compare)

    # plot
    p_p = subparsers.add_parser("plot", help="matplotlib PNG comparing a metric across N runs.")
    p_p.add_argument("run_ids", nargs="+")
    p_p.add_argument("--archive-root", type=Path, required=True)
    p_p.add_argument("--metric", required=True)
    p_p.add_argument("--namespace", default=None)
    p_p.add_argument("--out", type=Path, required=True)
    p_p.add_argument("--title", default=None)
    p_p.add_argument("--ymin", type=float, default=None)
    p_p.add_argument("--ymax", type=float, default=None)
    p_p.add_argument("--logy", action="store_true")
    p_p.set_defaults(func=_cmd_plot)

    # rollout-frames
    p_r = subparsers.add_parser(
        "rollout-frames",
        help="Dump per-step rollout sidecar JSONL records (ADR-017 §Schema appendix).",
    )
    p_r.add_argument("run_id")
    p_r.add_argument("--archive-root", type=Path, required=True)
    p_r.add_argument("--episode", type=int, default=None, help="Filter to a single global step.")
    p_r.add_argument("--field", default=None, help="Filter rows to one obs_summary / info key.")
    p_r.add_argument("--out", type=Path, default=None)
    p_r.set_defaults(func=_cmd_rollout_frames)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``chamber-analyze`` console script (P1.05.11; ADR-017 §Decisions).

    Args:
        argv: Optional list of CLI arguments (testing hook).

    Returns:
        ``0`` on success; ``2`` on a missing-archive or unknown-run_id.
    """
    args = _parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["main"]
