# SPDX-License-Identifier: Apache-2.0
"""ADR-002 risk-mitigation #1 trip-wire: empirical-guarantee assertion (T4b.13).

The empirical-guarantee experiment validates the working hypothesis behind
the project's choice of an ego-PPO substrate (ADR-002 §Decisions): when
the partner is *frozen*, the cooperative-multi-agent task degenerates to
a single-agent MDP from the ego's point of view, and an on-policy PPO
update on the ego's parameters alone is monotone-improving in expectation
(Huriot-Sibai 2025 Theorem 7).

This module ships the deterministic, off-line assertion side of that
experiment. A training run (driven by
:func:`chamber.benchmarks.training_runner.run_training`) produces a
:class:`~concerto.training.ego_aht.RewardCurve`; we then compute the
moving-window-of-K mean of the per-episode reward and check whether at
least ``threshold`` of consecutive intervals are non-decreasing. The
helper returns a :class:`GuaranteeReport` rather than raising — the
caller (the integration test, the ``chamber-spike train --check-guarantee``
CLI in Branch 4) decides what to do.

ADR-002 risk-mitigation #1 / plan/05 §6 criterion 4: if the assertion
fires under the production config, the project must stop and revisit the
algorithm choice — the project does not paper over a failed trip-wire by
widening the window or lowering the threshold. The pure-function shape
of :func:`assert_non_decreasing_window` keeps that contract testable.

Plotting (:func:`plot_reward_curve`) lives behind a lazy import of
``matplotlib`` so the module is importable in environments without the
``viz`` extra. The figure + JSON sidecar pair makes the result both
human-readable and machine-replayable: the PNG goes into
``docs/explanation/why-aht.md``; the JSON contains the curve + report so
a future reader can regenerate the figure exactly.

References:
- ADR-002 §Decisions ("on-policy PPO substrate") + ADR-002 §Risks #1
  ("frozen-partner monotonicity assumption — empirical trip-wire").
- plan/05 §6 criterion 4 (the assertion fires at 100k frames against the
  production ``mpe_cooperative_push.yaml``).
- plan/08 §4 (the helper itself is determinism-friendly — pure-Python
  list arithmetic over a frozen :class:`RewardCurve`).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from itertools import pairwise
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from pathlib import Path

    from concerto.training.ego_aht import RewardCurve

#: Default window size for the moving average (ADR-002 §Risks #1).
#:
#: Plan/05 §6 criterion 4 names "moving-window-of-10". The default is
#: pinned here so callers (integration test, CLI ``--check-guarantee``
#: path) share a single source of truth.
DEFAULT_WINDOW: int = 10

#: Default pass threshold for the fraction of non-decreasing intervals
#: (ADR-002 §Risks #1; plan/05 §6 criterion 4).
DEFAULT_THRESHOLD: float = 0.8


@dataclass(frozen=True)
class GuaranteeReport:
    """Result of one empirical-guarantee assertion (T4b.13; ADR-002 §Risks #1).

    The dataclass is frozen so a returned report cannot be mutated after
    the fact (the project's reproducibility contract: provenance bundles
    are immutable). Serialise to JSON via :func:`dataclasses.asdict`
    (used by :func:`plot_reward_curve`).

    Attributes:
        window: Moving-average window length the report was computed
            against (default :data:`DEFAULT_WINDOW`).
        threshold: Pass threshold the report was checked against (default
            :data:`DEFAULT_THRESHOLD`).
        n_intervals: Number of consecutive-window-pair intervals examined.
            Equals ``max(len(curve) - window, 0)`` for a non-degenerate
            run; zero when the curve is empty or shorter than ``window``.
        n_nondecreasing: Count of intervals whose later-window mean is
            ``>=`` the earlier-window mean.
        fraction: ``n_nondecreasing / n_intervals`` when ``n_intervals > 0``;
            ``1.0`` (vacuously) when ``n_intervals == 0`` so callers can
            treat the degenerate case uniformly.
        passed: ``fraction >= threshold``. Vacuously ``True`` when
            ``n_intervals == 0`` (the curve was too short to assert
            anything; the assertion is documented as vacuous in that
            regime).
    """

    window: int
    threshold: float
    n_intervals: int
    n_nondecreasing: int
    fraction: float
    passed: bool


def assert_non_decreasing_window(
    curve: RewardCurve,
    *,
    window: int = DEFAULT_WINDOW,
    threshold: float = DEFAULT_THRESHOLD,
) -> GuaranteeReport:
    """Check the moving-window-of-K mean for non-decreasing intervals (ADR-002 §Risks #1).

    Slides a window of length ``window`` over ``curve.per_episode_ego_rewards``
    and counts the fraction of consecutive interval pairs whose later-window
    mean is ``>=`` the earlier-window mean. A run is considered to have
    cleared the trip-wire when that fraction is ``>= threshold``.

    Plan/05 §6 criterion 4 names the canonical setting:
    ``window=10, threshold=0.8``. Callers MUST NOT silently widen the
    window or lower the threshold to coerce a passing report — if the
    assertion fires under the production config, hand control back to
    the maintainer and open a ``#scope-revision`` issue (ADR-002 §Risks
    #1 mitigation policy).

    Degenerate inputs (empty curve, ``len(curve) < window``) return a
    report with ``n_intervals=0`` and ``passed=True``. The pass is
    *vacuous* — there is not enough signal to assert anything — and is
    only useful as a sentinel for unit tests of the helper itself; do
    not rely on it to mask a too-short production run.

    Args:
        curve: The reward curve produced by
            :func:`concerto.training.ego_aht.train`. Only the
            ``per_episode_ego_rewards`` field is read; the per-step
            series is deliberately ignored because the assertion is
            episode-aggregated (Huriot-Sibai 2025 Theorem 7 is stated
            over the policy-improvement step, which lives at episode
            granularity).
        window: Window size (default :data:`DEFAULT_WINDOW`).
        threshold: Pass threshold on the non-decreasing fraction
            (default :data:`DEFAULT_THRESHOLD`).

    Returns:
        Frozen :class:`GuaranteeReport`. Does NOT raise on failure — the
        caller decides whether to fail the test, exit the CLI non-zero,
        or just log the report.
    """
    rewards = list(curve.per_episode_ego_rewards)
    if len(rewards) < window or window <= 0:
        return GuaranteeReport(
            window=window,
            threshold=threshold,
            n_intervals=0,
            n_nondecreasing=0,
            fraction=1.0,
            passed=True,
        )
    # Compute consecutive window means; n_intervals = len(means) - 1
    # because we count *pairs* of adjacent windows.
    means: list[float] = []
    running_sum = sum(rewards[:window])
    means.append(running_sum / window)
    for i in range(window, len(rewards)):
        running_sum += rewards[i] - rewards[i - window]
        means.append(running_sum / window)
    n_intervals = len(means) - 1
    n_nondecreasing = sum(1 for prev, nxt in pairwise(means) if nxt >= prev)
    fraction = n_nondecreasing / n_intervals if n_intervals > 0 else 1.0
    return GuaranteeReport(
        window=window,
        threshold=threshold,
        n_intervals=n_intervals,
        n_nondecreasing=n_nondecreasing,
        fraction=fraction,
        passed=fraction >= threshold,
    )


def plot_reward_curve(
    curve: RewardCurve,
    *,
    report: GuaranteeReport,
    out_dir: Path,
) -> tuple[Path, Path]:
    """Emit ``<run_id>.png`` + ``<run_id>.json`` for the assertion plot (T4b.13; ADR-002 §Risks #1).

    Writes two files under ``out_dir``:

    - ``<run_id>.png``: matplotlib line plot of the per-episode reward
      with the moving-window-of-``report.window`` mean overlaid. The
      report's ``fraction`` / ``passed`` are stamped into the title so
      ``docs/explanation/why-aht.md`` can embed the figure unchanged.
    - ``<run_id>.json``: machine-replayable sidecar carrying the raw
      per-episode curve + the report. A future reader regenerates the
      figure by replaying the JSON through this function.

    ``matplotlib`` is imported lazily inside the function so the
    surrounding module stays importable in environments without the
    ``viz`` extra (Phase-0 CI, headless containers). Callers that need
    the plot must install the ``viz`` extra (``uv sync --extra viz``).

    Args:
        curve: The reward curve produced by
            :func:`concerto.training.ego_aht.train`. ``curve.run_id``
            is used as the filename stem.
        report: The :class:`GuaranteeReport` produced by
            :func:`assert_non_decreasing_window`. Stamped into the figure
            title and copied into the JSON sidecar.
        out_dir: Output directory. Created if missing.

    Returns:
        ``(png_path, json_path)`` tuple — the absolute paths of the two
        emitted files.
    """
    # Lazy import: matplotlib is in the ``viz`` extra (pyproject.toml),
    # not the base install. Importing at module top would force every
    # caller of ``concerto.training`` to install matplotlib even when
    # they only want the assertion helper (e.g. CI, ``chamber-spike
    # train --check-guarantee``).
    import matplotlib  # noqa: PLC0415  # see lazy-import comment above.

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt  # noqa: PLC0415  # ditto.

    out_dir.mkdir(parents=True, exist_ok=True)
    rewards = list(curve.per_episode_ego_rewards)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    if rewards:
        ax.plot(range(len(rewards)), rewards, label="per-episode ego reward", alpha=0.4)
        if len(rewards) >= report.window > 0:
            running_sum = sum(rewards[: report.window])
            means: list[float] = [running_sum / report.window]
            for i in range(report.window, len(rewards)):
                running_sum += rewards[i] - rewards[i - report.window]
                means.append(running_sum / report.window)
            ax.plot(
                range(report.window - 1, len(rewards)),
                means,
                label=f"moving mean (window={report.window})",
                linewidth=2,
            )
    ax.set_xlabel("episode index")
    ax.set_ylabel("ego reward (sum over episode)")
    status = "PASS" if report.passed else "FAIL"
    ax.set_title(
        f"Empirical guarantee — run {curve.run_id} — {status} "
        f"({report.n_nondecreasing}/{report.n_intervals} "
        f"non-decreasing, threshold={report.threshold:.2f})"
    )
    if rewards:
        ax.legend(loc="lower right")
    ax.grid(visible=True, alpha=0.3)
    fig.tight_layout()
    png_path = out_dir / f"{curve.run_id}.png"
    fig.savefig(png_path, dpi=120)
    plt.close(fig)

    json_path = out_dir / f"{curve.run_id}.json"
    json_path.write_text(
        json.dumps(
            {
                "run_id": curve.run_id,
                "per_episode_ego_rewards": rewards,
                "report": asdict(report),
            },
            sort_keys=True,
            indent=2,
        ),
        encoding="utf-8",
    )
    return png_path, json_path


__all__ = [
    "DEFAULT_THRESHOLD",
    "DEFAULT_WINDOW",
    "GuaranteeReport",
    "assert_non_decreasing_window",
    "plot_reward_curve",
]
