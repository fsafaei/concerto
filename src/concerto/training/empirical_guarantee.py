# SPDX-License-Identifier: Apache-2.0
# pyright: reportAttributeAccessIssue=false
#
# Suppresses pyright's complaints about ``scipy.stats.linregress``
# attribute access. The function returns a ``LinregressResult`` named
# tuple with ``slope`` / ``intercept`` / ``rvalue`` / ``pvalue`` /
# ``stderr`` / ``intercept_stderr`` fields (documented at
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.linregress.html),
# but scipy's stub annotates the return as a private named tuple class
# whose attribute surface pyright cannot resolve. The fields ARE the
# canonical public API. Same noqa rationale as
# ``src/chamber/benchmarks/ego_ppo_trainer.py``'s pyright header.
"""ADR-002 risk-mitigation #1 trip-wire: empirical-guarantee assertion (T4b.13).

The empirical-guarantee experiment validates the working hypothesis behind
the project's choice of an ego-PPO substrate (ADR-002 §Decisions): when
the partner is *frozen*, the cooperative-multi-agent task degenerates to
a single-agent MDP from the ego's point of view, and an on-policy PPO
update on the ego's parameters alone is monotone-improving in expectation
(Huriot-Sibai 2025 Theorem 7).

This module ships two assertion shapes:

- :func:`assert_positive_learning_slope` — **canonical**. A least-squares
  linear regression of per-episode reward against episode index, with a
  one-sided significance test on the slope. Robust to per-episode noise
  because it integrates over the whole curve; an empirical-guarantee
  pass means "the curve has a statistically significant positive trend
  with confidence ``1 - alpha``". This is the gate the 100k integration
  test runs in production.
- :func:`assert_non_decreasing_window` — **legacy comparison**. The
  Phase-0 moving-window-of-K non-decreasing-fraction statistic.
  Retained because its threshold/window invariants and unit tests are
  the project's reference for what the original Pardo-2017-aware trip-
  wire shape looked like before issue #62. Not the gate in CI; do not
  base decisions on it without a justified diff against the slope test.

Both helpers return frozen reports (:class:`GuaranteeReport` /
:class:`SlopeReport`) rather than raising — the caller (the integration
test, the ``chamber-spike train --check-guarantee`` CLI in M4b-9b)
decides what to do.

ADR-002 risk-mitigation #1 / plan/05 §6 criterion 4: if the canonical
assertion fires under the production config, the project must stop and
revisit the algorithm choice — the project does not paper over a failed
trip-wire by lowering the alpha threshold or shortening the budget. The
pure-function shape of both helpers keeps that contract testable.

Plotting helpers (:func:`plot_reward_curve` and :func:`plot_slope_curve`)
live behind a lazy import of ``matplotlib`` so the module is importable
in environments without the ``viz`` extra. The figure + JSON sidecar
pair makes the result both human-readable and machine-replayable: the
PNG goes into ``docs/explanation/why-aht.md``; the JSON contains the
curve + report so a future reader can regenerate the figure exactly.

References:
- ADR-002 §Decisions ("on-policy PPO substrate") + ADR-002 §Risks #1
  ("frozen-partner monotonicity assumption — empirical trip-wire").
- plan/05 §6 criterion 4 (the canonical slope assertion runs at 100k
  frames against the production ``mpe_cooperative_push.yaml``).
- plan/08 §4 (the helpers themselves are determinism-friendly — pure
  arithmetic over a frozen :class:`RewardCurve`).
- Issue #62 — root-cause writeup for why the moving-window-of-K
  statistic is unfavorable on per-episode rewards at this signal/noise
  ratio, and why the slope test replaces it as the canonical gate.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from itertools import pairwise
from typing import TYPE_CHECKING

import numpy as np
from scipy import stats

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


#: Default minimum number of episodes the slope test requires (ADR-002 §Risks #1).
#:
#: Below this, the linear regression's p-value loses statistical power
#: and the assertion returns a vacuous pass to avoid false alarms on
#: too-short curves (e.g. unit-test smoke runs at ``total_frames=100``).
#: The Phase-0 production config runs ~2000 episodes per 100k frames, well
#: above this floor.
DEFAULT_MIN_EPISODES: int = 20

#: Default significance threshold for the slope assertion (ADR-002 §Risks #1).
#:
#: ``alpha = 0.05`` matches the convention in Pardo et al. 2017 and the
#: empirical PPO literature; the test is one-sided (slope > 0), so the
#: report compares ``p_value / 2`` against ``alpha`` and additionally
#: requires ``slope > 0``. The canonical 100k-frame run gives
#: ``p_value ≪ 1e-10`` across seeds, so the project sits comfortably
#: above any reasonable alpha; the parameter is exposed so unit tests
#: can pin the closed-form behavior at the boundary.
DEFAULT_ALPHA: float = 0.05


@dataclass(frozen=True)
class SlopeReport:
    """Result of one slope-test empirical-guarantee assertion (T4b.13; ADR-002 §Risks #1).

    Captures everything an external reader needs to interpret the trip-
    wire outcome without re-running the experiment: the least-squares
    slope itself, the coefficient of determination, the one-sided
    p-value, the alpha threshold the run was checked against, the number
    of episodes the regression saw, and whether the run cleared the
    gate.

    The dataclass is frozen so a returned report cannot be mutated after
    the fact (the project's reproducibility contract: provenance bundles
    are immutable). Serialise to JSON via :func:`dataclasses.asdict`
    (used by :func:`plot_slope_curve`).

    Attributes:
        slope: Least-squares slope of per-episode reward versus episode
            index, in reward-units-per-episode. Positive ⇒ learning;
            zero or negative ⇒ no learning or regressing.
        intercept: Least-squares intercept. Useful for the regression
            overlay in :func:`plot_slope_curve` and as a sanity check
            on the early-rollout reward scale.
        r_squared: Coefficient of determination ``R²``. Low ``R²`` plus
            ``passed=True`` is acceptable on noisy tasks (the slope is
            still statistically significant); the project does not gate
            on ``R²`` directly.
        p_value: One-sided p-value for the null "slope <= 0", computed
            as ``two_sided_p / 2`` when the observed slope is positive
            and ``1 - two_sided_p / 2`` when non-positive. The
            two-sided p-value comes from
            :func:`scipy.stats.linregress`.
        alpha: Significance threshold the assertion was checked against
            (default :data:`DEFAULT_ALPHA`).
        n_episodes: Number of episodes in the regressed curve. Below
            :data:`DEFAULT_MIN_EPISODES` the report is vacuously passing.
        passed: ``slope > 0`` and ``p_value < alpha``. Vacuously
            ``True`` when ``n_episodes`` is below the minimum.
    """

    slope: float
    intercept: float
    r_squared: float
    p_value: float
    alpha: float
    n_episodes: int
    passed: bool


def assert_positive_learning_slope(
    curve: RewardCurve,
    *,
    alpha: float = DEFAULT_ALPHA,
    min_episodes: int = DEFAULT_MIN_EPISODES,
) -> SlopeReport:
    """Least-squares slope test on per-episode reward (ADR-002 §Risks #1; canonical).

    Issue #62 root-cause writeup: the legacy
    :func:`assert_non_decreasing_window` statistic has poor signal-to-
    noise on per-episode rewards at the Phase-0 100k-frame budget — the
    moving-window-of-10 mean stdev (~6.3) is comparable to the total
    drift (~10-20 reward units) the trainer can produce, and the
    non-decreasing-fraction sits near 0.5 (a random walk) even when the
    policy is materially improving. The slope test integrates over the
    whole curve and rejects the null "slope <= 0" with p-values
    astronomically below any reasonable alpha when learning is real.

    The test:

    1. Fit ``rewards ~ slope * episode_index + intercept`` via least
       squares (:func:`scipy.stats.linregress`).
    2. Take the one-sided p-value: ``two_sided / 2`` when the observed
       slope is positive, otherwise ``1 - two_sided / 2``. (scipy
       reports two-sided by default; the one-sided form is the natural
       gate for "positive learning".)
    3. Pass when ``slope > 0`` AND ``p_value < alpha``.

    Plan/05 §6 criterion 4 uses ``alpha=0.05`` as the canonical
    threshold; the project's production config gives ``p_value < 1e-14``
    across seeds, so the gate has very wide margin.

    Degenerate inputs:

    - Empty curve, or fewer than ``min_episodes`` entries: vacuously
      passing with ``slope=0``, ``p_value=1.0``. Documented as a
      sentinel for too-short unit-test smoke runs; do not rely on it to
      mask an undersized production run.
    - Constant (zero-variance) curve: ``slope=0`` and ``p_value=1.0``;
      report is failing because the slope is not strictly positive.

    Args:
        curve: The reward curve produced by
            :func:`concerto.training.ego_aht.train`. Only
            ``per_episode_ego_rewards`` is read.
        alpha: One-sided significance threshold. Default
            :data:`DEFAULT_ALPHA`.
        min_episodes: Minimum episode count for a non-vacuous report.
            Default :data:`DEFAULT_MIN_EPISODES`.

    Returns:
        Frozen :class:`SlopeReport`. Does NOT raise on failure — the
        caller decides whether to fail the test, exit the CLI non-zero,
        or just log the report.
    """
    rewards = np.asarray(list(curve.per_episode_ego_rewards), dtype=np.float64)
    n = int(rewards.shape[0])
    if n < min_episodes:
        return SlopeReport(
            slope=0.0,
            intercept=0.0,
            r_squared=0.0,
            p_value=1.0,
            alpha=alpha,
            n_episodes=n,
            passed=True,
        )
    # scipy.stats.linregress raises on zero-variance x or zero-variance
    # y. The episode index x = arange(n) always has positive variance
    # for n >= 2 (which is guaranteed by min_episodes >= 2), so we only
    # need to handle the constant-y case.
    if float(rewards.std()) == 0.0:
        return SlopeReport(
            slope=0.0,
            intercept=float(rewards[0]) if n > 0 else 0.0,
            r_squared=0.0,
            p_value=1.0,
            alpha=alpha,
            n_episodes=n,
            passed=False,
        )
    x = np.arange(n, dtype=np.float64)
    result = stats.linregress(x, rewards)
    slope = float(result.slope)
    intercept = float(result.intercept)
    r_squared = float(result.rvalue) ** 2
    two_sided_p = float(result.pvalue)
    # One-sided p for "slope > 0". scipy reports two-sided; halving is
    # correct under the standard t-distribution symmetry. For a
    # non-positive observed slope the one-sided p flips to the
    # complement so the assertion correctly fails.
    p_value = two_sided_p / 2.0 if slope > 0.0 else 1.0 - two_sided_p / 2.0
    passed = slope > 0.0 and p_value < alpha
    return SlopeReport(
        slope=slope,
        intercept=intercept,
        r_squared=r_squared,
        p_value=p_value,
        alpha=alpha,
        n_episodes=n,
        passed=passed,
    )


def plot_slope_curve(
    curve: RewardCurve,
    *,
    report: SlopeReport,
    out_dir: Path,
) -> tuple[Path, Path]:
    """Emit ``<run_id>.png`` + ``<run_id>.json`` for the slope plot (T4b.13; ADR-002 §Risks #1).

    Mirrors :func:`plot_reward_curve`'s contract but overlays the
    least-squares regression line instead of the moving-window mean.
    The status line in the title carries the slope, p-value, and
    pass/fail flag so ``docs/explanation/why-aht.md`` can embed the
    figure unchanged.

    Args:
        curve: The reward curve produced by
            :func:`concerto.training.ego_aht.train`. ``curve.run_id``
            is the filename stem.
        report: The :class:`SlopeReport` produced by
            :func:`assert_positive_learning_slope`. Stamped into the
            figure title and copied into the JSON sidecar.
        out_dir: Output directory. Created if missing.

    Returns:
        ``(png_path, json_path)`` tuple — the absolute paths of the two
        emitted files.
    """
    # Lazy import: matplotlib is in the ``viz`` extra (pyproject.toml).
    # See the docstring + import on :func:`plot_reward_curve` for the
    # rationale.
    import matplotlib  # noqa: PLC0415  # see lazy-import comment above.

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt  # noqa: PLC0415  # ditto.

    out_dir.mkdir(parents=True, exist_ok=True)
    rewards = list(curve.per_episode_ego_rewards)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    if rewards:
        x = list(range(len(rewards)))
        ax.plot(x, rewards, label="per-episode ego reward", alpha=0.35)
        ax.plot(
            x,
            [report.slope * xi + report.intercept for xi in x],
            label=(
                f"least-squares slope = {report.slope:+.4f}/episode (R²={report.r_squared:.3f})"
            ),
            linewidth=2,
        )
    ax.set_xlabel("episode index")
    ax.set_ylabel("ego reward (sum over episode)")
    status = "PASS" if report.passed else "FAIL"
    ax.set_title(
        f"Empirical guarantee (slope test) — run {curve.run_id} — {status} "
        f"(p={report.p_value:.2e}, alpha={report.alpha:.2g}, "
        f"n={report.n_episodes})"
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
    "DEFAULT_ALPHA",
    "DEFAULT_MIN_EPISODES",
    "DEFAULT_THRESHOLD",
    "DEFAULT_WINDOW",
    "GuaranteeReport",
    "SlopeReport",
    "assert_non_decreasing_window",
    "assert_positive_learning_slope",
    "plot_reward_curve",
    "plot_slope_curve",
]
