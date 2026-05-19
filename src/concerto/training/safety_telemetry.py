# SPDX-License-Identifier: Apache-2.0
"""Safety-stack λ telemetry aggregator (P1.04.5; ADR-007 §Stage 1b).

The training loop in :func:`concerto.training.ego_aht.train` observes
``state.lambda_`` once per env step when the CBF-QP outer filter is
wired (per-step is too granular to JSONL-log directly: 100k frames x 5
seeds x ~4 cells = 2M lines/axis under naive logging). This module
ships :class:`SafetyAggregator` — a small running-statistics holder
that:

- ``observe(...)`` once per step,
- ``flush_window_stats()`` returns the per-rollout-update window
  aggregate (the training loop emits one ``safety_telemetry`` JSONL
  line per ``cfg.happo.rollout_length`` steps),
- ``finalise(...)`` returns the per-cell ``safety_telemetry_final``
  summary that carries the audit-gate predicate's inputs.

Audit-gate predicates (ADR-007 §Stage 1b implementation-details Rev 7;
``scripts/repro/stage1_{as,om}_stage1b.sh``):

- **Predicate A (saturation guard)** — ``lambda_steady_state <
  saturation_threshold * cartesian_accel_capacity`` → exit 0; otherwise
  exit 8.
- **Predicate B (adaptation invariant, conditional)** — if
  ``lambda_mean > 1e-6`` (λ adapted away from 0), then
  ``lambda_var > 1e-12`` → exit 0; otherwise exit 9 (λ stuck constant).
  If ``lambda_mean <= 1e-6``, the predicate is vacuously satisfied
  (the cell legitimately had no filter-fire activity).

The predicates capture the underlying invariant rather than codifying
condition-specific expectations (the asymmetry across AS-homo vs
AS-hetero / OM-* cells from D11 of the design pass): predicate B fires
on the AS-homo case where λ should have varied but didn't, without
false-positiving on AS-hetero / OM-* cells where λ legitimately stays
at zero (spatial-separation geometry keeps the filter from firing). The
condition-aware variant would have been the same class of foot-gun
ADR-016 §Decision closed by replacing the ``_DEFAULT_STAGE_BY_AXIS``
silent-fallback with the typed ``sub_stage`` field.

Forward-compat note (P1.04.6, next slice):
:class:`SafetyAggregator` will be extended in P1.04.6 with
``n_braking_fires`` and ``braking_fire_rate`` fields to support the
braking-fallback parity comparative audit (CBF-only vs CBF+braking λ
steady-state on a 200-step AS-homo smoke). The current Aggregator's
schema is forward-additive — new fields default to ``0`` for P1.04.5
JSONL records so the P1.04.6 audit can read both vintages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray

#: Threshold below which ``lambda_mean`` is treated as "λ stayed at 0"
#: for predicate B's conditional. 1e-6 is well below the conformal
#: learning-rate η * ε product (10⁻² * 10⁻¹ * O(1) per-pair-loss
#: ≈ 10⁻³ per step at saturation), so any meaningful adaptation
#: crosses this threshold within a handful of steps.
_LAMBDA_ADAPTATION_EPSILON: float = 1e-6

#: Threshold below which ``lambda_var`` is treated as "λ stuck constant"
#: for predicate B. 1e-12 is the float64 quantisation floor for a
#: variance over O(10⁴) observations of a O(10⁻²) signal; anything at
#: or below this indicates the running statistics never moved.
_LAMBDA_STUCK_EPSILON: float = 1e-12

#: Fraction of the final training window over which ``lambda_steady_state``
#: is computed. 10% gives O(10⁴) per-step samples for a 100k-frame
#: training run — enough to suppress within-rollout noise while still
#: tracking late-run dynamics.
_STEADY_STATE_TAIL_FRACTION: float = 0.10


@dataclass
class SafetyAggregator:
    """Rolling λ statistics for the ego-AHT training loop (P1.04.5; ADR-007 §Stage 1b).

    Constructed once per ``(seed, condition)`` cell by
    :class:`chamber.benchmarks.stage1_common.TrainedPolicyFactory` and
    handed to :func:`concerto.training.ego_aht.train`. The training
    loop calls :meth:`observe` once per step, :meth:`flush_window_stats`
    at every ``cfg.happo.rollout_length`` boundary (matching the
    PPO update cadence so the JSONL output aligns naturally with the
    trainer's own ``rollout_update`` events), and :meth:`finalise`
    once at end-of-cell.

    Statistics are accumulated incrementally via Welford-style running
    sums; the per-pair ``lambda_`` vector is reduced to scalar
    aggregates (mean over pairs) before incorporation into the running
    stats. Per-pair detail is retained in the *window* aggregate (the
    flush event carries the per-pair min/max so a future reviewer can
    spot a single saturating pair masked by a non-saturating mean).

    Attributes:
        n_pairs: Number of agent pairs in ``state.lambda_``
            (upper-triangular over the cell's uids).
        cartesian_accel_capacity: ``Bounds.cartesian_accel_capacity``
            for the cell (the audit-gate predicate A's RHS).
        saturation_threshold: ``cfg.safety.saturation_threshold``
            (0.9 by default per :class:`SafetyConfig`).
    """

    n_pairs: int
    cartesian_accel_capacity: float
    saturation_threshold: float = 0.9

    # ----- running statistics (private; reset by flush) -----
    _n_obs_total: int = 0
    _sum_lambda: float = 0.0
    _sum_lambda_sq: float = 0.0
    _max_lambda: float = field(default=-np.inf)
    _min_lambda: float = field(default=np.inf)
    _n_fallback_fires_total: int = 0
    _n_qp_infeasible_total: int = 0
    # Per-rollout-window state (cleared by flush_window_stats).
    _window_n_obs: int = 0
    _window_sum_lambda: float = 0.0
    _window_sum_lambda_sq: float = 0.0
    _window_max_lambda: float = field(default=-np.inf)
    _window_min_lambda: float = field(default=np.inf)
    _window_n_fallback_fires: int = 0
    _window_n_qp_infeasible: int = 0
    # Per-step λ history (for steady-state tail computation at finalise).
    # NOTE: this is the only O(N_steps) state. For the Stage-1b budget
    # (100k frames x 5 seeds x 4 cells), a single cell's history is
    # 100k float64 entries ≈ 800 KB — small enough to keep in memory
    # without a ring buffer. P1.04.6's per-step CPU budget for the
    # comparative audit re-uses this same buffer.
    _lambda_history: list[float] = field(default_factory=list)

    def observe(
        self,
        lambda_: NDArray[np.float64],
        *,
        fallback_fired: bool = False,
        qp_infeasible: bool = False,
    ) -> None:
        """Record one per-step λ vector + filter-event flags (ADR-007 §Stage 1b).

        Args:
            lambda_: Per-pair conformal slack vector
                (``state.lambda_``). Shape ``(n_pairs,)``,
                dtype float64. The aggregator reduces to a scalar
                (mean across pairs) for the running stats; per-pair
                min/max is preserved on the window aggregate.
            fallback_fired: Whether the CBF-QP backbone reported a
                fallback-fire on this step (from
                ``FilterInfo["fallback_fired"]``). Counted into the
                window + total fire-count.
            qp_infeasible: Whether the QP raised
                :class:`concerto.safety.errors.ConcertoSafetyInfeasible`
                on this step. Counted separately so the audit-gate
                can distinguish "fallback fired" (a recoverable event)
                from "QP infeasible" (a worse signal).
        """
        if lambda_.shape != (self.n_pairs,):
            msg = (
                f"SafetyAggregator: lambda_ shape {lambda_.shape} mismatches "
                f"expected ({self.n_pairs},) — the n_pairs constructed at "
                "cell start must match SafetyState.lambda_ size."
            )
            raise ValueError(msg)
        scalar = float(lambda_.mean())
        per_pair_max = float(lambda_.max())
        per_pair_min = float(lambda_.min())
        # Total running stats.
        self._n_obs_total += 1
        self._sum_lambda += scalar
        self._sum_lambda_sq += scalar * scalar
        self._max_lambda = max(self._max_lambda, per_pair_max)
        self._min_lambda = min(self._min_lambda, per_pair_min)
        if fallback_fired:
            self._n_fallback_fires_total += 1
        if qp_infeasible:
            self._n_qp_infeasible_total += 1
        # Window running stats.
        self._window_n_obs += 1
        self._window_sum_lambda += scalar
        self._window_sum_lambda_sq += scalar * scalar
        self._window_max_lambda = max(self._window_max_lambda, per_pair_max)
        self._window_min_lambda = min(self._window_min_lambda, per_pair_min)
        if fallback_fired:
            self._window_n_fallback_fires += 1
        if qp_infeasible:
            self._window_n_qp_infeasible += 1
        # Per-step history for the steady-state tail computation at
        # finalise. Single scalar per step; cheap.
        self._lambda_history.append(scalar)

    def flush_window_stats(self) -> dict[str, object]:
        """Return one ``safety_telemetry`` JSONL record + reset the window.

        P1.04.5; ADR-007 §Stage 1b.

        Called by :func:`concerto.training.ego_aht.train` at every
        ``(step + 1) % cfg.happo.rollout_length == 0`` boundary (one
        event per PPO update). The training loop emits the returned
        dict as a structlog event; the audit-gate hook ignores these
        and reads only the final-cell summary from :meth:`finalise`.

        Returns:
            ``{n_obs, lambda_mean, lambda_var, lambda_max, lambda_min,
            n_fallback_fires, n_qp_infeasible}`` for the window. Window
            state is cleared after the return; the per-cell total
            state is untouched.

            ``n_obs=0`` indicates an empty window (the call site
            invoked flush before any observe). The dict is returned
            with zero / -inf / +inf placeholders for downstream
            JSONL stability — callers should branch on ``n_obs > 0``
            before reading stats.
        """
        if self._window_n_obs == 0:
            return {
                "n_obs": 0,
                "lambda_mean": 0.0,
                "lambda_var": 0.0,
                "lambda_max": 0.0,
                "lambda_min": 0.0,
                "n_fallback_fires": 0,
                "n_qp_infeasible": 0,
            }
        n = self._window_n_obs
        mean = self._window_sum_lambda / n
        var = max(0.0, self._window_sum_lambda_sq / n - mean * mean)
        record: dict[str, object] = {
            "n_obs": n,
            "lambda_mean": mean,
            "lambda_var": var,
            "lambda_max": self._window_max_lambda,
            "lambda_min": self._window_min_lambda,
            "n_fallback_fires": self._window_n_fallback_fires,
            "n_qp_infeasible": self._window_n_qp_infeasible,
        }
        # Reset the window.
        self._window_n_obs = 0
        self._window_sum_lambda = 0.0
        self._window_sum_lambda_sq = 0.0
        self._window_max_lambda = -np.inf
        self._window_min_lambda = np.inf
        self._window_n_fallback_fires = 0
        self._window_n_qp_infeasible = 0
        return record

    def finalise(
        self,
        *,
        safety_enabled: bool,
        predictor_kind: str = "constant_velocity",
    ) -> dict[str, object]:
        """Return the per-cell ``safety_telemetry_final`` summary (P1.04.5).

        Emitted by :func:`concerto.training.ego_aht.train` exactly once
        per cell at end-of-training. The audit-gate hook in
        ``scripts/repro/stage1_{as,om}_stage1b.sh`` reads this record
        from the JSONL and evaluates both predicates:

        - **Predicate A** (saturation):
          ``lambda_steady_state >= saturation_threshold *
          cartesian_accel_capacity`` → exit 8.
        - **Predicate B** (adaptation invariant): if ``lambda_mean >
          ε_adapt`` then ``lambda_var > ε_stuck`` is required; else
          vacuously OK. Exit 9 on "λ stuck".

        Both predicates' inputs ship in the record so the hook is
        pure-JSON-comparison (no recomputation, no off-script logic).

        Args:
            safety_enabled: Echoes ``cfg.safety.enabled``. ``False``
                means the training loop skipped the filter entirely;
                the audit-gate hook detects this and emits a non-
                failing "safety disabled by operator override" path.
            predictor_kind: Echoes ``cfg.safety.predictor_kind``.
                Forward-compat for the AoI-conditioned predictor
                variants flagged in ADR-004 §Open questions.

        Returns:
            The full summary record. Schema (forward-additive — new
            fields default to safe values in older JSONL parsers):

            .. code-block:: python

                {
                    "event": "safety_telemetry_final",
                    "safety_enabled": bool,
                    "predictor_kind": str,
                    "n_filter_calls": int,
                    "n_fallback_fires": int,
                    "n_qp_infeasible": int,
                    "lambda_mean": float,
                    "lambda_var": float,
                    "lambda_max_observed": float,
                    "lambda_min_observed": float,
                    "lambda_steady_state": float,
                    "cartesian_accel_capacity": float,
                    "saturation_threshold": float,
                    "saturated": bool,
                    "n_braking_fires": int,  # P1.04.6 forward-compat (0 here)
                    "braking_fire_rate": float,  # P1.04.6 forward-compat (0.0 here)
                }
        """
        record: dict[str, object] = {
            "event": "safety_telemetry_final",
            "safety_enabled": safety_enabled,
            "predictor_kind": predictor_kind,
            "n_filter_calls": self._n_obs_total,
            "n_fallback_fires": self._n_fallback_fires_total,
            "n_qp_infeasible": self._n_qp_infeasible_total,
            "cartesian_accel_capacity": self.cartesian_accel_capacity,
            "saturation_threshold": self.saturation_threshold,
            # P1.04.6 forward-compat fields. Zero / 0.0 in P1.04.5
            # records; P1.04.6 will populate them and the audit-gate
            # can branch on safety_enabled to know which vintage.
            "n_braking_fires": 0,
            "braking_fire_rate": 0.0,
        }
        if self._n_obs_total == 0:
            # Edge case: zero-step run (Tier-1 fake or cfg.total_frames=0).
            # Emit safe defaults; the audit-gate hook's predicate B
            # vacuously passes (lambda_mean is 0), and predicate A's
            # comparison is 0 < threshold (passes).
            record.update(
                lambda_mean=0.0,
                lambda_var=0.0,
                lambda_max_observed=0.0,
                lambda_min_observed=0.0,
                lambda_steady_state=0.0,
                saturated=False,
            )
            return record
        n = self._n_obs_total
        mean = self._sum_lambda / n
        var = max(0.0, self._sum_lambda_sq / n - mean * mean)
        # Steady-state: mean over the final fraction of the history.
        # int(...) floors; max(1, ...) ensures we read at least one
        # sample for non-empty runs.
        tail_size = max(1, int(n * _STEADY_STATE_TAIL_FRACTION))
        steady_state = float(np.mean(self._lambda_history[-tail_size:]))
        threshold_value = self.saturation_threshold * self.cartesian_accel_capacity
        saturated = steady_state >= threshold_value
        record.update(
            lambda_mean=mean,
            lambda_var=var,
            lambda_max_observed=self._max_lambda,
            lambda_min_observed=self._min_lambda,
            lambda_steady_state=steady_state,
            saturated=saturated,
        )
        return record


__all__ = ["SafetyAggregator"]
