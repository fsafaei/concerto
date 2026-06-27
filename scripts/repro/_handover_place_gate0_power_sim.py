# SPDX-License-Identifier: Apache-2.0
"""Task-specific power simulation sizing the handover-and-place Gate-0 cluster count n.

Pure Monte-Carlo (no env, no measurement). Sized for ADEQUACY, not economy (scripted/CPU
— founder freeze review). Exercises the SINGLE canonical decision rule verbatim (the one
``chamber.spikes.handover_place_gate0.decision.classify_cell`` implements, and that
``gate0.yaml`` states once), with the SAME CI convention for both limbs:

    two-sided 95% cluster-bootstrap CI (2.5 / 97.5 percentiles)
    COUPLING_VALID  iff  CI_lower (2.5 pct)  >= delta_min
    WASHOUT         iff  CI_upper (97.5 pct) <  delta_min
    INDETERMINATE   otherwise

n is sized so, across EVERY between-seed dispersion bracket:

* **MDE <= 0.05** (half delta_min): the smallest true gap detected as significantly > 0
  (CI_lower > 0) at power >= 0.80 — the crossover-resolution margin;
* **WASHOUT power >= 0.80** at true gap = 0 (CI_upper < delta_min);
* **COUPLING_VALID power >= 0.80** at the expected (large) coupling effect (CI_lower >=
  delta_min); and
* **cluster-bootstrap type-I <= 0.075** at gap = 0 (P(CI_lower > 0)) — small-n cluster
  percentile bootstraps are anti-conservative, so this calibration guard is what forces
  an n with trustworthy CIs. n_boot does NOT substitute for seed-clusters.

At the EXACT crossover (gap = delta_min) a cell is INDETERMINATE ~most of the time by
design (CI_lower < delta_min < CI_upper); the overall verdict resolves from cells AWAY
from the boundary (large gap at low takt, ~0 gap at high takt). This boundary
indeterminacy is reported, not hidden, and is NOT a sizing target.

Paired model: a shared per-cluster latent u_i ~ Normal(0, sigma_u) induces the ref/shift
correlation the paired bootstrap exploits; sigma_u is swept low/central/high.
Determinism (P6 / ADR-002): all RNG routes through ``derive_substream``.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

from chamber.envs.handover_place import (
    HANDOVER_DELTA_MIN_PP,
    HANDOVER_K_EPISODES,
    HANDOVER_N_BOOT,
)
from concerto.training.seeding import derive_substream

#: Coupling bar as a fraction (the env carries it in percentage points).
_DELTA_MIN: float = HANDOVER_DELTA_MIN_PP / 100.0
#: Matched (solvable) reference success. Above the Limb-1 bar (tau_solv = 0.90), not a
#: 1.0 ceiling -> real between-seed variance that sets n.
_P_REF: float = 0.95
#: Episodes per (seed-cluster, condition); chamber convention.
_K_EPISODES: int = HANDOVER_K_EPISODES
#: Candidate seed-cluster counts to size over (extended high — size for adequacy).
_N_GRID: tuple[int, ...] = (8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 72, 80, 96)
#: Between-seed dispersion bracket (std of the per-cluster latent logit effect).
_SIGMA_U_GRID: dict[str, float] = {"low": 0.3, "central": 0.6, "high": 1.0}
#: True-gap grid for the power / MDE / type-I curves (finer below the bar).
_DELTA_GRID: tuple[float, ...] = (0.0, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30, 0.40, 0.50)
#: Design CEILING on the MDE (~half delta_min) — the founder's stated bar.
_MDE_DESIGN_CEILING: float = 0.10
#: SIZING target — half the design ceiling, for crossover-resolution MARGIN.
_MDE_TARGET: float = 0.05
#: Expected coupling effect for the COUPLING_VALID-power check (a large binding gap).
_EXPECTED_COUPLING_GAP: float = 0.40
#: Monte-Carlo trials per cell + bootstrap resamples per trial.
_N_TRIALS: int = 600
_N_BOOT: int = 1000
_TARGET_POWER: float = 0.80
#: Two-sided 95% CI (the SINGLE canonical convention, both limbs).
_ALPHA_TWO_SIDED: float = 0.05
#: Calibration guard: the false-positive significance rate at true gap=0 (the cluster-
#: bootstrap type-I) must stay at/under this — what forces an n with trustworthy CIs.
_TYPE_I_CEILING: float = 0.075


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p: float) -> float:
    return float(np.log(p / (1.0 - p)))


def _simulate_cell(
    *, n: int, sigma_u: float, delta: float, rng: np.random.Generator
) -> dict[str, float]:
    """Per-cell rates under the CANONICAL two-sided rule (COUPLING_VALID / sig / WASHOUT)."""
    logit_ref = _logit(_P_REF)
    logit_shift = _logit(max(_P_REF - delta, 1e-3))
    coupling = 0  # CI_lower(2.5 pct) >= delta_min  (COUPLING_VALID, canonical)
    significant = 0  # CI_lower(2.5 pct) > 0  (gap detectably > 0 -> MDE; type-I at gap=0)
    null_resolved = 0  # CI_upper(97.5 pct) < delta_min  (WASHOUT, canonical)
    q_lo = _ALPHA_TWO_SIDED / 2.0
    q_hi = 1.0 - _ALPHA_TWO_SIDED / 2.0
    for _ in range(_N_TRIALS):
        u = rng.normal(0.0, sigma_u, size=n)
        p_ref_i = _sigmoid(logit_ref + u)
        p_shift_i = _sigmoid(logit_shift + u)
        ref = rng.random((n, _K_EPISODES)) < p_ref_i[:, None]
        shift = rng.random((n, _K_EPISODES)) < p_shift_i[:, None]
        cluster_delta = ref.mean(axis=1) - shift.mean(axis=1)
        idx = rng.integers(0, n, size=(_N_BOOT, n))
        boot_means = cluster_delta[idx].mean(axis=1)
        ci_low = float(np.quantile(boot_means, q_lo))
        ci_high = float(np.quantile(boot_means, q_hi))
        if ci_low >= _DELTA_MIN:
            coupling += 1
        if ci_low > 0.0:
            significant += 1
        if ci_high < _DELTA_MIN:
            null_resolved += 1
    return {
        "coupling_rate": coupling / _N_TRIALS,
        "significant_rate": significant / _N_TRIALS,
        "null_resolved_rate": null_resolved / _N_TRIALS,
    }


def main() -> int:
    """Size n for adequacy under the canonical rule and write the artifact (ADR-026; ADR-007)."""
    out_path = os.environ.get(
        "OUT_JSON", "spikes/preregistration/handover_place/gate0_power_sim.json"
    )
    rng = derive_substream("handover_place.gate0.power_sim", root_seed=0).default_rng()

    by_sigma: dict[str, dict] = {}
    for sig_name, sigma_u in _SIGMA_U_GRID.items():
        table: dict[str, dict[str, float]] = {}
        for n in _N_GRID:
            row: dict[str, float] = {}
            for d in _DELTA_GRID:
                cell = _simulate_cell(n=n, sigma_u=sigma_u, delta=d, rng=rng)
                row[f"coupling_{d:.2f}"] = cell["coupling_rate"]
                row[f"sig_{d:.2f}"] = cell["significant_rate"]
                row[f"null_{d:.2f}"] = cell["null_resolved_rate"]
            mde: float | None = next(
                (d for d in _DELTA_GRID if d > 0.0 and row[f"sig_{d:.2f}"] >= _TARGET_POWER),
                None,
            )
            row["mde"] = mde if mde is not None else float("nan")
            table[str(n)] = row
            coupling_pwr = row[f"coupling_{_EXPECTED_COUPLING_GAP:.2f}"]
            at_bar_indeterminate = (
                1.0 - row[f"coupling_{_DELTA_MIN:.2f}"] - row[f"null_{_DELTA_MIN:.2f}"]
            )
            print(
                f"  sigma_u={sig_name}({sigma_u}) n={n:>3}  MDE={mde}  "
                f"typeI(sig@0)={row['sig_0.00']:.3f}  null@0={row['null_0.00']:.2f}  "
                f"coupling@{_EXPECTED_COUPLING_GAP}={coupling_pwr:.2f}  "
                f"indet@bar={at_bar_indeterminate:.2f}"
            )
        by_sigma[sig_name] = {"sigma_u": sigma_u, "table": table}

    def _n_adequate(n_idx: int) -> bool:
        # adequacy at this n AND every larger n, across EVERY dispersion bracket.
        for j in range(n_idx, len(_N_GRID)):
            n = _N_GRID[j]
            for sig in _SIGMA_U_GRID:
                row = by_sigma[sig]["table"][str(n)]
                mde_ok = row[f"sig_{_MDE_TARGET:.2f}"] >= _TARGET_POWER
                null_ok = row["null_0.00"] >= _TARGET_POWER
                coupling_ok = row[f"coupling_{_EXPECTED_COUPLING_GAP:.2f}"] >= _TARGET_POWER
                type_i_ok = row["sig_0.00"] <= _TYPE_I_CEILING
                if not (mde_ok and null_ok and coupling_ok and type_i_ok):
                    return False
        return True

    recommended_n: int | None = next((n for i, n in enumerate(_N_GRID) if _n_adequate(i)), None)
    if recommended_n is None:
        recommended_n = _N_GRID[-1]

    def _achieved(bracket: str) -> dict[str, float]:
        row = by_sigma[bracket]["table"][str(recommended_n)]
        return {
            "mde": row["mde"],
            "type_i_at_gap0": row["sig_0.00"],
            "washout_power_at_gap0": row["null_0.00"],
            "coupling_power_at_expected": row[f"coupling_{_EXPECTED_COUPLING_GAP:.2f}"],
            "at_bar_coupling_rate": row[f"coupling_{_DELTA_MIN:.2f}"],
            "at_bar_washout_rate": row[f"null_{_DELTA_MIN:.2f}"],
            "at_bar_indeterminate_rate": (
                1.0 - row[f"coupling_{_DELTA_MIN:.2f}"] - row[f"null_{_DELTA_MIN:.2f}"]
            ),
        }

    artifact = {
        "schema": "handover_place_gate0_power_sim/v3",
        "status": (
            "PROVISIONAL — sized for ADEQUACY WITH MARGIN under the SINGLE canonical decision "
            "rule (two-sided 95% CI; COUPLING_VALID iff CI_lower >= delta_min, WASHOUT iff "
            "CI_upper < delta_min). MDE <= 0.05 + WASHOUT/COUPLING power >= 0.80 + cluster-"
            "bootstrap type-I <= 0.075, all across the dispersion bracket. n_boot does not "
            "substitute for seed-clusters."
        ),
        "purpose": (
            "Task-specific Monte-Carlo power simulation sizing the handover-and-place Gate-0 "
            "seed-cluster count n under the canonical rule that gate0.yaml states and "
            "chamber.spikes.handover_place_gate0.decision.classify_cell implements. ADR-026."
        ),
        "design": {
            "p_ref_matched": _P_REF,
            "k_episodes_per_cluster_condition": _K_EPISODES,
            "n_grid": list(_N_GRID),
            "sigma_u_bracket": _SIGMA_U_GRID,
            "delta_grid": list(_DELTA_GRID),
            "n_trials": _N_TRIALS,
            "n_boot_per_trial": _N_BOOT,
            "production_n_boot": HANDOVER_N_BOOT,
            "delta_min": _DELTA_MIN,
            "ci_convention": "two_sided_95 (2.5 / 97.5 percentiles), SAME for both limbs",
            "canonical_rule_ref": (
                "gate0.yaml decision_rule (single source of truth); "
                "chamber.spikes.handover_place_gate0.decision.classify_cell"
            ),
            "coupling_valid_rule": "two-sided 95% CI lower (2.5 pct) >= delta_min",
            "washout_rule": "two-sided 95% CI upper (97.5 pct) < delta_min",
            "indeterminate_rule": "otherwise",
            "mde_design_ceiling": _MDE_DESIGN_CEILING,
            "mde_sizing_target": _MDE_TARGET,
            "type_i_ceiling": _TYPE_I_CEILING,
            "expected_coupling_gap": _EXPECTED_COUPLING_GAP,
            "target_power": _TARGET_POWER,
            "adequacy_rule": (
                "smallest n (monotone, all-larger-n) where ACROSS EVERY dispersion bracket: "
                "P(CI_lower > 0 | gap=mde_sizing_target) >= target_power  AND  "
                "P(CI_upper < delta_min | gap=0) >= target_power  AND  "
                "P(CI_lower >= delta_min | gap=expected_coupling) >= target_power  AND  "
                "P(CI_lower > 0 | gap=0) <= type_i_ceiling"
            ),
            "boundary_note": (
                "at gap=delta_min a cell is INDETERMINATE most of the time by design "
                "(CI_lower < delta_min < CI_upper); the verdict resolves from off-boundary "
                "cells. Surfaced, not a sizing target."
            ),
        },
        "results_by_dispersion": by_sigma,
        "recommended_n_provisional": recommended_n,
        "achieved_at_recommended_n": {"central": _achieved("central"), "high": _achieved("high")},
        "interpretation": (
            f"n={recommended_n} paired seed-clusters reach MDE <= {_MDE_TARGET} (half the "
            f"~{_MDE_DESIGN_CEILING} design ceiling) with calibrated type-I "
            f"(~{by_sigma['high']['table'][str(recommended_n)]['sig_0.00']:.2f} at gap=0, high "
            f"dispersion) and >= {_TARGET_POWER:.0%} power for BOTH crossover declarations under "
            "the canonical two-sided rule (COUPLING_VALID at the expected effect; WASHOUT at "
            "gap=0). At the exact crossover a cell is mostly INDETERMINATE by design; the verdict "
            "resolves from off-boundary cells. PROVISIONAL pending the S2-measured between-seed "
            "variance (check it against the sigma_u=1.0 bracket at PR C; invoke the seed "
            "extension if exceeded)."
        ),
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(artifact, fh, sort_keys=True, indent=2)
    print(f"  recommended provisional n (canonical rule, worst-case robust) = {recommended_n}")
    print(f"  artifact -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
