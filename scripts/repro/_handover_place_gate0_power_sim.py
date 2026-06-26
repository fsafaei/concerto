# SPDX-License-Identifier: Apache-2.0
"""Task-specific power simulation sizing the handover-and-place Gate-0 cluster count n.

Pure Monte-Carlo (no env, no measurement) — run at PR B to size the seed-cluster count
``n`` per (clearance, takt) cell from a task-specific analysis, exercising the
pre-committed coupling rule verbatim: paired cluster bootstrap over seed-clusters,
one-sided 95% CI; coupling-valid iff pooled mean gap >= delta_min AND the one-sided 95%
CI lower bound > 0. A NULL is resolved by the equivalence / CI bound (two-sided upper <
delta_min), never a power claim.

Why a task-specific n. The matched (solvable) reference success is high but not a 1.0
ceiling (~0.95, the Limb-1 bar is tau_solv = 0.90), so there is real between-seed
variance; that variance sets n and is bracketed low/central/high. At-bar power
(gap = delta_min) is structurally ~0.5 and is surfaced honestly, NEVER represented as
adequate; n is sized by the minimum-detectable-effect ceiling.

Determinism (P6 / ADR-002): all RNG routes through ``derive_substream``. No ad-hoc
``np.random``; the run is byte-reproducible.
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
#: Candidate seed-cluster counts to size over.
_N_GRID: tuple[int, ...] = (8, 10, 12, 16, 20, 24, 28, 32, 40, 48)
#: Between-seed dispersion bracket (std of the per-cluster latent logit effect).
_SIGMA_U_GRID: dict[str, float] = {"low": 0.3, "central": 0.6, "high": 1.0}
#: True-gap grid for the power / type-I curves (finer near the bar).
_DELTA_GRID: tuple[float, ...] = (0.0, 0.10, 0.15, 0.20, 0.22, 0.25, 0.28, 0.30, 0.40)
#: MDE ceiling for the n recommendation (at-bar power is structurally ~0.5; n cannot fix
#: that, so size by the MDE).
_MDE_CEILING: float = 0.25
#: Monte-Carlo trials per cell + bootstrap resamples per trial.
_N_TRIALS: int = 500
_N_BOOT: int = 1000
_TARGET_POWER: float = 0.80
_ALPHA_ONE_SIDED: float = 0.05


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p: float) -> float:
    return float(np.log(p / (1.0 - p)))


def _simulate_cell(
    *, n: int, sigma_u: float, delta: float, rng: np.random.Generator
) -> dict[str, float]:
    logit_ref = _logit(_P_REF)
    logit_shift = _logit(max(_P_REF - delta, 1e-3))
    lands = 0
    null_resolved = 0
    for _ in range(_N_TRIALS):
        u = rng.normal(0.0, sigma_u, size=n)
        p_ref_i = _sigmoid(logit_ref + u)
        p_shift_i = _sigmoid(logit_shift + u)
        ref = rng.random((n, _K_EPISODES)) < p_ref_i[:, None]
        shift = rng.random((n, _K_EPISODES)) < p_shift_i[:, None]
        cluster_delta = ref.mean(axis=1) - shift.mean(axis=1)
        pooled_mean = float(cluster_delta.mean())
        idx = rng.integers(0, n, size=(_N_BOOT, n))
        boot_means = cluster_delta[idx].mean(axis=1)
        ci_lower = float(np.quantile(boot_means, _ALPHA_ONE_SIDED))
        ci_upper = float(np.quantile(boot_means, 1.0 - _ALPHA_ONE_SIDED / 2.0))
        if pooled_mean >= _DELTA_MIN and ci_lower > 0.0:
            lands += 1
        if ci_upper < _DELTA_MIN:
            null_resolved += 1
    return {"lands_rate": lands / _N_TRIALS, "null_resolved_rate": null_resolved / _N_TRIALS}


def main() -> int:
    """Size n per dispersion bracket and write the power-sim artifact (ADR-026; ADR-007)."""
    out_path = os.environ.get(
        "OUT_JSON", "spikes/preregistration/handover_place/gate0_power_sim.json"
    )
    rng = derive_substream("handover_place.gate0.power_sim", root_seed=0).default_rng()

    by_sigma: dict[str, dict] = {}
    for sig_name, sigma_u in _SIGMA_U_GRID.items():
        power_table: dict[str, dict[str, float]] = {}
        mde_by_n: dict[str, float | None] = {}
        for n in _N_GRID:
            row: dict[str, float] = {}
            for d in _DELTA_GRID:
                cell = _simulate_cell(n=n, sigma_u=sigma_u, delta=d, rng=rng)
                row[f"{d:.2f}"] = cell["lands_rate"]
                if d == 0.0:
                    row["type_i"] = cell["lands_rate"]
                    row["null_resolved_at_0"] = cell["null_resolved_rate"]
            power_table[str(n)] = row
            mde: float | None = None
            for d in _DELTA_GRID:
                if d > 0.0 and row[f"{d:.2f}"] >= _TARGET_POWER:
                    mde = d
                    break
            mde_by_n[str(n)] = mde
            print(
                f"  sigma_u={sig_name}({sigma_u}) n={n:>3}  "
                f"at-bar={row[f'{_DELTA_MIN:.2f}']:.2f}  MDE@{_TARGET_POWER:.0%}={mde}  "
                f"type-I={row['type_i']:.2f}"
            )
        by_sigma[sig_name] = {"sigma_u": sigma_u, "power_table": power_table, "mde_by_n": mde_by_n}

    ceiling_key = f"{_MDE_CEILING:.2f}"

    def _all_brackets_pass(n_idx: int) -> bool:
        return all(
            by_sigma[sig]["power_table"][str(_N_GRID[j])][ceiling_key] >= _TARGET_POWER
            for j in range(n_idx, len(_N_GRID))
            for sig in _SIGMA_U_GRID
        )

    robust_n: int | None = None
    for i, n in enumerate(_N_GRID):
        if _all_brackets_pass(i):
            robust_n = n
            break
    recommended_n = robust_n if robust_n is not None else _N_GRID[-1]
    at_bar = by_sigma["central"]["power_table"][str(recommended_n)][f"{_DELTA_MIN:.2f}"]

    artifact = {
        "schema": "handover_place_gate0_power_sim/v1",
        "status": "PROVISIONAL — sizes n per (clearance, takt) cell before any measured run.",
        "purpose": (
            "Task-specific Monte-Carlo power simulation sizing the handover-and-place "
            "Gate-0 seed-cluster count n. Exercises the pre-committed coupling rule "
            "(paired cluster bootstrap, one-sided 95% CI; coupling-valid iff pooled mean "
            "gap >= delta_min AND CI lower > 0). ADR-026 §Decision."
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
            "target_power": _TARGET_POWER,
            "decision_rule": "pooled mean gap >= delta_min AND one-sided 95% CI lower > 0",
            "null_rule": (
                "two-sided 95% CI upper < delta_min (equivalence bound); never a power claim"
            ),
        },
        "mde_ceiling": _MDE_CEILING,
        "results_by_dispersion": by_sigma,
        "recommended_n_provisional": recommended_n,
        "at_bar_blind_band_power": at_bar,
        "interpretation": (
            f"At the bar (gap=delta_min={_DELTA_MIN}) power is ~{at_bar:.2f} and n cannot "
            "raise it (the point estimate is the binding constraint); at-bar power is "
            "structurally ~0.5 (surfaced, NOT represented as adequate). n is sized by the "
            f"MDE: under the central bracket n~{recommended_n} paired seed-clusters reach "
            f"~{_TARGET_POWER:.0%} power at a detectable gap <= {_MDE_CEILING}. The value is "
            "PROVISIONAL; a null is resolved by the equivalence/CI bound, never a power claim."
        ),
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(artifact, fh, sort_keys=True, indent=2)
    print(f"  recommended provisional n (worst-case robust) = {recommended_n}")
    print(f"  artifact -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
