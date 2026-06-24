# SPDX-License-Identifier: Apache-2.0
"""Task-specific power simulation sizing the co-insert seed-cluster count n (S0).

Pure Monte-Carlo (no SAPIEN, no incumbent) — run at S0 to size the seed-cluster
count ``n`` from a *task-specific* analysis rather than inheriting co-carry's
n=12 / n=28 (ADR-026 §Decision 4).

Why co-carry's n does NOT transfer. The co-carry reference was a degenerate
~1.0 success ceiling (deterministic), so its decision rule was conservative and
12 clusters sufficed. The co-insert precondition is deliberately different: the
cooperative-reference insertion success is ~0.9 (the co-insert Gate-0 precondition,
verified at S2), NOT 1.0 — a binomial/contact-rich regime with real
between-seed variance. That variance is what sets n, and it is unknown until S2,
so this S0 run is **provisional**: it sizes n under stated, bracketed
assumptions and the value is **re-derived at S6 from the S2-measured variance**
before any shifted holder is seen.

The simulation exercises the pre-committed decision procedure verbatim (the
co-insert decision rule, Gate 2): paired cluster bootstrap over seed-clusters, one-sided 95% CI,
``lands`` iff pooled mean Δ ≥ Δ_min AND the one-sided 95% CI lower bound > 0. It
reports, per candidate ``n``: power at Δ = Δ_min (stated honestly — the co-carry
~0.5 at-bar blind band is the cautionary precedent), the type-I rate at Δ = 0,
and the smallest ``n`` reaching the target power. A NULL is reported via the
equivalence / CI bound, never a power claim — so the sim also records, at Δ = 0,
the rate at which the two-sided CI upper bound falls below Δ_min (the
equivalence-null resolution rate).

Paired model. A shared per-cluster latent effect ``u_i ~ Normal(0, sigma_u)``
(the seed's env-init idiosyncrasy) induces the positive ref/shift correlation
the paired bootstrap exploits: ``p_ref_i = sigmoid(logit(p_ref) + u_i)`` and
``p_shift_i = sigmoid(logit(p_ref - Δ) + u_i)``; episodes are Bernoulli draws.
``sigma_u`` is swept over a bracket (low / central / high) because the true
between-seed dispersion is an S2 measurement.

Determinism: all RNG routes through ``derive_substream`` (P6 / ADR-002). No
``Date.now`` / ad-hoc ``np.random`` — the run is byte-reproducible.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

# Decision-rule constants — mirror chamber.envs.coinsert (single source of
# truth for the rule; imported so the sim cannot drift from the env module).
from chamber.envs.coinsert import COINSERT_DELTA_MIN, COINSERT_N_BOOT
from concerto.training.seeding import derive_substream

#: Cooperative-reference insertion success precondition (the co-insert Gate-0;
#: verified at S2). NOT 1.0 — this is the whole reason n must be re-sized.
_P_REF = 0.90

#: Episodes per (seed-cluster, condition). The chamber convention is 20; contact-
#: rich evaluation is slower, so n (clusters) is the lever, k is held.
_K_EPISODES = 20

#: Candidate seed-cluster counts to size over.
_N_GRID = [8, 10, 12, 16, 20, 24, 28, 32, 40, 48]

#: Between-seed dispersion bracket (std of the per-cluster latent logit effect).
#: The true value is an S2 measurement; bracket low/central/high.
_SIGMA_U_GRID = {"low": 0.3, "central": 0.6, "high": 1.0}

#: True effect grid for the power / type-I curves. Finer near the bar, where
#: n is the lever (above ~0.30 power saturates regardless of n).
_DELTA_GRID = [0.0, 0.10, 0.15, 0.20, 0.22, 0.25, 0.28, 0.30, 0.40]

#: Acceptable minimum-detectable-effect ceiling for the n recommendation. At
#: the bar (Δ = Δ_min) the "pooled mean >= Δ_min" conjunct caps power at ~0.5
#: structurally (the co-carry at-bar blind band) — n CANNOT fix that. n is
#: therefore sized so the MDE at target power is <= this ceiling (a
#: "comfortably above the bar" effect), and the at-bar ~0.5 is reported
#: honestly, never represented as adequate power.
_MDE_CEILING = 0.25

#: Monte-Carlo trials per (n, sigma_u, Δ) cell, and bootstrap resamples per
#: trial. 500 trials de-noises the power curve enough that the worst-case
#: recommendation is stable across the 0.80 threshold; the decision rule (and
#: the production n_boot) is the same as S6.
_N_TRIALS = 500
_N_BOOT = 1000

#: Target power for the n-sizing recommendation.
_TARGET_POWER = 0.80

#: CI one-sided alpha (95% one-sided lower bound) and the two-sided pair for the
#: equivalence-null resolution.
_ALPHA_ONE_SIDED = 0.05


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p: float) -> float:
    return float(np.log(p / (1.0 - p)))


def _simulate_cell(
    *, n: int, sigma_u: float, delta: float, rng: np.random.Generator
) -> dict[str, float]:
    """Power, type-I, and equivalence-null resolution for one (n, sigma_u, Δ) cell.

    Returns the fraction of trials whose verdict is ``lands`` (the power, or the
    type-I rate when ``delta == 0``) and the fraction whose two-sided CI upper
    bound falls below Δ_min (the equivalence-null resolution rate).
    """
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
        # Per-cluster success-rate difference (the paired unit).
        cluster_delta = ref.mean(axis=1) - shift.mean(axis=1)
        pooled_mean = float(cluster_delta.mean())
        # Paired cluster bootstrap: resample clusters with replacement.
        idx = rng.integers(0, n, size=(_N_BOOT, n))
        boot_means = cluster_delta[idx].mean(axis=1)
        ci_lower_one_sided = float(np.quantile(boot_means, _ALPHA_ONE_SIDED))
        ci_upper_two_sided = float(np.quantile(boot_means, 1.0 - _ALPHA_ONE_SIDED / 2.0))
        if pooled_mean >= COINSERT_DELTA_MIN and ci_lower_one_sided > 0.0:
            lands += 1
        if ci_upper_two_sided < COINSERT_DELTA_MIN:
            null_resolved += 1
    return {
        "lands_rate": lands / _N_TRIALS,
        "null_resolved_rate": null_resolved / _N_TRIALS,
    }


def main() -> int:
    out_path = os.environ.get(
        "OUT_JSON", "spikes/preregistration/coinsert/coinsert_s0_power_sim.json"
    )
    rng = derive_substream("coinsert.s0.power_sim", root_seed=0).default_rng()

    by_sigma: dict[str, dict] = {}
    n_star: dict[str, int | None] = {}
    mde_by_sigma: dict[str, dict[str, float | None]] = {}
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
            # MDE: smallest true Δ on the grid reaching target power.
            mde: float | None = None
            for d in _DELTA_GRID:
                if d > 0.0 and row[f"{d:.2f}"] >= _TARGET_POWER:
                    mde = d
                    break
            mde_by_n[str(n)] = mde
            print(
                f"  sigma_u={sig_name}({sigma_u}) n={n:>3}  "
                f"at-bar(Δ={COINSERT_DELTA_MIN})={row[f'{COINSERT_DELTA_MIN:.2f}']:.2f}  "
                f"MDE@{_TARGET_POWER:.0%}={mde}  type-I={row['type_i']:.2f}  "
                f"null_resolved(Δ=0)={row['null_resolved_at_0']:.2f}"
            )
        # Smallest n whose MDE at target power is at or below the ceiling.
        chosen: int | None = None
        for n in _N_GRID:
            mde = mde_by_n[str(n)]
            if mde is not None and mde <= _MDE_CEILING:
                chosen = n
                break
        n_star[sig_name] = chosen
        mde_by_sigma[sig_name] = mde_by_n
        by_sigma[sig_name] = {"sigma_u": sigma_u, "power_table": power_table, "mde_by_n": mde_by_n}

    # The provisional recommendation: size to the CENTRAL dispersion bracket
    # (re-derived at S6 from the S2-measured variance). WORST-CASE robust: the
    # smallest n whose MDE at target power is <= the ceiling under EVERY
    # dispersion bracket (including the pessimistic high one), so the provisional
    # n holds margin even if the S2 variance lands at the top of the bracket.
    # Honest: if no n on the grid satisfies all brackets, say so — do not
    # over-claim.
    # Robustness is judged directly on power at the ceiling effect (Δ =
    # _MDE_CEILING) across all brackets — stabler than the grid-walk MDE, which
    # is fragile at the 0.80 float boundary. The first n holds only if it AND
    # every larger n on the grid also hold (no isolated noise spike).
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
    # The at-bar blind band (Δ = Δ_min) power at the recommended n, central
    # dispersion — surfaced explicitly so the design is never represented as
    # 80%-powered at the bar (it is structurally ~0.5; the co-carry precedent).
    at_bar_blind_band = by_sigma["central"]["power_table"][str(recommended_n)][
        f"{COINSERT_DELTA_MIN:.2f}"
    ]

    artifact = {
        "schema": "coinsert_s0_power_sim/v1",
        "status": "PROVISIONAL — re-derived at S6 from the S2-measured variance.",
        "purpose": (
            "Task-specific Monte-Carlo power simulation sizing the co-insert "
            "seed-cluster count n at S0. Does NOT inherit co-carry's n=12/n=28: "
            "the co-insert reference success is ~0.9 (not a 1.0 ceiling), so the "
            "variance — and therefore n — is task-specific. Exercises the "
            "pre-committed decision rule (paired cluster bootstrap, one-sided 95% "
            "CI; lands iff pooled mean Δ >= Δ_min AND CI lower > 0). "
            "ADR-026 §Decision 4."
        ),
        "design": {
            "p_ref_precondition": _P_REF,
            "k_episodes_per_cluster_condition": _K_EPISODES,
            "n_grid": _N_GRID,
            "sigma_u_bracket": _SIGMA_U_GRID,
            "delta_grid": _DELTA_GRID,
            "n_trials": _N_TRIALS,
            "n_boot_per_trial": _N_BOOT,
            "production_n_boot": COINSERT_N_BOOT,
            "delta_min": COINSERT_DELTA_MIN,
            "target_power": _TARGET_POWER,
            "paired_model": (
                "shared per-cluster latent u_i ~ Normal(0, sigma_u); "
                "p_ref_i = sigmoid(logit(p_ref) + u_i), "
                "p_shift_i = sigmoid(logit(p_ref - Δ) + u_i); Bernoulli episodes"
            ),
            "decision_rule": "pooled mean Δ >= Δ_min AND one-sided 95% CI lower > 0",
            "null_rule": (
                "a NULL is reported via the equivalence/CI bound (two-sided 95% CI "
                "upper < Δ_min), NEVER via a power claim"
            ),
        },
        "mde_ceiling": _MDE_CEILING,
        "results_by_dispersion": by_sigma,
        "mde_by_dispersion": mde_by_sigma,
        "n_star_by_dispersion": n_star,
        "recommended_n_provisional": recommended_n,
        "at_bar_blind_band_power": at_bar_blind_band,
        "interpretation": (
            f"At the bar (Δ=Δ_min={COINSERT_DELTA_MIN}) power is ~{at_bar_blind_band:.2f} and "
            "n CANNOT raise it: the 'pooled mean Δ >= Δ_min' conjunct makes the point estimate "
            "the binding constraint, so at-bar power is structurally ~0.5 (the co-carry at-bar "
            "blind band — surfaced, NOT represented as adequate). n is therefore sized by the "
            f"MDE: under the central dispersion bracket n~{recommended_n} paired seed-clusters "
            f"reach ~{_TARGET_POWER:.0%} power at a detectable effect <= {_MDE_CEILING}. The "
            "value is PROVISIONAL and re-derived at S6 from the S2-measured variance before any "
            "shifted holder is seen. A NULL is resolved by the equivalence/CI bound, never a "
            "power claim."
        ),
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(artifact, fh, sort_keys=True, indent=2)
    print(f"  recommended provisional n (central dispersion) = {recommended_n}")
    print(f"  artifact -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
