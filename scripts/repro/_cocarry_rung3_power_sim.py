# SPDX-License-Identifier: Apache-2.0
"""Pre-registration power simulation for the Rung-3 PH measurement (R-2026-06-B §15 Rung 3).

Pure Monte-Carlo (no SAPIEN, no incumbent) — run BEFORE the measurement to fix
Δ_min against an archived power analysis. It exercises the *actual* decision
procedure (:func:`chamber.benchmarks.cocarry_ph.cluster_bootstrap_delta` +
:func:`~chamber.benchmarks.cocarry_ph.decide`) on synthetic Bernoulli outcomes:
the reference is the ~100% incumbent ceiling (deterministic success); each of
the K calibrated teammates degrades independently to success rate ``1 - Δ``.
For a sweep of true Δ it reports the power (P[verdict = drop]) and, at Δ = 0,
the type-I error — establishing that 12 paired seed-clusters give the
pre-registered Δ_min = 0.20 adequate power, and that the rule is conservative
under the null. The MDE is the smallest Δ on the grid reaching power >= 0.80.
"""

from __future__ import annotations

import json
import os
import sys
from typing import TYPE_CHECKING

from chamber.benchmarks import cocarry_ph as ph
from concerto.training.seeding import derive_substream

if TYPE_CHECKING:
    import numpy as np

_N_SEEDS = 12
_N_TEAMMATES = 3  # the admitted set size (>= 3)
_N_TRIALS = 400
_N_BOOT = 1500
_DELTA_GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
_TARGET_POWER = 0.80


def _simulate_power(delta: float, rng: np.random.Generator) -> float:
    """Fraction of trials whose verdict is 'drop' at a true pooled effect ``delta``."""
    p_shift = 1.0 - delta
    drops = 0
    for _ in range(_N_TRIALS):
        # Reference = the ~100% incumbent ceiling (deterministic success).
        reference = dict.fromkeys(range(_N_SEEDS), True)
        teammates = {
            f"t{k}": {s: bool(rng.random() < p_shift) for s in range(_N_SEEDS)}
            for k in range(_N_TEAMMATES)
        }
        # Use a per-trial bootstrap root_seed so the trials are independent but
        # each is internally P6-deterministic (the procedure under test).
        boot = ph.cluster_bootstrap_delta(
            reference, teammates, n_boot=_N_BOOT, root_seed=int(rng.integers(0, 2**31 - 1))
        )
        dec = ph.decide(
            pooled_mean_delta=boot["pooled_mean_delta"],
            pooled_ci_lower_one_sided=boot["pooled_ci_lower_one_sided"],
            pooled_ci_upper_for_null=boot["pooled_ci_two_sided"][1],
            positive_control_holds=True,
            any_excluded=True,
            delta_min=ph.DELTA_MIN,
        )
        if dec["verdict"] == ph.VERDICT_DROP:
            drops += 1
    return drops / _N_TRIALS


def main() -> int:
    out_path = os.environ.get(
        "OUT_JSON", "spikes/results/cocarry/rung3/cocarry_rung3_power_sim.json"
    )
    rng = derive_substream("cocarry.rung3.power_sim", root_seed=0).default_rng()
    power_by_delta = {}
    for d in _DELTA_GRID:
        pw = _simulate_power(d, rng)
        power_by_delta[f"{d:.2f}"] = pw
        print(f"    Δ={d:.2f}  power(drop)={pw:.2f}")

    type_i = power_by_delta["0.00"]
    power_at_delta_min = power_by_delta[f"{ph.DELTA_MIN:.2f}"]
    mde = None
    for d in _DELTA_GRID:
        if power_by_delta[f"{d:.2f}"] >= _TARGET_POWER:
            mde = d
            break

    artifact = {
        "schema": "cocarry_rung3_power_sim/v1",
        "purpose": (
            "Pre-registration power analysis fixing Δ_min for the Rung-3 PH "
            "measurement. Monte-Carlo over the ACTUAL decision procedure "
            "(cluster bootstrap one-sided CI + the pre-committed decision rule) on "
            "synthetic Bernoulli outcomes: reference = the ~100% incumbent ceiling, "
            "each of K teammates degraded to success 1-Δ. ADR-026 §Decision 4; "
            "R-2026-06-B §15 Rung 3."
        ),
        "design": {
            "n_seed_clusters": _N_SEEDS,
            "n_teammates": _N_TEAMMATES,
            "n_trials": _N_TRIALS,
            "n_boot": _N_BOOT,
            "ci_alpha": ph.CI_ALPHA,
            "delta_min": ph.DELTA_MIN,
            "reference_model": "deterministic success (incumbent ceiling ~1.0)",
            "teammate_model": "independent Bernoulli(1 - Δ) per seed",
            "decision_rule": "pooled mean Δ >= Δ_min AND one-sided 95% CI lower > 0",
        },
        "power_by_delta": power_by_delta,
        "type_i_error_at_delta_0": type_i,
        "power_at_delta_min": power_at_delta_min,
        "target_power": _TARGET_POWER,
        "mde_at_target_power": mde,
        "interpretation": (
            f"At the pre-registered Δ_min={ph.DELTA_MIN}, 12 paired seed-clusters over "
            f"{_N_TEAMMATES} teammates give power ~{power_at_delta_min:.2f} to declare a drop; "
            f"the minimum detectable effect at {_TARGET_POWER:.0%} power is ~{mde}. Under the "
            f"null (Δ=0) the type-I rate is ~{type_i:.2f} (the one-sided 95% rule is "
            "conservative because the reference is at the 1.0 ceiling). Δ_min=0.20 is a "
            "meaningful cooperation drop that this n can resolve — n is the precision lever, "
            "NOT the effect-size lever (R-2026-06-B §15: treat fewer clusters as exploration)."
        ),
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(artifact, fh, sort_keys=True, indent=2)
    print(
        f"    type-I(Δ=0)={type_i:.2f}  power(Δ_min={ph.DELTA_MIN})={power_at_delta_min:.2f}  "
        f"MDE@{_TARGET_POWER:.0%}={mde}"
    )
    print(f"    artifact -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
