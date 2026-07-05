# SPDX-License-Identifier: Apache-2.0
"""One-seed power pilots for the CB admission preregs (ADR-027 §Admission protocol).

Runs each admission cell for ONE pilot seed (disjoint from every
measurement seed) and records success statistics + wall-clock cost, so
the admission pre-registrations can justify their episode budgets from
measured CI width rather than assertion (ADR-007 §Discipline: the
pilot is explicitly ``run_purpose: power`` and is excluded from every
admission report).

Usage::

    uv run python scripts/repro/_admission_power_pilot.py --task cocarry
    uv run python scripts/repro/_admission_power_pilot.py --task stage1_pickplace_as

Writes ``spikes/preregistration/admission/pilots/<task>_power_pilot.json``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from chamber.benchmarks.admission_cells import resolve_cell_runner
from chamber.evaluation.admission import AdmissionCellSpec, stress_statistics
from chamber.evaluation.bundles import compute_summary

#: The single pilot seed — disjoint from the committed measurement seed
#: ranges (co-carry 90000-90006, pick-place 91000-91006).
PILOT_SEED = 89900

#: Pilot episode budget per cell.
PILOT_EPISODES = 12

_CELLS: dict[str, list[AdmissionCellSpec]] = {
    "cocarry": [
        AdmissionCellSpec(
            cell_id="pilot_a1_reference",
            runner="cocarry_scripted",
            policy_id="ref_script_impedance",
            partner_name="cocarry_impedance",
            params={"condition_id": "cocarry_matched_panda_pair", "ego": "impedance"},
        ),
        AdmissionCellSpec(
            cell_id="pilot_a2_single_arm",
            runner="cocarry_scripted",
            policy_id="ref_script_impedance",
            partner_name="partner_ablated_zero",
            params={"condition_id": "cocarry_single_arm_positive_control", "ego": "impedance"},
        ),
        AdmissionCellSpec(
            cell_id="pilot_a3_blind",
            runner="cocarry_scripted",
            policy_id="b_blind_impedance",
            partner_name="cocarry_impedance",
            params={"condition_id": "cocarry_matched_panda_pair", "ego": "blind"},
        ),
    ],
    "stage1_pickplace_as": [
        AdmissionCellSpec(
            cell_id="pilot_a1_reference",
            runner="pickplace_scripted",
            policy_id="ref_script",
            partner_name="scripted_heuristic",
            params={"variant": "reference"},
        ),
        AdmissionCellSpec(
            cell_id="pilot_a2_partner_ablated",
            runner="pickplace_scripted",
            policy_id="ref_script",
            partner_name="partner_ablated_zero",
            params={"variant": "partner_ablated"},
        ),
        AdmissionCellSpec(
            cell_id="pilot_a3_partner_blind",
            runner="pickplace_scripted",
            policy_id="b_blind",
            partner_name="scripted_heuristic",
            params={"variant": "partner_blind"},
        ),
    ],
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=sorted(_CELLS))
    args = parser.parse_args()

    out: dict[str, object] = {
        "run_purpose": "power",
        "task_id": args.task,
        "pilot_seed": PILOT_SEED,
        "episodes": PILOT_EPISODES,
        "excluded_from_report": True,
        "repro_command": (
            f"uv run python scripts/repro/_admission_power_pilot.py --task {args.task}"
        ),
        "cells": {},
    }
    cells: dict[str, object] = {}
    for cell in _CELLS[args.task]:
        runner = resolve_cell_runner(cell.runner)
        start = time.perf_counter()
        run = runner(
            cell=cell,
            seeds=[PILOT_SEED],
            episodes_per_seed=PILOT_EPISODES,
            root_seed=0,
            render_backend=None,
        )
        elapsed = time.perf_counter() - start
        episodes = [ep for eps in run.episodes_by_seed.values() for ep in eps]
        summary = compute_summary(episodes, n_resamples=2000, bootstrap_root_seed=0)
        cells[cell.cell_id] = {
            "n_episodes": summary.n_episodes,
            "success_mean": summary.success_mean,
            "success_iqm": summary.success_iqm,
            "success_ci_low": summary.success_ci_low,
            "success_ci_high": summary.success_ci_high,
            "stress": stress_statistics(episodes, successes_only=False),
            "wallclock_s": round(elapsed, 2),
        }
        print(
            f"{cell.cell_id}: mean {summary.success_mean:.3f} "
            f"IQM {summary.success_iqm:.3f} [{summary.success_ci_low:.3f}, "
            f"{summary.success_ci_high:.3f}] in {elapsed:.1f}s"
        )
    out["cells"] = cells

    dest = Path("spikes/preregistration/admission/pilots") / f"{args.task}_power_pilot.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
