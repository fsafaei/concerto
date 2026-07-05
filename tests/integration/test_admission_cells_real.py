# SPDX-License-Identifier: Apache-2.0
# pyright: reportPrivateImportUsage=false, reportAttributeAccessIssue=false
"""Tier-2 SAPIEN-gated smokes for the admission cell runners (ADR-027 §Admission protocol).

One-episode sanity of the measured cells on a Vulkan/GPU host: the
co-carry scripted cell yields a well-formed episode record with the
wrist stress channel populated, and the pick-place REF-SCRIPT cell
completes an episode. The full preregistered grids run through
``chamber-eval admission`` (the 2a/2b campaign), not through pytest.
"""

from __future__ import annotations

import pytest

from chamber.evaluation.admission import AdmissionCellSpec
from chamber.utils.device import sapien_gpu_available

pytestmark = [
    pytest.mark.slow,
    pytest.mark.gpu,
    pytest.mark.skipif(
        not sapien_gpu_available(),
        reason="Requires Vulkan/GPU (SAPIEN); skipped on CPU-only machines",
    ),
]


def test_cocarry_reference_cell_one_episode() -> None:
    from chamber.benchmarks.admission_cells import run_cocarry_cell

    cell = AdmissionCellSpec(
        cell_id="a1_reference",
        runner="cocarry_scripted",
        policy_id="ref_script_impedance",
        partner_name="cocarry_impedance",
        params={"condition_id": "cocarry_matched_panda_pair", "ego": "impedance"},
    )
    run = run_cocarry_cell(
        cell=cell, seeds=[123], episodes_per_seed=1, root_seed=0, render_backend=None
    )
    (episode,) = run.episodes_by_seed[123]
    assert episode.force_peak is not None
    assert episode.force_peak > 0.0
    assert "ego:cocarry_impedance" in run.partner_hashes


def test_pickplace_reference_cell_one_episode() -> None:
    from chamber.benchmarks.admission_cells import run_pickplace_cell

    cell = AdmissionCellSpec(
        cell_id="a1_reference",
        runner="pickplace_scripted",
        policy_id="ref_script",
        partner_name="scripted_heuristic",
        params={"variant": "reference"},
    )
    run = run_pickplace_cell(
        cell=cell, seeds=[0], episodes_per_seed=1, root_seed=0, render_backend=None
    )
    (episode,) = run.episodes_by_seed[0]
    assert episode.metadata["variant"] == "reference"
    assert episode.success  # the REF-SCRIPT oracle solves the canonical cell
