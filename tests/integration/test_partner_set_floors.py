# SPDX-License-Identifier: Apache-2.0
"""Partner-set floor integration tests (ADR-009 §Decision as amended 2026-07-05).

Every public member of a v1 set must clear its set's committed
matched-pair floor with the reference ego on the committed probe suite
— a member no ego can work with measures nothing (the co-insert Gate-0
lesson: a member that freezes is a wall, not a partner).

Tier split: the handover-place set runs Tier-1 (the env is pure-Python
kinematics — the per-prompt "Tier-1 kinematic path where the env
supports it"); the co-carry and pick-place sets are SAPIEN-gated Tier-2
(GPU host), on a reduced probe schedule to bound runtime — the full
committed schedule is the archived fingerprint run
(``spikes/results/partner-fingerprints/``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import chamber.partners  # noqa: F401 - registers the v1 sets
from chamber.benchmarks.partner_probe import fingerprint_statistics, run_member_probe
from chamber.partners.sets import PartnerSetSpec, get_partner_set, resolve_set_members
from chamber.utils.device import sapien_gpu_available

if TYPE_CHECKING:
    from chamber.evaluation.admission import CellRun

_GPU_GATE = pytest.mark.skipif(
    not sapien_gpu_available(),
    reason="Requires Vulkan/GPU (SAPIEN); skipped on CPU-only machines",
)


def _flatten(run: CellRun) -> list:
    return [ep for seed in sorted(run.episodes_by_seed) for ep in run.episodes_by_seed[seed]]


def _reduced(set_spec: PartnerSetSpec, *, seeds: int, episodes: int) -> PartnerSetSpec:
    """A shorter probe schedule for CI-bounded Tier-2 runs (schedule fields only)."""
    return set_spec.model_copy(
        update={
            "probe_seeds": list(set_spec.probe_seeds)[:seeds],
            "probe_episodes_per_seed": episodes,
        }
    )


class TestHandoverFloorsTier1:
    """Pure-Python kinematic path — the full committed probe schedule."""

    def test_every_public_member_clears_the_floor(self) -> None:
        set_spec = get_partner_set("handover_place_partners", version=1)
        assert set_spec.floor_probe == "free_regrasp"
        for member, params in resolve_set_members(set_spec):
            run = run_member_probe(set_spec, member, params, variant="free_regrasp")
            success = fingerprint_statistics(_flatten(run))["success_rate"]
            assert success >= set_spec.floor, (
                f"{member.member_name}: free-regrasp success {success:.3f} < floor {set_spec.floor}"
            )

    def test_fingerprint_probe_discriminates_members(self) -> None:
        """The anchor-cell fingerprint separates the mismatch family (measured coupling)."""
        set_spec = get_partner_set("handover_place_partners", version=1)
        by_name = {m.member_name: (m, p) for m, p in resolve_set_members(set_spec)}
        member_15, params_15 = by_name["presenter_mismatch_15"]
        member_45, params_45 = by_name["presenter_mismatch_45"]
        fp_15 = fingerprint_statistics(_flatten(run_member_probe(set_spec, member_15, params_15)))
        fp_45 = fingerprint_statistics(_flatten(run_member_probe(set_spec, member_45, params_45)))
        assert fp_45["success_rate"] < fp_15["success_rate"]
        assert fp_45["action_abs_mean"] > fp_15["action_abs_mean"]


@pytest.mark.slow
@pytest.mark.gpu
@_GPU_GATE
class TestCoCarryFloorsTier2:
    def test_every_public_member_clears_the_floor_reduced_schedule(self) -> None:
        set_spec = _reduced(get_partner_set("cocarry_partners", version=1), seeds=1, episodes=4)
        for member, params in resolve_set_members(set_spec):
            run = run_member_probe(
                set_spec, member, params, variant="fingerprint", render_backend="none"
            )
            success = fingerprint_statistics(_flatten(run))["success_rate"]
            assert success >= set_spec.floor, (
                f"{member.member_name}: matched-pair success {success:.3f} < floor {set_spec.floor}"
            )


@pytest.mark.slow
@pytest.mark.gpu
@_GPU_GATE
class TestPickPlaceFloorsTier2:
    def test_every_public_member_clears_the_floor_reduced_schedule(self) -> None:
        set_spec = _reduced(
            get_partner_set("stage1_pickplace_as_partners", version=1), seeds=1, episodes=4
        )
        for member, params in resolve_set_members(set_spec):
            run = run_member_probe(
                set_spec, member, params, variant="fingerprint", render_backend="none"
            )
            success = fingerprint_statistics(_flatten(run))["success_rate"]
            assert success >= set_spec.floor, (
                f"{member.member_name}: matched-pair success {success:.3f} < floor {set_spec.floor}"
            )
