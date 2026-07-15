# SPDX-License-Identifier: Apache-2.0
"""Tier-1 (no-SAPIEN-scene) tests for the co-carry decouple-ablation wiring (ADR-026; R-2026-06-C).

Covers the pure-Python decouple-condition wiring of
``scripts/repro/_cocarry_decouple_ablation.py`` without building a SAPIEN scene:

- **C1 spring-off**: ``drive_stiffness=0`` resolves to a zero-force coupling via
  :func:`chamber.envs.cocarry.cocarry_coupling` (stiffness 0, derived damping 0,
  unbounded force limit — the compliant freed-axis spring transmits no force),
  distinct from the rigid ``None`` default.
- **C2 partner-removed**: the zero-action partner seat constructs from the
  registered instrument (``partner_ablated_zero``, ``action_dim`` 8 — the same
  class the admission A2 cell used) and emits all-zero float32 actions
  regardless of the observation.
- The pre-committed decision rule (COOPERATION_CONTINGENT / COUPLING_TRIVIAL /
  INDETERMINATE), the seed-disjointness guard, and the locked prereg's grid
  endpoints behave per the pre-registration.

ADR-026 §Decision 1-2; ADR-026 §Open-questions (coupling stiffness is a task
parameter); R-2026-06-C condition 5.
"""

from __future__ import annotations

import functools
import importlib.util
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from types import ModuleType

_REPO = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO / "scripts/repro/_cocarry_decouple_ablation.py"
_PREREG = _REPO / "spikes/preregistration/cocarry_decouple_ablation_prereg.json"


@functools.lru_cache(maxsize=1)
def _driver() -> ModuleType:
    """Load the repro driver by path (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location("_cocarry_decouple_ablation", _SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@functools.lru_cache(maxsize=1)
def _prereg() -> dict[str, Any]:
    return json.loads(_PREREG.read_text("utf-8"))


# --------------------------------------------------------------------------
# C1 — decoupled (spring off).
# --------------------------------------------------------------------------


class TestC1SpringOff:
    """K=0 resolves to a coupling that transmits no force (ADR-026 §Open-questions)."""

    def test_k_zero_is_zero_force_spring(self) -> None:
        from chamber.envs.cocarry import cocarry_coupling

        stiffness, damping, force_limit = cocarry_coupling(0.0)
        assert stiffness == 0.0
        assert damping == 0.0  # derived at the rig's fixed ratio: 0 * 0.1
        assert force_limit > 0  # unbounded sentinel, not a clamp to zero

    def test_k_zero_differs_from_rigid_default(self) -> None:
        from chamber.envs.cocarry import cocarry_coupling

        rigid_stiffness, rigid_damping, _ = cocarry_coupling(None)
        assert rigid_stiffness > 0.0
        assert rigid_damping > 0.0

    def test_prereg_down_sweep_endpoints(self) -> None:
        prereg = _prereg()
        grid = prereg["conditions"]["C3_down_sweep"]["drive_stiffness_grid_npm"]
        assert grid[0] == prereg["fixed_base_config"]["baseline_drive_stiffness"] == 8000.0
        assert grid[-1] == 0.0  # the C1 endpoint
        assert grid == sorted(grid, reverse=True)  # a DOWN-sweep, monotone


# --------------------------------------------------------------------------
# C2 — partner removed (zero-action seat, coupling intact).
# --------------------------------------------------------------------------


class TestC2NoopPartner:
    """The zero-action seat constructs from the registered instrument (ADR-027 A2 precedent)."""

    def test_constructs_registered_instrument(self) -> None:
        from chamber.partners.ablation import PartnerAblatedZero

        partner = _driver().build_noop_partner(seed=71000)
        assert isinstance(partner, PartnerAblatedZero)

    def test_emits_all_zero_action_of_panda_width(self) -> None:
        partner = _driver().build_noop_partner(seed=71000)
        partner.reset(seed=71000)
        action = partner.act({"agent": {}, "extra": {}})
        assert action.shape == (8,)  # 7 arm joint deltas + gripper
        assert action.dtype == np.float32
        assert np.all(action == 0.0)

    def test_ignores_observation_and_is_deterministic(self) -> None:
        partner = _driver().build_noop_partner(seed=71001)
        a1 = partner.act({})
        a2 = partner.act({"agent": {"panda_partner": {"qpos": np.ones(9)}}})
        np.testing.assert_array_equal(a1, a2)


# --------------------------------------------------------------------------
# The pre-committed decision rule + guards.
# --------------------------------------------------------------------------


class TestDecisionRule:
    """The falsifiable-both-ways rule matches the locked prereg bounds (2026-07-14 form)."""

    @staticmethod
    def _verdict(c0: float, c1: float, c2: float, gap_lo: float) -> str:
        bounds = _prereg()["decision_rule"]["bounds"]
        return _driver().decouple_verdict(
            c0_rate=c0,
            c1_rate=c1,
            c2_rate=c2,
            gap_ci_lower=gap_lo,
            c0_min=bounds["c0_min"],
            c2_max=bounds["c2_max"],
            gap_ci_lower_min=bounds["gap_ci_lower_min"],
            trivial_min=bounds["trivial_min"],
        )

    def test_cooperation_contingent(self) -> None:
        assert self._verdict(1.0, 0.0, 0.0, 0.9) == "COOPERATION_CONTINGENT"

    def test_cooperation_contingent_at_exact_bounds(self) -> None:
        assert self._verdict(0.90, 0.0, 0.10, 0.50) == "COOPERATION_CONTINGENT"

    def test_contingent_does_not_require_c1_collapse(self) -> None:
        # C1 mid is nuance reported via the dose-response, not a contingency conjunct.
        assert self._verdict(1.0, 0.5, 0.0, 0.9) == "COOPERATION_CONTINGENT"

    def test_coupling_trivial_via_c2(self) -> None:
        assert self._verdict(1.0, 0.0, 1.0, -0.05) == "COUPLING_TRIVIAL"
        assert self._verdict(1.0, 0.0, 0.90, 0.0) == "COUPLING_TRIVIAL"

    def test_coupling_trivial_via_c1_takes_precedence(self) -> None:
        # Base wins with no coupling at all: a trivial-solve even if C2 collapses.
        assert self._verdict(1.0, 0.95, 0.0, 0.9) == "COUPLING_TRIVIAL"
        assert self._verdict(1.0, 0.90, 0.0, 0.9) == "COUPLING_TRIVIAL"

    def test_indeterminate_middle_ground(self) -> None:
        assert self._verdict(1.0, 0.0, 0.5, 0.3) == "INDETERMINATE"

    def test_indeterminate_when_gap_ci_too_wide(self) -> None:
        # Collapse in point estimates but an under-powered gap bound: NOT contingent.
        assert self._verdict(1.0, 0.0, 0.0, 0.49) == "INDETERMINATE"

    def test_broken_anchor_cannot_be_contingent(self) -> None:
        assert self._verdict(0.5, 0.0, 0.0, 0.9) == "INDETERMINATE"


class TestSeedDisjointness:
    """The pre-registered block is disjoint from every prior co-carry seed set."""

    def test_prereg_block_is_disjoint(self) -> None:
        seeds = [int(s) for s in _prereg()["seeds"]]
        assert seeds == list(range(71000, 71020))
        assert _driver().verify_seed_disjointness(seeds) == {}

    def test_collision_is_flagged(self) -> None:
        collisions = _driver().verify_seed_disjointness([70010])
        assert "base_probe" in collisions
        assert "rung4b_coupling_sweep" in collisions
        assert collisions["base_probe"] == [70010]


class TestRateBootstrap:
    """The per-cell seed-bootstrap CI is deterministic (P6) and sane at the edges."""

    def test_degenerate_all_success(self) -> None:
        per_seed = dict.fromkeys(range(71000, 71020), True)
        lo, hi = _driver().bootstrap_rate_ci(per_seed, n_boot=200)
        assert (lo, hi) == (1.0, 1.0)

    def test_deterministic_across_calls(self) -> None:
        per_seed = {s: (s % 3 == 0) for s in range(71000, 71020)}
        assert _driver().bootstrap_rate_ci(per_seed, n_boot=200) == _driver().bootstrap_rate_ci(
            per_seed, n_boot=200
        )
