# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``chamber.partners.selection`` (T4.8).

Covers ADR-009 §Consequences (Phase-0 draft zoo) and ADR-009 §Validation
criteria (Phase-1 select_zoo stub).
"""

from __future__ import annotations

import pytest

from chamber.partners.api import PartnerSpec
from chamber.partners.selection import make_phase0_draft_zoo, select_zoo


class TestPhase0DraftZoo:
    def test_zoo_has_three_specs(self) -> None:
        """ADR-009 §Consequences: Phase-0 ships 3-partner draft zoo."""
        zoo = make_phase0_draft_zoo()
        assert len(zoo) == 3

    def test_specs_are_partner_specs(self) -> None:
        """Plan/04 §3.8: draft zoo consists of PartnerSpec instances."""
        zoo = make_phase0_draft_zoo()
        for spec in zoo:
            assert isinstance(spec, PartnerSpec)

    def test_first_entry_is_scripted_heuristic(self) -> None:
        """ADR-009 §Decision: heuristic is the first stratum entry."""
        zoo = make_phase0_draft_zoo()
        assert zoo[0].class_name == "scripted_heuristic"

    def test_second_entry_is_frozen_mappo(self) -> None:
        """ADR-009 §Decision: frozen MAPPO is the second stratum entry."""
        zoo = make_phase0_draft_zoo()
        assert zoo[1].class_name == "frozen_mappo"

    def test_third_entry_is_frozen_harl_50pct(self) -> None:
        """ADR-009 §Decision: frozen HARL at 50%-reward is the third stratum entry."""
        zoo = make_phase0_draft_zoo()
        spec = zoo[2]
        assert spec.class_name == "frozen_harl"
        assert spec.extra["checkpoint_tier"] == "50pct_reward"

    def test_partner_ids_are_unique(self) -> None:
        """ADR-006 risk #3: every draft-zoo partner has a distinct partner_id."""
        zoo = make_phase0_draft_zoo()
        ids = [spec.partner_id for spec in zoo]
        assert len(set(ids)) == 3

    def test_function_returns_fresh_list(self) -> None:
        """Mutating the returned list does not corrupt the canonical zoo."""
        a = make_phase0_draft_zoo()
        a.append(
            PartnerSpec(
                class_name="phantom",
                seed=0,
                checkpoint_step=None,
                weights_uri=None,
            )
        )
        b = make_phase0_draft_zoo()
        assert len(b) == 3

    def test_uids_match_adr_001_smoke_robot_tuple(self) -> None:
        """ADR-001: smoke robot tuple is (panda_wristcam, fetch, allegro_hand_right)."""
        zoo = make_phase0_draft_zoo()
        uids = {spec.extra["uid"] for spec in zoo}
        assert uids == {"panda_wristcam", "fetch", "allegro_hand_right"}

    def test_frozen_rl_specs_carry_local_artifact_uris(self) -> None:
        """Plan/04 §3.8: frozen-RL checkpoints live under local://artifacts/."""
        zoo = make_phase0_draft_zoo()
        for spec in zoo[1:]:
            assert spec.weights_uri is not None
            assert spec.weights_uri.startswith("local://artifacts/")


class TestPhase1SelectZooStub:
    def test_select_zoo_raises_not_implemented(self) -> None:
        """ADR-009 §Validation criteria: Phase-1 zoo construction is stubbed in Phase 0."""
        with pytest.raises(NotImplementedError, match="Phase-1"):
            select_zoo()

    def test_select_zoo_message_cites_adr_009(self) -> None:
        """Plan/04 §3.8: error message points at the ADR-009 validation section."""
        with pytest.raises(NotImplementedError, match="ADR-009"):
            select_zoo()

    def test_select_zoo_message_redirects_to_draft_zoo(self) -> None:
        """Plan/04 §3.8: error message tells Phase-0 callers what to use instead."""
        with pytest.raises(NotImplementedError, match="make_phase0_draft_zoo"):
            select_zoo()

    def test_select_zoo_accepts_arbitrary_args(self) -> None:
        """Reserved signature: positional and keyword args are absorbed by the stub."""
        with pytest.raises(NotImplementedError):
            select_zoo("alpha", n=42, target="stage0_smoke")


class TestFrozenRLRegistrationStatus:
    """Pin which draft-zoo specs ``load_partner`` resolves (ADR-009 §Decision).

    Plan/04 §1: the heuristic (T4.4) + frozen-MAPPO (T4.5) adapters are
    registered in M4 Phase 1/2; the frozen-HARL adapter (T4.6) lands in a
    follow-up PR. Until then, calling
    :func:`chamber.partners.registry.load_partner` on the third spec
    raises :class:`KeyError` listing the registered keys — loud failure
    surfaces the deferral. Once T4.6 lands, the
    :meth:`test_load_frozen_harl_spec_raises_until_phase3` assertion
    flips and the M4-gate integration test (T4.9) takes over.
    """

    def test_load_frozen_mappo_spec_succeeds(self) -> None:
        """T4.5: ``frozen_mappo`` is registered and ``load_partner`` returns the wrapper."""
        from chamber.partners.frozen_mappo import FrozenMAPPOPartner
        from chamber.partners.registry import load_partner

        partner = load_partner(make_phase0_draft_zoo()[1])
        assert isinstance(partner, FrozenMAPPOPartner)

    def test_load_frozen_harl_spec_raises_until_phase3(self) -> None:
        """T4.6 deferred: ``frozen_harl`` is not registered until the follow-up PR."""
        from chamber.partners.registry import load_partner

        with pytest.raises(KeyError, match="frozen_harl"):
            load_partner(make_phase0_draft_zoo()[2])

    def test_load_heuristic_spec_succeeds(self) -> None:
        """T4.4 ships in M4a Phase 1: ``scripted_heuristic`` IS registered."""
        from chamber.partners.heuristic import ScriptedHeuristicPartner
        from chamber.partners.registry import load_partner

        partner = load_partner(make_phase0_draft_zoo()[0])
        assert isinstance(partner, ScriptedHeuristicPartner)
