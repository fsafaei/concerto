# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``chamber.partners.stubs`` (T4.7).

Covers ADR-010 §Decision Option B (Phase-1 stubs raise loudly).
"""

from __future__ import annotations

import pytest

from chamber.partners.api import PartnerSpec
from chamber.partners.registry import list_registered, load_partner
from chamber.partners.stubs.crossformer import CrossFormerPartner
from chamber.partners.stubs.openvla import OpenVLAPartner


def _openvla_spec() -> PartnerSpec:
    return PartnerSpec(
        class_name="openvla_lora_specialist",
        seed=0,
        checkpoint_step=None,
        weights_uri="local://lora.pt",
    )


def _crossformer_spec() -> PartnerSpec:
    return PartnerSpec(
        class_name="crossformer_zero_shot",
        seed=0,
        checkpoint_step=None,
        weights_uri=None,
    )


class TestOpenVLAStub:
    def test_reset_raises_not_implemented_with_adr_010_reference(self) -> None:
        """ADR-010 §Decision Option B: stub raises with Phase-1 ticket reference."""
        partner = OpenVLAPartner(_openvla_spec())
        with pytest.raises(NotImplementedError, match="ADR-010"):
            partner.reset()

    def test_act_raises_not_implemented_with_adr_010_reference(self) -> None:
        """ADR-010 §Decision Option B: act raises rather than silently returning."""
        partner = OpenVLAPartner(_openvla_spec())
        with pytest.raises(NotImplementedError, match="ADR-010"):
            partner.act({})

    def test_reset_message_points_at_phase1_ticket(self) -> None:
        """Plan/04 §3.7: NotImplementedError carries the Phase-1 ticket URL."""
        partner = OpenVLAPartner(_openvla_spec())
        with pytest.raises(NotImplementedError, match="TBD-Phase-1-openvla"):
            partner.reset()

    def test_act_message_points_at_phase1_ticket(self) -> None:
        """Plan/04 §3.7: NotImplementedError carries the Phase-1 ticket URL."""
        partner = OpenVLAPartner(_openvla_spec())
        with pytest.raises(NotImplementedError, match="TBD-Phase-1-openvla"):
            partner.act({})

    def test_construction_does_not_raise(self) -> None:
        """ADR-010 §Decision Option B: construction is cheap; only reset/act raise."""
        OpenVLAPartner(_openvla_spec())


class TestCrossFormerStub:
    def test_reset_raises_not_implemented_with_adr_010_reference(self) -> None:
        """ADR-010 §Decision Option B: stub raises with Phase-1 ticket reference."""
        partner = CrossFormerPartner(_crossformer_spec())
        with pytest.raises(NotImplementedError, match="ADR-010"):
            partner.reset()

    def test_act_raises_not_implemented_with_adr_010_reference(self) -> None:
        """ADR-010 §Decision Option B: act raises rather than silently returning."""
        partner = CrossFormerPartner(_crossformer_spec())
        with pytest.raises(NotImplementedError, match="ADR-010"):
            partner.act({})

    def test_reset_message_points_at_phase1_ticket(self) -> None:
        """Plan/04 §3.7: NotImplementedError carries the Phase-1 ticket URL."""
        partner = CrossFormerPartner(_crossformer_spec())
        with pytest.raises(NotImplementedError, match="TBD-Phase-1-crossformer"):
            partner.reset()

    def test_act_message_points_at_phase1_ticket(self) -> None:
        """Plan/04 §3.7: NotImplementedError carries the Phase-1 ticket URL."""
        partner = CrossFormerPartner(_crossformer_spec())
        with pytest.raises(NotImplementedError, match="TBD-Phase-1-crossformer"):
            partner.act({})

    def test_construction_does_not_raise(self) -> None:
        """ADR-010 §Decision Option B: construction is cheap; only reset/act raise."""
        CrossFormerPartner(_crossformer_spec())


class TestStubsAreRegistered:
    """Pin the @register_partner side-effect so an accidental drop of the
    decorator in Phase 1 is caught at import time (ADR-010 §Decision Option B).
    """

    def test_openvla_lora_specialist_is_registered(self) -> None:
        """The Phase-1 stub surfaces in the registry so the seam is visible."""
        assert "openvla_lora_specialist" in list_registered()

    def test_crossformer_zero_shot_is_registered(self) -> None:
        """The Phase-1 stub surfaces in the registry so the seam is visible."""
        assert "crossformer_zero_shot" in list_registered()

    def test_load_partner_builds_openvla_stub(self) -> None:
        """ADR-009 §Decision: registry seam works end-to-end for the OpenVLA stub."""
        partner = load_partner(_openvla_spec())
        assert isinstance(partner, OpenVLAPartner)

    def test_load_partner_builds_crossformer_stub(self) -> None:
        """ADR-009 §Decision: registry seam works end-to-end for the CrossFormer stub."""
        partner = load_partner(_crossformer_spec())
        assert isinstance(partner, CrossFormerPartner)


class TestStubsInheritShield:
    """Pin that the ``_FORBIDDEN_ATTRS`` shield (PR 26 / ADR-009 §Consequences)
    still applies to the Phase-1 stubs — a future author who overrides
    ``__getattr__`` for the inference harness must NOT bypass the shield.
    """

    def test_openvla_train_attribute_raises_with_adr_009(self) -> None:
        """ADR-009 §Consequences: ``train`` shield applies even to a stub partner."""
        partner = OpenVLAPartner(_openvla_spec())
        with pytest.raises(AttributeError, match="ADR-009"):
            partner.train  # noqa: B018  # access has the side-effect we care about

    def test_openvla_load_state_dict_raises_with_adr_009(self) -> None:
        """ADR-009 §Consequences: weight-mutation surface is shielded too."""
        partner = OpenVLAPartner(_openvla_spec())
        with pytest.raises(AttributeError, match="ADR-009"):
            partner.load_state_dict  # noqa: B018  # access has the side-effect

    def test_crossformer_train_attribute_raises_with_adr_009(self) -> None:
        """ADR-009 §Consequences: same shield contract for the CrossFormer stub."""
        partner = CrossFormerPartner(_crossformer_spec())
        with pytest.raises(AttributeError, match="ADR-009"):
            partner.train  # noqa: B018  # access has the side-effect we care about
