# SPDX-License-Identifier: Apache-2.0
"""Tier-1 tests for the partner-card renderers (ADR-009 as amended; ADR-027 §Consequences).

Cards render strictly from the set registry + a committed fingerprint
payload: every member gets a card, private members never leak parameter
values, and a missing archive/member is a loud error (cards never
render without evidence).
"""

from __future__ import annotations

from typing import Any

import pytest

import chamber.partners  # noqa: F401 - registers the v1 sets
from chamber.partners.cards import (
    archive_rel_dir,
    cards_rel_dir,
    render_all_partner_cards,
    render_member_card,
    render_set_index,
)
from chamber.partners.sets import get_partner_set, list_partner_sets, parse_set_slug


def _fake_payload(slug: str) -> dict[str, Any]:
    set_id, version = parse_set_slug(slug)
    spec = get_partner_set(set_id, version=version)
    members: dict[str, Any] = {}
    for m in spec.members:
        members[m.member_name] = {
            "partner_id": m.partner_id,
            "params_sha256": m.params_sha256,
            "registry_class": m.registry_class,
            "role": m.role,
            "split": m.split,
            "fingerprint": {
                "n_episodes": 20.0,
                "success_rate": 1.0,
                "action_abs_mean": 0.1234,
            },
            "floor_probe": spec.floor_probe,
            "floor_success": 1.0,
            "floor_pass": True,
        }
    return {"members": members}


def _all_payloads() -> dict[str, dict[str, Any]]:
    return {slug: _fake_payload(slug) for slug in list_partner_sets()}


class TestRenderAllPartnerCards:
    def test_every_member_gets_a_card_plus_indexes(self) -> None:
        cards = render_all_partner_cards(_all_payloads())
        assert "index.md" in cards
        total_members = 0
        for slug in list_partner_sets():
            set_id, version = parse_set_slug(slug)
            spec = get_partner_set(set_id, version=version)
            assert f"{cards_rel_dir(spec)}/index.md" in cards
            for m in spec.members:
                assert f"{cards_rel_dir(spec)}/{m.member_name}.md" in cards
            total_members += len(spec.members)
        assert len(cards) == total_members + len(list_partner_sets()) + 1

    def test_missing_set_payload_is_loud(self) -> None:
        payloads = _all_payloads()
        payloads.pop("cocarry_partners@v1")
        with pytest.raises(KeyError, match="no fingerprint payload"):
            render_all_partner_cards(payloads)

    def test_missing_member_entry_is_loud(self) -> None:
        payloads = _all_payloads()
        payloads["cocarry_partners@v1"]["members"].pop("imp_nominal")
        with pytest.raises(KeyError, match="imp_nominal"):
            render_all_partner_cards(payloads)

    def test_render_is_deterministic(self) -> None:
        assert render_all_partner_cards(_all_payloads()) == render_all_partner_cards(
            _all_payloads()
        )


class TestPrivateRedaction:
    def test_private_card_shows_hash_and_fingerprint_never_values(self) -> None:
        spec = get_partner_set("cocarry_partners", version=1)
        (private_member, *_rest) = spec.private_members
        payload = _fake_payload("cocarry_partners@v1")["members"][private_member.member_name]
        card = render_member_card(spec, private_member, payload)
        assert private_member.partner_id in card
        assert private_member.params_sha256 in card
        assert "*withheld*" in card
        assert "withheld parameters" in card

    def test_public_card_commits_the_values(self) -> None:
        spec = get_partner_set("cocarry_partners", version=1)
        (public_member, *_rest) = spec.public_members
        assert public_member.params is not None
        payload = _fake_payload("cocarry_partners@v1")["members"][public_member.member_name]
        card = render_member_card(spec, public_member, payload)
        for value in public_member.params.values():
            assert value in card


class TestSetIndex:
    def test_roster_rows_and_archive_pointer(self) -> None:
        spec = get_partner_set("handover_place_partners", version=1)
        index = render_set_index(spec, _fake_payload("handover_place_partners@v1"))
        for m in spec.members:
            assert m.member_name in index
            assert m.partner_id in index
        assert archive_rel_dir(spec) in index
        assert "no hand-picking" in index
