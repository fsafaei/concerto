# SPDX-License-Identifier: Apache-2.0
"""CHAMBER partner zoo — concrete implementations of the FrozenPartner Protocol.

ADR-009 §Decision (zoo construction); ADR-010 §Decision (FM partner selection);
plan/04 §3 (architecture). Phase-0 ships, in this PR:

- The :class:`FrozenPartner` Protocol + :class:`PartnerSpec` dataclass
  (:mod:`chamber.partners.api`).
- The plug-in :func:`register_partner` decorator + :func:`load_partner`
  factory + :func:`list_registered` (:mod:`chamber.partners.registry`).
- :class:`PartnerBase` with the ``_FORBIDDEN_ATTRS`` shield enforcing the
  black-box AHT no-joint-training constraint at runtime
  (:mod:`chamber.partners.interface`).
- :class:`ScriptedHeuristicPartner` — Stage-3 draft-zoo entry #1
  (:mod:`chamber.partners.heuristic`).
- :class:`OpenVLAPartner` and :class:`CrossFormerPartner` Phase-1 stubs
  (:mod:`chamber.partners.stubs`) — ADR-010 §Decision Option B; both
  raise :class:`NotImplementedError` referencing the Phase-1 ticket.
- The Phase-0 :func:`make_phase0_draft_zoo` (real) and the Phase-1
  :func:`select_zoo` stub (:mod:`chamber.partners.selection`).

The Phase-1 real frozen-RL adapters (T4.5 / T4.6 / T4.9) are blocked-by
M4b which produces the checkpoints (plan/04 §1).
"""

from chamber.partners.api import FrozenPartner, PartnerSpec
from chamber.partners.heuristic import ScriptedHeuristicPartner
from chamber.partners.interface import PartnerBase
from chamber.partners.registry import list_registered, load_partner, register_partner
from chamber.partners.selection import make_phase0_draft_zoo, select_zoo
from chamber.partners.stubs.crossformer import CrossFormerPartner
from chamber.partners.stubs.openvla import OpenVLAPartner

__all__ = [
    "CrossFormerPartner",
    "FrozenPartner",
    "OpenVLAPartner",
    "PartnerBase",
    "PartnerSpec",
    "ScriptedHeuristicPartner",
    "list_registered",
    "load_partner",
    "make_phase0_draft_zoo",
    "register_partner",
    "select_zoo",
]
