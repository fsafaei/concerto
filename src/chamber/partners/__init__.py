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

Subsequent M4a PRs add the 3-partner Phase-0 draft zoo (heuristic + frozen
MAPPO + frozen HARL) and the OpenVLA + CrossFormer Phase-1 stubs
(ADR-009 §Consequences "draft-zoo scoping for ADR-007 Stage 3";
ADR-010 §Decision Option B). The Phase-1 real frozen-RL adapters
(T4.5 / T4.6 / T4.9) are blocked-by M4b which produces the checkpoints
(plan/04 §1).
"""

from chamber.partners.api import FrozenPartner, PartnerSpec
from chamber.partners.interface import PartnerBase
from chamber.partners.registry import list_registered, load_partner, register_partner

__all__ = [
    "FrozenPartner",
    "PartnerBase",
    "PartnerSpec",
    "list_registered",
    "load_partner",
    "register_partner",
]
