# SPDX-License-Identifier: Apache-2.0
"""CHAMBER partner zoo — concrete implementations of the FrozenPartner Protocol.

ADR-009 §Decision (zoo construction); ADR-010 §Decision (FM partner selection);
plan/04 §3 (architecture). Phase-0 ships:

- The :class:`FrozenPartner` Protocol + :class:`PartnerSpec` dataclass
  (:mod:`chamber.partners.api`).
- The plug-in :func:`register_partner` decorator + :func:`load_partner`
  factory + :func:`list_registered` (:mod:`chamber.partners.registry`).
- :class:`PartnerBase` with the ``_FORBIDDEN_ATTRS`` shield enforcing the
  black-box AHT no-joint-training constraint at runtime
  (:mod:`chamber.partners.interface`).
- :class:`ScriptedHeuristicPartner` — Stage-3 draft-zoo entry #1
  (:mod:`chamber.partners.heuristic`).
- :class:`FrozenMAPPOPartner` — Stage-3 draft-zoo entry #2 (T4.5;
  :mod:`chamber.partners.frozen_mappo`). Loads a torch state-dict via
  :func:`concerto.training.checkpoints.load_checkpoint`, freezes every
  parameter, runs inference under :func:`torch.no_grad`.
- :class:`FrozenHARLPartner` — Stage-3 draft-zoo entry #3 (T4.6;
  :mod:`chamber.partners.frozen_harl`). Loads the HARL HAPPO actor
  checkpoint produced by
  :class:`chamber.benchmarks.ego_ppo_trainer.EgoPPOTrainer`, rebuilds
  the :class:`StochasticPolicy` with the same args, freezes every
  parameter, runs ``deterministic=True`` inference under
  :func:`torch.no_grad`.
- :class:`CoCarryImpedancePartner` — the matched hand-written co-carry
  controller (:mod:`chamber.partners.cocarry_impedance`; ADR-026
  §Decision 1). The frozen teammate the Rung-2 learned incumbent trains
  against (R-2026-06-B §15); imported here so it registers package-wide.
- :class:`OpenVLAPartner` and :class:`CrossFormerPartner` Phase-1 stubs
  (:mod:`chamber.partners.stubs`) — ADR-010 §Decision Option B; both
  raise :class:`NotImplementedError` referencing the Phase-1 ticket.
- The Phase-0 :func:`make_phase0_draft_zoo` (real) and the Phase-1
  :func:`select_zoo` stub (:mod:`chamber.partners.selection`).

The M4-gate integration test (T4.9) lands in a follow-up PR (plan/04 §1).
"""

from chamber.partners.ablation import PartnerAblatedZero
from chamber.partners.api import FrozenPartner, PartnerSpec
from chamber.partners.cocarry_blind import CoCarryBlindImpedancePartner
from chamber.partners.cocarry_impedance import CoCarryImpedancePartner
from chamber.partners.coinsert_impedance import CoInsertBaseInserter, CoInsertReferenceHolder
from chamber.partners.frozen_harl import FrozenHARLPartner
from chamber.partners.frozen_mappo import FrozenMAPPOPartner
from chamber.partners.handover_presenter import HandoverPresenterPartner
from chamber.partners.heuristic import ScriptedHeuristicPartner
from chamber.partners.interface import PartnerBase
from chamber.partners.registry import list_registered, load_partner, register_partner
from chamber.partners.selection import make_phase0_draft_zoo, select_zoo
from chamber.partners.sets import (
    PartnerMemberSpec,
    PartnerSetSpec,
    WithheldParametersError,
    compute_split,
    get_partner_set,
    list_partner_sets,
    register_partner_set,
    resolve_set_members,
)
from chamber.partners.stubs.crossformer import CrossFormerPartner
from chamber.partners.stubs.openvla import OpenVLAPartner

# Imported for the @register_partner_set side effects: the v1 sets register
# on package import, mirroring how chamber.tasks populates its registry
# via chamber.tasks.ladder (ADR-027 §Versioning; ADR-009 as amended).
import chamber.partners.set_definitions  # noqa: F401  isort: skip

__all__ = [
    "CoCarryBlindImpedancePartner",
    "CoCarryImpedancePartner",
    "CoInsertBaseInserter",
    "CoInsertReferenceHolder",
    "CrossFormerPartner",
    "FrozenHARLPartner",
    "FrozenMAPPOPartner",
    "FrozenPartner",
    "HandoverPresenterPartner",
    "OpenVLAPartner",
    "PartnerAblatedZero",
    "PartnerBase",
    "PartnerMemberSpec",
    "PartnerSetSpec",
    "PartnerSpec",
    "ScriptedHeuristicPartner",
    "WithheldParametersError",
    "compute_split",
    "get_partner_set",
    "list_partner_sets",
    "list_registered",
    "load_partner",
    "make_phase0_draft_zoo",
    "register_partner",
    "register_partner_set",
    "resolve_set_members",
    "select_zoo",
]
