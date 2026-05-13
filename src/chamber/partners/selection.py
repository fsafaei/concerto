# SPDX-License-Identifier: Apache-2.0
"""Phase-0 draft zoo + Phase-1 zoo construction stub (ADR-009 §Decision).

ADR-009 §Decision specifies FCP/MEP-style construction with a Population
Entropy admission filter, target zoo size 3 strata x 16 seeds x 3 checkpoints
= 144 partners. Phase-0 ships only the 3-partner draft zoo demanded by the
ADR-007 rev-3 Stage-3 PF spike (one heuristic + one frozen MAPPO checkpoint
+ one frozen HARL checkpoint at 50%-reward), wired through this module.

The Phase-1 :func:`select_zoo` constructor is intentionally a hard stub: it
raises :class:`NotImplementedError` referencing ADR-009 §Validation criteria
"By Phase-1 end" so any caller that drifts into Phase-1 territory in Phase-0
fails loudly.
"""

from __future__ import annotations

from typing import NoReturn

from chamber.partners.api import PartnerSpec


def make_phase0_draft_zoo() -> list[PartnerSpec]:
    """Return the 3-partner Stage-3 draft zoo (ADR-009 §Consequences "draft-zoo scoping").

    The draft zoo is the minimum viable zoo that lets the ADR-007 rev-3
    Stage-3 PF spike measure trained-with vs frozen-novel ≥20pp gap and
    instrument the partner-swap λ-reset transient (ADR-006 risk #3 / ADR-004
    §risk-mitigation #2). All three specs target the Stage-0 smoke task so
    they share an env shape; the per-partner ``uid`` matches the ADR-001
    smoke robot tuple (panda_wristcam / fetch / allegro_hand_right).

    The frozen-RL checkpoint URIs (``local://artifacts/...``) are produced
    by M4b training runs. The :class:`~chamber.partners.frozen_mappo.FrozenMAPPOPartner`
    adapter (T4.5) is registered as ``frozen_mappo`` and loads the second
    spec via :func:`chamber.partners.registry.load_partner`. The
    ``FrozenHARLPartner`` adapter (T4.6) for the third spec lands in a
    follow-up PR; calling ``load_partner`` on that spec raises
    :class:`KeyError` in the interim — by design (loud failure surfaces the
    deferral) per plan/04 §1.

    Returns:
        A fresh list of 3 :class:`~chamber.partners.api.PartnerSpec`
        instances. Callers may append to the list freely; the function
        rebuilds it every time so the canonical zoo is immutable.
    """
    return [
        PartnerSpec(
            class_name="scripted_heuristic",
            seed=0,
            checkpoint_step=None,
            weights_uri=None,
            extra={
                "uid": "fetch",
                "task": "stage0_smoke",
                "target_xy": "0.0,0.0",
                "action_dim": "2",
            },
        ),
        PartnerSpec(
            class_name="frozen_mappo",
            seed=42,
            checkpoint_step=100_000,
            weights_uri="local://artifacts/mappo_seed42_step100k.pt",
            extra={
                "uid": "panda_wristcam",
                "task": "stage0_smoke",
            },
        ),
        PartnerSpec(
            class_name="frozen_harl",
            seed=7,
            checkpoint_step=50_000,
            weights_uri="local://artifacts/happo_seed7_step50k.pt",
            extra={
                "uid": "allegro_hand_right",
                "task": "stage0_smoke",
                "checkpoint_tier": "50pct_reward",
            },
        ),
    ]


def select_zoo(*args: object, **kwargs: object) -> NoReturn:
    """FCP/MEP-style zoo construction — Phase-1 stub (ADR-009 §Decision).

    The Phase-1 implementation is the three-axis pipeline (policy class x
    seed x checkpoint) with MEP Population Entropy as the per-stratum
    admission filter (ADR-009 §Validation criteria "By Phase-1 end").

    Args:
        *args: Reserved for the Phase-1 signature.
        **kwargs: Reserved for the Phase-1 signature.

    Raises:
        NotImplementedError: Always. Phase-0 callers must use
            :func:`make_phase0_draft_zoo` instead.
    """
    del args, kwargs
    raise NotImplementedError(
        "Phase-1 work; see ADR-009 §Validation criteria 'By Phase-1 end' for "
        "the FCP/MEP-style construction protocol with Population Entropy "
        "admission filter. Phase-0 callers should use make_phase0_draft_zoo()."
    )


__all__ = ["make_phase0_draft_zoo", "select_zoo"]
