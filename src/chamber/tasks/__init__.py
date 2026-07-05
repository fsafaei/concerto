# SPDX-License-Identifier: Apache-2.0
"""CHAMBER-Bench task registry — the single source of truth for the suite (ADR-027 §Versioning).

``chamber.tasks`` pins what CHAMBER-Bench contains: one frozen
:class:`TaskSpec` per task version, registered on the ADR-027 §Tier
ladder by :mod:`chamber.tasks.ladder` (imported here, so the registry
is populated by a plain ``import chamber.tasks``). The generated
manifest (``chamber-eval manifest`` / :func:`manifest`), the README
task table, and the ``docs/reference/tasks/`` cards all render from
this package (ADR-027 §Consequences).

Tier-1 contract (ADR-001 §Risks / P2): importing this package touches
no SAPIEN/ManiSkill module; :func:`make` resolves env factories lazily.
"""

from __future__ import annotations

from chamber.tasks import ladder as _ladder  # noqa: F401  (import populates the registry)
from chamber.tasks.cards import CARD_PROSE, TaskCardProse, card_prose
from chamber.tasks.registry import (
    SUITE_NAME,
    SUITE_VERSION,
    get,
    list_registered,
    make,
    manifest,
    register_task,
)
from chamber.tasks.spec import (
    HETEROGENEITY_AXES,
    AdmissionStatus,
    AxisValidity,
    TaskSpec,
)

__all__ = [
    "CARD_PROSE",
    "HETEROGENEITY_AXES",
    "SUITE_NAME",
    "SUITE_VERSION",
    "AdmissionStatus",
    "AxisValidity",
    "TaskCardProse",
    "TaskSpec",
    "card_prose",
    "get",
    "list_registered",
    "make",
    "manifest",
    "register_task",
]
