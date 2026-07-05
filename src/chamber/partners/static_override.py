# SPDX-License-Identifier: Apache-2.0
"""Deprecated re-export — moved to :mod:`chamber.partners.exploratory.static_override`.

The EXPLORATORY zero-action partner override lives under the
``chamber.partners.exploratory`` namespace so exploratory partners are
structurally separated from the leaderboard-facing partner surface
(ADR-009 §Decision; ADR-027 §Reporting rules). Importing this module
emits :class:`DeprecationWarning`; update imports to the new path.
"""

from __future__ import annotations

import warnings

from chamber.partners.exploratory.static_override import (
    ExploratoryStaticPartnerOverride,
)

warnings.warn(
    "chamber.partners.static_override has moved to "
    "chamber.partners.exploratory.static_override (exploratory partners "
    "are quarantined from the leaderboard-facing surface; ADR-009 "
    "§Decision, ADR-027 §Reporting rules). Update the import.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["ExploratoryStaticPartnerOverride"]
