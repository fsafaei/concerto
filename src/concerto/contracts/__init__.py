# SPDX-License-Identifier: Apache-2.0
"""Task contracts: what an observation and an action mean, per task version.

``get("pickcube")`` returns the frozen :class:`TaskContract` both the sim env
and a real-robot env must satisfy. Importing this package touches no
SAPIEN/ManiSkill module; :mod:`concerto.contracts.sim` builds the sim env.
"""

from __future__ import annotations

from concerto.contracts import pickcube as _pickcube  # noqa: F401  (import registers the contract)
from concerto.contracts.spec import (
    N_ARM_JOINTS,
    ActionField,
    ObsField,
    TaskContract,
    get,
    list_contracts,
    register,
    versions,
)

__all__ = [
    "N_ARM_JOINTS",
    "ActionField",
    "ObsField",
    "TaskContract",
    "get",
    "list_contracts",
    "register",
    "versions",
]
