# SPDX-License-Identifier: Apache-2.0
"""CHAMBER — the benchmark for heterogeneous multi-robot ad-hoc teamwork.

CHAMBER (Contact-rich Heterogeneous Ad-hoc Manipulation Benchmark for
Embodied Robots) is the evaluation suite that wraps ManiSkill v3 (ADR-001,
ADR-005) and provides the partner zoo (ADR-009, ADR-010), communication
stack (ADR-003, ADR-006), evaluation harness (ADR-008, ADR-014), and
spike runners (ADR-007) used to evaluate the CONCERTO method.

Dependency direction: ``chamber.*`` may import from ``concerto.api.*``
and from ``concerto.safety`` / ``concerto.training``. The reverse
direction is forbidden (CI-enforced).
"""

from importlib.metadata import version as _dist_version

# CHAMBER ships in the same wheel as CONCERTO, so the distribution name is
# ``concerto`` (see ``[tool.hatch.build.targets.wheel]`` in pyproject.toml).
__version__ = _dist_version("concerto")
