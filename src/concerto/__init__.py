# SPDX-License-Identifier: Apache-2.0
"""CONCERTO — the method for heterogeneous multi-robot ad-hoc teamwork.

CONCERTO (Contact-rich Cooperation with Novel Cooperators under Embodiment
Heterogeneity, Trust bounds, and Opacity — the ``CITATION.cff`` expansion)
is the safety stack and ego-AHT training algorithm described in the
project's ADR suite. See ADR-INDEX.md for the full decision record.

The version is read from the installed ``concerto-multirobot`` distribution
metadata so that ``pyproject.toml`` is the single source of truth
(release-please bumps it on each release). The import package is ``concerto``;
the distribution name on PyPI is ``concerto-multirobot``.
"""

from importlib.metadata import version as _dist_version

__version__ = _dist_version("concerto-multirobot")
