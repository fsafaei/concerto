# SPDX-License-Identifier: Apache-2.0
"""CHAMBER environment wrappers above ManiSkill v3.

Implements the three wrapper layers from ADR-001 (fork-vs-build decision)
and ADR-005 (simulator base). All wrappers extend ManiSkill v3 without
modifying its internals (project principle P2).
"""
