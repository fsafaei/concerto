# SPDX-License-Identifier: Apache-2.0
"""CHAMBER evaluation harness — HRS bundle, leaderboard, safety reports.

Implements the HRS bundle composition (Option D, axis-survival rule) from
ADR-008 and consumes the three-table safety reporting format from ADR-014.
Statistical tooling (bootstrap CIs, paired tests) and the pre-registration
discipline (ADR-007 §Pre-registration) live in sub-modules of this package.
"""
