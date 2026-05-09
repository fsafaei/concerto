# SPDX-License-Identifier: Apache-2.0
"""Slow reproduction tests pinned to published results.

Plan/03 §5 "Reproduction" — tests under this directory reproduce
specific published numbers (e.g., Huriot & Sibai 2025 Table I) within a
small tolerance. They are marked ``@pytest.mark.slow`` so they skip on
PR CI and run nightly + before any release.
"""
