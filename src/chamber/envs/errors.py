# SPDX-License-Identifier: Apache-2.0
"""Custom errors for CHAMBER environment wrappers.

ADR-001 §Risks — wrappers raise this error rather than silently falling back
when a wrapped env does not satisfy the wrapper's structural assumptions.
"""

from __future__ import annotations


class ChamberEnvCompatibilityError(RuntimeError):
    """Raised when a wrapped env does not satisfy CHAMBER wrapper assumptions.

    ADR-001 §Risks: a silent fallback during a Stage-0 spike would invalidate
    the gate; failing loudly forces ADR re-review.
    """
