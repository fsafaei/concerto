# SPDX-License-Identifier: Apache-2.0
"""``chamber-spike`` console entry point (Phase-0 stub).

Launches pre-registered ADR-007 axis spikes. Pre-registration discipline
(project principle P4) is enforced: the CLI verifies the prereg YAML SHA
against its git tag before any run is launched.
"""

from __future__ import annotations

import chamber


def main() -> None:
    """Entry point for the ``chamber-spike`` console script.

    Phase-0 stub — sub-commands (run, verify-prereg) will be wired in M5.
    """
    print(f"chamber-spike  (CHAMBER {chamber.__version__})")
    print("Sub-commands: run, verify-prereg  [not yet implemented — M5]")
