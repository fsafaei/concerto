# SPDX-License-Identifier: Apache-2.0
"""``chamber-eval`` console entry point (Phase-0 stub).

Consumes spike results and emits the HRS bundle (ADR-008) and the
three-table safety report (ADR-014).
"""

from __future__ import annotations

import chamber


def main() -> None:
    """Entry point for the ``chamber-eval`` console script.

    Phase-0 stub — wired in M5.
    """
    print(f"chamber-eval  (CHAMBER {chamber.__version__})")
    print("Sub-commands: --spike, --runs  [not yet implemented — M5]")
