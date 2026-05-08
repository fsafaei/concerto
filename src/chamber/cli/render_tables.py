# SPDX-License-Identifier: Apache-2.0
"""``chamber-render-tables`` console entry point (Phase-0 stub).

Renders three-table safety reports (ADR-014) from spike result archives
into Markdown suitable for publication.
"""

from __future__ import annotations


def main() -> None:
    """Entry point for the ``chamber-render-tables`` console script.

    Phase-0 stub — wired in M5.
    """
    import chamber

    print(f"chamber-render-tables  (CHAMBER {chamber.__version__})")
    print("Sub-commands: --results-dir, --output  [not yet implemented — M5]")
