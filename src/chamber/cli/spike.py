# SPDX-License-Identifier: Apache-2.0
"""``chamber-spike`` console entry point (M4b-9b; T5b.1; ADR-002 §Decisions).

Thin argparse dispatcher over the per-subcommand modules:

- :mod:`chamber.cli._spike_train` — ``chamber-spike train`` (M4b-9b;
  plan/05 §3.5). Canonical ego-AHT training entry point used by
  ``scripts/repro/zoo_seed.sh`` and
  ``scripts/repro/empirical_guarantee.sh``.
- :mod:`chamber.cli._spike_verify_prereg` — ``chamber-spike verify-prereg``
  (T5b.1; ADR-007 §Discipline). Refuses to launch a spike whose YAML
  blob SHA does not match the tagged blob SHA.
- :mod:`chamber.cli._spike_list` — ``chamber-spike list-axes`` and
  ``chamber-spike list-profiles`` (T5b.1; ADR-007 §3.4 / ADR-006
  §Decision). Enumerate the project-frozen axis and comm-profile
  registries.

The dispatcher itself owns only the no-subcommand banner path
(``chamber-spike`` with empty argv) and the argparse glue; every
subcommand owns its own argparse surface + run function. Adding a
future subcommand (e.g. ``run`` / ``next-stage`` in B7) lands as a new
``_spike_<name>.py`` module wired into :func:`_build_parser` and
:func:`main`'s dispatch table.
"""

from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING

import chamber
from chamber.cli import (
    _spike_list,
    _spike_next_stage,
    _spike_run,
    _spike_train,
    _spike_verify_prereg,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

#: Subcommand → ``run(args)`` dispatch table. Built once at module
#: import so ``_SUBCOMMANDS`` (used in the top-level description) stays
#: in sync with the actual dispatch (no hand-maintained string list to
#: drift). A future subcommand drops in as a new ``_spike_<name>.py``
#: module + one line here.
_DISPATCH: dict[str, Callable[[argparse.Namespace], int]] = {
    "train": _spike_train.run,
    "verify-prereg": _spike_verify_prereg.run,
    "list-axes": _spike_list.run_axes,
    "list-profiles": _spike_list.run_profiles,
    "run": _spike_run.run,
    "next-stage": _spike_next_stage.run,
}
_SUBCOMMANDS: tuple[str, ...] = tuple(_DISPATCH)


def _build_parser() -> argparse.ArgumentParser:
    """Build the top-level ``chamber-spike`` argparse surface (M4b-9b; T5b.1)."""
    parser = argparse.ArgumentParser(
        prog="chamber-spike",
        description=(
            f"CHAMBER {chamber.__version__} — spike CLI. Sub-commands: {', '.join(_SUBCOMMANDS)}."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=False)
    _spike_train.add_parser(sub)
    _spike_verify_prereg.add_parser(sub)
    _spike_list.add_axes_parser(sub)
    _spike_list.add_profiles_parser(sub)
    _spike_run.add_parser(sub)
    _spike_next_stage.add_parser(sub)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``chamber-spike`` console script (M4b-9b; T5b.1; ADR-002 §Decisions).

    Args:
        argv: Optional argv override for tests. ``None`` (default)
            reads from :data:`sys.argv` per argparse's usual contract.

    Returns:
        Process exit code; ``0`` on success. Non-zero values are
        documented per-subcommand (``train`` exit 3 for the
        empirical-guarantee trip-wire; ``verify-prereg`` exit 4 for a
        SHA / tag mismatch).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        # No sub-command — print version banner + help, exit 0. Keeps the
        # previous entry point's friendly behavior for users who type
        # ``chamber-spike`` without arguments (the existing
        # ``test_chamber_spike_main_banner`` smoke pins this path).
        print(f"chamber-spike  (CHAMBER {chamber.__version__})")
        parser.print_help()
        return 0
    runner = _DISPATCH.get(args.command)
    if runner is None:
        parser.error(f"unknown command {args.command!r}")
        return 2  # pragma: no cover  # parser.error sys.exits before this returns.
    return runner(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
