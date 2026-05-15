# SPDX-License-Identifier: Apache-2.0
"""``chamber-spike list-axes`` and ``list-profiles`` subcommands (T5b.1).

Enumerates two project-frozen registries so an outside contributor (or
a CI script) can grep the available axes / comm profiles without
reading the source.

- ``list-axes`` prints the six ADR-007 §3.4 Option D axes (AS, OM, CR,
  CM, PF, SA) in canonical order along with their Phase-0 stage
  assignment (Stage 1/2/3 per ADR-007 §Implementation staging).
- ``list-profiles`` prints the six URLLC + 3GPP Release 17 anchored
  comm degradation profiles from :data:`chamber.comm.URLLC_3GPP_R17`
  (ideal, urllc, factory, wifi, lossy, saturation) along with each
  row's numeric anchors. This is the table the Stage-2 CM spike
  (plan/07 §3) sweeps over.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from chamber.comm import URLLC_3GPP_R17

if TYPE_CHECKING:
    import argparse

#: ADR-007 §3.4 Option D axes + their Phase-0 stage assignment
#: (ADR-007 §Implementation staging). Tuple-of-tuples (rather than a
#: dict) so iteration order is the canonical
#: AS / OM / CR / CM / PF / SA. Iteration order is easier to read at a
#: glance than the HRS-bundle ordering (which lives separately in
#: :data:`chamber.evaluation.hrs.DEFAULT_AXIS_WEIGHTS`).
#:
#: Description strings are pure ASCII so ``chamber-spike list-axes``
#: works on a Windows console (cp1252) and on non-UTF-8 CI pipes.
_AXES: tuple[tuple[str, int, str], ...] = (
    ("AS", 1, "Action space: 7-DOF arm vs 2-DOF base on shared pick-place."),
    ("OM", 1, "Observation modality: vision-only vs vision+FT+proprio fused."),
    ("CR", 2, "Control rate: dual-clock 500 Hz vs 50 Hz on a single embodiment."),
    ("CM", 2, "Communication: URLLC / 5G-TSN profile sweep on the fixed-format channel."),
    ("PF", 3, "Partner familiarity: trained-with vs novel partner with mid-episode swap."),
    ("SA", 3, "Safety: heterogeneous force-limit / SIL-PL pair on a contact-rich task."),
)


def add_axes_parser(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register the ``list-axes`` subparser (T5b.1; ADR-007 §3.4)."""
    sub.add_parser(
        "list-axes",
        help="List the six ADR-007 §3.4 Option D heterogeneity axes.",
        description=(
            "Prints each axis label, its Phase-0 stage assignment "
            "(ADR-007 §Implementation staging), and a one-line description."
        ),
    )


def add_profiles_parser(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register the ``list-profiles`` subparser (T5b.1; ADR-006 §Decision)."""
    sub.add_parser(
        "list-profiles",
        help="List the six URLLC + 3GPP R17 comm degradation profiles.",
        description=(
            "Prints each profile name and its numeric anchors (latency mean / "
            "latency std / drop rate) from chamber.comm.URLLC_3GPP_R17 (ADR-006 "
            "§Decision). The Stage-2 CM spike (plan/07 §3) sweeps over these."
        ),
    )


def run_axes(args: argparse.Namespace) -> int:
    """Implementation of ``chamber-spike list-axes`` (T5b.1; ADR-007 §3.4)."""
    del args
    print("axis  stage  description")
    print("----  -----  -----------")
    for axis, stage, description in _AXES:
        print(f"{axis:<4}  {stage:>5}  {description}")
    return 0


def run_profiles(args: argparse.Namespace) -> int:
    """Implementation of ``chamber-spike list-profiles`` (T5b.1; ADR-006 §Decision).

    The profile registry is :data:`chamber.comm.URLLC_3GPP_R17` —
    bumping any row requires an ADR amendment per
    :mod:`chamber.comm.profiles`.
    """
    del args
    print(f"{'profile':<12}  {'latency_mean_ms':>15}  {'latency_std_ms':>14}  {'drop_rate':>9}")
    print(f"{'-------':<12}  {'---------------':>15}  {'--------------':>14}  {'---------':>9}")
    for name, profile in URLLC_3GPP_R17.items():
        print(
            f"{name:<12}  "
            f"{profile.latency_mean_ms:>15.4g}  "
            f"{profile.latency_std_ms:>14.4g}  "
            f"{profile.drop_rate:>9.2e}"
        )
    return 0


__all__ = [
    "add_axes_parser",
    "add_profiles_parser",
    "run_axes",
    "run_profiles",
]
