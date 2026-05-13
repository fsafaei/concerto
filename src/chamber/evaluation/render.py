# SPDX-License-Identifier: Apache-2.0
"""Three-table safety report + leaderboard renderers (ADR-014 §Decision, ADR-008 §Decision).

Two consumer-facing renderers live in this module:

- :func:`render_three_table_safety_report` consumes the JSON payload
  emitted by :mod:`concerto.safety.reporting` and produces Markdown
  + LaTeX. The "fallback fired" column is rendered as a *separate*
  column from "constraint violation" per ADR-014 §Decision and the
  reviewer P0 finding on conflating the two.
- :func:`render_leaderboard` consumes a sequence of
  :class:`chamber.evaluation.results.LeaderboardEntry` records and
  emits the README-compatible Markdown leaderboard. The HRS vector
  is shown as a sub-row under each method per reviewer P1-8; HRS
  scalar is the ranking column per ADR-008 §Decision.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Any

    from chamber.evaluation.results import LeaderboardEntry


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _fmt_float(x: float, *, digits: int = 4) -> str:
    return f"{x:.{digits}g}"


def render_three_table_safety_report(
    report: Mapping[str, Any],
    *,
    fmt: str = "markdown",
) -> str:
    """Render the ADR-014 three-table safety report (ADR-014 §Decision).

    Consumes the JSON-serialised payload produced by
    :func:`concerto.safety.reporting.emit_three_tables` (or any
    structurally equivalent ``ThreeTableReport.to_jsonable()``
    output) and emits Markdown or LaTeX. The fallback-fired column
    in Table 2 is rendered separately from constraint-violation per
    ADR-014 §Decision and the reviewer P0 review on safety reporting.

    Args:
        report: A dict matching the
            :class:`concerto.safety.reporting.ThreeTableReport`
            ``to_jsonable()`` shape (``table_1``, ``table_2``,
            ``table_3``).
        fmt: Output format — ``"markdown"`` or ``"latex"``.

    Returns:
        The rendered three-table report as a string.

    Raises:
        ValueError: When ``fmt`` is not one of the supported
            formats, or the report dict is missing required keys.
    """
    for key in ("table_1", "table_2", "table_3"):
        if key not in report:
            msg = f"three-table report is missing required key {key!r}"
            raise ValueError(msg)

    if fmt == "markdown":
        return _three_table_markdown(report)
    if fmt == "latex":
        return _three_table_latex(report)
    msg = f"fmt must be one of {{'markdown', 'latex'}}, got {fmt!r}"
    raise ValueError(msg)


def _three_table_markdown(report: Mapping[str, Any]) -> str:
    table_1 = report["table_1"]
    table_2 = report["table_2"]
    table_3 = report["table_3"]
    lines: list[str] = [
        "# Three-table safety report (ADR-014)",
        "",
        "## Table 1 — Per-assumption violation rates",
        "",
        "| Assumption | Description | Violations | N steps |",
        "| --- | --- | --- | --- |",
    ]
    lines.extend(
        f"| {row['assumption']} | {row['description']} | {row['violations']} | {row['n_steps']} |"
        for row in table_1
    )
    lines += [
        "",
        "## Table 2 — Per-condition safety (constraint violations vs. fallback fires)",
        "",
        "| Predictor | Conformal mode | Vendor compliance | N episodes "
        "| Constraint violations | Fallback fires |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    lines.extend(
        f"| {row['predictor']} | {row['conformal_mode']} "
        f"| {'—' if row.get('vendor_compliance') is None else row['vendor_compliance']} "
        f"| {row['n_episodes']} | {row['violations']} | {row['fallback_fires']} |"
        for row in table_2
    )
    lines += [
        "",
        "Note: constraint violations and fallback fires are reported in",
        "separate columns per ADR-014 §Decision. A fallback fire means the",
        "braking layer engaged to keep the system inside the safe set; it",
        "is *not* a violation of the CBF-constraint.",
        "",
        "## Table 3 — Conservativeness gap vs. oracle",
        "",
        "| Condition | λ mean | λ var | Oracle λ mean |",
        "| --- | --- | --- | --- |",
    ]
    lines.extend(
        f"| {row['condition']} "
        f"| {_fmt_float(float(row['lambda_mean']))} "
        f"| {_fmt_float(float(row['lambda_var']))} "
        f"| {_fmt_float(float(row['oracle_lambda_mean']))} |"
        for row in table_3
    )
    return "\n".join(lines) + "\n"


def _three_table_latex(report: Mapping[str, Any]) -> str:
    table_1 = report["table_1"]
    table_2 = report["table_2"]
    table_3 = report["table_3"]
    lines: list[str] = [
        "% ADR-014 §Decision — three-table safety report",
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Table 1 — Per-assumption empirical violation rates.}",
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"Assumption & Description & Violations & N steps \\",
        r"\midrule",
    ]
    lines.extend(
        f"{row['assumption']} & {row['description']} & {row['violations']} & {row['n_steps']} \\\\"
        for row in table_1
    )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Table 2 — Per-condition safety (constraint violations vs. fallback fires).}",
        r"\begin{tabular}{lllrrr}",
        r"\toprule",
        r"Predictor & Conformal mode & Vendor compliance & N episodes "
        r"& Constraint violations & Fallback fires \\",
        r"\midrule",
    ]
    lines.extend(
        f"{row['predictor']} & {row['conformal_mode']} & "
        f"{'--' if row.get('vendor_compliance') is None else row['vendor_compliance']} & "
        f"{row['n_episodes']} & {row['violations']} & {row['fallback_fires']} \\\\"
        for row in table_2
    )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Table 3 — Conservativeness gap vs. oracle CBF.}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Condition & $\lambda$ mean & $\lambda$ var & Oracle $\lambda$ mean \\",
        r"\midrule",
    ]
    lines.extend(
        f"{row['condition']} & {_fmt_float(float(row['lambda_mean']))} & "
        f"{_fmt_float(float(row['lambda_var']))} & "
        f"{_fmt_float(float(row['oracle_lambda_mean']))} \\\\"
        for row in table_3
    )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def render_leaderboard(entries: Sequence[LeaderboardEntry]) -> str:
    """Render the CHAMBER leaderboard (ADR-008 §Decision; reviewer P1-8).

    Entries are sorted by ``hrs_scalar`` descending. Each method has
    two rows: the headline row carrying the scalar + aggregate
    violation / fallback rates, and a sub-row showing the HRS
    vector as ``axis: score (weight)`` pairs. The vector sub-row is
    *unconditional* per reviewer P1-8 — entries lacking an HRSVector
    raise :class:`ValueError`.

    Entries built from a single spike-run archive are tagged
    ``[PARTIAL: <axis>]`` in front of the method name (reviewer P1-3)
    so a one-axis row is never mistaken for a complete HRS-bundle row
    (ADR-008 §Decision binds the bundle to the surviving-axis set).

    Args:
        entries: Sequence of :class:`LeaderboardEntry` records.

    Returns:
        Markdown table suitable for embedding in README or
        leaderboard pages.

    Raises:
        ValueError: When any entry's :class:`HRSVector` is empty
            (no surviving axes) — the leaderboard refuses methods
            that did not produce a HRS vector per reviewer P1-8.
    """
    for entry in entries:
        if not entry.hrs_vector.entries:
            msg = (
                f"leaderboard entry for method {entry.method_id!r} has an empty HRS vector; "
                "ADR-008 §Decision requires the vector to be emitted unconditionally "
                "(reviewer P1-8)"
            )
            raise ValueError(msg)

    sorted_entries = sorted(entries, key=lambda e: e.hrs_scalar, reverse=True)
    lines: list[str] = [
        "| Rank | Method | HRS scalar | Violation rate | Fallback rate |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for rank, entry in enumerate(sorted_entries, start=1):
        if len(entry.spike_runs) == 1:
            partial_axis = entry.hrs_vector.entries[0].axis
            method_label = f"[PARTIAL: {partial_axis}] `{entry.method_id}`"
        else:
            method_label = f"`{entry.method_id}`"
        lines.append(
            f"| {rank} | {method_label} | {entry.hrs_scalar:.3f} "
            f"| {_fmt_pct(entry.violation_rate)} "
            f"| {_fmt_pct(entry.fallback_rate)} |"
        )
        vector_repr = ", ".join(
            f"{e.axis}={e.score:.3f}(w={e.weight:.2f})" for e in entry.hrs_vector.entries
        )
        lines.append(f"|   | ↳ HRS vector | {vector_repr} | | |")
    return "\n".join(lines) + "\n"


__all__ = [
    "render_leaderboard",
    "render_three_table_safety_report",
]
