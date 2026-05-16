# SPDX-License-Identifier: Apache-2.0
"""Three-table safety report emitter (ADR-014 §Decision; T3.9).

ADR-014 §Decision pins the three-table format for every Phase-1
result:

- **Table 1** — per-assumption empirical violation rates (one row per
  CDT assumption A1-A3 from note 42 §III; ADR-014 §Consequences flags
  an A4 row for ISO 10218-2:2025 once Stage-3 SA resolves the safety-
  axis decomposition).
- **Table 2** — per-condition safety violation rates (one row per
  experimental condition: predictor type x conformal mode, mirroring
  Huriot & Sibai 2025 Table I; "fallback fired" column counts braking-
  fallback fires from PR7 :func:`concerto.safety.braking.maybe_brake`).
- **Table 3** — conservativeness gap vs. oracle CBF (conformal lambda
  mean and variance vs. the gt/noLearn baseline from note 42 Table I).

This module emits the three tables as plain JSON + Markdown
side-by-side: the JSON is canonical (downstream consumers parse it
back to typed dataclasses for the leaderboard); the Markdown is for
human review. Both files share a SHA-256 content hash computed over
the canonical JSON so tampering is detectable (P6 reproducibility).
The hash is excluded from the canonical JSON used to compute it (the
hash field is added on emit, removed on parse) so re-rendering the
Markdown does not change the hash.

Plan/03 §3.6 mentions JSON-LD for linked-data framing; Phase-0 ships
plain JSON because the consumer is a single Python downstream
(``chamber.evaluation.reports``) and JSON-LD's ``@context`` is not
load-bearing here. JSON-LD is a Phase-1 upgrade if cross-tool linking
materialises.

The emitter is consumed by ``chamber.evaluation.reports`` (chamber-side
LaTeX rendering + leaderboard.json composition land later in M5/M6 per
plan/10 §2's dependency direction).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pathlib

#: ADR-014 schema version. Bumping is a breaking change to the
#: three-table wire format. A *forward-additive* bump (legacy payloads
#: parse cleanly with the new fields defaulting) lands via an ADR-014
#: §Revision history amendment; a *backward-incompatible* bump (a
#: field is renamed, removed, or changes semantics) requires a fresh
#: ADR-014a. The 2026-05-16 v1 → v2 bump follows the first pattern:
#: ``ConditionRow.max_slack`` and ``ConditionRow.slack_l2`` default to
#: ``0.0`` on read so v1 payloads survive the parse.
#:
#: - **2**: ``ConditionRow`` gains ``max_slack`` and ``slack_l2`` columns
#:   (external-review P0-3, 2026-05-16); see ADR-014 §Decision and
#:   §Revision history. The corresponding telemetry on the OSCBF solver
#:   side ships via :class:`concerto.safety.oscbf.OSCBFResult`.
#: - **1**: Initial three-table format (M3).
SCHEMA_VERSION: int = 2


@dataclass(frozen=True)
class AssumptionRow:
    """Row in Table 1 — per-assumption violation rate (ADR-014 §Decision).

    The CDT assumptions A1-A3 are defined in note 42 §III. ADR-014
    §Consequences notes that ADR-007 rev 3 may add an A4 row for
    "ISO 10218-2:2025 SIL/PL precondition satisfied" once Stage-3 SA
    resolves Open Q #4 (safety-axis decomposition); the schema is
    open under ``assumption: str`` so adding A4 doesn't bump
    :data:`SCHEMA_VERSION` on its own.

    Attributes:
        assumption: Assumption label (e.g. "A1", "A2", "A3", "A4").
        description: One-line human-readable description.
        violations: Count of empirical violations.
        n_steps: Total number of steps observed.
    """

    assumption: str
    description: str
    violations: int
    n_steps: int

    def to_jsonable(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict (ADR-014 §Decision)."""
        return asdict(self)

    @classmethod
    def from_jsonable(cls, data: dict[str, Any]) -> AssumptionRow:
        """Reconstruct from the dict produced by :meth:`to_jsonable` (ADR-014 §Decision)."""
        return cls(
            assumption=str(data["assumption"]),
            description=str(data["description"]),
            violations=int(data["violations"]),
            n_steps=int(data["n_steps"]),
        )


@dataclass(frozen=True)
class ConditionRow:
    """Row in Table 2 — per-condition violation rate (ADR-014 §Decision).

    Condition labels mirror Huriot & Sibai 2025 Table I:
    ``predictor in {"gt", "pred"}`` x ``conformal_mode in
    {"noLearn", "Learn"}``. The ``vendor_compliance`` slot is reserved
    for ADR-007 rev 3 Open Q #4 (decomposition into ISO/TS 15066
    force-pressure and ISO 10218-2 SIL/PL); ``None`` until Stage-3 SA
    resolves.

    Schema v2 (2026-05-16; external-review P0-3) adds the per-condition
    slack-aggregate columns ``max_slack`` and ``slack_l2``. These
    distinguish *constraint-satisfaction* from *constraint-relaxation
    via slack*: a row that "succeeds" only via large slack is not safe
    in the intended sense. The source signal is
    :class:`concerto.safety.oscbf.OSCBFResult.slack` aggregated across
    the condition's steps.

    Attributes:
        predictor: ``"gt"`` (ground-truth) or ``"pred"`` (black-box).
        conformal_mode: ``"noLearn"`` (lambda=eta=0) or ``"Learn"``
            (eta > 0; PR6 conformal layer engaged).
        vendor_compliance: ADR-007 rev 3 placeholder; ``None`` in
            Phase-0.
        n_episodes: Number of episodes in this condition.
        violations: Count of CBF-constraint violations.
        fallback_fires: Count of braking-fallback fires (PR7;
            :func:`concerto.safety.braking.maybe_brake`).
        max_slack: Maximum per-step OSCBF slack observed in this
            condition. ``0.0`` for runs that do not exercise the OSCBF
            inner filter (Phase-0 default; the EGO_ONLY outer filter is
            untouched). Schema v2.
        slack_l2: Mean across steps of ``||slack||_2`` from
            :class:`concerto.safety.oscbf.OSCBFResult`. ``0.0`` under
            the same caveat. Schema v2.
    """

    predictor: str
    conformal_mode: str
    vendor_compliance: str | None
    n_episodes: int
    violations: int
    fallback_fires: int
    max_slack: float = 0.0
    slack_l2: float = 0.0

    def to_jsonable(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict (ADR-014 §Decision)."""
        return asdict(self)

    @classmethod
    def from_jsonable(cls, data: dict[str, Any]) -> ConditionRow:
        """Reconstruct from the dict produced by :meth:`to_jsonable` (ADR-014 §Decision).

        The schema-v2 columns default to ``0.0`` when reading legacy
        v1 payloads — see :data:`SCHEMA_VERSION` for the migration
        envelope.
        """
        vendor = data.get("vendor_compliance")
        return cls(
            predictor=str(data["predictor"]),
            conformal_mode=str(data["conformal_mode"]),
            vendor_compliance=None if vendor is None else str(vendor),
            n_episodes=int(data["n_episodes"]),
            violations=int(data["violations"]),
            fallback_fires=int(data["fallback_fires"]),
            max_slack=float(data.get("max_slack", 0.0)),
            slack_l2=float(data.get("slack_l2", 0.0)),
        )


@dataclass(frozen=True)
class GapRow:
    """Row in Table 3 — conservativeness gap vs. oracle (ADR-014 §Decision).

    The conformal lambda statistics are computed over the steps in a
    given condition; the oracle baseline is gt/noLearn (Huriot & Sibai
    2025 Table I notation), which uses ``lambda = eta = 0`` and
    therefore has zero conformal slack — but it serves as the
    "minimum conservativeness" reference.

    Attributes:
        condition: Free-form label (typically ``"<predictor>/<mode>"``).
        lambda_mean: Mean of conformal slack over the condition.
        lambda_var: Variance of conformal slack.
        oracle_lambda_mean: Mean for the oracle gt/noLearn baseline.
    """

    condition: str
    lambda_mean: float
    lambda_var: float
    oracle_lambda_mean: float

    def to_jsonable(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict (ADR-014 §Decision)."""
        return asdict(self)

    @classmethod
    def from_jsonable(cls, data: dict[str, Any]) -> GapRow:
        """Reconstruct from the dict produced by :meth:`to_jsonable` (ADR-014 §Decision)."""
        return cls(
            condition=str(data["condition"]),
            lambda_mean=float(data["lambda_mean"]),
            lambda_var=float(data["lambda_var"]),
            oracle_lambda_mean=float(data["oracle_lambda_mean"]),
        )


@dataclass(frozen=True)
class ThreeTableReport:
    """Aggregate of the three Phase-1 safety tables (ADR-014 §Decision).

    Attributes:
        table_1: Per-assumption rows (Table 1).
        table_2: Per-condition rows (Table 2).
        table_3: Conservativeness-gap rows (Table 3).
        schema_version: ADR-014 schema version (default
            :data:`SCHEMA_VERSION`).
    """

    table_1: tuple[AssumptionRow, ...]
    table_2: tuple[ConditionRow, ...]
    table_3: tuple[GapRow, ...]
    schema_version: int = SCHEMA_VERSION

    def to_jsonable(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict (ADR-014 §Decision)."""
        return {
            "schema_version": self.schema_version,
            "table_1": [r.to_jsonable() for r in self.table_1],
            "table_2": [r.to_jsonable() for r in self.table_2],
            "table_3": [r.to_jsonable() for r in self.table_3],
        }

    @classmethod
    def from_jsonable(cls, data: dict[str, Any]) -> ThreeTableReport:
        """Reconstruct from the dict produced by :meth:`to_jsonable` (ADR-014 §Decision)."""
        return cls(
            table_1=tuple(AssumptionRow.from_jsonable(r) for r in data["table_1"]),
            table_2=tuple(ConditionRow.from_jsonable(r) for r in data["table_2"]),
            table_3=tuple(GapRow.from_jsonable(r) for r in data["table_3"]),
            schema_version=int(data.get("schema_version", SCHEMA_VERSION)),
        )

    def to_canonical_json(self) -> str:
        """Deterministic JSON serialisation for hashing (ADR-014 §Decision; P6).

        Sorted keys, fixed indentation and separator strings — same
        report ⇒ same bytes. The ``content_hash`` field is **not**
        included so the hash itself can be embedded without changing
        what it covers.
        """
        return json.dumps(
            self.to_jsonable(),
            sort_keys=True,
            indent=2,
            separators=(",", ": "),
        )

    def content_hash(self) -> str:
        """SHA-256 of the canonical JSON (ADR-014 §Decision; P6 tamper detection)."""
        return hashlib.sha256(self.to_canonical_json().encode("utf-8")).hexdigest()


def emit_three_tables(
    *,
    out_dir: pathlib.Path,
    report: ThreeTableReport,
) -> tuple[pathlib.Path, pathlib.Path]:
    """Emit the three-table report as JSON + Markdown (ADR-014 §Decision; T3.9).

    Writes ``three_tables.json`` and ``three_tables.md`` under
    ``out_dir``. Both files share the same SHA-256 content hash
    computed over :meth:`ThreeTableReport.to_canonical_json` (which
    excludes the hash itself), so tampering with either file's
    payload makes the hash mismatch on parse (P6 reproducibility).

    Args:
        out_dir: Destination directory; created if missing.
        report: The :class:`ThreeTableReport` to emit.

    Returns:
        Pair ``(json_path, md_path)`` of the written files.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    sha = report.content_hash()

    json_path = out_dir / "three_tables.json"
    md_path = out_dir / "three_tables.md"

    json_payload = {
        **report.to_jsonable(),
        "content_hash": sha,
    }
    json_path.write_text(
        json.dumps(
            json_payload,
            sort_keys=True,
            indent=2,
            separators=(",", ": "),
        ),
        encoding="utf-8",
    )
    md_path.write_text(_render_markdown(report, sha), encoding="utf-8")
    return json_path, md_path


def parse_three_tables(json_path: pathlib.Path) -> ThreeTableReport:
    """Round-trip parse a three-table JSON file (ADR-014 §Decision; T3.9).

    Reads the JSON file, reconstructs the typed
    :class:`ThreeTableReport`, and verifies the embedded
    ``content_hash`` against a fresh recomputation over the parsed
    report. A mismatch raises :class:`ValueError` so downstream
    consumers fail loudly on tampering.

    Args:
        json_path: Path to the JSON file emitted by
            :func:`emit_three_tables`.

    Returns:
        The reconstructed :class:`ThreeTableReport`.

    Raises:
        ValueError: When the embedded ``content_hash`` does not match
            the recomputed hash of the parsed report.
    """
    data: dict[str, Any] = json.loads(json_path.read_text(encoding="utf-8"))
    expected_hash = data.pop("content_hash", None)
    report = ThreeTableReport.from_jsonable(data)
    if expected_hash is not None:
        actual_hash = report.content_hash()
        if str(expected_hash) != actual_hash:
            msg = f"content_hash mismatch: file claims {expected_hash}, recomputed {actual_hash}"
            raise ValueError(msg)
    return report


def _render_markdown(report: ThreeTableReport, content_hash: str) -> str:
    """Deterministic Markdown rendering of the three tables.

    Header carries the schema version + content hash so the rendered
    file is self-describing.
    """
    lines: list[str] = [
        f"<!-- content_hash: {content_hash} -->",
        f"<!-- schema_version: {report.schema_version} -->",
        "",
        "# Three-table safety report (ADR-014)",
        "",
        "## Table 1 — Per-assumption violation rates",
        "",
        "| Assumption | Description | Violations | N steps |",
        "| --- | --- | --- | --- |",
    ]
    lines.extend(
        f"| {row.assumption} | {row.description} | {row.violations} | {row.n_steps} |"
        for row in report.table_1
    )
    lines += [
        "",
        "## Table 2 — Per-condition violation rates",
        "",
        "| Predictor | Conformal mode | Vendor compliance "
        "| N episodes | Violations | Fallback fires "
        "| Max slack | Slack L2 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    lines.extend(
        f"| {row.predictor} | {row.conformal_mode} "
        f"| {'—' if row.vendor_compliance is None else row.vendor_compliance} "
        f"| {row.n_episodes} | {row.violations} | {row.fallback_fires} "
        f"| {row.max_slack:.6g} | {row.slack_l2:.6g} |"
        for row in report.table_2
    )
    lines += [
        "",
        "## Table 3 — Conservativeness gap vs. oracle",
        "",
        "| Condition | λ mean | λ var | Oracle λ mean |",
        "| --- | --- | --- | --- |",
    ]
    lines.extend(
        f"| {row.condition} | {row.lambda_mean:.6g} "
        f"| {row.lambda_var:.6g} | {row.oracle_lambda_mean:.6g} |"
        for row in report.table_3
    )
    return "\n".join(lines) + "\n"


__all__ = [
    "SCHEMA_VERSION",
    "AssumptionRow",
    "ConditionRow",
    "GapRow",
    "ThreeTableReport",
    "emit_three_tables",
    "parse_three_tables",
]
