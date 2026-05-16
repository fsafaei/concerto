# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``concerto.safety.reporting`` (T3.9; ADR-014 §Decision).

Covers the M3 acceptance criterion #6: "ADR-014 three-table renderer
emits JSON + Markdown that round-trip to dataclasses byte-for-byte."
"""

from __future__ import annotations

import dataclasses
import json
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from concerto.safety.reporting import (
    SCHEMA_VERSION,
    AssumptionRow,
    ConditionRow,
    GapRow,
    ThreeTableReport,
    emit_three_tables,
    parse_three_tables,
)


def _sample_report() -> ThreeTableReport:
    return ThreeTableReport(
        table_1=(
            AssumptionRow(
                assumption="A1",
                description="bounded gradient norms",
                violations=2,
                n_steps=1000,
            ),
            AssumptionRow(
                assumption="A2",
                description="bounded prediction error",
                violations=5,
                n_steps=1000,
            ),
            AssumptionRow(
                assumption="A3",
                description="QP feasibility with lambda",
                violations=0,
                n_steps=1000,
            ),
        ),
        table_2=(
            ConditionRow(
                predictor="gt",
                conformal_mode="noLearn",
                vendor_compliance=None,
                n_episodes=100,
                violations=12,
                fallback_fires=0,
            ),
            ConditionRow(
                predictor="gt",
                conformal_mode="Learn",
                vendor_compliance=None,
                n_episodes=100,
                violations=3,
                fallback_fires=1,
            ),
            ConditionRow(
                predictor="pred",
                conformal_mode="noLearn",
                vendor_compliance=None,
                n_episodes=100,
                violations=18,
                fallback_fires=2,
            ),
            ConditionRow(
                predictor="pred",
                conformal_mode="Learn",
                vendor_compliance=None,
                n_episodes=100,
                violations=7,
                fallback_fires=1,
            ),
        ),
        table_3=(
            GapRow(
                condition="gt/Learn",
                lambda_mean=0.05,
                lambda_var=0.001,
                oracle_lambda_mean=0.0,
            ),
            GapRow(
                condition="pred/Learn",
                lambda_mean=0.07,
                lambda_var=0.003,
                oracle_lambda_mean=0.0,
            ),
        ),
    )


def test_schema_version_pinned() -> None:
    """Schema bumped to 2 for the ADR-014 ``max_slack`` / ``slack_l2`` columns
    (external-review P0-3, 2026-05-16). Bumping this further is a breaking
    change and requires a new ADR amendment.
    """
    assert SCHEMA_VERSION == 2


def test_round_trip_json_to_dataclass(tmp_path: Path) -> None:
    """M3 §6 acceptance #6: JSON ↔ dataclass round-trip."""
    report = _sample_report()
    json_path, _ = emit_three_tables(out_dir=tmp_path, report=report)
    parsed = parse_three_tables(json_path)
    assert parsed == report


def test_emit_writes_json_and_markdown(tmp_path: Path) -> None:
    report = _sample_report()
    json_path, md_path = emit_three_tables(out_dir=tmp_path, report=report)
    assert json_path == tmp_path / "three_tables.json"
    assert md_path == tmp_path / "three_tables.md"
    assert json_path.is_file()
    assert md_path.is_file()


def test_emit_creates_missing_out_dir(tmp_path: Path) -> None:
    nested = tmp_path / "deep" / "nested"
    assert not nested.exists()
    emit_three_tables(out_dir=nested, report=_sample_report())
    assert (nested / "three_tables.json").is_file()


def test_byte_for_byte_re_emit_under_same_inputs(tmp_path: Path) -> None:
    """Acceptance #6: re-emit is byte-stable for both files."""
    report = _sample_report()
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    emit_three_tables(out_dir=out_a, report=report)
    emit_three_tables(out_dir=out_b, report=report)

    assert (out_a / "three_tables.json").read_bytes() == (out_b / "three_tables.json").read_bytes()
    assert (out_a / "three_tables.md").read_bytes() == (out_b / "three_tables.md").read_bytes()


def test_content_hash_in_json_matches_report_hash(tmp_path: Path) -> None:
    report = _sample_report()
    json_path, _ = emit_three_tables(out_dir=tmp_path, report=report)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["content_hash"] == report.content_hash()


def test_content_hash_in_markdown_matches_report_hash(tmp_path: Path) -> None:
    report = _sample_report()
    _, md_path = emit_three_tables(out_dir=tmp_path, report=report)
    md_text = md_path.read_text(encoding="utf-8")
    assert f"content_hash: {report.content_hash()}" in md_text
    assert f"schema_version: {report.schema_version}" in md_text


def test_canonical_json_excludes_content_hash() -> None:
    """The canonical-json hash subject MUST NOT include the hash itself."""
    report = _sample_report()
    canonical = report.to_canonical_json()
    assert "content_hash" not in canonical


def test_parse_rejects_tampered_payload(tmp_path: Path) -> None:
    """ADR-014 §Decision + P6: tampering with the payload makes the hash mismatch."""
    report = _sample_report()
    json_path, _ = emit_three_tables(out_dir=tmp_path, report=report)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    payload["table_2"][0]["violations"] = 9999  # tamper
    json_path.write_text(
        json.dumps(payload, sort_keys=True, indent=2, separators=(",", ": ")),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="content_hash mismatch"):
        parse_three_tables(json_path)


def test_parse_without_content_hash_returns_report(tmp_path: Path) -> None:
    """A JSON without ``content_hash`` parses fine — hash check is opt-in."""
    report = _sample_report()
    payload = report.to_jsonable()  # no content_hash
    json_path = tmp_path / "no_hash.json"
    json_path.write_text(json.dumps(payload), encoding="utf-8")
    parsed = parse_three_tables(json_path)
    assert parsed == report


def test_dataclasses_are_frozen() -> None:
    a = AssumptionRow(assumption="A1", description="x", violations=0, n_steps=10)
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.violations = 99  # type: ignore[misc]

    c = ConditionRow(
        predictor="gt",
        conformal_mode="Learn",
        vendor_compliance=None,
        n_episodes=1,
        violations=0,
        fallback_fires=0,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.violations = 99  # type: ignore[misc]

    g = GapRow(
        condition="x",
        lambda_mean=0.0,
        lambda_var=0.0,
        oracle_lambda_mean=0.0,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        g.lambda_mean = 99.0  # type: ignore[misc]


def test_vendor_compliance_round_trips_when_set(tmp_path: Path) -> None:
    """ADR-007 Q4 placeholder: when vendor_compliance is set, it round-trips."""
    report = ThreeTableReport(
        table_1=(),
        table_2=(
            ConditionRow(
                predictor="gt",
                conformal_mode="Learn",
                vendor_compliance="ISO_10218-2_2025",
                n_episodes=10,
                violations=0,
                fallback_fires=0,
            ),
        ),
        table_3=(),
    )
    json_path, _ = emit_three_tables(out_dir=tmp_path, report=report)
    parsed = parse_three_tables(json_path)
    assert parsed.table_2[0].vendor_compliance == "ISO_10218-2_2025"


def test_markdown_em_dash_for_none_vendor_compliance(tmp_path: Path) -> None:
    """None vendor_compliance renders as em-dash placeholder in Markdown."""
    report = ThreeTableReport(
        table_1=(),
        table_2=(
            ConditionRow(
                predictor="gt",
                conformal_mode="Learn",
                vendor_compliance=None,
                n_episodes=10,
                violations=0,
                fallback_fires=0,
            ),
        ),
        table_3=(),
    )
    _, md_path = emit_three_tables(out_dir=tmp_path, report=report)
    md = md_path.read_text(encoding="utf-8")
    # Em-dash placeholder appears in the rendered table cell.
    assert "—" in md


def test_empty_report_round_trips(tmp_path: Path) -> None:
    empty = ThreeTableReport(table_1=(), table_2=(), table_3=())
    json_path, _ = emit_three_tables(out_dir=tmp_path, report=empty)
    parsed = parse_three_tables(json_path)
    assert parsed == empty


def test_schema_version_round_trips(tmp_path: Path) -> None:
    report = dataclasses.replace(_sample_report(), schema_version=42)
    json_path, _ = emit_three_tables(out_dir=tmp_path, report=report)
    parsed = parse_three_tables(json_path)
    assert parsed.schema_version == 42


def test_condition_row_slack_columns_round_trip(tmp_path: Path) -> None:
    """ADR-014 §Decision (2026-05-16): ``max_slack`` and ``slack_l2`` survive JSON ↔ dataclass."""
    report = ThreeTableReport(
        table_1=(),
        table_2=(
            ConditionRow(
                predictor="gt",
                conformal_mode="Learn",
                vendor_compliance=None,
                n_episodes=42,
                violations=3,
                fallback_fires=1,
                max_slack=0.275,
                slack_l2=0.183,
            ),
        ),
        table_3=(),
    )
    json_path, _ = emit_three_tables(out_dir=tmp_path, report=report)
    parsed = parse_three_tables(json_path)
    assert parsed.table_2[0].max_slack == pytest.approx(0.275)
    assert parsed.table_2[0].slack_l2 == pytest.approx(0.183)


def test_condition_row_legacy_v1_payload_defaults_slack_columns(tmp_path: Path) -> None:
    """Reading a v1 payload (no slack columns) defaults the new fields to 0.0.

    The schema bump from v1 to v2 is forward-additive: a legacy payload
    written before the slack columns existed must parse cleanly with
    ``max_slack=slack_l2=0.0``. The content_hash check is opt-in so the
    test can skip it (hash would differ on the missing keys).
    """
    legacy_payload: dict[str, object] = {
        "schema_version": 1,
        "table_1": [],
        "table_2": [
            {
                "predictor": "gt",
                "conformal_mode": "Learn",
                "vendor_compliance": None,
                "n_episodes": 10,
                "violations": 0,
                "fallback_fires": 0,
                # No max_slack / slack_l2 fields — pre-schema-v2 payload.
            }
        ],
        "table_3": [],
    }
    legacy_path = tmp_path / "legacy.json"
    legacy_path.write_text(json.dumps(legacy_payload), encoding="utf-8")
    parsed = parse_three_tables(legacy_path)
    assert parsed.table_2[0].max_slack == 0.0
    assert parsed.table_2[0].slack_l2 == 0.0


def test_markdown_renders_slack_columns(tmp_path: Path) -> None:
    """Schema-v2 Markdown rendering carries the slack columns inline."""
    report = ThreeTableReport(
        table_1=(),
        table_2=(
            ConditionRow(
                predictor="gt",
                conformal_mode="Learn",
                vendor_compliance=None,
                n_episodes=1,
                violations=0,
                fallback_fires=0,
                max_slack=0.012345,
                slack_l2=0.06789,
            ),
        ),
        table_3=(),
    )
    _, md_path = emit_three_tables(out_dir=tmp_path, report=report)
    md = md_path.read_text(encoding="utf-8")
    assert "Max slack" in md
    assert "Slack L2" in md
    # 6-significant-digit formatting matches Table 3 lambda-mean rendering.
    assert "0.012345" in md
    assert "0.06789" in md
