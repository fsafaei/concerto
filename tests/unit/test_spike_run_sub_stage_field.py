# SPDX-License-Identifier: Apache-2.0
"""Schema-level tests for ``SpikeRun.sub_stage`` (ADR-016 §Decision; ADR-007 §Stage 1a).

Pins the wire-format contract for the field added in PR 2 of the
2026-05-17 Stage-1a triage:

- ``sub_stage`` is a required typed field on :class:`SpikeRun` ranging
  over ``Literal["1a", "1b", "2", "3"]``.
- Constructing without ``sub_stage`` raises
  :class:`pydantic.ValidationError`.
- Constructing with an out-of-set value (e.g. ``"2a"``, ``"1B"``,
  ``"stage1"``) raises :class:`pydantic.ValidationError`.
- JSON round-trip preserves the field verbatim.
- :data:`SCHEMA_VERSION` is pinned at 2 (PR 2's bump). Bumping it
  further is a breaking change and requires a new ADR amendment per
  the module docstring on :mod:`chamber.evaluation.results`.

The Stage-1a routing behaviour (sub_stage="1a" → Defer) is pinned
separately in :mod:`tests.unit.test_summarize_month3`.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from chamber.evaluation.results import (
    SCHEMA_VERSION,
    ConditionPair,
    EpisodeResult,
    SpikeRun,
)


def _minimal_spike_run_kwargs(*, sub_stage: str = "1a") -> dict[str, Any]:
    """Build the minimum kwargs for a valid :class:`SpikeRun` fixture.

    Centralises the boilerplate so each test only varies the field
    under test. Uses ``"0" * 40`` for ``prereg_sha`` per the existing
    convention in :mod:`tests.integration.test_evaluation_spine`. The
    return type is ``dict[str, Any]`` because the kwargs are
    heterogeneous (str, Literal, ConditionPair, list[int],
    list[EpisodeResult]); unpacking through Pydantic's ``__init__``
    re-validates everything at runtime, so the loose typing here is
    bounded by the model's own ``extra="forbid"`` Pydantic config.
    """
    return {
        "spike_id": "test_spike",
        "prereg_sha": "0" * 40,
        "git_tag": "prereg/test",
        "axis": "AS",
        "sub_stage": sub_stage,
        "condition_pair": ConditionPair(
            homogeneous_id="homo_id",
            heterogeneous_id="hetero_id",
        ),
        "seeds": [0],
        "episode_results": [
            EpisodeResult(
                seed=0,
                episode_idx=0,
                initial_state_seed=0,
                success=True,
                metadata={"condition": "homo_id"},
            )
        ],
    }


class TestSubStageFieldShape:
    """ADR-016 §Decision: ``sub_stage`` is a typed first-class SpikeRun field."""

    @pytest.mark.parametrize("stage", ["1a", "1b", "2", "3"])
    def test_construct_with_valid_sub_stage(self, stage: str) -> None:
        """Every value in the ADR-007 §Implementation-staging set constructs cleanly."""
        run = SpikeRun(**_minimal_spike_run_kwargs(sub_stage=stage))
        assert run.sub_stage == stage

    @pytest.mark.parametrize("bad", ["2a", "1B", "stage1", "", "1", "4"])
    def test_construct_with_invalid_sub_stage_raises(self, bad: str) -> None:
        """Out-of-set values fail loudly at construction time.

        Type-enforced rejection is the load-bearing reason for option
        (a)'s schema-bump approach over the prior metadata-dict
        affordance (ADR-016 §Rationale; 2026-05-17 incident).
        """
        with pytest.raises(ValidationError):
            SpikeRun(**_minimal_spike_run_kwargs(sub_stage=bad))

    def test_sub_stage_is_required(self) -> None:
        """No default — the empty-default is the foot-gun the field was added to close."""
        kwargs = _minimal_spike_run_kwargs()
        kwargs.pop("sub_stage")
        with pytest.raises(ValidationError, match="sub_stage"):
            SpikeRun(**kwargs)


class TestSubStageJSONRoundTrip:
    """ADR-016 §Decision: ``sub_stage`` survives JSON serialisation."""

    @pytest.mark.parametrize("stage", ["1a", "1b", "2", "3"])
    def test_dump_then_validate_preserves_sub_stage(self, stage: str) -> None:
        """``model_dump_json`` + ``model_validate_json`` round-trips the field."""
        run = SpikeRun(**_minimal_spike_run_kwargs(sub_stage=stage))
        dumped = run.model_dump_json()
        round_tripped = SpikeRun.model_validate_json(dumped)
        assert round_tripped.sub_stage == stage

    def test_dumped_json_carries_sub_stage_key(self) -> None:
        """Wire-format pin: ``sub_stage`` appears verbatim as a JSON key."""
        run = SpikeRun(**_minimal_spike_run_kwargs(sub_stage="1a"))
        dumped = run.model_dump_json()
        assert '"sub_stage":"1a"' in dumped or '"sub_stage": "1a"' in dumped


class TestSchemaVersionPin:
    """ADR-016 §Decision: SCHEMA_VERSION bumped 1 → 2 in PR 2."""

    def test_schema_version_is_two(self) -> None:
        """Bumping further is a breaking change; requires a new ADR amendment."""
        assert SCHEMA_VERSION == 2

    def test_new_spike_run_carries_schema_version_two(self) -> None:
        """``SpikeRun.schema_version`` default tracks the module constant."""
        run = SpikeRun(**_minimal_spike_run_kwargs())
        assert run.schema_version == 2
