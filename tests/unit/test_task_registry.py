# SPDX-License-Identifier: Apache-2.0
"""Tier-1 tests for the ``chamber.tasks`` registry (ADR-027 §Versioning).

Pure-Python surface only: registry round-trip, version resolution,
loud-fail lookups, manifest determinism, lazy factory resolution, and
the evidence-path / axis-cell contracts of the registered v1.0 ladder.
No SAPIEN import happens anywhere in this file (ADR-001 §Risks / P2).
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

import chamber.tasks
from chamber.tasks import registry as registry_module
from chamber.tasks.spec import HETEROGENEITY_AXES, TaskSpec

_REPO_ROOT = Path(__file__).parents[2]

#: The exact v1.0 ladder pinned by CB-scope ADR-027 §Tier ladder.
_EXPECTED_SLUGS = [
    "amr_handover_dynamic@v0",
    "co_hold_secure@v1",
    "cocarry@v1",
    "coinsert@v1",
    "handover_place@v1",
    "mpe_cooperative_push@v1",
    "stage0_smoke@v1",
    "stage1_pickplace_as@v1",
    "stage1_pickplace_om@v1",
]


def _spec_kwargs(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "task_id": "dummy_task",
        "version": 1,
        "tier": 0,
        "title": "Dummy",
        "env_factory": "tests.unit.test_task_registry.fake_env_factory",
        "sim_backend": "pure_python",
        "n_agents": 2,
        "action_space_summary": "n/a",
        "observation_summary": "n/a",
        "stress_channel": None,
        "axes": dict.fromkeys(HETEROGENEITY_AXES, "untested"),
        "admission_status": "DIAGNOSTIC",
        "evidence": ["adr/ADR-027-chamber-bench-v1-protocol.md"],
        "notes": "",
    }
    base.update(overrides)
    return base


def fake_env_factory(**kwargs: Any) -> dict[str, Any]:
    """Test-local stand-in factory; returns its kwargs for merge assertions."""
    return kwargs


@pytest.fixture
def isolated_registry(monkeypatch: pytest.MonkeyPatch) -> dict[str, dict[int, TaskSpec]]:
    """Fresh registry table so tests never pollute the process-global ladder."""
    fresh: dict[str, dict[int, TaskSpec]] = {}
    monkeypatch.setattr(registry_module, "_REGISTRY", fresh)
    return fresh


class TestTaskSpec:
    def test_round_trip(self) -> None:
        spec = TaskSpec(**_spec_kwargs())
        assert TaskSpec.model_validate(spec.model_dump()) == spec

    def test_frozen(self) -> None:
        spec = TaskSpec(**_spec_kwargs())
        with pytest.raises(ValidationError):
            spec.task_id = "other"  # type: ignore[misc]

    def test_extra_forbidden(self) -> None:
        with pytest.raises(ValidationError, match="bogus"):
            TaskSpec(**_spec_kwargs(bogus=1))

    def test_axes_must_cover_all_six(self) -> None:
        with pytest.raises(ValidationError, match="missing"):
            TaskSpec(**_spec_kwargs(axes={"AS": "untested"}))
        with pytest.raises(ValidationError, match="extra"):
            TaskSpec(
                **_spec_kwargs(axes={**dict.fromkeys(HETEROGENEITY_AXES, "untested"), "EH": "null"})
            )

    def test_env_factory_must_be_dotted(self) -> None:
        with pytest.raises(ValidationError, match="dotted"):
            TaskSpec(**_spec_kwargs(env_factory="notdotted"))

    def test_slug(self) -> None:
        assert TaskSpec(**_spec_kwargs()).slug == "dummy_task@v1"


class TestRegistry:
    def test_duplicate_registration_loud_fails(
        self, isolated_registry: dict[str, dict[int, TaskSpec]]
    ) -> None:
        @registry_module.register_task
        def build() -> TaskSpec:
            return TaskSpec(**_spec_kwargs())

        with pytest.raises(ValueError, match="already registered"):
            registry_module.register_task(build)

    def test_unknown_id_lists_known(
        self, isolated_registry: dict[str, dict[int, TaskSpec]]
    ) -> None:
        registry_module.register_task(lambda: TaskSpec(**_spec_kwargs()))
        with pytest.raises(KeyError, match="dummy_task"):
            registry_module.get("nope")

    def test_version_resolution_defaults_to_latest(
        self, isolated_registry: dict[str, dict[int, TaskSpec]]
    ) -> None:
        registry_module.register_task(lambda: TaskSpec(**_spec_kwargs(version=1)))
        registry_module.register_task(lambda: TaskSpec(**_spec_kwargs(version=2, notes="v2")))
        assert registry_module.get("dummy_task").version == 2
        assert registry_module.get("dummy_task", version=1).version == 1
        with pytest.raises(KeyError, match="registered versions: 1, 2"):
            registry_module.get("dummy_task", version=9)

    def test_make_merges_defaults_under_overrides(
        self, isolated_registry: dict[str, dict[int, TaskSpec]]
    ) -> None:
        registry_module.register_task(
            lambda: TaskSpec(**_spec_kwargs(factory_defaults={"a": 1, "b": 2}))
        )
        assert registry_module.make("dummy_task", b=3, c=4) == {"a": 1, "b": 3, "c": 4}

    def test_make_placeholder_raises_clearly(
        self, isolated_registry: dict[str, dict[int, TaskSpec]]
    ) -> None:
        registry_module.register_task(
            lambda: TaskSpec(**_spec_kwargs(version=0, tier=3, env_factory=None))
        )
        with pytest.raises(NotImplementedError, match="spec-only"):
            registry_module.make("dummy_task")


class TestLadder:
    def test_registered_slugs_pin_the_v1_suite(self) -> None:
        assert chamber.tasks.list_registered() == _EXPECTED_SLUGS

    def test_manifest_is_deterministic_and_ordered(self) -> None:
        first = json.dumps(chamber.tasks.manifest(), indent=2)
        second = json.dumps(chamber.tasks.manifest(), indent=2)
        assert first == second
        manifest = chamber.tasks.manifest()
        assert manifest["suite"] == "CHAMBER-Bench"
        assert manifest["suite_version"] == "1.0"
        keys = [(t["tier"], t["task_id"], t["version"]) for t in manifest["tasks"]]
        assert keys == sorted(keys)
        assert len(manifest["tasks"]) == len(_EXPECTED_SLUGS)

    @pytest.mark.parametrize("slug", _EXPECTED_SLUGS)
    def test_env_factory_dotted_paths_import(self, slug: str) -> None:
        task_id, _, version = slug.partition("@v")
        spec = chamber.tasks.get(task_id, version=int(version))
        if spec.env_factory is None:
            assert spec.version == 0, "only version-0 placeholders may omit the factory"
            return
        module_name, _, attr = spec.env_factory.rpartition(".")
        factory = getattr(importlib.import_module(module_name), attr)
        assert callable(factory)

    @pytest.mark.parametrize("slug", _EXPECTED_SLUGS)
    def test_evidence_paths_exist_in_repo(self, slug: str) -> None:
        task_id, _, version = slug.partition("@v")
        spec = chamber.tasks.get(task_id, version=int(version))
        assert spec.evidence, "every task version carries evidence"
        missing = [p for p in spec.evidence if not (_REPO_ROOT / p).exists()]
        assert not missing, f"evidence paths missing from repo: {missing}"

    def test_make_constructs_the_tier1_diagnostic(self) -> None:
        env = chamber.tasks.make("mpe_cooperative_push", episode_length=7)
        obs, _info = env.reset(seed=0)
        assert set(obs["agent"]) == {"ego", "partner"}

    def test_placeholders_raise_not_implemented(self) -> None:
        # co_hold_secure left this list at ADR-029 (env factory wired, @v1).
        for task_id in ("amr_handover_dynamic",):
            with pytest.raises(NotImplementedError, match="spec-only"):
                chamber.tasks.make(task_id)

    def test_cocarry_canonical_instrument_is_pinned(self) -> None:
        spec = chamber.tasks.get("cocarry")
        assert spec.stress_channel is not None
        assert "130.5697" in spec.stress_channel
        assert "secondary telemetry" in spec.stress_channel
        assert spec.axes["AS"] == "null"

    def test_stage1_as_is_the_invalid_control(self) -> None:
        spec = chamber.tasks.get("stage1_pickplace_as")
        assert spec.admission_status == "CONTROL"
        assert spec.axes["AS"] == "invalid"

    def test_handover_place_pins_the_measured_valid_region(self) -> None:
        spec = chamber.tasks.get("handover_place")
        assert spec.admission_status == "ADMITTED"
        assert "22 of 216" in spec.notes
        assert "prereg-handover-place-gate0-rev2-2026-06-26" in spec.notes
        assert (
            "spikes/results/admission/handover_place-2026-07-05/admission_report.json"
            in spec.evidence
        )

    def test_cocarry_is_admitted_by_the_committed_report(self) -> None:
        spec = chamber.tasks.get("cocarry")
        assert spec.admission_status == "ADMITTED"
        assert "spikes/results/admission/cocarry-2026-07-05/admission_report.json" in spec.evidence
