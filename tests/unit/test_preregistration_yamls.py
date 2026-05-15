# SPDX-License-Identifier: Apache-2.0
"""Schema-load tests for the Phase-0 pre-registration YAMLs (T5a.7; plan/06 §6 #1).

Verifies that the six ADR-007 axis YAMLs under
``spikes/preregistration/`` validate against the canonical
:class:`chamber.evaluation.prereg.PreregistrationSpec` schema and
that each carries the per-axis invariants the Stage-1/2/3 spike
runner relies on (cluster bootstrap, leaderboard run purpose,
five seeds, hundred-episode budget per condition).

What this test does NOT do:

- It does NOT call :func:`chamber.evaluation.prereg.verify_git_tag`.
  Tag-cutting is the maintainer's step per ADR-007 §Discipline;
  the ``git_tag`` field in each YAML is a draft that the maintainer
  tags on the post-merge commit. The launch-time tag-verification
  belongs to ``chamber-spike verify-prereg`` (B6, T5b.1).
- It does NOT launch a spike. The schema is independent of the
  Stage-N benchmark adapter (B8/B9, T5b.2) — the
  ``condition_pair`` identifiers are free-form strings the
  adapter resolves into concrete env builds.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from chamber.evaluation.prereg import PreregistrationSpec, load_prereg

#: Repository root, derived from this test file's location.
_REPO_ROOT: Path = Path(__file__).resolve().parents[2]

#: Directory the six prereg YAMLs live in (plan/06 §3.3).
_PREREG_DIR: Path = _REPO_ROOT / "spikes" / "preregistration"

#: Canonical ADR-007 §3.4 axes — the Option D shortlist. The
#: HRS-bundle ordering (CM > PF > CR > SA > OM > AS) lives in
#: :data:`chamber.evaluation.hrs.DEFAULT_AXIS_WEIGHTS` per ADR-008
#: §Decision; the six prereg files themselves are order-independent.
_AXES: tuple[str, ...] = ("AS", "OM", "CR", "CM", "PF", "SA")

#: Plan/07 §2 sample-size contract: 5 seeds x 20 episodes per
#: (seed, condition) = 100 episodes per condition total.
#: ``episodes_per_seed`` carries the per-(seed, condition) budget.
_EXPECTED_SEEDS: list[int] = [0, 1, 2, 3, 4]
_EXPECTED_EPISODES_PER_SEED: int = 20
_EXPECTED_TOTAL_EPISODES_PER_CONDITION: int = 100


def _prereg_path(axis: str) -> Path:
    return _PREREG_DIR / f"{axis}.yaml"


def test_every_axis_yaml_exists() -> None:
    """plan/06 §6 #1: six pre-registration YAMLs under spikes/preregistration/."""
    for axis in _AXES:
        path = _prereg_path(axis)
        assert path.exists(), f"missing pre-registration YAML for axis {axis!r}: {path}"


@pytest.mark.parametrize("axis", _AXES)
def test_yaml_loads_via_load_prereg(axis: str) -> None:
    """ADR-007 §Discipline: every YAML round-trips through load_prereg.

    :func:`chamber.evaluation.prereg.load_prereg` parses YAML, validates
    against the Pydantic v2 schema, and asserts the axis label is on
    the ADR-007 §3.4 shortlist. A schema or axis-label failure here
    means the YAML drifted from the canonical contract.
    """
    spec = load_prereg(_prereg_path(axis))
    assert isinstance(spec, PreregistrationSpec)
    assert spec.axis == axis


@pytest.mark.parametrize("axis", _AXES)
def test_condition_pair_uses_distinct_identifiers(axis: str) -> None:
    """ADR-007 §Validation criteria: homo / hetero conditions must differ.

    A homogeneous_id == heterogeneous_id collision would mean the
    spike is comparing a condition against itself — the ≥20pp gate
    is undefined under that degenerate case.
    """
    spec = load_prereg(_prereg_path(axis))
    homo = spec.condition_pair.homogeneous_id
    hetero = spec.condition_pair.heterogeneous_id
    assert homo != hetero, f"{axis}: condition_pair has identical homo/hetero ids ({homo!r})"
    assert homo, f"{axis}: condition_pair.homogeneous_id is empty"
    assert hetero, f"{axis}: condition_pair.heterogeneous_id is empty"


@pytest.mark.parametrize("axis", _AXES)
def test_sample_size_matches_plan_07_contract(axis: str) -> None:
    """plan/07 §2: 5 seeds x 20 episodes per (seed, condition) = 100 per condition."""
    spec = load_prereg(_prereg_path(axis))
    assert spec.seeds == _EXPECTED_SEEDS, (
        f"{axis}: expected seeds {_EXPECTED_SEEDS}; got {spec.seeds}"
    )
    assert spec.episodes_per_seed == _EXPECTED_EPISODES_PER_SEED, (
        f"{axis}: episodes_per_seed expected {_EXPECTED_EPISODES_PER_SEED}; "
        f"got {spec.episodes_per_seed}"
    )
    assert spec.episodes_per_seed * len(spec.seeds) == _EXPECTED_TOTAL_EPISODES_PER_CONDITION, (
        f"{axis}: total episodes per condition expected "
        f"{_EXPECTED_TOTAL_EPISODES_PER_CONDITION}; got "
        f"{spec.episodes_per_seed * len(spec.seeds)}"
    )


@pytest.mark.parametrize("axis", _AXES)
def test_leaderboard_runs_use_cluster_bootstrap(axis: str) -> None:
    """reviewer P1-9: leaderboard entries use cluster (or hierarchical) bootstrap.

    The pooled IID bootstrap on seed-clustered episode data
    understates CI width; admitting it to the leaderboard would
    let entries claim tighter intervals than the data supports.
    """
    spec = load_prereg(_prereg_path(axis))
    assert spec.run_purpose == "leaderboard"
    assert spec.bootstrap_method in {"cluster", "hierarchical"}


@pytest.mark.parametrize("axis", _AXES)
def test_failure_policy_is_strict(axis: str) -> None:
    """plan/06 §6: Phase-0 spikes fail loudly on any seed error."""
    spec = load_prereg(_prereg_path(axis))
    assert spec.failure_policy == "strict"


@pytest.mark.parametrize("axis", _AXES)
def test_git_tag_is_a_draft_placeholder(axis: str) -> None:
    """ADR-007 §Discipline: each YAML names the tag the maintainer will cut.

    The tag does not yet exist in the repository — that is intentional.
    The format ``prereg-stage<N>-<axis>-YYYY-MM-DD`` is the project's
    convention (plan/06 §2; plan/08 §9). The test pins the format so
    a future change to the tag-naming scheme is intentional, not
    accidental.
    """
    spec = load_prereg(_prereg_path(axis))
    assert spec.git_tag.startswith("prereg-stage"), (
        f"{axis}: git_tag {spec.git_tag!r} does not start with 'prereg-stage'"
    )
    assert f"-{axis}-" in spec.git_tag, (
        f"{axis}: git_tag {spec.git_tag!r} does not contain axis label {axis!r}"
    )


def test_axis_stage_mapping_matches_adr_007() -> None:
    """ADR-007 §Implementation staging: AS+OM → Stage 1, CR+CM → Stage 2, PF+SA → Stage 3."""
    expected_stage = {
        "AS": "stage1",
        "OM": "stage1",
        "CR": "stage2",
        "CM": "stage2",
        "PF": "stage3",
        "SA": "stage3",
    }
    for axis, stage in expected_stage.items():
        spec = load_prereg(_prereg_path(axis))
        assert stage in spec.git_tag, (
            f"{axis}: git_tag {spec.git_tag!r} does not contain {stage!r} "
            f"(ADR-007 §Implementation staging)"
        )
