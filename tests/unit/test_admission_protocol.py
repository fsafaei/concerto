# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the executable admission protocol (ADR-027 §Admission protocol).

Covers the pre-committed threshold rules, the spec-from-prereg loader,
and the full :func:`chamber.evaluation.admission.run_admission` flow
against fake cell runners inside a throwaway git repo — every measured
cell must come out as a ``chamber-eval verify``-passing v3 bundle
(ADR-028 §Decision 3), the verdict table must match ADR-027 (CONTROL
demotion, NOT_SOLVABLE short-circuit, the A4 UNINSTRUMENTABLE
brittle-instrument verdict per ADR-027 §Admission A4, INDETERMINATE +
the single pre-committed seed extension), and wrapped evidence must be
SHA-verified, never trusted (I8).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import yaml
from pydantic import ValidationError

from chamber.evaluation.admission import (
    AdmissionCellSpec,
    AdmissionError,
    AdmissionReport,
    AdmissionSpec,
    CellRun,
    WrappedEvidenceSpec,
    a1_outcome,
    a2_outcome,
    a3_outcome,
    a4_outcome,
    admission_spec_from_prereg,
    load_admission_report,
    overall_verdict,
    run_admission,
    stress_statistics,
)
from chamber.evaluation.bundles import sha256_file, verify_bundle_dir
from chamber.evaluation.prereg import load_prereg_document
from chamber.evaluation.results import EpisodeResult
from chamber.partners.api import PartnerSpec

if TYPE_CHECKING:
    from collections.abc import Callable

_REPO_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Threshold rules (pure).
# ---------------------------------------------------------------------------


class TestThresholdRules:
    """The pre-committed A1/A2/A3 rules (ADR-027 §Admission protocol)."""

    @pytest.mark.parametrize(
        ("low", "high", "expected"),
        [(0.96, 1.0, "PASS"), (0.90, 0.94, "FAIL"), (0.90, 0.97, "INDETERMINATE")],
    )
    def test_a1(self, low: float, high: float, expected: str) -> None:
        assert a1_outcome(low, high, 0.95) == expected

    @pytest.mark.parametrize(
        ("low", "high", "expected"),
        [(0.0, 0.04, "PASS"), (0.06, 0.30, "FAIL"), (0.02, 0.30, "INDETERMINATE")],
    )
    def test_a2(self, low: float, high: float, expected: str) -> None:
        assert a2_outcome(low, high, 0.05) == expected

    @pytest.mark.parametrize(
        ("low", "high", "expected"),
        [(0.25, 0.60, "PASS"), (0.0, 0.15, "FAIL"), (0.10, 0.40, "INDETERMINATE")],
    )
    def test_a3(self, low: float, high: float, expected: str) -> None:
        assert a3_outcome(low, high, 0.20) == expected

    @pytest.mark.parametrize(
        ("profile", "expected"),
        [
            # Every admitted partner clears the floor -> PASS.
            ({"m1": (0.9, 1.0), "m2": (0.8, 0.95)}, "PASS"),
            # The R-2026-06-C shape: matched-only success; any partner
            # whose upper bound sits below the floor is brittle -> FAIL.
            ({"matched": (1.0, 1.0), "selfish_effort": (0.0, 0.0)}, "FAIL"),
            # A FAIL partner dominates another partner's straddle.
            ({"m1": (0.6, 0.9), "m2": (0.0, 0.1)}, "FAIL"),
            # Weakest partner straddles the floor -> INDETERMINATE.
            ({"m1": (0.9, 1.0), "m2": (0.6, 0.9)}, "INDETERMINATE"),
        ],
    )
    def test_a4(self, profile: dict[str, tuple[float, float]], expected: str) -> None:
        assert a4_outcome(profile, 0.75) == expected

    def test_a4_empty_profile_is_loud(self) -> None:
        with pytest.raises(AdmissionError, match="non-empty"):
            a4_outcome({}, 0.75)

    def test_verdict_table(self) -> None:
        assert overall_verdict("PASS", "PASS", "PASS") == "ADMITTED"
        assert overall_verdict("FAIL", None, None) == "NOT_SOLVABLE"
        assert overall_verdict("PASS", "FAIL", "PASS") == "CONTROL"
        assert overall_verdict("PASS", "PASS", "FAIL") == "CONTROL"
        assert overall_verdict("PASS", "INDETERMINATE", "PASS") == "INDETERMINATE"
        assert overall_verdict("INDETERMINATE", "PASS", "PASS") == "INDETERMINATE"

    def test_verdict_table_with_a4(self) -> None:
        """The A4 fold (ADR-027 §Admission A4): FAIL -> UNINSTRUMENTABLE; None = legacy."""
        assert overall_verdict("PASS", "PASS", "PASS", "PASS") == "ADMITTED"
        assert overall_verdict("PASS", "PASS", "PASS", "FAIL") == "UNINSTRUMENTABLE"
        assert overall_verdict("PASS", "PASS", "PASS", "INDETERMINATE") == "INDETERMINATE"
        # The task-level CONTROL demotion outranks the instrument verdict.
        assert overall_verdict("PASS", "FAIL", "PASS", "FAIL") == "CONTROL"
        assert overall_verdict("FAIL", None, None, None) == "NOT_SOLVABLE"
        # a4 is None (not an instrument contrast): the 3-check table is unchanged.
        assert overall_verdict("PASS", "PASS", "PASS", None) == "ADMITTED"

    def test_stress_statistics_successes_only(self) -> None:
        eps = [
            _episode(0, 0, success=True, force_peak=10.0),
            _episode(0, 1, success=False, force_peak=500.0),
            _episode(0, 2, success=True, force_peak=30.0),
        ]
        stats = stress_statistics(eps)
        assert stats["stress_n"] == 2.0
        assert stats["stress_max"] == 30.0
        assert stress_statistics([_episode(0, 0, success=True)]) == {}


# ---------------------------------------------------------------------------
# Fixtures: throwaway repo + tagged prereg + fake cell runners.
# ---------------------------------------------------------------------------


def _episode(
    seed: int,
    idx: int,
    *,
    success: bool,
    force_peak: float | None = None,
    member: str | None = None,
) -> EpisodeResult:
    metadata: dict[str, str] = {"condition": "fake"}
    if member is not None:
        metadata["member"] = member
    return EpisodeResult(
        seed=seed,
        episode_idx=idx,
        initial_state_seed=idx,
        success=success,
        force_peak=force_peak,
        metadata=metadata,
    )


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)  # noqa: S603,S607


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)  # noqa: S603,S607
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    _git(repo, "config", "commit.gpgsign", "false")
    _git(repo, "config", "tag.gpgsign", "false")
    return repo


_CELL = {
    "cell_id": "a1_reference",
    "runner": "fake",
    "policy_id": "ref_script",
    "partner_name": "scripted_heuristic",
    "params": {},
}

#: A4 instrument cell (ADR-027 §Admission A4): policy_id names the
#: instrument under test, partner_name the admitted set it sweeps.
_A4_CELL = {
    "cell_id": "a4_instrument",
    "runner": "fake",
    "policy_id": "residual_incumbent",
    "partner_name": "cocarry_partners@v1",
    "params": {},
}

#: A1/A2/A3 success functions for an otherwise-admittable task.
_TASK_PASSES = {
    "a1_reference": lambda _s, _e: True,
    "a2_ablated": lambda _s, _e: False,
    "a3_blind": lambda _s, _e: False,
}


def _prereg_payload(**admission_overrides: object) -> dict[str, object]:
    admission: dict[str, object] = {
        "task_version": 1,
        "tau_solv": 0.9,
        "stress_limit": 100.0,
        "tau_infeasible": 0.05,
        "delta_min": 0.2,
        "seeds": [0, 1],
        "episodes_per_seed": 4,
        "extension_seeds": [2],
        "n_resamples": 200,
        "root_seed": 0,
        "a1": dict(_CELL),
        "a2": {**_CELL, "cell_id": "a2_ablated", "partner_name": "partner_ablated_zero"},
        "a3": {**_CELL, "cell_id": "a3_blind", "policy_id": "b_blind"},
    }
    admission.update(admission_overrides)
    return {
        "schema_version": 1,
        "task_id": "faketask",
        "git_tag": "prereg/adm-test",
        "parameters": {"admission": admission},
        "decision_rules": "A1/A2/A3 per ADR-027 §Admission protocol.",
    }


def _write_tagged_prereg(repo: Path, payload: dict[str, object]) -> Path:
    prereg = repo / "prereg.yaml"
    prereg.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")
    _git(repo, "add", "prereg.yaml")
    _git(repo, "commit", "-q", "-m", "lock prereg")
    _git(repo, "tag", "prereg/adm-test")
    return prereg


def _fake_partner_material() -> tuple[list[dict[str, object]], dict[str, str]]:
    spec = PartnerSpec("scripted_heuristic", 0, None, None, {"action_dim": "2"})
    return (
        [
            {
                "name": "partner:scripted_heuristic",
                "class_name": spec.class_name,
                "seed": spec.seed,
                "checkpoint_step": None,
                "weights_uri": None,
                "extra": dict(spec.extra),
            }
        ],
        {"partner:scripted_heuristic": spec.partner_id},
    )


def _make_fake_resolver(
    success_by_cell: dict[str, Callable[[int, int], bool]],
    *,
    force_peak: float | None = 40.0,
    members_by_cell: dict[str, dict[str, Callable[[int, int], bool]]] | None = None,
    member_force_peak: dict[str, float | None] | None = None,
) -> Callable[[str], Callable[..., CellRun]]:
    """Fake cell-runner resolver: success per (seed, episode) by cell id.

    A cell listed in ``members_by_cell`` instead sweeps a fake admitted
    partner set — per member a success function, episodes stamped with
    ``metadata["member"]`` (the A4 instrument-cell convention);
    ``member_force_peak`` overrides ``force_peak`` per member.
    """

    def _resolver(name: str) -> Callable[..., CellRun]:
        assert name == "fake"

        def _runner(
            *,
            cell: AdmissionCellSpec,
            seeds: list[int],
            episodes_per_seed: int,
            root_seed: int,
            render_backend: str | None = None,
        ) -> CellRun:
            del root_seed, render_backend
            base_cell = cell.cell_id.split("-")[0]
            material, hashes = _fake_partner_material()
            members = (members_by_cell or {}).get(base_cell)
            if members is not None:
                episodes_by_seed = {
                    s: [
                        _episode(
                            s,
                            m_idx * episodes_per_seed + e,
                            success=member_fn(s, e),
                            force_peak=(member_force_peak or {}).get(member, force_peak),
                            member=member,
                        )
                        for m_idx, (member, member_fn) in enumerate(sorted(members.items()))
                        for e in range(episodes_per_seed)
                    ]
                    for s in seeds
                }
            else:
                fn = success_by_cell[base_cell]
                episodes_by_seed = {
                    s: [
                        _episode(s, e, success=fn(s, e), force_peak=force_peak)
                        for e in range(episodes_per_seed)
                    ]
                    for s in seeds
                }
            return CellRun(
                episodes_by_seed=episodes_by_seed,
                partner_material=material,
                partner_hashes=hashes,
                substream_labels=["fake.substream"],
            )

        return _runner

    return _resolver


def _run(
    repo: Path,
    prereg: Path,
    resolver: Callable[[str], Callable[..., CellRun]],
    *,
    out_name: str = "admission",
) -> tuple[object, Path]:
    spec = admission_spec_from_prereg(load_prereg_document(prereg))
    out_dir = repo / "spikes" / out_name
    report = run_admission(
        spec,
        out_dir=out_dir,
        repo_path=repo,
        prereg_path=prereg,
        date_stamp="2026-07-05",
        repro_command="chamber-eval admission --prereg prereg.yaml",
        cell_runner_resolver=resolver,
    )
    return report, out_dir


# ---------------------------------------------------------------------------
# The full flow.
# ---------------------------------------------------------------------------


class TestRunAdmission:
    """End-to-end protocol runs against fake cells (ADR-027 §Admission protocol)."""

    def test_admitted_flow_bundles_verify(self, tmp_path: Path) -> None:
        """All-pass → ADMITTED; every cell bundle passes chamber-eval verify."""
        repo = _init_repo(tmp_path)
        prereg = _write_tagged_prereg(repo, _prereg_payload())
        resolver = _make_fake_resolver(
            {
                "a1_reference": lambda _s, _e: True,
                "a2_ablated": lambda _s, _e: False,
                "a3_blind": lambda _s, _e: False,
            }
        )
        report, out_dir = _run(repo, prereg, resolver)
        assert report.verdict == "ADMITTED"  # type: ignore[attr-defined]
        for name in ("a1_reference", "a2_ablated", "a3_blind"):
            rows = verify_bundle_dir(out_dir / name, repo_path=repo)
            assert all(r.ok for r in rows), [r for r in rows if not r.ok]
        loaded = load_admission_report(out_dir / "admission_report.json")
        assert loaded.verdict == "ADMITTED"
        assert loaded.binding_evidence["stress_max"] == 40.0
        assert (out_dir / "ADMISSION_REPORT.md").is_file()
        assert (out_dir / "SHA256SUMS.txt").is_file()

    def test_control_demotion_on_a3_null(self, tmp_path: Path) -> None:
        """Blind ego matches the reference → A3 FAIL → CONTROL (the demotion rule)."""
        repo = _init_repo(tmp_path)
        prereg = _write_tagged_prereg(repo, _prereg_payload())
        resolver = _make_fake_resolver(
            {
                "a1_reference": lambda _s, _e: True,
                "a2_ablated": lambda _s, _e: False,
                "a3_blind": lambda _s, _e: True,  # blind solves it — ego-solvable
            }
        )
        report, _ = _run(repo, prereg, resolver)
        assert report.verdict == "CONTROL"  # type: ignore[attr-defined]
        a3 = next(c for c in report.checks if c.check == "A3")  # type: ignore[attr-defined]
        assert a3.outcome == "FAIL"
        assert a3.statistics["delta_iqm"] == 0.0

    def test_control_on_a2_failure(self, tmp_path: Path) -> None:
        """Ablated variant still succeeds → A2 FAIL → CONTROL (single-robot-solvable)."""
        repo = _init_repo(tmp_path)
        prereg = _write_tagged_prereg(repo, _prereg_payload())
        resolver = _make_fake_resolver(
            {
                "a1_reference": lambda _s, _e: True,
                "a2_ablated": lambda _s, _e: True,
                "a3_blind": lambda _s, _e: False,
            }
        )
        report, _ = _run(repo, prereg, resolver)
        assert report.verdict == "CONTROL"  # type: ignore[attr-defined]

    def test_not_solvable_short_circuits(self, tmp_path: Path) -> None:
        """A1 FAIL → NOT_SOLVABLE; A2/A3 never run (thresholds do not move)."""
        repo = _init_repo(tmp_path)
        prereg = _write_tagged_prereg(repo, _prereg_payload())
        resolver = _make_fake_resolver(
            {
                "a1_reference": lambda _s, _e: False,
                "a2_ablated": lambda _s, _e: False,
                "a3_blind": lambda _s, _e: False,
            }
        )
        report, out_dir = _run(repo, prereg, resolver)
        assert report.verdict == "NOT_SOLVABLE"  # type: ignore[attr-defined]
        assert [c.check for c in report.checks] == ["A1"]  # type: ignore[attr-defined]
        assert not (out_dir / "a2_ablated").exists()
        assert not (out_dir / "a3_blind").exists()

    def test_stress_violation_fails_a1(self, tmp_path: Path) -> None:
        """A success-rate pass with stress over the committed limit is still an A1 FAIL."""
        repo = _init_repo(tmp_path)
        prereg = _write_tagged_prereg(repo, _prereg_payload())
        resolver = _make_fake_resolver(
            {
                "a1_reference": lambda _s, _e: True,
                "a2_ablated": lambda _s, _e: False,
                "a3_blind": lambda _s, _e: False,
            },
            force_peak=150.0,  # > stress_limit 100.0
        )
        report, _ = _run(repo, prereg, resolver)
        assert report.verdict == "NOT_SOLVABLE"  # type: ignore[attr-defined]

    def test_missing_stress_channel_with_committed_limit_is_loud(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        prereg = _write_tagged_prereg(repo, _prereg_payload())
        resolver = _make_fake_resolver(
            {
                "a1_reference": lambda _s, _e: True,
                "a2_ablated": lambda _s, _e: False,
                "a3_blind": lambda _s, _e: False,
            },
            force_peak=None,
        )
        with pytest.raises(AdmissionError, match="force_peak"):
            _run(repo, prereg, resolver)

    def test_indeterminate_consumes_single_extension(self, tmp_path: Path) -> None:
        """A straddled A2 CI runs the pre-committed extension once, then is final."""
        repo = _init_repo(tmp_path)
        prereg = _write_tagged_prereg(repo, _prereg_payload())

        # Ablated succeeds on every episode of seed 1 only → the 2-cluster
        # bootstrap straddles tau_infeasible; extension seed 2 fails all.
        resolver = _make_fake_resolver(
            {
                "a1_reference": lambda _s, _e: True,
                "a2_ablated": lambda s, _e: s == 1,
                "a3_blind": lambda _s, _e: False,
            }
        )
        report, out_dir = _run(repo, prereg, resolver)
        a2 = next(c for c in report.checks if c.check == "A2")  # type: ignore[attr-defined]
        assert a2.extended is True
        assert report.seed_extension_used is True  # type: ignore[attr-defined]
        assert a2.outcome in ("PASS", "FAIL")
        assert (out_dir / "a2_ablated-ext").is_dir()

    def test_dirty_tree_refused(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        prereg = _write_tagged_prereg(repo, _prereg_payload())
        (repo / "scratch.txt").write_text("dirty", encoding="utf-8")
        resolver = _make_fake_resolver({"a1_reference": lambda _s, _e: True})
        with pytest.raises(AdmissionError, match="dirty"):
            _run(repo, prereg, resolver)

    def test_non_empty_archive_dir_refused(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        prereg = _write_tagged_prereg(repo, _prereg_payload())
        out_dir = repo / "spikes" / "admission"
        out_dir.mkdir(parents=True)
        (out_dir / "existing.txt").write_text("x", encoding="utf-8")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "seed archive dir")
        resolver = _make_fake_resolver({"a1_reference": lambda _s, _e: True})
        with pytest.raises(AdmissionError, match="non-empty"):
            _run(repo, prereg, resolver)


class TestA4InstrumentGate:
    """The A4 ego-robustness gate end-to-end (ADR-027 §Admission A4; ADR-026 §Decision 2)."""

    def test_brittle_instrument_is_uninstrumentable(self, tmp_path: Path) -> None:
        """The R-2026-06-C finding, executable: matched-only success is a FAIL.

        The instrument succeeds 1.0 with its matched partner but ~0.5 /
        0.5 / 0.0 with the rest of the admitted set (the Rung-5
        ``CONFOUNDED_BY_INCUMBENT_BRITTLENESS`` shape) — A4 FAIL, the
        verdict is UNINSTRUMENTABLE (not CONTROL, not a null), and the
        per-partner profile is written to the report.
        """
        repo = _init_repo(tmp_path)
        prereg = _write_tagged_prereg(repo, _prereg_payload(c_min_ego=0.75, a4=dict(_A4_CELL)))
        resolver = _make_fake_resolver(
            dict(_TASK_PASSES),
            members_by_cell={
                "a4_instrument": {
                    "imp_matched": lambda _s, _e: True,
                    "admittance": lambda _s, e: e % 2 == 0,
                    "selfish_goal": lambda _s, e: e % 2 == 0,
                    "selfish_effort": lambda _s, _e: False,
                }
            },
        )
        report, out_dir = _run(repo, prereg, resolver)
        assert report.verdict == "UNINSTRUMENTABLE"  # type: ignore[attr-defined]
        assert [c.check for c in report.checks] == ["A1", "A2", "A3", "A4"]  # type: ignore[attr-defined]
        a4 = next(c for c in report.checks if c.check == "A4")  # type: ignore[attr-defined]
        assert a4.outcome == "FAIL"
        assert a4.statistics["n_partners"] == 4.0
        assert "selfish_effort" in a4.notes
        rows = verify_bundle_dir(out_dir / "a4_instrument", repo_path=repo)
        assert all(r.ok for r in rows), [r for r in rows if not r.ok]
        payload = json.loads((out_dir / "admission_report.json").read_text(encoding="utf-8"))
        profile = payload["ego_robustness_profile"]
        assert set(profile) == {"imp_matched", "admittance", "selfish_goal", "selfish_effort"}
        assert profile["imp_matched"]["success_mean"] == 1.0
        assert profile["selfish_effort"]["success_mean"] == 0.0
        loaded = load_admission_report(out_dir / "admission_report.json")
        assert loaded.schema_version == 1
        assert loaded.ego_robustness_profile == profile
        md = (out_dir / "ADMISSION_REPORT.md").read_text(encoding="utf-8")
        assert "A4 instrument robustness" in md
        assert "selfish_effort" in md
        assert "`c_min_ego` = 0.75" in md

    def test_robust_instrument_is_admitted(self, tmp_path: Path) -> None:
        """Every admitted partner clears c_min_ego -> A4 PASS -> ADMITTED, profile reported."""
        repo = _init_repo(tmp_path)
        prereg = _write_tagged_prereg(repo, _prereg_payload(c_min_ego=0.75, a4=dict(_A4_CELL)))
        resolver = _make_fake_resolver(
            dict(_TASK_PASSES),
            members_by_cell={
                "a4_instrument": {
                    "imp_matched": lambda _s, _e: True,
                    "admittance": lambda _s, _e: True,
                    "selfish_goal": lambda _s, _e: True,
                }
            },
        )
        report, out_dir = _run(repo, prereg, resolver)
        assert report.verdict == "ADMITTED"  # type: ignore[attr-defined]
        a4 = next(c for c in report.checks if c.check == "A4")  # type: ignore[attr-defined]
        assert a4.outcome == "PASS"
        assert a4.statistics["min_success_ci_low"] == 1.0
        loaded = load_admission_report(out_dir / "admission_report.json")
        assert loaded.ego_robustness_profile is not None
        assert set(loaded.ego_robustness_profile) == {"imp_matched", "admittance", "selfish_goal"}

    def test_a4_straddle_consumes_single_extension(self, tmp_path: Path) -> None:
        """A straddling weakest partner runs the pre-committed extension once, then is final."""
        repo = _init_repo(tmp_path)
        prereg = _write_tagged_prereg(repo, _prereg_payload(c_min_ego=0.75, a4=dict(_A4_CELL)))
        # One member succeeds on seed >= 1 only: the 2-cluster bootstrap
        # straddles the floor; extension seed 2 also succeeds.
        resolver = _make_fake_resolver(
            dict(_TASK_PASSES),
            members_by_cell={
                "a4_instrument": {
                    "imp_matched": lambda _s, _e: True,
                    "flaky": lambda s, _e: s >= 1,
                }
            },
        )
        report, out_dir = _run(repo, prereg, resolver)
        a4 = next(c for c in report.checks if c.check == "A4")  # type: ignore[attr-defined]
        assert a4.extended is True
        assert a4.outcome in ("PASS", "FAIL")
        assert report.seed_extension_used is True  # type: ignore[attr-defined]
        assert (out_dir / "a4_instrument-ext").is_dir()

    def test_a4_per_partner_stress_violation_fails(self, tmp_path: Path) -> None:
        """A success-rate pass over the stress ceiling with one partner is still a FAIL.

        The A4 bar is the task's own success+stress bar, held per
        partner (ADR-027 §Admission A4).
        """
        repo = _init_repo(tmp_path)
        prereg = _write_tagged_prereg(repo, _prereg_payload(c_min_ego=0.75, a4=dict(_A4_CELL)))
        resolver = _make_fake_resolver(
            dict(_TASK_PASSES),
            members_by_cell={
                "a4_instrument": {
                    "imp_matched": lambda _s, _e: True,
                    "imp_stiff": lambda _s, _e: True,
                }
            },
            member_force_peak={"imp_stiff": 150.0},  # > stress_limit 100.0
        )
        report, _ = _run(repo, prereg, resolver)
        assert report.verdict == "UNINSTRUMENTABLE"  # type: ignore[attr-defined]
        a4 = next(c for c in report.checks if c.check == "A4")  # type: ignore[attr-defined]
        assert a4.outcome == "FAIL"

    def test_a4_missing_stress_channel_with_committed_limit_is_loud(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        prereg = _write_tagged_prereg(repo, _prereg_payload(c_min_ego=0.75, a4=dict(_A4_CELL)))
        resolver = _make_fake_resolver(
            dict(_TASK_PASSES),
            members_by_cell={"a4_instrument": {"imp_matched": lambda _s, _e: True}},
            member_force_peak={"imp_matched": None},
        )
        with pytest.raises(AdmissionError, match="A4 stress_limit"):
            _run(repo, prereg, resolver)

    def test_a4_unstamped_members_are_loud(self, tmp_path: Path) -> None:
        """An instrument cell that fails to stamp metadata['member'] cannot pass silently."""
        repo = _init_repo(tmp_path)
        prereg = _write_tagged_prereg(repo, _prereg_payload(c_min_ego=0.75, a4=dict(_A4_CELL)))
        resolver = _make_fake_resolver({**_TASK_PASSES, "a4_instrument": lambda _s, _e: True})
        with pytest.raises(AdmissionError, match="member"):
            _run(repo, prereg, resolver)

    def test_a4_ragged_per_seed_counts_are_loud(self, tmp_path: Path) -> None:
        """An instrument runner that under-delivers on one seed fails the schedule check."""
        repo = _init_repo(tmp_path)
        prereg = _write_tagged_prereg(repo, _prereg_payload(c_min_ego=0.75, a4=dict(_A4_CELL)))
        inner = _make_fake_resolver(
            dict(_TASK_PASSES),
            members_by_cell={"a4_instrument": {"imp_matched": lambda _s, _e: True}},
        )

        def _resolver(name: str) -> Callable[..., CellRun]:
            runner = inner(name)

            def _ragged(
                *,
                cell: AdmissionCellSpec,
                seeds: list[int],
                episodes_per_seed: int,
                root_seed: int,
                render_backend: str | None = None,
            ) -> CellRun:
                run = runner(
                    cell=cell,
                    seeds=seeds,
                    episodes_per_seed=episodes_per_seed,
                    root_seed=root_seed,
                    render_backend=render_backend,
                )
                if not cell.cell_id.startswith("a4_instrument"):
                    return run
                first = min(run.episodes_by_seed)
                trimmed = {
                    s: (records[:-1] if s == first else records)
                    for s, records in run.episodes_by_seed.items()
                }
                return CellRun(
                    episodes_by_seed=trimmed,
                    partner_material=run.partner_material,
                    partner_hashes=run.partner_hashes,
                    substream_labels=run.substream_labels,
                )

            return _ragged

        with pytest.raises(AdmissionError, match="episodes_per_seed"):
            _run(repo, prereg, _resolver)

    def test_a4_extension_member_drift_is_loud(self, tmp_path: Path) -> None:
        """The pre-committed extension must sweep the same admitted member set."""
        repo = _init_repo(tmp_path)
        prereg = _write_tagged_prereg(repo, _prereg_payload(c_min_ego=0.75, a4=dict(_A4_CELL)))
        base = _make_fake_resolver(
            dict(_TASK_PASSES),
            members_by_cell={
                "a4_instrument": {
                    "imp_matched": lambda _s, _e: True,
                    "flaky": lambda s, _e: s >= 1,  # straddles -> triggers the extension
                }
            },
        )
        drifted = _make_fake_resolver(
            dict(_TASK_PASSES),
            members_by_cell={"a4_instrument": {"imp_matched": lambda _s, _e: True}},
        )

        def _resolver(name: str) -> Callable[..., CellRun]:
            base_runner = base(name)
            drift_runner = drifted(name)

            def _runner(
                *,
                cell: AdmissionCellSpec,
                seeds: list[int],
                episodes_per_seed: int,
                root_seed: int,
                render_backend: str | None = None,
            ) -> CellRun:
                # The extension run is identified by its committed seeds
                # (the runner sees the same cell spec for both runs).
                runner = drift_runner if seeds == [2] else base_runner
                return runner(
                    cell=cell,
                    seeds=seeds,
                    episodes_per_seed=episodes_per_seed,
                    root_seed=root_seed,
                    render_backend=render_backend,
                )

            return _runner

        with pytest.raises(AdmissionError, match="member set"):
            _run(repo, prereg, _resolver)

    def test_legacy_admission_without_a4_is_unchanged(self, tmp_path: Path) -> None:
        """No c_min_ego committed -> A4 skipped, three checks, no profile field content."""
        repo = _init_repo(tmp_path)
        prereg = _write_tagged_prereg(repo, _prereg_payload())
        report, out_dir = _run(repo, prereg, _make_fake_resolver(dict(_TASK_PASSES)))
        assert report.verdict == "ADMITTED"  # type: ignore[attr-defined]
        assert [c.check for c in report.checks] == ["A1", "A2", "A3"]  # type: ignore[attr-defined]
        payload = json.loads((out_dir / "admission_report.json").read_text(encoding="utf-8"))
        assert payload["ego_robustness_profile"] is None
        assert not (out_dir / "a4_instrument").exists()


class TestWrappedEvidence:
    """Wrap = SHA-verify + re-extract, never trust (I8)."""

    def _wrapped_payload(self, repo: Path, *, tamper: bool = False) -> dict[str, object]:
        archive = repo / "spikes" / "gate0"
        archive.mkdir(parents=True)
        cells = [
            {
                "takt_s": 1.5,
                "clearance_factor": 0.2,
                "mismatch_bias_deg": 30.0,
                "arm_basis": "fast",
                "matched_iqm": 1.0,
                "matched_ci_low": 1.0,
                "gap_pp": 45.5,
                "gap_ci_low_pp": 45.5,
                "gap_ci_high_pp": 100.0,
                "coupling_valid": True,
            },
            {
                "takt_s": 2.0,
                "clearance_factor": 0.2,
                "mismatch_bias_deg": 45.0,
                "arm_basis": "slow",
                "matched_iqm": 1.0,
                "matched_ci_low": 1.0,
                "gap_pp": 100.0,
                "gap_ci_low_pp": 100.0,
                "gap_ci_high_pp": 100.0,
                "coupling_valid": True,
            },
        ]
        crossover = archive / "crossover_curves.json"
        crossover.write_text(json.dumps({"cells": cells}), encoding="utf-8")
        sha = sha256_file(crossover)
        rel = "spikes/gate0/crossover_curves.json"
        if tamper:
            crossover.write_text(json.dumps({"cells": cells[:1]}), encoding="utf-8")
        wrapped_a1 = {
            "archive": "spikes/gate0",
            "files": {rel: sha},
            "extractor": "handover_gate0_limb1",
            "params": {"takt_band_s": [1.0, 5.0]},
        }
        wrapped_a3 = {
            "archive": "spikes/gate0",
            "files": {rel: sha},
            "extractor": "handover_gate0_limb2",
            "params": {
                "clearance_factor": 0.2,
                "mismatch_bias_deg": [30.0, 45.0],
                "takt_band_s": [1.0, 5.0],
            },
        }
        return _prereg_payload(a1=wrapped_a1, a3=wrapped_a3, stress_limit=None)

    def test_wrapped_a1_a3_admit_with_fresh_a2(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        payload = self._wrapped_payload(repo)
        _git(repo, "add", "-A")
        prereg = _write_tagged_prereg(repo, payload)
        resolver = _make_fake_resolver({"a2_ablated": lambda _s, _e: False}, force_peak=None)
        report, out_dir = _run(repo, prereg, resolver)
        assert report.verdict == "ADMITTED"  # type: ignore[attr-defined]
        a1 = next(c for c in report.checks if c.check == "A1")  # type: ignore[attr-defined]
        assert a1.bundles == []
        assert a1.statistics["success_ci_low"] == 1.0
        a3 = next(c for c in report.checks if c.check == "A3")  # type: ignore[attr-defined]
        assert a3.statistics["delta_ci_low"] == pytest.approx(0.455)
        rows = verify_bundle_dir(out_dir / "a2_ablated", repo_path=repo)
        assert all(r.ok for r in rows)

    def test_tampered_wrapped_evidence_is_refused(self, tmp_path: Path) -> None:
        repo = _init_repo(tmp_path)
        payload = self._wrapped_payload(repo, tamper=True)
        _git(repo, "add", "-A")
        prereg = _write_tagged_prereg(repo, payload)
        resolver = _make_fake_resolver({"a2_ablated": lambda _s, _e: False}, force_peak=None)
        with pytest.raises(AdmissionError, match="SHA-256 mismatch"):
            _run(repo, prereg, resolver)


def _write_bundle_episodes(
    repo: Path, rel_dir: str, episodes_by_seed: dict[int, list[EpisodeResult]]
) -> dict[str, str]:
    """Write committed-bundle-style episode files; return their rel-path -> SHA-256 pins."""
    out = repo / rel_dir
    out.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}
    for seed, records in sorted(episodes_by_seed.items()):
        path = out / f"episodes_seed{seed}.jsonl"
        path.write_text(
            "".join(record.model_dump_json() + "\n" for record in records), encoding="utf-8"
        )
        files[f"{rel_dir}/episodes_seed{seed}.jsonl"] = sha256_file(path)
    return files


class TestWrappedA4:
    """Wrapped A4: SHA-verify + re-extract the profile, same rule, straddle final (ADR-027 A4)."""

    _MEMBERS = ("imp_a", "imp_b")

    def _fully_wrapped_payload(
        self,
        repo: Path,
        *,
        member_success: dict[str, Callable[[int, int], bool]] | None = None,
        member_force: dict[str, float | None] | None = None,
        members_pin: list[str] | None = None,
        a1_extractor: str = "bundle_success_summary",
        a4_extractor: str = "bundle_ego_robustness_profile",
        a1_force: float | None = 40.0,
        a2_success: Callable[[int, int], bool] | None = None,
        tamper_a4: bool = False,
    ) -> dict[str, object]:
        """A fully wrapped admission payload — the retrospective re-issue shape (I8)."""
        seeds = [0, 1]
        n_eps = 4
        success = member_success or {m: (lambda _s, _e: True) for m in self._MEMBERS}
        force: dict[str, float | None] = member_force or {}
        a2_fn = a2_success or (lambda _s, _e: False)
        a1_files = _write_bundle_episodes(
            repo,
            "spikes/wrapped/a1_reference",
            {
                s: [_episode(s, e, success=True, force_peak=a1_force) for e in range(n_eps)]
                for s in seeds
            },
        )
        a2_files = _write_bundle_episodes(
            repo,
            "spikes/wrapped/a2_single_arm",
            {
                s: [_episode(s, e, success=a2_fn(s, e), force_peak=40.0) for e in range(n_eps)]
                for s in seeds
            },
        )
        a3_files = _write_bundle_episodes(
            repo,
            "spikes/wrapped/a3_blind",
            {
                s: [_episode(s, e, success=False, force_peak=40.0) for e in range(n_eps)]
                for s in seeds
            },
        )
        a4_files = _write_bundle_episodes(
            repo,
            "spikes/wrapped/a4_instrument",
            {
                s: [
                    _episode(
                        s,
                        m_idx * n_eps + e,
                        success=success[m](s, e),
                        force_peak=force.get(m, 40.0),
                        member=m,
                    )
                    for m_idx, m in enumerate(self._MEMBERS)
                    for e in range(n_eps)
                ]
                for s in seeds
            },
        )
        if tamper_a4:
            path = repo / "spikes/wrapped/a4_instrument/episodes_seed0.jsonl"
            path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        if a1_extractor == "bundle_ego_robustness_profile":
            # The shape-guard case: point A1 at member-stamped episodes so
            # the extractor returns a nested profile to a scalar check.
            wrapped_a1 = {
                "archive": "spikes/wrapped",
                "files": a4_files,
                "extractor": a1_extractor,
                "params": {"bundle_dir": "spikes/wrapped/a4_instrument", "n_resamples": 200},
            }
        else:
            wrapped_a1 = {
                "archive": "spikes/wrapped",
                "files": a1_files,
                "extractor": a1_extractor,
                "params": {"bundle_dir": "spikes/wrapped/a1_reference", "n_resamples": 200},
            }
        wrapped_a2 = {
            "archive": "spikes/wrapped",
            "files": a2_files,
            "extractor": "bundle_success_summary",
            "params": {
                "bundle_dir": "spikes/wrapped/a2_single_arm",
                "n_resamples": 200,
                "stress_successes_only": False,
            },
        }
        wrapped_a3 = {
            "archive": "spikes/wrapped",
            "files": {**a1_files, **a3_files},
            "extractor": "bundle_paired_delta",
            "params": {
                "reference_dir": "spikes/wrapped/a1_reference",
                "blind_dir": "spikes/wrapped/a3_blind",
                "n_resamples": 200,
            },
        }
        wrapped_a4 = {
            "archive": "spikes/wrapped/a4_instrument",
            "files": a4_files,
            "extractor": a4_extractor,
            "params": {"members": members_pin or list(self._MEMBERS), "n_resamples": 200},
        }
        return _prereg_payload(
            a1=wrapped_a1, a2=wrapped_a2, a3=wrapped_a3, c_min_ego=0.75, a4=wrapped_a4
        )

    def _run_wrapped(
        self, tmp_path: Path, **payload_kwargs: object
    ) -> tuple[AdmissionReport, Path]:
        repo = _init_repo(tmp_path)
        payload = self._fully_wrapped_payload(repo, **payload_kwargs)  # type: ignore[arg-type]
        _git(repo, "add", "-A")
        prereg = _write_tagged_prereg(repo, payload)
        resolver = _make_fake_resolver({})  # nothing is measured on a fully wrapped run
        report, out_dir = _run(repo, prereg, resolver)
        assert isinstance(report, AdmissionReport)
        return report, out_dir

    def test_fully_wrapped_admission_admits_and_writes_profile(self, tmp_path: Path) -> None:
        report, out_dir = self._run_wrapped(tmp_path)
        assert report.verdict == "ADMITTED"
        a4 = next(c for c in report.checks if c.check == "A4")
        assert a4.outcome == "PASS"
        assert a4.bundles == []
        assert a4.extended is False
        assert a4.statistics["n_partners"] == 2.0
        assert "wrapped committed evidence" in a4.notes
        profile = report.ego_robustness_profile
        assert profile is not None
        assert set(profile) == set(self._MEMBERS)
        assert profile["imp_a"]["success_ci_low"] == 1.0
        assert profile["imp_a"]["stress_max"] == 40.0
        # A fully wrapped archive carries no cell bundles — report files only.
        assert sorted(p.name for p in out_dir.iterdir()) == [
            "ADMISSION_REPORT.md",
            "SHA256SUMS.txt",
            "admission_report.json",
        ]
        assert load_admission_report(out_dir / "admission_report.json") == report

    def test_wrapped_a4_straddle_is_final_fail(self, tmp_path: Path) -> None:
        """A straddling member fails the wrapped gate — no extension on committed evidence."""
        report, _ = self._run_wrapped(
            tmp_path,
            member_success={"imp_a": lambda _s, _e: True, "imp_b": lambda s, _e: s == 0},
        )
        assert report.verdict == "UNINSTRUMENTABLE"
        a4 = next(c for c in report.checks if c.check == "A4")
        assert a4.outcome == "FAIL"
        assert a4.extended is False
        assert report.seed_extension_used is False
        profile = report.ego_robustness_profile
        assert profile is not None
        assert profile["imp_b"]["success_ci_low"] < 0.75 < profile["imp_b"]["success_ci_high"]

    def test_wrapped_a4_brittle_member_is_named(self, tmp_path: Path) -> None:
        report, _ = self._run_wrapped(
            tmp_path,
            member_success={"imp_a": lambda _s, _e: True, "imp_b": lambda _s, _e: False},
        )
        assert report.verdict == "UNINSTRUMENTABLE"
        a4 = next(c for c in report.checks if c.check == "A4")
        assert a4.outcome == "FAIL"
        assert "brittle partners" in a4.notes
        assert "imp_b" in a4.notes

    def test_wrapped_a4_per_partner_stress_over_limit_fails(self, tmp_path: Path) -> None:
        """The task's stress bar is held per partner on the wrapped path too."""
        report, _ = self._run_wrapped(tmp_path, member_force={"imp_b": 150.0})
        assert report.verdict == "UNINSTRUMENTABLE"
        a4 = next(c for c in report.checks if c.check == "A4")
        assert a4.outcome == "FAIL"
        profile = report.ego_robustness_profile
        assert profile is not None
        assert profile["imp_b"]["stress_max"] == 150.0

    def test_wrapped_a1_stress_over_limit_fails_not_solvable(self, tmp_path: Path) -> None:
        """The committed stress bar holds on the wrapped A1 path too."""
        report, _ = self._run_wrapped(tmp_path, a1_force=150.0)
        assert report.verdict == "NOT_SOLVABLE"
        a1 = next(c for c in report.checks if c.check == "A1")
        assert a1.outcome == "FAIL"
        assert a1.statistics["stress_max"] == 150.0

    def test_wrapped_a1_pass_without_stress_channel_is_loud(self, tmp_path: Path) -> None:
        """A wrapped A1 PASS cannot be granted without successful-episode stress evidence."""
        with pytest.raises(AdmissionError, match="stress statistics"):
            self._run_wrapped(tmp_path, a1_force=None)

    def test_wrapped_a2_straddle_is_final_fail(self, tmp_path: Path) -> None:
        """A wrapped A2 straddle is FAIL (no extension) -> the CONTROL demotion."""
        report, _ = self._run_wrapped(tmp_path, a2_success=lambda s, _e: s == 0)
        assert report.verdict == "CONTROL"
        a2 = next(c for c in report.checks if c.check == "A2")
        assert a2.outcome == "FAIL"
        assert a2.extended is False
        assert report.seed_extension_used is False

    def test_wrapped_a4_pass_without_stress_channel_is_loud(self, tmp_path: Path) -> None:
        """A wrapped A4 PASS cannot be granted from a profile with no stress channel."""
        with pytest.raises(AdmissionError, match="A4 stress_limit is committed"):
            self._run_wrapped(tmp_path, member_force=dict.fromkeys(self._MEMBERS))

    def test_extractor_refuses_unpinned_episode_files(self, tmp_path: Path) -> None:
        """Only SHA-pinned files can contribute evidence (I8)."""
        from chamber.benchmarks.admission_cells import extract_bundle_success_summary

        repo = _init_repo(tmp_path)
        files = _write_bundle_episodes(
            repo,
            "spikes/wrapped/a1_reference",
            {0: [_episode(0, 0, success=True, force_peak=40.0)]},
        )
        spec = WrappedEvidenceSpec(
            archive="spikes/wrapped",
            files=files,
            extractor="bundle_success_summary",
            params={"bundle_dir": "spikes/wrapped/other_cell"},
        )
        with pytest.raises(AdmissionError, match="SHA-pinned"):
            extract_bundle_success_summary(repo_path=repo, spec=spec)

    def test_paired_delta_requires_both_bundle_dirs(self, tmp_path: Path) -> None:
        from chamber.benchmarks.admission_cells import extract_bundle_paired_delta

        repo = _init_repo(tmp_path)
        files = _write_bundle_episodes(
            repo,
            "spikes/wrapped/a1_reference",
            {0: [_episode(0, 0, success=True, force_peak=40.0)]},
        )
        spec = WrappedEvidenceSpec(
            archive="spikes/wrapped",
            files=files,
            extractor="bundle_paired_delta",
            params={"reference_dir": "spikes/wrapped/a1_reference"},
        )
        with pytest.raises(AdmissionError, match="blind_dir"):
            extract_bundle_paired_delta(repo_path=repo, spec=spec)

    def test_tampered_wrapped_a4_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(AdmissionError, match="SHA-256 mismatch"):
            self._run_wrapped(tmp_path, tamper_a4=True)

    def test_wrapped_a4_member_pin_mismatch_is_loud(self, tmp_path: Path) -> None:
        with pytest.raises(AdmissionError, match="pins"):
            self._run_wrapped(tmp_path, members_pin=["imp_a", "imp_b", "imp_c"])

    def test_profile_extractor_on_scalar_check_is_loud(self, tmp_path: Path) -> None:
        with pytest.raises(AdmissionError, match="nested per-member profile"):
            self._run_wrapped(tmp_path, a1_extractor="bundle_ego_robustness_profile")

    def test_scalar_extractor_on_a4_is_loud(self, tmp_path: Path) -> None:
        with pytest.raises(AdmissionError, match="per-member profile extractor"):
            self._run_wrapped(tmp_path, a4_extractor="bundle_success_summary")


class TestCommittedCocarryA4Evidence:
    """The retrospective co-carry A4 wrap over the committed b-aht bundle (ADR-027 A4; I8).

    CI-runnable real-data tests: the evidence files are committed to the
    repo, the prereg pins their SHA-256, and the extractor re-derives
    the per-partner profile through the same seed-cluster bootstrap the
    measured path uses.
    """

    _PREREG = (
        _REPO_ROOT / "spikes" / "preregistration" / "admission" / "cocarry_admission_a4_rev2.yaml"
    )
    _MEMBERS = frozenset(
        {
            "imp_stiff_low",
            "imp_stiff_high",
            "imp_damp_low",
            "imp_damp_high",
            "imp_lag_bounded",
            "imp_blend_b",
            "imp_blend_c",
        }
    )

    def _admission_block(self) -> dict[str, object]:
        payload = yaml.safe_load(self._PREREG.read_text(encoding="utf-8"))
        block = payload["parameters"]["admission"]
        assert isinstance(block, dict)
        return block

    def test_prereg_loads_as_fully_wrapped_spec(self) -> None:
        doc = load_prereg_document(self._PREREG)
        spec = admission_spec_from_prereg(doc)
        assert isinstance(spec.a1, WrappedEvidenceSpec)
        assert isinstance(spec.a2, WrappedEvidenceSpec)
        assert isinstance(spec.a3, WrappedEvidenceSpec)
        assert isinstance(spec.a4, WrappedEvidenceSpec)
        # The admitted set's capability floor C_min, never the aggregate
        # tau_solv (ADR-027 §Revision history 2026-07-15 disambiguation).
        assert spec.c_min_ego == 0.75
        assert spec.tau_solv == 0.95

    def test_committed_baht_profile_matches_the_committed_data(self) -> None:
        """The extracted per-partner means reproduce the committed b-aht episodes."""
        from chamber.benchmarks.admission_cells import extract_bundle_ego_robustness_profile

        block = self._admission_block()
        spec = WrappedEvidenceSpec.model_validate(block["a4"])
        for rel, expected in spec.files.items():
            assert sha256_file(_REPO_ROOT / rel) == expected, f"committed pin drifted: {rel}"
        profile = extract_bundle_ego_robustness_profile(repo_path=_REPO_ROOT, spec=spec)
        assert set(profile) == self._MEMBERS
        assert all(stats["n_episodes"] == 250.0 for stats in profile.values())
        assert profile["imp_lag_bounded"]["success_mean"] == pytest.approx(0.884)
        assert profile["imp_damp_low"]["success_mean"] == 1.0
        stress_limit = block["stress_limit"]
        assert isinstance(stress_limit, float)
        assert all(stats["stress_max"] <= stress_limit for stats in profile.values())
        # At the committed floor (c_min_ego = C_min = 0.75, the admitted
        # set's capability floor — ADR-027 §Revision history 2026-07-15)
        # every admitted member clears: the weakest CI-lower is
        # imp_blend_c at ~0.833 >= 0.75 -> A4 PASS.
        c_min_ego = block["c_min_ego"]
        assert isinstance(c_min_ego, float)
        assert c_min_ego == 0.75
        cis = {m: (s["success_ci_low"], s["success_ci_high"]) for m, s in profile.items()}
        assert a4_outcome(cis, c_min_ego) == "PASS"
        assert min(ci_low for ci_low, _ in cis.values()) == pytest.approx(0.8331, abs=1e-3)

    def test_committed_a3_wrap_reproduces_the_committed_gap(self) -> None:
        """The paired-delta extractor byte-reproduces the 2026-07-05 committed A3 CI."""
        from chamber.benchmarks.admission_cells import extract_bundle_paired_delta

        block = self._admission_block()
        spec = WrappedEvidenceSpec.model_validate(block["a3"])
        stats = extract_bundle_paired_delta(repo_path=_REPO_ROOT, spec=spec)
        committed = load_admission_report(
            _REPO_ROOT
            / "spikes"
            / "results"
            / "admission"
            / "cocarry-2026-07-05"
            / "admission_report.json"
        )
        committed_a3 = next(c for c in committed.checks if c.check == "A3")
        for key in ("n_pairs", "delta_iqm", "delta_mean", "delta_ci_low", "delta_ci_high"):
            assert stats[key] == committed_a3.statistics[key], key


class TestSpecLoading:
    """Spec-from-prereg loading (ADR-028 §Decision 2 document form)."""

    def test_missing_admission_block_is_loud(self, tmp_path: Path) -> None:
        del tmp_path
        from chamber.evaluation.prereg import PreregDocument

        doc = PreregDocument(task_id="t", git_tag="tag", parameters={}, decision_rules="none")
        with pytest.raises(AdmissionError, match="admission"):
            admission_spec_from_prereg(doc)

    def test_cell_and_wrapped_union_discriminates(self) -> None:
        payload = _prereg_payload()
        admission = payload["parameters"]["admission"]  # type: ignore[index]
        spec = AdmissionSpec.model_validate(
            {**admission, "task_id": "t", "git_tag": "tag"}  # type: ignore[dict-item]
        )
        assert isinstance(spec.a1, AdmissionCellSpec)
        wrapped = {
            "archive": "a",
            "files": {},
            "extractor": "handover_gate0_limb1",
        }
        spec2 = AdmissionSpec.model_validate(
            {**admission, "a1": wrapped, "task_id": "t", "git_tag": "tag"}  # type: ignore[dict-item]
        )
        assert isinstance(spec2.a1, WrappedEvidenceSpec)

    def test_report_schema_version_is_pinned(self, tmp_path: Path) -> None:
        bogus = tmp_path / "r.json"
        bogus.write_text(json.dumps({"schema_version": 99}), encoding="utf-8")
        with pytest.raises(ValueError, match="schema_version"):
            load_admission_report(bogus)

    def test_a4_fields_travel_together(self) -> None:
        """c_min_ego and the a4 cell are committed together or not at all (ADR-027 A4)."""
        payload = _prereg_payload()
        admission = payload["parameters"]["admission"]  # type: ignore[index]
        base = {**admission, "task_id": "t", "git_tag": "tag"}  # type: ignore[dict-item]
        with pytest.raises(ValidationError, match="c_min_ego"):
            AdmissionSpec.model_validate({**base, "c_min_ego": 0.75})
        with pytest.raises(ValidationError, match="c_min_ego"):
            AdmissionSpec.model_validate({**base, "a4": dict(_A4_CELL)})
        spec = AdmissionSpec.model_validate({**base, "c_min_ego": 0.75, "a4": dict(_A4_CELL)})
        assert spec.c_min_ego == 0.75
        assert isinstance(spec.a4, AdmissionCellSpec)


class TestReportSchemaCompatibility:
    """A4 additions are optional: schema_version stays 1 (ADR-027 §Open questions; I9)."""

    def test_profile_round_trips_at_schema_1(self, tmp_path: Path) -> None:
        report = AdmissionReport(
            task_id="t",
            task_version=1,
            prereg_git_tag="tag",
            prereg_blob_sha="blob",
            git_sha="sha",
            dirty=False,
            date_stamp="2026-07-15",
            checks=[],
            verdict="UNINSTRUMENTABLE",
            seed_extension_used=False,
            binding_evidence={},
            ego_robustness_profile={
                "imp_matched": {"success_ci_low": 1.0, "success_ci_high": 1.0},
                "selfish_effort": {"success_ci_low": 0.0, "success_ci_high": 0.0},
            },
        )
        path = tmp_path / "admission_report.json"
        path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        loaded = load_admission_report(path)
        assert loaded == report
        assert loaded.schema_version == 1

    @pytest.mark.parametrize(
        "archive",
        ["cocarry-2026-07-05", "handover_place-2026-07-05", "stage1_pickplace_as-2026-07-05"],
    )
    def test_committed_admission_reports_still_load(self, archive: str) -> None:
        """The three pre-A4 committed archives load unchanged through the exact-match gate."""
        path = _REPO_ROOT / "spikes" / "results" / "admission" / archive / "admission_report.json"
        report = load_admission_report(path)
        assert report.schema_version == 1
        assert report.ego_robustness_profile is None
        assert [c.check for c in report.checks][:1] == ["A1"]
