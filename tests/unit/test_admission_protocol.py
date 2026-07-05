# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the executable admission protocol (ADR-027 §Admission protocol).

Covers the pre-committed threshold rules, the spec-from-prereg loader,
and the full :func:`chamber.evaluation.admission.run_admission` flow
against fake cell runners inside a throwaway git repo — every measured
cell must come out as a ``chamber-eval verify``-passing v3 bundle
(ADR-028 §Decision 3), the verdict table must match ADR-027 (CONTROL
demotion, NOT_SOLVABLE short-circuit, INDETERMINATE + the single
pre-committed seed extension), and wrapped evidence must be
SHA-verified, never trusted (I8).
"""

from __future__ import annotations

import json
import subprocess
from typing import TYPE_CHECKING

import pytest
import yaml

from chamber.evaluation.admission import (
    AdmissionCellSpec,
    AdmissionError,
    AdmissionSpec,
    CellRun,
    WrappedEvidenceSpec,
    a1_outcome,
    a2_outcome,
    a3_outcome,
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
    from pathlib import Path

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

    def test_verdict_table(self) -> None:
        assert overall_verdict("PASS", "PASS", "PASS") == "ADMITTED"
        assert overall_verdict("FAIL", None, None) == "NOT_SOLVABLE"
        assert overall_verdict("PASS", "FAIL", "PASS") == "CONTROL"
        assert overall_verdict("PASS", "PASS", "FAIL") == "CONTROL"
        assert overall_verdict("PASS", "INDETERMINATE", "PASS") == "INDETERMINATE"
        assert overall_verdict("INDETERMINATE", "PASS", "PASS") == "INDETERMINATE"

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
) -> EpisodeResult:
    return EpisodeResult(
        seed=seed,
        episode_idx=idx,
        initial_state_seed=idx,
        success=success,
        force_peak=force_peak,
        metadata={"condition": "fake"},
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
) -> Callable[[str], Callable[..., CellRun]]:
    """Fake cell-runner resolver: success per (seed, episode) by cell id."""

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
            fn = success_by_cell[cell.cell_id.split("-")[0]]
            material, hashes = _fake_partner_material()
            return CellRun(
                episodes_by_seed={
                    s: [
                        _episode(s, e, success=fn(s, e), force_peak=force_peak)
                        for e in range(episodes_per_seed)
                    ]
                    for s in seeds
                },
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
