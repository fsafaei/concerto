# SPDX-License-Identifier: Apache-2.0
"""Tier-1 tests for ``chamber-eval run --partner-set`` (ADR-009 as amended; ADR-028).

Covers: unknown-set / task-mismatch / ``--include-private``-without-
withheld-parameters refusals (exit 2, nothing written), and the happy
path over a temporarily-registered CPU set on ``mpe_cooperative_push``
— the bundle records the set slug + per-member custody hashes, runs the
full grid per member, and passes ``chamber-eval verify`` end to end.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import chamber.partners  # noqa: F401 - registers the v1 sets
from chamber.cli import _eval_run as eval_run_module
from chamber.cli import eval as eval_cli
from chamber.evaluation.bundles import GitProvenance, verify_bundle_dir
from chamber.partners import sets as sets_module
from chamber.partners.sets import (
    PRIVATE_PARAMS_ENV,
    ParamRange,
    PartnerMemberSpec,
    PartnerSetSpec,
    compute_split,
    derive_member_params,
    params_sha256,
)

_TEST_ROOT_SEED = 90210


def _mpe_set(*, with_private: bool = False) -> PartnerSetSpec:
    """A CPU-runnable heuristic set on the Tier-0 diagnostic task."""
    box = {
        "target_x": ParamRange(lo=-0.5, hi=0.5),
        "target_y": ParamRange(lo=-0.5, hi=0.5),
    }
    n = 4 if with_private else 3  # N=4 → 3 public + 1 private; N=3 → all public
    drawn = {
        f"reach_{i}": derive_member_params(
            "mpe_test_partners", f"reach_{i}", box, root_seed=_TEST_ROOT_SEED
        )
        for i in range(n)
    }
    prototypes = {
        name: PartnerMemberSpec(
            member_name=name,
            registry_class="scripted_heuristic",
            role="partner",
            split="public",
            param_box=box,
            params=params,
            params_sha256=params_sha256(params),
        )
        for name, params in drawn.items()
    }
    split = compute_split([m.partner_id for m in prototypes.values()])
    members = [
        PartnerMemberSpec(
            member_name=m.member_name,
            registry_class=m.registry_class,
            role=m.role,
            split=split[m.partner_id],
            param_box=m.param_box,
            params=m.params if split[m.partner_id] == "public" else None,
            params_sha256=m.params_sha256,
        )
        for m in prototypes.values()
    ]
    return PartnerSetSpec(
        set_id="mpe_test_partners",
        version=1,
        task_id="mpe_cooperative_push",
        task_version=1,
        floor=0.0,
        probe_seeds=[0],
        probe_episodes_per_seed=1,
        members=members,
    )


@pytest.fixture
def register_mpe_set(monkeypatch: pytest.MonkeyPatch):
    """Temporarily register the CPU test set without polluting the global registry."""

    def _register(*, with_private: bool = False) -> PartnerSetSpec:
        spec = _mpe_set(with_private=with_private)
        registry = dict(sets_module._SET_REGISTRY)
        registry[spec.set_id] = {spec.version: spec}
        monkeypatch.setattr(sets_module, "_SET_REGISTRY", registry)
        return spec

    return _register


@pytest.fixture
def clean_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        eval_run_module, "git_provenance", lambda _repo: GitProvenance(sha="a" * 40, dirty=False)
    )


def _run_args(out: Path, *extra: str) -> list[str]:
    return [
        "run",
        "--task",
        "mpe_cooperative_push",
        "--policy",
        "random",
        "--partner-set",
        "mpe_test_partners@v1",
        "--seeds",
        "2",
        "--episodes",
        "2",
        "--out",
        str(out),
        *extra,
    ]


class TestPartnerSetRefusals:
    def test_unknown_set_exits_2(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        rc = eval_cli.main(
            [
                "run",
                "--task",
                "mpe_cooperative_push",
                "--policy",
                "random",
                "--partner-set",
                "nope",
                "--seeds",
                "1",
                "--episodes",
                "1",
                "--out",
                str(tmp_path / "b"),
            ]
        )
        assert rc == 2
        assert "unknown partner-set id" in capsys.readouterr().err

    def test_task_mismatch_exits_2(
        self, tmp_path: Path, register_mpe_set, capsys: pytest.CaptureFixture[str]
    ) -> None:
        register_mpe_set()
        rc = eval_cli.main(
            [
                "run",
                "--task",
                "cocarry",
                "--policy",
                "random",
                "--partner-set",
                "mpe_test_partners",
                "--seeds",
                "1",
                "--episodes",
                "1",
                "--out",
                str(tmp_path / "b"),
            ]
        )
        assert rc == 2
        assert "sets are per-task" in capsys.readouterr().err

    def test_include_private_refused_without_withheld_params(
        self,
        tmp_path: Path,
        register_mpe_set,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        register_mpe_set(with_private=True)
        monkeypatch.delenv(PRIVATE_PARAMS_ENV, raising=False)
        rc = eval_cli.main(_run_args(tmp_path / "b", "--include-private"))
        assert rc == 2
        assert "withheld" in capsys.readouterr().err
        assert not (tmp_path / "b" / "bundle.json").exists()

    def test_include_private_without_partner_set_exits_2(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rc = eval_cli.main(
            [
                "run",
                "--task",
                "mpe_cooperative_push",
                "--policy",
                "random",
                "--partner",
                "scripted_heuristic",
                "--include-private",
                "--seeds",
                "1",
                "--episodes",
                "1",
                "--out",
                str(tmp_path / "b"),
            ]
        )
        assert rc == 2
        assert "only applies to --partner-set" in capsys.readouterr().err


class TestPartnerSetRun:
    def test_public_grid_bundle_verifies(
        self, tmp_path: Path, register_mpe_set, clean_tree, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        spec = register_mpe_set()
        monkeypatch.delenv(PRIVATE_PARAMS_ENV, raising=False)
        out = tmp_path / "bundle"
        rc = eval_cli.main(_run_args(out))
        assert rc == 0
        bundle = json.loads((out / "bundle.json").read_text(encoding="utf-8"))
        assert bundle["partner_set_id"] == "mpe_test_partners@v1"
        assert set(bundle["partner_hashes"]) == {m.member_name for m in spec.members}
        # 2 seeds x 2 episodes x 3 members.
        assert bundle["summary"]["n_episodes"] == 12
        assert bundle["seed_schedule"]["episodes_per_seed"] == 6
        rows = verify_bundle_dir(out, repo_path=Path.cwd())
        failing = [r for r in rows if not r.ok]
        assert not failing, failing

    def test_include_private_runs_the_full_roster(
        self, tmp_path: Path, register_mpe_set, clean_tree, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        spec = register_mpe_set(with_private=True)
        monkeypatch.setenv(PRIVATE_PARAMS_ENV, str(_TEST_ROOT_SEED))
        out = tmp_path / "bundle"
        rc = eval_cli.main(_run_args(out, "--include-private"))
        assert rc == 0
        bundle = json.loads((out / "bundle.json").read_text(encoding="utf-8"))
        assert len(bundle["partner_hashes"]) == 4
        partners = json.loads((out / "partners.json").read_text(encoding="utf-8"))
        by_name = {entry["name"]: entry for entry in partners}
        (private_member,) = [m for m in spec.members if m.split == "private"]
        # Private redaction: the withheld marker, never the drawn values.
        assert by_name[private_member.member_name]["extra"] == {
            "withheld_params_sha256": private_member.params_sha256
        }
        rows = verify_bundle_dir(out, repo_path=Path.cwd())
        assert all(r.ok for r in rows), [r for r in rows if not r.ok]

    def test_members_see_identical_initial_states(
        self, tmp_path: Path, register_mpe_set, clean_tree
    ) -> None:
        register_mpe_set()
        out = tmp_path / "bundle"
        assert eval_cli.main(_run_args(out)) == 0
        lines = (out / "episodes_seed0.jsonl").read_text(encoding="utf-8").splitlines()
        episodes = [json.loads(line) for line in lines]
        by_member: dict[str, list[int]] = {}
        for ep in episodes:
            by_member.setdefault(ep["metadata"]["member"], []).append(ep["initial_state_seed"])
        state_seeds = {tuple(sorted(v)) for v in by_member.values()}
        assert len(state_seeds) == 1  # matched draws across members
