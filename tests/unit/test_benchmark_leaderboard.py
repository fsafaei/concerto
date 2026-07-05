# SPDX-License-Identifier: Apache-2.0
"""Tier-1 pins for the per-task benchmark leaderboard (ADR-027 §Reporting rules).

Verified-bundles-only admission (an unverifiable entry refuses the
whole render), the ADR-011 row mapping + mandated labels, the
recomputed row statistics, and the README marker block. Bundles are
built with the same helpers ``chamber-eval run`` uses, so what passes
here is exactly what ``chamber-eval verify`` admits (mirrors
``tests/unit/test_result_bundle.py``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from chamber.evaluation.bundles import compute_summary, write_bundle_dir
from chamber.evaluation.leaderboard import (
    LEADERBOARD_MANIFEST,
    README_LEADERBOARD_BEGIN,
    README_LEADERBOARD_END,
    LeaderboardInputError,
    build_rows,
    render_readme_leaderboard,
    render_task_leaderboard,
    row_id_for_policy,
)
from chamber.evaluation.results import (
    EpisodeResult,
    PlatformFingerprint,
    ResultBundle,
    SeedSchedule,
)
from chamber.partners.api import PartnerSpec

if TYPE_CHECKING:
    from pathlib import Path

_PARTNER_SPEC = PartnerSpec(
    class_name="scripted_heuristic",
    seed=0,
    checkpoint_step=None,
    weights_uri=None,
    extra={"uid": "partner", "target_xy": "0.0,0.0", "action_dim": "2"},
)

_MEMBERS = ("imp_stiff_low", "imp_blend_b")


def _episodes(seed: int) -> list[EpisodeResult]:
    records = []
    for member_index, member in enumerate(_MEMBERS):
        for episode in range(4):
            idx = member_index * 4 + episode
            records.append(
                EpisodeResult(
                    seed=seed,
                    episode_idx=idx,
                    initial_state_seed=seed * 1000 + episode,
                    # imp_stiff_low mostly succeeds; imp_blend_b mostly fails —
                    # gives the per-partner range a real spread.
                    success=(episode < 3) if member_index == 0 else (episode == 0),
                    force_peak=60.0 + 10.0 * idx,
                    metadata={"member": member, "max_tilt_deg": 5.0, "n_steps": 100.0},
                )
            )
    return records


def _write_row_bundle(out_dir: Path, *, policy_id: str) -> None:
    episodes_by_seed = {0: _episodes(0), 1: _episodes(1)}
    all_eps = [ep for eps in episodes_by_seed.values() for ep in eps]
    bundle = ResultBundle(
        task_id="cocarry",
        task_version=1,
        policy_id=policy_id,
        partner_set_id="cocarry_partners@v1",
        partner_hashes={"scripted_heuristic": _PARTNER_SPEC.partner_id},
        git_sha="f" * 40,
        dirty=False,
        package_version="0.0.0-test",
        seed_schedule=SeedSchedule(
            root_seed=0, seeds=[0, 1], episodes_per_seed=8, substream_labels=["a.{seed}"]
        ),
        repro_command="chamber-eval run --task cocarry",
        platform=PlatformFingerprint(os="t", python="3", numpy="0", torch=None, device="cpu"),
        manifest={},
        summary=compute_summary(all_eps),
    )
    write_bundle_dir(
        out_dir,
        bundle_without_manifest=bundle,
        episodes_by_seed=episodes_by_seed,
        partner_specs=[
            {
                "name": "scripted_heuristic",
                "class_name": "scripted_heuristic",
                "seed": 0,
                "checkpoint_step": None,
                "weights_uri": None,
                "extra": dict(_PARTNER_SPEC.extra),
            }
        ],
        repro_command="chamber-eval run --task cocarry",
    )


def _task_dir(tmp_path: Path, *, policies: dict[str, str]) -> Path:
    task_dir = tmp_path / "spikes" / "results" / "benchmark" / "cocarry-v1"
    task_dir.mkdir(parents=True)
    lines = ["# committed leaderboard bundles"]
    for name, policy in policies.items():
        bundle_dir = task_dir / name
        _write_row_bundle(bundle_dir, policy_id=policy)
        lines.append(str(bundle_dir.relative_to(tmp_path)))
    (task_dir / LEADERBOARD_MANIFEST).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return task_dir


class TestRowMapping:
    """The ADR-011 policy → row mapping + mandated labels."""

    def test_row_ids(self) -> None:
        assert row_id_for_policy("random") == "B-RND"
        assert row_id_for_policy("static") == "B-STAT"
        assert row_id_for_policy("ref_script_cocarry_impedance") == "REF-SCRIPT"
        assert row_id_for_policy("happo_manifest:manifests/b_aht.json") == "B-AHT"
        assert row_id_for_policy("happo_blind_manifest:m.json") == "B-BLIND"
        assert row_id_for_policy("joint_ego:local://artifacts/pair.pt") == "B-JOINT"
        # Unknown policies keep their own id — no silent row adoption.
        assert row_id_for_policy("mystery") == "mystery"


class TestRenderFromVerifiedBundles:
    """Verified-only admission + the recomputed row table (ADR-027; ADR-028)."""

    def test_renders_rows_with_labels_and_range(self, tmp_path: Path) -> None:
        task_dir = _task_dir(
            tmp_path,
            policies={
                "ref-script-2026-07-05": "ref_script_cocarry_impedance",
                "b-rnd-2026-07-05": "random",
            },
        )
        table = render_task_leaderboard(task_dir, repo_path=tmp_path)
        assert "| REF-SCRIPT | *oracle reference* |" in table
        assert "| B-RND |" in table
        # Per-partner breakdown extremes name the members.
        assert "imp_blend_b" in table
        assert "imp_stiff_low" in table
        # REF-SCRIPT is ordered before B-RND (ROW_ORDER).
        assert table.index("REF-SCRIPT") < table.index("B-RND")

    def test_joint_row_pools_per_seed_bundles(self, tmp_path: Path) -> None:
        task_dir = _task_dir(
            tmp_path,
            policies={
                "b-joint-seed0": "joint_ego:local://artifacts/pair_s0.pt",
                "b-joint-seed1": "joint_ego:local://artifacts/pair_s1.pt",
            },
        )
        rows = build_rows(task_dir, repo_path=tmp_path)
        assert len(rows) == 1
        row = rows[0]
        assert row.row_id == "B-JOINT"
        assert row.label == "non-AHT upper anchor"
        assert len(row.bundle_paths) == 2
        assert row.n_episodes == 32  # 2 bundles x 2 seeds x 8 episodes

    def test_tampered_bundle_refuses_render(self, tmp_path: Path) -> None:
        task_dir = _task_dir(tmp_path, policies={"b-rnd": "random"})
        episodes_file = task_dir / "b-rnd" / "episodes_seed0.jsonl"
        episodes_file.write_text(
            episodes_file.read_text(encoding="utf-8").replace("false", "true "),
            encoding="utf-8",
        )
        with pytest.raises(LeaderboardInputError, match="failed verification"):
            render_task_leaderboard(task_dir, repo_path=tmp_path)

    def test_missing_manifest_is_loud(self, tmp_path: Path) -> None:
        task_dir = tmp_path / "empty-task"
        task_dir.mkdir()
        with pytest.raises(LeaderboardInputError, match=LEADERBOARD_MANIFEST):
            render_task_leaderboard(task_dir, repo_path=tmp_path)


class TestReadmeBlock:
    """The README marker block (the CB-02 drift-check convention)."""

    def test_placeholder_when_no_manifests(self, tmp_path: Path) -> None:
        block = render_readme_leaderboard(repo_path=tmp_path)
        assert block.startswith(README_LEADERBOARD_BEGIN)
        assert block.endswith(README_LEADERBOARD_END)
        assert "No verified leaderboard bundles committed yet" in block

    def test_renders_task_sections(self, tmp_path: Path) -> None:
        _task_dir(tmp_path, policies={"b-stat": "static"})
        block = render_readme_leaderboard(repo_path=tmp_path)
        assert "#### cocarry-v1" in block
        assert "| B-STAT |" in block
