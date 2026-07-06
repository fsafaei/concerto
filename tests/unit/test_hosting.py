"""Tier-1 tests for the hosting packager (ADR-028; ADR-027 §Versioning).

Pure-Python: a miniature repository tree is synthesised per test, with
fake checkpoint payloads for every checkpoint URI the (real) partner
registry references, so the tests exercise the real planning logic
without the 300 MB local artifact store.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from chamber.evaluation import hosting
from chamber.evaluation.hosting import (
    HOSTING_ARTIFACTS,
    HostingSourceError,
    PackagePlan,
    build_all,
    build_package,
    plan_leaderboard_bundles,
    plan_partner_sets,
    plan_reference_trajectories,
)
from chamber.partners.sets import get_partner_set, list_partner_sets, parse_set_slug

REPO_ROOT = Path(__file__).resolve().parents[2]

_BUNDLE_FILES = {
    "bundle.json": '{"schema_version": 3}\n',
    "episodes_seed0.jsonl": '{"episode": 0}\n',
    "REPRO.txt": (
        "chamber-eval run --task cocarry --policy joint_ego:local://artifacts/fake_pair.pt "
        "--partner frozen_cocarry_joint --partner-weights local://artifacts/fake_pair.pt "
        "--seeds 0, --episodes 1 --out spikes/results/benchmark/cocarry-v1/row-a\n"
    ),
    "SHA256SUMS.txt": "0" * 64 + "  bundle.json\n",
}


def _registry_checkpoint_uris() -> set[str]:
    uris: set[str] = set()
    for slug in list_partner_sets():
        set_id, version = parse_set_slug(slug)
        spec = get_partner_set(set_id, version)
        uris.update(
            member.checkpoint_uri
            for member in spec.public_members
            if member.checkpoint_uri is not None
        )
    return uris


@pytest.fixture
def mini_repo(tmp_path: Path) -> Path:
    """A synthetic repo tree covering every source the planners read."""
    repo = tmp_path / "repo"
    # Partner cards + fingerprints.
    cards = repo / "docs/reference/partners"
    (cards / "some_set-v1").mkdir(parents=True)
    (cards / "index.md").write_text("# partners\n", encoding="utf-8")
    (cards / "some_set-v1" / "member.md").write_text("# member\n", encoding="utf-8")
    fingerprints = repo / "spikes/results/partner-fingerprints"
    for slug in list_partner_sets():
        set_dir = fingerprints / slug.replace("@", "-")
        set_dir.mkdir(parents=True)
        (set_dir / "fingerprints.json").write_text("{}\n", encoding="utf-8")
        (set_dir / "SHA256SUMS.txt").write_text("x\n", encoding="utf-8")
    # Fake checkpoint payloads for every URI the real registry references,
    # plus the one referenced by the fake bundle below.
    artifacts = repo / "artifacts/artifacts"
    artifacts.mkdir(parents=True)
    for uri in _registry_checkpoint_uris() | {"local://artifacts/fake_pair.pt"}:
        name = uri.rsplit("/", 1)[-1]
        (artifacts / name).write_bytes(b"payload:" + name.encode())
        (artifacts / f"{name}.json").write_text('{"sidecar": true}\n', encoding="utf-8")
    # One leaderboard task with one bundle + selection artifacts.
    task_dir = repo / "spikes/results/benchmark/cocarry-v1"
    bundle = task_dir / "row-a"
    bundle.mkdir(parents=True)
    for name, content in _BUNDLE_FILES.items():
        (bundle / name).write_text(content, encoding="utf-8")
    (task_dir / "LEADERBOARD_BUNDLES.txt").write_text(
        "# manifest\nspikes/results/benchmark/cocarry-v1/row-a\n", encoding="utf-8"
    )
    (task_dir / "CAMPAIGN_REPORT.md").write_text("# report\n", encoding="utf-8")
    selection = task_dir / "selection"
    selection.mkdir()
    (selection / "b-aht_selected_manifest.json").write_text(
        json.dumps({"0": "local://artifacts/fake_pair.pt"}), encoding="utf-8"
    )
    # Admission evidence.
    admission = repo / "spikes/results/admission/cocarry-2026-07-05"
    admission.mkdir(parents=True)
    (admission / "admission_report.json").write_text('{"verdict": "ADMITTED"}\n', encoding="utf-8")
    # One power-pilot bundle (excluded from the leaderboard manifest).
    pilot = task_dir / "power-pilot-ref-script-2026-07-05"
    pilot.mkdir()
    (pilot / "bundle.json").write_text('{"run_purpose": "power"}\n', encoding="utf-8")
    # Committed card + croissant sources for all three artifacts.
    for name in HOSTING_ARTIFACTS:
        source = repo / "release/hosting" / name
        source.mkdir(parents=True)
        (source / "README.md").write_text(f"# {name}\n", encoding="utf-8")
        (source / "croissant.jsonld").write_text('{"@type": "sc:Dataset"}\n', encoding="utf-8")
    return repo


class TestPlanPartnerSets:
    def test_rosters_cards_fingerprints_checkpoints(self, mini_repo: Path) -> None:
        plan = plan_partner_sets(mini_repo)
        assert plan.name == "chamber-bench-partner-sets"
        assert "cards/index.md" in plan.copies
        assert "cards/some_set-v1/member.md" in plan.copies
        for slug in list_partner_sets():
            assert f"sets/{slug.replace('@', '-')}.json" in plan.generated
            assert f"fingerprints/{slug.replace('@', '-')}.json" in plan.copies
        for uri in _registry_checkpoint_uris():
            name = uri.rsplit("/", 1)[-1]
            assert f"checkpoints/{name}" in plan.copies
            assert f"checkpoints/{name}.json" in plan.copies

    def test_private_members_have_no_params(self, mini_repo: Path) -> None:
        plan = plan_partner_sets(mini_repo)
        for rel, content in plan.generated.items():
            assert rel.startswith("sets/")
            roster = json.loads(content)
            for member in roster["members"]:
                if member["split"] == "private":
                    assert member["params"] is None

    def test_private_param_literal_refuses_packaging(
        self, mini_repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        slug = list_partner_sets()[0]
        set_id, version = parse_set_slug(slug)
        spec = get_partner_set(set_id, version)
        leaky = spec.model_copy(deep=True)
        object.__setattr__(leaky.members[0], "split", "private")
        object.__setattr__(leaky.members[0], "params", {"k": "1.0"})
        monkeypatch.setattr(hosting, "get_partner_set", lambda *_a, **_k: leaky)
        monkeypatch.setattr(hosting, "list_partner_sets", lambda: [slug])
        with pytest.raises(HostingSourceError, match=r"parameter literals"):
            plan_partner_sets(mini_repo)


class TestPlanLeaderboardBundles:
    def test_bundles_reports_selection_checkpoints(self, mini_repo: Path) -> None:
        plan = plan_leaderboard_bundles(mini_repo)
        prefix = "benchmark/cocarry-v1"
        assert f"{prefix}/LEADERBOARD_BUNDLES.txt" in plan.copies
        assert f"{prefix}/CAMPAIGN_REPORT.md" in plan.copies
        assert f"{prefix}/selection/b-aht_selected_manifest.json" in plan.copies
        for name in _BUNDLE_FILES:
            assert f"{prefix}/row-a/{name}" in plan.copies
        assert "checkpoints/fake_pair.pt" in plan.copies
        assert "checkpoints/fake_pair.pt.json" in plan.copies
        assert "admission/cocarry-2026-07-05/admission_report.json" in plan.copies
        # Power pilots are reference evidence, not leaderboard evidence.
        assert not any("power-pilot" in rel for rel in plan.copies)

    def test_missing_admission_tree_fails(self, mini_repo: Path) -> None:
        import shutil

        shutil.rmtree(mini_repo / "spikes/results/admission")
        with pytest.raises(HostingSourceError, match="admission"):
            plan_leaderboard_bundles(mini_repo)

    def test_missing_checkpoint_is_listed(self, mini_repo: Path) -> None:
        (mini_repo / "artifacts/artifacts/fake_pair.pt").unlink()
        with pytest.raises(HostingSourceError, match=r"fake_pair\.pt"):
            plan_leaderboard_bundles(mini_repo)

    def test_listed_but_missing_bundle_fails(self, mini_repo: Path) -> None:
        manifest = mini_repo / "spikes/results/benchmark/cocarry-v1/LEADERBOARD_BUNDLES.txt"
        manifest.write_text("spikes/results/benchmark/cocarry-v1/ghost\n", encoding="utf-8")
        with pytest.raises(HostingSourceError, match="ghost"):
            plan_leaderboard_bundles(mini_repo)

    def test_no_manifests_fails(self, mini_repo: Path) -> None:
        (mini_repo / "spikes/results/benchmark/cocarry-v1/LEADERBOARD_BUNDLES.txt").unlink()
        with pytest.raises(HostingSourceError, match=r"no LEADERBOARD_BUNDLES\.txt"):
            plan_leaderboard_bundles(mini_repo)


class TestPlanReferenceTrajectories:
    def test_covers_fingerprints_and_power_pilots(self, mini_repo: Path) -> None:
        plan = plan_reference_trajectories(mini_repo)
        assert any(rel.startswith("partner-fingerprints/") for rel in plan.copies)
        assert (
            "power-pilots/cocarry-v1/power-pilot-ref-script-2026-07-05/bundle.json" in plan.copies
        )
        # Admission evidence ships with the leaderboard bundles instead.
        assert not any(rel.startswith("admission/") for rel in plan.copies)

    def test_missing_fingerprints_tree_fails(self, mini_repo: Path) -> None:
        import shutil

        shutil.rmtree(mini_repo / "spikes/results/partner-fingerprints")
        with pytest.raises(HostingSourceError, match="partner-fingerprints"):
            plan_reference_trajectories(mini_repo)

    def test_no_power_pilots_fails(self, mini_repo: Path) -> None:
        import shutil

        shutil.rmtree(
            mini_repo / "spikes/results/benchmark/cocarry-v1/power-pilot-ref-script-2026-07-05"
        )
        with pytest.raises(HostingSourceError, match="power-pilot"):
            plan_reference_trajectories(mini_repo)


class TestBuildPackage:
    def test_build_all_is_deterministic(self, mini_repo: Path, tmp_path: Path) -> None:
        first = build_all(mini_repo, tmp_path / "a", git_sha="deadbeef")
        second = build_all(mini_repo, tmp_path / "b", git_sha="deadbeef")
        assert [p.name for p in first] == list(HOSTING_ARTIFACTS)
        for pkg_a, pkg_b in zip(first, second, strict=True):
            for name in ("manifest.json", "SHA256SUMS.txt"):
                assert (pkg_a / name).read_bytes() == (pkg_b / name).read_bytes()

    def test_manifest_and_sums_cover_everything(self, mini_repo: Path, tmp_path: Path) -> None:
        package = build_all(mini_repo, tmp_path / "dist", git_sha="deadbeef")[2]
        manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["artifact"] == "chamber-bench-reference-trajectories"
        assert manifest["git_sha"] == "deadbeef"
        # Only the package-root manifest and digest list are excluded;
        # nested payload files that happen to be named SHA256SUMS.txt
        # (result bundles carry their own) are payload.
        on_disk = {
            p.relative_to(package).as_posix()
            for p in package.rglob("*")
            if p.is_file()
            and p.relative_to(package).as_posix() not in ("manifest.json", "SHA256SUMS.txt")
        }
        listed = {entry["path"] for entry in manifest["files"]}
        assert listed == on_disk
        assert any(rel.endswith("SHA256SUMS.txt") for rel in listed)  # nested bundle sums
        assert "README.md" in listed
        assert "croissant.jsonld" in listed
        for entry in manifest["files"]:
            digest = hashlib.sha256((package / entry["path"]).read_bytes()).hexdigest()
            assert digest == entry["sha256"]
        sums = {
            line.split("  ", 1)[1]
            for line in (package / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines()
        }
        assert sums == on_disk | {"manifest.json"}

    def test_refuses_nonempty_destination(self, mini_repo: Path, tmp_path: Path) -> None:
        dest = tmp_path / "dist"
        plan = plan_reference_trajectories(mini_repo)
        build_package(plan, mini_repo, dest, git_sha=None)
        with pytest.raises(HostingSourceError, match="not empty"):
            build_package(plan, mini_repo, dest, git_sha=None)

    def test_missing_committed_sources_fail(self, mini_repo: Path, tmp_path: Path) -> None:
        (
            mini_repo / "release/hosting/chamber-bench-reference-trajectories/croissant.jsonld"
        ).unlink()
        plan = plan_reference_trajectories(mini_repo)
        with pytest.raises(HostingSourceError, match=r"croissant\.jsonld"):
            build_package(plan, mini_repo, tmp_path / "dist", git_sha=None)

    def test_missing_planned_payload_fails(self, mini_repo: Path, tmp_path: Path) -> None:
        plan = plan_reference_trajectories(mini_repo)
        broken = PackagePlan(
            name=plan.name,
            copies={**plan.copies, "ghost.txt": mini_repo / "nope.txt"},
            generated=plan.generated,
        )
        with pytest.raises(HostingSourceError, match=r"ghost|nope"):
            build_package(broken, mini_repo, tmp_path / "dist", git_sha=None)


class TestCommittedCroissantSources:
    """The committed croissant.jsonld files stay structurally sound."""

    RAI_KEYS = (
        "rai:dataCollection",
        "rai:dataCollectionType",
        "rai:dataBiases",
        "rai:dataLimitations",
        "rai:dataUseCases",
        "rai:personalSensitiveInformation",
    )

    @pytest.mark.parametrize("name", HOSTING_ARTIFACTS)
    def test_croissant_structure(self, name: str) -> None:
        path = REPO_ROOT / "release/hosting" / name / "croissant.jsonld"
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["@type"] == "sc:Dataset"
        assert data["name"] == name
        assert data["conformsTo"] == "http://mlcommons.org/croissant/1.0"
        assert data["license"] == "https://www.apache.org/licenses/LICENSE-2.0"
        for key in self.RAI_KEYS:
            assert data[key], f"missing Responsible-AI field {key}"
        ids = [entry["@id"] for entry in data["distribution"]]
        assert "repo" in ids
        assert "manifest.json" in ids
        assert "SHA256SUMS.txt" in ids
        assert "README.md" in ids

    @pytest.mark.parametrize("name", HOSTING_ARTIFACTS)
    def test_dataset_card_exists(self, name: str) -> None:
        card = REPO_ROOT / "release/hosting" / name / "README.md"
        text = card.read_text(encoding="utf-8")
        # Hugging Face dataset-card YAML frontmatter, then the heading.
        assert text.startswith("---\nlicense: apache-2.0\n")
        assert f"\n# {name}\n" in text
        assert "Apache-2.0" in text
