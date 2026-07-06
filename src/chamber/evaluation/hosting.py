"""Deterministic packaging of the CHAMBER-Bench v1.0 hosted data artifacts.

Three dataset artifacts leave the repository for hosting (ADR-028
§Decision provenance contract; ADR-027 §Versioning; licence per
ADR-012 §Decision):

1. ``chamber-partner-sets`` — the versioned partner zoo: rendered
   cards, per-set machine-readable rosters (public members' committed
   parameter literals; private members' identity hashes only — the
   withheld-parameter custody rule of ADR-009 as amended), per-set
   behavioural fingerprints, and the public learned members'
   checkpoints.
2. ``chamber-leaderboard-bundles`` — the verbatim result bundles
   listed in the per-task ``LEADERBOARD_BUNDLES.txt`` manifests, the
   campaign reports, the committed checkpoint-selection artifacts,
   and the checkpoints the learned rows load.
3. ``chamber-reference-trajectories`` — the fingerprint probe
   archives and the admission-report archives.

Every package is built deterministically from the committed tree
(sorted walks, no timestamps): building twice from the same tree
yields byte-identical ``manifest.json`` and ``SHA256SUMS`` files.
Each package also carries its committed ``DATASET_CARD.md`` and
Croissant ``croissant.json`` (sources under ``release/hosting/``).

The thin CLI wrapper is ``scripts/release/prepare_hosting.py``; the
founder-side upload lives in ``scripts/release/upload_hosting.py``.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from chamber.partners import set_definitions as _set_definitions
from chamber.partners.sets import get_partner_set, list_partner_sets, parse_set_slug

# Importing chamber.partners.set_definitions populates the partner-set
# registry (its documented import side effect); the alias assignment
# keeps the import from being pruned as unused.
_ = _set_definitions

#: Package names, in build order (ADR-028 §Decision; stable public ids).
HOSTING_ARTIFACTS: tuple[str, str, str] = (
    "chamber-partner-sets",
    "chamber-leaderboard-bundles",
    "chamber-reference-trajectories",
)

#: Committed per-artifact sources (dataset card + Croissant metadata).
HOSTING_SOURCE_DIRNAME: str = "release/hosting"

#: Default build destination (git-ignored; ADR-028 §Decision).
HOSTING_DEST_DIRNAME: str = "dist/hosting"

#: Files every package must carry besides its payload.
PACKAGE_CARD_NAME: str = "DATASET_CARD.md"
PACKAGE_CROISSANT_NAME: str = "croissant.json"
PACKAGE_MANIFEST_NAME: str = "manifest.json"
PACKAGE_SUMS_NAME: str = "SHA256SUMS"

_BENCHMARK_ROOT = Path("spikes/results/benchmark")
_FINGERPRINTS_ROOT = Path("spikes/results/partner-fingerprints")
_ADMISSION_ROOT = Path("spikes/results/admission")
_PARTNER_CARDS_ROOT = Path("docs/reference/partners")
_LEADERBOARD_MANIFEST = "LEADERBOARD_BUNDLES.txt"
_LOCAL_URI_PREFIX = "local://"
#: Any checkpoint URI in a REPRO.txt (``--policy joint_ego:local://…``
#: and ``--partner-weights local://…`` alike).
_REPRO_CHECKPOINT_URI = re.compile(r"local://\S+\.pt")


class HostingSourceError(RuntimeError):
    """A required source file for a hosting package is missing (ADR-028 §Decision)."""


@dataclass(frozen=True)
class PackagePlan:
    """One hosting package, fully planned before any byte is written.

    ADR-028 §Decision (the provenance contract the packages extend to
    hosting). ``copies`` maps destination-relative paths to absolute
    source files; ``generated`` maps destination-relative paths to
    already-rendered file content.
    """

    name: str
    copies: dict[str, Path]
    generated: dict[str, str]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_local_uri(uri: str, repo_root: Path) -> Path:
    """Resolve a ``local://artifacts/<name>.pt`` URI against the repo tree.

    Mirrors ``concerto.training.checkpoints.resolve_uri`` with the
    repository-conventional ``artifacts/`` root (ADR-002; ADR-018
    custody URIs).
    """
    if not uri.startswith(_LOCAL_URI_PREFIX):
        msg = f"unsupported checkpoint URI scheme for hosting: {uri!r}"
        raise HostingSourceError(msg)
    return repo_root / "artifacts" / uri[len(_LOCAL_URI_PREFIX) :]


def _copy_tree(entries: dict[str, Path], dest_prefix: str, source_root: Path) -> None:
    """Add every file under ``source_root`` to ``entries``, sorted, prefix-mapped."""
    for path in sorted(p for p in source_root.rglob("*") if p.is_file()):
        rel = path.relative_to(source_root).as_posix()
        entries[f"{dest_prefix}/{rel}"] = path


def _checkpoint_copies(uris: set[str], repo_root: Path) -> dict[str, Path]:
    """Map checkpoint URIs (plus JSON sidecars) to package entries.

    Missing payloads are a hard error, listed all at once — a hosting
    package must never silently ship without the checkpoints its
    rosters and reproduction commands reference (ADR-018 custody;
    ADR-028 §Decision).
    """
    entries: dict[str, Path] = {}
    missing: list[str] = []
    for uri in sorted(uris):
        payload = _resolve_local_uri(uri, repo_root)
        if not payload.is_file():
            missing.append(f"{uri} -> {payload}")
            continue
        entries[f"checkpoints/{payload.name}"] = payload
        sidecar = payload.with_suffix(payload.suffix + ".json")
        if sidecar.is_file():
            entries[f"checkpoints/{sidecar.name}"] = sidecar
    if missing:
        msg = "missing checkpoint payload(s):\n  " + "\n  ".join(missing)
        raise HostingSourceError(msg)
    return entries


def _set_roster_json(slug: str) -> str:
    """Render one partner set as JSON with the custody rule enforced.

    Public members carry their committed parameter literals; private
    members carry ``params: null`` plus the identity digest — exactly
    the committed source surface, re-checked here so a hosting build
    can never leak a withheld parameter (ADR-009 as amended).
    """
    set_id, version = parse_set_slug(slug)
    spec = get_partner_set(set_id, version)
    payload = spec.model_dump(mode="json")
    for member in spec.members:
        if member.split == "private" and member.params is not None:
            msg = (
                f"partner set {slug!r}: private member {member.member_name!r} "
                "carries parameter literals; refusing to package"
            )
            raise HostingSourceError(msg)
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def plan_partner_sets(repo_root: Path) -> PackagePlan:
    """Plan the ``chamber-partner-sets`` package (ADR-009 as amended; ADR-027 §Versioning)."""
    copies: dict[str, Path] = {}
    generated: dict[str, str] = {}
    _copy_tree(copies, "cards", repo_root / _PARTNER_CARDS_ROOT)
    checkpoint_uris: set[str] = set()
    for slug in list_partner_sets():
        set_id, version = parse_set_slug(slug)
        spec = get_partner_set(set_id, version)
        generated[f"sets/{slug.replace('@', '-')}.json"] = _set_roster_json(slug)
        fingerprints = repo_root / _FINGERPRINTS_ROOT / slug.replace("@", "-") / "fingerprints.json"
        if fingerprints.is_file():
            copies[f"fingerprints/{slug.replace('@', '-')}.json"] = fingerprints
        checkpoint_uris.update(
            member.checkpoint_uri
            for member in spec.public_members
            if member.checkpoint_uri is not None
        )
    copies.update(_checkpoint_copies(checkpoint_uris, repo_root))
    return PackagePlan(name=HOSTING_ARTIFACTS[0], copies=copies, generated=generated)


def _leaderboard_bundle_dirs(task_dir: Path, repo_root: Path) -> list[Path]:
    manifest = task_dir / _LEADERBOARD_MANIFEST
    bundles: list[Path] = []
    for raw_line in manifest.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#"):
            bundle = repo_root / line
            if not bundle.is_dir():
                msg = f"{manifest}: listed bundle {line!r} is not a directory"
                raise HostingSourceError(msg)
            bundles.append(bundle)
    return bundles


def plan_leaderboard_bundles(repo_root: Path) -> PackagePlan:
    """Plan the ``chamber-leaderboard-bundles`` package (ADR-028; ADR-027 §Reporting rules)."""
    copies: dict[str, Path] = {}
    checkpoint_uris: set[str] = set()
    benchmark_root = repo_root / _BENCHMARK_ROOT
    task_dirs = sorted(path.parent for path in benchmark_root.glob(f"*/{_LEADERBOARD_MANIFEST}"))
    if not task_dirs:
        msg = f"no {_LEADERBOARD_MANIFEST} manifests under {benchmark_root}"
        raise HostingSourceError(msg)
    for task_dir in task_dirs:
        prefix = f"benchmark/{task_dir.name}"
        copies[f"{prefix}/{_LEADERBOARD_MANIFEST}"] = task_dir / _LEADERBOARD_MANIFEST
        campaign_report = task_dir / "CAMPAIGN_REPORT.md"
        if campaign_report.is_file():
            copies[f"{prefix}/CAMPAIGN_REPORT.md"] = campaign_report
        selection = task_dir / "selection"
        if selection.is_dir():
            _copy_tree(copies, f"{prefix}/selection", selection)
            for manifest in sorted(selection.glob("*_selected_manifest.json")):
                selected: dict[str, str] = json.loads(manifest.read_text(encoding="utf-8"))
                checkpoint_uris.update(selected.values())
        for bundle in _leaderboard_bundle_dirs(task_dir, repo_root):
            _copy_tree(copies, f"{prefix}/{bundle.name}", bundle)
            repro = (bundle / "REPRO.txt").read_text(encoding="utf-8")
            checkpoint_uris.update(_REPRO_CHECKPOINT_URI.findall(repro))
    copies.update(_checkpoint_copies(checkpoint_uris, repo_root))
    return PackagePlan(name=HOSTING_ARTIFACTS[1], copies=copies, generated={})


def plan_reference_trajectories(repo_root: Path) -> PackagePlan:
    """Plan the ``chamber-reference-trajectories`` package (ADR-027 §Decision)."""
    copies: dict[str, Path] = {}
    for prefix, root in (
        ("partner-fingerprints", repo_root / _FINGERPRINTS_ROOT),
        ("admission", repo_root / _ADMISSION_ROOT),
    ):
        if not root.is_dir():
            msg = f"required evidence tree missing: {root}"
            raise HostingSourceError(msg)
        _copy_tree(copies, prefix, root)
    return PackagePlan(name=HOSTING_ARTIFACTS[2], copies=copies, generated={})


def plan_all(repo_root: Path) -> list[PackagePlan]:
    """Plan the three hosting packages (ADR-028 §Decision), in stable order."""
    return [
        plan_partner_sets(repo_root),
        plan_leaderboard_bundles(repo_root),
        plan_reference_trajectories(repo_root),
    ]


def build_package(
    plan: PackagePlan,
    repo_root: Path,
    dest_root: Path,
    *,
    git_sha: str | None = None,
) -> Path:
    """Materialise one package under ``dest_root/<name>`` (ADR-028 §Decision).

    Layout: payload files, the committed ``DATASET_CARD.md`` and
    ``croissant.json`` (from ``release/hosting/<name>/``), then
    ``manifest.json`` (path → size + SHA-256 over everything except
    itself and ``SHA256SUMS``), then ``SHA256SUMS`` (everything except
    itself) — the same two-layer integrity shape as a result bundle.
    Deterministic: identical inputs produce identical bytes.

    Returns:
        The package directory.

    Raises:
        HostingSourceError: a committed card/croissant source or a
            planned payload file is missing, or the destination is
            non-empty.
    """
    package_dir = dest_root / plan.name
    if package_dir.exists() and any(package_dir.iterdir()):
        msg = f"destination {package_dir} exists and is not empty"
        raise HostingSourceError(msg)
    source_dir = repo_root / HOSTING_SOURCE_DIRNAME / plan.name
    for required in (PACKAGE_CARD_NAME, PACKAGE_CROISSANT_NAME):
        if not (source_dir / required).is_file():
            msg = f"missing committed source {source_dir / required}"
            raise HostingSourceError(msg)
    package_dir.mkdir(parents=True, exist_ok=True)

    for rel, source in sorted(plan.copies.items()):
        if not source.is_file():
            msg = f"planned payload missing: {source}"
            raise HostingSourceError(msg)
        target = package_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    for rel, content in sorted(plan.generated.items()):
        target = package_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    shutil.copyfile(source_dir / PACKAGE_CARD_NAME, package_dir / PACKAGE_CARD_NAME)
    shutil.copyfile(source_dir / PACKAGE_CROISSANT_NAME, package_dir / PACKAGE_CROISSANT_NAME)

    payload = sorted(
        path
        for path in package_dir.rglob("*")
        if path.is_file() and path.name not in (PACKAGE_MANIFEST_NAME, PACKAGE_SUMS_NAME)
    )
    manifest = {
        "artifact": plan.name,
        "git_sha": git_sha,
        "files": [
            {
                "path": path.relative_to(package_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
            for path in payload
        ],
    }
    manifest_path = package_dir / PACKAGE_MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    sums_lines = [
        f"{_sha256_file(path)}  {path.relative_to(package_dir).as_posix()}"
        for path in (*payload, manifest_path)
    ]
    (package_dir / PACKAGE_SUMS_NAME).write_text(
        "\n".join(sorted(sums_lines)) + "\n", encoding="utf-8"
    )
    return package_dir


def build_all(
    repo_root: Path,
    dest_root: Path,
    *,
    git_sha: str | None = None,
) -> list[Path]:
    """Build the three packages (ADR-028 §Decision); returns their directories."""
    return [
        build_package(plan, repo_root, dest_root, git_sha=git_sha) for plan in plan_all(repo_root)
    ]
