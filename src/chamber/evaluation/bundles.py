# SPDX-License-Identifier: Apache-2.0
"""Result-bundle IO — write, hash, verify v3 bundles (ADR-028 §Decision 1, 3).

A bundle directory is the unit of reviewer-facing evidence: it holds
``bundle.json`` (the :class:`chamber.evaluation.results.ResultBundle`),
per-seed episode JSONL files (one
:class:`chamber.evaluation.results.EpisodeResult` per line),
``partners.json`` (the partner-spec material the identity hashes are
recomputed from), an optional ``prereg.yaml`` copy, ``REPRO.txt``, and
``SHA256SUMS.txt``. :func:`write_bundle_dir` produces it;
:func:`verify_bundle_dir` is the leaderboard admission gate (ADR-028
§Decision 3) — it re-derives everything derivable and fails loudly on
any mismatch, CPU-only, on a bundle produced anywhere.

Integrity is two-layer because "SHA-256 of every file" cannot be
literal (a file cannot contain its own hash): ``bundle.json``'s
``manifest`` covers every file except ``bundle.json`` and
``SHA256SUMS.txt``; ``SHA256SUMS.txt`` covers every file except itself
(including ``bundle.json``). Recorded in ADR-028 §Revision history.

Summary statistics are byte-reproducible (ADR-002 P6): the bootstrap
rng routes through :func:`concerto.training.seeding.derive_substream`
with the ``bootstrap_root_seed`` pinned in the bundle, so
:func:`verify_bundle_dir` recomputes identical numbers up to JSON
float round-trip.
"""

from __future__ import annotations

import hashlib
import json
import platform as _platform
import shutil
import subprocess
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version
from typing import TYPE_CHECKING

import numpy as np

from chamber.evaluation.bootstrap import cluster_bootstrap
from chamber.evaluation.prereg import PreregistrationError, _git
from chamber.evaluation.results import (
    BundleSummary,
    EpisodeResult,
    PlatformFingerprint,
    ResultBundle,
    load_run_archive,
)
from chamber.partners.api import PartnerSpec
from chamber.partners.registry import list_registered
from concerto.training.seeding import derive_substream

if TYPE_CHECKING:
    from pathlib import Path

#: Exit code for the dirty-working-tree refusal in ``chamber-eval run``
#: (ADR-028 §Rev 2026-07-05). Distinct from argparse's 2, the training
#: trip-wire's 3, prereg-mismatch 4, next-stage 5, adapter-not-wired 6.
DIRTY_TREE_EXIT_CODE: int = 7

#: ``derive_substream`` label for the bundle summary bootstrap
#: (ADR-002 P6; ADR-028 §Decision 3 — pinned so verify recomputes
#: byte-identical statistics).
BOOTSTRAP_SUBSTREAM: str = "evaluation.bundle_bootstrap"

#: Bootstrap resample count for bundle summaries (ADR-008 default).
DEFAULT_N_RESAMPLES: int = 2000

#: Relative tolerance for the verify-time summary comparison — floats
#: survive a JSON round-trip well inside 1e-9 (ADR-028 §Decision 3).
SUMMARY_REL_TOL: float = 1e-9

_BUNDLE_JSON = "bundle.json"
_SHA256SUMS = "SHA256SUMS.txt"
_PARTNERS_JSON = "partners.json"
_REPRO_TXT = "REPRO.txt"


@dataclass(frozen=True)
class GitProvenance:
    """Launch-commit provenance of a run (ADR-028 §Decision 1).

    ``dirty`` is ``True`` when the working tree has uncommitted
    changes (``git status --porcelain`` non-empty) **or** when git
    state cannot be determined at all — an unprovable tree is treated
    as dirty, never silently as clean.
    """

    sha: str
    dirty: bool


@dataclass(frozen=True)
class CheckResult:
    """One row of the ``chamber-eval verify`` PASS/FAIL table (ADR-028 §Decision 3)."""

    name: str
    ok: bool
    detail: str


def sha256_file(path: Path) -> str:
    """SHA-256 hex digest of a file's bytes (ADR-028 §Decision 1 manifest hashing)."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_provenance(repo_path: Path) -> GitProvenance:
    """Resolve the launch commit + dirty flag once, at bundle-write time (ADR-028 §Decision 1).

    Follows the ADR-002 §Rev 2026-06-12 capture-once discipline: the
    caller resolves this a single time per run and stamps it into the
    bundle. Any git failure yields ``("unknown", dirty=True)``.
    """
    git = shutil.which("git")
    if git is None:
        return GitProvenance(sha="unknown", dirty=True)
    try:
        sha = subprocess.run(  # noqa: S603 — git binary resolved via shutil.which
            [git, "-C", str(repo_path), "rev-parse", "HEAD"],
            capture_output=True,
            check=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        status = subprocess.run(  # noqa: S603
            [git, "-C", str(repo_path), "status", "--porcelain"],
            capture_output=True,
            check=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return GitProvenance(sha="unknown", dirty=True)
    return GitProvenance(sha=sha, dirty=bool(status))


def _optional_dist_version(name: str) -> str | None:
    try:
        return _dist_version(name)
    except PackageNotFoundError:
        return None


def platform_fingerprint() -> PlatformFingerprint:
    """Capture the ADR-028 §Decision 1 platform fingerprint.

    Dependency versions come from installed-distribution metadata (no
    ``import torch`` — the fingerprint must be capturable on the
    CPU-only verify path without pulling heavyweight imports).
    """
    device = "cpu"
    try:  # a GPU name is provenance gold when present, never required
        import torch  # noqa: PLC0415 — deliberately lazy; CPU-only verify must not pay the torch import

        if torch.cuda.is_available():  # pragma: no cover - GPU-host only
            device = torch.cuda.get_device_name(0)
    except Exception:
        device = "cpu"
    return PlatformFingerprint(
        os=_platform.platform(),
        python=_platform.python_version(),
        numpy=np.__version__,
        torch=_optional_dist_version("torch"),
        device=device,
    )


def compute_summary(
    episodes: list[EpisodeResult],
    *,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    bootstrap_root_seed: int = 0,
) -> BundleSummary:
    """Compute the recomputable bundle summary from raw episodes (ADR-028 §Decision 3).

    Pure function of ``(episodes, n_resamples, bootstrap_root_seed)``:
    the bootstrap rng derives from :data:`BOOTSTRAP_SUBSTREAM`, so run
    time and verify time produce identical numbers (ADR-002 P6).
    """
    by_seed: dict[int, list[float]] = {}
    for ep in episodes:
        by_seed.setdefault(ep.seed, []).append(1.0 if ep.success else 0.0)
    rng = derive_substream(BOOTSTRAP_SUBSTREAM, root_seed=bootstrap_root_seed).default_rng()
    ci = cluster_bootstrap(by_seed, n_resamples=n_resamples, rng=rng)
    scores = [1.0 if ep.success else 0.0 for ep in episodes]
    return BundleSummary(
        n_episodes=len(episodes),
        success_mean=float(np.mean(scores)) if scores else 0.0,
        success_iqm=ci.iqm,
        success_ci_low=ci.ci_low,
        success_ci_high=ci.ci_high,
        n_resamples=n_resamples,
        bootstrap_root_seed=bootstrap_root_seed,
    )


def episode_file_name(seed: int) -> str:
    """Canonical per-seed episode JSONL file name (ADR-028 §Decision 1)."""
    return f"episodes_seed{seed}.jsonl"


def write_bundle_dir(
    out_dir: Path,
    *,
    bundle_without_manifest: ResultBundle,
    episodes_by_seed: dict[int, list[EpisodeResult]],
    partner_specs: list[dict[str, object]],
    repro_command: str,
    prereg_source: Path | None = None,
) -> ResultBundle:
    """Write a complete v3 bundle directory (ADR-028 §Decision 1).

    Writes the episode JSONL files, ``partners.json``, ``REPRO.txt``,
    and the optional ``prereg.yaml`` copy first; computes the manifest
    over them; writes ``bundle.json`` with the manifest folded in; and
    finishes with ``SHA256SUMS.txt`` over everything except itself.
    Refuses a non-empty ``out_dir`` (bundles are immutable evidence,
    invariant I8 — never write into an existing archive).

    Args:
        out_dir: Bundle directory to create.
        bundle_without_manifest: The bundle record with an empty
            ``manifest`` (filled in here).
        episodes_by_seed: Raw episode records keyed by cluster seed.
        partner_specs: Serialized partner-spec material (the fields
            ``PartnerSpec.partner_id`` hashes over, plus ``name``).
        repro_command: Exact reproduction invocation.
        prereg_source: Pre-registration file to copy in as
            ``prereg.yaml``, when the run was preregistered.

    Returns:
        The final :class:`ResultBundle` as written to ``bundle.json``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.iterdir()):
        msg = f"refusing to write bundle into non-empty directory {out_dir} (I8)"
        raise FileExistsError(msg)

    for seed in sorted(episodes_by_seed):
        lines = [ep.model_dump_json() for ep in episodes_by_seed[seed]]
        (out_dir / episode_file_name(seed)).write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / _PARTNERS_JSON).write_text(
        json.dumps(partner_specs, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out_dir / _REPRO_TXT).write_text(repro_command + "\n", encoding="utf-8")
    if prereg_source is not None:
        shutil.copyfile(prereg_source, out_dir / "prereg.yaml")

    manifest = {
        p.name: sha256_file(p)
        for p in sorted(out_dir.iterdir())
        if p.name not in (_BUNDLE_JSON, _SHA256SUMS)
    }
    bundle = bundle_without_manifest.model_copy(update={"manifest": manifest})
    (out_dir / _BUNDLE_JSON).write_text(bundle.model_dump_json(indent=2) + "\n", encoding="utf-8")

    sums = "".join(
        f"{sha256_file(p)}  {p.name}\n" for p in sorted(out_dir.iterdir()) if p.name != _SHA256SUMS
    )
    (out_dir / _SHA256SUMS).write_text(sums, encoding="utf-8")
    return bundle


def _check_manifest(bundle: ResultBundle, bundle_dir: Path) -> list[CheckResult]:
    rows: list[CheckResult] = []
    for name, expected in sorted(bundle.manifest.items()):
        target = bundle_dir / name
        if not target.is_file():
            rows.append(CheckResult(f"manifest:{name}", ok=False, detail="file missing"))
            continue
        actual = sha256_file(target)
        ok = actual == expected
        detail = "sha256 match" if ok else f"sha256 mismatch ({actual[:12]}… != {expected[:12]}…)"
        rows.append(CheckResult(f"manifest:{name}", ok=ok, detail=detail))
    unlisted = [
        p.name
        for p in sorted(bundle_dir.iterdir())
        if p.name not in bundle.manifest and p.name not in (_BUNDLE_JSON, _SHA256SUMS)
    ]
    rows.append(
        CheckResult(
            "manifest:membership",
            ok=not unlisted,
            detail="no unmanifested files" if not unlisted else f"unmanifested: {unlisted}",
        )
    )
    return rows


def _check_sha256sums(bundle_dir: Path) -> list[CheckResult]:
    sums_path = bundle_dir / _SHA256SUMS
    if not sums_path.is_file():
        return [CheckResult("sha256sums", ok=False, detail=f"{_SHA256SUMS} missing")]
    try:
        sums_text = sums_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        return [CheckResult("sha256sums", ok=False, detail=f"{_SHA256SUMS} unreadable: {exc!r}")]
    rows: list[CheckResult] = []
    for line in sums_text.splitlines():
        if not line.strip():
            continue
        expected, _, name = line.partition("  ")
        target = bundle_dir / name
        if not target.is_file():
            rows.append(CheckResult(f"sha256sums:{name}", ok=False, detail="file missing"))
            continue
        ok = sha256_file(target) == expected
        rows.append(
            CheckResult(
                f"sha256sums:{name}",
                ok=ok,
                detail="sha256 match" if ok else "sha256 mismatch",
            )
        )
    return rows


def _load_episodes(bundle: ResultBundle, bundle_dir: Path) -> list[EpisodeResult]:
    episodes: list[EpisodeResult] = []
    for seed in bundle.seed_schedule.seeds:
        path = bundle_dir / episode_file_name(seed)
        episodes.extend(
            EpisodeResult.model_validate_json(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return episodes


def _check_summary(bundle: ResultBundle, bundle_dir: Path) -> list[CheckResult]:
    try:
        episodes = _load_episodes(bundle, bundle_dir)
    except (OSError, ValueError) as exc:
        return [CheckResult("summary:recompute", ok=False, detail=f"episode load failed: {exc}")]
    recomputed = compute_summary(
        episodes,
        n_resamples=bundle.summary.n_resamples,
        bootstrap_root_seed=bundle.summary.bootstrap_root_seed,
    )
    rows: list[CheckResult] = []
    expected_total = bundle.seed_schedule.episodes_per_seed * len(bundle.seed_schedule.seeds)
    rows.append(
        CheckResult(
            "summary:n_episodes",
            ok=recomputed.n_episodes == bundle.summary.n_episodes == expected_total,
            detail=f"{recomputed.n_episodes} episodes (schedule expects {expected_total})",
        )
    )
    for field in ("success_mean", "success_iqm", "success_ci_low", "success_ci_high"):
        stated = getattr(bundle.summary, field)
        fresh = getattr(recomputed, field)
        ok = abs(fresh - stated) <= SUMMARY_REL_TOL * max(1.0, abs(stated))
        rows.append(
            CheckResult(
                f"summary:{field}",
                ok=ok,
                detail=f"recomputed {fresh:.6f} vs stated {stated:.6f}",
            )
        )
    return rows


def _check_partner_hashes(bundle: ResultBundle, bundle_dir: Path) -> list[CheckResult]:
    path = bundle_dir / _PARTNERS_JSON
    if not path.is_file():
        return [CheckResult("partners", ok=False, detail=f"{_PARTNERS_JSON} missing")]
    # A tampered partners.json must yield a FAIL row, never a traceback
    # (the tamper-detection property, ADR-028 §Decision 3).
    try:
        specs = json.loads(path.read_text(encoding="utf-8"))
        registered = set(list_registered())
        rows: list[CheckResult] = []
        seen: dict[str, str] = {}
        for entry in specs:
            name = str(entry["name"])
            spec = PartnerSpec(
                class_name=str(entry["class_name"]),
                seed=int(entry["seed"]),
                checkpoint_step=entry["checkpoint_step"],
                weights_uri=entry["weights_uri"],
                extra={str(k): str(v) for k, v in dict(entry.get("extra", {})).items()},
            )
            seen[name] = spec.partner_id
            in_registry = spec.class_name in registered
            rows.append(
                CheckResult(
                    f"partner:{name}:registered",
                    ok=in_registry,
                    detail=spec.class_name if in_registry else f"{spec.class_name} not in registry",
                )
            )
    except (KeyError, TypeError, ValueError, UnicodeDecodeError) as exc:
        return [
            CheckResult(
                "partners",
                ok=False,
                detail=f"{_PARTNERS_JSON} unreadable/malformed: {exc!r}",
            )
        ]
    rows.append(
        CheckResult(
            "partner:hashes",
            ok=seen == bundle.partner_hashes,
            detail="identity hashes match"
            if seen == bundle.partner_hashes
            else f"recomputed {seen} != stated {bundle.partner_hashes}",
        )
    )
    return rows


def _check_prereg(bundle: ResultBundle, bundle_dir: Path, *, repo_path: Path) -> list[CheckResult]:
    if bundle.prereg_git_tag is None:
        return [CheckResult("prereg", ok=True, detail="unpreregistered run (no tag referenced)")]
    rows: list[CheckResult] = []
    prereg_copy = bundle_dir / "prereg.yaml"
    if not prereg_copy.is_file():
        return [CheckResult("prereg:copy", ok=False, detail="prereg.yaml missing from bundle")]
    try:
        _git("rev-parse", "--verify", f"refs/tags/{bundle.prereg_git_tag}", repo_path=repo_path)
        tag_ok = True
        tag_detail = f"tag {bundle.prereg_git_tag} exists"
    except PreregistrationError as exc:
        tag_ok = False
        tag_detail = str(exc)
    rows.append(CheckResult("prereg:tag", ok=tag_ok, detail=tag_detail))
    try:
        on_disk = _git("hash-object", "--", str(prereg_copy), repo_path=repo_path)
    except PreregistrationError as exc:
        return [*rows, CheckResult("prereg:blob", ok=False, detail=str(exc))]
    ok = on_disk == bundle.prereg_blob_sha
    rows.append(
        CheckResult(
            "prereg:blob",
            ok=ok,
            detail="bundle copy matches verified blob SHA"
            if ok
            else f"bundle copy {on_disk[:12]}… != recorded {str(bundle.prereg_blob_sha)[:12]}…",
        )
    )
    return rows


def verify_bundle_dir(
    bundle_dir: Path,
    *,
    repo_path: Path,
    allow_dirty: bool = False,
) -> list[CheckResult]:
    """Verify a bundle directory end to end (ADR-028 §Decision 3).

    The leaderboard admission gate: schema validation, both integrity
    layers, deterministic summary recomputation, partner-hash custody,
    prereg-tag linkage, and the dirty-flag eligibility rule. CPU-only;
    works on a bundle produced anywhere.

    Args:
        bundle_dir: The bundle directory.
        repo_path: Git working tree the prereg tag is resolved in.
        allow_dirty: Downgrade the dirty-bundle eligibility FAIL to an
            annotated pass — for local development loops only; the
            leaderboard gate never sets this.

    Returns:
        The full check table; the bundle passes iff every row is ok.
    """
    bundle_path = bundle_dir / _BUNDLE_JSON
    if not bundle_path.is_file():
        return [CheckResult("bundle", ok=False, detail=f"{_BUNDLE_JSON} missing")]
    try:
        loaded = load_run_archive(bundle_path)
    except Exception as exc:
        return [CheckResult("bundle:schema", ok=False, detail=f"load failed: {exc}")]
    if not isinstance(loaded, ResultBundle):
        return [
            CheckResult(
                "bundle:schema",
                ok=False,
                detail=f"schema_version {loaded.schema_version} is not a v3 bundle "
                "(v2 archives are historical inputs, not verifiable bundles)",
            )
        ]
    rows = [CheckResult("bundle:schema", ok=True, detail=f"v{loaded.schema_version} ResultBundle")]
    if loaded.dirty:
        eligibility = CheckResult(
            "bundle:eligibility",
            ok=allow_dirty,
            detail="dirty bundle — ineligible for leaderboard use"
            + (" (tolerated: --allow-dirty)" if allow_dirty else ""),
        )
    else:
        eligibility = CheckResult("bundle:eligibility", ok=True, detail="clean working tree")
    rows.append(eligibility)
    rows.extend(_check_manifest(loaded, bundle_dir))
    rows.extend(_check_sha256sums(bundle_dir))
    rows.extend(_check_summary(loaded, bundle_dir))
    rows.extend(_check_partner_hashes(loaded, bundle_dir))
    rows.extend(_check_prereg(loaded, bundle_dir, repo_path=repo_path))
    return rows


def render_check_table(rows: list[CheckResult]) -> str:
    """Render the verify PASS/FAIL table (ADR-028 §Decision 3)."""
    width = max(len(r.name) for r in rows)
    lines = [f"{'PASS' if r.ok else 'FAIL'}  {r.name.ljust(width)}  {r.detail}" for r in rows]
    verdict = "PASS" if all(r.ok for r in rows) else "FAIL"
    lines.append(f"verify: {verdict} ({sum(r.ok for r in rows)}/{len(rows)} checks)")
    return "\n".join(lines)


__all__ = [
    "BOOTSTRAP_SUBSTREAM",
    "DEFAULT_N_RESAMPLES",
    "DIRTY_TREE_EXIT_CODE",
    "SUMMARY_REL_TOL",
    "CheckResult",
    "GitProvenance",
    "compute_summary",
    "episode_file_name",
    "git_provenance",
    "platform_fingerprint",
    "render_check_table",
    "sha256_file",
    "verify_bundle_dir",
    "write_bundle_dir",
]
