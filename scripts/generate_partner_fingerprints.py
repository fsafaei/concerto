#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate a partner-set fingerprint archive (ADR-009 as amended 2026-07-05; ADR-028).

Runs every member of the named partner set(s) — public **and** private
— through the committed probe suite
(:mod:`chamber.benchmarks.partner_probe`) against the task's reference
ego, and writes the immutable archive
``spikes/results/partner-fingerprints/<set_id>-v<N>/``:

- ``<member_name>/`` — one ADR-028 v3 result bundle per member (each
  passes ``chamber-eval verify``); private members' ``partners.json``
  entries are redacted to the withheld-parameters marker.
- ``<member_name>-floor/`` — the separate floor-probe bundle where the
  set's floor is measured on a different variant (the handover
  free-re-grasp endpoint).
- ``fingerprints.json`` — per-member behavioural fingerprints + the
  committed-floor verdicts the cards render from.
- ``SHA256SUMS.txt`` — recursive digests over the archive (I8).

Exits non-zero if any member fails the set's committed floor (the
member must then be dropped from the set — a member no ego can work
with measures nothing).

Requires the maintainer-held private-parameter seed
(``CHAMBER_PRIVATE_PARTNER_SEED``) since private members are probed
too, and a clean working tree (bundles are evidence; ADR-028).

Usage::

    uv run python scripts/generate_partner_fingerprints.py            # all sets
    uv run python scripts/generate_partner_fingerprints.py --set cocarry_partners@v1
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

import chamber
import chamber.partners  # registers the v1 sets
from chamber.benchmarks.partner_probe import (
    REFERENCE_EGO_IDS,
    fingerprint_statistics,
    run_member_probe,
)
from chamber.evaluation.bundles import (
    compute_summary,
    git_provenance,
    platform_fingerprint,
    sha256_file,
    write_bundle_dir,
)
from chamber.evaluation.results import ResultBundle, SeedSchedule
from chamber.partners.sets import (
    PartnerSetSpec,
    get_partner_set,
    list_partner_sets,
    parse_set_slug,
    private_seed_available,
    resolve_set_members,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_ARCHIVE_ROOT = _REPO_ROOT / "spikes" / "results" / "partner-fingerprints"

#: Fingerprint-archive schema version (I9: independent of the ADR-028
#: bundle SCHEMA_VERSION; bumping requires ADR review).
FINGERPRINTS_SCHEMA_VERSION: int = 1

#: Exit code when a member fails the committed floor.
FLOOR_FAIL_EXIT_CODE: int = 3


def archive_dir_name(set_spec: PartnerSetSpec) -> str:
    """``<set_id>-v<N>`` — the archive directory name (ADR-009 as amended)."""
    return f"{set_spec.set_id}-v{set_spec.version}"


def _write_member_bundle(
    out_dir: Path,
    set_spec: PartnerSetSpec,
    run_episodes_by_seed: dict[int, list],
    material: list[dict[str, object]],
    hashes: dict[str, str],
    substream_labels: list[str],
    *,
    git_sha: str,
    dirty: bool,
    repro_command: str,
) -> None:
    episodes = [ep for seed in sorted(run_episodes_by_seed) for ep in run_episodes_by_seed[seed]]
    bundle = ResultBundle(
        task_id=set_spec.task_id,
        task_version=set_spec.task_version,
        policy_id=REFERENCE_EGO_IDS[set_spec.task_id],
        partner_set_id=set_spec.slug,
        partner_hashes=hashes,
        git_sha=git_sha,
        dirty=dirty,
        package_version=chamber.__version__,
        seed_schedule=SeedSchedule(
            root_seed=0,
            seeds=list(set_spec.probe_seeds),
            episodes_per_seed=set_spec.probe_episodes_per_seed,
            substream_labels=substream_labels,
        ),
        repro_command=repro_command,
        platform=platform_fingerprint(),
        manifest={},
        summary=compute_summary(episodes, bootstrap_root_seed=0),
        prereg_git_tag=None,
        prereg_blob_sha=None,
    )
    write_bundle_dir(
        out_dir,
        bundle_without_manifest=bundle,
        episodes_by_seed=run_episodes_by_seed,
        partner_specs=material,
        repro_command=repro_command,
    )


def generate_set_archive(
    set_spec: PartnerSetSpec,
    *,
    render_backend: str | None,
    allow_dirty: bool,
    argv: list[str],
) -> tuple[Path, list[str]]:
    """Probe every member and write the set's archive; return (path, floor failures)."""
    provenance = git_provenance(_REPO_ROOT)
    if provenance.dirty and not allow_dirty:
        msg = (
            "working tree is dirty; a fingerprint archive must be traceable to a "
            "commit (ADR-028). Commit first, or pass --allow-dirty for a dev run."
        )
        raise SystemExit(msg)
    out_root = _ARCHIVE_ROOT / archive_dir_name(set_spec)
    out_root.mkdir(parents=True, exist_ok=True)
    if any(out_root.iterdir()):
        msg = f"refusing to write into non-empty archive {out_root} (I8)"
        raise SystemExit(msg)
    repro_command = "generate_partner_fingerprints.py " + shlex.join(argv)

    failures: list[str] = []
    members_payload: dict[str, dict[str, object]] = {}
    for member, params in resolve_set_members(set_spec, include_private=True):
        print(f"[{set_spec.slug}] probing {member.member_name} ({member.split}) …", flush=True)
        run = run_member_probe(
            set_spec, member, params, variant="fingerprint", render_backend=render_backend
        )
        _write_member_bundle(
            out_root / member.member_name,
            set_spec,
            run.episodes_by_seed,
            list(run.partner_material),
            dict(run.partner_hashes),
            list(run.substream_labels),
            git_sha=provenance.sha,
            dirty=provenance.dirty,
            repro_command=repro_command,
        )
        fingerprint_eps = [
            ep for seed in sorted(run.episodes_by_seed) for ep in run.episodes_by_seed[seed]
        ]
        fingerprint = fingerprint_statistics(fingerprint_eps)

        if set_spec.floor_probe == "free_regrasp":
            floor_run = run_member_probe(
                set_spec, member, params, variant="free_regrasp", render_backend=render_backend
            )
            _write_member_bundle(
                out_root / f"{member.member_name}-floor",
                set_spec,
                floor_run.episodes_by_seed,
                list(floor_run.partner_material),
                dict(floor_run.partner_hashes),
                list(floor_run.substream_labels),
                git_sha=provenance.sha,
                dirty=provenance.dirty,
                repro_command=repro_command,
            )
            floor_eps = [
                ep
                for seed in sorted(floor_run.episodes_by_seed)
                for ep in floor_run.episodes_by_seed[seed]
            ]
            floor_success = fingerprint_statistics(floor_eps)["success_rate"]
        else:
            floor_success = fingerprint["success_rate"]

        floor_pass = floor_success >= set_spec.floor
        if not floor_pass:
            failures.append(
                f"{member.member_name}: floor_success {floor_success:.3f} < "
                f"committed floor {set_spec.floor}"
            )
        members_payload[member.member_name] = {
            "partner_id": member.partner_id,
            "params_sha256": member.params_sha256,
            "registry_class": member.registry_class,
            "role": member.role,
            "split": member.split,
            "fingerprint": fingerprint,
            "floor_probe": set_spec.floor_probe,
            "floor_success": floor_success,
            "floor_pass": floor_pass,
        }
        print(
            f"[{set_spec.slug}]   success {fingerprint['success_rate']:.3f}  "
            f"floor {floor_success:.3f} ({'PASS' if floor_pass else 'FAIL'})",
            flush=True,
        )

    payload = {
        "schema_version": FINGERPRINTS_SCHEMA_VERSION,
        "set_id": set_spec.set_id,
        "set_version": set_spec.version,
        "task": f"{set_spec.task_id}@v{set_spec.task_version}",
        "floor": set_spec.floor,
        "floor_probe": set_spec.floor_probe,
        "probe_seeds": list(set_spec.probe_seeds),
        "probe_episodes_per_seed": set_spec.probe_episodes_per_seed,
        "git_sha": provenance.sha,
        "dirty": provenance.dirty,
        "members": members_payload,
    }
    (out_root / "fingerprints.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    sums_lines = [
        f"{sha256_file(path)}  {path.relative_to(out_root)}\n"
        for path in sorted(out_root.rglob("*"))
        if path.is_file() and path.name != "SHA256SUMS.txt"
    ]
    (out_root / "SHA256SUMS.txt").write_text("".join(sums_lines), encoding="utf-8")
    return out_root, failures


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    parser = argparse.ArgumentParser(
        description="Generate partner-set fingerprint archives (ADR-009 as amended)."
    )
    parser.add_argument(
        "--set",
        dest="sets",
        action="append",
        default=None,
        help="Set slug (set_id[@vN]); repeatable. Default: every registered set.",
    )
    parser.add_argument(
        "--render-backend",
        default=None,
        help="Render backend forwarded to SAPIEN env factories ('none' on headless hosts).",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Tolerate a dirty tree (dev only; the bundles are marked dirty).",
    )
    args = parser.parse_args(argv)
    argv_list = list(sys.argv[1:]) if argv is None else list(argv)

    if not private_seed_available():
        print(
            "error: the fingerprint archive covers private members too; set "
            "CHAMBER_PRIVATE_PARTNER_SEED (the maintainer-held seed) first.",
            file=sys.stderr,
        )
        return 2

    slugs = args.sets if args.sets else list_partner_sets()
    all_failures: list[str] = []
    for slug in slugs:
        set_id, version = parse_set_slug(slug)
        set_spec = get_partner_set(set_id, version=version)
        out_root, failures = generate_set_archive(
            set_spec,
            render_backend=args.render_backend,
            allow_dirty=args.allow_dirty,
            argv=argv_list,
        )
        print(f"[{set_spec.slug}] archive written: {out_root}")
        all_failures.extend(f"{set_spec.slug}/{f}" for f in failures)
    if all_failures:
        print("FLOOR FAILURES (drop these members from the set):", file=sys.stderr)
        for line in all_failures:
            print(f"  {line}", file=sys.stderr)
        return FLOOR_FAIL_EXIT_CODE
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
