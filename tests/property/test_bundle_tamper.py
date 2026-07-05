# SPDX-License-Identifier: Apache-2.0
"""Property tests: bundle round-trip + single-byte tamper detection (ADR-028 §Decision 3).

Two properties over a canonical bundle built once per session (no env
runs — fabricated episode records keep every example fast):

1. Write → read round-trip: the ``bundle.json`` on disk validates back
   to the exact :class:`ResultBundle` that was written.
2. Tamper detection: flipping **any** single byte of **any** file in
   the bundle directory makes ``verify_bundle_dir`` fail — the
   ADR-028 §Decision 3 integrity guarantee the smoke-eval CI job spot
   checks with one byte, generalized.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from chamber.evaluation.bundles import (
    compute_summary,
    verify_bundle_dir,
    write_bundle_dir,
)
from chamber.evaluation.results import (
    EpisodeResult,
    PlatformFingerprint,
    ResultBundle,
    SeedSchedule,
    load_run_archive,
)
from chamber.partners.api import PartnerSpec

_SEEDS = (0, 1)
_EPISODES_PER_SEED = 3


def _episodes(seed: int) -> list[EpisodeResult]:
    return [
        EpisodeResult(
            seed=seed,
            episode_idx=idx,
            initial_state_seed=idx,
            success=((seed + idx) % 2 == 0),
            metadata={"condition": "property_test", "final_reward": -float(idx)},
        )
        for idx in range(_EPISODES_PER_SEED)
    ]


@pytest.fixture(scope="module")
def canonical_bundle_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One bundle for the whole module; examples copy it, never mutate it."""
    out = tmp_path_factory.mktemp("bundle_src") / "bundle"
    episodes_by_seed = {seed: _episodes(seed) for seed in _SEEDS}
    all_episodes = [ep for eps in episodes_by_seed.values() for ep in eps]
    bundle = ResultBundle(
        task_id="mpe_cooperative_push",
        task_version=1,
        policy_id="random",
        partner_set_id="adhoc:scripted_heuristic",
        partner_hashes={
            "scripted_heuristic": PartnerSpec(
                class_name="scripted_heuristic",
                seed=0,
                checkpoint_step=None,
                weights_uri=None,
                extra={"uid": "partner", "target_xy": "0.0,0.0", "action_dim": "2"},
            ).partner_id
        },
        git_sha="f" * 40,
        dirty=False,
        package_version="0.0.0-test",
        seed_schedule=SeedSchedule(
            root_seed=0,
            seeds=list(_SEEDS),
            episodes_per_seed=_EPISODES_PER_SEED,
            substream_labels=["property.{seed}"],
        ),
        repro_command="chamber-eval run --task mpe_cooperative_push",
        platform=PlatformFingerprint(
            os="test-os", python="3.11", numpy="0", torch=None, device="cpu"
        ),
        manifest={},
        summary=compute_summary(all_episodes, n_resamples=100),
    )
    write_bundle_dir(
        out,
        bundle_without_manifest=bundle,
        episodes_by_seed=episodes_by_seed,
        partner_specs=[
            {
                "name": "scripted_heuristic",
                "class_name": "scripted_heuristic",
                "seed": 0,
                "checkpoint_step": None,
                "weights_uri": None,
                "extra": {"uid": "partner", "target_xy": "0.0,0.0", "action_dim": "2"},
            }
        ],
        repro_command="chamber-eval run --task mpe_cooperative_push",
    )
    return out


def test_write_read_round_trip(canonical_bundle_dir: Path) -> None:
    """Property 1 (degenerate case is enough: write path is deterministic)."""
    loaded = load_run_archive(canonical_bundle_dir / "bundle.json")
    assert isinstance(loaded, ResultBundle)
    assert loaded.manifest, "manifest must be populated by write_bundle_dir"
    rows = verify_bundle_dir(canonical_bundle_dir, repo_path=Path.cwd())
    assert all(r.ok for r in rows)


@settings(max_examples=40, deadline=None)
@given(data=st.data())
def test_any_single_byte_tamper_fails_verify(
    canonical_bundle_dir: Path, data: st.DataObject, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Property 2: no single-byte mutation anywhere survives verify."""
    work = tmp_path_factory.mktemp("tampered") / "bundle"
    shutil.copytree(canonical_bundle_dir, work)

    files = sorted(p for p in work.iterdir() if p.is_file())
    target = data.draw(st.sampled_from(files), label="file")
    raw = bytearray(target.read_bytes())
    offset = data.draw(st.integers(min_value=0, max_value=len(raw) - 1), label="offset")
    flip = data.draw(st.integers(min_value=1, max_value=255), label="xor")
    raw[offset] ^= flip
    target.write_bytes(bytes(raw))

    rows = verify_bundle_dir(work, repo_path=Path.cwd())
    assert not all(r.ok for r in rows), (
        f"verify passed after flipping byte {offset} of {target.name} with 0x{flip:02x}"
    )
