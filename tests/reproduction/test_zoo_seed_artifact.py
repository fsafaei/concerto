# SPDX-License-Identifier: Apache-2.0
"""Reproduction test for the published M4a zoo-seed manifest (T4b.14; plan/05 §6 #5).

Verifies that the on-disk ``local://artifacts/happo_seed7_step50k.pt``
artefact matches the SHA-256 manifest committed at
``scripts/repro/artifacts/happo_seed7_step50k.pt.sha256``. The test is
the fast-lane substitute for the 2 h GPU run named in
``scripts/repro/zoo_seed.sh`` — once the artefact has been published
(via ``make zoo-seed-gpu`` on a GPU host, then ``make zoo-seed-pull``
to fetch it onto any host), this test re-checks integrity in seconds.

Threat-model coverage (plan/08 §8): the committed manifest is the
fingerprint a tampered checkpoint must not be able to forge. A byte
change in the ``.pt`` payload trips the SHA-256 mismatch path in
:func:`concerto.training.checkpoints.load_checkpoint` (the sidecar-
vs-payload check). The manifest in the public repo is the off-host
fingerprint that closes the loop against an attacker who has write
access to both ``.pt`` and sidecar on the artefact host — the test
pins ``metadata.sha256 == committed_manifest_sha`` to catch that
class of tamper.

The test is :mod:`@pytest.mark.slow` (it reads the full ``.pt`` from
disk) but is **not** GPU-gated — verification runs anywhere. The
``CONCERTO_ARTIFACTS_ROOT`` env var lets a CI host with the artefact
pre-staged elsewhere point the resolver at the right directory.

Skip-when-absent: the canonical artefact is not committed to the
public repo (it lives in out-of-tree storage; see ``make zoo-seed-pull``
for the fetch path). When the artefact is absent the test
:func:`pytest.skip`-s with a message pointing the reader at the pull
target. Mismatch is **not** a skip — a present but tampered ``.pt``
is a loud failure (plan/08 §8 threat model).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from concerto.training.checkpoints import CheckpointError, load_checkpoint

pytestmark = pytest.mark.slow

#: Canonical artefact URI (matches
#: :func:`chamber.partners.selection.make_phase0_draft_zoo`'s frozen-HARL
#: spec ``weights_uri``).
_ARTEFACT_URI: str = "local://artifacts/happo_seed7_step50k.pt"

#: SHA-256 manifest path, committed alongside this test.
_MANIFEST_PATH: Path = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "repro"
    / "artifacts"
    / "happo_seed7_step50k.pt.sha256"
)

#: Env var the loader reads when present (ADR-002 §Revision-history 2026-05-13).
_ARTIFACTS_ROOT_ENV: str = "CONCERTO_ARTIFACTS_ROOT"

#: Default artefact root when the env var is unset (matches
#: :attr:`concerto.training.config.EgoAHTConfig.artifacts_root`).
_DEFAULT_ARTIFACTS_ROOT: Path = Path("./artifacts")


def _resolve_artifacts_root() -> Path:
    raw = os.environ.get(_ARTIFACTS_ROOT_ENV)
    return Path(raw) if raw else _DEFAULT_ARTIFACTS_ROOT


def _expected_sha256() -> str:
    """Read the committed manifest. Strips trailing newlines."""
    return _MANIFEST_PATH.read_text(encoding="utf-8").strip()


def _payload_path() -> Path:
    """Resolve the ``.pt`` payload path under the current artefact root."""
    # local://artifacts/happo_seed7_step50k.pt → <root>/artifacts/happo_seed7_step50k.pt
    relative = _ARTEFACT_URI.removeprefix("local://")
    return _resolve_artifacts_root() / relative


def test_manifest_is_a_64_char_hex_sha256() -> None:
    """ADR-002 §Decisions: the committed manifest is a well-formed SHA-256 hex digest."""
    expected = _expected_sha256()
    assert len(expected) == 64, f"sha256 manifest must be 64 hex chars; got {len(expected)}"
    int(expected, 16)  # raises if non-hex.


def test_artefact_matches_committed_sha256() -> None:
    """plan/05 §6 #5: on-disk ``.pt`` SHA-256 must match the committed manifest.

    Skipped when the artefact is absent (the public repo does not
    commit the binary; see ``make zoo-seed-pull``). When present, a
    SHA-256 mismatch is a loud :class:`CheckpointError` from
    :func:`concerto.training.checkpoints.load_checkpoint`, NOT a skip.
    """
    payload = _payload_path()
    if not payload.exists():
        pytest.skip(
            f"Zoo-seed artefact not present at {payload}. Run `make zoo-seed-pull` to "
            f"fetch it (or set CONCERTO_ARTIFACTS_ROOT to a directory that contains "
            f"the staged artefact)."
        )
    expected = _expected_sha256()
    try:
        _state_dict, metadata = load_checkpoint(
            uri=_ARTEFACT_URI, artifacts_root=_resolve_artifacts_root()
        )
    except CheckpointError as exc:
        pytest.fail(
            f"load_checkpoint failed for {payload}: {exc}. The published .pt + "
            f"sidecar should round-trip cleanly through load_checkpoint."
        )
    # load_checkpoint already verified sidecar.sha256 against on-disk
    # bytes; the test pins that sidecar.sha256 also matches the
    # committed manifest. A mismatch here means the published .pt has
    # been silently replaced even though the sidecar was updated to
    # match — which is exactly what the committed manifest exists to
    # detect.
    assert metadata.sha256 == expected, (
        f"sidecar.sha256={metadata.sha256!r} but committed manifest={expected!r}. "
        f"The published .pt has drifted from the M4a manifest at {_MANIFEST_PATH}."
    )
