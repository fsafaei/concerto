# SPDX-License-Identifier: Apache-2.0
"""Checkpoint save/load with SHA-256 integrity (T4b.12; ADR-002 §Decisions).

The training stack writes ``torch`` state-dicts as ``.pt`` files referenced
from M4a's :class:`~chamber.partners.api.PartnerSpec` ``weights_uri`` field
(plan/04 §3.8). This module owns the wire format:

- A ``local://artifacts/<name>.pt`` URI resolves to ``<artifacts_root>/<name>.pt``.
- Each ``.pt`` payload is paired with a ``<name>.pt.json`` metadata sidecar
  carrying :class:`CheckpointMetadata` (run_id, seed, step, git_sha,
  pyproject_hash, sha256). The sidecar is the authoritative provenance
  record; the ``.pt`` file is the opaque tensor payload.
- :func:`load_checkpoint` re-hashes the loaded ``.pt`` payload and refuses
  to return tensors when the digest disagrees with the sidecar
  (ADR-002 §Decisions; P6 reproducibility).

Deploy order: **payload first, sidecar second**. A half-finished deploy
(payload landed but sidecar not yet) hits the "missing sidecar" failure
path and refuses to load tensors. The reverse order (sidecar first,
stale payload) would silently load tensors with the wrong provenance —
do not invert.

ADR-009 §Consequences "draft-zoo scoping": the M4a Phase-0 draft-zoo
specs reference ``local://artifacts/mappo_seed42_step100k.pt`` and
``local://artifacts/happo_seed7_step50k.pt`` but the concrete frozen-RL
adapters (T4.5 / T4.6) load them only in M4 Phase 3. Until then the SHA-
verified loader fails loudly when the sidecar is missing — by design.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping
    from pathlib import Path

#: URI scheme used for project-local checkpoint paths (plan/04 §3.8).
#:
#: Phase-0 only. Phase-1 may add ``hf://`` (Hugging Face Hub) and ``s3://``
#: schemes for partner zoo distribution; both require an ADR amendment.
LOCAL_URI_SCHEME: str = "local://"

#: Sidecar filename suffix appended to every payload path (plan/05 §5).
#:
#: A checkpoint at ``artifacts/foo.pt`` carries its provenance + SHA-256
#: digest at ``artifacts/foo.pt.json``. The two-file convention keeps the
#: ``.pt`` payload format-clean (no embedded metadata) and lets external
#: tools inspect provenance without unpickling tensors.
_SIDECAR_SUFFIX: str = ".json"


class CheckpointError(RuntimeError):
    """Loud-fail signal for checkpoint I/O failures (ADR-002 §Decisions).

    Raised on:
    - URI scheme mismatch (only ``local://`` is supported in Phase 0).
    - Missing ``.pt`` payload or missing sidecar JSON.
    - SHA-256 mismatch between the stored sidecar digest and the
      re-computed payload digest on load.

    Subclasses :class:`RuntimeError` so a careless caller catching
    ``Exception`` still sees the failure but can disambiguate via
    ``isinstance(exc, CheckpointError)``.
    """


@dataclass(frozen=True)
class CheckpointMetadata:
    """Provenance + integrity sidecar for one checkpoint (T4b.12; plan/05 §5).

    Mirrors the :class:`~concerto.training.logging.RunContext` provenance
    bundle (run_id / seed / git_sha / pyproject_hash) plus the per-
    checkpoint ``step`` and the integrity-protecting ``sha256`` of the
    ``.pt`` payload. All fields are required so a sidecar with missing
    keys raises :class:`CheckpointError` on load (loud-fail per ADR-002).

    Attributes:
        run_id: 16-hex hash of the run that produced this checkpoint.
            Cross-references the run's JSONL log file via
            :class:`concerto.training.logging.RunContext`.
        seed: Root seed of the producing run; same value passed to
            :func:`concerto.training.seeding.derive_substream`.
        step: Training step at which this checkpoint was taken (e.g.
            50_000 for the canonical "50%-reward" tier in
            ADR-009 §Decision).
        git_sha: Full SHA of the commit the producing run was launched
            from. ``"unknown"`` is permitted (sentinel from
            :class:`concerto.training.logging.GIT_SHA_UNKNOWN`).
        pyproject_hash: SHA-256 of ``pyproject.toml`` at run launch.
            Detects silent dependency drift between save and load.
        sha256: SHA-256 of the raw ``.pt`` payload bytes, computed at
            save time and verified on every load. Mismatch →
            :class:`CheckpointError`.
    """

    run_id: str
    seed: int
    step: int
    git_sha: str
    pyproject_hash: str
    sha256: str

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> CheckpointMetadata:
        """Build a :class:`CheckpointMetadata` from a JSON-decoded mapping (ADR-002 §Decisions).

        Refuses partial sidecars: every field declared on the dataclass
        must be present, and types are coerced (``int``/``str``) without
        loss. Missing fields raise :class:`CheckpointError` so a half-
        written sidecar is caught at load time, not at first inference.

        Args:
            raw: A ``json.load(open(sidecar))``-style dict.

        Returns:
            A frozen :class:`CheckpointMetadata` instance.

        Raises:
            CheckpointError: If any required field is missing.
        """
        required = ("run_id", "seed", "step", "git_sha", "pyproject_hash", "sha256")
        missing = [k for k in required if k not in raw]
        if missing:
            raise CheckpointError(
                f"Checkpoint sidecar is missing required field(s): {missing}; "
                "expected the full provenance bundle (ADR-002 §Decisions)."
            )
        try:
            return cls(
                run_id=str(raw["run_id"]),
                seed=int(raw["seed"]),  # type: ignore[arg-type]
                step=int(raw["step"]),  # type: ignore[arg-type]
                git_sha=str(raw["git_sha"]),
                pyproject_hash=str(raw["pyproject_hash"]),
                sha256=str(raw["sha256"]),
            )
        except (TypeError, ValueError) as exc:
            # Hand-edited sidecar with malformed seed/step (e.g. "50000.0",
            # "abc") would otherwise raise an unwrapped ValueError; wrap so
            # callers catching CheckpointError see all sidecar-failure modes
            # uniformly (ADR-002 §Decisions).
            raise CheckpointError(
                f"Checkpoint sidecar contains malformed integer field(s): {exc}; "
                "expected ``seed`` and ``step`` to coerce to int (ADR-002 §Decisions)."
            ) from exc


def resolve_uri(uri: str, *, artifacts_root: Path) -> Path:
    """Resolve ``local://artifacts/<name>.pt`` URI to filesystem path (ADR-002; plan/04 §3.8).

    Phase-0 supports the ``local://`` scheme only. Non-``local://`` URIs
    raise :class:`CheckpointError` so Phase-1 distribution schemes
    (``hf://``, ``s3://``) cannot silently fall through to the local
    filesystem before an ADR amendment is in place.

    The path immediately under ``local://`` is appended to ``artifacts_root``;
    e.g. ``local://artifacts/foo.pt`` against ``artifacts_root=/data`` resolves
    to ``/data/artifacts/foo.pt``. The leading ``artifacts/`` segment in the
    URI is preserved by design — it makes the URI self-describing in logs
    and keeps the on-disk layout grep-able.

    Args:
        uri: The ``weights_uri`` from a
            :class:`~chamber.partners.api.PartnerSpec` (plan/04 §3.8).
        artifacts_root: Filesystem directory the URI is resolved against.

    Returns:
        Absolute filesystem path to the ``.pt`` payload (the file may
        not exist yet — :func:`save_checkpoint` creates it).

    Raises:
        CheckpointError: If ``uri`` does not start with
            :data:`LOCAL_URI_SCHEME`.
    """
    if not uri.startswith(LOCAL_URI_SCHEME):
        raise CheckpointError(
            f"Unsupported URI scheme: {uri!r}. Phase-0 supports only "
            f"{LOCAL_URI_SCHEME!r} (plan/04 §3.8)."
        )
    relative = uri[len(LOCAL_URI_SCHEME) :]
    return artifacts_root / relative


def _sidecar_path(payload_path: Path) -> Path:
    """Sidecar JSON path for a payload (``foo.pt`` -> ``foo.pt.json``)."""
    return payload_path.with_suffix(payload_path.suffix + _SIDECAR_SUFFIX)


def _sha256_of(path: Path) -> str:
    """SHA-256 hex digest of the file at ``path`` (T4b.12; P6).

    Streams the file in 1-MiB chunks so a 7 GB OpenVLA LoRA checkpoint
    (ADR-010 §Decision) does not have to be fully loaded into memory.
    """
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_checkpoint(
    *,
    state_dict: Mapping[str, torch.Tensor],
    uri: str,
    metadata: CheckpointMetadata,
    artifacts_root: Path,
) -> Path:
    """Save a torch state-dict + sidecar (T4b.12; ADR-002 §Decisions).

    The save is two-phase: write the ``.pt`` payload first, then compute
    its SHA-256, write the sidecar JSON with the digest stitched in. The
    ``sha256`` field on the input ``metadata`` is overwritten with the
    digest computed at save time so callers cannot accidentally ship a
    stale digest that would crash :func:`load_checkpoint`.

    Args:
        state_dict: A ``Mapping[str, torch.Tensor]`` exactly as produced
            by ``model.state_dict()``. Saved verbatim via :func:`torch.save`.
        uri: Destination URI (e.g.
            ``"local://artifacts/happo_seed7_step50k.pt"``).
        metadata: Provenance bundle. The ``sha256`` field is recomputed
            at save time and overrides whatever the caller passed.
        artifacts_root: Filesystem directory the URI resolves against.

    Returns:
        Absolute filesystem path to the written ``.pt`` payload.

    Raises:
        CheckpointError: If ``uri`` is not a ``local://`` URI.
    """
    payload_path = resolve_uri(uri, artifacts_root=artifacts_root)
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state_dict), payload_path)
    digest = _sha256_of(payload_path)
    pinned = CheckpointMetadata(
        run_id=metadata.run_id,
        seed=metadata.seed,
        step=metadata.step,
        git_sha=metadata.git_sha,
        pyproject_hash=metadata.pyproject_hash,
        sha256=digest,
    )
    _sidecar_path(payload_path).write_text(
        json.dumps(asdict(pinned), sort_keys=True, indent=2),
        encoding="utf-8",
    )
    return payload_path


def load_checkpoint(
    *,
    uri: str,
    artifacts_root: Path,
) -> tuple[dict[str, torch.Tensor], CheckpointMetadata]:
    """Load a torch state-dict + verify SHA-256 (T4b.12; ADR-002 §Decisions; P6).

    Refuses to return tensors when:
    - The ``.pt`` payload is missing.
    - The sidecar JSON is missing.
    - The sidecar is missing required fields (per
      :meth:`CheckpointMetadata.from_dict`).
    - The sidecar's stored ``sha256`` disagrees with the recomputed
      digest of the on-disk payload.

    All four failure modes raise :class:`CheckpointError` so a partial
    deployment (e.g. payload deployed but sidecar not yet) cannot
    silently load tensors that the project has no provenance for.

    Args:
        uri: Source URI (e.g.
            ``"local://artifacts/happo_seed7_step50k.pt"``).
        artifacts_root: Filesystem directory the URI resolves against.

    Returns:
        A tuple ``(state_dict, metadata)`` where ``state_dict`` is the
        verbatim mapping loaded from the ``.pt`` payload and
        ``metadata`` is the verified :class:`CheckpointMetadata`.

    Raises:
        CheckpointError: On any of the four loud-fail conditions above.
    """
    payload_path = resolve_uri(uri, artifacts_root=artifacts_root)
    if not payload_path.exists():
        raise CheckpointError(
            f"Checkpoint payload not found at {payload_path} (resolved from {uri!r})."
        )
    sidecar = _sidecar_path(payload_path)
    if not sidecar.exists():
        raise CheckpointError(
            f"Checkpoint sidecar not found at {sidecar}; refusing to load tensors "
            "without a provenance record (ADR-002 §Decisions)."
        )
    metadata = CheckpointMetadata.from_dict(json.loads(sidecar.read_text(encoding="utf-8")))
    actual_digest = _sha256_of(payload_path)
    if actual_digest != metadata.sha256:
        raise CheckpointError(
            f"Checkpoint integrity check failed for {payload_path}: "
            f"sidecar sha256={metadata.sha256!r} but on-disk sha256={actual_digest!r}. "
            "The payload may have been corrupted, partially overwritten, or "
            "swapped without updating the sidecar (ADR-002 §Decisions; P6)."
        )
    # weights_only=True restricts the unpickler to safe tensor primitives,
    # so we do not execute arbitrary code from a tampered .pt file. Pinned
    # explicitly (rather than relying on the default) because the project
    # supports torch>=2.3, and the default flipped to True only at 2.4.
    state_dict = torch.load(
        payload_path,
        weights_only=True,
        map_location="cpu",
    )
    return state_dict, metadata


__all__ = [
    "LOCAL_URI_SCHEME",
    "CheckpointError",
    "CheckpointMetadata",
    "load_checkpoint",
    "resolve_uri",
    "save_checkpoint",
]
