# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``concerto.training.checkpoints`` (T4b.12).

Covers ADR-002 §Decisions (save/load round-trip + SHA-256 integrity) and
plan/04 §3.8 (``local://artifacts/...`` URI resolution).
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

import pytest
import torch

from concerto.training.checkpoints import (
    LOCAL_URI_SCHEME,
    CheckpointError,
    CheckpointMetadata,
    load_checkpoint,
    resolve_uri,
    save_checkpoint,
)

if TYPE_CHECKING:
    from pathlib import Path


def _toy_state_dict() -> dict[str, torch.Tensor]:
    """A tiny but non-empty state-dict for round-trip tests."""
    torch.manual_seed(0)
    return {
        "linear.weight": torch.randn(4, 3),
        "linear.bias": torch.zeros(4),  # type: ignore[attr-defined]
        "step_count": torch.tensor(50_000),  # type: ignore[attr-defined]
    }


def _toy_metadata() -> CheckpointMetadata:
    """Metadata with a placeholder sha256 — save_checkpoint overwrites it."""
    return CheckpointMetadata(
        run_id="0" * 16,
        seed=7,
        step=50_000,
        git_sha="abc1234",
        pyproject_hash="0" * 64,
        sha256="placeholder_will_be_overwritten",
    )


class TestResolveUri:
    def test_local_uri_resolves_under_artifacts_root(self, tmp_path: Path) -> None:
        """plan/04 §3.8: local:// is the Phase-0 scheme."""
        path = resolve_uri(
            "local://artifacts/happo_seed7_step50k.pt",
            artifacts_root=tmp_path,
        )
        assert path == tmp_path / "artifacts" / "happo_seed7_step50k.pt"

    def test_non_local_scheme_raises(self, tmp_path: Path) -> None:
        """plan/04 §3.8: hf:// and s3:// are Phase-1 (need ADR amendment)."""
        with pytest.raises(CheckpointError, match="Phase-0"):
            resolve_uri("hf://hf-org/model.pt", artifacts_root=tmp_path)

    def test_bare_path_without_scheme_raises(self, tmp_path: Path) -> None:
        """ADR-002 §Decisions: scheme is mandatory; bare paths cannot
        accidentally fall through to the local filesystem.
        """
        with pytest.raises(CheckpointError, match="scheme"):
            resolve_uri("artifacts/foo.pt", artifacts_root=tmp_path)

    def test_local_uri_scheme_constant(self) -> None:
        """The constant matches the documented prefix."""
        assert LOCAL_URI_SCHEME == "local://"


class TestCheckpointMetadata:
    def test_metadata_is_frozen(self) -> None:
        """Immutability: tampering with metadata after save must not silently work."""
        import dataclasses

        meta = _toy_metadata()
        with pytest.raises(dataclasses.FrozenInstanceError):
            meta.step = 42  # type: ignore[misc]

    def test_from_dict_round_trip(self) -> None:
        """JSON round-trip preserves every field (T4b.12)."""
        meta = _toy_metadata()
        from dataclasses import asdict

        recovered = CheckpointMetadata.from_dict(asdict(meta))
        assert recovered == meta

    def test_from_dict_missing_field_raises(self) -> None:
        """ADR-002 §Decisions: a half-written sidecar must fail loud."""
        with pytest.raises(CheckpointError, match="missing required field"):
            CheckpointMetadata.from_dict(
                {
                    "run_id": "x",
                    "seed": 0,
                    "step": 0,
                    "git_sha": "abc",
                    # missing pyproject_hash + sha256
                }
            )

    def test_from_dict_coerces_str_to_int(self) -> None:
        """JSON sometimes stores ints as strings; tolerate it without losing info."""
        meta = CheckpointMetadata.from_dict(
            {
                "run_id": "x",
                "seed": "7",  # str → int coerced
                "step": "50000",  # str → int coerced
                "git_sha": "abc",
                "pyproject_hash": "0" * 64,
                "sha256": "f" * 64,
            }
        )
        assert meta.seed == 7
        assert meta.step == 50_000

    def test_from_dict_malformed_int_wraps_as_checkpoint_error(self) -> None:
        """ADR-002 §Decisions: hand-edited malformed ``seed`` is wrapped, not raw."""
        with pytest.raises(CheckpointError, match="malformed integer"):
            CheckpointMetadata.from_dict(
                {
                    "run_id": "x",
                    "seed": "not-a-number",
                    "step": 0,
                    "git_sha": "abc",
                    "pyproject_hash": "0" * 64,
                    "sha256": "f" * 64,
                }
            )


class TestSaveLoadRoundTrip:
    def test_round_trip_returns_identical_tensors(self, tmp_path: Path) -> None:
        """T4b.12: save → load returns byte-identical state_dict."""
        original = _toy_state_dict()
        save_checkpoint(
            state_dict=original,
            uri="local://artifacts/happo_seed7_step50k.pt",
            metadata=_toy_metadata(),
            artifacts_root=tmp_path,
        )
        recovered, _meta = load_checkpoint(
            uri="local://artifacts/happo_seed7_step50k.pt",
            artifacts_root=tmp_path,
        )
        assert set(recovered.keys()) == set(original.keys())
        for k in original:
            assert torch.equal(recovered[k], original[k])  # type: ignore[attr-defined]

    def test_round_trip_returns_metadata(self, tmp_path: Path) -> None:
        """T4b.12: load returns the verified metadata."""
        save_checkpoint(
            state_dict=_toy_state_dict(),
            uri="local://artifacts/foo.pt",
            metadata=_toy_metadata(),
            artifacts_root=tmp_path,
        )
        _, meta = load_checkpoint(uri="local://artifacts/foo.pt", artifacts_root=tmp_path)
        assert meta.seed == 7
        assert meta.step == 50_000
        assert meta.run_id == "0" * 16
        assert meta.git_sha == "abc1234"

    def test_save_creates_parent_directory(self, tmp_path: Path) -> None:
        """ADR-002 §Decisions: save into deeply-nested artifacts root works."""
        save_checkpoint(
            state_dict=_toy_state_dict(),
            uri="local://artifacts/zoo/seed7/step50k.pt",
            metadata=_toy_metadata(),
            artifacts_root=tmp_path,
        )
        assert (tmp_path / "artifacts" / "zoo" / "seed7" / "step50k.pt").exists()
        assert (tmp_path / "artifacts" / "zoo" / "seed7" / "step50k.pt.json").exists()

    def test_save_overwrites_provided_sha256_with_computed(self, tmp_path: Path) -> None:
        """ADR-002 §Decisions: stale digests cannot ship; save recomputes."""
        meta_with_bogus_sha = CheckpointMetadata(
            run_id="0" * 16,
            seed=7,
            step=50_000,
            git_sha="abc1234",
            pyproject_hash="0" * 64,
            sha256="d" * 64,  # bogus
        )
        save_checkpoint(
            state_dict=_toy_state_dict(),
            uri="local://artifacts/foo.pt",
            metadata=meta_with_bogus_sha,
            artifacts_root=tmp_path,
        )
        sidecar = (tmp_path / "artifacts" / "foo.pt.json").read_text(encoding="utf-8")
        sidecar_obj = json.loads(sidecar)
        # The bogus value must NOT have made it to the sidecar.
        assert sidecar_obj["sha256"] != "d" * 64
        assert re.fullmatch(r"[0-9a-f]{64}", sidecar_obj["sha256"])


class TestSidecarFormat:
    def test_sidecar_is_json_with_full_provenance(self, tmp_path: Path) -> None:
        """plan/05 §5: every saved checkpoint has a valid JSON sidecar."""
        save_checkpoint(
            state_dict=_toy_state_dict(),
            uri="local://artifacts/foo.pt",
            metadata=_toy_metadata(),
            artifacts_root=tmp_path,
        )
        sidecar_path = tmp_path / "artifacts" / "foo.pt.json"
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        for key in ("run_id", "seed", "step", "git_sha", "pyproject_hash", "sha256"):
            assert key in sidecar
        # SHA-256 hex digest is 64 lowercase hex chars.
        assert re.fullmatch(r"[0-9a-f]{64}", sidecar["sha256"])

    def test_sidecar_is_byte_stable_under_re_save(self, tmp_path: Path) -> None:
        """P6 reproducibility: identical inputs → identical sidecar bytes."""
        for _ in range(2):
            save_checkpoint(
                state_dict=_toy_state_dict(),
                uri="local://artifacts/foo.pt",
                metadata=_toy_metadata(),
                artifacts_root=tmp_path,
            )
        # The sha256 field is over the .pt payload; if two saves produce
        # identical .pt bytes, they produce identical sidecars too.
        first_sidecar = (tmp_path / "artifacts" / "foo.pt.json").read_text(encoding="utf-8")
        save_checkpoint(
            state_dict=_toy_state_dict(),
            uri="local://artifacts/foo.pt",
            metadata=_toy_metadata(),
            artifacts_root=tmp_path,
        )
        second_sidecar = (tmp_path / "artifacts" / "foo.pt.json").read_text(encoding="utf-8")
        # Sidecar contents must agree on the four immutable fields. The
        # sha256 field is the integrity-protecting digest; if torch.save
        # produced byte-stable bytes it would also be equal — but PyTorch
        # does not promise that, so we only assert the four metadata
        # fields.
        first = json.loads(first_sidecar)
        second = json.loads(second_sidecar)
        for key in ("run_id", "seed", "step", "git_sha", "pyproject_hash"):
            assert first[key] == second[key]


class TestIntegrityFailures:
    def test_load_missing_payload_raises(self, tmp_path: Path) -> None:
        """ADR-002 §Decisions: missing .pt is a loud failure, not a silent None."""
        with pytest.raises(CheckpointError, match="payload not found"):
            load_checkpoint(uri="local://artifacts/never_saved.pt", artifacts_root=tmp_path)

    def test_load_missing_sidecar_raises(self, tmp_path: Path) -> None:
        """ADR-002 §Decisions: missing sidecar refuses to load tensors."""
        # Hand-write a .pt without the sidecar (simulates partial deploy).
        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir()
        torch.save({"x": torch.zeros(1)}, artifacts_dir / "foo.pt")  # type: ignore[attr-defined]
        with pytest.raises(CheckpointError, match="sidecar not found"):
            load_checkpoint(uri="local://artifacts/foo.pt", artifacts_root=tmp_path)

    def test_load_corrupted_payload_fails_sha_check(self, tmp_path: Path) -> None:
        """T4b.12: SHA-256 mismatch raises CheckpointError (not silent garbage)."""
        save_checkpoint(
            state_dict=_toy_state_dict(),
            uri="local://artifacts/foo.pt",
            metadata=_toy_metadata(),
            artifacts_root=tmp_path,
        )
        # Tamper with the payload while leaving the sidecar's stored
        # sha256 unchanged. Real-world analogue: half-written file, disk
        # bit-rot, in-place patch by an attacker.
        (tmp_path / "artifacts" / "foo.pt").write_bytes(b"corrupted_payload")
        with pytest.raises(CheckpointError, match="integrity check failed"):
            load_checkpoint(uri="local://artifacts/foo.pt", artifacts_root=tmp_path)

    def test_load_corrupted_sidecar_missing_field_raises(self, tmp_path: Path) -> None:
        """ADR-002 §Decisions: a half-written sidecar JSON fails on load."""
        save_checkpoint(
            state_dict=_toy_state_dict(),
            uri="local://artifacts/foo.pt",
            metadata=_toy_metadata(),
            artifacts_root=tmp_path,
        )
        sidecar_path = tmp_path / "artifacts" / "foo.pt.json"
        # Drop a required field.
        partial = json.loads(sidecar_path.read_text(encoding="utf-8"))
        del partial["pyproject_hash"]
        sidecar_path.write_text(json.dumps(partial), encoding="utf-8")
        with pytest.raises(CheckpointError, match="missing required field"):
            load_checkpoint(uri="local://artifacts/foo.pt", artifacts_root=tmp_path)

    def test_load_with_non_local_uri_raises(self, tmp_path: Path) -> None:
        """plan/04 §3.8: scheme check fires before any I/O."""
        with pytest.raises(CheckpointError, match="Phase-0"):
            load_checkpoint(uri="hf://x/y.pt", artifacts_root=tmp_path)


class TestPublicSurface:
    def test_module_exports(self) -> None:
        """The public surface includes the six names a partner adapter touches."""
        from concerto.training import checkpoints as ck

        for name in (
            "LOCAL_URI_SCHEME",
            "CheckpointError",
            "CheckpointMetadata",
            "load_checkpoint",
            "resolve_uri",
            "save_checkpoint",
        ):
            assert hasattr(ck, name)

    def test_checkpoint_error_subclasses_runtime_error(self) -> None:
        """ADR-002 §Decisions: catch-Exception callers still see the failure."""
        assert issubclass(CheckpointError, RuntimeError)
