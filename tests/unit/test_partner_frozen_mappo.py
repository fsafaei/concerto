# SPDX-License-Identifier: Apache-2.0
# pyright: reportPrivateImportUsage=false
"""Unit tests for ``chamber.partners.frozen_mappo`` (T4.5).

Covers ADR-009 §Decision (frozen-partner contract; ``requires_grad=False``
on every parameter; ``torch.no_grad`` inference) and plan/04 §3.5 (Phase-0
shared-parameter MAPPO actor; URI + uid contract).

The Phase-0 MAPPO checkpoint produced by an M4b training run is *not* a
prerequisite for this test suite: every test stages a tmp_path fixture
via :func:`concerto.training.checkpoints.save_checkpoint` so the suite
stays hermetic on a Mac development host (plan/04 §8 Notes: "If M4a is
being implemented before M4b finishes, ship with stub ``.pt`` files
containing one randomly-initialised network and document this in the
test fixture.").
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from chamber.partners.api import PartnerSpec
from chamber.partners.frozen_mappo import (
    _FC1_WEIGHT_KEY,
    FrozenMAPPOPartner,
    _MAPPOActor,
)
from concerto.training.checkpoints import (
    CheckpointError,
    CheckpointMetadata,
    save_checkpoint,
)

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

_OBS_DIM = 4
_HIDDEN_DIM = 8
_ACTION_DIM = 3
_URI = "local://artifacts/mappo_test.pt"


def _make_actor_state_dict(*, seed: int = 0) -> dict[str, torch.Tensor]:
    """Build a deterministic state-dict for a :class:`_MAPPOActor`."""
    torch.manual_seed(seed)
    actor = _MAPPOActor(obs_dim=_OBS_DIM, hidden_dim=_HIDDEN_DIM, action_dim=_ACTION_DIM)
    return {name: tensor.clone().detach() for name, tensor in actor.state_dict().items()}


def _stage_checkpoint(
    *,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    state_dict: dict[str, torch.Tensor] | None = None,
    payload_override: dict[str, object] | None = None,
) -> Path:
    """Write a fixture .pt + sidecar under tmp_path and point the env var at it.

    Args:
        tmp_path: Pytest tmp_path fixture for the artefact root.
        monkeypatch: Pytest monkeypatch fixture for the env var.
        state_dict: Actor tensors to wrap under the canonical
            ``{"actor": ...}`` layout. Mutually exclusive with
            ``payload_override``.
        payload_override: Drop-in payload dict bypassing the canonical
            nesting — used by malformed-layout tests to verify the
            error path. Mutually exclusive with ``state_dict``.
    """
    if payload_override is not None:
        payload: dict[str, object] = payload_override
    else:
        if state_dict is None:
            state_dict = _make_actor_state_dict()
        payload = {"actor": dict(state_dict)}
    metadata = CheckpointMetadata(
        run_id="0" * 16,
        seed=42,
        step=100_000,
        git_sha="abc1234",
        pyproject_hash="0" * 64,
        sha256="placeholder",
    )
    path = save_checkpoint(
        state_dict=payload,  # type: ignore[arg-type]
        uri=_URI,
        metadata=metadata,
        artifacts_root=tmp_path,
    )
    monkeypatch.setenv("CONCERTO_ARTIFACTS_ROOT", str(tmp_path))
    return path


def _spec(**extra: str) -> PartnerSpec:
    extra_full = {"uid": "panda_wristcam", "task": "stage0_smoke"}
    extra_full.update(extra)
    return PartnerSpec(
        class_name="frozen_mappo",
        seed=42,
        checkpoint_step=100_000,
        weights_uri=_URI,
        extra=extra_full,
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_construction_is_io_free(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """plan/04 §3.3: construction must not touch disk; load is deferred to reset/act."""
        monkeypatch.setenv("CONCERTO_ARTIFACTS_ROOT", str(tmp_path))
        partner = FrozenMAPPOPartner(_spec())
        # No file staged yet — but construction succeeded, proving no I/O happened.
        assert partner.spec.weights_uri == _URI

    def test_missing_weights_uri_raises(self) -> None:
        """ADR-009 §Decision: a frozen-RL partner needs a checkpoint URI."""
        spec = PartnerSpec(
            class_name="frozen_mappo",
            seed=0,
            checkpoint_step=None,
            weights_uri=None,
            extra={"uid": "panda_wristcam"},
        )
        with pytest.raises(ValueError, match="weights_uri"):
            FrozenMAPPOPartner(spec)

    def test_missing_uid_raises(self) -> None:
        """plan/04 §3.5: spec.extra['uid'] is the env-side agent uid."""
        spec = PartnerSpec(
            class_name="frozen_mappo",
            seed=0,
            checkpoint_step=100_000,
            weights_uri=_URI,
            extra={},
        )
        with pytest.raises(ValueError, match="uid"):
            FrozenMAPPOPartner(spec)


# ---------------------------------------------------------------------------
# Freezing contract
# ---------------------------------------------------------------------------


class TestFreezeContract:
    def test_all_parameters_have_requires_grad_false(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ADR-009 §Decision: every parameter is frozen post-load."""
        _stage_checkpoint(tmp_path=tmp_path, monkeypatch=monkeypatch)
        partner = FrozenMAPPOPartner(_spec())
        partner.reset(seed=0)
        actor = partner._ensure_loaded()
        for name, param in actor.named_parameters():
            assert param.requires_grad is False, f"parameter {name} still requires grad"

    def test_actor_in_eval_mode(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Eval mode pins dropout/batchnorm; the MLP has neither but the contract is general."""
        _stage_checkpoint(tmp_path=tmp_path, monkeypatch=monkeypatch)
        partner = FrozenMAPPOPartner(_spec())
        partner.reset(seed=0)
        actor = partner._ensure_loaded()
        assert actor.training is False

    def test_forbidden_attrs_shield_still_bites(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ADR-009 §Consequences: the runtime shield from PartnerBase still applies."""
        _stage_checkpoint(tmp_path=tmp_path, monkeypatch=monkeypatch)
        partner = FrozenMAPPOPartner(_spec())
        # Attribute lookups for forbidden names hit PartnerBase.__getattr__ since
        # FrozenMAPPOPartner does not define them locally.
        for forbidden in ("train", "learn", "update", "fit", "set_weights"):
            with pytest.raises(AttributeError, match="ADR-009"):
                getattr(partner, forbidden)


# ---------------------------------------------------------------------------
# Determinism + shape contract
# ---------------------------------------------------------------------------


class TestActionContract:
    def test_action_shape_matches_head(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """plan/04 §3.5: action vector length = inferred action_dim."""
        _stage_checkpoint(tmp_path=tmp_path, monkeypatch=monkeypatch)
        partner = FrozenMAPPOPartner(_spec())
        obs = {"agent": {"panda_wristcam": {"state": np.zeros(_OBS_DIM, dtype=np.float32)}}}
        action = partner.act(obs)
        assert action.shape == (_ACTION_DIM,)

    def test_action_dtype_is_float32(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """plan/04 §3.5: actions are float32 to match Box action spaces."""
        _stage_checkpoint(tmp_path=tmp_path, monkeypatch=monkeypatch)
        partner = FrozenMAPPOPartner(_spec())
        obs = {"agent": {"panda_wristcam": {"state": np.zeros(_OBS_DIM, dtype=np.float32)}}}
        action = partner.act(obs)
        assert action.dtype == np.float32

    def test_two_calls_same_obs_identical(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """P6 + ADR-009 §Decision: deterministic head + frozen weights → identical output."""
        _stage_checkpoint(tmp_path=tmp_path, monkeypatch=monkeypatch)
        partner = FrozenMAPPOPartner(_spec())
        state = np.linspace(-1.0, 1.0, _OBS_DIM, dtype=np.float32)
        obs = {"agent": {"panda_wristcam": {"state": state}}}
        a = partner.act(obs)
        b = partner.act(obs)
        np.testing.assert_array_equal(a, b)

    def test_deterministic_kwarg_ignored(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The Phase-0 head is a deterministic linear layer; deterministic kwarg is a no-op."""
        _stage_checkpoint(tmp_path=tmp_path, monkeypatch=monkeypatch)
        partner = FrozenMAPPOPartner(_spec())
        obs = {"agent": {"panda_wristcam": {"state": np.ones(_OBS_DIM, dtype=np.float32)}}}
        a = partner.act(obs, deterministic=True)
        b = partner.act(obs, deterministic=False)
        np.testing.assert_array_equal(a, b)

    def test_seed_change_in_reset_does_not_change_output(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """plan/04 §2: partner is stateless across episodes; reset(seed) is a no-op."""
        _stage_checkpoint(tmp_path=tmp_path, monkeypatch=monkeypatch)
        partner = FrozenMAPPOPartner(_spec())
        obs = {"agent": {"panda_wristcam": {"state": np.ones(_OBS_DIM, dtype=np.float32)}}}
        partner.reset(seed=0)
        a = partner.act(obs)
        partner.reset(seed=12345)
        b = partner.act(obs)
        np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# Load failure modes
# ---------------------------------------------------------------------------


class TestLoadFailures:
    def test_missing_payload_raises_checkpoint_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ADR-002 §Decisions: missing .pt is a loud failure."""
        monkeypatch.setenv("CONCERTO_ARTIFACTS_ROOT", str(tmp_path))
        partner = FrozenMAPPOPartner(_spec())
        with pytest.raises(CheckpointError, match="not found"):
            partner.reset(seed=0)

    def test_wrong_state_dict_layout_raises_value_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """plan/04 §3.5: unrecognised layer keys → ValueError, not silent fall-through."""
        bogus = {"alien.weight": torch.zeros(2, 2), "alien.bias": torch.zeros(2)}
        _stage_checkpoint(tmp_path=tmp_path, monkeypatch=monkeypatch, state_dict=bogus)
        partner = FrozenMAPPOPartner(_spec())
        with pytest.raises(ValueError, match="fc1/fc2/head"):
            partner.reset(seed=0)

    def test_obs_shape_mismatch_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """plan/04 §3.5: obs state dim must match the inferred actor input dim."""
        _stage_checkpoint(tmp_path=tmp_path, monkeypatch=monkeypatch)
        partner = FrozenMAPPOPartner(_spec())
        obs = {"agent": {"panda_wristcam": {"state": np.zeros(_OBS_DIM + 2, dtype=np.float32)}}}
        with pytest.raises(ValueError, match="fc1 input dim"):
            partner.act(obs)

    def test_missing_agent_uid_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """plan/04 §3.5: a partner without its uid in obs cannot act."""
        _stage_checkpoint(tmp_path=tmp_path, monkeypatch=monkeypatch)
        partner = FrozenMAPPOPartner(_spec())
        with pytest.raises(ValueError, match="panda_wristcam"):
            partner.act({"agent": {}})


# ---------------------------------------------------------------------------
# Canonical layout (nested under "actor")
# ---------------------------------------------------------------------------


class TestCanonicalLayout:
    def test_nested_state_dict_layout_loads(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """plan/04 §3.5: ``{"actor": ...}`` is the canonical (and only) wire format."""
        _stage_checkpoint(tmp_path=tmp_path, monkeypatch=monkeypatch)
        partner = FrozenMAPPOPartner(_spec())
        obs = {"agent": {"panda_wristcam": {"state": np.zeros(_OBS_DIM, dtype=np.float32)}}}
        action = partner.act(obs)
        assert action.shape == (_ACTION_DIM,)

    def test_missing_actor_key_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """plan/04 §3.5: a .pt without an ``"actor"`` key is rejected loudly."""
        flat = _make_actor_state_dict()
        _stage_checkpoint(
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
            payload_override=dict(flat),
        )
        partner = FrozenMAPPOPartner(_spec())
        with pytest.raises(ValueError, match="'actor'"):
            partner.reset(seed=0)

    def test_actor_key_not_dict_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """plan/04 §3.5: ``loaded["actor"]`` must be a dict, not a bare tensor."""
        _stage_checkpoint(
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
            payload_override={"actor": torch.zeros(2, 2)},
        )
        partner = FrozenMAPPOPartner(_spec())
        with pytest.raises(ValueError, match="'actor'"):
            partner.reset(seed=0)

    def test_non_tensor_layer_value_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """plan/04 §3.5: a non-tensor value at a layer key is flagged with the right error."""
        good = _make_actor_state_dict()
        tampered = dict(good)
        tampered[_FC1_WEIGHT_KEY] = "not-a-tensor"  # type: ignore[assignment]
        _stage_checkpoint(
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
            payload_override={"actor": tampered},
        )
        partner = FrozenMAPPOPartner(_spec())
        with pytest.raises(ValueError, match=r"must be torch\.Tensor"):
            partner.reset(seed=0)


# ---------------------------------------------------------------------------
# Targeted error-path coverage (plan/04 §3.5 mismatched-shape branches)
# ---------------------------------------------------------------------------


class TestShapeMismatchErrors:
    """Each test triggers a specific shape-disagreement branch in the actor builder."""

    def test_fc2_shape_disagrees_with_hidden_dim_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """plan/04 §3.5: fc2 must be ``(hidden_dim, hidden_dim)``."""
        bad = _make_actor_state_dict()
        bad["fc2.weight"] = torch.zeros(_HIDDEN_DIM + 1, _HIDDEN_DIM)
        bad["fc2.bias"] = torch.zeros(_HIDDEN_DIM + 1)
        _stage_checkpoint(tmp_path=tmp_path, monkeypatch=monkeypatch, state_dict=bad)
        partner = FrozenMAPPOPartner(_spec())
        with pytest.raises(ValueError, match="fc2 shape"):
            partner.reset(seed=0)

    def test_head_input_dim_disagrees_with_hidden_dim_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """plan/04 §3.5: head's input dim must match the inferred hidden_dim."""
        bad = _make_actor_state_dict()
        bad["head.weight"] = torch.zeros(_ACTION_DIM, _HIDDEN_DIM + 2)
        bad["head.bias"] = torch.zeros(_ACTION_DIM)
        _stage_checkpoint(tmp_path=tmp_path, monkeypatch=monkeypatch, state_dict=bad)
        partner = FrozenMAPPOPartner(_spec())
        with pytest.raises(ValueError, match="head input dim"):
            partner.reset(seed=0)

    def test_load_state_dict_runtime_error_remapped_to_value_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ADR-002 §Decisions: torch RuntimeError is wrapped so callers see one error type."""
        bad = _make_actor_state_dict()
        # Drop fc1.bias to force a load_state_dict missing-key RuntimeError.
        del bad["fc1.bias"]
        _stage_checkpoint(tmp_path=tmp_path, monkeypatch=monkeypatch, state_dict=bad)
        partner = FrozenMAPPOPartner(_spec())
        with pytest.raises(ValueError, match="load_state_dict failed"):
            partner.reset(seed=0)


class TestObservationErrors:
    """Each test triggers a specific malformed-obs branch in ``_read_flat_state``."""

    def test_missing_agent_key_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """plan/04 §3.4: obs without ``'agent'`` cannot drive the partner."""
        _stage_checkpoint(tmp_path=tmp_path, monkeypatch=monkeypatch)
        partner = FrozenMAPPOPartner(_spec())
        with pytest.raises(ValueError, match="panda_wristcam"):
            partner.act({})

    def test_uid_entry_missing_state_key_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """plan/04 §3.4: ``obs['agent'][uid]`` must carry a ``'state'`` channel."""
        _stage_checkpoint(tmp_path=tmp_path, monkeypatch=monkeypatch)
        partner = FrozenMAPPOPartner(_spec())
        with pytest.raises(ValueError, match="state"):
            partner.act({"agent": {"panda_wristcam": {}}})

    def test_state_not_1d_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """plan/04 §3.4: state must be 1-D; 2-D inputs are a fixture bug, not silent flatten."""
        _stage_checkpoint(tmp_path=tmp_path, monkeypatch=monkeypatch)
        partner = FrozenMAPPOPartner(_spec())
        state = np.zeros((_OBS_DIM, 2), dtype=np.float32)
        with pytest.raises(ValueError, match="1-D"):
            partner.act({"agent": {"panda_wristcam": {"state": state}}})
