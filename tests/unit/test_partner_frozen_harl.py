# SPDX-License-Identifier: Apache-2.0
# pyright: reportPrivateImportUsage=false
"""Unit tests for ``chamber.partners.frozen_harl`` (T4.6).

Covers ADR-009 §Decision (frozen-partner contract; ``requires_grad=False``
on every parameter; ``torch.no_grad`` inference) and plan/04 §3.6 (the
HARL HAPPO checkpoint adapter shape).

The Phase-0 zoo-seed checkpoint produced by an M4b training run is *not* a
prerequisite for this unit suite: every test stages a tmp_path fixture
via :func:`concerto.training.checkpoints.save_checkpoint` over a freshly-
constructed HARL :class:`StochasticPolicy` (plan/04 §8 Notes — same
rationale as the FrozenMAPPOPartner tests). The :func:`B4 reproduction
test <tests.reproduction.test_zoo_seed_artifact>` is the one that pins
behaviour against the real GPU-produced artefact.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import gymnasium as gym
import numpy as np
import pytest
import torch

# HARL is scoped to the ``[dependency-groups].train`` PEP 735 group
# (ADR-002 §Revision-history 2026-05-16; #131; pyproject.toml). Skip the
# whole module if the group is not installed. ``pytest.importorskip``
# returns the imported module; the module-level binding below mirrors
# the original ``from`` import so test bodies do not need to change.
_harl_stochastic_policy = pytest.importorskip(
    "harl.models.policy_models.stochastic_policy",
    reason="HARL not installed; run `uv sync --group train` to enable HARL tests.",
)
StochasticPolicy = _harl_stochastic_policy.StochasticPolicy

from chamber.partners.api import PartnerSpec  # noqa: E402
from chamber.partners.frozen_harl import (  # noqa: E402
    _HARL_INFERENCE_ARGS,
    FrozenHARLPartner,
)
from concerto.training.checkpoints import (  # noqa: E402
    CheckpointError,
    CheckpointMetadata,
    save_checkpoint,
)

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

_OBS_DIM = 6
_HIDDEN_DIM = 16
_ACTION_DIM = 4
_URI = "local://artifacts/harl_test.pt"


def _build_args(*, hidden_dim: int = _HIDDEN_DIM) -> dict[str, object]:
    args = dict(_HARL_INFERENCE_ARGS)
    args["hidden_sizes"] = [hidden_dim, hidden_dim]
    return args


def _build_actor(
    *,
    obs_dim: int = _OBS_DIM,
    hidden_dim: int = _HIDDEN_DIM,
    action_dim: int = _ACTION_DIM,
) -> StochasticPolicy:
    """Construct a real HARL StochasticPolicy for the fixture."""
    args = _build_args(hidden_dim=hidden_dim)
    obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
    act_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)
    return StochasticPolicy(args, obs_space, act_space, torch.device("cpu"))


def _make_actor_state_dict(*, seed: int = 0) -> dict[str, torch.Tensor]:
    """Deterministic HARL StochasticPolicy state-dict for the fixture."""
    torch.manual_seed(seed)
    actor = _build_actor()
    return {name: tensor.clone().detach() for name, tensor in actor.state_dict().items()}


def _stage_checkpoint(
    *,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    state_dict: dict[str, torch.Tensor] | None = None,
    payload_override: dict[str, object] | None = None,
) -> Path:
    """Write a fixture .pt + sidecar and point the env var at it."""
    if payload_override is not None:
        payload: dict[str, object] = payload_override
    else:
        if state_dict is None:
            state_dict = _make_actor_state_dict()
        payload = {"actor": dict(state_dict)}
    metadata = CheckpointMetadata(
        run_id="0" * 16,
        seed=7,
        step=50_000,
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
    extra_full = {"uid": "allegro_hand_right", "task": "stage0_smoke"}
    extra_full.update(extra)
    return PartnerSpec(
        class_name="frozen_harl",
        seed=7,
        checkpoint_step=50_000,
        weights_uri=_URI,
        extra=extra_full,
    )


def _obs(state: NDArray[np.floating] | None = None) -> dict[str, object]:
    if state is None:
        state = np.zeros(_OBS_DIM, dtype=np.float32)
    return {"agent": {"allegro_hand_right": {"state": state}}}


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_construction_is_io_free(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """plan/04 §3.3: construction must not touch disk; load is deferred to reset/act."""
        monkeypatch.setenv("CONCERTO_ARTIFACTS_ROOT", str(tmp_path))
        partner = FrozenHARLPartner(_spec())
        assert partner.spec.weights_uri == _URI

    def test_missing_weights_uri_raises(self) -> None:
        """ADR-009 §Decision: a frozen-RL partner needs a checkpoint URI."""
        spec = PartnerSpec(
            class_name="frozen_harl",
            seed=0,
            checkpoint_step=None,
            weights_uri=None,
            extra={"uid": "allegro_hand_right"},
        )
        with pytest.raises(ValueError, match="weights_uri"):
            FrozenHARLPartner(spec)

    def test_missing_uid_raises(self) -> None:
        """plan/04 §3.6: spec.extra['uid'] is the env-side agent uid."""
        spec = PartnerSpec(
            class_name="frozen_harl",
            seed=0,
            checkpoint_step=50_000,
            weights_uri=_URI,
            extra={},
        )
        with pytest.raises(ValueError, match="uid"):
            FrozenHARLPartner(spec)


# ---------------------------------------------------------------------------
# Freezing contract — the test the prompt §4 names explicitly
# ---------------------------------------------------------------------------


class TestFreezeContract:
    def test_all_parameters_have_requires_grad_false(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ADR-009 §Decision: every parameter is frozen post-load."""
        _stage_checkpoint(tmp_path=tmp_path, monkeypatch=monkeypatch)
        partner = FrozenHARLPartner(_spec())
        partner.reset(seed=0)
        actor = partner._ensure_loaded()
        params = list(actor.named_parameters())
        assert params, "actor must expose at least one parameter after load"
        for name, param in params:
            assert param.requires_grad is False, f"parameter {name} still requires grad"

    def test_actor_in_eval_mode(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Eval mode pins LayerNorm / Dropout behaviour; HARL's MLPBase uses LayerNorm."""
        _stage_checkpoint(tmp_path=tmp_path, monkeypatch=monkeypatch)
        partner = FrozenHARLPartner(_spec())
        partner.reset(seed=0)
        actor = partner._ensure_loaded()
        assert actor.training is False

    def test_forbidden_attrs_shield_still_bites(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ADR-009 §Consequences: the runtime shield from PartnerBase still applies."""
        _stage_checkpoint(tmp_path=tmp_path, monkeypatch=monkeypatch)
        partner = FrozenHARLPartner(_spec())
        for forbidden in ("train", "learn", "update", "fit", "set_weights"):
            with pytest.raises(AttributeError, match="ADR-009"):
                getattr(partner, forbidden)


# ---------------------------------------------------------------------------
# Action contract — finite, expected shape, deterministic
# ---------------------------------------------------------------------------


class TestActionContract:
    def test_action_shape_matches_head(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """plan/04 §3.6: action vector length = inferred action_dim."""
        _stage_checkpoint(tmp_path=tmp_path, monkeypatch=monkeypatch)
        partner = FrozenHARLPartner(_spec())
        action = partner.act(_obs())
        assert action.shape == (_ACTION_DIM,)

    def test_action_dtype_is_float32(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """plan/04 §3.6: actions are float32 to match Box action spaces."""
        _stage_checkpoint(tmp_path=tmp_path, monkeypatch=monkeypatch)
        partner = FrozenHARLPartner(_spec())
        action = partner.act(_obs())
        assert action.dtype == np.float32

    def test_action_is_finite(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """ADR-009 §Decision: the action vector must be finite (no NaN/inf)."""
        _stage_checkpoint(tmp_path=tmp_path, monkeypatch=monkeypatch)
        partner = FrozenHARLPartner(_spec())
        state = np.linspace(-2.0, 2.0, _OBS_DIM, dtype=np.float32)
        action = partner.act(_obs(state=state))
        assert np.all(np.isfinite(action))

    def test_two_calls_same_obs_identical(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """P6 + ADR-009 §Decision: deterministic mode + frozen weights → identical output."""
        _stage_checkpoint(tmp_path=tmp_path, monkeypatch=monkeypatch)
        partner = FrozenHARLPartner(_spec())
        state = np.linspace(-1.0, 1.0, _OBS_DIM, dtype=np.float32)
        a = partner.act(_obs(state=state))
        b = partner.act(_obs(state=state))
        np.testing.assert_array_equal(a, b)

    def test_deterministic_kwarg_does_not_sample(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Phase-0 P6: the partner always returns the distribution mode regardless of the kwarg.

        DiagGaussian.sample() reads torch's global RNG; honouring
        ``deterministic=False`` would couple a frozen partner's actions to
        whatever else in the process touched the global generator. Pin the
        deterministic-only behaviour so a future regression surfaces here.
        """
        _stage_checkpoint(tmp_path=tmp_path, monkeypatch=monkeypatch)
        partner = FrozenHARLPartner(_spec())
        obs = _obs(state=np.ones(_OBS_DIM, dtype=np.float32))
        a = partner.act(obs, deterministic=True)
        b = partner.act(obs, deterministic=False)
        np.testing.assert_array_equal(a, b)

    def test_action_matches_actor_mean_head(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Equivalence check: partner.act(obs) matches the reference actor on the same weights.

        Independent reference implementation: build a fresh HARL actor with
        the same args + weights, run its forward pass directly, and confirm
        the partner returns the same tensor. Catches any silent batching /
        squeezing / dtype drift in the partner's inference path.
        """
        sd = _make_actor_state_dict()
        _stage_checkpoint(tmp_path=tmp_path, monkeypatch=monkeypatch, state_dict=sd)
        partner = FrozenHARLPartner(_spec())
        state = np.linspace(-0.7, 0.7, _OBS_DIM, dtype=np.float32)
        action = partner.act(_obs(state=state))

        ref_actor = _build_actor()
        ref_actor.load_state_dict(sd)
        ref_actor.eval()
        obs_t = torch.from_numpy(state).reshape(1, -1)
        rnn_states = np.zeros((1, 1, _HIDDEN_DIM), dtype=np.float32)
        masks = np.ones((1, 1), dtype=np.float32)
        with torch.no_grad():
            ref_action_t, _, _ = ref_actor(obs_t, rnn_states, masks, None, True)
        ref_action = ref_action_t.detach().cpu().numpy().squeeze(0).astype(np.float32)
        np.testing.assert_allclose(action, ref_action, atol=1e-6)

    def test_seed_change_in_reset_does_not_change_output(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """plan/04 §2: partner is stateless across episodes; reset(seed) is a no-op."""
        _stage_checkpoint(tmp_path=tmp_path, monkeypatch=monkeypatch)
        partner = FrozenHARLPartner(_spec())
        obs = _obs(state=np.ones(_OBS_DIM, dtype=np.float32))
        partner.reset(seed=0)
        a = partner.act(obs)
        partner.reset(seed=99999)
        b = partner.act(obs)
        np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# Load failure modes
# ---------------------------------------------------------------------------


class TestLoadFailures:
    def test_missing_payload_raises_checkpoint_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ADR-002 §Decisions: a missing .pt is a loud failure."""
        monkeypatch.setenv("CONCERTO_ARTIFACTS_ROOT", str(tmp_path))
        partner = FrozenHARLPartner(_spec())
        with pytest.raises(CheckpointError, match="not found"):
            partner.reset(seed=0)

    def test_missing_actor_key_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """plan/04 §3.6: a .pt without an ``"actor"`` key is rejected loudly."""
        flat = _make_actor_state_dict()
        _stage_checkpoint(tmp_path=tmp_path, monkeypatch=monkeypatch, payload_override=dict(flat))
        partner = FrozenHARLPartner(_spec())
        with pytest.raises(ValueError, match="'actor'"):
            partner.reset(seed=0)

    def test_actor_key_not_dict_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """plan/04 §3.6: ``loaded["actor"]`` must be a dict, not a bare tensor."""
        _stage_checkpoint(
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
            payload_override={"actor": torch.zeros(2, 2)},
        )
        partner = FrozenHARLPartner(_spec())
        with pytest.raises(ValueError, match="'actor'"):
            partner.reset(seed=0)

    def test_missing_feature_norm_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """plan/04 §3.6: shape inference requires base.feature_norm.weight."""
        sd = _make_actor_state_dict()
        del sd["base.feature_norm.weight"]
        del sd["base.feature_norm.bias"]
        _stage_checkpoint(tmp_path=tmp_path, monkeypatch=monkeypatch, state_dict=sd)
        partner = FrozenHARLPartner(_spec())
        with pytest.raises(ValueError, match=r"base\.feature_norm"):
            partner.reset(seed=0)

    def test_missing_fc_mean_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """plan/04 §3.6: shape inference requires act.action_out.fc_mean.weight."""
        sd = _make_actor_state_dict()
        del sd["act.action_out.fc_mean.weight"]
        del sd["act.action_out.fc_mean.bias"]
        _stage_checkpoint(tmp_path=tmp_path, monkeypatch=monkeypatch, state_dict=sd)
        partner = FrozenHARLPartner(_spec())
        with pytest.raises(ValueError, match=r"act\.action_out\.fc_mean"):
            partner.reset(seed=0)

    def test_load_state_dict_runtime_error_remapped_to_value_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ADR-002 §Decisions: a torch RuntimeError is wrapped so callers see one error type."""
        sd = _make_actor_state_dict()
        # Reshape a Linear weight to an incompatible shape so HARL's
        # StochasticPolicy.load_state_dict raises mid-load.
        sd["base.mlp.fc.0.weight"] = torch.zeros(_HIDDEN_DIM + 4, _OBS_DIM)
        _stage_checkpoint(tmp_path=tmp_path, monkeypatch=monkeypatch, state_dict=sd)
        partner = FrozenHARLPartner(_spec())
        with pytest.raises(ValueError, match="load_state_dict failed"):
            partner.reset(seed=0)

    def test_obs_shape_mismatch_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """plan/04 §3.6: obs state dim must match the inferred actor input dim."""
        _stage_checkpoint(tmp_path=tmp_path, monkeypatch=monkeypatch)
        partner = FrozenHARLPartner(_spec())
        bad_state = np.zeros(_OBS_DIM + 3, dtype=np.float32)
        with pytest.raises(ValueError, match="input dim"):
            partner.act(_obs(state=bad_state))

    def test_missing_agent_uid_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """plan/04 §3.6: a partner without its uid in obs cannot act."""
        _stage_checkpoint(tmp_path=tmp_path, monkeypatch=monkeypatch)
        partner = FrozenHARLPartner(_spec())
        with pytest.raises(ValueError, match="allegro_hand_right"):
            partner.act({"agent": {}})


# ---------------------------------------------------------------------------
# Full EgoPPOTrainer.state_dict() shape (actor + critic + optims)
# ---------------------------------------------------------------------------


class TestFullTrainerCheckpoint:
    def test_loads_from_full_egoppo_state_dict(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """plan/04 §3.6: the wrapper ignores critic/optimizer sub-dicts in the .pt.

        The published Phase-0 zoo seed (``happo_seed7_step50k.pt``) is the
        full :meth:`EgoPPOTrainer.state_dict` output — a dict with keys
        ``"actor"``, ``"actor_optim"``, ``"critic"``, ``"critic_optim"``.
        :class:`FrozenHARLPartner` must load successfully from that shape
        by reading only the ``"actor"`` sub-key.
        """
        actor_sd = _make_actor_state_dict()
        full_payload: dict[str, object] = {
            "actor": dict(actor_sd),
            "actor_optim": {"state": {}, "param_groups": []},
            "critic": {"net.0.weight": torch.zeros(4, _OBS_DIM)},
            "critic_optim": {"state": {}, "param_groups": []},
        }
        _stage_checkpoint(
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
            payload_override=full_payload,
        )
        partner = FrozenHARLPartner(_spec())
        action = partner.act(_obs())
        assert action.shape == (_ACTION_DIM,)
