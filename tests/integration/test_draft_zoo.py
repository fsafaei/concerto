# SPDX-License-Identifier: Apache-2.0
# pyright: reportPrivateImportUsage=false
"""Phase-0 draft-zoo Stage-0 round-trip (T4.9; plan/04 §3.8; §6 #1 + #4).

The M4 gate proper. Builds the three canonical draft-zoo partners via
:func:`chamber.partners.selection.make_phase0_draft_zoo` +
:func:`chamber.partners.registry.load_partner`, wraps the env with
:class:`chamber.envs.PartnerIdAnnotationWrapper` (the producer side of
ADR-006 risk #3 / ADR-004 §risk-mitigation #2), and exercises
``reset()`` + ``act()`` 10 times per partner against the Stage-0 smoke
env.

Two tiers (mirrors :mod:`tests.integration.test_stage0_adapter_real` /
:mod:`tests.integration.test_stage0_adapter_fake`):

- Tier-1 (CPU, default): the canonical 3-uid
  :class:`tests.fakes.FakeMultiAgentEnv` stands in for the Stage-0
  rig. Fixture-staged ``.pt`` artefacts (under ``tmp_path`` keyed by
  the canonical zoo URIs) carry tiny actor weights matching the fake
  env's obs / action shape so the partners load via the registry
  without touching SAPIEN.
- Tier-2 (``@pytest.mark.gpu``, SAPIEN-gated): runs against the real
  Stage-0 env from :func:`chamber.benchmarks.stage0_smoke.make_stage0_env`.
  Skipped on CPU-only hosts; on a GPU host it reproduces the full M4
  gate (plan/04 §6 #1) including the wrapper-chain compatibility.

What the test pins (plan/04 §6 #1 + #4):

1. Every draft-zoo spec resolves via the registry — no ``KeyError``s.
2. Each partner's ``act(obs)`` returns a finite vector of the expected
   shape for 10 steps without raising.
3. ``obs["meta"]["partner_id"]`` matches
   :attr:`PartnerSpec.partner_id` on every step.
4. The ``_FORBIDDEN_ATTRS`` shield (ADR-009 §Consequences) still bites
   on every loaded partner after a real load.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from chamber.envs.partner_meta import PartnerIdAnnotationWrapper
from chamber.partners.frozen_harl import (
    _ARTIFACTS_ROOT_ENV,
    _HARL_INFERENCE_ARGS,
)
from chamber.partners.frozen_mappo import _MAPPOActor
from chamber.partners.registry import load_partner
from chamber.partners.selection import make_phase0_draft_zoo
from chamber.utils.device import sapien_gpu_available
from concerto.training.checkpoints import CheckpointMetadata, save_checkpoint
from tests.fakes import FakeMultiAgentEnv

if TYPE_CHECKING:
    from pathlib import Path

    from chamber.partners.api import PartnerSpec

#: Fake-env per-uid action dim (matches
#: :class:`tests.fakes.FakeMultiAgentEnv`'s 2-D Box action space).
_FAKE_ACTION_DIM = 2

#: Fake-env per-uid state-channel dim (matches FakeMultiAgentEnv's
#: ``obs["agent"][uid]["state"]`` shape ``(3,)``).
_FAKE_OBS_DIM = 3

#: Tiny hidden width for the fixture checkpoints. The wrapper's shape-
#: inference reads ``base.mlp.fc.0.weight.shape[0]`` (HARL) and
#: ``fc1.weight.shape[0]`` (MAPPO) from the saved tensors — a 4-unit
#: hidden layer is enough to exercise that path without bloating the
#: ``.pt`` fixtures.
_FIXTURE_HIDDEN_DIM = 4

_NUM_STEPS = 10


def _stage_mappo_checkpoint(
    *, artifacts_root: Path, uri: str, obs_dim: int, action_dim: int
) -> None:
    """Write a tiny MAPPO actor under ``uri`` matching the inner env's shape."""
    torch.manual_seed(0)
    actor = _MAPPOActor(obs_dim=obs_dim, hidden_dim=_FIXTURE_HIDDEN_DIM, action_dim=action_dim)
    sd = {name: tensor.clone().detach() for name, tensor in actor.state_dict().items()}
    save_checkpoint(
        state_dict={"actor": sd},  # type: ignore[arg-type]
        uri=uri,
        metadata=_metadata(seed=42, step=100_000),
        artifacts_root=artifacts_root,
    )


def _stage_harl_checkpoint(
    *, artifacts_root: Path, uri: str, obs_dim: int, action_dim: int
) -> None:
    """Write a tiny HARL StochasticPolicy actor under ``uri`` matching the env's shape."""
    import gymnasium as gym
    from harl.models.policy_models.stochastic_policy import StochasticPolicy

    args = dict(_HARL_INFERENCE_ARGS)
    args["hidden_sizes"] = [_FIXTURE_HIDDEN_DIM, _FIXTURE_HIDDEN_DIM]
    obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
    act_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)
    torch.manual_seed(7)
    actor = StochasticPolicy(args, obs_space, act_space, torch.device("cpu"))
    sd = {name: tensor.clone().detach() for name, tensor in actor.state_dict().items()}
    save_checkpoint(
        state_dict={"actor": sd},  # type: ignore[arg-type]
        uri=uri,
        metadata=_metadata(seed=7, step=50_000),
        artifacts_root=artifacts_root,
    )


def _metadata(*, seed: int, step: int) -> CheckpointMetadata:
    return CheckpointMetadata(
        run_id="0" * 16,
        seed=seed,
        step=step,
        git_sha="abc1234",
        pyproject_hash="0" * 64,
        sha256="placeholder",
    )


def _stage_zoo_artefacts(
    *,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    zoo: list[PartnerSpec],
    obs_dim: int,
    action_dim: int,
) -> None:
    """Stage matching checkpoints for every frozen-RL spec in ``zoo``."""
    monkeypatch.setenv(_ARTIFACTS_ROOT_ENV, str(tmp_path))
    for spec in zoo:
        if spec.class_name == "frozen_mappo":
            assert spec.weights_uri is not None
            _stage_mappo_checkpoint(
                artifacts_root=tmp_path,
                uri=spec.weights_uri,
                obs_dim=obs_dim,
                action_dim=action_dim,
            )
        elif spec.class_name == "frozen_harl":
            assert spec.weights_uri is not None
            _stage_harl_checkpoint(
                artifacts_root=tmp_path,
                uri=spec.weights_uri,
                obs_dim=obs_dim,
                action_dim=action_dim,
            )


def _zero_dict_action(uids: tuple[str, ...], action_dim: int) -> dict[str, np.ndarray]:
    return {uid: np.zeros(action_dim, dtype=np.float32) for uid in uids}


def _exercise_partner(
    *,
    inner_env,  # type: ignore[no-untyped-def]
    spec: PartnerSpec,
    inner_uids: tuple[str, ...],
    action_dim: int,
    n_steps: int = _NUM_STEPS,
) -> None:
    """Run a partner against ``inner_env`` for ``n_steps``; assert the M4 contract.

    Builds the partner via the registry, wraps the env with
    :class:`PartnerIdAnnotationWrapper`, drives ``reset + step`` for
    ``n_steps``, and asserts ``obs["meta"]["partner_id"]`` matches
    ``spec.partner_id`` at every step.
    """
    partner = load_partner(spec)
    env = PartnerIdAnnotationWrapper(inner_env, partner_id=spec.partner_id)
    obs, _ = env.reset(seed=0)
    partner.reset(seed=0)
    assert obs["meta"]["partner_id"] == spec.partner_id, (
        f"initial obs missing partner_id for {spec.class_name!r}"
    )
    partner_uid = spec.extra["uid"]

    for step_idx in range(n_steps):
        partner_action = partner.act(obs)
        assert partner_action.shape == (action_dim,), (
            f"{spec.class_name!r}: action shape {partner_action.shape} != ({action_dim},)"
        )
        assert np.all(np.isfinite(partner_action)), (
            f"{spec.class_name!r}: non-finite action at step {step_idx}"
        )
        action = _zero_dict_action(inner_uids, action_dim)
        action[partner_uid] = partner_action
        obs, _, _, _, _ = env.step(action)
        assert obs["meta"]["partner_id"] == spec.partner_id, (
            f"{spec.class_name!r}: obs partner_id drifted at step {step_idx}"
        )

    # ADR-009 §Consequences: the runtime shield still bites after a real load.
    for forbidden in ("train", "learn", "update", "set_weights"):
        with pytest.raises(AttributeError, match="ADR-009"):
            getattr(partner, forbidden)


# ---------------------------------------------------------------------------
# Tier-1: Fake multi-agent env (default; CPU)
# ---------------------------------------------------------------------------


class TestDraftZooFakeEnv:
    """Plan/04 §6 #1 + #4 via :class:`tests.fakes.FakeMultiAgentEnv`."""

    def test_all_three_partners_load_and_act_on_fake_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The canonical 3-partner draft zoo round-trips on the fake Stage-0 env."""
        zoo = make_phase0_draft_zoo()
        _stage_zoo_artefacts(
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
            zoo=zoo,
            obs_dim=_FAKE_OBS_DIM,
            action_dim=_FAKE_ACTION_DIM,
        )
        inner_uids = ("panda_wristcam", "fetch", "allegro_hand_right")
        for spec in zoo:
            inner = FakeMultiAgentEnv(agent_uids=inner_uids)
            _exercise_partner(
                inner_env=inner,
                spec=spec,
                inner_uids=inner_uids,
                action_dim=_FAKE_ACTION_DIM,
            )

    def test_partner_ids_are_pairwise_distinct(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """plan/04 §6 #4: every draft-zoo partner has a unique partner_id surface."""
        zoo = make_phase0_draft_zoo()
        _stage_zoo_artefacts(
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
            zoo=zoo,
            obs_dim=_FAKE_OBS_DIM,
            action_dim=_FAKE_ACTION_DIM,
        )
        inner_uids = ("panda_wristcam", "fetch", "allegro_hand_right")
        seen_ids: set[str] = set()
        for spec in zoo:
            inner = FakeMultiAgentEnv(agent_uids=inner_uids)
            wrapped = PartnerIdAnnotationWrapper(inner, partner_id=spec.partner_id)
            obs, _ = wrapped.reset(seed=0)
            seen_ids.add(obs["meta"]["partner_id"])
        assert len(seen_ids) == 3


# ---------------------------------------------------------------------------
# Tier-2: real Stage-0 env (GPU-gated)
# ---------------------------------------------------------------------------


@pytest.mark.smoke
@pytest.mark.gpu
@pytest.mark.skipif(
    not sapien_gpu_available(),
    reason="Requires Vulkan/GPU (SAPIEN); skipped on CPU-only machines",
)
def test_draft_zoo_round_trip_on_real_stage0_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """plan/04 §6 #1: the M4 gate proper — real Stage-0 env, 3 partners, 10 steps each.

    Mirrors the Tier-1 test but uses
    :func:`chamber.benchmarks.stage0_smoke.make_stage0_env`. The fixture
    checkpoints are sized to the real env's ``obs["agent"][uid]["state"]``
    and ``action_space[uid]`` shapes (probed by instantiating the env
    once before staging the .pt files).
    """
    import gymnasium as gym

    from chamber.benchmarks.stage0_smoke import make_stage0_env

    zoo = make_phase0_draft_zoo()
    probe = make_stage0_env()
    try:
        # Probe per-uid obs / action dims from the wrapper chain so the
        # staged checkpoints match the real env's shapes exactly.
        assert isinstance(probe.observation_space, gym.spaces.Dict)
        assert isinstance(probe.action_space, gym.spaces.Dict)
        agent_space = probe.observation_space.spaces["agent"]
        assert isinstance(agent_space, gym.spaces.Dict)
        uid_shapes: dict[str, tuple[int, int]] = {}
        for spec in zoo:
            uid = spec.extra["uid"]
            uid_obs = agent_space.spaces[uid]
            assert isinstance(uid_obs, gym.spaces.Dict)
            state_space = uid_obs.spaces["state"]
            assert isinstance(state_space, gym.spaces.Box)
            assert state_space.shape is not None
            obs_dim = int(state_space.shape[0])
            action_space = probe.action_space.spaces[uid]
            assert isinstance(action_space, gym.spaces.Box)
            assert action_space.shape is not None
            action_dim = int(action_space.shape[0])
            uid_shapes[uid] = (obs_dim, action_dim)
        inner_uids = tuple(probe.action_space.spaces.keys())
    finally:
        probe.close()

    # Tier-2 stages synthetic checkpoints matching the real env's shapes.
    # The published zoo-seed (``happo_seed7_step50k.pt``) is exercised by
    # ``tests/reproduction/test_zoo_seed_artifact.py`` in B4 — keeping the
    # M4-gate test decoupled from artefact storage avoids confusing
    # shape-mismatch failures here.
    monkeypatch.setenv(_ARTIFACTS_ROOT_ENV, str(tmp_path))
    for spec in zoo:
        uid = spec.extra["uid"]
        obs_dim, action_dim = uid_shapes[uid]
        if spec.class_name == "frozen_mappo":
            assert spec.weights_uri is not None
            _stage_mappo_checkpoint(
                artifacts_root=tmp_path,
                uri=spec.weights_uri,
                obs_dim=obs_dim,
                action_dim=action_dim,
            )
        elif spec.class_name == "frozen_harl":
            assert spec.weights_uri is not None
            _stage_harl_checkpoint(
                artifacts_root=tmp_path,
                uri=spec.weights_uri,
                obs_dim=obs_dim,
                action_dim=action_dim,
            )

    for spec in zoo:
        uid = spec.extra["uid"]
        _obs_dim, action_dim = uid_shapes[uid]
        inner = make_stage0_env()
        try:
            _exercise_partner(
                inner_env=inner,
                spec=spec,
                inner_uids=inner_uids,
                action_dim=action_dim,
            )
        finally:
            inner.close()
