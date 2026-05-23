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

from dataclasses import replace
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


class TestOverrideActionDimForScriptedHeuristic:
    """Pin the #194 override helper used by the Tier-2 round-trip.

    The helper is the engineering-only workaround for the
    canonical-zoo / real-env action-dim mismatch; the assertions
    below verify it (a) rewrites the scripted_heuristic spec, (b)
    leaves other specs untouched, and (c) preserves all other
    ``extra`` fields. If the #187 remediation lands the Option-B
    ``load_partner`` env-derivation API, this helper retires and
    these tests retire with it.
    """

    def _make_zoo_specs(self) -> list[PartnerSpec]:
        return list(make_phase0_draft_zoo())

    def test_overrides_scripted_heuristic_action_dim(self) -> None:
        """``extra["action_dim"]`` is rewritten to the override value for scripted_heuristic."""
        zoo = self._make_zoo_specs()
        scripted = next(s for s in zoo if s.class_name == "scripted_heuristic")
        assert scripted.extra["action_dim"] == "2"  # canonical-zoo invariant pin
        overridden = _override_action_dim_for_scripted_heuristic(scripted, action_dim=13)
        assert overridden.extra["action_dim"] == "13"

    def test_preserves_other_extra_fields(self) -> None:
        """Override only touches ``action_dim``; ``uid`` / ``target_xy`` / ``task`` flow through."""
        zoo = self._make_zoo_specs()
        scripted = next(s for s in zoo if s.class_name == "scripted_heuristic")
        overridden = _override_action_dim_for_scripted_heuristic(scripted, action_dim=13)
        for key in ("uid", "target_xy", "task"):
            assert overridden.extra[key] == scripted.extra[key]
        # PartnerSpec.partner_id is derived from spec identity; verify
        # we have not mutated any field the registry keys off.
        assert overridden.class_name == scripted.class_name
        assert overridden.seed == scripted.seed
        assert overridden.checkpoint_step == scripted.checkpoint_step
        assert overridden.weights_uri == scripted.weights_uri

    def test_non_scripted_heuristic_specs_pass_through_unchanged(self) -> None:
        """Frozen-RL specs (mappo / harl) carry no ``action_dim`` knob; override is a no-op."""
        zoo = self._make_zoo_specs()
        for spec in zoo:
            if spec.class_name == "scripted_heuristic":
                continue
            overridden = _override_action_dim_for_scripted_heuristic(spec, action_dim=13)
            assert overridden is spec  # identity preserved for non-scripted specs

    def test_override_does_not_mutate_input_spec(self) -> None:
        """``PartnerSpec`` is ``frozen=True``; the override returns a new instance."""
        zoo = self._make_zoo_specs()
        scripted = next(s for s in zoo if s.class_name == "scripted_heuristic")
        original_extra = dict(scripted.extra)
        overridden = _override_action_dim_for_scripted_heuristic(scripted, action_dim=13)
        assert scripted.extra == original_extra  # input untouched
        assert overridden is not scripted  # fresh instance


# ---------------------------------------------------------------------------
# Tier-2: real Stage-0 env (GPU-gated)
# ---------------------------------------------------------------------------


def _override_action_dim_for_scripted_heuristic(
    spec: PartnerSpec, *, action_dim: int
) -> PartnerSpec:
    """Return a copy of ``spec`` with ``extra["action_dim"]`` rewritten (#194).

    Local helper, intentionally not promoted to a shared utility — the
    correct long-term home for the env→partner action-dim derivation is
    inside ``load_partner`` (#194 Option B), which depends on the #187
    science decision about the scripted heuristic's planar-reach
    semantics. Keeping the override visible at the test call site flags
    it as a workaround rather than implying we've already chosen the
    Option-B API.

    Args:
        spec: A draft-zoo spec; for the ``scripted_heuristic`` row the
            ``extra["action_dim"]`` field is overwritten.
        action_dim: Probe-derived value from
            ``env.action_space.spaces[partner_uid].shape[0]``.

    Returns:
        A new :class:`PartnerSpec` (``PartnerSpec`` is ``frozen=True``);
        non-scripted_heuristic specs are returned unchanged.
    """
    if spec.class_name != "scripted_heuristic":
        return spec
    return replace(spec, extra={**spec.extra, "action_dim": str(action_dim)})


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

    **scripted_heuristic action_dim override (#194 — engineering invariant).**
    :func:`chamber.partners.selection.make_phase0_draft_zoo` ships the
    ``scripted_heuristic`` row with ``extra["action_dim"]="2"`` so it
    matches the Tier-1 :class:`tests.fakes.FakeMultiAgentEnv` per-uid
    action shape ``(2,)``. The real Stage-0 ``fetch`` URDF under
    ManiSkill v3's default control mode exposes a 13-D action space,
    so the canonical spec is shape-wrong here. This test patches the
    spec via :func:`_override_action_dim_for_scripted_heuristic` to
    rewrite ``extra["action_dim"]`` to the probe-derived value before
    calling ``load_partner``. The canonical zoo stays unchanged
    (Tier-1 continues to use its 2-D fake-env contract).

    **What the override does NOT address (#187 — open science question).**
    Even with the action-dim shape corrected, the heuristic still writes
    ``action[0]/action[1] = clipped xy delta toward target_xy`` and zeros
    the remaining components. For fetch's ManiSkill-v3 action layout,
    components 0/1 are wheel commands rather than Cartesian velocities,
    so the resulting partner motion is the joint-vs-Cartesian mis-port
    documented in #187. The Stage-2 CM axis needs that mis-port resolved
    (the partner must emit Cartesian pose for the comm channel to
    mediate meaningfully) but the M4 gate this test pins is purely the
    shape + round-trip contract. The shape fix is engineering-only; the
    semantics fix is gated on #187's founder remediation call.

    **Retirement path.** If #187 lands the Option-B API
    (``load_partner`` derives ``action_dim`` from an injected env
    handle, plus a comm-channel pose reader for the heuristic), this
    override and its helper can be deleted in the same commit.
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
        # #194 engineering override: patch the canonical
        # scripted_heuristic action_dim to match the real env. See the
        # function docstring above for the #187 retirement path.
        patched_spec = _override_action_dim_for_scripted_heuristic(spec, action_dim=action_dim)
        inner = make_stage0_env()
        try:
            _exercise_partner(
                inner_env=inner,
                spec=patched_spec,
                inner_uids=inner_uids,
                action_dim=action_dim,
            )
        finally:
            inner.close()
