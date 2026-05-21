# SPDX-License-Identifier: Apache-2.0
"""Tier-2 SAPIEN-gated tests for the Stage-1b dispatch path on the AS adapter (P1.05).

End-to-end smoke that ``chamber.benchmarks.stage1_as.run_axis`` with
``args.sub_stage='1b'`` builds the real
:class:`chamber.envs.stage1_pickplace.Stage1PickPlaceEnv` env per
condition and routes through
:class:`chamber.benchmarks.stage1_common.TrainedPolicyFactory`. The
full Stage-1b science launch (5 seeds x 100k frames per axis) is
out-of-scope here — the founder runs that via
``scripts/repro/stage1_as_stage1b.sh`` on the RTX 2080 / A100 box and
captures the archive under
``spikes/results/stage1-AS-stage1b-<UTC-date>/`` (PR 5b operational
artefact).

This Tier-2 test shrinks **both** the prereg sample sizes (1 seed x 1
episode x 2 conditions) AND the cfg's ``total_frames`` (1k instead of
100k) so the smoke completes in a couple of minutes on the RTX 2080
rather than the ~80 min the full-budget cfg would take. The dispatch
contract and the env build are what's exercised, not the science
gate.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest

from chamber.benchmarks.ego_ppo_trainer import EgoPPOTrainer
from chamber.benchmarks.stage1_as import run_axis
from chamber.envs.stage1_pickplace import make_stage1_pickplace_env
from chamber.evaluation.prereg import load_prereg
from chamber.evaluation.results import SpikeRun
from chamber.partners.api import PartnerSpec
from chamber.partners.heuristic import ScriptedHeuristicPartner
from chamber.utils.device import sapien_gpu_available
from concerto.training.config import (
    EgoAHTConfig,
    EnvConfig,
    HAPPOHyperparams,
    PartnerConfig,
    RuntimeConfig,
)
from concerto.training.config import load_config as _real_load_config

pytestmark = [
    pytest.mark.slow,
    pytest.mark.gpu,
    pytest.mark.skipif(
        not sapien_gpu_available(),
        reason="Stage-1b dispatch requires SAPIEN/Vulkan (the real ManiSkill pick-place env)",
    ),
]

_REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _tiny_prereg_for_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shrink the prereg + cfg for the Tier-2 smoke.

    Two shrinks happen here:

    1. ``_load_canonical_prereg`` is patched to return a 1-seed x
       1-episode-per-seed x 2-condition spec (so the adapter loop
       runs 2 (seed, condition) cells, each building one env + one
       trained policy).
    2. ``load_config`` is patched to wrap the real loader and rewrite
       ``total_frames``, ``rollout_length``, ``batch_size`` to tiny
       values so each cell's training run completes in ~30 s instead
       of the production ~25 min. The other cfg fields (env.task,
       env.condition_id, partner, safety) are honoured verbatim — the
       smoke is about the dispatch contract, not the gradient signal.
    """
    spec = load_prereg(_REPO_ROOT / "spikes" / "preregistration" / "AS.yaml")
    tiny = spec.model_copy(update={"seeds": [0], "episodes_per_seed": 1})

    def _fake_load_prereg(*args, **kwargs):  # type: ignore[no-untyped-def]
        del args, kwargs
        return tiny, _REPO_ROOT / "spikes" / "preregistration" / "AS.yaml"

    def _shrunken_load_config(*args, **kwargs):  # type: ignore[no-untyped-def]
        cfg = _real_load_config(*args, **kwargs)
        # Tier-2 smoke budget: 1k frames is enough to cover one PPO
        # update at rollout_length=256 + a handful of safety_telemetry
        # window flushes. Production runs at 100k per ADR-007 §Stage 1b.
        return cfg.model_copy(
            update={
                "total_frames": 1000,
                "happo": cfg.happo.model_copy(update={"rollout_length": 256, "batch_size": 64}),
            }
        )

    monkeypatch.setattr(
        "chamber.benchmarks.stage1_as._load_canonical_prereg",
        _fake_load_prereg,
    )
    monkeypatch.setattr(
        "concerto.training.config.load_config",
        _shrunken_load_config,
    )


class TestStage1ASStage1bDispatchSmoke:
    """The dispatch path runs end-to-end on real SAPIEN env + TrainedPolicyFactory."""

    def test_stage1b_dispatch_completes_end_to_end(self) -> None:
        """``run_axis`` with ``args.sub_stage='1b'`` returns a sub_stage=1b SpikeRun."""
        args = argparse.Namespace(axis="AS", sub_stage="1b")
        run = run_axis(args)
        assert isinstance(run, SpikeRun)
        assert run.sub_stage == "1b"
        assert run.axis == "AS"
        # 1 seed x 1 episode x 2 conditions = 2 episodes total in the smoke.
        assert len(run.episode_results) == 2

    def test_stage1b_dispatch_records_prereg_sha(self) -> None:
        """ADR-007 §Discipline: the prereg blob SHA is verified + recorded under sub_stage='1b'."""
        args = argparse.Namespace(axis="AS", sub_stage="1b")
        run = run_axis(args)
        assert len(run.prereg_sha) == 40


# AS conditions with their (ego_uid, partner_uid) tuples. Pinned here
# so the per-condition trainer-obs-dim assertion stays self-
# documenting; the canonical source is
# ``chamber.envs.stage1_pickplace._CONDITION_TABLE``.
_AS_HOMO = "stage1_pickplace_panda_only_mappo_shared_param"
_AS_HETERO = "stage1_pickplace_panda_plus_fetch_ego_aht_happo_per_agent"
_AS_CONDITIONS: tuple[tuple[str, str, str], ...] = (
    (_AS_HOMO, "panda_wristcam", "panda_partner"),
    (_AS_HETERO, "panda_wristcam", "fetch"),
)


def _make_minimal_cfg_for_condition(
    tmp_path,  # type: ignore[no-untyped-def]
    *,
    condition_id: str,
    partner_uid: str,
    partner_action_dim: int,
) -> EgoAHTConfig:
    """Smallest cfg that lets ``EgoPPOTrainer.from_config`` construct on a real Stage-1b env.

    No training is run; this fixture only exercises the
    ``ego_obs_space = env.observation_space["agent"][ego_uid]["state"]``
    derivation path at trainer construction. Hidden dim / rollout
    length / batch size are scaled down so the construction is fast
    on the RTX 2080.
    """
    return EgoAHTConfig(
        seed=0,
        total_frames=100,
        checkpoint_every=1000,
        artifacts_root=tmp_path / "artifacts",
        log_dir=tmp_path / "logs",
        env=EnvConfig(
            task="stage1_pickplace",
            episode_length=50,
            agent_uids=("panda_wristcam", partner_uid),
            condition_id=condition_id,
        ),
        partner=PartnerConfig(
            class_name="scripted_heuristic",
            extra={
                "uid": partner_uid,
                "target_xy": "0.0,0.0",
                "action_dim": str(partner_action_dim),
            },
        ),
        happo=HAPPOHyperparams(rollout_length=10, batch_size=10, hidden_dim=32),
        runtime=RuntimeConfig(device="cuda", deterministic_torch=False),
    )


def _build_frozen_partner(partner_uid: str, partner_action_dim: int) -> ScriptedHeuristicPartner:
    """Frozen scripted partner — passes the ADR-009 black-box-policy gate."""
    return ScriptedHeuristicPartner(
        spec=PartnerSpec(
            class_name="scripted_heuristic",
            seed=0,
            checkpoint_step=None,
            weights_uri=None,
            extra={
                "uid": partner_uid,
                "target_xy": "0.0,0.0",
                "action_dim": str(partner_action_dim),
            },
        )
    )


class TestTrainerObsDimMatchesWidenedStatePerCondition:
    """Pin the Surface 6 remediation against real SAPIEN per-condition shapes.

    ADR-007 §Stage 1b Rev 12 / P1.05.8: the widened
    :class:`Stage1ASStateSynthesizer` composes the ego's ``state``
    Box from
    ``[ego_qpos, ego_qvel, partner_qpos, partner_qvel, cube_pose,
    goal_pos, tcp_pose]``. Per-condition dim is env-emit-dependent
    (the panda_partner vs fetch qpos / qvel shapes differ); this test
    is the canonical ground-truth for what the trainer sees on real
    SAPIEN against each AS condition. The Tier-1 fake's assumption
    (``tests/unit/test_stage1_pickplace_tier1.py``) is self-
    consistent against the fake's encoded shapes; this Tier-2 test
    pins what the real env emits.

    The assertion is structural: ``trainer._obs_dim`` MUST equal the
    sum derived from ``env.observation_space`` directly (ego qpos +
    qvel + partner qpos + qvel + 7 + 3 + 7). If the SAPIEN env's
    fetch qpos shape is, say, 13-D instead of the Tier-1 fake's 15-D
    assumption, the equality still holds — both sides read off the
    same env-emit. The pin protects against a future regression where
    the synthesiser stops widening, where the trainer's
    ``_flat_ego_obs`` reverts to ego-only, or where a refactor breaks
    the env↔trainer obs wire.
    """

    @pytest.mark.parametrize(("condition_id", "ego_uid", "partner_uid"), _AS_CONDITIONS)
    def test_trainer_obs_dim_matches_widened_state_per_condition(
        self,
        tmp_path,  # type: ignore[no-untyped-def]
        condition_id: str,
        ego_uid: str,
        partner_uid: str,
    ) -> None:
        """Per-condition: trainer._obs_dim equals the sum of ego + partner + task slot dims."""
        env = make_stage1_pickplace_env(condition_id=condition_id, episode_length=50, root_seed=0)
        try:
            # Pull per-condition shapes off the real env's observation_space
            # rather than hardcoding — defends against fake-vs-real drift.
            agent_space = env.observation_space["agent"]  # type: ignore[index]
            ego_qpos_dim = int(np.prod(agent_space[ego_uid]["qpos"].shape))  # type: ignore[union-attr, index]
            ego_qvel_dim = int(np.prod(agent_space[ego_uid]["qvel"].shape))  # type: ignore[union-attr, index]
            partner_qpos_dim = int(np.prod(agent_space[partner_uid]["qpos"].shape))  # type: ignore[union-attr, index]
            partner_qvel_dim = int(np.prod(agent_space[partner_uid]["qvel"].shape))  # type: ignore[union-attr, index]
            expected_obs_dim = (
                ego_qpos_dim + ego_qvel_dim + partner_qpos_dim + partner_qvel_dim + 7 + 3 + 7
            )

            partner_action_dim = int(env.action_space[partner_uid].shape[0])  # type: ignore[index, union-attr]
            cfg = _make_minimal_cfg_for_condition(
                tmp_path,
                condition_id=condition_id,
                partner_uid=partner_uid,
                partner_action_dim=partner_action_dim,
            )
            partner = _build_frozen_partner(partner_uid, partner_action_dim)
            trainer = EgoPPOTrainer.from_config(
                cfg,
                env=env,  # type: ignore[arg-type]
                partner=partner,
                ego_uid=ego_uid,
            )

            # The contract: the trainer's flat-state input dim equals the
            # widened-state shape composed by Stage1ASStateSynthesizer.
            # Intentional private-attr access — ``_obs_dim`` is the
            # load-bearing proxy the trainer reads at construction time.
            assert trainer._obs_dim == expected_obs_dim, (
                f"trainer._obs_dim={trainer._obs_dim} does not match the widened "
                f"state per condition {condition_id!r}: expected "
                f"ego_qpos({ego_qpos_dim}) + ego_qvel({ego_qvel_dim}) + "
                f"partner_qpos({partner_qpos_dim}) + partner_qvel({partner_qvel_dim}) "
                f"+ 7 + 3 + 7 = {expected_obs_dim}. "
                "See ADR-007 §Stage 1b Rev 12 / P1.05.8."
            )
            # Sanity: the widening pushes _obs_dim well past the xfail-pin
            # threshold of 30 the Tier-1 regression net asserts.
            assert trainer._obs_dim >= 30
        finally:
            env.close()
