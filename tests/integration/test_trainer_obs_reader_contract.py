# SPDX-License-Identifier: Apache-2.0
# pyright: reportPrivateImportUsage=false, reportPrivateUsage=false
#
# torch stubs and HARL fork internals aren't fully advertised; the test
# also reads ``trainer._obs_dim`` (intentional private-attr access — the
# field is the load-bearing proxy for the contract under test).
"""Regression pin for the trainer's obs-reader contract (PR #182 Surface 6; ADR-002).

The Stage-1b AS launch (PR #182) surfaced a Tier-2 acceptance gap: the
P1.04 Tier-2 tests verified that :class:`EgoPPOTrainer` ran end-to-end
against a real :class:`Stage1PickPlaceEnv` instance but did NOT verify
what the trainer actually read from the env's obs dict. Surface 6 of
the failure investigation identified that the trainer's obs reader
(:func:`chamber.benchmarks.ego_ppo_trainer._flat_ego_obs`) indexed only
``obs["agent"][ego_uid]["state"]`` — a flat 18-D ``concat(qpos, qvel)``
synthesised by :class:`Stage1ASStateSynthesizer`. Cube pose, goal
position, ego TCP pose, and partner qpos+qvel were PRESENT in the env's
obs dict (under ``obs["extra"]`` and the partner's sibling subtree) but
lived in paths the trainer never read; the ego trained for 100k frames
blind to task-relevant signal. Full runtime evidence:
``spikes/results/stage1-failure-investigation/2026-05-20/surface_6_obs_contract_audit.txt``.

This module ships the regression pin that would have caught Surface 6
at Tier-2 acceptance time. The contract:

    When :class:`EgoPPOTrainer` is constructed against a
    :class:`Stage1PickPlaceEnv`-shaped env (or its Tier-1 fake), the
    ego state vector the trainer sizes its actor against MUST be wide
    enough to cover task-relevant fields (cube pose + goal position +
    partner state) in addition to the ego's own qpos+qvel.

Operationalised as a proxy assertion on ``trainer._obs_dim``: pre-
P1.05.8 the trainer read 18 floats per step; the P1.05.8 remediation
widens :class:`Stage1ASStateSynthesizer` to concatenate
``[ego_qpos, ego_qvel, partner_qpos, partner_qvel, cube_pose,
goal_pos, tcp_pose]`` and pushes ``_obs_dim`` well above the 30-float
threshold this pin asserts. The exact concat order is NOT pinned here
— the field-specific positive tests live in
``tests/unit/test_stage1_pickplace_tier1.py``; this pin is the
structural floor.

The test was originally marked ``@pytest.mark.xfail(strict=True)`` so
it xfailed on pre-P1.05.8 main and xpassed when the Surface 6
remediation landed. The xfail-strict discipline forced the
remediation PR (P1.05.8) to remove the marker in the same commit as
the synthesizer widening, closing the regression-pin lifecycle
cleanly. **As of 2026-05-21 / P1.05.8 the contract is met and the
xfail marker is removed; the assertion is now a positive Tier-1
regression net.** Closure rationale is the consultation brief at
``spikes/results/stage1-failure-investigation/2026-05-20/CONSULTATION_BRIEF.md``;
ADR amendments at ADR-002 §Revision history 2026-05-21, ADR-007
§Stage 1b Rev 12, and ADR-009 §Decision direct-observation-of-
partner-pose paragraph.

The test is Tier-1 (no SAPIEN, no GPU): it composes a
:class:`tests.fakes.FakeStage1PickPlaceObs` with the real
:class:`Stage1ASStateSynthesizer` and constructs
:class:`EgoPPOTrainer` via :meth:`EgoPPOTrainer.from_config`. The
contract under test is the static obs-space shape the trainer reads at
construction; SAPIEN kinematics are irrelevant to that shape. Tier-1
placement keeps the regression net firing on every ``make verify``,
not only on the GPU box where the original gap shipped from.
"""

from __future__ import annotations

from chamber.benchmarks.ego_ppo_trainer import EgoPPOTrainer
from chamber.envs.stage1_obs_filter import Stage1ASStateSynthesizer
from chamber.partners.api import PartnerSpec
from chamber.partners.heuristic import ScriptedHeuristicPartner
from concerto.training.config import (
    EgoAHTConfig,
    EnvConfig,
    HAPPOHyperparams,
    PartnerConfig,
    RuntimeConfig,
)
from tests.fakes import FakeStage1PickPlaceObs

# Conservative threshold: well above the current 18 (ego qpos+qvel
# only) and well below the expected ~50+ after any plausible
# remediation. The exact target depends on the consultation-brief
# outcome — see module docstring for the rationale on proxy over
# direct-offset.
_TASK_RELEVANT_OBS_DIM_MIN = 30


def _build_frozen_partner() -> ScriptedHeuristicPartner:
    """Build a scripted-heuristic partner sized for AS-hetero (fetch, 13-D)."""
    return ScriptedHeuristicPartner(
        spec=PartnerSpec(
            class_name="scripted_heuristic",
            seed=0,
            checkpoint_step=None,
            weights_uri=None,
            extra={"uid": "fetch", "target_xy": "0.0,0.0", "action_dim": "13"},
        )
    )


class TestTrainerObsReaderContract:
    """Pins the :class:`EgoPPOTrainer` obs-reader contract (PR #182 Surface 6).

    Single combined assertion on ``trainer._obs_dim`` — see module
    docstring for the rationale on a structural proxy over per-field
    introspection.
    """

    def test_trainer_obs_dim_covers_task_relevant_fields(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """The trainer's ego obs_dim must cover task-relevant channels.

        Composes the AS-hetero wrapper chain end-to-end on a Tier-1
        fake (no SAPIEN required) and constructs the trainer via the
        same :meth:`EgoPPOTrainer.from_config` path the spike adapter
        uses. Asserts ``trainer._obs_dim >= 30``; today it is 18.
        """
        inner = FakeStage1PickPlaceObs()
        wrapped = Stage1ASStateSynthesizer(inner)
        cfg = EgoAHTConfig(
            seed=0,
            total_frames=10,
            checkpoint_every=1000,
            artifacts_root=tmp_path / "artifacts",
            log_dir=tmp_path / "logs",
            env=EnvConfig(
                task="stage1_pickplace",
                episode_length=50,
                agent_uids=("panda_wristcam", "fetch"),
                condition_id=FakeStage1PickPlaceObs.AS_HETERO_CONDITION_ID,
            ),
            partner=PartnerConfig(
                class_name="scripted_heuristic",
                extra={"uid": "fetch", "target_xy": "0.0,0.0", "action_dim": "13"},
            ),
            happo=HAPPOHyperparams(rollout_length=10, batch_size=10),
            runtime=RuntimeConfig(device="cpu", deterministic_torch=True),
        )
        trainer = EgoPPOTrainer.from_config(
            cfg,
            env=wrapped,  # type: ignore[arg-type]
            partner=_build_frozen_partner(),
            ego_uid="panda_wristcam",
        )

        # Today: trainer._obs_dim == 18 (ego qpos+qvel only). Surface 6
        # remediation must widen this so the actor sees task-relevant
        # signal. The 30-float threshold is the structural proxy: any
        # plausible remediation (cube_pose 7-D + goal_pos 3-D + partner
        # state ≥18-D) pushes _obs_dim well above 30.
        assert trainer._obs_dim >= _TASK_RELEVANT_OBS_DIM_MIN, (
            f"trainer._obs_dim={trainer._obs_dim} does not cover task-relevant "
            f"channels (expected ≥ {_TASK_RELEVANT_OBS_DIM_MIN}). "
            "See PR #182 Surface 6 INVESTIGATION.md."
        )
