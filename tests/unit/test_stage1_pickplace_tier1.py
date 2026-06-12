# SPDX-License-Identifier: Apache-2.0
# pyright: reportPrivateImportUsage=false, reportAttributeAccessIssue=false, reportIndexIssue=false
#
# Same rationale as :mod:`tests.integration.test_stage0_adapter_real`:
# torch and pytorch_kinematics stubs do not advertise every public symbol;
# suppressed file-locally so the Tier-1 tests stay free of per-line noise.
"""Tier-1 (no-SAPIEN-scene) tests for the Stage-1b pick-place env (ADR-007 §Stage 1b).

Covers the parts of :mod:`chamber.envs.stage1_pickplace` that don't
require constructing a SAPIEN scene:

- :func:`resolve_condition` — table lookup correctness for all four
  Stage-1 ``condition_id`` strings, plus the loud-fail message on a
  bogus id.
- :func:`build_control_models_for_condition` — per-condition per-uid
  :class:`AgentControlModel` typing (panda -> JacobianControlModel;
  fetch -> DoubleIntegratorControlModel; both pandas in AS-homo ->
  JacobianControlModel).
- :class:`PandaJacobianProvider` — URDF parses, Jacobian shape is
  ``(3, 7)``, finite-difference agreement at a non-singular qpos.
- :class:`Stage1OMChannelFilter` — pass-through under non-OM conditions,
  zero-masking of proprio + force_torque under vision_only, no-op when
  the inner env exposes no ``condition_id`` attribute.
- :data:`PANDA_CARTESIAN_ACCEL_CAPACITY_MS2` is finite — Stage-1b's
  audit-gate predicate (Q3 mod 2; ADR-007 §Stage 1b §λ-telemetry
  handoff contract; the audit-gate jq block lands in P1.05) needs a
  finite RHS, which this constant supplies.

Tier-2 SAPIEN-gated coverage lives in
``tests/integration/test_stage1_pickplace_real.py`` — real env
construction, per-condition action/obs shapes, real Jacobian agreement
under SAPIEN qpos, 10-step rollout finite-reward smoke.

ADR-007 §Stage 1b; ADR-004 §Decision; ADR-001 §Decision (wrapper-only), #6.
"""

from __future__ import annotations

from typing import ClassVar

import gymnasium as gym
import numpy as np
import pytest

from chamber.agents.panda_jacobian import (
    CARTESIAN_POSITION_DIM,
    PANDA_ARM_DOF,
    PandaJacobianProvider,
)
from chamber.envs.stage1_obs_filter import Stage1ASStateSynthesizer, Stage1OMChannelFilter
from chamber.envs.stage1_pickplace import (
    DEFAULT_EPISODE_LENGTH,
    FETCH_CARTESIAN_ACCEL_CAPACITY_MS2,
    PANDA_CARTESIAN_ACCEL_CAPACITY_MS2,
    SYNTHESISED_FT_NOISE_SIGMA_N,
    ConditionConfig,
    build_control_models_for_condition,
    resolve_condition,
)
from concerto.safety.api import DoubleIntegratorControlModel, JacobianControlModel

_ALL_CONDITIONS: tuple[str, ...] = (
    "stage1_pickplace_panda_only_mappo_shared_param",
    "stage1_pickplace_panda_plus_fetch_ego_aht_happo_per_agent",
    "stage1_pickplace_vision_only",
    "stage1_pickplace_vision_plus_force_torque_plus_proprio",
)


# ----- resolve_condition (ADR-007 §Stage 1b condition_id resolution) -----


class TestResolveCondition:
    """Stage-1b ``condition_id`` table lookup (ADR-007 §Stage 1b)."""

    def test_as_homo(self) -> None:
        cfg = resolve_condition("stage1_pickplace_panda_only_mappo_shared_param")
        assert isinstance(cfg, ConditionConfig)
        assert cfg.agent_uids == ("panda_wristcam", "panda_partner")
        assert cfg.obs_mode == "state_dict"
        assert cfg.is_om_condition is False
        assert cfg.ft_synthesis_enabled is False

    def test_as_hetero(self) -> None:
        cfg = resolve_condition("stage1_pickplace_panda_plus_fetch_ego_aht_happo_per_agent")
        assert cfg.agent_uids == ("panda_wristcam", "fetch")
        assert cfg.obs_mode == "state_dict"
        assert cfg.is_om_condition is False
        assert cfg.ft_synthesis_enabled is False

    def test_om_homo_vision_only(self) -> None:
        cfg = resolve_condition("stage1_pickplace_vision_only")
        assert cfg.agent_uids == ("panda_wristcam", "fetch")
        assert "rgb" in cfg.obs_mode
        assert "depth" in cfg.obs_mode
        assert cfg.is_om_condition is True
        assert cfg.ft_synthesis_enabled is False

    def test_om_hetero_vision_plus_ft_proprio(self) -> None:
        cfg = resolve_condition("stage1_pickplace_vision_plus_force_torque_plus_proprio")
        assert cfg.agent_uids == ("panda_wristcam", "fetch")
        assert "rgb" in cfg.obs_mode
        assert "depth" in cfg.obs_mode
        assert cfg.is_om_condition is True
        assert cfg.ft_synthesis_enabled is True

    def test_unknown_condition_id_raises_value_error_naming_options(self) -> None:
        """Bogus IDs raise ValueError listing valid options + ADR-007 §Discipline."""
        with pytest.raises(ValueError, match="not one of the four"):
            resolve_condition("stage1_pickplace_unknown_condition")
        with pytest.raises(ValueError, match="ADR-007"):
            resolve_condition("")

    def test_all_four_conditions_covered(self) -> None:
        """Defensive: every Stage-1 condition_id resolves without raising."""
        for cid in _ALL_CONDITIONS:
            resolve_condition(cid)


# ----- build_control_models_for_condition (ADR-004 §Decision per-uid dispatch) -----


class TestBuildControlModelsForCondition:
    """Per-condition per-uid AgentControlModel typing (ADR-004 §Decision; ADR-007 §Stage 1b)."""

    def test_as_homo_two_pandas_get_jacobian_models(self) -> None:
        models = build_control_models_for_condition(
            "stage1_pickplace_panda_only_mappo_shared_param"
        )
        assert set(models) == {"panda_wristcam", "panda_partner"}
        assert all(isinstance(m, JacobianControlModel) for m in models.values())

    def test_as_hetero_panda_jacobian_fetch_double_integrator(self) -> None:
        models = build_control_models_for_condition(
            "stage1_pickplace_panda_plus_fetch_ego_aht_happo_per_agent"
        )
        assert set(models) == {"panda_wristcam", "fetch"}
        assert isinstance(models["panda_wristcam"], JacobianControlModel)
        assert isinstance(models["fetch"], DoubleIntegratorControlModel)

    def test_om_conditions_use_same_typing_as_as_hetero(self) -> None:
        """OM conditions share AS-hetero's panda+fetch tuple."""
        for cid in (
            "stage1_pickplace_vision_only",
            "stage1_pickplace_vision_plus_force_torque_plus_proprio",
        ):
            models = build_control_models_for_condition(cid)
            assert isinstance(models["panda_wristcam"], JacobianControlModel)
            assert isinstance(models["fetch"], DoubleIntegratorControlModel)

    def test_panda_jacobian_model_carries_finite_cartesian_accel_capacity(self) -> None:
        """Q3 mod 2 / ADR-007 §Stage 1b λ-telemetry handoff contract.

        ``max_cartesian_accel_value`` on the panda's JacobianControlModel
        must be finite so the eventual Stage-1b audit-gate predicate
        ``λ_steady_state < cartesian_accel_capacity`` (P1.05 dispatch
        slice) has a finite RHS. The :class:`JacobianControlModel`
        loud-fails at use when the value is left at the default ``nan``
        (cf. ``concerto/safety/api.py:600-606``); this regression test
        pins the env-side handoff contract.
        """
        models = build_control_models_for_condition(
            "stage1_pickplace_panda_only_mappo_shared_param"
        )
        panda = models["panda_wristcam"]
        assert isinstance(panda, JacobianControlModel)
        assert np.isfinite(panda.max_cartesian_accel_value)
        assert panda.max_cartesian_accel_value > 0.0
        # The constant the env wires in must match the module-level
        # source of truth (Stage-2 CR axis review will revisit this).
        assert panda.max_cartesian_accel_value == PANDA_CARTESIAN_ACCEL_CAPACITY_MS2

    def test_jacobian_fn_default_none_keeps_placeholder_loud_fail(self) -> None:
        """Tier-1 path (jacobian_fn=None) preserves JacobianControlModel's loud-fail.

        The Tier-1 build_control_models_for_condition call site (no
        env, no SAPIEN articulation) passes ``jacobian_fn=None``. The
        resulting JacobianControlModel must raise on first use so a
        downstream test that forgets to inject the closure fails loudly
        (cf. ``concerto/safety/api.py:552-559``).
        """
        models = build_control_models_for_condition(
            "stage1_pickplace_panda_only_mappo_shared_param"
        )
        panda = models["panda_wristcam"]
        assert isinstance(panda, JacobianControlModel)
        assert panda.jacobian_fn is None


# ----- PandaJacobianProvider (no SAPIEN scene; URDF + pytorch_kinematics only) -----


class TestPandaJacobianProvider:
    """URDF Jacobian provider Tier-1 surface (ADR-004 §Decision; ADR-007 §Stage 1b)."""

    def test_panda_v3_urdf_parses_to_seven_joint_chain(self) -> None:
        provider = PandaJacobianProvider()
        assert provider.ee_link == "panda_hand_tcp"
        # pytorch_kinematics returns a SerialChain; the chain to
        # panda_hand_tcp must contain exactly the 7 arm joints.
        assert provider._chain.n_joints == PANDA_ARM_DOF

    def test_jacobian_shape_is_three_by_seven(self) -> None:
        """Linear Jacobian rows of the SE(3) chain (Cartesian-only safety body)."""
        provider = PandaJacobianProvider()
        q = np.zeros(PANDA_ARM_DOF, dtype=np.float64)
        jac = provider(None, q)  # type: ignore[arg-type]
        assert jac.shape == (CARTESIAN_POSITION_DIM, PANDA_ARM_DOF)
        assert jac.dtype == np.float64

    def test_jacobian_qpos_shape_validation(self) -> None:
        """Mismatched qpos length raises ValueError (defensive)."""
        provider = PandaJacobianProvider()
        with pytest.raises(ValueError, match="qpos_arm shape"):
            provider(None, np.zeros(6))  # type: ignore[arg-type]

    def test_jacobian_finite_difference_agreement_non_singular_qpos(self) -> None:
        """Analytical Jacobian agrees with FD at a non-singular qpos (ADR-007 §Stage 1b)."""
        import torch

        provider = PandaJacobianProvider()
        # Non-singular reference qpos away from the home pose.
        rng = np.random.default_rng(42)
        q = rng.uniform(-1.0, 1.0, size=PANDA_ARM_DOF).astype(np.float64)
        jac_analytical = provider(None, q)  # type: ignore[arg-type]
        eps = 1e-3

        def fk(qv: np.ndarray) -> np.ndarray:
            qt = torch.tensor(qv, dtype=torch.float32).unsqueeze(0)
            mat = provider._chain.forward_kinematics(qt).get_matrix()
            return mat[0, :3, 3].detach().cpu().numpy()

        jac_fd = np.zeros_like(jac_analytical)
        for k in range(PANDA_ARM_DOF):
            dq = np.zeros(PANDA_ARM_DOF)
            dq[k] = eps
            jac_fd[:, k] = (fk(q + dq) - fk(q - dq)) / (2 * eps)
        # eps=1e-3 + float32 internals -> ~1e-4 floor; tolerate 5e-3 to
        # catch real Jacobian regressions while staying robust to noise.
        max_err = float(np.max(np.abs(jac_analytical - jac_fd)))
        assert max_err < 5e-3, f"Jacobian/FD disagreement: max={max_err:.2e}"


# ----- Stage1ASStateSynthesizer (qpos+qvel -> state; ADR-007 §Stage 1b AS axis) -----


#: Per-uid qpos / qvel dims encoded in the AS Tier-1 fake. Pinned
#: here so the per-condition-shape assertions below are self-
#: documenting; the real SAPIEN env's dims are pinned by the Tier-2
#: integration test (``test_stage1_as_real_stage1b.py``).
_FAKE_EGO_QPOS_DIM: int = 9  # panda_wristcam: 7 arm + 2 mimic-gripper
_FAKE_PANDA_PARTNER_QPOS_DIM: int = 9  # panda_partner mirrors the ego
_FAKE_FETCH_QPOS_DIM: int = 15  # fetch (Tier-1 assumption; Tier-2 pins truth)
_FAKE_CUBE_POSE_DIM: int = 7
_FAKE_GOAL_POS_DIM: int = 3
_FAKE_TCP_POSE_DIM: int = 7

#: Base offsets for the deterministic-counter obs the fake emits.
#: Used so the positional concat-order test can read each slot's
#: contribution to the widened ``state`` vector by inspection.
_OFFSET_EGO_QPOS: int = 100
_OFFSET_EGO_QVEL: int = 200
_OFFSET_PARTNER_QPOS: int = 300
_OFFSET_PARTNER_QVEL: int = 400
_OFFSET_CUBE_POSE: int = 500
_OFFSET_GOAL_POS: int = 600
_OFFSET_TCP_POSE: int = 700


class _FakeStateDictEnv(gym.Env):  # type: ignore[type-arg]
    """Mirrors ManiSkill v3 ``obs_mode="state_dict"`` shape for the AS axis.

    The real Stage1PickPlaceEnv under AS conditions emits per-agent
    ``Dict(qpos, qvel, ...)`` and ``obs["extra"]`` carrying the task
    fields ``cube_pose / goal_pos / tcp_pose``. This fake mirrors
    that contract for both AS-homo (panda + panda_partner) and
    AS-hetero (panda + fetch) so the Tier-1 wrapper tests exercise
    the same code paths the Tier-2 SAPIEN tests exercise end-to-end.

    Per-uid qpos / qvel dims are pinned by module-level constants
    (``_FAKE_EGO_QPOS_DIM = 9``, ``_FAKE_PANDA_PARTNER_QPOS_DIM = 9``,
    ``_FAKE_FETCH_QPOS_DIM = 15``). Whatever the real SAPIEN env's
    fetch qpos shape turns out to be, the synthesiser's
    shape-from-env logic adapts automatically (it reads
    ``inner.spaces["agent"][partner_uid]["qpos"].shape``); the
    Tier-2 integration test ``test_trainer_obs_dim_matches_widened_
    state_per_condition`` pins the real per-condition dim.

    The fake emits deterministic-counter values keyed by per-slot
    base offsets (``_OFFSET_EGO_QPOS = 100`` etc.) so the positional
    concat-order test can read each slot's contribution to the
    widened ``state`` vector by inspection. Pre-P1.05.8 the wrapper
    composed ``state`` from ego qpos+qvel only; post-P1.05.8 the
    ego's ``state`` concatenates
    ``[ego_qpos, ego_qvel, partner_qpos, partner_qvel, cube_pose,
    goal_pos, tcp_pose]`` per ADR-007 §Stage 1b Rev 12.
    """

    metadata: ClassVar[dict[str, object]] = {"render_modes": []}  # type: ignore[misc]

    def __init__(self, condition_id: str | None) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Dict(
                    {
                        "panda_wristcam": gym.spaces.Dict(
                            {
                                "qpos": gym.spaces.Box(
                                    -np.inf, np.inf, (1, _FAKE_EGO_QPOS_DIM), np.float32
                                ),
                                "qvel": gym.spaces.Box(
                                    -np.inf, np.inf, (1, _FAKE_EGO_QPOS_DIM), np.float32
                                ),
                            }
                        ),
                        "panda_partner": gym.spaces.Dict(
                            {
                                "qpos": gym.spaces.Box(
                                    -np.inf,
                                    np.inf,
                                    (1, _FAKE_PANDA_PARTNER_QPOS_DIM),
                                    np.float32,
                                ),
                                "qvel": gym.spaces.Box(
                                    -np.inf,
                                    np.inf,
                                    (1, _FAKE_PANDA_PARTNER_QPOS_DIM),
                                    np.float32,
                                ),
                            }
                        ),
                        "fetch": gym.spaces.Dict(
                            {
                                "qpos": gym.spaces.Box(
                                    -np.inf, np.inf, (1, _FAKE_FETCH_QPOS_DIM), np.float32
                                ),
                                "qvel": gym.spaces.Box(
                                    -np.inf, np.inf, (1, _FAKE_FETCH_QPOS_DIM), np.float32
                                ),
                            }
                        ),
                    }
                ),
                "extra": gym.spaces.Dict(
                    {
                        "cube_pose": gym.spaces.Box(
                            -np.inf, np.inf, (1, _FAKE_CUBE_POSE_DIM), np.float32
                        ),
                        "goal_pos": gym.spaces.Box(
                            -np.inf, np.inf, (1, _FAKE_GOAL_POS_DIM), np.float32
                        ),
                        "tcp_pose": gym.spaces.Box(
                            -np.inf, np.inf, (1, _FAKE_TCP_POSE_DIM), np.float32
                        ),
                    }
                ),
            }
        )
        self.action_space = gym.spaces.Dict({})
        if condition_id is not None:
            self.condition_id = condition_id

    @staticmethod
    def _sample_obs() -> dict:
        def _counter(offset: int, n: int) -> np.ndarray:  # type: ignore[type-arg]
            return (offset + np.arange(n, dtype=np.float32)).reshape(1, n)

        return {
            "agent": {
                "panda_wristcam": {
                    "qpos": _counter(_OFFSET_EGO_QPOS, _FAKE_EGO_QPOS_DIM),
                    "qvel": _counter(_OFFSET_EGO_QVEL, _FAKE_EGO_QPOS_DIM),
                },
                "panda_partner": {
                    "qpos": _counter(_OFFSET_PARTNER_QPOS, _FAKE_PANDA_PARTNER_QPOS_DIM),
                    "qvel": _counter(_OFFSET_PARTNER_QVEL, _FAKE_PANDA_PARTNER_QPOS_DIM),
                },
                "fetch": {
                    "qpos": _counter(_OFFSET_PARTNER_QPOS, _FAKE_FETCH_QPOS_DIM),
                    "qvel": _counter(_OFFSET_PARTNER_QVEL, _FAKE_FETCH_QPOS_DIM),
                },
            },
            "extra": {
                "cube_pose": _counter(_OFFSET_CUBE_POSE, _FAKE_CUBE_POSE_DIM),
                "goal_pos": _counter(_OFFSET_GOAL_POS, _FAKE_GOAL_POS_DIM),
                "tcp_pose": _counter(_OFFSET_TCP_POSE, _FAKE_TCP_POSE_DIM),
            },
        }


class TestStage1ASStateSynthesizer:
    """Synthesised widened ``state`` key for AS conditions (ADR-007 §Stage 1b Rev 12; P1.05.8)."""

    def test_as_hetero_injects_widened_ego_state_with_partner_and_task(self) -> None:
        """AS-hetero ego state concatenates ego + partner + task fields.

        Post-P1.05.8 (ADR-007 §Stage 1b Rev 12 / ADR-002 §Revision
        history 2026-05-21) the ego uid's synthesised ``state`` Box
        widens from ``concat(ego_qpos, ego_qvel)`` (shape ``(18,)``) to
        ``[ego_qpos, ego_qvel, partner_qpos, partner_qvel, cube_pose,
        goal_pos, tcp_pose]``. For the AS-hetero condition the partner
        is fetch (qpos+qvel = ``_FAKE_FETCH_QPOS_DIM * 2`` per the
        Tier-1 fake's encoding); per-condition shape is env-emit-
        dependent and pinned by the Tier-2 ``test_trainer_obs_dim_
        matches_widened_state_per_condition`` test on the real SAPIEN
        env. The partner uid's own ``state`` Box stays at
        ``concat(qpos, qvel)`` of the partner alone — partner does not
        see the task.

        EgoPPOTrainer.from_config derives ``obs_dim =
        ego_state_space.shape[0]`` at construction per cell, so the
        per-condition divergence does not trip any global constant.
        """
        inner = _FakeStateDictEnv(
            condition_id="stage1_pickplace_panda_plus_fetch_ego_aht_happo_per_agent"
        )
        wrap = Stage1ASStateSynthesizer(inner)  # type: ignore[arg-type]
        ego_space = wrap.observation_space["agent"]["panda_wristcam"]["state"]
        assert isinstance(ego_space, gym.spaces.Box)
        expected_ego_dim = (
            _FAKE_EGO_QPOS_DIM * 2
            + _FAKE_FETCH_QPOS_DIM * 2
            + _FAKE_CUBE_POSE_DIM
            + _FAKE_GOAL_POS_DIM
            + _FAKE_TCP_POSE_DIM
        )
        assert ego_space.shape == (expected_ego_dim,)
        assert ego_space.dtype == np.float32
        # Partner's own state stays at concat(qpos, qvel) of fetch alone
        # — no task fields, no cross-agent observation (ADR-009 §Decision
        # direct-observation-of-partner-pose is asymmetric: only the ego
        # sees the partner, not vice-versa).
        partner_space = wrap.observation_space["agent"]["fetch"]["state"]
        assert partner_space.shape == (_FAKE_FETCH_QPOS_DIM * 2,)

    def test_as_homo_injects_widened_ego_state_with_partner_and_task(self) -> None:
        """AS-homo ego state mirrors AS-hetero with panda_partner as the partner."""
        inner = _FakeStateDictEnv(condition_id="stage1_pickplace_panda_only_mappo_shared_param")
        wrap = Stage1ASStateSynthesizer(inner)  # type: ignore[arg-type]
        ego_space = wrap.observation_space["agent"]["panda_wristcam"]["state"]
        expected_ego_dim = (
            _FAKE_EGO_QPOS_DIM * 2
            + _FAKE_PANDA_PARTNER_QPOS_DIM * 2
            + _FAKE_CUBE_POSE_DIM
            + _FAKE_GOAL_POS_DIM
            + _FAKE_TCP_POSE_DIM
        )
        assert ego_space.shape == (expected_ego_dim,)
        partner_space = wrap.observation_space["agent"]["panda_partner"]["state"]
        assert partner_space.shape == (_FAKE_PANDA_PARTNER_QPOS_DIM * 2,)

    def test_observation_concat_order_is_load_bearing(self) -> None:
        """Concat order is pinned positionally.

        Order: ``[ego_qpos, ego_qvel, partner_qpos, partner_qvel,
        cube_pose, goal_pos, tcp_pose]``.

        Load-bearing per ADR-007 §Stage 1b Rev 12 — the audit-trail
        reproducibility of trained-checkpoint diffs across PRs and any
        future targeted feature ablation depend on this order. The
        deterministic-counter fake (``_OFFSET_*`` constants) makes
        each slot's contribution readable by inspection.
        """
        inner = _FakeStateDictEnv(
            condition_id="stage1_pickplace_panda_plus_fetch_ego_aht_happo_per_agent"
        )
        wrap = Stage1ASStateSynthesizer(inner)  # type: ignore[arg-type]
        obs = wrap.observation(_FakeStateDictEnv._sample_obs())
        state = obs["agent"]["panda_wristcam"]["state"]
        assert state.dtype == np.float32

        # Slot 1: ego qpos (offset 100, length 9).
        np.testing.assert_array_equal(
            state[:_FAKE_EGO_QPOS_DIM],
            _OFFSET_EGO_QPOS + np.arange(_FAKE_EGO_QPOS_DIM, dtype=np.float32),
        )
        cursor = _FAKE_EGO_QPOS_DIM
        # Slot 2: ego qvel (offset 200, length 9).
        np.testing.assert_array_equal(
            state[cursor : cursor + _FAKE_EGO_QPOS_DIM],
            _OFFSET_EGO_QVEL + np.arange(_FAKE_EGO_QPOS_DIM, dtype=np.float32),
        )
        cursor += _FAKE_EGO_QPOS_DIM
        # Slot 3: partner (fetch) qpos (offset 300, length 15).
        np.testing.assert_array_equal(
            state[cursor : cursor + _FAKE_FETCH_QPOS_DIM],
            _OFFSET_PARTNER_QPOS + np.arange(_FAKE_FETCH_QPOS_DIM, dtype=np.float32),
        )
        cursor += _FAKE_FETCH_QPOS_DIM
        # Slot 4: partner (fetch) qvel (offset 400, length 15).
        np.testing.assert_array_equal(
            state[cursor : cursor + _FAKE_FETCH_QPOS_DIM],
            _OFFSET_PARTNER_QVEL + np.arange(_FAKE_FETCH_QPOS_DIM, dtype=np.float32),
        )
        cursor += _FAKE_FETCH_QPOS_DIM
        # Slot 5: cube_pose (offset 500, length 7).
        np.testing.assert_array_equal(
            state[cursor : cursor + _FAKE_CUBE_POSE_DIM],
            _OFFSET_CUBE_POSE + np.arange(_FAKE_CUBE_POSE_DIM, dtype=np.float32),
        )
        cursor += _FAKE_CUBE_POSE_DIM
        # Slot 6: goal_pos (offset 600, length 3).
        np.testing.assert_array_equal(
            state[cursor : cursor + _FAKE_GOAL_POS_DIM],
            _OFFSET_GOAL_POS + np.arange(_FAKE_GOAL_POS_DIM, dtype=np.float32),
        )
        cursor += _FAKE_GOAL_POS_DIM
        # Slot 7: tcp_pose (offset 700, length 7).
        np.testing.assert_array_equal(
            state[cursor : cursor + _FAKE_TCP_POSE_DIM],
            _OFFSET_TCP_POSE + np.arange(_FAKE_TCP_POSE_DIM, dtype=np.float32),
        )
        cursor += _FAKE_TCP_POSE_DIM
        assert cursor == state.shape[0]

    def test_partner_state_excludes_task_and_ego(self) -> None:
        """Partner's own ``state`` is ``concat(qpos, qvel)`` only — asymmetry pinned.

        ADR-009 §Decision direct-observation-of-partner-pose is one-
        way: the ego sees the partner, but the partner does not see
        the task or the ego. A regression that symmetrised the
        widening would silently change the ad-hoc claim.
        """
        inner = _FakeStateDictEnv(
            condition_id="stage1_pickplace_panda_plus_fetch_ego_aht_happo_per_agent"
        )
        wrap = Stage1ASStateSynthesizer(inner)  # type: ignore[arg-type]
        obs = wrap.observation(_FakeStateDictEnv._sample_obs())
        partner_state = obs["agent"]["fetch"]["state"]
        assert partner_state.shape == (_FAKE_FETCH_QPOS_DIM * 2,)
        # First half = fetch qpos counter (offset 300), second half =
        # fetch qvel counter (offset 400). No cube / goal / tcp slot.
        np.testing.assert_array_equal(
            partner_state[:_FAKE_FETCH_QPOS_DIM],
            _OFFSET_PARTNER_QPOS + np.arange(_FAKE_FETCH_QPOS_DIM, dtype=np.float32),
        )
        np.testing.assert_array_equal(
            partner_state[_FAKE_FETCH_QPOS_DIM:],
            _OFFSET_PARTNER_QVEL + np.arange(_FAKE_FETCH_QPOS_DIM, dtype=np.float32),
        )

    def test_pass_through_when_inner_env_has_no_condition_id(self) -> None:
        inner = _FakeStateDictEnv(condition_id=None)
        wrap = Stage1ASStateSynthesizer(inner)  # type: ignore[arg-type]
        obs = wrap.observation(_FakeStateDictEnv._sample_obs())
        # No "state" injected — wrapper is inert without a recognised condition_id.
        assert "state" not in obs["agent"]["panda_wristcam"]

    def test_passthrough_under_om_condition(self) -> None:
        inner = _FakeStateDictEnv(condition_id="stage1_pickplace_vision_only")
        wrap = Stage1ASStateSynthesizer(inner)  # type: ignore[arg-type]
        obs = wrap.observation(_FakeStateDictEnv._sample_obs())
        # OM conditions are handled by Stage1OMChannelFilter, not this wrapper.
        # The OM remediation lives in P1.05.6 / issue #177 (the vision-head
        # EgoPPOTrainer extension); this short-circuit must stay verbatim.
        assert "state" not in obs["agent"]["panda_wristcam"]

    def test_loud_fail_when_extra_dict_is_missing_at_construction(self) -> None:
        """Missing ``obs["extra"]`` Dict in the observation_space raises TypeError.

        ADR-007 §Stage 1b Rev 12 loud-fail predicate — silent zero-fill
        would resurrect the Surface 6 class of bug (the whole point of
        widening is that the trainer sees real task signal).
        """
        inner = _FakeStateDictEnv(
            condition_id="stage1_pickplace_panda_plus_fetch_ego_aht_happo_per_agent"
        )
        # Drop the "extra" sub-space.
        broken_spaces = dict(inner.observation_space.spaces)
        del broken_spaces["extra"]
        inner.observation_space = gym.spaces.Dict(broken_spaces)
        with pytest.raises(TypeError, match=r"Stage1ASStateSynthesizer.*'extra'"):
            Stage1ASStateSynthesizer(inner)  # type: ignore[arg-type]

    def test_loud_fail_when_cube_pose_field_is_missing_at_construction(self) -> None:
        """Missing ``obs["extra"]["cube_pose"]`` raises TypeError naming the field."""
        inner = _FakeStateDictEnv(
            condition_id="stage1_pickplace_panda_plus_fetch_ego_aht_happo_per_agent"
        )
        extra_spaces = dict(inner.observation_space["extra"].spaces)
        del extra_spaces["cube_pose"]
        new_spaces = dict(inner.observation_space.spaces)
        new_spaces["extra"] = gym.spaces.Dict(extra_spaces)
        inner.observation_space = gym.spaces.Dict(new_spaces)
        with pytest.raises(TypeError, match="cube_pose"):
            Stage1ASStateSynthesizer(inner)  # type: ignore[arg-type]

    def test_loud_fail_when_extra_field_has_wrong_trailing_dim(self) -> None:
        """A task field with the wrong trailing dim raises TypeError naming the dim mismatch."""
        inner = _FakeStateDictEnv(
            condition_id="stage1_pickplace_panda_plus_fetch_ego_aht_happo_per_agent"
        )
        extra_spaces = dict(inner.observation_space["extra"].spaces)
        # Shrink goal_pos to (1, 2) — wrong trailing dim (expected 3).
        extra_spaces["goal_pos"] = gym.spaces.Box(-np.inf, np.inf, (1, 2), np.float32)
        new_spaces = dict(inner.observation_space.spaces)
        new_spaces["extra"] = gym.spaces.Dict(extra_spaces)
        inner.observation_space = gym.spaces.Dict(new_spaces)
        with pytest.raises(TypeError, match="goal_pos"):
            Stage1ASStateSynthesizer(inner)  # type: ignore[arg-type]

    def test_loud_fail_on_inconsistent_batch_dims_at_observation_time(self) -> None:
        """Mixed leading batch dims across concat inputs raise TypeError.

        P1.05.10 (ADR-007 §Stage 1b regime-alignment revision) extends
        the Rev 12 single-env contract to consistent ``(num_envs, dim)``
        batches; what stays loud-fail is an *inconsistent* batch — one
        field batched at ``num_envs > 1`` while its siblings are not —
        which indicates a wiring bug, not a broadcastable case.
        """
        inner = _FakeStateDictEnv(
            condition_id="stage1_pickplace_panda_plus_fetch_ego_aht_happo_per_agent"
        )
        wrap = Stage1ASStateSynthesizer(inner)  # type: ignore[arg-type]
        bad_obs = _FakeStateDictEnv._sample_obs()
        # Simulate a num_envs=2 batch on the ego's qpos ONLY (siblings
        # stay single-env): inconsistent — must raise.
        bad_obs["agent"]["panda_wristcam"]["qpos"] = np.zeros(
            (2, _FAKE_EGO_QPOS_DIM), dtype=np.float32
        )
        with pytest.raises(TypeError, match=r"batch dims|inconsistent"):
            wrap.observation(bad_obs)

    def test_consistent_batch_emits_batched_state(self) -> None:
        """A consistent ``(num_envs, dim)`` batch synthesises ``(num_envs, state_dim)``.

        P1.05.10 (ADR-007 §Stage 1b regime-alignment revision): the
        vectorised cell's obs path. Per-row content must equal the
        widened concat of that env's row (positional contract of
        Rev 12, applied per env).
        """
        inner = _FakeStateDictEnv(
            condition_id="stage1_pickplace_panda_plus_fetch_ego_aht_happo_per_agent"
        )
        wrap = Stage1ASStateSynthesizer(inner)  # type: ignore[arg-type]
        n = 3
        obs = _FakeStateDictEnv._sample_obs()

        def _batch(arr: np.ndarray, scale: float) -> np.ndarray:  # type: ignore[type-arg]
            base = np.asarray(arr, dtype=np.float32).reshape(1, -1)
            return np.concatenate([base * (i + 1) * scale for i in range(n)], axis=0)

        for uid in ("panda_wristcam", "fetch"):
            for key in ("qpos", "qvel"):
                obs["agent"][uid][key] = _batch(obs["agent"][uid][key], 1.0)
        for key in ("cube_pose", "goal_pos", "tcp_pose"):
            obs["extra"][key] = _batch(obs["extra"][key], 1.0)
        out = wrap.observation(obs)
        ego_state = out["agent"]["panda_wristcam"]["state"]
        partner_state = out["agent"]["fetch"]["state"]
        assert ego_state.ndim == 2
        assert ego_state.shape[0] == n
        assert partner_state.shape[0] == n
        # Row-wise positional contract: row i equals the flat concat of
        # row i's fields in the Rev 12 order.
        for i in range(n):
            expected = np.concatenate(
                [
                    np.asarray(obs["agent"]["panda_wristcam"]["qpos"][i]).ravel(),
                    np.asarray(obs["agent"]["panda_wristcam"]["qvel"][i]).ravel(),
                    np.asarray(obs["agent"]["fetch"]["qpos"][i]).ravel(),
                    np.asarray(obs["agent"]["fetch"]["qvel"][i]).ravel(),
                    np.asarray(obs["extra"]["cube_pose"][i]).ravel(),
                    np.asarray(obs["extra"]["goal_pos"][i]).ravel(),
                    np.asarray(obs["extra"]["tcp_pose"][i]).ravel(),
                ]
            ).astype(np.float32)
            np.testing.assert_allclose(ego_state[i], expected)


# ----- Stage1OMChannelFilter (per-condition obs masking; ADR-007 §Stage 1b OM axis) -----


class _FakeInnerEnv(gym.Env):  # type: ignore[type-arg]
    """Minimal gym.Env shape for filter tests (no SAPIEN dependency).

    Mirrors the real :class:`Stage1PickPlaceEnv`'s ``obs_mode="state_dict"``
    per-agent contract: each uid emits ``qpos`` / ``qvel`` (not the legacy
    ``joint_pos`` / ``joint_vel`` keys). Aligned with the new
    ``_FakeStateDictEnv`` fixture (added for the AS synthesizer's Tier-1
    regression pin) so a future contract-drift bug on the OM axis would
    surface in unit tests rather than only on the first GPU run
    (issue #165 §Finding 2).
    """

    metadata: ClassVar[dict[str, object]] = {"render_modes": []}  # type: ignore[misc]

    def __init__(self, condition_id: str | None) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Dict(
                    {
                        "panda_wristcam": gym.spaces.Dict(
                            {
                                "qpos": gym.spaces.Box(-1, 1, (7,), np.float32),
                                "qvel": gym.spaces.Box(-1, 1, (7,), np.float32),
                            }
                        ),
                        "fetch": gym.spaces.Dict(
                            {
                                "qpos": gym.spaces.Box(-1, 1, (12,), np.float32),
                            }
                        ),
                    }
                ),
                "extra": gym.spaces.Dict(
                    {
                        "tcp_pose": gym.spaces.Box(-1, 1, (7,), np.float32),
                        "goal_pos": gym.spaces.Box(-1, 1, (3,), np.float32),
                        "force_torque": gym.spaces.Box(-1, 1, (6,), np.float32),
                        "cube_pose": gym.spaces.Box(-1, 1, (7,), np.float32),
                    }
                ),
            }
        )
        self.action_space = gym.spaces.Dict({})
        if condition_id is not None:
            self.condition_id = condition_id

    @staticmethod
    def _sample_obs() -> dict:
        return {
            "agent": {
                "panda_wristcam": {
                    "qpos": np.ones(7, dtype=np.float32),
                    "qvel": np.ones(7, dtype=np.float32),
                },
                "fetch": {
                    "qpos": np.ones(12, dtype=np.float32),
                },
            },
            "extra": {
                "tcp_pose": np.full(7, 0.5, dtype=np.float32),
                "goal_pos": np.full(3, 0.3, dtype=np.float32),
                "force_torque": np.full(6, 2.0, dtype=np.float32),
                "cube_pose": np.full(7, 0.1, dtype=np.float32),
            },
        }


class TestStage1OMChannelFilter:
    """OM-axis per-condition obs masking (ADR-007 §Stage 1b OM axis)."""

    def test_pass_through_when_inner_env_has_no_condition_id(self) -> None:
        inner = _FakeInnerEnv(condition_id=None)
        wrap = Stage1OMChannelFilter(inner)  # type: ignore[arg-type]
        obs = wrap.observation(_FakeInnerEnv._sample_obs())
        # No mutation expected — every value matches the input
        # template.
        np.testing.assert_allclose(obs["extra"]["force_torque"], 2.0)
        np.testing.assert_allclose(obs["agent"]["panda_wristcam"]["qpos"], 1.0)

    def test_pass_through_under_as_homo_condition(self) -> None:
        inner = _FakeInnerEnv(condition_id="stage1_pickplace_panda_only_mappo_shared_param")
        wrap = Stage1OMChannelFilter(inner)  # type: ignore[arg-type]
        assert wrap.condition_id == "stage1_pickplace_panda_only_mappo_shared_param"
        obs = wrap.observation(_FakeInnerEnv._sample_obs())
        # AS conditions do not filter — FT + proprio untouched.
        np.testing.assert_allclose(obs["extra"]["force_torque"], 2.0)
        np.testing.assert_allclose(obs["agent"]["panda_wristcam"]["qpos"], 1.0)

    def test_pass_through_under_om_hetero(self) -> None:
        inner = _FakeInnerEnv(condition_id="stage1_pickplace_vision_plus_force_torque_plus_proprio")
        wrap = Stage1OMChannelFilter(inner)  # type: ignore[arg-type]
        obs = wrap.observation(_FakeInnerEnv._sample_obs())
        # OM-hetero keeps everything.
        np.testing.assert_allclose(obs["extra"]["force_torque"], 2.0)
        np.testing.assert_allclose(obs["agent"]["panda_wristcam"]["qpos"], 1.0)

    def test_om_vision_only_zeros_proprio_and_force_torque_but_keeps_tcp_goal(self) -> None:
        inner = _FakeInnerEnv(condition_id="stage1_pickplace_vision_only")
        wrap = Stage1OMChannelFilter(inner)  # type: ignore[arg-type]
        obs = wrap.observation(_FakeInnerEnv._sample_obs())
        # Per-uid proprio zero-masked.
        np.testing.assert_allclose(obs["agent"]["panda_wristcam"]["qpos"], 0.0)
        np.testing.assert_allclose(obs["agent"]["panda_wristcam"]["qvel"], 0.0)
        np.testing.assert_allclose(obs["agent"]["fetch"]["qpos"], 0.0)
        # Force-torque zero-masked.
        np.testing.assert_allclose(obs["extra"]["force_torque"], 0.0)
        # Cube-pose zero-masked (cube state is "state-info" not "vision").
        np.testing.assert_allclose(obs["extra"]["cube_pose"], 0.0)
        # TCP-pose + goal-pos PRESERVED — the policy needs at least the
        # gripper + target pose to be feasible at the 100-step budget.
        np.testing.assert_allclose(obs["extra"]["tcp_pose"], 0.5, rtol=1e-5)
        np.testing.assert_allclose(obs["extra"]["goal_pos"], 0.3, rtol=1e-5)

    def test_shape_preservation_under_filter(self) -> None:
        """Zero-masked channels must keep dtype + shape (TextureFilter parity)."""
        inner = _FakeInnerEnv(condition_id="stage1_pickplace_vision_only")
        wrap = Stage1OMChannelFilter(inner)  # type: ignore[arg-type]
        obs = wrap.observation(_FakeInnerEnv._sample_obs())
        assert obs["agent"]["panda_wristcam"]["qpos"].dtype == np.float32
        assert obs["agent"]["panda_wristcam"]["qpos"].shape == (7,)
        assert obs["extra"]["force_torque"].dtype == np.float32
        assert obs["extra"]["force_torque"].shape == (6,)

    def test_om_vision_only_masks_tensor_leaves_device_aware(self) -> None:
        """Tensor leaves mask via ``torch.zeros_like`` — container type, dtype,
        device, shape all preserved (issue #231; ADR-007 §Stage 1b Rev 17).

        Pre-fix the masking went through ``np.zeros_like``, which raises
        ``TypeError`` on CUDA tensors — the vision-only OM env could not
        ``reset()`` at ``num_envs > 1``. CUDA itself is Tier-2; this pin
        exercises the tensor dispatch branch with CPU tensors (the
        branch taken is type-, not device-, conditional, so the Tier-1
        duck covers the code path the GPU build takes).
        """
        import torch

        inner = _FakeInnerEnv(condition_id="stage1_pickplace_vision_only")
        wrap = Stage1OMChannelFilter(inner)  # type: ignore[arg-type]
        n = 4  # vectorised (num_envs, dim) layout — the failing regime
        obs_in = {
            "agent": {
                "panda_wristcam": {
                    "qpos": torch.ones((n, 7), dtype=torch.float32),
                    "qvel": torch.ones((n, 7), dtype=torch.float64),
                },
            },
            "extra": {
                "tcp_pose": torch.full((n, 7), 0.5, dtype=torch.float32),
                "goal_pos": torch.full((n, 3), 0.3, dtype=torch.float32),
                "force_torque": torch.full((n, 6), 2.0, dtype=torch.float32),
                "cube_pose": torch.full((n, 7), 0.1, dtype=torch.float32),
            },
        }
        obs = wrap.observation(obs_in)
        for masked in (
            obs["agent"]["panda_wristcam"]["qpos"],
            obs["agent"]["panda_wristcam"]["qvel"],
            obs["extra"]["force_torque"],
            obs["extra"]["cube_pose"],
        ):
            assert isinstance(masked, torch.Tensor)
            assert bool((masked == 0).all())
        # dtype + device + shape follow the input leaf.
        assert obs["agent"]["panda_wristcam"]["qpos"].dtype == torch.float32
        assert obs["agent"]["panda_wristcam"]["qpos"].shape == (n, 7)
        assert obs["agent"]["panda_wristcam"]["qpos"].device.type == "cpu"
        assert obs["agent"]["panda_wristcam"]["qvel"].dtype == torch.float64
        # Keep-set untouched: identity, not a copy.
        assert obs["extra"]["tcp_pose"] is obs_in["extra"]["tcp_pose"]
        assert obs["extra"]["goal_pos"] is obs_in["extra"]["goal_pos"]

    def test_om_vision_only_ndarray_path_byte_identical_to_pre_fix(self) -> None:
        """ndarray leaves keep the historical ``np.zeros_like`` masking (ADR-002).

        Issue #231 is a container-type fix only — the CPU/ndarray path's
        masking semantics (values, dtype, shape, container) must be
        byte-identical pre/post fix.
        """
        inner = _FakeInnerEnv(condition_id="stage1_pickplace_vision_only")
        wrap = Stage1OMChannelFilter(inner)  # type: ignore[arg-type]
        obs = wrap.observation(_FakeInnerEnv._sample_obs())
        masked = obs["agent"]["panda_wristcam"]["qpos"]
        assert isinstance(masked, np.ndarray)
        expected = np.zeros_like(_FakeInnerEnv._sample_obs()["agent"]["panda_wristcam"]["qpos"])
        assert masked.tobytes() == expected.tobytes()
        assert masked.dtype == expected.dtype
        assert masked.shape == expected.shape


# ----- Module-level constants + ADR handoff contracts -----


class TestModuleConstants:
    """Pin module-level constants so ADR-007 §Stage 1b handoff contract is testable."""

    def test_default_episode_length_is_100(self) -> None:
        """ADR-007 §Stage 1b compute budget pin."""
        assert DEFAULT_EPISODE_LENGTH == 100

    def test_ft_noise_sigma_default_is_half_newton(self) -> None:
        """Q2 / ADR-007 §Revision history rev. 5 pin.

        sigma=0.5 N lands at the lower bound of the real Franka wrist-FT
        sensor's noise floor (0.5-1.5 N). The value is documented in
        the ADR-007 revision-history paragraph as a Stage-1b
        implementation detail (NOT a prereg edit — tag rotation
        forbidden per ADR-007 §Discipline). This test guards against
        silent edits.
        """
        assert SYNTHESISED_FT_NOISE_SIGMA_N == 0.5

    def test_cartesian_accel_capacity_values_are_finite_and_positive(self) -> None:
        """Both per-embodiment caps are configured (ADR-004 §Decision)."""
        assert np.isfinite(PANDA_CARTESIAN_ACCEL_CAPACITY_MS2)
        assert PANDA_CARTESIAN_ACCEL_CAPACITY_MS2 > 0.0
        assert np.isfinite(FETCH_CARTESIAN_ACCEL_CAPACITY_MS2)
        assert FETCH_CARTESIAN_ACCEL_CAPACITY_MS2 > 0.0
