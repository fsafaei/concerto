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


class _FakeStateDictEnv(gym.Env):  # type: ignore[type-arg]
    """Mirrors ManiSkill v3 ``obs_mode="state_dict"`` shape for the AS axis.

    The real Stage1PickPlaceEnv under AS conditions emits per-agent
    ``Dict(qpos, qvel, ...)``; this fake matches that contract so the
    Tier-1 wrapper test exercises the same code path the Tier-2 SAPIEN
    test exercises end-to-end (regression-pin against further drift).
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
                                "qpos": gym.spaces.Box(-np.inf, np.inf, (1, 9), np.float32),
                                "qvel": gym.spaces.Box(-np.inf, np.inf, (1, 9), np.float32),
                            }
                        ),
                        "fetch": gym.spaces.Dict(
                            {
                                "qpos": gym.spaces.Box(-np.inf, np.inf, (1, 15), np.float32),
                                "qvel": gym.spaces.Box(-np.inf, np.inf, (1, 15), np.float32),
                            }
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
        return {
            "agent": {
                "panda_wristcam": {
                    "qpos": np.arange(9, dtype=np.float32).reshape(1, 9),
                    "qvel": np.arange(9, 18, dtype=np.float32).reshape(1, 9),
                },
                "fetch": {
                    "qpos": np.zeros((1, 15), dtype=np.float32),
                    "qvel": np.ones((1, 15), dtype=np.float32),
                },
            },
        }


class TestStage1ASStateSynthesizer:
    """Synthesised ``state`` key for AS conditions (ADR-007 §Stage 1b AS axis)."""

    def test_as_hetero_injects_state_box_with_qpos_plus_qvel_dim(self) -> None:
        """The ego-state Box's ``shape[0]`` must equal qpos_dim + qvel_dim.

        EgoPPOTrainer.from_config derives ``obs_dim = ego_state_space.shape[0]``
        (chamber.benchmarks.ego_ppo_trainer:608) — pinning the shape here
        is the regression guard against the P1.04 SAPIEN-gated failure
        rediscovering itself.
        """
        inner = _FakeStateDictEnv(
            condition_id="stage1_pickplace_panda_plus_fetch_ego_aht_happo_per_agent"
        )
        wrap = Stage1ASStateSynthesizer(inner)  # type: ignore[arg-type]
        ego_space = wrap.observation_space["agent"]["panda_wristcam"]["state"]
        assert isinstance(ego_space, gym.spaces.Box)
        assert ego_space.shape == (18,)  # 9 qpos + 9 qvel
        assert ego_space.dtype == np.float32
        partner_space = wrap.observation_space["agent"]["fetch"]["state"]
        assert partner_space.shape == (30,)  # 15 qpos + 15 qvel

    def test_observation_concat_order_qpos_then_qvel(self) -> None:
        """ManiSkill v3 obs_mode="state" concat orders qpos then qvel."""
        inner = _FakeStateDictEnv(
            condition_id="stage1_pickplace_panda_plus_fetch_ego_aht_happo_per_agent"
        )
        wrap = Stage1ASStateSynthesizer(inner)  # type: ignore[arg-type]
        obs = wrap.observation(_FakeStateDictEnv._sample_obs())
        state = obs["agent"]["panda_wristcam"]["state"]
        assert state.shape == (18,)
        assert state.dtype == np.float32
        # qpos = arange(9), qvel = arange(9, 18); concat preserves that.
        np.testing.assert_array_equal(state, np.arange(18, dtype=np.float32))

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
        assert "state" not in obs["agent"]["panda_wristcam"]

    def test_as_homo_also_synthesizes_state(self) -> None:
        inner = _FakeStateDictEnv(condition_id="stage1_pickplace_panda_only_mappo_shared_param")
        wrap = Stage1ASStateSynthesizer(inner)  # type: ignore[arg-type]
        obs = wrap.observation(_FakeStateDictEnv._sample_obs())
        assert obs["agent"]["panda_wristcam"]["state"].shape == (18,)


# ----- Stage1OMChannelFilter (per-condition obs masking; ADR-007 §Stage 1b OM axis) -----


class _FakeInnerEnv(gym.Env):  # type: ignore[type-arg]
    """Minimal gym.Env shape for filter tests (no SAPIEN dependency)."""

    metadata: ClassVar[dict[str, object]] = {"render_modes": []}  # type: ignore[misc]

    def __init__(self, condition_id: str | None) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Dict(
                    {
                        "panda_wristcam": gym.spaces.Dict(
                            {
                                "joint_pos": gym.spaces.Box(-1, 1, (7,), np.float32),
                                "joint_vel": gym.spaces.Box(-1, 1, (7,), np.float32),
                            }
                        ),
                        "fetch": gym.spaces.Dict(
                            {
                                "joint_pos": gym.spaces.Box(-1, 1, (12,), np.float32),
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
                    "joint_pos": np.ones(7, dtype=np.float32),
                    "joint_vel": np.ones(7, dtype=np.float32),
                },
                "fetch": {
                    "joint_pos": np.ones(12, dtype=np.float32),
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
        np.testing.assert_allclose(obs["agent"]["panda_wristcam"]["joint_pos"], 1.0)

    def test_pass_through_under_as_homo_condition(self) -> None:
        inner = _FakeInnerEnv(condition_id="stage1_pickplace_panda_only_mappo_shared_param")
        wrap = Stage1OMChannelFilter(inner)  # type: ignore[arg-type]
        assert wrap.condition_id == "stage1_pickplace_panda_only_mappo_shared_param"
        obs = wrap.observation(_FakeInnerEnv._sample_obs())
        # AS conditions do not filter — FT + proprio untouched.
        np.testing.assert_allclose(obs["extra"]["force_torque"], 2.0)
        np.testing.assert_allclose(obs["agent"]["panda_wristcam"]["joint_pos"], 1.0)

    def test_pass_through_under_om_hetero(self) -> None:
        inner = _FakeInnerEnv(condition_id="stage1_pickplace_vision_plus_force_torque_plus_proprio")
        wrap = Stage1OMChannelFilter(inner)  # type: ignore[arg-type]
        obs = wrap.observation(_FakeInnerEnv._sample_obs())
        # OM-hetero keeps everything.
        np.testing.assert_allclose(obs["extra"]["force_torque"], 2.0)
        np.testing.assert_allclose(obs["agent"]["panda_wristcam"]["joint_pos"], 1.0)

    def test_om_vision_only_zeros_proprio_and_force_torque_but_keeps_tcp_goal(self) -> None:
        inner = _FakeInnerEnv(condition_id="stage1_pickplace_vision_only")
        wrap = Stage1OMChannelFilter(inner)  # type: ignore[arg-type]
        obs = wrap.observation(_FakeInnerEnv._sample_obs())
        # Per-uid proprio zero-masked.
        np.testing.assert_allclose(obs["agent"]["panda_wristcam"]["joint_pos"], 0.0)
        np.testing.assert_allclose(obs["agent"]["panda_wristcam"]["joint_vel"], 0.0)
        np.testing.assert_allclose(obs["agent"]["fetch"]["joint_pos"], 0.0)
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
        assert obs["agent"]["panda_wristcam"]["joint_pos"].dtype == np.float32
        assert obs["agent"]["panda_wristcam"]["joint_pos"].shape == (7,)
        assert obs["extra"]["force_torque"].dtype == np.float32
        assert obs["extra"]["force_torque"].shape == (6,)


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
