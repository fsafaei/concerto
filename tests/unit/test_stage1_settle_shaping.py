# SPDX-License-Identifier: Apache-2.0
"""Tier-1 tests for the PBRS settle wrapper (P1.05.11; ADR-007 §Stage 1b Rev 18; issue #232).

Pins the Ng-Harada-Russell form against fakes exposing the privileged
``privileged_settle_state()`` accessor: per-transition
F = gamma*Phi(s') - Phi(s), Phi = -alpha*min(max-arm-|qvel|, cap)*1[placed];
the privileged Phi source (issue #232 — identical Phi trace under both
OM keep-sets, plus an AS-path regression fixture); boundary conventions
(terminal Phi == 0; truncation reads the actual pre-reset final state
through the inside-auto-reset placement; episode starts read the reset
state at step entry); the telescoping identity (the discounted shaping
sum equals -Phi(s_0) + gamma^T*Phi(s_T) for any trajectory — the
invariance mechanism); the construction loud-fails (alpha=0; missing
privileged accessor; ego_uid contradiction); and the ``run_training``
wiring (default-off gate + the inside-auto-reset placement).
"""

from __future__ import annotations

from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
import pytest

from chamber.envs.stage1_shaping import Stage1SettleShapingWrapper
from chamber.envs.stage1_vector import Stage1AutoResetWrapper

_ALPHA = 0.5
_CAP = 0.7
_GAMMA = 0.8
_EGO = "panda_wristcam"


def _priv(qvel7_max: float, *, placed: bool, n: int = 1) -> dict[str, Any]:
    """Fake privileged settle state with a controllable max arm |qvel| and placement."""
    qvel = np.zeros((n, 9), dtype=np.float32)
    qvel[:, 0] = qvel7_max
    qvel[:, 7:] = 5.0  # finger DOF — MUST be excluded from the reduction
    cube = np.zeros((n, 3), dtype=np.float32)
    goal = np.zeros((n, 3), dtype=np.float32)
    if not placed:
        goal[:, 0] = 1.0  # 1 m away — far outside goal_thresh
    return {"ego_qvel": qvel, "cube_pos": cube, "goal_pos": goal}


def _phi(qvel7_max: float, *, placed: bool) -> float:
    return -_ALPHA * min(qvel7_max, _CAP) * (1.0 if placed else 0.0)


class _ScriptedEnv(gym.Env):  # type: ignore[type-arg]
    """Fake env replaying a (priv_state, reward, term, trunc, info) tape.

    The tape entry's privileged state is the live state *after* the
    step — :meth:`privileged_settle_state` always reports the current
    live state, mirroring the real env's handles. ``reset(options=
    {"env_idx": ...})`` (the auto-reset partial-reset surface) swaps
    the live state for ``post_reset_priv``.
    """

    observation_space = gym.spaces.Box(-np.inf, np.inf, (1,), np.float32)
    action_space = gym.spaces.Box(-1.0, 1.0, (1,), np.float32)

    def __init__(
        self,
        reset_priv: dict[str, Any],
        tape: list[tuple],  # type: ignore[type-arg]
        *,
        post_reset_priv: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self._reset_priv = reset_priv
        self._post_reset_priv = post_reset_priv
        self._tape = list(tape)
        self._i = 0
        self._priv = reset_priv

    def privileged_settle_state(self) -> dict[str, Any]:
        return self._priv

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        options = kwargs.get("options")
        if isinstance(options, dict) and "env_idx" in options:
            # Auto-reset partial reset: swap in the post-reset state.
            assert self._post_reset_priv is not None
            self._priv = self._post_reset_priv
            return np.zeros(1, dtype=np.float32), {}
        self._i = 0
        self._priv = self._reset_priv
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action: Any) -> tuple:  # type: ignore[type-arg]
        del action
        priv, reward, term, trunc, info = self._tape[self._i]
        self._i += 1
        self._priv = priv
        return np.zeros(1, dtype=np.float32), reward, term, trunc, info


def _wrap(env: gym.Env) -> Stage1SettleShapingWrapper:  # type: ignore[type-arg]
    return Stage1SettleShapingWrapper(
        env,
        alpha=_ALPHA,
        qvel_cap=_CAP,
        gamma=_GAMMA,
        ego_uid=_EGO,
    )


class TestConstructionLoudFails:
    def test_alpha_zero_refused(self) -> None:
        """Default-off (settle_alpha=0) must never construct the wrapper (ADR-002)."""
        with pytest.raises(ValueError, match="alpha"):
            Stage1SettleShapingWrapper(
                _ScriptedEnv(_priv(0.0, placed=False), []),  # type: ignore[arg-type]
                alpha=0.0,
                qvel_cap=_CAP,
                gamma=_GAMMA,
                ego_uid=_EGO,
            )

    def test_missing_privileged_accessor_refused(self) -> None:
        """Issue #232: Phi's source is the privileged accessor — loud-fail without it."""

        class _NoPrivEnv(gym.Env):  # type: ignore[type-arg]
            observation_space = gym.spaces.Box(-np.inf, np.inf, (1,), np.float32)
            action_space = gym.spaces.Box(-1.0, 1.0, (1,), np.float32)

        with pytest.raises(TypeError, match="privileged_settle_state"):
            _wrap(_NoPrivEnv())  # type: ignore[arg-type]

    def test_ego_uid_contradiction_refused(self) -> None:
        """A mis-wired cell must surface at construction (ADR-007 §Stage 1b Rev 18)."""
        env = _ScriptedEnv(_priv(0.0, placed=False), [])
        env.ego_uid = "somebody_else"  # type: ignore[attr-defined]
        with pytest.raises(ValueError, match="ego_uid"):
            _wrap(env)  # type: ignore[arg-type]


class TestPotentialForm:
    def test_per_transition_f_and_finger_exclusion_and_cap(self) -> None:
        """F = gamma*Phi(s') - Phi(s); fingers excluded; cap applied; gate on placed."""
        # reset (not placed, fast) -> s1 (placed, 0.5) -> s2 (placed, 0.9 >= cap)
        env = _ScriptedEnv(
            _priv(0.6, placed=False),
            [
                (_priv(0.5, placed=True), 1.0, np.array([False]), np.array([False]), {}),
                (_priv(0.9, placed=True), 1.0, np.array([False]), np.array([False]), {}),
            ],
        )
        wrap = _wrap(env)  # type: ignore[arg-type]
        wrap.reset()
        # Phi(reset) = 0 (not placed, despite qvel 0.6 — gate works; finger
        # qvel 5.0 ignored — exclusion works).
        _, r1, _, _, _ = wrap.step(None)
        assert r1 == pytest.approx(1.0 + _GAMMA * _phi(0.5, placed=True) - 0.0)
        # Cap: Phi(s2) uses min(0.9, 0.7).
        _, r2, _, _, _ = wrap.step(None)
        assert r2 == pytest.approx(1.0 + _GAMMA * _phi(0.7, placed=True) - _phi(0.5, placed=True))

    def test_terminal_phi_is_zero(self) -> None:
        """True termination: F = -Phi(s) regardless of the final state (frozen convention)."""
        env = _ScriptedEnv(
            _priv(0.5, placed=True),
            [(_priv(0.6, placed=True), 2.0, np.array([True]), np.array([False]), {})],
        )
        wrap = _wrap(env)  # type: ignore[arg-type]
        wrap.reset()
        _, r, _, _, _ = wrap.step(None)
        assert r == pytest.approx(2.0 + _GAMMA * 0.0 - _phi(0.5, placed=True))

    def test_truncation_reads_pre_reset_final_state_inside_auto_reset(self) -> None:
        """Auto-reset path: Phi(s') is the actual pre-reset final state.

        Composes the wrapper INSIDE the real
        :class:`chamber.envs.stage1_vector.Stage1AutoResetWrapper` —
        the production placement on the vectorised path (issue #232) —
        and pins both halves of the boundary convention: the truncated
        transition's Phi(s') is the pre-reset final state (no
        Phi-zeroing at truncation), and the next transition's Phi(s)
        re-initialises from the post-reset state.
        """
        final = _priv(0.4, placed=True)
        post_reset = _priv(0.0, placed=False)
        env = _ScriptedEnv(
            _priv(0.5, placed=True),
            [
                (final, 1.0, np.array([False]), np.array([True]), {}),
                # Next transition: Phi(s) must have re-initialised from
                # the post-reset state (Phi=0), NOT from the final state.
                (_priv(0.3, placed=True), 1.0, np.array([False]), np.array([False]), {}),
            ],
            post_reset_priv=post_reset,
        )
        chain = Stage1AutoResetWrapper(_wrap(env))  # type: ignore[arg-type]
        chain.reset()
        _, r1, _, trunc1, info1 = chain.step(None)
        assert bool(np.asarray(trunc1).reshape(-1)[0])
        assert "final_observation" in info1  # trainer's Pardo surface intact
        assert r1 == pytest.approx(1.0 + _GAMMA * _phi(0.4, placed=True) - _phi(0.5, placed=True))
        _, r2, _, _, _ = chain.step(None)
        assert r2 == pytest.approx(1.0 + _GAMMA * _phi(0.3, placed=True) - 0.0)

    def test_telescoping_identity(self) -> None:
        """Discounted shaping sum = gamma^T Phi(s_T) - Phi(s_0) (invariance mechanism)."""
        rng = np.random.default_rng(7)
        qs = rng.uniform(0, 1, 12)
        ps = rng.integers(0, 2, 12)
        states = [_priv(float(q), placed=bool(p)) for q, p in zip(qs, ps, strict=True)]
        tape = [(s, 0.0, np.array([False]), np.array([False]), {}) for s in states]
        env = _ScriptedEnv(_priv(0.55, placed=True), tape)
        wrap = _wrap(env)  # type: ignore[arg-type]
        wrap.reset()
        disc_sum = 0.0
        for t in range(len(tape)):
            _, r, _, _, _ = wrap.step(None)
            disc_sum += (_GAMMA**t) * float(np.asarray(r).reshape(-1)[0])
        phi_t = _phi(float(np.float32(qs[-1])), placed=bool(ps[-1]))
        phi_0 = _phi(0.55, placed=True)
        # abs=1e-6: the privileged tensors are float32, so Phi carries
        # ~1e-8-scale quantisation per term; the identity is exact in
        # exact arithmetic.
        assert disc_sum == pytest.approx((_GAMMA ** len(tape)) * phi_t - phi_0, abs=1e-6)

    def test_vectorised_layout(self) -> None:
        """(num_envs,) rewards + (num_envs, dim) privileged state shape through per-env Phi."""
        n = 3
        env = _ScriptedEnv(
            _priv(0.2, placed=False, n=n),
            [
                (
                    _priv(0.5, placed=True, n=n),
                    np.ones(n, dtype=np.float32),
                    np.zeros(n, bool),
                    np.zeros(n, bool),
                    {},
                )
            ],
        )
        wrap = _wrap(env)  # type: ignore[arg-type]
        wrap.reset()
        _, r, _, _, _ = wrap.step(None)
        np.testing.assert_allclose(np.asarray(r), 1.0 + _GAMMA * _phi(0.5, placed=True), rtol=1e-6)


# ----- Privileged Phi source (issue #232) -----


class _FakeOMInnerEnv(gym.Env):  # type: ignore[type-arg]
    """Fake inner env satisfying BOTH the OM filter and the shaping wrapper.

    Emits the real env's obs keys (per-uid ``qpos``/``qvel`` +
    ``extra.cube_pose``/``goal_pos``/``tcp_pose``/``force_torque``) so
    :class:`chamber.envs.stage1_obs_filter.Stage1OMChannelFilter`
    masks it exactly as it masks the real vision-only build, while
    :meth:`privileged_settle_state` reports the (unmasked) live state.
    """

    metadata: ClassVar[dict[str, object]] = {"render_modes": []}  # type: ignore[misc]

    def __init__(self, condition_id: str, tape: list[dict[str, Any]]) -> None:
        super().__init__()
        self.condition_id = condition_id
        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Dict(
                    {
                        _EGO: gym.spaces.Dict(
                            {
                                "qpos": gym.spaces.Box(-np.inf, np.inf, (9,), np.float32),
                                "qvel": gym.spaces.Box(-np.inf, np.inf, (9,), np.float32),
                            }
                        )
                    }
                ),
                "extra": gym.spaces.Dict(
                    {
                        "tcp_pose": gym.spaces.Box(-np.inf, np.inf, (7,), np.float32),
                        "goal_pos": gym.spaces.Box(-np.inf, np.inf, (3,), np.float32),
                        "cube_pose": gym.spaces.Box(-np.inf, np.inf, (7,), np.float32),
                        "force_torque": gym.spaces.Box(-np.inf, np.inf, (6,), np.float32),
                    }
                ),
            }
        )
        self.action_space = gym.spaces.Dict({})
        self._tape = list(tape)
        self._i = 0
        self._priv = tape[0]

    def privileged_settle_state(self) -> dict[str, Any]:
        return self._priv

    def _obs_from_priv(self) -> dict[str, Any]:
        cube_pose = np.concatenate(
            [self._priv["cube_pos"][0], np.array([1, 0, 0, 0], dtype=np.float32)]
        )
        return {
            "agent": {
                _EGO: {
                    "qpos": np.ones(9, dtype=np.float32),
                    "qvel": self._priv["ego_qvel"][0],
                }
            },
            "extra": {
                "tcp_pose": np.full(7, 0.5, dtype=np.float32),
                "goal_pos": self._priv["goal_pos"][0],
                "cube_pose": cube_pose.astype(np.float32),
                "force_torque": np.full(6, 2.0, dtype=np.float32),
            },
        }

    def reset(self, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        del kwargs
        self._i = 0
        self._priv = self._tape[0]
        return self._obs_from_priv(), {}

    def step(self, action: Any) -> tuple:  # type: ignore[type-arg]
        del action
        self._i += 1
        self._priv = self._tape[self._i]
        return (
            self._obs_from_priv(),
            1.0,
            np.array([False]),
            np.array([False]),
            {},
        )


def _shaped_trace(condition_id: str, tape: list[dict[str, Any]]) -> list[float]:
    """Shaped-reward trace of one scripted state sequence under ``condition_id``."""
    from chamber.envs.stage1_obs_filter import Stage1OMChannelFilter

    inner = _FakeOMInnerEnv(condition_id, tape)
    chain = _wrap(Stage1OMChannelFilter(inner))  # type: ignore[arg-type]
    obs, _ = chain.reset()
    rewards: list[float] = []
    masked = bool(np.all(obs["agent"][_EGO]["qvel"] == 0))
    assert masked == (condition_id == "stage1_pickplace_vision_only"), (
        "fixture drift: the OM filter no longer masks the way this pin assumes"
    )
    for _ in range(len(tape) - 1):
        _, r, _, _, _ = chain.step(None)
        rewards.append(float(np.asarray(r).reshape(-1)[0]))
    return rewards


class TestPrivilegedPhiSource:
    """Issue #232: Phi never reads the (condition-filtered) observation."""

    _TAPE: ClassVar[list[dict[str, Any]]] = [
        _priv(0.6, placed=False),
        _priv(0.5, placed=True),
        _priv(0.9, placed=True),
        _priv(0.2, placed=False),
    ]

    def test_phi_trace_identical_under_both_om_keep_sets(self) -> None:
        """ADR-007 §Stage 1b Rev 18 condition-symmetry pin (exact equality).

        The vision-only keep-set zero-masks ``cube_pose`` and the
        per-uid proprio in the observation (asserted inside the
        helper); pre-#232 the obs-derived Phi therefore degenerated to
        0 under vision-only and the shaping differed between the two
        OM conditions. The privileged source makes the traces equal
        by construction — pinned here bit-exactly.
        """
        vision_only = _shaped_trace("stage1_pickplace_vision_only", self._TAPE)
        hetero = _shaped_trace("stage1_pickplace_vision_plus_force_torque_plus_proprio", self._TAPE)
        assert vision_only == hetero  # exact, not approx
        # And the trace is the true potential's, not the degenerate
        # masked one (which would be flat 1.0 everywhere).
        assert vision_only != [1.0, 1.0, 1.0]

    def test_as_path_phi_values_unchanged_against_recorded_fixture(self) -> None:
        """Regression pin: the #232 source fix does not move AS-path Phi values.

        Recorded from the pre-fix obs-derived implementation on the
        same underlying state sequence (alpha=0.5, cap=0.7, gamma=0.8):
        the privileged read returns the same live values the unfiltered
        AS obs carried, so the shaped rewards must be unchanged.
        """
        recorded = [0.8, 0.97, 1.35]
        env = _ScriptedEnv(
            self._TAPE[0],
            [(s, 1.0, np.array([False]), np.array([False]), {}) for s in self._TAPE[1:]],
        )
        wrap = _wrap(env)  # type: ignore[arg-type]
        wrap.reset()
        got = []
        for _ in recorded:
            _, r, _, _, _ = wrap.step(None)
            got.append(float(np.asarray(r).reshape(-1)[0]))
        assert got == pytest.approx(recorded, abs=1e-7)


# ----- run_training wiring -----


class TestRunTrainingWiring:
    def test_default_off_does_not_wrap(self, tmp_path: Any) -> None:
        """shaping.settle_alpha=0 (default) leaves the env unwrapped (ADR-002)."""
        from concerto.training.config import ShapingConfig

        assert ShapingConfig().settle_alpha == 0.0
        # The gate in run_training is `> 0.0`; the wrapper itself refuses
        # alpha<=0 (TestConstructionLoudFails), so a mis-wire cannot
        # silently produce a no-op shaped cell.

    @pytest.mark.parametrize("vectorised", [True, False])
    def test_shaping_seats_inside_auto_reset_on_the_vectorised_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any, *, vectorised: bool
    ) -> None:
        """Issue #232 placement: AutoReset(Shaping(chain)) when vectorised.

        The truncation-boundary convention (Phi(s') = actual final
        state) is realised positionally — the shaping wrapper must sit
        INSIDE the auto-reset wrapper so the done envs have not been
        partially reset when Phi(s') is read. Single-env builds (no
        auto-reset wrapper) keep the shaping outermost.
        """
        import chamber.benchmarks.training_runner as tr
        from concerto.training.config import (
            EgoAHTConfig,
            EnvConfig,
            HAPPOHyperparams,
            PartnerConfig,
            RuntimeConfig,
            ShapingConfig,
        )

        inner = _ScriptedEnv(_priv(0.0, placed=False), [])
        built = Stage1AutoResetWrapper(inner) if vectorised else inner
        captured: dict[str, Any] = {}

        def _fake_build_env(env_cfg: Any, *, root_seed: int) -> Any:
            del env_cfg, root_seed
            return built

        def _fake_train(cfg: Any, *, env: Any, **kwargs: Any) -> Any:
            del cfg, kwargs
            captured["env"] = env
            return "sentinel-training-result"

        monkeypatch.setattr(tr, "build_env", _fake_build_env)
        monkeypatch.setattr(tr, "train", _fake_train)
        cfg = EgoAHTConfig(
            seed=0,
            total_frames=50,
            checkpoint_every=50,
            artifacts_root=tmp_path / "artifacts",
            log_dir=tmp_path / "logs",
            env=EnvConfig(task="mpe_cooperative_push", episode_length=20),
            partner=PartnerConfig(
                class_name="scripted_heuristic",
                extra={"uid": "partner", "target_xy": "0.0,0.0", "action_dim": "2"},
            ),
            happo=HAPPOHyperparams(rollout_length=25),
            runtime=RuntimeConfig(device="cpu"),
            shaping=ShapingConfig(settle_alpha=0.5),
        )
        # The fake env has no ego_uid attr, so the construction-time
        # cross-check is skipped regardless of cfg.env.agent_uids.
        result = tr.run_training(cfg, repo_root=tmp_path)
        assert result == "sentinel-training-result"
        env = captured["env"]
        if vectorised:
            assert isinstance(env, Stage1AutoResetWrapper)
            assert isinstance(env.env, Stage1SettleShapingWrapper)
            assert env.env.env is inner
        else:
            assert isinstance(env, Stage1SettleShapingWrapper)
            assert env.env is inner
