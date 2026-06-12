# SPDX-License-Identifier: Apache-2.0
# pyright: reportPrivateImportUsage=false
#
# Same suppression rationale as :mod:`chamber.envs.stage1_vector`:
# ``torch.as_tensor`` is canonical public API absent from the stub's
# ``__all__``. Suppressed file-locally.
"""Potential-based settle shaping for the Stage-1b training cell (ADR-007 §Stage 1b Rev 18).

P1.05.11 PBRS-settle slice. Implements the founder-approved
potential-based reward transform in exact Ng-Harada-Russell form
(ICML 1999, Thm 1 — policy invariance for any state-only potential):

    F(s, s') = gamma * Phi(s') - Phi(s)
    Phi(s)   = -alpha * min(max-arm-|qvel|(s), cap) * 1[is_obj_placed(s)]

with ``gamma`` the **training MDP's discount** (the invariance theorem
is stated for the MDP's own gamma) and both Phi factors functions of
the state only: the qvel reduction is the ``is_static`` predicate's own
max over the ego's 7 arm joints (fingers excluded), and the placement
gate is the cube/goal distance against the env's goal threshold.

Phi source (issue #232; ADR-007 §Stage 1b Rev 18 condition-symmetry
mandate). Phi reads **privileged env state** via the inner env's
``privileged_settle_state()`` accessor — the same live handles
``compute_normalized_dense_reward`` uses — never the (possibly
condition-filtered) observation. Training-time reward computation is
privileged by construction. Pre-fix, Phi read ``obs["extra"]["cube_pose"]``
/ ``goal_pos`` / per-uid ``qvel`` from the *outer* (filtered) obs; the
OM vision-only keep-set zero-masks all three, so the shaping would have
differed between the two OM conditions. AS conditions are unaffected
(neither AS condition masks; the privileged values are the same tensors
the obs carried).

Boundary conventions (the bug class this project keeps catching;
pre-stated in the 2026-06-11 PBRS-settle PRESTATEMENT — unchanged by
the #232 source fix, only their realisation moved):

- **True termination:** Phi == 0 at the terminal state — the terminal
  transition contributes F = -Phi(s_T).
- **Time-limit truncation:** the episode does not terminate; Phi(s') is
  evaluated on the *actual* final state. Realised positionally: on the
  vectorised path the wrapper sits **inside**
  :class:`chamber.envs.stage1_vector.Stage1AutoResetWrapper` (see
  :func:`chamber.benchmarks.training_runner.run_training`), so the
  live state at step exit is still the pre-reset final state; on the
  single-env path the post-step live state *is* the final state. No
  Phi-zeroing at truncation; the trainer's Pardo-style bootstrap then
  operates on the shaped-reward MDP unchanged.
- **Episode starts:** Phi(s) is read live at step *entry*, after any
  auto-reset from the previous step (and after the single-env loop's
  own ``reset()``), so the first transition of every episode uses the
  reset state's potential.

The entry/exit live-read pair makes the wrapper **stateless** — there
is no cached Phi_prev to go stale across an auto-reset boundary.

Placement of the transform: applied by
:func:`chamber.benchmarks.training_runner.run_training` around the
*training* env only — inside the auto-reset wrapper on the vectorised
path, outermost on the single-env path — gated on
``cfg.shaping.settle_alpha > 0`` (default 0 = wrapper never constructed
= byte-identical pre-existing behaviour, ADR-002).
``Stage1PickPlaceEnv.compute_normalized_dense_reward`` and
``evaluate()`` are byte-untouched; every evaluation instrument measures
unshaped success.

Tier-1 contract (ADR-001): no SAPIEN/ManiSkill imports; torch handled
by duck-typing so the unit tests run on fakes exposing
``privileged_settle_state()``.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

__all__ = ["Stage1SettleShapingWrapper"]

#: ``Stage1PickPlaceEnv``'s placement threshold (``self._goal_thresh``,
#: upstream-matched PickCube value). Pinned here so the potential's
#: placement gate is the same state predicate ``evaluate()`` applies;
#: the wrapper loud-fails at construction if the env exposes a
#: different value.
_PLACED_GOAL_THRESH_M: float = 0.025

#: Number of ego arm joints in the ``is_static`` reduction (the 2
#: finger DOF are excluded, matching ``Panda.is_static``).
_ARM_DOF: int = 7


def _to_np(x: Any) -> np.ndarray:  # type: ignore[type-arg]  # noqa: ANN401 - torch.Tensor or np.ndarray depending on env backend
    """Detach a torch tensor to numpy; pass numpy through (ADR-007 §Stage 1b Rev 18)."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


class Stage1SettleShapingWrapper(gym.Wrapper):  # type: ignore[type-arg]
    """PBRS settle term around the Stage-1b training env (ADR-007 §Stage 1b Rev 18).

    See the module docstring for the form, the invariance argument, the
    privileged Phi source (issue #232), and the boundary conventions.
    Works for both the single-env layout (scalar reward) and the
    vectorised layout (per-env reward vector) — Phi never touches the
    observation, so the per-condition obs filtering is irrelevant to
    the shaping by construction.

    Args:
        env: The training env. On the vectorised path this is the
            wrapper chain *inside*
            :class:`chamber.envs.stage1_vector.Stage1AutoResetWrapper`
            (so the truncation-boundary Phi(s') reads the pre-reset
            state); on the single-env path it is the full chain. Must
            expose ``privileged_settle_state()`` (forwarded from
            :class:`chamber.envs.stage1_pickplace.Stage1PickPlaceEnv`
            through the chain's ``__getattr__``) — loud-fail at
            construction otherwise.
        alpha: Potential scale alpha > 0 (``cfg.shaping.settle_alpha``).
        qvel_cap: Cap on the qvel term in rad/s
            (``cfg.shaping.settle_qvel_cap``).
        gamma: The training MDP's discount (``cfg.happo.gamma``) — NHR
            invariance requires the MDP's own gamma.
        ego_uid: The ego agent's uid. The potential reads the ego's
            qvel through the privileged accessor (which is ego-routed
            inside the env); the kwarg is validated against the env's
            own ``ego_uid`` when the env exposes one, so a mis-wired
            cell surfaces at construction.

    Raises:
        ValueError: If ``alpha <= 0`` (default-off must never construct
            the wrapper), if the env's goal threshold diverges from the
            pinned predicate value, or if ``ego_uid`` contradicts the
            env's own ``ego_uid``.
        TypeError: If the env does not expose a callable
            ``privileged_settle_state`` (the Phi source contract;
            issue #232).
    """

    def __init__(
        self,
        env: gym.Env[Any, Any],
        *,
        alpha: float,
        qvel_cap: float,
        gamma: float,
        ego_uid: str,
    ) -> None:
        """Bind the potential's constants (ADR-007 §Stage 1b Rev 18)."""
        super().__init__(env)
        if alpha <= 0.0:
            msg = (
                "Stage1SettleShapingWrapper: alpha must be > 0 — the default-off "
                "path (shaping.settle_alpha=0) must not construct the wrapper "
                "(ADR-002 byte-identity; ADR-007 §Stage 1b Rev 18)."
            )
            raise ValueError(msg)
        env_thresh = getattr(env, "_goal_thresh", None)
        thresh_tol = 1e-9
        if env_thresh is not None and abs(float(env_thresh) - _PLACED_GOAL_THRESH_M) > thresh_tol:
            msg = (
                f"Stage1SettleShapingWrapper: env goal threshold {env_thresh} != "
                f"pinned {_PLACED_GOAL_THRESH_M}. The potential's placement gate "
                "must be the same state predicate evaluate() applies "
                "(ADR-007 §Stage 1b Rev 18)."
            )
            raise ValueError(msg)
        if not callable(getattr(env, "privileged_settle_state", None)):
            msg = (
                "Stage1SettleShapingWrapper: env does not expose "
                "privileged_settle_state(). Phi reads privileged env state — "
                "never the (possibly condition-filtered) observation — so the "
                "shaping is identical across OM keep-sets (issue #232; "
                "ADR-007 §Stage 1b Rev 18 condition-symmetry mandate)."
            )
            raise TypeError(msg)
        env_ego_uid = getattr(env, "ego_uid", None)
        if env_ego_uid is not None and env_ego_uid != ego_uid:
            msg = (
                f"Stage1SettleShapingWrapper: ego_uid={ego_uid!r} contradicts the "
                f"env's own ego_uid={env_ego_uid!r}. The potential is ego-routed; "
                "a mis-wired cell must surface at construction "
                "(ADR-007 §Stage 1b Rev 18)."
            )
            raise ValueError(msg)
        self._alpha = float(alpha)
        self._cap = float(qvel_cap)
        self._gamma = float(gamma)
        self._ego_uid = ego_uid

    # ----- potential -----

    def _phi(self) -> np.ndarray:  # type: ignore[type-arg]
        """Phi(s) per env, from privileged live state (state-only; NHR-eligible).

        Issue #232: training-time reward computation is privileged by
        construction — the reads come from the env's
        ``privileged_settle_state()`` (the same live handles
        ``compute_normalized_dense_reward`` uses), never from the
        filtered observation, so Phi is identical across OM keep-sets.
        """
        state = self.env.privileged_settle_state()  # type: ignore[attr-defined]
        qvel = _to_np(state["ego_qvel"]).reshape(-1, 9)[:, :_ARM_DOF]
        cube = _to_np(state["cube_pos"]).reshape(-1, 3)
        goal = _to_np(state["goal_pos"]).reshape(-1, 3)
        speed = np.minimum(np.abs(qvel).max(axis=1), self._cap)
        placed = np.linalg.norm(goal - cube, axis=1) <= _PLACED_GOAL_THRESH_M
        return (-self._alpha * speed * placed).astype(np.float64)

    # ----- gym surface -----

    def step(  # type: ignore[override]
        self,
        action: Any,  # noqa: ANN401 - dict-of-batched-tensors action
    ) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
        """Add F = gamma*Phi(s') - Phi(s) to the reward (ADR-007 §Stage 1b Rev 18).

        Phi(s) is read live at entry (post-any-reset, so episode starts
        use the reset state's potential) and Phi(s') live at exit.
        Boundary handling per the frozen pre-statement: terminated =>
        Phi(s') = 0; truncated => the exit read *is* the actual final
        state (on the vectorised path the wrapper sits inside the
        auto-reset wrapper, so the done envs have not been reset yet;
        on the single-env path the post-step state is the final state).
        """
        phi_prev = self._phi()
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward_any: Any = reward
        term = _to_np(terminated).reshape(-1).astype(bool)
        phi_next = np.where(term, 0.0, self._phi())  # terminal => Phi == 0
        shaping = self._gamma * phi_next - phi_prev
        if hasattr(reward_any, "detach"):
            import torch  # noqa: PLC0415 - lazy: Tier-1 import safety

            shaped = reward_any + torch.as_tensor(
                shaping, dtype=reward_any.dtype, device=reward_any.device
            ).reshape(reward_any.shape)
        else:
            shaped = reward_any + (float(shaping[0]) if np.ndim(reward_any) == 0 else shaping)
        return obs, shaped, terminated, truncated, info

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401 - mirrors the Stage1 wrappers' forwarding contract
        """Forward attribute access to the inner env (Tier-2 caller contract)."""
        if name == "env":
            raise AttributeError(name)
        return getattr(self.env, name)
