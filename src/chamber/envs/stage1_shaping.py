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
gate is recomputed from the observation (cube/goal positions vs the
env's goal threshold) so the potential never depends on info-dict
plumbing.

Boundary conventions (the bug class this project keeps catching;
pre-stated in the 2026-06-11 PBRS-settle PRESTATEMENT):

- **True termination:** Phi == 0 at the terminal state — the terminal
  transition contributes F = -Phi(s_T).
- **Time-limit truncation:** the episode does not terminate; Phi(s') is
  evaluated on the *actual* final observation. On the vectorised path
  that is ``info["final_observation"]`` from
  :class:`chamber.envs.stage1_vector.Stage1AutoResetWrapper` (the
  returned obs is already the next episode's reset obs); on the
  single-env path the post-step obs *is* the final obs. No Phi-zeroing
  at truncation; the trainer's Pardo-style bootstrap then operates on
  the shaped-reward MDP unchanged.
- **Episode starts:** Phi_prev re-initialises from the reset obs (both
  at ``reset()`` and, per-env, after an auto-reset boundary).

Placement of the transform: applied by
:func:`chamber.benchmarks.training_runner.run_training` around the
*training* env only, gated on ``cfg.shaping.settle_alpha > 0``
(default 0 = wrapper never constructed = byte-identical pre-existing
behaviour, ADR-002). ``Stage1PickPlaceEnv.compute_normalized_dense_reward``
and ``evaluate()`` are byte-untouched; every evaluation instrument
measures unshaped success.

Tier-1 contract (ADR-001): no SAPIEN/ManiSkill imports; torch handled
by duck-typing so the unit tests run on fakes.
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

    See the module docstring for the form, the invariance argument, and
    the boundary conventions. Works for both the single-env layout
    (scalar reward, flat obs) and the vectorised layout (per-env reward
    vector, ``(num_envs, dim)`` obs, auto-reset with
    ``info["final_observation"]``).

    Args:
        env: The training env (full wrapper chain; the AS synthesizer's
            obs layout with per-uid ``qvel`` and ``extra.cube_pose`` /
            ``extra.goal_pos`` is required — loud-fail at first step
            otherwise).
        alpha: Potential scale alpha > 0 (``cfg.shaping.settle_alpha``).
        qvel_cap: Cap on the qvel term in rad/s
            (``cfg.shaping.settle_qvel_cap``).
        gamma: The training MDP's discount (``cfg.happo.gamma``) — NHR
            invariance requires the MDP's own gamma.
        ego_uid: The ego agent's uid (the potential reads its qvel).
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
        self._alpha = float(alpha)
        self._cap = float(qvel_cap)
        self._gamma = float(gamma)
        self._ego_uid = ego_uid
        self._phi_prev: np.ndarray | None = None  # type: ignore[type-arg]

    # ----- potential -----

    def _phi(self, obs: Any) -> np.ndarray:  # type: ignore[type-arg]  # noqa: ANN401 - nested dict of tensors/arrays
        """Phi(s) per env, from the observation alone (state-only; NHR-eligible)."""
        try:
            qvel = _to_np(obs["agent"][self._ego_uid]["qvel"]).reshape(-1, 9)[:, :_ARM_DOF]
            cube = _to_np(obs["extra"]["cube_pose"]).reshape(-1, 7)[:, :3]
            goal = _to_np(obs["extra"]["goal_pos"]).reshape(-1, 3)
        except (KeyError, TypeError) as exc:
            msg = (
                "Stage1SettleShapingWrapper: obs is missing the per-uid 'qvel' or "
                "the 'extra.cube_pose'/'extra.goal_pos' fields the potential "
                "reads. The wrapper requires the Stage-1b AS obs layout "
                "(ADR-007 §Stage 1b Rev 18)."
            )
            raise TypeError(msg) from exc
        speed = np.minimum(np.abs(qvel).max(axis=1), self._cap)
        placed = np.linalg.norm(goal - cube, axis=1) <= _PLACED_GOAL_THRESH_M
        return (-self._alpha * speed * placed).astype(np.float64)

    # ----- gym surface -----

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:  # type: ignore[override]  # noqa: ANN401 - gym reset kwargs
        """Reset and re-initialise Phi_prev from the reset obs (ADR-007 §Stage 1b Rev 18)."""
        obs, info = self.env.reset(**kwargs)
        self._phi_prev = self._phi(obs)
        return obs, info

    def step(  # type: ignore[override]
        self,
        action: Any,  # noqa: ANN401 - dict-of-batched-tensors action
    ) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
        """Add F = gammaPhi(s') - Phi(s) to the reward (ADR-007 §Stage 1b Rev 18).

        Boundary handling per the frozen pre-statement: terminated =>
        Phi(s') = 0; truncated => Phi(s') from ``info["final_observation"]``
        when present (vectorised auto-reset path) else from the
        returned obs (single-env path, where the post-step obs *is*
        the final obs); Phi_prev for the next step re-initialises from
        the post-reset obs on done envs.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward_any: Any = reward
        if self._phi_prev is None:
            msg = "Stage1SettleShapingWrapper: step() before reset()."
            raise RuntimeError(msg)
        term = _to_np(terminated).reshape(-1).astype(bool)
        final_obs = info.get("final_observation") if isinstance(info, dict) else None
        # Phi(s') of the *pre-reset* next state: the auto-reset wrapper has
        # already swapped done envs' obs for the new episode's reset obs,
        # so the true s' for those envs lives in final_observation.
        phi_next = self._phi(final_obs) if final_obs is not None else self._phi(obs)
        phi_next = np.where(term, 0.0, phi_next)  # terminal => Phi == 0
        shaping = self._gamma * phi_next - self._phi_prev
        # Next step's Phi_prev: post-reset obs for done envs (the returned
        # obs is exactly that on the auto-reset path; on the single-env
        # path the training loop calls reset() before the next step,
        # which re-initialises Phi_prev anyway).
        self._phi_prev = self._phi(obs)
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
