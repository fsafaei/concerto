# SPDX-License-Identifier: Apache-2.0
# pyright: reportPrivateImportUsage=false
#
# Same suppression rationale as :mod:`chamber.envs.stage1_shaping`:
# ``torch.as_tensor`` is canonical public API absent from the stub's
# ``__all__``. Suppressed file-locally.
r"""Transport PBRS shaping for the Rung-2 co-carry training cell (ADR-026 §Decision 4).

Rung-2 remediation (COCARRY_RUNG2_REMEDIATION_2026-06-16). A potential-based
reward transform in exact Ng-Harada-Russell form (ICML 1999, Thm 1 — policy
invariance for any state-only potential), sharpening the gradient toward
goal-directed transport of the shared bar:

    F(s, s') = gamma * Phi(s') - Phi(s)
    Phi(s)   = -COCARRY_REWARD_TRANSPORT_PBRS_COEFF * dist(bar_centroid(s), goal)

with ``gamma`` the **training MDP's discount** (the invariance theorem is
stated for the MDP's own gamma) and ``Phi`` a function of state only (the
live bar-centroid-to-goal distance). Because PBRS is policy-invariant it
**cannot change the optimum** — it cannot bias the Rung-2 result, only
speed learning. The transport *outcome* term already in
:meth:`chamber.envs.cocarry.CoCarryEnv.compute_normalized_dense_reward`
stays; this only adds the progress gradient toward it. The keystone of the
remediation — the excess-internal-stress penalty — lives in the env reward;
this wrapper is the supporting term.

Phi source (mirrors :mod:`chamber.envs.stage1_shaping`, issue #232): Phi
reads **privileged env state** via the inner env's
:meth:`privileged_transport_distance` accessor — never the (synthesised)
observation — so training-time reward computation is privileged by
construction and the shaping is independent of the obs wrapper chain.

Boundary conventions (mirrors the settle PBRS pre-statement):

- **True termination:** Phi == 0 at the terminal state (F = -Phi(s)).
- **Time-limit truncation:** the episode does not terminate; Phi(s') is the
  live post-step (final) state — the co-carry cell is single-env, so the
  post-step live state *is* the final state. No Phi-zeroing at truncation;
  the trainer's Pardo-style bootstrap operates on the shaped MDP unchanged.
- **Episode starts:** Phi(s) is read live at step entry, after the
  single-env loop's ``reset()``, so the first transition uses the reset
  state's potential.

The entry/exit live-read pair makes the wrapper **stateless** (no cached
Phi_prev to go stale across a reset).

Placement: applied by
:func:`chamber.benchmarks.training_runner.run_training` around the *training*
env only, gated on ``cfg.shaping.transport_pbrs_coeff > 0`` (default 0 =
wrapper never constructed = byte-identical pre-existing behaviour, ADR-002).
The canonical env reward and ``evaluate()`` are untouched; every evaluation
instrument measures unshaped success.

Tier-1 contract (ADR-001): no SAPIEN/ManiSkill imports; torch handled by
duck-typing so the unit tests run on fakes exposing
``privileged_transport_distance()``.

References:
- ADR-026 §Decision 4 (the Phase-2 forward-design ladder; Rung-2 remediation).
- ADR-002 §Risks #1 (the learning-signal check the shaped curve feeds).
- R-2026-06-B §15 Rung 2; COCARRY_RUNG2_REMEDIATION_2026-06-16 (the note).
- :class:`chamber.envs.stage1_shaping.Stage1SettleShapingWrapper` (the
  parallel settle PBRS this mirrors).
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

__all__ = ["CoCarryTransportPBRSWrapper"]


def _to_np(x: Any) -> np.ndarray:  # type: ignore[type-arg]  # noqa: ANN401 - torch.Tensor or np.ndarray
    """Detach a torch tensor to numpy; pass numpy through (ADR-026 §Decision 4)."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


class CoCarryTransportPBRSWrapper(gym.Wrapper):  # type: ignore[type-arg]
    """Policy-invariant transport PBRS around the co-carry training env (ADR-026 §Decision 4).

    See the module docstring for the form, the invariance argument, the
    privileged Phi source, and the boundary conventions. Works for the
    single-env layout (scalar reward) and, for symmetry with the settle
    wrapper, a per-env reward vector — Phi never touches the observation.

    Args:
        env: The training env (the synthesizer-wrapped co-carry env). Must
            expose a callable ``privileged_transport_distance()`` (forwarded
            from :class:`chamber.envs.cocarry.CoCarryEnv` through the chain's
            ``__getattr__``) — loud-fail at construction otherwise.
        coeff: Potential scale > 0 (``cfg.shaping.transport_pbrs_coeff``,
            which must equal
            :data:`chamber.envs.cocarry.COCARRY_REWARD_TRANSPORT_PBRS_COEFF`).
        gamma: The training MDP's discount (``cfg.happo.gamma``) — NHR
            invariance requires the MDP's own gamma.

    Raises:
        ValueError: If ``coeff <= 0`` (the default-off path must never
            construct the wrapper).
        TypeError: If the env does not expose a callable
            ``privileged_transport_distance`` (the Phi-source contract).
    """

    def __init__(self, env: gym.Env[Any, Any], *, coeff: float, gamma: float) -> None:
        """Bind the potential's constants (ADR-026 §Decision 4)."""
        super().__init__(env)
        if coeff <= 0.0:
            msg = (
                "CoCarryTransportPBRSWrapper: coeff must be > 0 — the default-off "
                "path (shaping.transport_pbrs_coeff=0) must not construct the wrapper "
                "(ADR-002 byte-identity; ADR-026 §Decision 4)."
            )
            raise ValueError(msg)
        if not callable(getattr(env, "privileged_transport_distance", None)):
            msg = (
                "CoCarryTransportPBRSWrapper: env does not expose "
                "privileged_transport_distance(). Phi reads privileged env state — "
                "never the (synthesised) observation — so the shaping is "
                "obs-wrapper-independent (ADR-026 §Decision 4; mirrors issue #232)."
            )
            raise TypeError(msg)
        self._coeff = float(coeff)
        self._gamma = float(gamma)

    def _phi(self) -> np.ndarray:  # type: ignore[type-arg]
        """Phi(s) per env from privileged live state (state-only; NHR-eligible)."""
        dist = _to_np(self.env.privileged_transport_distance()).reshape(-1)  # type: ignore[attr-defined]
        return (-self._coeff * dist).astype(np.float64)

    def step(  # type: ignore[override]
        self,
        action: Any,  # noqa: ANN401 - dict-of-batched-tensors action
    ) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
        """Add ``F = gamma*Phi(s') - Phi(s)`` to the reward (ADR-026 §Decision 4).

        Phi(s) is read live at entry (post-any-reset, so episode starts use
        the reset state's potential) and Phi(s') live at exit, with terminal
        states zeroed (F = -Phi(s) on true termination).
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

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401 - mirrors the co-carry wrappers' forwarding
        """Forward attribute access to the inner env (Tier-2 caller contract)."""
        if name == "env":
            raise AttributeError(name)
        return getattr(self.env, name)
