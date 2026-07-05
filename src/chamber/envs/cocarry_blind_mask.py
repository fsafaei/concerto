# SPDX-License-Identifier: Apache-2.0
r"""Partner-blind ego observation mask for the co-carry env (ADR-011 §Decision as amended; ADR-027).

B-BLIND (ADR-011 §Decision as amended 2026-07-05) is the partner-blind
PPO ego: the same trainer, the same reward, the same action interface as
B-AHT, with **partner state and coupling feedback masked from the
observation**. This module is the masking wrapper — a pure observation
transform layered *outside*
:class:`chamber.envs.cocarry_obs.CoCarryEgoStateSynthesizer`; no trainer
changes (the actor keeps its 46-D input width, so B-BLIND and B-AHT
checkpoints stay layout-compatible and the eval machinery drives both
identically).

**Exactly which observation slice is masked** (the committed definition;
the campaign prereg quotes these indices):

The synthesised 46-D ego ``state`` layout is
``[ego_qpos(0:9), ego_qvel(9:18), partner_qpos(18:27),
partner_qvel(27:36), bar_pose(36:43), goal_pos(43:46)]``
(:mod:`chamber.envs.cocarry_obs`). The wrapper zeroes indices
``[18, 43)``:

- ``partner_qpos`` + ``partner_qvel`` (18:36) — the partner state.
- ``bar_pose`` (36:43) — the coupling feedback. The bar is rigidly
  welded to both grippers, so its pose (position *and* tilt) is the
  channel through which the partner's motion and the internal coupling
  stress are observable (:mod:`chamber.envs.cocarry_obs` documents bar
  tilt / wrist stress as recoverable from ``bar_pose``).

What remains — ``ego_qpos`` + ``ego_qvel`` + ``goal_pos`` — is exactly
the information set of the scripted A3 blind ego
(:class:`chamber.partners.cocarry_blind.CoCarryBlindImpedancePartner`:
world goal + own joints), so the learned and scripted B-BLIND variants
test the same construct. Known asymmetry, stated for the prereg: the
learned ego's *own measured* ``qpos``/``qvel`` still carry load-induced
deviation (the scripted variant dead-reckons them away); masking the
ego's own proprioception would make the policy input non-Markovian and
untrainable, so the residual proprioceptive coupling channel is
accepted and documented rather than hidden.

Only the **ego seat's** ``state`` leaf is touched. Every raw leaf
(``qpos`` / ``qvel`` / ``obs["extra"]``) and the partner's ``state``
pass through byte-identically, so the frozen scripted partner keeps
reading the same obs (ADR-009 §Decision) and telemetry-driven
evaluation is unaffected. Evaluation of a B-BLIND checkpoint MUST apply
this same wrapper — the actor was trained on the masked interface, and
an unmasked eval would be a train/eval interface mismatch, not a
measurement.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import gymnasium as gym
import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray

#: Half-open ``[start, stop)`` index range zeroed in the ego's 46-D
#: ``state`` vector (ADR-011 §Decision as amended: partner state
#: ``partner_qpos``+``partner_qvel`` at 18:36 plus coupling feedback
#: ``bar_pose`` at 36:43). Load-bearing for the campaign prereg and the
#: Tier-1 positional regression test — do not renumber without
#: re-deriving from :mod:`chamber.envs.cocarry_obs`'s concat order.
COCARRY_BLIND_MASK_START: int = 18
COCARRY_BLIND_MASK_STOP: int = 43


def mask_ego_state(state: NDArray[np.float32]) -> NDArray[np.float32]:
    """Zero the partner-state + coupling-feedback slice of an ego state (ADR-011 §Decision).

    Accepts the flat 1-D single-env layout or the batched
    ``(num_envs, dim)`` layout; returns a fresh array (the input is not
    mutated — the raw obs leaves are shared with the partner seat).

    Args:
        state: The synthesised ego ``state`` vector(s).

    Returns:
        A copy with indices ``[COCARRY_BLIND_MASK_START,
        COCARRY_BLIND_MASK_STOP)`` of the trailing axis zeroed.
    """
    masked = np.array(state, dtype=np.float32, copy=True)
    masked[..., COCARRY_BLIND_MASK_START:COCARRY_BLIND_MASK_STOP] = 0.0
    return masked


class CoCarryEgoBlindMask(gym.ObservationWrapper):  # type: ignore[type-arg]
    """Zero the ego-state partner + coupling slice (B-BLIND; ADR-011 §Decision as amended).

    Layered outside :class:`chamber.envs.cocarry_obs.CoCarryEgoStateSynthesizer`
    (which must already have injected ``obs["agent"][ego_uid]["state"]``).
    The observation space is unchanged — the masked entries keep their
    slots so B-BLIND / B-AHT actors share one input width (no trainer
    changes; ADR-011 §Decision as amended).

    Raises:
        TypeError: At construction, when the inner env does not expose
            an ego ``state`` entry (the synthesizer is missing — the
            wrapper order is load-bearing).
    """

    def __init__(self, env: gym.Env[Any, Any]) -> None:
        """Bind the ego uid and validate the synthesizer ran first (ADR-011 §Decision)."""
        super().__init__(env)
        self._ego_uid: str = str(env.get_wrapper_attr("ego_uid"))
        obs_space = env.observation_space
        try:
            ego_state_space = obs_space["agent"][self._ego_uid]["state"]  # type: ignore[index]
        except (KeyError, TypeError) as exc:
            msg = (
                "CoCarryEgoBlindMask requires the inner env to already expose "
                f"observation_space['agent'][{self._ego_uid!r}]['state'] — wrap "
                "CoCarryEgoStateSynthesizer first (ADR-011 §Decision as amended; "
                "ADR-026 §Decision 1)."
            )
            raise TypeError(msg) from exc
        state_dim = int(np.asarray(ego_state_space.shape)[-1])
        if state_dim < COCARRY_BLIND_MASK_STOP:
            msg = (
                f"CoCarryEgoBlindMask: ego state dim {state_dim} is narrower than the "
                f"masked slice [{COCARRY_BLIND_MASK_START}, {COCARRY_BLIND_MASK_STOP}) "
                "— the co-carry ego-state concat order has changed; re-derive the "
                "mask indices from chamber.envs.cocarry_obs (ADR-011 §Decision)."
            )
            raise TypeError(msg)

    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:  # type: ignore[override]
        """Zero the ego ``state`` slice; raw leaves untouched (ADR-011 §Decision as amended)."""
        agent = observation.get("agent")
        if not isinstance(agent, dict):
            return observation
        ego_sub = agent.get(self._ego_uid)
        if not isinstance(ego_sub, dict) or "state" not in ego_sub:
            msg = (
                f"CoCarryEgoBlindMask: runtime obs missing "
                f"obs['agent'][{self._ego_uid!r}]['state'] — the synthesizer must "
                "run inside this wrapper (ADR-011 §Decision as amended)."
            )
            raise TypeError(msg)
        masked_sub = dict(ego_sub)
        masked_sub["state"] = mask_ego_state(np.asarray(ego_sub["state"], dtype=np.float32))
        new_agent = dict(agent)
        new_agent[self._ego_uid] = masked_sub
        out = dict(observation)
        out["agent"] = new_agent
        return out

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401 - mirrors gym.Wrapper forwarding
        """Forward inner-env attributes (``ego_uid`` / ``get_telemetry`` / …).

        Mirrors :meth:`chamber.envs.cocarry_obs.CoCarryEgoStateSynthesizer.__getattr__`
        (Gymnasium 1.3 removed implicit forwarding).
        """
        if name == "env":
            raise AttributeError(name)
        return getattr(self.env, name)


__all__ = [
    "COCARRY_BLIND_MASK_START",
    "COCARRY_BLIND_MASK_STOP",
    "CoCarryEgoBlindMask",
    "mask_ego_state",
]
