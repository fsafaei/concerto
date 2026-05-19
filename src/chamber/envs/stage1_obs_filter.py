# SPDX-License-Identifier: Apache-2.0
# pyright: reportPrivateImportUsage=false
#
# Same rationale as :mod:`chamber.benchmarks.stage0_smoke`:
# ``torch.as_tensor`` is exported but not advertised in torch's stub
# ``__all__``. Suppressed file-locally so the synthesised-state helper
# stays free of per-line ``type: ignore`` noise.
"""Stage-1 observation wrappers (ADR-007 §Stage 1b).

Two wrappers, applied in fixed order by
:func:`chamber.envs.stage1_pickplace.make_stage1_pickplace_env`:

1. :class:`Stage1ASStateSynthesizer` — synthesises
   ``obs["agent"][uid]["state"]`` as ``concat(qpos, qvel)`` for AS
   conditions, bridging ManiSkill v3 ``obs_mode="state_dict"`` (which
   emits per-agent ``Dict(qpos, qvel, ...)``) to
   :class:`chamber.benchmarks.ego_ppo_trainer.EgoPPOTrainer`'s
   ``obs["agent"][ego_uid]["state"]`` contract. Pass-through for OM
   conditions and for envs without a recognised ``condition_id``.
2. :class:`Stage1OMChannelFilter` — per-condition channel keep-set for
   the OM axis. The Stage-1 AS conditions don't filter at all (they
   use ``obs_mode="state_dict"`` which already excludes the camera
   data the OM axis differentiates on); the wrapper passes those
   conditions through untouched.

For the two OM conditions:

- ``stage1_pickplace_vision_only`` — keep ``obs["sensor_data"]`` (the
  RGB-D camera channels) plus the task-extra fields the policy needs
  to be functional at the Stage-1b episode budget (``tcp_pose``,
  ``goal_pos``). **Zero-mask** the agent-proprio sub-dicts under
  ``obs["agent"][uid]`` and the synthesised ``obs["extra"]["force_torque"]``
  channel.
- ``stage1_pickplace_vision_plus_force_torque_plus_proprio`` —
  pass-through (no filtering); the policy sees everything.

ManiSkill v3 obs layout reminder (``mani_skill/envs/sapien_env.py:546-630``):
``obs["agent"][uid]`` carries proprio (state, joint_pos, joint_vel);
``obs["extra"]`` carries task-level fields from
:meth:`_get_obs_extra`; ``obs["sensor_data"]`` carries camera channels
when the env's ``obs_mode`` is image-bearing.

Shape preservation: zero-masked channels become zero arrays of the same
shape and dtype (mirrors
:class:`chamber.envs.texture_filter.TextureFilterObsWrapper`'s pattern).
This keeps downstream policy networks oblivious to which modalities a
given uid actually receives at runtime — only the *values* change
across conditions, not the *shapes*.

References:
- ADR-007 §Stage 1b (OM axis spec); ``spikes/preregistration/OM.yaml``
  (the condition_id strings this wrapper resolves).
- :class:`chamber.envs.texture_filter.TextureFilterObsWrapper` (the
  parallel pattern for the Stage-0 smoke env's per-uid keep-set).
- :class:`chamber.envs.stage1_pickplace.Stage1PickPlaceEnv` (the
  inner env that supplies the synthesised FT channel and the
  ``condition_id`` attribute).
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

#: Channels under ``obs["extra"]`` the OM-vision-only condition keeps.
#:
#: The policy needs at least *some* task-relevant signal beyond the
#: RGB-D camera to be solvable inside the Stage-1b episode budget
#: (100 steps); ``tcp_pose`` + ``goal_pos`` is the minimum "where is
#: the gripper" + "where is the target" pair. Cube pose / cube-relative
#: deltas / FT are zero-masked. If the AS-hetero pilot shows the gap
#: vs vision_only is artefact-of-this-keep-set, tighten or loosen here
#: in a follow-up patch — the keep-set is design-tunable inside the
#: condition_id's spec since the prereg only names the *modality
#: families* (vision vs vision+ft+proprio), not the per-field keep-set
#: (ADR-007 §Discipline).
_OM_VISION_ONLY_EXTRA_KEEP: frozenset[str] = frozenset({"tcp_pose", "goal_pos"})

#: Condition_ids the wrapper actually mutates obs for. Conditions
#: outside this set pass through untouched.
_FILTERED_CONDITIONS: frozenset[str] = frozenset({"stage1_pickplace_vision_only"})


class Stage1ASStateSynthesizer(gym.ObservationWrapper):  # type: ignore[type-arg]
    """Synthesise ``obs["agent"][uid]["state"]`` for AS conditions (ADR-007 §Stage 1b).

    ManiSkill v3.0.1 ``obs_mode="state_dict"`` (used by the Stage-1 AS
    conditions) emits per-agent ``Dict(qpos: Box, qvel: Box, ...)`` with
    no top-level ``"state"`` key. But
    :meth:`chamber.benchmarks.ego_ppo_trainer.EgoPPOTrainer.from_config`
    reads ``env.observation_space["agent"][ego_uid]["state"]`` as a 1-D
    Box and derives ``obs_dim = ego_state_space.shape[0]``. ManiSkill v3
    ``obs_mode="state"`` flattens ``concat(qpos, qvel)`` into a single
    ``"state"`` key with that shape; this wrapper replicates that concat
    in obs-space so callers can keep ``obs_mode="state_dict"`` for the
    synthesised-channel injection (``force_torque``, ``tcp_pose``, ...)
    the OM-hetero condition needs while still satisfying the trainer's
    1-D ``state`` contract.

    Pass-through for OM conditions (``is_om_condition=True`` — those go
    through :class:`Stage1OMChannelFilter` instead) and for envs whose
    ``condition_id`` is missing or unrecognised (Tier-1 safety: keeps
    the wrapper a no-op for envs that don't satisfy the Stage-1b
    ``condition_id`` contract).

    Args:
        env: Inner env exposing a ``condition_id`` attribute.

    Raises:
        TypeError: If ``env.observation_space`` is not a
            :class:`gym.spaces.Dict`.
    """

    def __init__(self, env: gym.Env[Any, Any]) -> None:
        """Detect AS-condition envs at construction (ADR-007 §Stage 1b)."""
        super().__init__(env)
        if not isinstance(env.observation_space, gym.spaces.Dict):
            msg = (
                "Stage1ASStateSynthesizer requires a gym.spaces.Dict observation "
                f"space; got {type(env.observation_space).__name__}."
            )
            raise TypeError(msg)
        condition_id = getattr(env, "condition_id", None)
        if condition_id is None and hasattr(env, "get_wrapper_attr"):
            try:
                condition_id = env.get_wrapper_attr("condition_id")
            except AttributeError:
                condition_id = None
        # Local import to avoid the chamber.envs.stage1_pickplace ↔
        # chamber.envs.stage1_obs_filter circular import that arises now
        # that make_stage1_pickplace_env applies this wrapper.
        from chamber.envs.stage1_pickplace import resolve_condition

        config = None
        if isinstance(condition_id, str):
            try:
                config = resolve_condition(condition_id)
            except ValueError:
                config = None
        self._active: bool = config is not None and not config.is_om_condition
        if self._active:
            self.observation_space = self._build_observation_space(env.observation_space)

    @staticmethod
    def _build_observation_space(inner: gym.spaces.Dict) -> gym.spaces.Dict:
        """Inject a 1-D ``state`` Box per agent under ``obs["agent"][uid]``.

        The injected Box's shape is ``(qpos_dim + qvel_dim,)`` —
        flattened to 1-D so :attr:`gym.spaces.Box.shape[0]` matches the
        ``obs_dim`` HARL's MLPBase reads at construction.
        """
        agent = inner.spaces.get("agent")
        if not isinstance(agent, gym.spaces.Dict):
            return inner
        new_agent_spaces: dict[str, gym.spaces.Space[Any]] = {}
        for uid, sub in agent.spaces.items():
            if isinstance(sub, gym.spaces.Dict):
                qpos = sub.spaces.get("qpos")
                qvel = sub.spaces.get("qvel")
                if isinstance(qpos, gym.spaces.Box) and isinstance(qvel, gym.spaces.Box):
                    state_dim = int(np.prod(qpos.shape)) + int(np.prod(qvel.shape))
                    state_box = gym.spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(state_dim,),
                        dtype=np.float32,
                    )
                    augmented = dict(sub.spaces)
                    augmented["state"] = state_box
                    new_agent_spaces[uid] = gym.spaces.Dict(augmented)
                    continue
            new_agent_spaces[uid] = sub
        new_spaces = dict(inner.spaces)
        new_spaces["agent"] = gym.spaces.Dict(new_agent_spaces)
        return gym.spaces.Dict(new_spaces)

    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:  # type: ignore[override]
        """Inject synthesised ``state`` per agent (ADR-007 §Stage 1b)."""
        if not self._active:
            return observation
        agent = observation.get("agent")
        if not isinstance(agent, dict):
            return observation
        new_agent: dict[str, Any] = {}
        for uid, sub in agent.items():
            if isinstance(sub, dict) and "qpos" in sub and "qvel" in sub:
                augmented = dict(sub)
                augmented["state"] = _flat_state_concat(sub["qpos"], sub["qvel"])
                new_agent[uid] = augmented
            else:
                new_agent[uid] = sub
        out = dict(observation)
        out["agent"] = new_agent
        return out

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401 - mirrors gym.Wrapper's inherited __getattr__ signature
        """Forward attribute access to the inner env (Tier-2 caller contract).

        Gymnasium 1.3 removed :class:`gym.Wrapper`'s implicit
        ``__getattr__``; existing P1.03 callers of
        :func:`make_stage1_pickplace_env` read SAPIEN-env attributes
        directly (``env.agent``, ``env._jacobian_provider``,
        ``env.condition_config``...). Forwarding here keeps that
        contract without forcing every caller to switch to
        :meth:`gym.Wrapper.get_wrapper_attr`.

        ``__getattr__`` only fires when normal attribute lookup misses;
        ``self.env`` is set by :meth:`gym.Wrapper.__init__` before any
        downstream access, so the delegation is safe. The ``env``-name
        guard is a defensive backstop against recursive lookup on the
        env attr itself (only reachable in pathological half-init
        paths).
        """
        if name == "env":
            raise AttributeError(name)
        return getattr(self.env, name)


def _flat_state_concat(qpos: Any, qvel: Any) -> np.ndarray:  # type: ignore[type-arg]  # noqa: ANN401 - inputs are torch.Tensor or np.ndarray depending on env source; Tier-1 fakes pass np.ndarray, real SAPIEN env passes torch.Tensor
    """Concat ``qpos`` + ``qvel`` into a 1-D float32 ndarray.

    ManiSkill v3 ``obs_mode="state"`` flattens with the order
    ``concat(qpos, qvel)``; mirrored here so anyone running the
    underlying env with the flat ``obs_mode`` would see the same shape.

    Handles both torch tensors (real SAPIEN env) and numpy arrays
    (Tier-1 fakes). Squeezes the ``num_envs`` batch dim implicitly via
    ``ravel`` — Stage-1b runs single-env per cell (plan/07 §3).
    """
    try:
        import torch

        if isinstance(qpos, torch.Tensor) or isinstance(qvel, torch.Tensor):
            qpos_t = torch.as_tensor(qpos).detach().cpu().numpy()
            qvel_t = torch.as_tensor(qvel).detach().cpu().numpy()
            return np.concatenate([qpos_t.ravel(), qvel_t.ravel()]).astype(np.float32)
    except ImportError:
        # Torch is a hard dep of the project (chamber.benchmarks.*),
        # but the local import keeps this module Tier-1 importable on
        # the rare host that ships without it.
        pass
    return np.concatenate([np.asarray(qpos).ravel(), np.asarray(qvel).ravel()]).astype(np.float32)


class Stage1OMChannelFilter(gym.ObservationWrapper):  # type: ignore[type-arg]
    """OM-axis per-condition channel filter (ADR-007 §Stage 1b).

    Reads the inner env's ``condition_id`` at construction time and
    captures the resulting filter policy. The wrapper does not re-read
    ``condition_id`` on every step — Stage-1b envs are built once per
    ``(seed, condition)`` cell (see ``stage1_as.py:253-275``) and the
    condition is fixed for the env's lifetime.

    Args:
        env: Inner env exposing a ``condition_id`` attribute. The
            :class:`chamber.envs.stage1_pickplace.Stage1PickPlaceEnv`
            class satisfies this; the wrapper falls back to a
            pass-through identity if ``condition_id`` is missing or
            not one of the two OM strings.

    Raises:
        TypeError: If the inner env's observation space is not a
            :class:`gym.spaces.Dict` (the wrapper relies on dict-shaped
            obs).
    """

    def __init__(self, env: gym.Env[Any, Any]) -> None:
        """Store the per-condition filter policy (ADR-007 §Stage 1b)."""
        super().__init__(env)
        if not isinstance(env.observation_space, gym.spaces.Dict):
            msg = (
                "Stage1OMChannelFilter requires a gym.spaces.Dict observation "
                f"space; got {type(env.observation_space).__name__}."
            )
            raise TypeError(msg)
        # Read condition_id once at construction; the env-per-cell
        # cadence makes this safe and saves a hot-path attr lookup.
        condition_id = getattr(env, "condition_id", None)
        if condition_id is None and hasattr(env, "get_wrapper_attr"):
            try:
                condition_id = env.get_wrapper_attr("condition_id")
            except AttributeError:
                condition_id = None
        self._condition_id: str | None = condition_id
        self._filter_active: bool = condition_id in _FILTERED_CONDITIONS

    @property
    def condition_id(self) -> str | None:
        """ADR-007 §Stage 1b OM axis: the condition_id resolved at construction."""
        return self._condition_id

    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:  # type: ignore[override]
        """Apply the per-condition channel mask (ADR-007 §Stage 1b).

        OM-vision-only zero-masks the per-uid agent-proprio sub-dicts
        and the ``obs["extra"]["force_torque"]`` channel; keeps
        ``obs["extra"]["tcp_pose"]`` + ``obs["extra"]["goal_pos"]``
        (the minimal task-relevant fields) plus ``obs["sensor_data"]``
        (RGB-D camera) untouched. Other conditions pass through.
        """
        if not self._filter_active:
            return observation
        out = dict(observation)
        # Zero-mask per-uid agent proprio (the "no proprioception" half
        # of the vision-only condition_id semantics).
        agent = observation.get("agent")
        if isinstance(agent, dict):
            zeroed_agent: dict[str, Any] = {}
            for uid, sub in agent.items():
                if isinstance(sub, dict):
                    zeroed_agent[uid] = {
                        ch: np.zeros_like(val) if hasattr(val, "shape") else val
                        for ch, val in sub.items()
                    }
                else:
                    zeroed_agent[uid] = sub
            out["agent"] = zeroed_agent
        # Zero-mask task extras NOT in the vision-only keep-set (the
        # "no force-torque" half of the vision-only semantics).
        extra = observation.get("extra")
        if isinstance(extra, dict):
            filtered_extra: dict[str, Any] = {}
            for ch, val in extra.items():
                if ch in _OM_VISION_ONLY_EXTRA_KEEP:
                    filtered_extra[ch] = val
                elif hasattr(val, "shape"):
                    filtered_extra[ch] = np.zeros_like(val)
                else:
                    filtered_extra[ch] = val
            out["extra"] = filtered_extra
        return out

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401 - mirrors Stage1ASStateSynthesizer's forwarding contract
        """Forward attribute access to the inner env (Tier-2 caller contract).

        Mirrors :meth:`Stage1ASStateSynthesizer.__getattr__`: Gymnasium 1.3
        removed :class:`gym.Wrapper`'s implicit forwarding, so callers of
        :func:`chamber.envs.stage1_pickplace.make_stage1_pickplace_env`
        — which now applies this wrapper on top of the AS synthesizer —
        would otherwise lose access to SAPIEN-env attributes such as
        ``env.agent``, ``env.obs_mode``, ``env.condition_config``.
        """
        if name == "env":
            raise AttributeError(name)
        return getattr(self.env, name)


__all__ = ["Stage1ASStateSynthesizer", "Stage1OMChannelFilter"]
