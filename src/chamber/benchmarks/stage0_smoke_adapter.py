# SPDX-License-Identifier: Apache-2.0
"""Stage-0 smoke env adapter for ego-AHT training (T4b.3; ADR-001 §Validation criteria).

The Stage-0 smoke env (:func:`chamber.benchmarks.stage0_smoke.make_stage0_env`)
is the 3-robot rig-validation env from ADR-001 — heterogeneous robot
uids, per-agent action-repeat, texture-channel filtering, comm shaping.
Its primary purpose is GPU-host smoke validation, not training; the
training side of M4b needs an adapter that surfaces the env in the
shape :func:`chamber.benchmarks.ego_ppo_trainer.EgoPPOTrainer.from_config`
expects:

- ``observation_space["agent"][ego_uid]["state"]`` as a
  :class:`gym.spaces.Box` — the ego state vector the trainer reads via
  :func:`chamber.benchmarks.ego_ppo_trainer._flat_ego_obs`.
- ``action_space[ego_uid]`` as a :class:`gym.spaces.Box` — the ego
  action space the trainer samples.

plan/05 §3.4 ("rollout/update logic on the CONCERTO side, not in the
fork") and ADR-001 §Validation criteria set the role split: the env's
multi-robot rig stays in :mod:`chamber.benchmarks.stage0_smoke`; the
training-side glue lives here on the CONCERTO/Chamber benchmark side.

The adapter satisfies the
:class:`concerto.training.ego_aht.EnvLike` Protocol structurally; the
Protocol is imported (not re-declared) from
:mod:`concerto.training.ego_aht`.

Two-tier test coverage mirroring
``tests/integration/test_stage0_smoke.py``:

- Tier 1 (``tests/integration/test_stage0_adapter_fake.py``) — wrapper-
  structure tests via :class:`tests.fakes.FakeMultiAgentEnv` with
  :data:`chamber.benchmarks.stage0_smoke._SMOKE_ROBOT_UIDS`; always
  runs on CPU.
- Tier 2 (``tests/integration/test_stage0_adapter_real.py``) — real-
  SAPIEN smoke through :func:`make_stage0_training_env`; skipped on
  CPU-only machines.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import gymnasium as gym
import numpy as np

from chamber.benchmarks.stage0_smoke import _SMOKE_ROBOT_UIDS, make_stage0_env
from chamber.envs.errors import ChamberEnvCompatibilityError

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping

    from numpy.typing import NDArray

    from concerto.training.ego_aht import EnvLike

#: Default truncation horizon for Stage-0 training rollouts (ADR-001 §Validation criteria).
#:
#: The Stage-0 smoke env itself has no episode boundary (it is a rig-
#: validation env, not a task). The training loop expects truncation, so
#: the adapter enforces a horizon. 100 ticks matches the
#: ``configs/training/ego_aht_happo/stage0_smoke.yaml`` default and is
#: long enough that the slow-rate agent (action-repeat=10) gets at
#: least 10 action updates per episode.
DEFAULT_EPISODE_LENGTH: int = 100

#: Per-agent state-channel name the trainer reads (ADR-002 §Decisions).
#:
#: Matches the convention in
#: :func:`chamber.benchmarks.ego_ppo_trainer._flat_ego_obs`. Held as a
#: module constant so the synthesis path and the pass-through path read
#: the same name.
_STATE_KEY: str = "state"

#: Rank of the vectorised-by-num_envs obs shapes ManiSkill emits when
#: ``num_envs=1`` adds a leading batch dimension — ``(1, n)`` is rank 2.
#: Used to relax the 1-D Box check in :func:`_synthesised_state_space`.
_VECTORISED_RANK: int = 2


class _Stage0TrainingAdapter(gym.Env):  # type: ignore[type-arg]
    """EnvLike adapter over the Stage-0 multi-robot env (T4b.3; ADR-001 §Validation criteria).

    Bridges three impedance mismatches between the Stage-0 smoke env
    and the training loop's :class:`concerto.training.ego_aht.EnvLike`
    contract:

    1. **Action augmentation** — the Stage-0 env exposes 3 robot uids
       but the training loop only drives 2 (one ego + one frozen
       partner per :data:`agent_uids`). Any inner uid not in
       ``agent_uids`` gets a zero-vector action injected by
       :meth:`step` so the underlying env receives a complete dict-
       action.
    2. **State-channel surfacing** — the trainer reads
       ``obs["agent"][ego_uid]["state"]``. If the inner env already
       exposes that channel (e.g.
       :class:`tests.fakes.FakeMultiAgentEnv`), it is passed through.
       Otherwise the adapter synthesises a flat state vector by
       concatenating every 1-D :class:`gym.spaces.Box` channel under
       ``obs["agent"][ego_uid]`` in alphabetical key order. This handles
       the production Stage-0 case where
       :class:`chamber.envs.TextureFilterObsWrapper` keeps
       ``joint_pos`` + ``joint_vel`` for ``panda_wristcam`` but no
       ``state`` field.
    3. **Truncation** — the Stage-0 env has no native episode boundary;
       the adapter enforces ``episode_length`` ticks of truncation so
       the training loop's per-episode reward accounting works.

    The adapter does NOT touch the comm channel or any other obs key
    the wrapper chain attaches (``obs["comm"]``, etc.); they flow
    through untouched.
    """

    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": []}  # type: ignore[misc]

    def __init__(
        self,
        inner: gym.Env[Any, Any],
        *,
        agent_uids: tuple[str, ...],
        episode_length: int = DEFAULT_EPISODE_LENGTH,
        root_seed: int = 0,
    ) -> None:
        """Wrap an inner Stage-0-style env (T4b.3; ADR-001 §Validation criteria).

        Args:
            inner: A Gymnasium-conformant multi-agent env exposing
                ``action_space`` as a :class:`gym.spaces.Dict` keyed by
                agent uid and ``observation_space["agent"][uid]`` as a
                :class:`gym.spaces.Dict` (the Stage-0 wrapper-chain
                shape, or :class:`tests.fakes.FakeMultiAgentEnv` for
                Tier-1 tests).
            agent_uids: The training-side ego + partner uids. The first
                entry is the ego the trainer updates; subsequent entries
                are the partners the trainer expects to receive actions
                for. Every entry must appear in
                ``inner.action_space.spaces``.
            episode_length: Truncation horizon in adapter ticks (default
                :data:`DEFAULT_EPISODE_LENGTH`).
            root_seed: Stored for downstream identity propagation; the
                Stage-0 env's RNG is seeded via Gymnasium's ``reset(seed=)``
                pathway, not via the adapter (the Stage-0 wrapper chain
                already routes through
                :func:`concerto.training.seeding.derive_substream`).

        Raises:
            ValueError: If ``agent_uids`` is empty or any uid is not in
                ``inner.action_space``.
            TypeError: If ``inner.action_space`` is not a
                :class:`gym.spaces.Dict` or any
                ``action_space[uid]`` is not a :class:`gym.spaces.Box`.
        """
        super().__init__()
        if len(agent_uids) < 1:
            raise ValueError("agent_uids must have at least one entry (the ego).")
        inner_action_space = inner.action_space
        if not isinstance(inner_action_space, gym.spaces.Dict):
            raise TypeError(
                "Stage-0 adapter requires inner.action_space to be a gym.spaces.Dict; "
                f"got {type(inner_action_space).__name__}."
            )
        inner_uids: tuple[str, ...] = tuple(inner_action_space.spaces.keys())
        for uid in agent_uids:
            if uid not in inner_uids:
                raise ValueError(
                    f"agent_uid {uid!r} not in inner env uids {inner_uids!r}; "
                    "the training config's env.agent_uids must be a subset of "
                    "chamber.benchmarks.stage0_smoke._SMOKE_ROBOT_UIDS."
                )
            sub = inner_action_space.spaces[uid]
            if not isinstance(sub, gym.spaces.Box):
                raise TypeError(
                    f"inner.action_space[{uid!r}] must be a gym.spaces.Box; "
                    f"got {type(sub).__name__}."
                )

        self._inner = inner
        self._agent_uids = tuple(agent_uids)
        self._ego_uid = self._agent_uids[0]
        self._inner_uids = inner_uids
        self._episode_length = int(episode_length)
        self._root_seed = int(root_seed)
        self._step_count = 0
        # Zero-action template for inner uids not driven by the trainer.
        self._zero_action = _build_zero_action(inner_action_space, self._agent_uids)
        self.action_space = gym.spaces.Dict(
            {uid: inner_action_space.spaces[uid] for uid in self._agent_uids}
        )
        # State channel: pass-through if present in the inner obs space;
        # synthesise from 1-D Box channels otherwise.
        (
            self.observation_space,
            self._synthesise_state_for,
            self._state_synthesis_keys,
        ) = _build_adapted_obs_space(inner.observation_space, inner_uids)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset the inner env and clear the truncation counter (ADR-001 §Validation criteria).

        Forwards ``seed`` to the inner env so the wrapper chain's
        :func:`concerto.training.seeding.derive_substream` substreams
        rebind deterministically per the project's P6 reproducibility
        contract.
        """
        obs, info = self._inner.reset(seed=seed, options=options)
        self._step_count = 0
        return self._transform_obs(obs), dict(info)

    def step(
        self,
        action: dict[str, NDArray[np.floating]],
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Augment action, step inner, surface state, enforce truncation (ADR-001).

        Args:
            action: Dict keyed by training-side ``agent_uids``. Every
                entry must be present; missing keys raise
                :class:`ValueError`. Inner uids not in ``agent_uids``
                are augmented with zero-actions before the inner step.

        Returns:
            ``(obs, reward, terminated, truncated, info)`` with
            ``obs["agent"][ego_uid]["state"]`` guaranteed to be a 1-D
            ``np.float32`` array. ``truncated`` is the inner env's
            truncated flag OR-ed with the adapter's own
            ``step_count >= episode_length`` boundary.
        """
        for uid in self._agent_uids:
            if uid not in action:
                raise ValueError(
                    f"step() action missing uid {uid!r}; got keys {list(action.keys())}"
                )
        full_action: dict[str, NDArray[np.floating]] = dict(action)
        for uid, zero in self._zero_action.items():
            full_action.setdefault(uid, zero)
        obs, reward, terminated, truncated, info = self._inner.step(full_action)
        self._step_count += 1
        adapter_truncated = self._step_count >= self._episode_length
        return (
            self._transform_obs(obs),
            float(reward),
            bool(terminated),
            bool(truncated) or adapter_truncated,
            dict(info),
        )

    def _transform_obs(self, obs: Mapping[str, Any]) -> dict[str, Any]:
        """Inject synthesised ``state`` keys where needed; pass-through otherwise."""
        if not self._synthesise_state_for:
            # Hot path: no synthesis needed (Tier-1 FakeMultiAgentEnv).
            return dict(obs)
        out: dict[str, Any] = dict(obs)
        agent_in = obs.get("agent", {})
        agent_out: dict[str, Any] = dict(agent_in)
        for uid in self._synthesise_state_for:
            sub_in = agent_in.get(uid, {})
            if not isinstance(sub_in, dict):
                continue
            keys = self._state_synthesis_keys[uid]
            pieces = [np.asarray(sub_in[k], dtype=np.float32).ravel() for k in keys]
            sub_out = dict(sub_in)
            sub_out[_STATE_KEY] = (
                np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
            )
            agent_out[uid] = sub_out
        out["agent"] = agent_out
        return out


def _build_zero_action(
    inner_action_space: gym.spaces.Dict,
    training_uids: tuple[str, ...],
) -> dict[str, NDArray[np.float32]]:
    """Pre-compute zero-action vectors for un-driven inner uids (ADR-001 §Validation)."""
    zero_action: dict[str, NDArray[np.float32]] = {}
    for uid, sub in inner_action_space.spaces.items():
        if uid in training_uids:
            continue
        if not isinstance(sub, gym.spaces.Box):
            raise TypeError(
                f"inner.action_space[{uid!r}] must be a gym.spaces.Box; got {type(sub).__name__}."
            )
        zero_action[uid] = np.zeros(sub.shape, dtype=np.float32)
    return zero_action


def _build_adapted_obs_space(
    inner_obs_space: gym.spaces.Space[Any],
    inner_uids: tuple[str, ...],
) -> tuple[gym.spaces.Dict, set[str], dict[str, tuple[str, ...]]]:
    """Per-uid obs adaptation: pass-through ``state`` or synthesise it (ADR-001 §Validation).

    Returns ``(adapted_space, synthesise_for_uids, synthesis_keys_per_uid)``.
    """
    if not isinstance(inner_obs_space, gym.spaces.Dict):
        raise TypeError(
            "Stage-0 adapter requires inner.observation_space to be a gym.spaces.Dict; "
            f"got {type(inner_obs_space).__name__}."
        )
    inner_agent_space = inner_obs_space.spaces["agent"]
    if not isinstance(inner_agent_space, gym.spaces.Dict):
        raise TypeError(
            "Stage-0 adapter requires inner.observation_space['agent'] to be a "
            f"gym.spaces.Dict; got {type(inner_agent_space).__name__}."
        )
    synthesise_for: set[str] = set()
    synthesis_keys: dict[str, tuple[str, ...]] = {}
    adapted_agent: dict[str, gym.spaces.Space[Any]] = {}
    for uid in inner_uids:
        sub_space = inner_agent_space.spaces[uid]
        if not isinstance(sub_space, gym.spaces.Dict):
            adapted_agent[uid] = sub_space
            continue
        if _STATE_KEY in sub_space.spaces:
            adapted_agent[uid] = sub_space
            continue
        synth_box, synth_keys = _synthesised_state_space(sub_space)
        synthesise_for.add(uid)
        synthesis_keys[uid] = synth_keys
        adapted_sub_dict: dict[str, gym.spaces.Space[Any]] = dict(sub_space.spaces)
        adapted_sub_dict[_STATE_KEY] = synth_box
        adapted_agent[uid] = gym.spaces.Dict(adapted_sub_dict)
    adapted_top: dict[str, gym.spaces.Space[Any]] = dict(inner_obs_space.spaces)
    adapted_top["agent"] = gym.spaces.Dict(adapted_agent)
    return gym.spaces.Dict(adapted_top), synthesise_for, synthesis_keys


def _synthesised_state_space(
    sub_space: gym.spaces.Dict,
) -> tuple[gym.spaces.Box, tuple[str, ...]]:
    """Flat state Box from 1-D Box channels of ``sub_space`` (ADR-001 §Validation criteria).

    Concatenates every 1-D :class:`gym.spaces.Box` channel of
    ``sub_space`` in alphabetical key order. Image-shaped channels
    (rank > 1) are deliberately ignored — the trainer's
    :func:`chamber.benchmarks.ego_ppo_trainer._flat_ego_obs` consumes a
    flat vector, and the right place to fold images into a flat
    representation is an explicit encoder, not the adapter.

    Args:
        sub_space: The inner env's ``observation_space["agent"][uid]``
            sub-Dict for the uid that needs synthesised state.

    Returns:
        ``(box, keys)`` where ``box`` is the synthesised state space
        (low/high concatenated, dtype taken from the first piece) and
        ``keys`` is the alphabetised tuple of source channel names so
        the adapter can rebuild the state vector at step/reset time
        from the same channels.

    Raises:
        ChamberEnvCompatibilityError: If no 1-D Box channels are
            available — the trainer cannot proceed without a flat state
            vector and the wrapper chain would need adjustment (or the
            ego uid would need to change).
    """
    pieces: list[gym.spaces.Box] = []
    keys: list[str] = []
    for key in sorted(sub_space.spaces.keys()):
        sub = sub_space.spaces[key]
        if not isinstance(sub, gym.spaces.Box) or sub.shape is None:
            continue
        # Accept 1-D shapes ``(n,)`` and vectorised ``(1, n)`` shapes — the
        # latter is what ManiSkill emits when ``num_envs=1`` adds a leading
        # batch dimension to every per-agent obs channel. ``_transform_obs``
        # already handles any shape via ``.ravel()``; only the space-building
        # check needed the relaxation (ADR-001 §Validation criteria).
        if len(sub.shape) == 1:
            pieces.append(sub)
            keys.append(key)
        elif len(sub.shape) == _VECTORISED_RANK and sub.shape[0] == 1:
            squeezed = gym.spaces.Box(
                low=np.asarray(sub.low[0], dtype=np.float32),
                high=np.asarray(sub.high[0], dtype=np.float32),
                dtype=np.float32,
            )
            pieces.append(squeezed)
            keys.append(key)
    if not pieces:
        raise ChamberEnvCompatibilityError(
            "Stage-0 adapter: inner env exposes no 1-D Box channels under "
            "obs['agent'][ego_uid]; the trainer needs a flat state vector. "
            "Adjust the chamber.benchmarks.stage0_smoke wrapper chain (e.g. "
            "include 'state' or 'joint_pos' in _SMOKE_KEEP for the ego uid) "
            "or pick a different ego uid (ADR-001 §Validation criteria)."
        )
    low = np.concatenate([np.asarray(p.low, dtype=np.float32).ravel() for p in pieces])
    high = np.concatenate([np.asarray(p.high, dtype=np.float32).ravel() for p in pieces])
    return gym.spaces.Box(low=low, high=high, dtype=np.float32), tuple(keys)


def make_stage0_training_env(
    *,
    agent_uids: tuple[str, ...],
    episode_length: int = DEFAULT_EPISODE_LENGTH,
    root_seed: int = 0,
) -> EnvLike:
    """Build the Stage-0 training env (T4b.3; ADR-001 §Validation criteria; plan/05 §3.4).

    Wraps :func:`chamber.benchmarks.stage0_smoke.make_stage0_env` in
    :class:`_Stage0TrainingAdapter` so the result satisfies
    :class:`concerto.training.ego_aht.EnvLike` and can be passed to
    :func:`concerto.training.ego_aht.train` via the chamber-side
    :func:`chamber.benchmarks.training_runner.build_env` dispatch.

    Args:
        agent_uids: Training-side ego + partner uids. Must be a subset
            of :data:`chamber.benchmarks.stage0_smoke._SMOKE_ROBOT_UIDS`.
            The first entry is the ego the trainer updates.
        episode_length: Truncation horizon in adapter ticks.
        root_seed: Project-wide root seed (forwarded for downstream
            identity propagation; the Stage-0 wrapper chain seeds its
            own substreams via
            :func:`concerto.training.seeding.derive_substream`).

    Returns:
        An :class:`~concerto.training.ego_aht.EnvLike`-conformant env
        ready for :func:`concerto.training.ego_aht.train`.

    Raises:
        ChamberEnvCompatibilityError: If SAPIEN / Vulkan is unavailable
            (raised by :func:`make_stage0_env`; see ADR-001 §Risks).
        ValueError: If ``agent_uids`` is not a subset of
            :data:`_SMOKE_ROBOT_UIDS`.
    """
    for uid in agent_uids:
        if uid not in _SMOKE_ROBOT_UIDS:
            raise ValueError(
                f"agent_uid {uid!r} not in _SMOKE_ROBOT_UIDS {_SMOKE_ROBOT_UIDS!r}; "
                "Stage-0 training requires one of the three rig-validated robots "
                "(ADR-001 §Validation criteria)."
            )
    inner = make_stage0_env()
    return _Stage0TrainingAdapter(
        inner,
        agent_uids=agent_uids,
        episode_length=episode_length,
        root_seed=root_seed,
    )


__all__ = [
    "DEFAULT_EPISODE_LENGTH",
    "make_stage0_training_env",
]
