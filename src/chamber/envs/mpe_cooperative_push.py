# SPDX-License-Identifier: Apache-2.0
"""2-agent MPE Cooperative-Push env for the empirical-guarantee experiment (T4b.13).

Plan/05 §3.5 (decision row "Empirical guarantee Phase-0 verification") +
ADR-002 risk-mitigation #1: the Phase-0 ego-AHT empirical-guarantee
experiment runs on a 2-agent MPE-style cooperative continuous-control
task. This module ships a deterministic hand-rolled equivalent of
PettingZoo's ``simple_spread_v3`` rather than depending on PettingZoo
itself — Phase 0 keeps the dependency footprint minimal (PettingZoo
deprecated MPE in 1.25 and split it into the younger ``mpe2`` package;
either lift would mean a dep-auditor pass for a single Phase-0 task).
The task structure matches:

- Two agents (default uids ``ego`` and ``partner``) on the unit-square
  plane ``[-1, 1]^2`` with continuous 2-D velocity actions clipped to
  ``[-1, 1]`` per axis.
- Two landmarks fixed per episode at random positions sampled from a
  determinism-harnessed numpy ``Generator`` (seed routed via
  :func:`concerto.training.seeding.derive_substream`).
- Shared reward (both agents see the same scalar):
  ``r = -sum_landmark min_agent ||agent_pos - landmark_pos||``. The
  cooperative-coverage gradient encourages the agents to pick distinct
  landmarks; a single agent cannot drive ``r`` to zero on its own.
- Episode length fixed at :data:`DEFAULT_EPISODE_LENGTH`; truncation only
  (no terminal state).

The env is intentionally Gymnasium-compatible with CONCERTO's other
wrappers (M2 comm + M3 safety): observation is keyed by ``obs["agent"][uid]``
with the ``state`` channel that
:class:`chamber.partners.heuristic.ScriptedHeuristicPartner` already
reads in its three-tier fallback (plan/04 §3.4). Wrapping this env
through :class:`chamber.envs.action_repeat.PerAgentActionRepeatWrapper`,
:class:`chamber.envs.comm_shaping.CommShapingWrapper`, etc. is therefore
a zero-config drop-in.

Determinism (P6): two ``reset(seed=K)`` calls on the same ``root_seed``
produce byte-identical landmark layouts and agent start poses; the same
action stream then produces a byte-identical state trajectory and reward
curve. ``test_mpe_cooperative_push.py::test_determinism_byte_identical_traj``
pins this contract.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import gymnasium as gym
import numpy as np

from concerto.training.seeding import derive_substream

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray

#: Default episode length in env ticks (plan/05 §3.5).
#:
#: 50 ticks is the canonical PettingZoo simple_spread default; long enough
#: for a learner to demonstrate the cooperative-coverage gradient but short
#: enough that 100k frames (T4b.13) fit in 30 minutes of CPU wall-time.
DEFAULT_EPISODE_LENGTH: int = 50

#: Number of cooperative landmarks (plan/05 §3.5).
#:
#: Matched to the agent count (2) so the cooperative gradient has a unique
#: coverage solution — one agent per landmark. Single-landmark variants
#: collapse the task to "both agents head toward the same point" which has
#: no cooperation signal.
N_LANDMARKS: int = 2

#: Plane half-width — every position is clipped to ``[-POSITION_BOUND, +POSITION_BOUND]``.
POSITION_BOUND: float = 1.0

#: Per-axis velocity-action clip — every action component is clipped to
#: ``[-VELOCITY_BOUND, +VELOCITY_BOUND]`` before integration (plan/05 §3.5).
VELOCITY_BOUND: float = 1.0

#: Integration timestep for the perfect-velocity-tracking dynamics
#: (``x[t+1] = clip(x[t] + a * DT, ±bound)``). 0.1 keeps the trajectory
#: roughly the same length scale as PettingZoo simple_spread for a fair
#: comparison if Phase 1 ever swaps in the real PettingZoo env.
DT: float = 0.1

#: Substream name used to seed the env's deterministic RNG (P6).
_SUBSTREAM_NAME: str = "env.mpe_cooperative_push"


class MPECooperativePushEnv(gym.Env):  # type: ignore[type-arg]
    """2-agent MPE-style cooperative continuous-control env (T4b.13; ADR-002 risk-mitigation #1).

    See module docstring for the task design. The env exposes the standard
    Gymnasium ``reset`` + ``step`` API with multi-agent dict actions / dict
    observations keyed by ``uid``, matching :class:`tests.fakes.FakeMultiAgentEnv`
    so the M2/M3 wrapper stack drops in unchanged (plan/01 §3).
    """

    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": []}  # type: ignore[misc]

    #: Number of agents the env supports — fixed at 2 by the cooperative-coverage task.
    _N_AGENTS: ClassVar[int] = 2

    def __init__(
        self,
        *,
        agent_uids: tuple[str, str] = ("ego", "partner"),
        episode_length: int = DEFAULT_EPISODE_LENGTH,
        root_seed: int = 0,
    ) -> None:
        """Build a 2-agent Cooperative-Push env (T4b.13; plan/05 §3.5).

        Args:
            agent_uids: The two uids the env exposes. The default
                ``("ego", "partner")`` matches the ego-AHT training loop
                convention. Reordering swaps which agent the trainer
                updates and which is held frozen — the env is agnostic.
            episode_length: Truncation horizon in env ticks (default
                :data:`DEFAULT_EPISODE_LENGTH`).
            root_seed: Project-wide root seed; the env derives a
                deterministic numpy ``Generator`` from this via
                :func:`concerto.training.seeding.derive_substream`
                (P6 reproducibility).

        Raises:
            ValueError: If ``agent_uids`` does not have exactly two
                distinct entries.
        """
        super().__init__()
        if len(agent_uids) != self._N_AGENTS or agent_uids[0] == agent_uids[1]:
            raise ValueError(
                f"agent_uids must be a 2-tuple of distinct strings; got {agent_uids!r}"
            )
        self._uids: tuple[str, str] = agent_uids
        self._episode_length: int = int(episode_length)
        self._root_seed: int = int(root_seed)

        self.action_space = gym.spaces.Dict(
            {
                uid: gym.spaces.Box(
                    low=-VELOCITY_BOUND,
                    high=VELOCITY_BOUND,
                    shape=(2,),
                    dtype=np.float32,
                )
                for uid in self._uids
            }
        )
        # state channel: self_pos(2) + self_vel(2) + landmark_rel(4) + partner_rel(2) = 10.
        agent_state = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Dict(
                    {uid: gym.spaces.Dict({"state": agent_state}) for uid in self._uids}
                )
            }
        )

        self._rng: np.random.Generator = derive_substream(
            _SUBSTREAM_NAME, root_seed=self._root_seed
        ).default_rng()
        # Per-episode mutable state — set by reset.
        self._step_count: int = 0
        self._positions: dict[str, NDArray[np.float64]] = {}
        self._velocities: dict[str, NDArray[np.float64]] = {}
        self._landmark_positions: NDArray[np.float64] = np.zeros((N_LANDMARKS, 2))

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset to a determinism-harnessed initial state (T4b.13; ADR-002 risk-mitigation #1; P6).

        .. note::

            This env's seeding semantics intentionally diverge from
            Gymnasium's convention (where ``reset(seed=K)`` is the
            *complete* seed source). Here, ``reset(seed=K)`` is folded
            with the constructor's ``root_seed`` via
            :func:`concerto.training.seeding.derive_substream` using the
            substream name ``"env.mpe_cooperative_push.episode.{K}"`` —
            so two training runs at distinct ``root_seed`` values do
            *not* collide on the same per-episode RNG even at the same
            ``K``. This matches the project-wide seeding philosophy
            (P6 + ADR-002 §Decisions): the ``root_seed`` is the
            run-level pin, and per-episode ``seed`` is an episode index
            scoped under it.

        Args:
            seed: Per-episode seed (episode index, NOT a complete seed).
                When provided, re-derives the env's RNG via
                ``derive_substream(f"env.mpe_cooperative_push.episode.{seed}",
                root_seed=self._root_seed)``. When ``None``, continues
                from the RNG's existing state (Gymnasium convention).
            options: Reserved for the Gymnasium API; ignored.

        Returns:
            ``(obs, info)`` where ``obs`` matches the
            :attr:`observation_space` shape and ``info`` is empty.
        """
        del options
        if seed is not None:
            self._rng = derive_substream(
                f"{_SUBSTREAM_NAME}.episode.{seed}",
                root_seed=self._root_seed,
            ).default_rng()
        self._step_count = 0
        self._landmark_positions = self._rng.uniform(
            -POSITION_BOUND, POSITION_BOUND, size=(N_LANDMARKS, 2)
        )
        for uid in self._uids:
            self._positions[uid] = self._rng.uniform(-POSITION_BOUND, POSITION_BOUND, size=(2,))
            self._velocities[uid] = np.zeros(2, dtype=np.float64)
        return self._build_obs(), {}

    def step(
        self,
        action: dict[str, NDArray[np.floating]],
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Advance one tick (T4b.13; ADR-002 risk-mitigation #1; plan/05 §3.5).

        Args:
            action: Dict keyed by ``uid``, each value a 2-D velocity
                command. Components are clipped to
                ``[-VELOCITY_BOUND, +VELOCITY_BOUND]`` before integration
                so a misbehaving partner cannot escape the plane.

        Returns:
            ``(obs, reward, terminated, truncated, info)``. ``reward`` is
            the *shared* cooperative-coverage scalar (both agents see the
            same value); the trainer's ego/partner split happens upstream
            in :func:`concerto.training.ego_aht.train`. ``terminated`` is
            always ``False`` (no terminal state); ``truncated`` is ``True``
            on the last tick of the episode.

        Raises:
            ValueError: If ``action`` is missing a required uid.
        """
        for uid in self._uids:
            if uid not in action:
                raise ValueError(
                    f"step() action missing uid {uid!r}; got keys {list(action.keys())}"
                )
        for uid in self._uids:
            a = np.clip(
                np.asarray(action[uid], dtype=np.float64),
                -VELOCITY_BOUND,
                VELOCITY_BOUND,
            )
            self._velocities[uid] = a
            self._positions[uid] = np.clip(
                self._positions[uid] + a * DT, -POSITION_BOUND, POSITION_BOUND
            )
        self._step_count += 1
        truncated = self._step_count >= self._episode_length
        return self._build_obs(), self._compute_reward(), False, truncated, {}

    def _build_obs(self) -> dict[str, Any]:
        """Pack the Gymnasium-conformant obs dict (plan/01 §3; plan/04 §3.4)."""
        out_agents: dict[str, dict[str, NDArray[np.float32]]] = {}
        for uid in self._uids:
            self_pos = self._positions[uid]
            self_vel = self._velocities[uid]
            landmark_rel = (self._landmark_positions - self_pos).flatten()
            partner_uid = self._uids[1] if uid == self._uids[0] else self._uids[0]
            partner_rel = self._positions[partner_uid] - self_pos
            state = np.concatenate([self_pos, self_vel, landmark_rel, partner_rel]).astype(
                np.float32
            )
            out_agents[uid] = {"state": state}
        return {"agent": out_agents}

    def _compute_reward(self) -> float:
        """Shared cooperative-coverage scalar (plan/05 §3.5).

        ``r = -sum_landmark min_agent dist(agent, landmark)``. Both
        agents always see the same value so the trainer can route it to
        the ego's policy update without a credit-assignment step.

        Edge case: when an agent sits exactly on a landmark, that
        landmark contributes ``0`` to the sum. Perfect coverage (one
        agent per landmark, both at distance ``0``) yields ``r = 0`` —
        the natural ceiling. The empirical-guarantee assertion in
        T4b.13 is on *moving-window-of-10 non-decreasing*, not on
        approaching this ceiling.

        ``np.linalg.norm`` matches PettingZoo ``simple_spread_v3``'s
        Euclidean reward (squared distance would change the gradient
        shape and break the "matches simple_spread" claim should
        Phase 1 ever cross-validate against the real env).
        """
        # Phase 1 TODO: cross-validate trajectory against
        # mpe2.simple_spread_v3 once the dep-auditor has cleared mpe2.
        total = 0.0
        for landmark in self._landmark_positions:
            distances = [
                float(np.linalg.norm(self._positions[uid] - landmark)) for uid in self._uids
            ]
            total += min(distances)
        return -total


__all__ = [
    "DEFAULT_EPISODE_LENGTH",
    "DT",
    "N_LANDMARKS",
    "POSITION_BOUND",
    "VELOCITY_BOUND",
    "MPECooperativePushEnv",
]
# Sorted alphabetically; ruff RUF022 enforces the order.
