# SPDX-License-Identifier: Apache-2.0
# pyright: reportPrivateImportUsage=false
#
# torch's distributed stubs do not export ``device``, ``from_numpy``, etc.
# via ``__all__``, even though they are the canonical public API documented
# at https://pytorch.org/docs/stable/. Suppress the rule file-locally rather
# than scattering per-line pragmas; per-line is the noisier option and
# pyright's stub-driven false positives here do not reflect runtime risk.
"""Ego-PPO trainer wrapping HARL's HAPPO actor (M4b-8a; ADR-002 §Decisions).

The :class:`EgoPPOTrainer` is the concrete :class:`~concerto.training.ego_aht.EgoTrainer`
that drives the Phase-0 ego-AHT empirical-guarantee experiment (T4b.13 /
plan/05 §3.5). It satisfies the Protocol with a thin layer over HARL's
:class:`harl.algorithms.actors.happo.HAPPO` actor + a hand-rolled MLP critic
+ a hand-rolled rollout buffer.

Architecture (plan/05 §3.5; ADR-002 §Decisions):

- The HARL fork's :class:`harl.algorithms.actors.ego_aht_happo.EgoAHTHAPPO`
  is a thin reference subclass for external researchers; its
  :meth:`from_config` body is intentionally :class:`NotImplementedError`.
- CONCERTO's training run binds :meth:`EgoPPOTrainer.from_config` instead.
  The bridge from :class:`~concerto.training.config.EgoAHTConfig` (project-
  specific Pydantic) to HARL's 18-key ``args`` dict lives here, on the
  CONCERTO side, where project-specific surfaces belong.
- Rollout / GAE / advantage normalization / PPO update / critic update all
  run in this module — *not* in the fork, *not* via HARL's
  :class:`OnPolicyHARunner`. This satisfies the architecture rule in
  plan/05 §3.5 ("rollout/update logic on the CONCERTO side") and lets the
  ``test_advantage_decomposition.py`` property test compare against a
  hand-rolled GAE reference without tautology.

Public surface:

- :func:`compute_gae` — module-level GAE recurrence (ADR-002 §Decisions;
  Schulman et al. 2016 §6.3). Used both by :class:`EgoPPOTrainer` and by
  the property test (T4b.13 / plan/05 §5).
- :func:`normalize_advantages` — module-level mean/std normalization. Same
  formula HARL applies inside :meth:`HAPPO.train` for state_type ``"EP"``
  (without nan-masking, since the ego is always active in our 2-agent
  task — documented in the function's docstring).
- :class:`EgoPPOTrainer` — the trainer.

ADR-002 risk-mitigation #1: the empirical-guarantee assertion in T4b.13 /
plan/05 §6 criterion 4 is a trip-wire for the framework choice. Do not
change the GAE / normalization / PPO update math without re-running the
property test and re-validating the reference implementation against
the canonical PPO paper formulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import gymnasium as gym
import numpy as np
import torch
from harl.algorithms.actors.happo import HAPPO
from torch import nn

from concerto.training.seeding import derive_substream

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping

    from numpy.typing import NDArray

    from concerto.training.config import EgoAHTConfig, HAPPOHyperparams
    from concerto.training.ego_aht import EnvLike

#: Keys HARL's :class:`HAPPO` + :class:`StochasticPolicy` + :class:`MLPBase` +
#: :class:`DiagGaussian` read from a single ``args`` dict. Construction values
#: not driven by :class:`~concerto.training.config.HAPPOHyperparams` are
#: filled in from :func:`_build_harl_args` using these defaults. Holding
#: them here (rather than inside :func:`_build_harl_args`) keeps the
#: surface inspectable from a debugger and from the property test fixtures.
#:
#: ADR-002 §Decisions: the chosen values match HARL's published
#: Bi-DexHands defaults where they translate. ``hidden_sizes`` is filled
#: from :attr:`HAPPOHyperparams.hidden_dim` per layer; ``ppo_epoch`` and
#: ``clip_param`` follow ``n_epochs`` / ``clip_eps``; the rest are
#: HARL-only knobs we hold constant in Phase 0.
_HARL_FIXED_DEFAULTS: dict[str, Any] = {
    # OnPolicyBase + StochasticPolicy
    "opti_eps": 1e-5,
    "weight_decay": 0.0,
    "data_chunk_length": 10,
    "use_recurrent_policy": False,
    "use_naive_recurrent_policy": False,
    "use_policy_active_masks": False,
    # ``"prod"`` is the right action_aggregation for diag-Gaussian policies:
    # ``prod(exp(diff)) == exp(sum(diff))`` recovers the joint per-step
    # importance ratio. ``"sum"`` would be incorrect (sum of per-dim
    # ratios, not joint) — see :meth:`HAPPO.update`.
    "action_aggregation": "prod",
    "recurrent_n": 1,
    # HAPPO
    "entropy_coef": 0.01,
    "use_max_grad_norm": True,
    "max_grad_norm": 10.0,
    # MLPBase
    "use_feature_normalization": True,
    "activation_func": "relu",
    "initialization_method": "orthogonal_",
    # ACTLayer + DiagGaussian
    "gain": 0.01,
    "std_x_coef": 1.0,
    "std_y_coef": 0.5,
}

#: Substream name for torch's global RNG seed (P6 / ADR-002 §"deterministic
#: seeding harness"). HARL's :class:`DiagGaussian` samples actions via
#: :meth:`torch.distributions.Normal.sample`, which reads from torch's
#: *global* RNG (no per-distribution generator surface). We seed that
#: global RNG once at trainer construction; subsequent action draws and
#: PPO updates inherit determinism. The cost is one global side-effect
#: per trainer; documented as a Phase-0 limitation in the class docstring.
_TORCH_SUBSTREAM: str = "training.ego_ppo.torch"

#: Substream name for the in-trainer numpy RNG used to permute minibatch
#: indices on each PPO epoch (P6).
_MINIBATCH_SUBSTREAM: str = "training.ego_ppo.minibatch"


def compute_gae(
    rewards: NDArray[np.float32],
    values: NDArray[np.float32],
    next_values: NDArray[np.float32],
    episode_boundaries: NDArray[np.bool_],
    *,
    gamma: float,
    gae_lambda: float,
) -> NDArray[np.float32]:
    """GAE with per-step bootstrap (Schulman 2016 §6.3; Pardo 2017; ADR-002 §Decisions).

    Pardo et al. 2017 ("Time Limits in Reinforcement Learning") motivates
    the per-step bootstrap target shape: time-limit truncation is **not**
    a terminal state — the policy should still receive the value of the
    state it would have transitioned to. Treating truncation as
    termination zeros that bootstrap and biases advantages
    systematically, slowing learning materially (see project issue #62).

    The recurrence is::

        delta_t = r_t + gamma * next_values[t]     - V(s_t)
        A_t     = delta_t + gamma * lambda * (1 - boundary_t) * A_{t+1}
        A_T     = delta_T

    where ``next_values[t]`` is the *caller-supplied* bootstrap target —
    in production it is:

    - ``values[t + 1]`` for non-boundary steps (typical mid-rollout).
    - ``V(s_truncated_final_t)`` at truncation boundaries (the value of
      the actual post-step obs of the truncated episode, computed by the
      caller via the critic).
    - ``0.0`` at termination boundaries (true terminal state — no value
      after).
    - The rollout-tail bootstrap (``V(s_T+1)`` if the rollout ends
      mid-episode; ``0.0`` if it ends on an episode boundary) at the
      final timestep ``t = T - 1``.

    ``episode_boundaries[t]`` is ``True`` iff step ``t`` ended an episode
    (terminated *or* truncated). It zeros GAE propagation across the
    boundary — advantages from a *different* episode (steps ``> t`` after
    a boundary) must not contribute to ``A_t``.

    plan/05 §5 + the property test
    ``tests/property/test_advantage_decomposition.py`` verify this formula
    matches a freshly hand-rolled reference to within 1e-6 on synthetic
    trajectories that include both termination and truncation boundaries.

    Args:
        rewards: Per-step ego reward. Shape ``(T,)``.
        values: Per-step critic value estimate ``V(s_t)``. Shape ``(T,)``.
        next_values: Per-step bootstrap target. Shape ``(T,)``. Caller
            responsibility (see module docstring + issue #62 root-cause
            writeup for why this split matters).
        episode_boundaries: ``True`` at steps that ended an episode
            (terminated OR truncated). Shape ``(T,)``. Zeros GAE
            propagation across the boundary.
        gamma: Discount factor in ``[0, 1]``.
        gae_lambda: GAE exponential smoothing in ``[0, 1]``.

    Returns:
        Advantages ``A_t``, same shape and dtype (float32) as ``rewards``.
    """
    n = rewards.shape[0]
    advantages = np.zeros(n, dtype=np.float32)
    # Accumulate ``gae`` in Python float64 — keeping intermediate
    # advantages at higher precision avoids the 1e-6 rounding drift that
    # float32 in-loop accumulation accrues over long horizons. The final
    # store to ``advantages[t]`` casts back to the declared float32
    # storage. The property test against ``reference_gae`` relies on
    # both implementations using the same accumulation precision.
    gae = 0.0
    for t in reversed(range(n)):
        boundary = bool(episode_boundaries[t])
        delta = float(rewards[t]) + gamma * float(next_values[t]) - float(values[t])
        # Zero GAE propagation across episode boundaries: advantages from
        # the next episode must not flow back into this episode (Pardo 2017
        # §4; issue #62 root-cause writeup).
        gae = delta + gamma * gae_lambda * (0.0 if boundary else 1.0) * gae
        advantages[t] = np.float32(gae)
    return advantages


def normalize_advantages(advantages: NDArray[np.float32]) -> NDArray[np.float32]:
    """Mean/std-normalize advantages (ADR-002 §Decisions; matches HAPPO state_type='EP').

    The formula::

        A_normalized = (A - mean(A)) / (std(A) + 1e-5)

    matches the EP-state branch in :meth:`HAPPO.train`
    (lines 122-127 of ``harl/algorithms/actors/happo.py``) without the
    nan-masking it applies to inactive agents. The ego in our 2-agent
    task is never inactive (``use_policy_active_masks=False`` in
    :data:`_HARL_FIXED_DEFAULTS`), so the nan-masking branch is dead
    code in our setting and the formula collapses to a plain
    mean/std normalization.

    plan/05 §5: pulled out of HARL into this module so the property test
    (T4b.13) can verify the formula matches single-agent PPO without
    tautology.

    Args:
        advantages: Raw advantages (typically from :func:`compute_gae`).

    Returns:
        Normalized advantages, same shape and dtype.
    """
    mean = float(advantages.mean())
    std = float(advantages.std())
    return ((advantages - mean) / (std + 1e-5)).astype(np.float32)


class _EgoCritic(nn.Module):
    """Value head for ego PPO (Phase-0; plan/05 §3.5).

    Hand-rolled MLP rather than HARL's :class:`harl.algorithms.critics.v_critic.VCritic`
    so the critic's args dict + buffer-shape contracts stay out of the
    CONCERTO side. Architecture matches HARL's MLPBase: two Tanh hidden
    layers of ``hidden_dim`` width plus a scalar head. Tanh activation
    matches the published Bi-DexHands HAPPO critic; cheap to swap to
    ReLU if Phase-1 needs it.

    ADR-002 §Decisions: critic state ships in
    :meth:`EgoPPOTrainer.state_dict` so the M4b-9 zoo-seed reproducer
    can resume mid-training (T4b.14).
    """

    def __init__(self, *, obs_dim: int, hidden_dim: int) -> None:
        """Build a 2-hidden-layer MLP critic (ADR-002 §Decisions; plan/05 §3.5)."""
        super().__init__()
        self.net: nn.Sequential = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Map ``(batch, obs_dim)`` → ``(batch, 1)`` (ADR-002 §Decisions)."""
        return self.net(obs)


def _build_harl_args(*, happo: HAPPOHyperparams) -> dict[str, Any]:
    """Bridge :class:`HAPPOHyperparams` (Pydantic) to HARL's args dict (ADR-002 §Decisions).

    The ``actor_num_mini_batch`` is derived as
    ``max(1, rollout_length // batch_size)`` so every PPO epoch sweeps the
    full rollout once. Setting ``batch_size > rollout_length`` therefore
    collapses to one minibatch per epoch (single-batch SGD) rather than
    silently dropping data.

    plan/05 §3.5: lives on the CONCERTO side because :class:`HAPPOHyperparams`
    is a project-specific surface — the fork must stay free of
    project-specific concerns.

    Args:
        happo: Validated :class:`HAPPOHyperparams` from
            :class:`~concerto.training.config.EgoAHTConfig`.

    Returns:
        The 18-key args dict HARL's :class:`HAPPO` constructor expects.
    """
    args = dict(_HARL_FIXED_DEFAULTS)
    args["lr"] = happo.lr
    args["clip_param"] = happo.clip_eps
    args["ppo_epoch"] = happo.n_epochs
    args["actor_num_mini_batch"] = max(1, happo.rollout_length // happo.batch_size)
    args["hidden_sizes"] = [happo.hidden_dim, happo.hidden_dim]
    return args


@dataclass(frozen=True)
class _PendingStep:
    """Pre-step cache populated by :meth:`EgoPPOTrainer.act` (M4b-8a; plan/05 §3.5).

    The four fields are computed together at action time and consumed
    together at observe time. Wrapping them in a frozen dataclass lets a
    single ``self._pending is not None`` check narrow all four for type
    checkers, eliminating the per-attribute defensive asserts that
    otherwise trip ruff's S101 rule.
    """

    obs: NDArray[np.float32]
    action: NDArray[np.float32]
    log_prob: NDArray[np.float32]
    value: float


def _flat_ego_obs(obs: Mapping[str, Any], ego_uid: str) -> NDArray[np.float32]:
    """Extract the ego's flat ``state`` vector from a nested env obs dict (plan/04 §3.4).

    The CONCERTO env contract (plan/04 §3.4 + plan/01 §3) keys observations
    as ``obs["agent"][uid]["state"]`` so the M2 comm + M3 safety wrappers
    drop in unchanged. HARL's MLPBase, however, expects a flat
    ``(batch, obs_dim)`` ndarray; this helper bridges the two.
    """
    return np.asarray(obs["agent"][ego_uid]["state"], dtype=np.float32)


class EgoPPOTrainer:
    """Ego-PPO trainer wrapping HARL's HAPPO actor (M4b-8a; ADR-002 §Decisions).

    Implements the :class:`~concerto.training.ego_aht.EgoTrainer` Protocol
    over a HARL :class:`HAPPO` actor + a hand-rolled :class:`_EgoCritic`.
    The training loop in :func:`concerto.training.ego_aht.train` calls
    :meth:`act` per step, :meth:`observe` per step (with the *next* obs +
    reward + done), and :meth:`update` at every ``rollout_length``
    boundary. The trainer's internal rollout buffer accumulates
    ``(obs, action, log_prob, value, reward, done)`` tuples between
    update calls; :meth:`update` then runs GAE + normalization + a HARL
    PPO update + a hand-rolled critic update, then clears the buffer.

    Determinism caveat (P6 / ADR-002): HARL's :class:`DiagGaussian` samples
    actions via :meth:`torch.distributions.Normal.sample`, which uses
    torch's *global* RNG. The trainer seeds that global RNG once at
    construction time via
    :func:`concerto.training.seeding.derive_substream`; this is enough
    for byte-identical CPU runs at the same seed but means concurrent
    trainers in the same process would alias each other's draws.
    Phase-0 runs are single-trainer per process, so this is acceptable.

    See plan/05 §3.5 for the architecture rationale (rollout/update logic
    on the CONCERTO side, not in the fork) and plan/05 §5 for the
    empirical-guarantee verification contract (T4b.13).
    """

    #: Class-level marker for the property test ``test_advantage_decomposition.py``
    #: to assert this trainer routes through the module-level
    #: :func:`compute_gae` + :func:`normalize_advantages` (rather than a
    #: shadow re-implementation). Keeps the test honest if a refactor
    #: ever inlines the GAE math into :meth:`update`.
    USES_MODULE_GAE: ClassVar[bool] = True

    def __init__(
        self,
        *,
        cfg: EgoAHTConfig,
        ego_uid: str,
        ego_obs_space: gym.spaces.Box,  # type: ignore[type-arg]
        ego_act_space: gym.spaces.Box,  # type: ignore[type-arg]
        device: torch.device | None = None,
    ) -> None:
        """Build the trainer (M4b-8a; ADR-002 §Decisions; plan/05 §3.5).

        Prefer :meth:`from_config` over direct construction — it derives
        ``ego_obs_space`` and ``ego_act_space`` from the env so the call
        site cannot drift from the env's actual shapes.

        Args:
            cfg: Validated :class:`~concerto.training.config.EgoAHTConfig`.
                The trainer reads ``cfg.seed``, ``cfg.happo.*``.
            ego_uid: The env-side uid the ego acts on (matches
                ``cfg.env.agent_uids[0]``).
            ego_obs_space: The Box space for the ego's flat state vector.
                Synthesized by :meth:`from_config` from the env's nested
                observation_space.
            ego_act_space: The Box space for the ego's action vector.
                Read directly from ``env.action_space[ego_uid]``.
            device: Torch device. Defaults to CPU; the M4b plan budgets
                everything for CPU (Mac M5, no CUDA).
        """
        self._device = device or torch.device("cpu")
        self._ego_uid = ego_uid
        self._gamma = cfg.happo.gamma
        self._gae_lambda = cfg.happo.gae_lambda
        self._n_epochs = cfg.happo.n_epochs
        self._batch_size = cfg.happo.batch_size
        self._rollout_length = cfg.happo.rollout_length
        # Seed torch's global RNG. ``derive_substream`` is the project-wide
        # P6 reproducibility entry-point; we cast its 64-bit draw down to
        # int32 because ``torch.manual_seed`` documents 64-bit input but
        # has historically failed silently on values > 2**63 on macOS.
        torch_seed_int = int(
            derive_substream(_TORCH_SUBSTREAM, root_seed=cfg.seed)
            .default_rng()
            .integers(0, 2**31 - 1)
        )
        torch.manual_seed(torch_seed_int)
        self._minibatch_rng: np.random.Generator = derive_substream(
            _MINIBATCH_SUBSTREAM, root_seed=cfg.seed
        ).default_rng()

        harl_args = _build_harl_args(happo=cfg.happo)
        self._harl_args = harl_args
        self._happo: HAPPO = HAPPO(harl_args, ego_obs_space, ego_act_space, self._device)
        obs_dim = int(ego_obs_space.shape[0])
        self._obs_dim = obs_dim
        self._act_dim = int(ego_act_space.shape[0])
        self._critic = _EgoCritic(obs_dim=obs_dim, hidden_dim=cfg.happo.hidden_dim).to(self._device)
        self._critic_optim = torch.optim.Adam(self._critic.parameters(), lr=cfg.happo.lr)

        # Rollout buffer (cleared after each update).
        self._buf_obs: list[NDArray[np.float32]] = []
        self._buf_actions: list[NDArray[np.float32]] = []
        self._buf_log_probs: list[NDArray[np.float32]] = []
        self._buf_values: list[float] = []
        self._buf_rewards: list[float] = []
        # ADR-002 §Decisions + Pardo 2017 / issue #62: terminated and
        # truncated are stored *separately* so :meth:`update` can build a
        # per-step bootstrap target. Treating truncation as termination
        # zeros V(s_next) at every time-limit boundary, biasing advantages
        # and slowing learning 2-3x on time-limit-only envs.
        self._buf_terminated: list[bool] = []
        self._buf_truncated: list[bool] = []
        # V(s_truncated_final) computed via the critic at observe time
        # when a truncation occurs. Indexed parallel to the buffer; unused
        # at non-truncation steps but populated with 0.0 so the array
        # shapes stay aligned.
        self._buf_truncation_bootstraps: list[float] = []
        # Pre-step cache (filled by :meth:`act`, consumed by :meth:`observe`).
        self._pending: _PendingStep | None = None
        # Last next-obs (cached by :meth:`observe`) for the critic
        # bootstrap at update time.
        self._last_next_obs: NDArray[np.float32] | None = None

    @classmethod
    def from_config(
        cls,
        cfg: EgoAHTConfig,
        *,
        env: EnvLike,
        ego_uid: str,
    ) -> EgoPPOTrainer:
        """Build from :class:`EgoAHTConfig` + a concrete env (TrainerFactory; ADR-002 §Decisions).

        This is the :class:`~concerto.training.ego_aht.TrainerFactory`
        callable that :func:`concerto.training.ego_aht.train` plugs in via
        dependency injection. The factory reads the ego's observation +
        action shapes off the env directly so the trainer's network
        widths cannot drift from what the env actually emits.

        Per plan/05 §3.5 the bridge from :class:`EgoAHTConfig` to HARL's
        args dict lives in this CONCERTO-side module rather than in the
        fork's :meth:`EgoAHTHAPPO.from_config` (which intentionally
        stays :class:`NotImplementedError` — see
        ``scripts/harl-fork-patches/v0.1.0-aht/`` README for the
        role-split rationale).

        Args:
            cfg: Validated :class:`~concerto.training.config.EgoAHTConfig`.
            env: Concrete env satisfying :class:`EnvLike`. Must expose
                ``observation_space["agent"][ego_uid]["state"]`` as a
                :class:`gym.spaces.Box` and ``action_space[ego_uid]``
                as a :class:`gym.spaces.Box`.
            ego_uid: The env-side uid the ego acts on.

        Returns:
            A constructed :class:`EgoPPOTrainer` ready for
            :func:`concerto.training.ego_aht.train`.

        Raises:
            TypeError: If the env's ego observation or action space is
                not a :class:`gym.spaces.Box`.
        """
        env_obs_space = env.observation_space  # type: ignore[attr-defined]
        ego_state_space = env_obs_space["agent"][ego_uid]["state"]
        if not isinstance(ego_state_space, gym.spaces.Box):
            raise TypeError(
                "EgoPPOTrainer.from_config requires "
                f"env.observation_space['agent'][{ego_uid!r}]['state'] to be "
                f"a gym.spaces.Box; got {type(ego_state_space).__name__}."
            )
        env_act_space = env.action_space  # type: ignore[attr-defined]
        ego_act_space = env_act_space[ego_uid]
        if not isinstance(ego_act_space, gym.spaces.Box):
            raise TypeError(
                "EgoPPOTrainer.from_config requires "
                f"env.action_space[{ego_uid!r}] to be a gym.spaces.Box; "
                f"got {type(ego_act_space).__name__}."
            )
        return cls(
            cfg=cfg,
            ego_uid=ego_uid,
            ego_obs_space=ego_state_space,
            ego_act_space=ego_act_space,
        )

    def act(
        self,
        obs: Mapping[str, Any],
        *,
        deterministic: bool = False,
    ) -> NDArray[np.floating]:
        """Sample the ego action; cache log-prob + value for :meth:`observe` (ADR-002 §Decisions).

        Caches ``(obs, action, log_prob, value)`` in private state so the
        next call to :meth:`observe` can insert them into the rollout
        buffer with the actual reward + done flag. The pairing relies on
        the contract in :func:`concerto.training.ego_aht.train` (each
        ``act(obs)`` is followed by exactly one
        ``observe(obs_next, reward, done)`` before the next ``act``).

        Args:
            obs: Env obs dict. The ego's flat state vector is read from
                ``obs["agent"][ego_uid]["state"]``.
            deterministic: When ``True``, returns the distribution mode
                instead of a sample. Used by evaluation paths
                (M4b-8b's per-checkpoint deterministic eval).

        Returns:
            The ego action as a length-``act_dim`` float32 array, ready
            to be packed into the env's dict-action.
        """
        flat = _flat_ego_obs(obs, self._ego_uid)
        # HARL expects shape ``(batch, obs_dim)`` and a positional
        # ``(rnn_states, masks)`` even when the policy is feedforward.
        # rnn_states is ignored downstream (use_recurrent_policy=False);
        # masks=1.0 means "do not reset the RNN state" — also a no-op
        # without an RNN. We pass shape-correct placeholders so HARL's
        # internal ``check`` calls accept them.
        obs_t = flat.reshape(1, -1)
        rnn_states = np.zeros(
            (1, self._harl_args["recurrent_n"], self._harl_args["hidden_sizes"][-1]),
            dtype=np.float32,
        )
        masks = np.ones((1, 1), dtype=np.float32)
        with torch.no_grad():
            action_t, log_prob_t, _ = self._happo.get_actions(
                obs_t, rnn_states, masks, None, deterministic
            )
            value_t = self._critic(torch.from_numpy(flat).to(self._device).unsqueeze(0))
        action = action_t.detach().cpu().numpy().squeeze(0).astype(np.float32)
        log_prob = log_prob_t.detach().cpu().numpy().squeeze(0).astype(np.float32)
        value = float(value_t.detach().cpu().numpy().squeeze())
        self._pending = _PendingStep(obs=flat, action=action, log_prob=log_prob, value=value)
        return action

    def observe(
        self,
        obs: Mapping[str, Any],
        reward: float,
        done: bool,
        *,
        truncated: bool = False,
    ) -> None:
        """Buffer ``(pending, reward, terminated, truncated)`` (ADR-002 §Decisions; Pardo 2017).

        The trainer keeps a single ``(obs, action, log_prob, value)``
        cache — set by :meth:`act` — and pairs it with the
        ``(reward, done, truncated)`` from :meth:`observe`. The post-step
        ``obs`` is also remembered as the bootstrap target for
        :meth:`update`.

        ``done`` keeps the historical "this step ended the episode" meaning
        (terminated OR truncated). ``truncated`` is the new keyword-only
        flag that distinguishes time-limit truncation from true
        termination. The trainer derives
        ``terminated = done and not truncated`` internally; the GAE
        bootstrap then treats truncation correctly per Pardo et al. 2017
        (V(s_truncated_final) flows through, not 0) — see :func:`compute_gae`
        + project issue #62 for the root-cause writeup.

        At truncation boundaries this method calls the critic on the
        post-step obs once to capture V(s_truncated_final); that value
        is later threaded into :func:`compute_gae` as the per-step
        bootstrap target. At non-truncation steps the recorded value is
        ``0.0`` and unused.

        Args:
            obs: Post-step env obs (the obs the env returned after the
                action passed to the most recent :meth:`act` call).
            reward: Per-step ego reward (the env's shared scalar).
            done: ``True`` if the env terminated *or* truncated this step
                (i.e. the loop should reset the env for the next step).
            truncated: ``True`` iff this step ended the episode via
                time-limit truncation (env returned ``truncated=True``).
                Default ``False`` is the conservative legacy behavior —
                callers that don't yet distinguish truncation from
                termination see the historical "treat as terminal"
                semantics.
        """
        pending = self._pending
        if pending is None:
            # No prior :meth:`act` — defensive no-op so the trainer can
            # safely ignore a stray ``observe`` (eg. a pre-rollout
            # bookkeeping call).
            return
        terminated = bool(done) and not bool(truncated)
        truncated_b = bool(truncated)
        self._buf_obs.append(pending.obs)
        self._buf_actions.append(pending.action)
        self._buf_log_probs.append(pending.log_prob)
        self._buf_values.append(pending.value)
        self._buf_rewards.append(float(reward))
        self._buf_terminated.append(terminated)
        self._buf_truncated.append(truncated_b)
        next_flat = _flat_ego_obs(obs, self._ego_uid)
        # Capture V(s_truncated_final) eagerly at the boundary. Doing it
        # here (rather than at update time) is the cleanest place because
        # ``next_flat`` is the actual post-step obs of the truncated
        # episode; by update time the training loop has already reset
        # the env, and the per-step "next obs in buffer" at index
        # ``t + 1`` is from the *new* episode. The cost is one extra
        # critic forward pass per truncation boundary.
        if truncated_b and not terminated:
            with torch.no_grad():
                v_trunc = float(
                    self._critic(torch.from_numpy(next_flat).to(self._device).unsqueeze(0))
                    .squeeze()
                    .item()
                )
            self._buf_truncation_bootstraps.append(v_trunc)
        else:
            self._buf_truncation_bootstraps.append(0.0)
        self._last_next_obs = next_flat
        # Clear the pre-step cache so a stray double-observe cannot
        # double-insert the same tuple.
        self._pending = None

    def _build_next_values(
        self,
        values: NDArray[np.float32],
        terminated: NDArray[np.bool_],
        truncated: NDArray[np.bool_],
        truncation_bootstraps: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Per-step ``V(s_{t+1})`` bootstrap target for GAE (Pardo 2017 §4; issue #62).

        Mid-rollout non-boundary steps use ``values[t + 1]``. At
        terminations the bootstrap is ``0`` (true terminal: no value
        after). At truncations it is ``V(s_truncated_final)`` captured
        at observe time. The rollout-tail step (``t = n-1``) follows the
        same three-way rule, with the live critic call against the
        remembered next-obs as the non-boundary tail bootstrap.
        """
        n_steps = values.shape[0]
        next_values = np.zeros(n_steps, dtype=np.float32)
        for t in range(n_steps - 1):
            if terminated[t]:
                next_values[t] = 0.0
            elif truncated[t]:
                next_values[t] = truncation_bootstraps[t]
            else:
                next_values[t] = values[t + 1]
        last_t = n_steps - 1
        if terminated[last_t]:
            next_values[last_t] = 0.0
        elif truncated[last_t]:
            next_values[last_t] = truncation_bootstraps[last_t]
        elif self._last_next_obs is None:
            # Defensive: no remembered next-obs (rollout never observed
            # anything past the last step). Falls back to zero — safe
            # because the rollout will be cleared and the next one starts
            # fresh.
            next_values[last_t] = 0.0
        else:
            with torch.no_grad():
                next_values[last_t] = float(
                    self._critic(
                        torch.from_numpy(self._last_next_obs).to(self._device).unsqueeze(0)
                    )
                    .squeeze()
                    .item()
                )
        return next_values

    def update(self) -> None:
        """Compute GAE; run PPO + critic updates; clear the buffer (ADR-002 §Decisions; Pardo 2017).

        Order of operations (Schulman 2017, lifted to ego-only / frozen-
        partner per plan/05 §3.5):

        1. Build the per-step ``next_values`` bootstrap array (Pardo 2017
           §4): ``V(s_truncated_final_t)`` at truncation boundaries (cached
           at observe time), ``0.0`` at terminations, ``values[t + 1]``
           otherwise; rollout-tail bootstrap at ``t = T - 1`` is the same
           rule applied to ``self._last_next_obs``.
        2. :func:`compute_gae` → raw advantages ``A_t``.
        3. ``returns_t = A_t + V(s_t)`` (the critic regression target).
        4. :func:`normalize_advantages` → normalized ``A_t`` for the
           PPO importance-weighted surrogate loss.
        5. For each of ``n_epochs`` epochs, shuffle indices and walk
           ``actor_num_mini_batch`` minibatches. For each minibatch, build
           the 9-tuple :meth:`HAPPO.update` expects and call it directly
           (skipping :meth:`HAPPO.train`'s buffer-machinery wrapper).
        6. Train the critic on the same minibatch with MSE on returns.
        7. Clear the buffer.

        The ``factor_batch`` argument :meth:`HAPPO.update` requires for
        HAPPO's sequential multi-agent update is set to all ``1.0`` — the
        ego is the only updating agent (ADR-009 §Decision: black-box AHT,
        partner is frozen), so previous-agent importance weights are
        identically 1.
        """
        if not self._buf_rewards:
            return

        rewards = np.asarray(self._buf_rewards, dtype=np.float32)
        values = np.asarray(self._buf_values, dtype=np.float32)
        terminated = np.asarray(self._buf_terminated, dtype=np.bool_)
        truncated = np.asarray(self._buf_truncated, dtype=np.bool_)
        truncation_bootstraps = np.asarray(self._buf_truncation_bootstraps, dtype=np.float32)
        next_values = self._build_next_values(values, terminated, truncated, truncation_bootstraps)
        episode_boundaries = terminated | truncated
        raw_advantages = compute_gae(
            rewards,
            values,
            next_values,
            episode_boundaries,
            gamma=self._gamma,
            gae_lambda=self._gae_lambda,
        )
        returns = (raw_advantages + values).astype(np.float32)
        norm_advantages = normalize_advantages(raw_advantages)

        obs_arr = np.stack(self._buf_obs).astype(np.float32)
        actions_arr = np.stack(self._buf_actions).astype(np.float32)
        log_probs_arr = np.stack(self._buf_log_probs).astype(np.float32)

        n_steps = rewards.shape[0]
        n_minibatches = max(1, n_steps // self._batch_size)
        mb_size = max(1, n_steps // n_minibatches)

        for _epoch in range(self._n_epochs):
            perm = self._minibatch_rng.permutation(n_steps)
            for mb in range(n_minibatches):
                start = mb * mb_size
                end = start + mb_size if mb < n_minibatches - 1 else n_steps
                mb_idx = perm[start:end]
                mb_obs = obs_arr[mb_idx]
                mb_actions = actions_arr[mb_idx]
                mb_old_log_probs = log_probs_arr[mb_idx]
                # HAPPO expects ``adv_targ`` shape ``(mb, 1)``. log-probs
                # for diag-Gaussian are already per-action-dim; HAPPO's
                # action_aggregation="prod" reduces them inside update().
                mb_advantages = norm_advantages[mb_idx].reshape(-1, 1).astype(np.float32)
                this_mb_size = mb_idx.shape[0]
                mb_rnn_states = np.zeros(
                    (
                        this_mb_size,
                        self._harl_args["recurrent_n"],
                        self._harl_args["hidden_sizes"][-1],
                    ),
                    dtype=np.float32,
                )
                mb_masks = np.ones((this_mb_size, 1), dtype=np.float32)
                mb_active_masks = np.ones((this_mb_size, 1), dtype=np.float32)
                # ``factor_batch`` = 1 because the ego is the only updating
                # agent (ADR-009 §Decision); see plan/05 §3.5.
                mb_factor = np.ones((this_mb_size, 1), dtype=np.float32)
                sample = (
                    mb_obs,
                    mb_rnn_states,
                    mb_actions,
                    mb_masks,
                    mb_active_masks,
                    mb_old_log_probs,
                    mb_advantages,
                    None,
                    mb_factor,
                )
                self._happo.update(sample)

                # Critic update: simple MSE regression on returns.
                mb_returns = torch.from_numpy(returns[mb_idx]).to(self._device).float()
                mb_obs_t = torch.from_numpy(mb_obs).to(self._device).float()
                value_pred = self._critic(mb_obs_t).squeeze(-1)
                critic_loss = ((value_pred - mb_returns) ** 2).mean()
                self._critic_optim.zero_grad()
                critic_loss.backward()
                self._critic_optim.step()

        # Reset rollout buffer.
        self._buf_obs.clear()
        self._buf_actions.clear()
        self._buf_log_probs.clear()
        self._buf_values.clear()
        self._buf_rewards.clear()
        self._buf_terminated.clear()
        self._buf_truncated.clear()
        self._buf_truncation_bootstraps.clear()
        self._last_next_obs = None

    def state_dict(self) -> dict[str, Any]:
        """Return a flat dict with actor + critic + optimizer states (ADR-002 §Decisions; T4b.12).

        The returned dict is what :func:`concerto.training.checkpoints.save_checkpoint`
        ``torch.save``s to the ``.pt`` artefact. Loading is the dual:
        each component reads its own sub-key off the loaded dict via
        :meth:`load_state_dict`.
        """
        return {
            "actor": self._happo.actor.state_dict(),
            "actor_optim": self._happo.actor_optimizer.state_dict(),
            "critic": self._critic.state_dict(),
            "critic_optim": self._critic_optim.state_dict(),
        }


__all__ = [
    "EgoPPOTrainer",
    "compute_gae",
    "normalize_advantages",
]
