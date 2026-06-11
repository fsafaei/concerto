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

    import structlog
    from numpy.typing import NDArray

    from concerto.training.config import EgoAHTConfig, HAPPOHyperparams
    from concerto.training.ego_aht import EnvLike, PartnerLike

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


def _set_torch_determinism(enabled: bool) -> None:
    """Wrap :func:`torch.use_deterministic_algorithms` for test patching (plan/08 §4).

    Routed through a module-level helper so
    ``tests/unit/test_ego_ppo_trainer.py`` can patch it during the
    trainer-construction tests — calling the real torch function in a
    unit test would impose a process-global side effect on every
    subsequent test in the session, which is exactly the kind of
    leakage the trainer's docstring warns about.

    ``warn_only=True`` so ops without a deterministic implementation
    (rare; nearly all of torch's MLP / softmax / Adam path is already
    deterministic) emit a warning rather than raising — the trainer's
    main loop must keep running. ADR-002 §Decisions: the determinism
    contract is per-tensor-op, not strict-mode.
    """
    torch.use_deterministic_algorithms(enabled, warn_only=True)


def _assert_partner_is_frozen(partner: object) -> None:
    """Refuse a partner with any trainable parameter (ADR-009 §Consequences; plan/05 §6 #3).

    The black-box AHT contract (ADR-009 §Decision) requires every
    partner to be frozen during ego training. Two enforcement paths
    coexist in this project:

    1. The :data:`chamber.partners.interface._FORBIDDEN_ATTRS` shield
       on :class:`~chamber.partners.interface.PartnerBase` raises
       :class:`AttributeError` on ``partner.named_parameters``
       look-ups, citing ADR-009 §Consequences. Any project-shipped
       partner subclasses :class:`PartnerBase` and therefore enters
       this branch — we catch the :class:`AttributeError` and treat
       it as ``contract enforced by the shield``.

    2. A custom partner adapter (a research-fork experiment, a
       regression-test fixture, a future zoo partner that bypasses
       :class:`PartnerBase`) exposes ``named_parameters()`` directly.
       We walk it and raise :class:`ValueError` on the first parameter
       with ``requires_grad is True``. The error message cites
       ``ADR-009 §Consequences`` and the offending parameter path
       so the caller can jump straight to the unfrozen weight.

    Wired into :meth:`EgoPPOTrainer.from_config` as the *first*
    construction step (before any tensor allocation) so a bad partner
    aborts the run cheaply, with a message that points at the
    architectural rule it violates.

    Note on the dual-path design: the obvious implementation would walk
    ``partner.named_parameters()`` unconditionally, but the existing
    :class:`PartnerBase` shield (project-wide since M4) intentionally
    raises :class:`AttributeError` on that lookup — bypassing the
    shield to walk the params is a project anti-pattern (the shield's
    purpose is the same ADR-009 §Consequences contract this helper
    enforces). The dual path preserves the shield's invariant for
    every project-shipped partner *and* adds the explicit walk for
    custom partners that lack a shield.
    """
    # The PartnerBase shield raises AttributeError on the *attribute*
    # lookup (``partner.named_parameters``), not on the call. We narrow
    # the AttributeError catch to the lookup only — a bug in a custom
    # partner's ``named_parameters`` implementation (e.g. a property
    # whose getter raises AttributeError on a misspelled internal
    # attribute) must surface as a loud error rather than be silently
    # swallowed as "no params, partner is frozen".
    try:
        params_method = partner.named_parameters  # type: ignore[attr-defined]
    except AttributeError:
        # Either the partner has no ``named_parameters`` at all (no
        # torch state to leak — pure-Python heuristic partners), or
        # the PartnerBase shield intercepted the look-up (the shield's
        # AttributeError is itself an ADR-009 §Consequences ack, so the
        # contract is enforced). Proceed to construct the trainer.
        return
    # Call the bound method explicitly. Any error raised here — TypeError
    # from a non-callable attribute, a runtime error inside the
    # method body — propagates so a bug in the partner adapter is loud,
    # not silent.
    params_iter = params_method()
    # The partner exposed torch params: walk and refuse on the first
    # parameter with ``requires_grad=True``.
    for name, param in params_iter:
        if getattr(param, "requires_grad", False):
            cls_name = type(partner).__name__
            msg = (
                f"EgoPPOTrainer requires a frozen partner: partner of class "
                f"{cls_name!r} has parameter {name!r} with requires_grad=True. "
                f"ADR-009 §Consequences: black-box AHT — the partner is frozen "
                f"during ego training; set ``param.requires_grad = False`` for "
                f"every parameter before passing the partner to the trainer, "
                f"or wrap the partner in a "
                f"``chamber.partners.interface.PartnerBase`` subclass (which "
                f"enforces the no-joint-training shield at attribute-lookup "
                f"time). See plan/05 §6 #3 for the project rationale."
            )
            raise ValueError(msg)


def _resolve_device(requested: str) -> torch.device:
    """Resolve a :class:`RuntimeConfig` ``device`` string (ADR-002 §Decisions).

    ``"auto"`` defers to :func:`chamber.utils.device.torch_device`, which
    picks the best available backend (CUDA > MPS > CPU). Explicit
    strings ``"cpu"`` / ``"cuda"`` / ``"mps"`` raise
    :class:`RuntimeError` if the requested backend is unavailable.

    Args:
        requested: ``cfg.runtime.device`` value (literal-checked at
            :class:`RuntimeConfig` validation time, so only the four
            documented strings reach this resolver).

    Returns:
        A concrete :class:`torch.device`.

    Raises:
        RuntimeError: If ``requested in {"cuda", "mps"}`` and the
            backend is unavailable. Error message cites ADR-002
            §Decisions so a confused user searching for the message
            lands on the ADR.
    """
    if requested == "auto":
        from chamber.utils.device import torch_device

        return torch.device(torch_device())
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "RuntimeConfig.device='cuda' requested but torch.cuda.is_available() "
                "is False. Either flip the config to 'auto' (resolves to CPU on this "
                "host) or run on a CUDA-capable Linux box. See ADR-002 §Decisions for "
                "the device-resolution contract."
            )
        return torch.device("cuda")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "RuntimeConfig.device='mps' requested but torch.backends.mps.is_available() "
                "is False. Flip the config to 'auto' or 'cpu'. See ADR-002 §Decisions for "
                "the device-resolution contract."
            )
        return torch.device("mps")
    return torch.device("cpu")


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

    Vectorised cells (P1.05.10; ADR-007 §Stage 1b regime-alignment
    revision): ``obs`` / ``action`` / ``log_prob`` carry a leading
    ``(num_envs, …)`` batch dim and ``value`` is the per-env value
    array. The single-env layout (1-D obs, scalar value) is unchanged.
    """

    obs: NDArray[np.float32]
    action: NDArray[np.float32]
    log_prob: NDArray[np.float32]
    value: float | NDArray[np.float32]


def _flat_ego_obs(obs: Mapping[str, Any], ego_uid: str) -> NDArray[np.float32]:
    """Extract the ego's flat ``state`` vector from a nested env obs dict (plan/04 §3.4).

    The CONCERTO env contract (plan/04 §3.4 + plan/01 §3) keys observations
    as ``obs["agent"][uid]["state"]`` so the M2 comm + M3 safety wrappers
    drop in unchanged. HARL's MLPBase, however, expects a flat
    ``(batch, obs_dim)`` ndarray; this helper bridges the two. Vectorised
    cells (P1.05.10) emit ``state`` as ``(num_envs, obs_dim)`` — passed
    through with the batch dim intact. Torch tensors (ManiSkill GPU obs)
    are detached to numpy.
    """
    state = obs["agent"][ego_uid]["state"]
    if hasattr(state, "detach"):
        state = state.detach().cpu().numpy()
    return np.asarray(state, dtype=np.float32)


def _to_float_vec(value: Any, n: int) -> NDArray[np.float32]:  # noqa: ANN401 - scalar / ndarray / torch tensor
    """Coerce a per-step reward to a ``(n,)`` float32 vector (P1.05.10; ADR-007 §Stage 1b)."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32).reshape(n)


def _to_bool_vec(value: Any, n: int) -> NDArray[np.bool_]:  # noqa: ANN401 - scalar / ndarray / torch tensor
    """Coerce a done/truncated flag to a ``(n,)`` bool vector (P1.05.10; ADR-007 §Stage 1b)."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value)
    if arr.ndim == 0:
        arr = np.full(n, bool(arr))
    return arr.reshape(n).astype(bool)


def _per_env_box(space: gym.spaces.Box) -> gym.spaces.Box:  # type: ignore[type-arg]
    """Strip a leading batch dim off a ManiSkill batched Box (P1.05.10; ADR-007 §Stage 1b).

    Vectorised ManiSkill envs expose batched ``(num_envs, dim)`` spaces;
    HARL's HAPPO + the critic are sized per env. Returns the input
    unchanged when it is already 1-D (the single-env path — no object
    churn, byte-identical construction).
    """
    if space.shape is None or len(space.shape) <= 1:
        return space
    low = np.asarray(space.low)[0]
    high = np.asarray(space.high)[0]
    return gym.spaces.Box(low=low, high=high, shape=(int(space.shape[-1]),), dtype=np.float32)


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
        partner: PartnerLike,
        device: torch.device | None = None,
        logger: structlog.BoundLogger | None = None,
    ) -> None:
        """Build the trainer (M4b-8a; ADR-002 §Decisions; plan/05 §3.5).

        Prefer :meth:`from_config` over direct construction — it derives
        ``ego_obs_space`` and ``ego_act_space`` from the env so the call
        site cannot drift from the env's actual shapes.

        The partner-freeze gate (ADR-009 §Consequences; plan/05 §6 #3)
        runs first, before any tensor allocation: a partner with any
        ``requires_grad=True`` parameter (and no
        :class:`~chamber.partners.interface.PartnerBase` shield blocking
        ``named_parameters`` access) aborts the construction with a
        :class:`ValueError` that names the offending parameter path.

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
            partner: The frozen partner instance the ego will be trained
                against. Validated via :func:`_assert_partner_is_frozen`
                before any tensor allocation. The trainer does not retain
                a reference — the partner is consumed by the training
                loop in :func:`concerto.training.ego_aht.train`, not by
                the trainer itself.
            device: Torch device. Defaults to CPU when called directly;
                :meth:`from_config` resolves it from
                :attr:`RuntimeConfig.device` (auto / cpu / cuda / mps)
                via :func:`_resolve_device` and applies the determinism
                flag from :attr:`RuntimeConfig.deterministic_torch` on
                CPU before constructing the trainer (ADR-002 §Decisions;
                plan/08 §4).
            logger: Optional structlog ``BoundLogger`` from
                :func:`concerto.training.logging.bind_run_logger`.
                When non-``None``, every PPO update emits one
                ``event="scalar"`` line per rollout via
                :func:`concerto.training.logging.log_scalars` (with
                ``metric_namespace="train"``) carrying HARL's
                ``policy_loss``, ``dist_entropy``, ``actor_grad_norm``
                + the trainer's locally-computed ``critic_loss``,
                ``approx_kl``, ``clip_fraction``, ``ratio_min/max``,
                advantage stats, and learning rate. Default ``None``
                preserves the pre-P1.05.11 silent-trainer behaviour
                so existing callers (tests, the non-Stage-1b paths)
                see no change.

        Raises:
            ValueError: If ``partner`` exposes a torch parameter with
                ``requires_grad=True`` (ADR-009 §Consequences).
        """
        # ADR-009 §Consequences + plan/05 §6 #3: partner-freeze gate is
        # the first construction step, before any tensor allocation.
        _assert_partner_is_frozen(partner)
        self._device = device or torch.device("cpu")
        self._ego_uid = ego_uid
        self._gamma = cfg.happo.gamma
        self._gae_lambda = cfg.happo.gae_lambda
        self._n_epochs = cfg.happo.n_epochs
        self._batch_size = cfg.happo.batch_size
        self._rollout_length = cfg.happo.rollout_length
        # P1.05.11 / ADR-017: capture PPO clip + learning rate locally so
        # ``update()`` can compute clip_fraction + emit lr per rollout.
        # No behaviour change — these are read-only references to cfg.
        self._clip_param: float = cfg.happo.clip_eps
        self._learning_rate: float = cfg.happo.lr
        # P1.05.11 / ADR-017: optional bound logger from
        # :func:`concerto.training.logging.bind_run_logger`. ``None``
        # (default) preserves the pre-P1.05.11 silent-trainer behaviour.
        # The driver
        # :func:`concerto.training.ego_aht.train` threads its bound
        # logger here via :meth:`from_config`'s ``logger=`` kwarg.
        self._logger: structlog.BoundLogger | None = logger
        # P1.05.11 / ADR-017: global-step counter incremented per
        # :meth:`observe` call (one env step). ``update()`` emits its
        # scalar line at this step so the W&B x-axis matches frames.
        self._global_step: int = 0
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

        # Rollout buffer (cleared after each update). Single-env cells
        # buffer scalars per step; vectorised cells (P1.05.10; ADR-007
        # §Stage 1b regime-alignment) buffer per-env vectors per step —
        # :meth:`update` branches on the buffered layout.
        self._buf_obs: list[NDArray[np.float32]] = []
        self._buf_actions: list[NDArray[np.float32]] = []
        self._buf_log_probs: list[NDArray[np.float32]] = []
        self._buf_values: list[float | NDArray[np.float32]] = []
        self._buf_rewards: list[float | NDArray[np.float32]] = []
        # ADR-002 §Decisions + Pardo 2017 / issue #62: terminated and
        # truncated are stored *separately* so :meth:`update` can build a
        # per-step bootstrap target. Treating truncation as termination
        # zeros V(s_next) at every time-limit boundary, biasing advantages
        # and slowing learning 2-3x on time-limit-only envs.
        self._buf_terminated: list[bool | NDArray[np.bool_]] = []
        self._buf_truncated: list[bool | NDArray[np.bool_]] = []
        # V(s_truncated_final) computed via the critic at observe time
        # when a truncation occurs. Indexed parallel to the buffer; unused
        # at non-truncation steps but populated with 0.0 so the array
        # shapes stay aligned.
        self._buf_truncation_bootstraps: list[float | NDArray[np.float32]] = []
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
        partner: PartnerLike,
        ego_uid: str,
        logger: structlog.BoundLogger | None = None,
    ) -> EgoPPOTrainer:
        """Build from :class:`EgoAHTConfig` + a concrete env + the frozen partner.

        ADR-002 §Decisions / ADR-009 §Consequences / plan/05 §3.5 + §6 #3.

        This is the :class:`~concerto.training.ego_aht.TrainerFactory`
        callable that :func:`concerto.training.ego_aht.train` plugs in via
        dependency injection. The factory reads the ego's observation +
        action shapes off the env directly so the trainer's network
        widths cannot drift from what the env actually emits.

        The partner-freeze gate (:func:`_assert_partner_is_frozen`) runs
        first, before any tensor allocation, so a bad partner aborts
        cheaply (plan/05 §6 #3).

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
            partner: Frozen partner instance the ego will train against.
                Validated by :func:`_assert_partner_is_frozen` as the
                first construction step (ADR-009 §Consequences).
            ego_uid: The env-side uid the ego acts on.
            logger: Optional structlog ``BoundLogger`` from
                :func:`concerto.training.logging.bind_run_logger`.
                P1.05.11 / ADR-017 §Decisions: when supplied, every
                PPO update emits one namespaced ``event="scalar"``
                line. Default ``None`` preserves the pre-P1.05.11
                silent-trainer behaviour. Threaded through
                :class:`~concerto.training.ego_aht.TrainerFactory` by
                the canonical training driver.

        Returns:
            A constructed :class:`EgoPPOTrainer` ready for
            :func:`concerto.training.ego_aht.train`.

        Raises:
            ValueError: If ``partner`` exposes a torch parameter with
                ``requires_grad=True`` (ADR-009 §Consequences).
            TypeError: If the env's ego observation or action space is
                not a :class:`gym.spaces.Box`.
        """
        # ADR-009 §Consequences / plan/05 §6 #3: partner-freeze gate
        # runs first, before any env-space introspection or tensor
        # allocation. A bad partner aborts the construction cheaply.
        _assert_partner_is_frozen(partner)
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
        # P1.05.10 (ADR-007 §Stage 1b regime-alignment): vectorised
        # ManiSkill envs expose batched (num_envs, dim) spaces; the
        # actor/critic are sized per env. _per_env_box is the identity
        # for the historical 1-D spaces.
        ego_state_space = _per_env_box(ego_state_space)
        ego_act_space = _per_env_box(ego_act_space)
        device = _resolve_device(cfg.runtime.device)
        if device.type == "cpu" and cfg.runtime.deterministic_torch:
            # CPU determinism is the project's P6 contract (plan/08 §4).
            # Skip on CUDA/MPS where the contract is relaxed because
            # cuDNN / Metal kernels are not bit-deterministic even with
            # this flag.
            _set_torch_determinism(True)
        return cls(
            cfg=cfg,
            ego_uid=ego_uid,
            ego_obs_space=ego_state_space,
            ego_act_space=ego_act_space,
            partner=partner,
            device=device,
            logger=logger,
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
        if flat.ndim == 2:  # noqa: PLR2004 - rank-2 is the vectorised (num_envs, obs_dim) layout
            # P1.05.10 (ADR-007 §Stage 1b regime-alignment): vectorised
            # path — one HARL forward over the whole (num_envs, obs_dim)
            # batch; the action / log-prob / value keep the batch dim
            # so :meth:`observe` can buffer per-env rows.
            n = flat.shape[0]
            rnn_states_b = np.zeros(
                (n, self._harl_args["recurrent_n"], self._harl_args["hidden_sizes"][-1]),
                dtype=np.float32,
            )
            masks_b = np.ones((n, 1), dtype=np.float32)
            with torch.no_grad():
                action_t, log_prob_t, _ = self._happo.get_actions(
                    flat, rnn_states_b, masks_b, None, deterministic
                )
                value_t = self._critic(torch.from_numpy(flat).to(self._device))
            action_b = action_t.detach().cpu().numpy().astype(np.float32)
            log_prob_b = log_prob_t.detach().cpu().numpy().astype(np.float32)
            value_b = value_t.detach().cpu().numpy().reshape(-1).astype(np.float32)
            self._pending = _PendingStep(
                obs=flat, action=action_b, log_prob=log_prob_b, value=value_b
            )
            return action_b
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
        reward: Any,  # noqa: ANN401 - scalar float (single-env) or (num_envs,) array/tensor (vectorised; P1.05.10)
        done: Any,  # noqa: ANN401 - scalar bool or (num_envs,) array/tensor
        *,
        truncated: Any = False,  # noqa: ANN401 - scalar bool or (num_envs,) array/tensor
        final_obs: Mapping[str, Any] | None = None,
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
            final_obs: Vectorised cells only (P1.05.10; ADR-007
                §Stage 1b regime-alignment): the *pre-reset* batched
                obs of this step, surfaced by
                :class:`chamber.envs.stage1_vector.Stage1AutoResetWrapper`
                via ``info["final_observation"]``. Required whenever
                any env truncated this step — the auto-reset wrapper
                has already replaced ``obs`` with the new episode's
                reset obs, so the Pardo truncation bootstrap
                ``V(s_truncated_final)`` must be computed against
                ``final_obs`` instead. Ignored on the single-env path
                (the loop's act-then-observe-then-reset ordering means
                ``obs`` *is* the final obs there).
        """
        pending = self._pending
        if pending is None:
            # No prior :meth:`act` — defensive no-op so the trainer can
            # safely ignore a stray ``observe`` (eg. a pre-rollout
            # bookkeeping call).
            return
        if pending.action.ndim == 2:  # noqa: PLR2004 - rank-2 is the vectorised layout
            self._observe_vectorised(
                obs, reward, done, truncated=truncated, final_obs=final_obs, pending=pending
            )
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
        # P1.05.11 / ADR-017: bump the global step counter so the next
        # :meth:`update` emission carries the current frame index.
        self._global_step += 1

    def _observe_vectorised(
        self,
        obs: Mapping[str, Any],
        reward: Any,  # noqa: ANN401 - (num_envs,) array/tensor
        done: Any,  # noqa: ANN401 - (num_envs,) array/tensor
        *,
        truncated: Any,  # noqa: ANN401 - (num_envs,) array/tensor
        final_obs: Mapping[str, Any] | None,
        pending: _PendingStep,
    ) -> None:
        """Buffer one batched step (P1.05.10; ADR-007 §Stage 1b regime-alignment; Pardo 2017).

        The vectorised analogue of the single-env :meth:`observe` body:
        per-env rows are buffered with the batch dim intact (stacked to
        ``(T, num_envs, …)`` at :meth:`update` time), and the Pardo
        truncation bootstrap is computed against ``final_obs`` (the
        pre-reset obs from
        :class:`chamber.envs.stage1_vector.Stage1AutoResetWrapper`)
        for exactly the envs that truncated without terminating.
        """
        n = pending.action.shape[0]
        reward_arr = _to_float_vec(reward, n)
        done_arr = _to_bool_vec(done, n)
        trunc_arr = _to_bool_vec(truncated, n)
        terminated_arr = done_arr & ~trunc_arr
        self._buf_obs.append(pending.obs)
        self._buf_actions.append(pending.action)
        self._buf_log_probs.append(pending.log_prob)
        self._buf_values.append(np.asarray(pending.value, dtype=np.float32))  # type: ignore[arg-type]
        self._buf_rewards.append(reward_arr)  # type: ignore[arg-type]
        self._buf_terminated.append(terminated_arr)  # type: ignore[arg-type]
        self._buf_truncated.append(trunc_arr)  # type: ignore[arg-type]
        bootstraps = np.zeros(n, dtype=np.float32)
        trunc_only = trunc_arr & ~terminated_arr
        if bool(trunc_only.any()):
            if final_obs is None:
                msg = (
                    "EgoPPOTrainer._observe_vectorised: a truncation boundary "
                    "occurred but no final_obs was supplied. Vectorised cells "
                    "must surface the pre-reset obs via "
                    "info['final_observation'] (Stage1AutoResetWrapper) so the "
                    "Pardo truncation bootstrap is computed against the actual "
                    "terminal obs, not the new episode's reset obs "
                    "(issue #62; ADR-007 §Stage 1b regime-alignment)."
                )
                raise ValueError(msg)
            final_flat = _flat_ego_obs(final_obs, self._ego_uid)
            idx = np.nonzero(trunc_only)[0]
            with torch.no_grad():
                v_trunc = (
                    self._critic(torch.from_numpy(final_flat[idx]).to(self._device))
                    .detach()
                    .cpu()
                    .numpy()
                    .reshape(-1)
                )
            bootstraps[idx] = v_trunc.astype(np.float32)
        self._buf_truncation_bootstraps.append(bootstraps)  # type: ignore[arg-type]
        # Post-step obs (for done envs this is the auto-reset obs — only
        # consumed as the rollout-tail bootstrap for *non-boundary* tail
        # envs, where it is the genuine next obs).
        self._last_next_obs = _flat_ego_obs(obs, self._ego_uid)
        self._pending = None
        self._global_step += n

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

    def _build_next_values_vectorised(
        self,
        values: NDArray[np.float32],
        terminated: NDArray[np.bool_],
        truncated: NDArray[np.bool_],
        truncation_bootstraps: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Per-(step, env) GAE bootstrap targets (P1.05.10; Pardo 2017 §4; issue #62).

        The ``(T, N)`` analogue of :meth:`_build_next_values`: mid-rollout
        non-boundary entries use ``values[t + 1, e]``; terminations
        bootstrap ``0``; truncations use the per-env
        ``V(s_truncated_final)`` captured at observe time from
        ``final_obs``. The rollout-tail row applies the same three-way
        rule, with one batched critic call over the remembered
        ``(N, obs_dim)`` next obs as the non-boundary tail bootstrap.
        """
        n_steps, n_envs = values.shape
        next_values = np.zeros((n_steps, n_envs), dtype=np.float32)
        if n_steps > 1:
            next_values[:-1] = values[1:]
            next_values[:-1][truncated[:-1]] = truncation_bootstraps[:-1][truncated[:-1]]
            next_values[:-1][terminated[:-1]] = 0.0
        last_t = n_steps - 1
        if self._last_next_obs is None:
            tail = np.zeros(n_envs, dtype=np.float32)
        else:
            with torch.no_grad():
                tail = (
                    self._critic(torch.from_numpy(self._last_next_obs).to(self._device))
                    .detach()
                    .cpu()
                    .numpy()
                    .reshape(-1)
                    .astype(np.float32)
                )
        next_values[last_t] = tail
        next_values[last_t][truncated[last_t]] = truncation_bootstraps[last_t][truncated[last_t]]
        next_values[last_t][terminated[last_t]] = 0.0
        return next_values

    def update(self) -> None:  # noqa: PLR0915 - P1.05.11 added metric accumulation; refactor blocked by HARL.update tuple shape
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

        vectorised = np.ndim(self._buf_rewards[0]) == 1
        if vectorised:
            # P1.05.10 (ADR-007 §Stage 1b regime-alignment): batched
            # rollout. Stack to (T, N, …), run GAE independently per env
            # column through the SAME module-level compute_gae the
            # property test pins (USES_MODULE_GAE stays honest), then
            # flatten to (T*N, …) for the minibatch sweep.
            rewards_tn = np.stack(self._buf_rewards).astype(np.float32)  # (T, N)
            values_tn = np.stack(self._buf_values).astype(np.float32)
            terminated_tn = np.stack(self._buf_terminated).astype(np.bool_)
            truncated_tn = np.stack(self._buf_truncated).astype(np.bool_)
            bootstraps_tn = np.stack(self._buf_truncation_bootstraps).astype(np.float32)
            next_values_tn = self._build_next_values_vectorised(
                values_tn, terminated_tn, truncated_tn, bootstraps_tn
            )
            boundaries_tn = terminated_tn | truncated_tn
            n_envs = rewards_tn.shape[1]
            raw_adv_tn = np.stack(
                [
                    compute_gae(
                        rewards_tn[:, e],
                        values_tn[:, e],
                        next_values_tn[:, e],
                        boundaries_tn[:, e],
                        gamma=self._gamma,
                        gae_lambda=self._gae_lambda,
                    )
                    for e in range(n_envs)
                ],
                axis=1,
            )
            returns_tn = (raw_adv_tn + values_tn).astype(np.float32)
            rewards = rewards_tn.reshape(-1)
            values = values_tn.reshape(-1)
            raw_advantages = raw_adv_tn.reshape(-1)
            returns = returns_tn.reshape(-1)
            norm_advantages = normalize_advantages(raw_advantages)
            obs_arr = np.stack(self._buf_obs).astype(np.float32).reshape(-1, self._obs_dim)
            actions_arr = np.stack(self._buf_actions).astype(np.float32).reshape(-1, self._act_dim)
            log_probs_arr = (
                np.stack(self._buf_log_probs).astype(np.float32).reshape(obs_arr.shape[0], -1)
            )
        else:
            rewards = np.asarray(self._buf_rewards, dtype=np.float32)
            values = np.asarray(self._buf_values, dtype=np.float32)
            terminated = np.asarray(self._buf_terminated, dtype=np.bool_)
            truncated = np.asarray(self._buf_truncated, dtype=np.bool_)
            truncation_bootstraps = np.asarray(self._buf_truncation_bootstraps, dtype=np.float32)
            next_values = self._build_next_values(
                values, terminated, truncated, truncation_bootstraps
            )
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

        # P1.05.11 / ADR-017: per-update metric accumulators. Sums are
        # mean-reduced post-loop so a single ``log_scalars`` call
        # carries the rollout-aggregated PPO health snapshot. Captured
        # iff a logger is bound — short-circuits the accumulation work
        # otherwise so the silent-trainer path stays free.
        emit_metrics = self._logger is not None
        sum_policy_loss = 0.0
        sum_critic_loss = 0.0
        sum_dist_entropy = 0.0
        sum_grad_norm = 0.0
        sum_approx_kl = 0.0
        sum_clip_fraction = 0.0
        ratio_min_seen = float("inf")
        ratio_max_seen = float("-inf")
        n_updates = 0

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
                # P1.05.11 / ADR-017: capture HARL's 4-tuple return value.
                # Pre-P1.05.11 the trainer discarded it; commit 3 wires the
                # values through the bound logger. ``imp_weights`` is the
                # PPO importance ratio (exp of new-vs-old log-prob diff).
                policy_loss, dist_entropy, actor_grad_norm, imp_weights = self._happo.update(sample)

                # Critic update: simple MSE regression on returns.
                mb_returns = torch.from_numpy(returns[mb_idx]).to(self._device).float()
                mb_obs_t = torch.from_numpy(mb_obs).to(self._device).float()
                value_pred = self._critic(mb_obs_t).squeeze(-1)
                critic_loss = ((value_pred - mb_returns) ** 2).mean()
                self._critic_optim.zero_grad()
                critic_loss.backward()
                self._critic_optim.step()

                if emit_metrics:
                    # P1.05.11 / ADR-017: per-minibatch metric capture.
                    # Locally compute approx_kl and clip_fraction; HARL
                    # does not expose them. The torch.no_grad() guard
                    # is defensive — these are read-only reductions
                    # against detached tensors.
                    with torch.no_grad():
                        ratio = imp_weights.detach()
                        approx_kl = float(((ratio - 1.0) ** 2).mean().item()) * 0.5
                        clip_fraction = float(
                            ((ratio - 1.0).abs() > self._clip_param).float().mean().item()
                        )
                        r_min = float(ratio.min().item())
                        r_max = float(ratio.max().item())
                    sum_policy_loss += float(policy_loss.detach().item())
                    sum_critic_loss += float(critic_loss.detach().item())
                    sum_dist_entropy += float(dist_entropy.detach().item())
                    sum_grad_norm += float(
                        actor_grad_norm.detach().item()
                        if torch.is_tensor(actor_grad_norm)
                        else actor_grad_norm
                    )
                    sum_approx_kl += approx_kl
                    sum_clip_fraction += clip_fraction
                    ratio_min_seen = min(ratio_min_seen, r_min)
                    ratio_max_seen = max(ratio_max_seen, r_max)
                    n_updates += 1

        # P1.05.11 / ADR-017: emit one ``event="scalar"`` line per
        # ``update()`` call with the rollout-aggregated PPO health snapshot.
        # Skipped when no logger is bound (the existing silent path).
        if emit_metrics and self._logger is not None and n_updates > 0:
            # Lazy import: keeping the structlog helper out of module-top
            # imports preserves the trainer's Tier-1 importability story
            # (the trainer's TYPE_CHECKING block already gates structlog).
            from concerto.training.logging import log_scalars

            log_scalars(
                self._logger,
                step=self._global_step,
                namespace="train",
                policy_loss=sum_policy_loss / n_updates,
                value_loss=sum_critic_loss / n_updates,
                dist_entropy=sum_dist_entropy / n_updates,
                actor_grad_norm=sum_grad_norm / n_updates,
                approx_kl=sum_approx_kl / n_updates,
                clip_fraction=sum_clip_fraction / n_updates,
                ratio_min=ratio_min_seen,
                ratio_max=ratio_max_seen,
                advantage_mean=float(raw_advantages.mean()),
                advantage_std=float(raw_advantages.std()),
                value_mean=float(values.mean()),
                value_std=float(values.std()),
                learning_rate=self._learning_rate,
            )

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
# ``_assert_partner_is_frozen`` is intentionally private (leading underscore)
# — the public partner-freeze surface is the trainer constructor itself,
# which calls the helper. Tests import it directly for contract pinning
# (tests/unit/test_ego_ppo_trainer_rejects_non_frozen_partner.py); the
# helper's docstring + ADR-009 §Consequences citation are the documentation.
