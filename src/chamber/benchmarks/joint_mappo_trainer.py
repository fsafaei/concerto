# SPDX-License-Identifier: Apache-2.0
# pyright: reportPrivateImportUsage=false
#
# Same rationale as :mod:`chamber.benchmarks.ego_ppo_trainer`: torch's
# stubs do not export ``from_numpy`` / ``no_grad`` via ``__all__`` even
# though they are documented public API.
r"""B-JOINT: the jointly-trained MAPPO co-carry pair (ADR-011 as amended; ADR-018 §Consequences).

ADR-011 §Decision (as amended 2026-07-05) defines B-JOINT as *"a
jointly-trained MAPPO pair, evaluated as a pair — the upper anchor"*,
wired on the installed ``harl-aht`` MAPPO trainer
(``harl.algorithms.actors.mappo.MAPPO``; availability verified by
import inspection 2026-07-05). B-JOINT **trains outside the AHT
setting by construction** — both co-carry seats update — so this module
deliberately does NOT route through
:func:`concerto.training.ego_aht.train` (whose one-learner +
frozen-partner seams, including
``EgoPPOTrainer._assert_partner_is_frozen``, encode the AHT contract
this baseline is the anchor *against*). Its **evaluation** obeys the
ADR-018 black-box contract: checkpoints are frozen and driven through
:class:`chamber.partners.frozen_cocarry_joint.FrozenCoCarryJointPartner`
/ the co-carry eval driver, never trained further.

**Parameter sharing, recorded as implemented (ADR-011):** the pair is
trained with **per-agent parameters** — two independent
:class:`harl.algorithms.actors.mappo.MAPPO` actors (one per seat) plus
one shared central critic — the non-parameter-sharing MAPPO form. The
committed shared-parameter joint-PPO fallback clause was not needed.
``JointMAPPOConfig.param_sharing`` pins the literal ``"per_agent"`` so
the fact rides in every config dump and report.

Observation interfaces: the ego seat reads the synthesised 46-D
``state`` (:mod:`chamber.envs.cocarry_obs`); the partner seat reads the
symmetric full state
(:func:`chamber.partners.frozen_cocarry_joint.joint_partner_full_state`)
— the same function the frozen evaluation wrapper uses, so the
training and evaluation interfaces cannot drift. The central critic
reads the ego-view state (it already contains both seats + bar + goal).

Rollout/update math is reused from
:mod:`chamber.benchmarks.ego_ppo_trainer` (:func:`compute_gae`,
:func:`normalize_advantages`, the HARL args bridge, the critic MLP) —
one GAE implementation project-wide (ADR-002 §Decisions). MAPPO's
simultaneous update is realised by calling each actor's
:meth:`MAPPO.update` on the same shared normalised advantages within
each minibatch (no HAPPO sequential ``factor``).

Determinism (P6 / ADR-002): torch's global RNG is seeded once at
trainer construction via
:func:`concerto.training.seeding.derive_substream`; episode reseeds use
the ``training.joint_mappo.episode.{index}`` substream family
(label-disjoint from the ego-AHT loop's ``training.episode.*``).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import gymnasium as gym
import numpy as np
import torch
import yaml
from harl.algorithms.actors.mappo import MAPPO
from pydantic import BaseModel, ConfigDict, Field

from chamber.benchmarks.ego_ppo_trainer import (
    _build_harl_args,
    _EgoCritic,
    _flat_ego_obs,
    _resolve_device,
    compute_gae,
    normalize_advantages,
)
from chamber.partners.frozen_cocarry_joint import joint_partner_full_state
from concerto.training.checkpoints import CheckpointMetadata, save_checkpoint
from concerto.training.config import (
    EnvConfig,
    HAPPOHyperparams,
    RuntimeConfig,
    ShapingConfig,
)
from concerto.training.logging import compute_run_metadata
from concerto.training.seeding import derive_substream

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping

    from numpy.typing import NDArray

#: ``compute_run_metadata`` run-kind label; folded into the run_id hash
#: so B-JOINT runs never collide with same-seed ego-AHT runs (ADR-002).
JOINT_MAPPO_RUN_KIND: str = "joint_mappo"

#: Substream labels (P6 / ADR-002) — disjoint from the ego-AHT loop's
#: ``training.ego_ppo.*`` / ``training.episode.*`` families.
_TORCH_SUBSTREAM: str = "training.joint_mappo.torch"
_MINIBATCH_SUBSTREAM: str = "training.joint_mappo.minibatch"
_EPISODE_SUBSTREAM: str = "training.joint_mappo.episode.{index}"


def _to_scalar_float(value: Any) -> float:  # noqa: ANN401 - torch/np scalar
    """Coerce a torch / numpy scalar-or-(1,) value to a Python float (env 0)."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return float(np.asarray(value).reshape(-1)[0])


class JointMAPPOConfig(BaseModel):
    """B-JOINT training-run configuration (ADR-011 §Decision as amended; ADR-002 §Decisions).

    Deliberately mirrors :class:`concerto.training.config.EgoAHTConfig`'s
    shape (reusing its ``EnvConfig`` / ``HAPPOHyperparams`` /
    ``RuntimeConfig`` / ``ShapingConfig`` sub-models) minus the
    ``partner`` block — B-JOINT has no frozen partner; both seats learn.

    Attributes:
        algo: Pinned ``"joint_mappo"`` (the config-file discriminator).
        param_sharing: Pinned ``"per_agent"`` — the recorded
            parameter-sharing fact (ADR-011: two independent MAPPO
            actors + one shared central critic; see module docstring).
        seed: Root seed (P6 / ADR-002).
        total_frames: Training budget in env frames (single-env ticks).
        checkpoint_every: Checkpoint stride in frames; the CB-06
            checkpoint-selection rule (ADR-027 §Reporting rules)
            selects among the saved steps.
        artifacts_root: ``local://artifacts/...`` resolution root.
        env: Env block; ``task`` must be ``"cocarry"`` and ``num_envs``
            must be 1 (:func:`run_joint_training` loud-fails otherwise).
        mappo: PPO hyperparameters (the :class:`HAPPOHyperparams`
            fields map 1:1 onto MAPPO's knobs via the shared HARL args
            bridge).
        runtime: Device + determinism knobs.
        shaping: Reward-shaping block — B-JOINT trains under the same
            remediated co-carry reward as B-AHT (transport PBRS parity;
            ADR-026 §Decision 4) so the anchor comparison is
            reward-identical.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    algo: Literal["joint_mappo"] = "joint_mappo"
    param_sharing: Literal["per_agent"] = "per_agent"
    seed: int = 0
    total_frames: int = Field(gt=0)
    checkpoint_every: int = Field(gt=0)
    artifacts_root: Path = Path("./artifacts")
    env: EnvConfig
    mappo: HAPPOHyperparams
    runtime: RuntimeConfig = RuntimeConfig()
    shaping: ShapingConfig = ShapingConfig()


def load_joint_config(config_path: Path, overrides: list[str] | None = None) -> JointMAPPOConfig:
    """Load + validate a B-JOINT YAML config (ADR-011 §Decision as amended).

    Plain ``yaml.safe_load`` + Pydantic validation (no Hydra composition
    — the B-JOINT surface is one file with occasional smoke overrides).

    Args:
        config_path: The YAML file (e.g.
            ``configs/training/joint_mappo/cocarry_matched.yaml``).
        overrides: Optional ``dotted.key=value`` strings (values parsed
            with ``yaml.safe_load``), e.g. ``total_frames=2000`` /
            ``runtime.device=cpu`` for smoke runs.

    Returns:
        The validated frozen :class:`JointMAPPOConfig`.

    Raises:
        ValueError: On a malformed override string or non-mapping YAML.
    """
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        msg = f"{config_path}: expected a YAML mapping at top level"
        raise ValueError(msg)
    for item in overrides or []:
        key, sep, value = item.partition("=")
        if not sep or not key:
            msg = f"override {item!r} is not of the form dotted.key=value"
            raise ValueError(msg)
        node = raw
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                msg = f"override {item!r}: {part!r} is not a mapping in the config"
                raise ValueError(msg)
        node[parts[-1]] = yaml.safe_load(value)
    return JointMAPPOConfig.model_validate(raw)


class _SeatPending:
    """Per-step pre-step cache for both seats (mirrors ``_PendingStep``)."""

    def __init__(
        self,
        inputs: dict[str, NDArray[np.float32]],
        actions: dict[str, NDArray[np.float32]],
        log_probs: dict[str, NDArray[np.float32]],
        value: float,
    ) -> None:
        self.inputs = inputs
        self.actions = actions
        self.log_probs = log_probs
        self.value = value


class JointMAPPOTrainer:
    """Two-seat MAPPO trainer for the B-JOINT co-carry pair (ADR-011 §Decision as amended).

    Per-agent parameters: one :class:`harl.algorithms.actors.mappo.MAPPO`
    actor per seat + one shared central critic
    (:class:`~chamber.benchmarks.ego_ppo_trainer._EgoCritic` topology on
    the ego-view full state). Single-env only — the B-JOINT cell
    mirrors the B-AHT cell's regime (ADR-027 §Reporting rules demand
    comparable rows, not a differently-parallelised anchor).

    The rollout/update math reuses the project GAE + normalisation
    (:func:`chamber.benchmarks.ego_ppo_trainer.compute_gae` /
    :func:`~chamber.benchmarks.ego_ppo_trainer.normalize_advantages`;
    ADR-002 §Decisions: one GAE implementation project-wide).
    """

    def __init__(
        self,
        *,
        cfg: JointMAPPOConfig,
        ego_uid: str,
        partner_uid: str,
        state_dim: int,
        act_dims: dict[str, int],
        device: torch.device | None = None,
    ) -> None:
        """Build both actors + the central critic (ADR-011 §Decision as amended).

        Args:
            cfg: Validated :class:`JointMAPPOConfig`.
            ego_uid: The ego seat uid (``cfg.env.agent_uids[0]``).
            partner_uid: The partner seat uid (``cfg.env.agent_uids[1]``).
            state_dim: Width of the (symmetric) full-state vector both
                seats read (46 for the matched Panda pair).
            act_dims: Per-uid action dimensionality.
            device: Torch device; ``None`` → CPU.
        """
        self._device = device or torch.device("cpu")
        self._ego_uid = ego_uid
        self._partner_uid = partner_uid
        self._uids = (ego_uid, partner_uid)
        self._gamma = cfg.mappo.gamma
        self._gae_lambda = cfg.mappo.gae_lambda
        self._n_epochs = cfg.mappo.n_epochs
        self._batch_size = cfg.mappo.batch_size
        self._rollout_length = cfg.mappo.rollout_length
        torch_seed_int = int(
            derive_substream(_TORCH_SUBSTREAM, root_seed=cfg.seed)
            .default_rng()
            .integers(0, 2**31 - 1)
        )
        torch.manual_seed(torch_seed_int)
        self._minibatch_rng: np.random.Generator = derive_substream(
            _MINIBATCH_SUBSTREAM, root_seed=cfg.seed
        ).default_rng()

        harl_args = _build_harl_args(happo=cfg.mappo)
        self._harl_args = harl_args
        self._state_dim = state_dim
        self._act_dims = dict(act_dims)
        obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32)
        self._actors: dict[str, MAPPO] = {
            uid: MAPPO(
                harl_args,
                obs_space,
                gym.spaces.Box(low=-1.0, high=1.0, shape=(act_dims[uid],), dtype=np.float32),
                self._device,
            )
            for uid in self._uids
        }
        self._critic = _EgoCritic(obs_dim=state_dim, hidden_dim=cfg.mappo.hidden_dim).to(
            self._device
        )
        self._critic_optim = torch.optim.Adam(self._critic.parameters(), lr=cfg.mappo.lr)

        self._buf_inputs: dict[str, list[NDArray[np.float32]]] = {u: [] for u in self._uids}
        self._buf_actions: dict[str, list[NDArray[np.float32]]] = {u: [] for u in self._uids}
        self._buf_log_probs: dict[str, list[NDArray[np.float32]]] = {u: [] for u in self._uids}
        self._buf_values: list[float] = []
        self._buf_rewards: list[float] = []
        self._buf_terminated: list[bool] = []
        self._buf_truncated: list[bool] = []
        self._buf_truncation_bootstraps: list[float] = []
        self._pending: _SeatPending | None = None
        self._last_next_state: NDArray[np.float32] | None = None

    def _seat_inputs(self, obs: Mapping[str, Any]) -> dict[str, NDArray[np.float32]]:
        """Per-seat policy inputs: ego = synthesised state, partner = symmetric mirror."""
        ego_state = _flat_ego_obs(obs, self._ego_uid)
        partner_state = joint_partner_full_state(
            obs, own_uid=self._partner_uid, other_uid=self._ego_uid
        )
        return {self._ego_uid: ego_state, self._partner_uid: partner_state}

    def act(
        self, obs: Mapping[str, Any], *, deterministic: bool = False
    ) -> dict[str, NDArray[np.float32]]:
        """Sample both seats' actions; cache log-probs + central value (ADR-011; ADR-002).

        Args:
            obs: Env obs dict (raw leaves + synthesised ego ``state``).
            deterministic: When ``True``, both actors return the
                distribution mode (evaluation paths).

        Returns:
            ``{uid: action}`` ready for the env's dict-action step.
        """
        inputs = self._seat_inputs(obs)
        actions: dict[str, NDArray[np.float32]] = {}
        log_probs: dict[str, NDArray[np.float32]] = {}
        rnn_states = np.zeros(
            (1, self._harl_args["recurrent_n"], self._harl_args["hidden_sizes"][-1]),
            dtype=np.float32,
        )
        masks = np.ones((1, 1), dtype=np.float32)
        with torch.no_grad():
            for uid in self._uids:
                action_t, log_prob_t, _ = self._actors[uid].get_actions(
                    inputs[uid].reshape(1, -1), rnn_states, masks, None, deterministic
                )
                actions[uid] = action_t.detach().cpu().numpy().squeeze(0).astype(np.float32)
                log_probs[uid] = log_prob_t.detach().cpu().numpy().squeeze(0).astype(np.float32)
            value = float(
                self._critic(torch.from_numpy(inputs[self._ego_uid]).to(self._device).unsqueeze(0))
                .squeeze()
                .item()
            )
        self._pending = _SeatPending(inputs, actions, log_probs, value)
        return actions

    def observe(
        self,
        obs: Mapping[str, Any],
        reward: Any,  # noqa: ANN401 - scalar float / torch scalar
        done: Any,  # noqa: ANN401 - scalar bool / torch scalar
        *,
        truncated: Any = False,  # noqa: ANN401 - scalar bool / torch scalar
    ) -> None:
        """Buffer one shared-reward step for both seats (ADR-002 §Decisions; Pardo 2017).

        Mirrors :meth:`chamber.benchmarks.ego_ppo_trainer.EgoPPOTrainer.observe`'s
        single-env body: termination vs truncation are stored
        separately, and ``V(s_truncated_final)`` is captured eagerly at
        the boundary for the Pardo bootstrap (issue #62).
        """
        pending = self._pending
        if pending is None:
            return
        done_b = bool(_to_scalar_float(done))
        trunc_b = bool(_to_scalar_float(truncated))
        terminated = done_b and not trunc_b
        for uid in self._uids:
            self._buf_inputs[uid].append(pending.inputs[uid])
            self._buf_actions[uid].append(pending.actions[uid])
            self._buf_log_probs[uid].append(pending.log_probs[uid])
        self._buf_values.append(pending.value)
        self._buf_rewards.append(_to_scalar_float(reward))
        self._buf_terminated.append(terminated)
        self._buf_truncated.append(trunc_b)
        next_state = self._seat_inputs(obs)[self._ego_uid]
        if trunc_b and not terminated:
            with torch.no_grad():
                v_trunc = float(
                    self._critic(torch.from_numpy(next_state).to(self._device).unsqueeze(0))
                    .squeeze()
                    .item()
                )
            self._buf_truncation_bootstraps.append(v_trunc)
        else:
            self._buf_truncation_bootstraps.append(0.0)
        self._last_next_state = next_state
        self._pending = None

    def update(self) -> None:  # noqa: PLR0915 - the GAE + two-actor + critic sweep is one atomic rollout consumption (mirrors EgoPPOTrainer.update)
        """GAE on the shared reward; simultaneous per-agent MAPPO updates + critic MSE.

        MAPPO semantics (ADR-011 §Decision as amended): both actors
        update on the same shared normalised advantages within each
        minibatch — simultaneous, no HAPPO sequential ``factor``. The
        central critic regresses on the shared returns.
        """
        if not self._buf_rewards:
            return
        rewards = np.asarray(self._buf_rewards, dtype=np.float32)
        values = np.asarray(self._buf_values, dtype=np.float32)
        terminated = np.asarray(self._buf_terminated, dtype=np.bool_)
        truncated = np.asarray(self._buf_truncated, dtype=np.bool_)
        bootstraps = np.asarray(self._buf_truncation_bootstraps, dtype=np.float32)
        n_steps = rewards.shape[0]
        next_values = np.zeros(n_steps, dtype=np.float32)
        for t in range(n_steps - 1):
            if terminated[t]:
                next_values[t] = 0.0
            elif truncated[t]:
                next_values[t] = bootstraps[t]
            else:
                next_values[t] = values[t + 1]
        last_t = n_steps - 1
        if terminated[last_t]:
            next_values[last_t] = 0.0
        elif truncated[last_t]:
            next_values[last_t] = bootstraps[last_t]
        elif self._last_next_state is not None:
            with torch.no_grad():
                next_values[last_t] = float(
                    self._critic(
                        torch.from_numpy(self._last_next_state).to(self._device).unsqueeze(0)
                    )
                    .squeeze()
                    .item()
                )
        boundaries = terminated | truncated
        raw_advantages = compute_gae(
            rewards,
            values,
            next_values,
            boundaries,
            gamma=self._gamma,
            gae_lambda=self._gae_lambda,
        )
        returns = (raw_advantages + values).astype(np.float32)
        norm_advantages = normalize_advantages(raw_advantages)

        inputs_arr = {uid: np.stack(self._buf_inputs[uid]).astype(np.float32) for uid in self._uids}
        actions_arr = {
            uid: np.stack(self._buf_actions[uid]).astype(np.float32) for uid in self._uids
        }
        log_probs_arr = {
            uid: np.stack(self._buf_log_probs[uid]).astype(np.float32) for uid in self._uids
        }

        n_minibatches = max(1, n_steps // self._batch_size)
        mb_size = max(1, n_steps // n_minibatches)
        for _epoch in range(self._n_epochs):
            perm = self._minibatch_rng.permutation(n_steps)
            for mb in range(n_minibatches):
                start = mb * mb_size
                end = start + mb_size if mb < n_minibatches - 1 else n_steps
                mb_idx = perm[start:end]
                this_mb_size = mb_idx.shape[0]
                mb_advantages = norm_advantages[mb_idx].reshape(-1, 1).astype(np.float32)
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
                for uid in self._uids:
                    # MAPPO's 8-tuple sample layout (no HAPPO ``factor``):
                    # simultaneous updates on the shared advantages.
                    sample = (
                        inputs_arr[uid][mb_idx],
                        mb_rnn_states,
                        actions_arr[uid][mb_idx],
                        mb_masks,
                        mb_active_masks,
                        log_probs_arr[uid][mb_idx],
                        mb_advantages,
                        None,
                    )
                    self._actors[uid].update(sample)
                mb_returns = torch.from_numpy(returns[mb_idx]).to(self._device).float()
                mb_obs_t = (
                    torch.from_numpy(inputs_arr[self._ego_uid][mb_idx]).to(self._device).float()
                )
                value_pred = self._critic(mb_obs_t).squeeze(-1)
                critic_loss = ((value_pred - mb_returns) ** 2).mean()
                self._critic_optim.zero_grad()
                critic_loss.backward()
                self._critic_optim.step()

        for uid in self._uids:
            self._buf_inputs[uid].clear()
            self._buf_actions[uid].clear()
            self._buf_log_probs[uid].clear()
        self._buf_values.clear()
        self._buf_rewards.clear()
        self._buf_terminated.clear()
        self._buf_truncated.clear()
        self._buf_truncation_bootstraps.clear()
        self._last_next_state = None

    def state_dict(self) -> dict[str, Any]:
        """The pair-checkpoint payload (ADR-011 §Decision as amended; ADR-002 §Decisions).

        Layout consumed by
        :class:`chamber.partners.frozen_cocarry_joint.FrozenCoCarryJointPartner`
        (``actor_ego`` / ``actor_partner`` sub-keys) and the co-carry
        eval driver; optimizer + critic states ship for resume parity
        with :meth:`EgoPPOTrainer.state_dict`.
        """
        ego, partner = self._ego_uid, self._partner_uid
        return {
            "actor_ego": self._actors[ego].actor.state_dict(),
            "actor_ego_optim": self._actors[ego].actor_optimizer.state_dict(),
            "actor_partner": self._actors[partner].actor.state_dict(),
            "actor_partner_optim": self._actors[partner].actor_optimizer.state_dict(),
            "critic": self._critic.state_dict(),
            "critic_optim": self._critic_optim.state_dict(),
        }


class JointTrainingResult:
    """Outcome of one B-JOINT training run (ADR-011 §Decision as amended).

    Attributes:
        run_id: The :func:`concerto.training.logging.compute_run_metadata`
            16-hex run id (config fingerprint folded in; issue #214).
        checkpoint_paths: Absolute paths of the saved pair checkpoints,
            in step order (the CB-06 checkpoint-selection inputs).
        per_episode_rewards: Shared episode returns, diagnostic only.
        trainer: The live trainer (in-memory final policy).
    """

    def __init__(
        self,
        *,
        run_id: str,
        checkpoint_paths: list[Path],
        per_episode_rewards: list[float],
        trainer: JointMAPPOTrainer,
    ) -> None:
        """Bind the run's provenance + artefacts (ADR-011 §Decision as amended)."""
        self.run_id = run_id
        self.checkpoint_paths = checkpoint_paths
        self.per_episode_rewards = per_episode_rewards
        self.trainer = trainer


def _config_fingerprint(cfg: JointMAPPOConfig) -> str:
    """Stable config hash folded into the run_id (issue #214 convention).

    Excludes ``artifacts_root`` (an operator-side sink, not run
    semantics) — mirrors :func:`concerto.training.ego_aht.train`'s
    fingerprint exclusions.
    """
    dumped = cfg.model_dump(mode="json", exclude={"artifacts_root"})
    return hashlib.sha256(repr(sorted(dumped.items())).encode("utf-8")).hexdigest()


def run_joint_training(  # noqa: PLR0915 - env build + probe + loop + checkpoint flush is one linear run script
    cfg: JointMAPPOConfig, *, repo_root: Path | None = None
) -> JointTrainingResult:
    """Build the co-carry env and train the B-JOINT pair (ADR-011 §Decision as amended).

    Single-env loop mirroring :func:`concerto.training.ego_aht.train`'s
    scalar path: first reset at ``cfg.seed``, per-episode reseeds from
    the ``training.joint_mappo.episode.{index}`` substream, updates
    every ``mappo.rollout_length`` frames, checkpoints every
    ``checkpoint_every`` frames plus a terminal flush (the issue-#215
    rider W1 convention). The reward is the remediated co-carry reward
    plus the same transport PBRS the B-AHT cell trains under
    (ADR-026 §Decision 4) — anchor comparability is reward-identical.

    Args:
        cfg: Validated :class:`JointMAPPOConfig`.
        repo_root: Working-tree root for run-metadata provenance;
            ``None`` → ``Path.cwd()``.

    Returns:
        The :class:`JointTrainingResult`.

    Raises:
        ValueError: When ``cfg.env.task != "cocarry"`` or
            ``cfg.env.num_envs != 1`` (the B-JOINT cell regime).
    """
    if cfg.env.task != "cocarry":
        msg = (
            f"run_joint_training: B-JOINT v1 is wired for task='cocarry' only "
            f"(ADR-011 §Decision as amended); got {cfg.env.task!r}."
        )
        raise ValueError(msg)
    if cfg.env.num_envs != 1:
        msg = (
            "run_joint_training: the B-JOINT cell is single-env (regime parity "
            f"with the B-AHT cell; ADR-027 §Reporting rules); got num_envs="
            f"{cfg.env.num_envs}."
        )
        raise ValueError(msg)
    from chamber.benchmarks.training_runner import build_env

    env = build_env(cfg.env, root_seed=cfg.seed)
    if cfg.shaping.transport_pbrs_coeff > 0.0:
        # Same wrapper + gamma convention as run_training (ADR-026 §D4).
        from chamber.envs.cocarry_shaping import CoCarryTransportPBRSWrapper

        env = cast(
            "Any",
            CoCarryTransportPBRSWrapper(
                cast("gym.Env[Any, Any]", env),
                coeff=cfg.shaping.transport_pbrs_coeff,
                gamma=cfg.mappo.gamma,
            ),
        )
    ctx = compute_run_metadata(
        seed=cfg.seed,
        run_kind=JOINT_MAPPO_RUN_KIND,
        repo_root=repo_root if repo_root is not None else Path.cwd(),
        config_fingerprint=_config_fingerprint(cfg),
    )
    ego_uid, partner_uid = cfg.env.agent_uids
    device = _resolve_device(cfg.runtime.device)

    obs, _ = env.reset(seed=cfg.seed)
    probe_inputs = {
        ego_uid: _flat_ego_obs(obs, ego_uid),
        partner_uid: joint_partner_full_state(obs, own_uid=partner_uid, other_uid=ego_uid),
    }
    state_dim = int(probe_inputs[ego_uid].shape[0])
    if probe_inputs[partner_uid].shape[0] != state_dim:
        msg = (
            f"run_joint_training: asymmetric seat state widths "
            f"(ego {state_dim}, partner {probe_inputs[partner_uid].shape[0]}) — "
            "B-JOINT v1 assumes the matched Panda pair (ADR-011 as amended)."
        )
        raise ValueError(msg)
    act_dims = {
        uid: int(env.action_space[uid].shape[0])  # type: ignore[attr-defined,index]
        for uid in (ego_uid, partner_uid)
    }
    trainer = JointMAPPOTrainer(
        cfg=cfg,
        ego_uid=ego_uid,
        partner_uid=partner_uid,
        state_dim=state_dim,
        act_dims=act_dims,
        device=device,
    )

    checkpoint_paths: list[Path] = []
    per_episode_rewards: list[float] = []
    episode_acc = 0.0
    episode_index = 0
    last_ckpt_frames = 0
    frames_done = 0
    for step in range(cfg.total_frames):
        actions = trainer.act(obs)
        obs, reward, terminated_v, truncated_v, _info = env.step(actions)
        terminated = bool(_to_scalar_float(terminated_v))
        truncated = bool(_to_scalar_float(truncated_v))
        done = terminated or truncated
        trainer.observe(obs, reward, done, truncated=truncated)
        episode_acc += _to_scalar_float(reward)
        frames_done = step + 1
        if done:
            per_episode_rewards.append(episode_acc)
            episode_acc = 0.0
            episode_index += 1
            episode_seed_int = int(
                derive_substream(_EPISODE_SUBSTREAM.format(index=episode_index), root_seed=cfg.seed)
                .default_rng()
                .integers(0, 2**31 - 1)
            )
            obs, _ = env.reset(seed=episode_seed_int)
        if frames_done % cfg.mappo.rollout_length == 0:
            trainer.update()
        if frames_done % cfg.checkpoint_every == 0:
            last_ckpt_frames = frames_done
            checkpoint_paths.append(
                _save_pair_checkpoint(
                    cfg=cfg,
                    ctx_run_id=ctx.run_id,
                    step=frames_done,
                    git_sha=ctx.git_sha,
                    pyproject_hash=ctx.pyproject_hash,
                    state_dict=trainer.state_dict(),
                )
            )
    if frames_done > 0 and last_ckpt_frames != frames_done:
        # Terminal flush (the issue-#215 rider W1 convention).
        checkpoint_paths.append(
            _save_pair_checkpoint(
                cfg=cfg,
                ctx_run_id=ctx.run_id,
                step=frames_done,
                git_sha=ctx.git_sha,
                pyproject_hash=ctx.pyproject_hash,
                state_dict=trainer.state_dict(),
            )
        )
    env.close()  # type: ignore[attr-defined]
    return JointTrainingResult(
        run_id=ctx.run_id,
        checkpoint_paths=checkpoint_paths,
        per_episode_rewards=per_episode_rewards,
        trainer=trainer,
    )


def _save_pair_checkpoint(
    *,
    cfg: JointMAPPOConfig,
    ctx_run_id: str,
    step: int,
    git_sha: str,
    pyproject_hash: str,
    state_dict: dict[str, Any],
) -> Path:
    """Write one pair checkpoint .pt + sidecar (ADR-002 §Decisions payload+sidecar contract)."""
    uri = f"local://artifacts/{ctx_run_id}_step{step}.pt"
    metadata = CheckpointMetadata(
        run_id=ctx_run_id,
        seed=cfg.seed,
        step=step,
        git_sha=git_sha,
        pyproject_hash=pyproject_hash,
        sha256="",  # save_checkpoint recomputes from the .pt bytes.
    )
    return save_checkpoint(
        state_dict=state_dict,
        uri=uri,
        metadata=metadata,
        artifacts_root=cfg.artifacts_root,
    )


__all__ = [
    "JOINT_MAPPO_RUN_KIND",
    "JointMAPPOConfig",
    "JointMAPPOTrainer",
    "JointTrainingResult",
    "load_joint_config",
    "run_joint_training",
]
