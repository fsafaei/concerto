# SPDX-License-Identifier: Apache-2.0
r"""Frozen Rung-2 co-carry incumbent: deterministic load + matched eval (ADR-026 §Decision 4).

The Rung-2 incumbent is a learned Panda ego, trained against the frozen
matched :class:`chamber.partners.cocarry_impedance.CoCarryImpedancePartner`
and then **frozen** (R-2026-06-B §15). This module is the deterministic
load-and-act surface Rungs 3-4 hold fixed while swapping teammates against
it:

- :func:`load_frozen_incumbent` — rebuild the
  :class:`chamber.benchmarks.ego_ppo_trainer.EgoPPOTrainer` from the
  training config + the eval env's spaces, restore the checkpointed actor
  + critic, and return a deterministic per-step ``act`` closure
  (``trainer.act(obs, deterministic=True)`` returns the policy **mode** —
  no RNG — so two loads of the same checkpoint emit byte-identical actions
  on identical obs, on any device; the §3.4 deterministic-reload contract).
- :func:`build_matched_eval` — build the synthesizer-wrapped matched
  co-carry env + the frozen partner-seat impedance controller, the pair the
  incumbent is evaluated against.
- :func:`evaluate_incumbent_matched` — roll the frozen incumbent (ego)
  against the matched frozen partner over a seed sweep and return the
  per-seed :class:`chamber.benchmarks.cocarry_runner.EpisodeMetrics`, so
  the incumbent's joint-success rate can be compared to the Rung-1 matched
  reference (the Step-1 stop criterion: cannot reach it ⇒ STOP).

The incumbent's policy reads ``obs["agent"][ego_uid]["state"]`` (synthesised
by :class:`chamber.envs.cocarry_obs.CoCarryEgoStateSynthesizer`); the frozen
partner reads the raw ``obs["agent"][partner_uid]["qpos"]`` +
``obs["extra"]["goal_pos"]`` the synthesizer leaves untouched (ADR-009
§Decision). This module never edits :mod:`chamber.benchmarks.cocarry_runner`
(the Rung-0/1 surface stays as shipped in PR #245); it reuses its
:class:`EpisodeMetrics` + :func:`summarize`.

References:
- ADR-026 §Decision 4 (the Phase-2 forward-design ladder; the frozen
  incumbent is the fixed point Rungs 3-4 measure against).
- ADR-002 §Decisions (the checkpoint payload+sidecar SHA-256 load contract;
  ``trainer.act(deterministic=True)`` is the policy mode).
- ADR-009 §Decision (frozen black-box partner; partner reads raw leaves).
- R-2026-06-B §15 Rung 2 (the train-to-reference + freeze design).
- :mod:`chamber.benchmarks.cocarry_runner`, :mod:`chamber.envs.cocarry_obs`,
  :class:`chamber.benchmarks.ego_ppo_trainer.EgoPPOTrainer`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from chamber.benchmarks.cocarry_runner import EpisodeMetrics, _to_float, build_matched_controllers
from chamber.envs.cocarry import COCARRY_DEFAULT_EPISODE_LENGTH
from chamber.envs.cocarry_obs import make_cocarry_training_env

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Mapping
    from pathlib import Path

    import gymnasium as gym
    from numpy.typing import NDArray

    from chamber.partners.api import FrozenPartner
    from concerto.training.config import EgoAHTConfig


def slope_report_from_curve(
    curve: Any,  # noqa: ANN401 - concerto.training.ego_aht.RewardCurve (avoids a top-level concerto import)
    *,
    alpha: float = 0.05,
    min_episodes: int = 20,
) -> Any:  # noqa: ANN401 - concerto.training.learning_signal_check.SlopeReport
    """Run the ADR-002 learning-signal slope check on a co-carry reward curve (R-2026-06-B §15).

    The single-env ManiSkill co-carry cell (``num_envs=1``) returns the
    per-step ``reward`` as a ``(1,)``-shaped tensor, so
    :class:`~concerto.training.ego_aht.RewardCurve`'s
    ``per_episode_ego_rewards`` holds ``(1,)`` arrays rather than Python
    floats. Fed straight to
    :func:`concerto.training.learning_signal_check.check_positive_learning_slope`,
    that makes the internal ``rewards`` array 2-D and trips
    ``scipy.stats.linregress`` ("all x values are identical"). This helper
    coerces every per-episode (and per-step) reward to a scalar float —
    lossless for a single-env cell — then runs the canonical alpha=0.05
    one-sided slope check (ADR-002 §Risks #1). It does **not** alter the
    canonical training loop (concerto core stays untouched).

    Args:
        curve: The :class:`~concerto.training.ego_aht.RewardCurve` from
            :func:`chamber.benchmarks.training_runner.run_training`.
        alpha: One-sided significance threshold (ADR-002 default 0.05).
        min_episodes: Minimum episodes for a non-vacuous verdict (default 20).

    Returns:
        The :class:`~concerto.training.learning_signal_check.SlopeReport`.
    """
    from concerto.training.ego_aht import RewardCurve
    from concerto.training.learning_signal_check import check_positive_learning_slope

    def _scalar(x: Any) -> float:  # noqa: ANN401 - float / ndarray / torch scalar
        return float(np.asarray(x).reshape(-1)[0])

    clean = RewardCurve(
        run_id=curve.run_id,
        per_step_ego_rewards=[_scalar(s) for s in curve.per_step_ego_rewards],
        per_episode_ego_rewards=[_scalar(r) for r in curve.per_episode_ego_rewards],
        checkpoint_paths=list(curve.checkpoint_paths),
    )
    return check_positive_learning_slope(clean, alpha=alpha, min_episodes=min_episodes)


def load_frozen_incumbent(
    *,
    cfg: EgoAHTConfig,
    env: gym.Env[Any, Any],
    partner: FrozenPartner,
    checkpoint_uri: str,
    artifacts_root: Path,
    ego_uid: str | None = None,
) -> Callable[[Mapping[str, Any]], NDArray[np.float32]]:
    """Load the frozen incumbent into a deterministic per-step ``act`` closure (ADR-026 §D4).

    Rebuilds the :class:`chamber.benchmarks.ego_ppo_trainer.EgoPPOTrainer`
    from ``cfg`` + the eval ``env``'s observation/action spaces (so the
    actor/critic widths cannot drift from what the env emits), restores the
    checkpointed actor + critic via
    :func:`concerto.training.checkpoints.load_checkpoint` (SHA-256-verified),
    and returns ``lambda obs: trainer.act(obs, deterministic=True)``. The
    deterministic mode carries no RNG, so two calls to this function on the
    same checkpoint produce byte-identical actions on identical obs
    (the §3.4 deterministic-reload contract).

    Args:
        cfg: The training :class:`~concerto.training.config.EgoAHTConfig`
            (its ``happo`` widths + ``runtime.device`` rebuild the trainer).
        env: The synthesizer-wrapped eval env (must expose the ego
            ``state`` Box). Built by :func:`build_matched_eval`.
        partner: A frozen partner instance — passed to
            :meth:`EgoPPOTrainer.from_config` so its partner-freeze gate
            (``_assert_partner_is_frozen``; ADR-009 §Consequences) runs.
        checkpoint_uri: ``local://...`` checkpoint URI (the incumbent
            ``.pt`` produced by the training run).
        artifacts_root: Filesystem root the URI resolves against.
        ego_uid: Ego uid; defaults to ``cfg.env.agent_uids[0]``.

    Returns:
        A deterministic ``obs -> action`` closure for the frozen incumbent.
    """
    from chamber.benchmarks.ego_ppo_trainer import EgoPPOTrainer
    from concerto.training.checkpoints import load_checkpoint

    resolved_ego = ego_uid if ego_uid is not None else cfg.env.agent_uids[0]
    # from_config takes the structural EnvLike Protocol; the synthesizer-
    # wrapped gym.Env satisfies it at runtime (reset/step signatures), so
    # cast for pyright (mirrors training_runner.build_env's EnvLike cast).
    trainer = EgoPPOTrainer.from_config(
        cfg, env=cast("Any", env), partner=partner, ego_uid=resolved_ego
    )
    state_dict, _ = load_checkpoint(uri=checkpoint_uri, artifacts_root=artifacts_root)
    # Restore is per-component (EgoPPOTrainer.state_dict() is a flat 4-key
    # dict of nested sub-state-dicts; there is no monolithic load). Only the
    # actor matters for the deterministic policy, but the critic is restored
    # too so a resumed trainer is faithful (ADR-002 §Decisions; mirrors
    # tests/unit/test_ego_ppo_trainer.py round-trip). load_checkpoint's
    # return is annotated dict[str, torch.Tensor]; the values are actually
    # the nested module sub-state-dicts — cast for the load_state_dict call.
    sd = cast("dict[str, Any]", state_dict)
    trainer._happo.actor.load_state_dict(sd["actor"])
    trainer._critic.load_state_dict(sd["critic"])

    def _act(obs: Mapping[str, Any]) -> NDArray[np.float32]:
        """Deterministic frozen-incumbent action (policy mode; ADR-026 §Decision 4)."""
        return np.asarray(trainer.act(obs, deterministic=True), dtype=np.float32)

    return _act


def build_matched_eval(
    *,
    root_seed: int,
    episode_length: int = COCARRY_DEFAULT_EPISODE_LENGTH,
    render_backend: str | None = None,
) -> tuple[gym.Env[Any, Any], FrozenPartner]:
    """Build the matched eval env + the frozen partner-seat controller (ADR-026 §Decision 1-2).

    The env is the synthesizer-wrapped matched co-carry env (so the ego
    ``state`` channel is present for the incumbent); the partner is the
    matched :class:`CoCarryImpedancePartner` on the partner seat
    (``panda_partner``, the bar's -x end), built from the env's
    single-source-of-truth geometry via
    :func:`chamber.benchmarks.cocarry_runner.build_matched_controllers`.

    Args:
        root_seed: Episode seed (P6 / ADR-002).
        episode_length: Truncation horizon.
        render_backend: Optional ManiSkill render backend (``"none"`` on
            headless hosts).

    Returns:
        ``(env, partner)`` — the eval env and the frozen partner-seat
        controller. The caller is responsible for ``env.close()``.
    """
    env = make_cocarry_training_env(
        condition_id="cocarry_matched_panda_pair",
        episode_length=episode_length,
        root_seed=root_seed,
        render_backend=render_backend,
    )
    partner = build_matched_controllers()[env.get_wrapper_attr("partner_uid")]
    return env, partner


def rollout_incumbent_episode(
    *,
    ego_act: Callable[[Mapping[str, Any]], NDArray[np.float32]],
    env: gym.Env[Any, Any],
    partner: FrozenPartner,
    seed: int,
    episode_length: int,
) -> EpisodeMetrics:
    """Roll one matched episode with ego = frozen incumbent (ADR-026 §Decision 1; R-2026-06-B §15).

    Mirrors :func:`chamber.benchmarks.cocarry_runner.rollout_episode`'s
    matched branch but drives the ego seat with the incumbent's ``ego_act``
    closure instead of the impedance controller; the partner seat keeps the
    frozen impedance controller (ADR-009 §Decision).

    Args:
        ego_act: The frozen incumbent's deterministic ``obs -> action``
            closure (from :func:`load_frozen_incumbent`).
        env: The matched eval env (from :func:`build_matched_eval`).
        partner: The frozen partner-seat controller.
        seed: Episode seed (P6).
        episode_length: Truncation horizon.

    Returns:
        The per-episode :class:`EpisodeMetrics` (joint success + binding
        telemetry maxima).
    """
    ego_uid = env.get_wrapper_attr("ego_uid")
    partner_uid = env.get_wrapper_attr("partner_uid")
    obs, _ = env.reset(seed=seed)
    partner.reset(seed=seed)
    info: dict[str, Any] = {}
    for _ in range(episode_length):
        action = {ego_uid: ego_act(obs), partner_uid: partner.act(obs)}
        obs, _, terminated, truncated, info = env.step(action)
        if bool(np.asarray(terminated).reshape(-1)[0]) or bool(
            np.asarray(truncated).reshape(-1)[0]
        ):
            break
    tel = env.get_wrapper_attr("get_telemetry")()
    return EpisodeMetrics(
        seed=seed,
        success=bool(np.asarray(info["success"]).reshape(-1)[0]),
        centroid_to_goal=_to_float(tel["centroid_to_goal"]),
        max_tilt_deg=_to_float(tel["max_tilt_deg"]),
        max_stress_proxy=_to_float(tel["max_stress_proxy"]),
        n_steps=episode_length,
    )


def evaluate_incumbent_matched(
    *,
    cfg: EgoAHTConfig,
    checkpoint_uri: str,
    artifacts_root: Path,
    seeds: list[int],
    episode_length: int = COCARRY_DEFAULT_EPISODE_LENGTH,
    render_backend: str | None = None,
) -> list[EpisodeMetrics]:
    """Evaluate the frozen incumbent + matched partner over a seed sweep (ADR-026 §Decision 4).

    For each seed: build the matched eval env + frozen partner, load the
    frozen incumbent into the ego seat, roll one episode, and collect the
    :class:`EpisodeMetrics`. Aggregate with
    :func:`chamber.benchmarks.cocarry_runner.summarize` and compare the
    success rate to the Rung-1 matched reference (the Step-1 stop
    criterion). A fresh env per seed mirrors the Rung-0/1 runner.

    Args:
        cfg: The training config (rebuilds the trainer / spaces).
        checkpoint_uri: The incumbent ``.pt`` URI.
        artifacts_root: Filesystem root the URI resolves against.
        seeds: Episode seeds (P6).
        episode_length: Truncation horizon.
        render_backend: Optional ManiSkill render backend.

    Returns:
        Per-seed :class:`EpisodeMetrics`.
    """
    metrics: list[EpisodeMetrics] = []
    for s in seeds:
        env, partner = build_matched_eval(
            root_seed=s, episode_length=episode_length, render_backend=render_backend
        )
        try:
            ego_act = load_frozen_incumbent(
                cfg=cfg,
                env=env,
                partner=partner,
                checkpoint_uri=checkpoint_uri,
                artifacts_root=artifacts_root,
            )
            metrics.append(
                rollout_incumbent_episode(
                    ego_act=ego_act,
                    env=env,
                    partner=partner,
                    seed=s,
                    episode_length=episode_length,
                )
            )
        finally:
            env.close()
    return metrics


__all__ = [
    "build_matched_eval",
    "evaluate_incumbent_matched",
    "load_frozen_incumbent",
    "rollout_incumbent_episode",
    "slope_report_from_curve",
]
