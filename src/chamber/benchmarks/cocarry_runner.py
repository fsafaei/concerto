# SPDX-License-Identifier: Apache-2.0
r"""Co-carry Rung-0/1 rollout + measurement harness (ADR-026 §Decision 1-2; R-2026-06-B Rungs 0-1).

Thin, SAPIEN-gated driver shared by the Rung-0 stability smoke
(``scripts/repro/cocarry_rung0_stability.sh``), the Rung-1 matched
competence + coupling positive-control
(``scripts/repro/cocarry_rung1_matched.sh``), and their Tier-2 tests.
Keeps the rollout + aggregation logic in one place so the scripts and
tests do not drift.

What it provides:

- :func:`build_matched_controllers` — the two
  :class:`chamber.partners.cocarry_impedance.CoCarryImpedancePartner`
  instances, built from the env's single-source-of-truth geometry
  (:func:`chamber.envs.cocarry.cocarry_matched_controller_specs`).
- :func:`rollout_hold` — Rung-0: zero-action hold; asserts the dual-hold
  attach holds the bar (telemetry finite, tilt + wrist stress bounded).
- :func:`rollout_episode` — Rung-1: drive the held arm(s) with the matched
  controller(s) and return the per-episode :class:`EpisodeMetrics` (joint
  success + the binding telemetry maxima).
- :func:`evaluate_condition` — run a condition across a seed list and
  return the per-seed metrics (the matched competence rate / single-arm
  ~=0 positive-control / the constraint-force distribution feeding the
  ``f_max`` derivation).

Governance: this module builds nothing on the Phase-1 gate path and reuses
the existing partner interface (ADR-009) + env (ADR-026). Single-env
(num_envs=1) only — the Rung-0/1 regime.

References:
- ADR-026 §Decision 1-2 (coupling-valid task + falsifiable positive-control).
- ADR-009 §Decision (partner interface the matched controllers route through).
- R-2026-06-B Rungs 0-1 (stability gate, matched competence, geometric +
  empirical single-arm infeasibility, constraint-binding, ``f_max``).
- :mod:`chamber.envs.cocarry`, :mod:`chamber.partners.cocarry_impedance`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from chamber.envs.cocarry import (
    COCARRY_DEFAULT_EPISODE_LENGTH,
    COCARRY_STRESS_MAX_PROXY_N,
    COCARRY_TILT_MAX_DEG,
    cocarry_matched_controller_specs,
    make_cocarry_env,
)
from chamber.partners.api import PartnerSpec
from chamber.partners.registry import load_partner

if TYPE_CHECKING:  # pragma: no cover
    import gymnasium as gym

    from chamber.partners.api import FrozenPartner

# Import for the @register_partner side effect so "cocarry_impedance" is
# resolvable when this module is the entry point (scripts / tests).
import chamber.partners.cocarry_impedance  # noqa: F401


def _to_float(value: Any) -> float:  # noqa: ANN401 - torch/np scalar
    """Coerce a torch / numpy scalar-or-(1,) telemetry value to a Python float (env 0)."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return float(np.asarray(value).reshape(-1)[0])


@dataclass(frozen=True)
class EpisodeMetrics:
    """Per-episode co-carry outcome + binding telemetry (ADR-026 §Decision 1-2).

    Attributes:
        seed: The episode seed.
        success: The joint co-carry success (placed + level + unstressed +
            both-static + settled).
        centroid_to_goal: Final bar-centroid-to-goal distance, metres.
        max_tilt_deg: Episode-max bar tilt (post-settle), degrees.
        max_stress_proxy: Episode-max wrist constraint-solver force proxy
            (post-settle), Newtons.
        n_steps: Steps actually rolled.
    """

    seed: int
    success: bool
    centroid_to_goal: float
    max_tilt_deg: float
    max_stress_proxy: float
    n_steps: int


@dataclass(frozen=True)
class HoldMetrics:
    """Rung-0 zero-action hold stability summary (ADR-026 §Decision 1; R-2026-06-B stability gate).

    Attributes:
        seed: The episode seed.
        finite: Whether all telemetry stayed finite (no solver blow-up).
        max_tilt_deg: Max bar tilt over the hold, degrees.
        max_stress_proxy: Max wrist constraint-solver force proxy, Newtons.
    """

    seed: int
    finite: bool
    max_tilt_deg: float
    max_stress_proxy: float


def build_matched_controllers() -> dict[str, FrozenPartner]:
    """Build the two matched co-carry controllers (ADR-026 §Decision 1; ADR-009 §Decision).

    Returns:
        ``{uid: FrozenPartner}`` for the ego + partner uids, each a
        :class:`chamber.partners.cocarry_impedance.CoCarryImpedancePartner`
        built from the env's geometry via
        :func:`chamber.envs.cocarry.cocarry_matched_controller_specs`.
    """
    specs = cocarry_matched_controller_specs()
    return {
        uid: load_partner(PartnerSpec("cocarry_impedance", 0, None, None, extra))
        for uid, extra in specs.items()
    }


def _zero_actions(env: gym.Env[Any, Any]) -> dict[str, np.ndarray]:
    """Per-uid zero (hold) action dict matching the env's action space."""
    return {
        uid: np.zeros(box.shape, dtype=np.float32)
        for uid, box in env.action_space.spaces.items()  # type: ignore[attr-defined]
    }


def rollout_hold(
    *,
    seed: int,
    n_steps: int = 80,
    render_backend: str | None = None,
) -> HoldMetrics:
    """Rung-0 stability smoke: zero-action hold of the attached bar (ADR-026 §Decision 1).

    Builds the matched env, holds both arms (zero joint delta) for
    ``n_steps``, and reports whether the telemetry stayed finite and the
    bounded tilt / wrist-stress maxima. The Rung-0 gate asserts the
    dual-hold attach is stable (the bar is held, no solver blow-up,
    constraint force bounded) before Rung 1 is attempted.

    Args:
        seed: Episode seed (P6).
        n_steps: Hold duration in env ticks.
        render_backend: Optional ManiSkill render backend (``"none"`` on
            headless hosts).

    Returns:
        The :class:`HoldMetrics` summary.
    """
    env = make_cocarry_env(
        condition_id="cocarry_matched_panda_pair",
        episode_length=n_steps + 5,
        root_seed=seed,
        render_backend=render_backend,
    )
    try:
        env.reset(seed=seed)
        hold = _zero_actions(env)
        max_tilt = 0.0
        max_stress = 0.0
        finite = True
        for _ in range(n_steps):
            _, _, _, _, _ = env.step(hold)
            tel = env.get_telemetry()  # type: ignore[attr-defined]
            tilt = _to_float(tel["tilt_deg"])
            stress = _to_float(tel["stress_proxy"])
            if not (np.isfinite(tilt) and np.isfinite(stress)):
                finite = False
                break
            max_tilt = max(max_tilt, tilt)
            max_stress = max(max_stress, stress)
        return HoldMetrics(
            seed=seed, finite=finite, max_tilt_deg=max_tilt, max_stress_proxy=max_stress
        )
    finally:
        env.close()


def rollout_episode(
    *,
    condition_id: str,
    seed: int,
    episode_length: int,
    render_backend: str | None = None,
) -> EpisodeMetrics:
    """Rung-1 rollout of one co-carry episode (ADR-026 §Decision 1-2).

    Drives the held arm(s) with the matched controller; in the single-arm
    positive-control condition the partner is retracted by the env and is
    held (zero delta), so the ego attempts the task alone.

    Args:
        condition_id: ``"cocarry_matched_panda_pair"`` or
            ``"cocarry_single_arm_positive_control"``.
        seed: Episode seed (P6).
        episode_length: Truncation horizon, env ticks.
        render_backend: Optional ManiSkill render backend.

    Returns:
        The per-episode :class:`EpisodeMetrics`.
    """
    env = make_cocarry_env(
        condition_id=condition_id,
        episode_length=episode_length,
        root_seed=seed,
        render_backend=render_backend,
    )
    single_arm = bool(env.single_arm)  # type: ignore[attr-defined]
    try:
        controllers = build_matched_controllers()
        obs, _ = env.reset(seed=seed)
        for c in controllers.values():
            c.reset(seed=seed)
        ego_uid = env.ego_uid  # type: ignore[attr-defined]
        partner_uid = env.partner_uid  # type: ignore[attr-defined]
        info: dict[str, Any] = {}
        for _ in range(episode_length):
            action = {ego_uid: controllers[ego_uid].act(obs)}
            if single_arm:
                # Partner is disabled + retracted by the env; hold it
                # (zero joint delta) so the ego attempts the task alone.
                action[partner_uid] = np.zeros(
                    env.action_space.spaces[partner_uid].shape,  # type: ignore[attr-defined]
                    dtype=np.float32,
                )
            else:
                action[partner_uid] = controllers[partner_uid].act(obs)
            obs, _, terminated, truncated, info = env.step(action)
            if bool(np.asarray(terminated).reshape(-1)[0]) or bool(
                np.asarray(truncated).reshape(-1)[0]
            ):
                break
        tel = env.get_telemetry()  # type: ignore[attr-defined]
        return EpisodeMetrics(
            seed=seed,
            success=bool(np.asarray(info["success"]).reshape(-1)[0]),
            centroid_to_goal=_to_float(tel["centroid_to_goal"]),
            max_tilt_deg=_to_float(tel["max_tilt_deg"]),
            max_stress_proxy=_to_float(tel["max_stress_proxy"]),
            n_steps=episode_length,
        )
    finally:
        env.close()


def evaluate_condition(
    *,
    condition_id: str,
    seeds: list[int],
    episode_length: int = COCARRY_DEFAULT_EPISODE_LENGTH,
    render_backend: str | None = None,
) -> list[EpisodeMetrics]:
    """Run a condition across ``seeds`` and return per-seed metrics (ADR-026 §Decision 2)."""
    return [
        rollout_episode(
            condition_id=condition_id,
            seed=s,
            episode_length=episode_length,
            render_backend=render_backend,
        )
        for s in seeds
    ]


def summarize(metrics: list[EpisodeMetrics]) -> dict[str, float]:
    """Aggregate per-seed metrics into a small report dict (ADR-026 §Decision 2).

    Includes the success rate, the centroid/tilt summaries, and the
    constraint-force percentiles that feed the ``f_max`` derivation, plus
    how close the tilt/stress maxima run to their limits (the binding
    evidence).
    """
    if not metrics:
        return {}
    succ = float(np.mean([m.success for m in metrics]))
    tilts = np.asarray([m.max_tilt_deg for m in metrics])
    stresses = np.asarray([m.max_stress_proxy for m in metrics])
    dists = np.asarray([m.centroid_to_goal for m in metrics])
    succ_stresses = np.asarray([m.max_stress_proxy for m in metrics if m.success])
    return {
        "n": float(len(metrics)),
        "success_rate": succ,
        "centroid_to_goal_p50": float(np.percentile(dists, 50)),
        "max_tilt_p50": float(np.percentile(tilts, 50)),
        "max_tilt_p90": float(np.percentile(tilts, 90)),
        "max_tilt_max": float(np.max(tilts)),
        "tilt_limit": COCARRY_TILT_MAX_DEG,
        "stress_p50": float(np.percentile(stresses, 50)),
        "stress_p90": float(np.percentile(stresses, 90)),
        "stress_p99": float(np.percentile(stresses, 99)),
        "stress_max": float(np.max(stresses)),
        "stress_limit": COCARRY_STRESS_MAX_PROXY_N,
        "success_stress_p95": float(np.percentile(succ_stresses, 95))
        if succ_stresses.size
        else float("nan"),
    }


__all__ = [
    "EpisodeMetrics",
    "HoldMetrics",
    "build_matched_controllers",
    "evaluate_condition",
    "rollout_episode",
    "rollout_hold",
    "summarize",
]
