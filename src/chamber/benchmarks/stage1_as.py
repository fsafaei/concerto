# SPDX-License-Identifier: Apache-2.0
"""Stage-1 Action-Space (AS) spike adapter (T5b.2; plan/07 §3).

Implements the ``chamber.benchmarks.stage1_as.run_axis`` entry point
:mod:`chamber.cli._spike_run` dispatches to when a user invokes
``chamber-spike run --axis AS`` without ``--dry-run``. The adapter:

1. Loads the canonical AS pre-registration from
   ``spikes/preregistration/AS.yaml`` via
   :func:`chamber.evaluation.prereg.load_prereg`.
2. For each seed x condition x episode (5 x 2 x 20 = 200 per AS
   spike per plan/07 §2), builds the per-condition env, drives a
   scripted-heuristic partner against a deterministic ego policy,
   and records the per-episode success.
3. Aggregates the episode results into a
   :class:`chamber.evaluation.results.SpikeRun` and returns it to
   the CLI for serialisation.

Phase-0 scoping (READ THIS BEFORE THINKING THE SCIENCE IS DONE).
Plan/07 §3 calls for distinct task envs per condition:

- Homo: 7-DOF panda arm only on a shared pick-place task; the
  baseline is shared-parameter MAPPO.
- Hetero: 7-DOF panda arm + 2-DOF differential-drive fetch base on
  the same task; the baseline is per-agent EgoAHTHAPPO.

Neither task env is shipped yet — building real ManiSkill v3
pick-place envs with the named robot tuples is plan/07 §T5b.2
follow-up work (Phase 1). For Phase-0, the adapter uses
:class:`chamber.envs.mpe_cooperative_push.MPECooperativePushEnv` as
a CPU-friendly stand-in: both conditions share the MPE physics but
diverge on the ``agent_uids`` tuple to encode the AS distinction
(``("panda_wristcam", "panda_partner")`` vs
``("panda_wristcam", "fetch")``). Episode success is rule-based
(``mean_reward > _SUCCESS_THRESHOLD``); the ego is a frozen-zero
policy (no training).

The ego trainer integration (``concerto.training.ego_aht.train``
per plan/07 §T5b.2) is scaffolded behind the ``_make_ego_action``
callable injection point — Phase-1 work wires a trained
``EgoTrainer`` here without touching the SpikeRun aggregation
shape. Same for the real Stage-1 pick-place env: replacing
:func:`_default_env_factory` is a one-spot edit.

The Tier-1 fake-env test injects :class:`tests.fakes.FakeMultiAgentEnv`
via the ``env_factory`` kwarg; the Tier-2 SAPIEN-gated test runs
the real env path on a Vulkan host.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from chamber.envs.mpe_cooperative_push import MPECooperativePushEnv
from chamber.evaluation.prereg import PreregistrationSpec, load_prereg
from chamber.evaluation.results import EpisodeResult, SpikeRun
from chamber.partners.api import PartnerSpec
from chamber.partners.heuristic import ScriptedHeuristicPartner
from concerto.training.seeding import derive_substream

if TYPE_CHECKING:
    import argparse
    from collections.abc import Callable, Mapping

    import gymnasium as gym
    import numpy as np
    from numpy.typing import NDArray

    EnvFactory = Callable[[str, tuple[str, str], int], gym.Env[Any, Any]]
    EgoActionFn = Callable[[str, Mapping[str, Any], NDArray[np.float32]], NDArray[np.float32]]

#: ADR-007 §3.4 axis label this adapter implements.
_AXIS: str = "AS"

#: Maximum env steps per evaluation episode (Phase-0 stand-in budget).
#:
#: MPE Cooperative-Push uses 50-step episodes by default; the adapter
#: walks the full horizon so the partner has time to drive a
#: meaningful trajectory. Phase-1 raises this when the real Stage-1
#: pick-place env's natural horizon is longer.
_MAX_STEPS_PER_EPISODE: int = 50

#: Success threshold on the mean per-step reward over the episode
#: (MPE Cooperative-Push rewards are negative; ``mean_reward >
#: -0.30`` ≈ "agents got reasonably close to landmarks"). Phase-1
#: success rules use the env's terminal-success flag where
#: applicable; for Phase-0 stand-in MPE there is no terminal flag.
_SUCCESS_THRESHOLD: float = -0.30

#: Default per-condition agent-uid tuples (Phase-0 stand-in mapping
#: of plan/07 §3 condition_id strings onto MPE-compatible 2-tuples).
_CONDITION_UIDS: dict[str, tuple[str, str]] = {
    # Homo: "panda only" — both MPE agents share the panda embodiment
    # label so the AS distinction is "homogeneous arm" vs "arm + base".
    "stage1_pickplace_panda_only_mappo_shared_param": (
        "panda_wristcam",
        "panda_partner",
    ),
    # Hetero: panda arm + differential-drive fetch base.
    "stage1_pickplace_panda_plus_fetch_ego_aht_happo_per_agent": (
        "panda_wristcam",
        "fetch",
    ),
}

#: Canonical prereg path. Resolved relative to the repo root the
#: caller's CWD lives under; the prereg file is part of B5's
#: ``spikes/preregistration/`` set.
_PREREG_RELATIVE_PATH: Path = Path("spikes") / "preregistration" / "AS.yaml"


def run_axis(args: argparse.Namespace) -> SpikeRun:
    """Run the Stage-1 AS spike end-to-end (T5b.2; ADR-007 §Implementation staging).

    Entry point :mod:`chamber.cli._spike_run` dispatches to when a
    user invokes ``chamber-spike run --axis AS`` without ``--dry-run``.
    Cites ADR-007 §Implementation staging (Stage 1 axis grouping),
    ADR-007 §Validation criteria (≥20pp gap rule), and plan/07 §3
    (per-condition spike specification).

    Args:
        args: The argparse namespace from
            :func:`chamber.cli._spike_run.add_parser`. Only
            ``args.axis`` is consulted directly; the prereg drives
            the rest (seeds, episodes_per_seed, condition_pair,
            git_tag).

    Returns:
        A :class:`SpikeRun` carrying the per-episode results,
        ready to be serialised by the CLI dispatcher.

    Raises:
        ValueError: If ``args.axis`` is not ``"AS"`` (defensive;
            B7's dispatch routes by axis name) or if a prereg
            condition_id is not in the Phase-0 condition map.
        FileNotFoundError: If the canonical prereg is missing
            (typically because the maintainer has not yet checked
            out the B5 commit; the friendly message names the path).
    """
    if args.axis != _AXIS:
        msg = f"stage1_as.run_axis: expected axis={_AXIS!r}, got {args.axis!r}"
        raise ValueError(msg)
    return _run_axis_with_factories(
        prereg=_load_canonical_prereg(),
        env_factory=_default_env_factory,
        ego_action_fn=_zero_ego_action,
    )


def _load_canonical_prereg() -> PreregistrationSpec:
    """Resolve and load ``spikes/preregistration/AS.yaml`` (plan/06 §6 #1)."""
    here = Path.cwd()
    prereg_path = here / _PREREG_RELATIVE_PATH
    if not prereg_path.exists():
        msg = (
            f"stage1_as.run_axis: pre-registration not found at {prereg_path}. "
            "The Stage-1 AS adapter expects spikes/preregistration/AS.yaml "
            "(shipped by plan/06 §6 #1; PR #109). Run chamber-spike from the "
            "repo root, or check out a commit that includes the file."
        )
        raise FileNotFoundError(msg)
    return load_prereg(prereg_path)


def _run_axis_with_factories(
    *,
    prereg: PreregistrationSpec,
    env_factory: EnvFactory,
    ego_action_fn: EgoActionFn,
) -> SpikeRun:
    """Drive the spike with injectable env + ego factories (T5b.2; plan/07 §3).

    Pulled out of :func:`run_axis` so the Tier-1 fake-env test can
    inject :class:`tests.fakes.FakeMultiAgentEnv` and a synthetic
    ego-action callable without monkey-patching module globals.
    Phase-1 will inject a real ``concerto.training.ego_aht.train``-
    backed ego instead of the zero-action placeholder.

    Args:
        prereg: The loaded :class:`PreregistrationSpec`.
        env_factory: Callable returning the per-condition env;
            signature ``(condition_id, agent_uids, root_seed) -> gym.Env``.
        ego_action_fn: Callable returning the ego action vector for
            one step; signature
            ``(ego_uid, obs, partner_action) -> NDArray[float32]``.

    Returns:
        The aggregated :class:`SpikeRun`.
    """
    condition_pair = prereg.condition_pair
    conditions = (condition_pair.homogeneous_id, condition_pair.heterogeneous_id)

    episode_results: list[EpisodeResult] = []
    for seed in prereg.seeds:
        for condition_id in conditions:
            agent_uids = _CONDITION_UIDS.get(condition_id)
            if agent_uids is None:
                msg = (
                    f"stage1_as.run_axis: prereg condition_id {condition_id!r} "
                    f"is not in the Phase-0 condition map {sorted(_CONDITION_UIDS)}."
                )
                raise ValueError(msg)
            ego_uid, partner_uid = agent_uids
            for episode_idx in range(prereg.episodes_per_seed):
                initial_state_seed = _derive_episode_seed(seed=seed, episode_idx=episode_idx)
                env = env_factory(condition_id, agent_uids, seed)
                partner = _make_scripted_partner(partner_uid=partner_uid)
                episode_results.append(
                    _run_one_episode(
                        env=env,
                        partner=partner,
                        ego_uid=ego_uid,
                        partner_uid=partner_uid,
                        ego_action_fn=ego_action_fn,
                        seed=seed,
                        episode_idx=episode_idx,
                        initial_state_seed=initial_state_seed,
                        condition_id=condition_id,
                    )
                )

    return SpikeRun(
        spike_id=f"stage1_as_{prereg.git_tag}",
        prereg_sha="",  # filled in by the launch-time chamber-spike verify-prereg step
        git_tag=prereg.git_tag,
        axis=_AXIS,
        condition_pair=condition_pair,
        seeds=list(prereg.seeds),
        episode_results=episode_results,
    )


def _run_one_episode(
    *,
    env: gym.Env[Any, Any],
    partner: ScriptedHeuristicPartner,
    ego_uid: str,
    partner_uid: str,
    ego_action_fn: EgoActionFn,
    seed: int,
    episode_idx: int,
    initial_state_seed: int,
    condition_id: str,
) -> EpisodeResult:
    """Roll one evaluation episode out and return the success record (plan/07 §3)."""
    obs_tuple = env.reset(seed=initial_state_seed)
    obs: Mapping[str, Any] = obs_tuple[0]
    partner.reset(seed=initial_state_seed)
    total_reward = 0.0
    n_steps = 0
    terminated = False
    truncated = False
    while n_steps < _MAX_STEPS_PER_EPISODE:
        partner_action = partner.act(obs, deterministic=True)
        ego_action = ego_action_fn(ego_uid, obs, partner_action)
        action_dict = {ego_uid: ego_action, partner_uid: partner_action}
        step_out = env.step(action_dict)
        obs = step_out[0]
        reward = float(step_out[1])
        terminated = bool(step_out[2])
        truncated = bool(step_out[3])
        total_reward += reward
        n_steps += 1
        if terminated or truncated:
            break
    mean_reward = total_reward / max(1, n_steps)
    success = mean_reward > _SUCCESS_THRESHOLD or terminated
    return EpisodeResult(
        seed=seed,
        episode_idx=episode_idx,
        initial_state_seed=initial_state_seed,
        success=success,
        metadata={
            "condition": condition_id,
            "mean_reward": f"{mean_reward:.4f}",
            "n_steps": str(n_steps),
        },
    )


def _default_env_factory(
    condition_id: str, agent_uids: tuple[str, str], root_seed: int
) -> gym.Env[Any, Any]:
    """Build the per-condition env (Phase-0 MPE stand-in; plan/07 §T5b.2 deferral).

    The real Stage-1 pick-place env tuples (panda-only vs panda+fetch
    on shared pick-place) live in Phase-1 work — see plan/07 §T5b.2.
    For Phase-0 the adapter uses
    :class:`chamber.envs.mpe_cooperative_push.MPECooperativePushEnv`
    so the SpikeRun shape, the partner-stack contract, and the
    chamber-side dispatch are all exercisable on CPU. The
    ``agent_uids`` tuple encodes the AS distinction at the obs / action
    keying level (the only project surface the partner stack and
    chamber-eval bootstrap actually see).
    """
    del condition_id  # tuple encodes the distinction; module docstring explains.
    return MPECooperativePushEnv(agent_uids=agent_uids, root_seed=root_seed)


def _zero_ego_action(
    ego_uid: str, obs: Mapping[str, Any], partner_action: NDArray[np.float32]
) -> NDArray[np.float32]:
    """Ego policy placeholder: always-zero action vector (Phase-0; plan/07 §T5b.2 deferral).

    Phase-1 swaps this for a ``concerto.training.ego_aht.train``-
    backed ego that loads a trained checkpoint and runs deterministic
    inference. The function's signature is the injection contract.
    """
    del ego_uid, obs
    import numpy as np

    return np.zeros_like(partner_action, dtype=np.float32)


def _make_scripted_partner(*, partner_uid: str) -> ScriptedHeuristicPartner:
    """Construct the scripted-heuristic partner with the right uid (plan/04 §3.4)."""
    spec = PartnerSpec(
        class_name="scripted_heuristic",
        seed=0,
        checkpoint_step=None,
        weights_uri=None,
        extra={"uid": partner_uid, "target_xy": "0.0,0.0", "action_dim": "2"},
    )
    return ScriptedHeuristicPartner(spec)


def _derive_episode_seed(*, seed: int, episode_idx: int) -> int:
    """Deterministic per-(seed, episode_idx) sub-seed (ADR-002 P6).

    Routes through :func:`concerto.training.seeding.derive_substream`
    so two runs at the same ``seed`` produce byte-identical
    ``initial_state_seed`` streams.
    """
    rng = derive_substream(
        f"chamber.benchmarks.stage1_as.episode.{episode_idx}", root_seed=seed
    ).default_rng()
    return int(rng.integers(0, 2**31 - 1))


__all__ = ["run_axis"]
