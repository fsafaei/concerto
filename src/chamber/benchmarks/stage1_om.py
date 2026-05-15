# SPDX-License-Identifier: Apache-2.0
"""Stage-1 Observation-Modality (OM) spike adapter (T5b.2; plan/07 §3).

Implements the ``chamber.benchmarks.stage1_om.run_axis`` entry point
:mod:`chamber.cli._spike_run` dispatches to when a user invokes
``chamber-spike run --axis OM`` without ``--dry-run``. The adapter:

1. Loads the canonical OM pre-registration from
   ``spikes/preregistration/OM.yaml`` via
   :func:`chamber.evaluation.prereg.load_prereg`.
2. For each seed x condition x episode (5 x 2 x 20 = 200 per OM
   spike per plan/07 §2), builds the per-condition env, drives a
   scripted-heuristic partner against a deterministic ego policy,
   and records the per-episode success.
3. Aggregates the episode results into a
   :class:`chamber.evaluation.results.SpikeRun` and returns it to
   the CLI for serialisation.

Phase-0 scoping (READ THIS BEFORE THINKING THE SCIENCE IS DONE).
Plan/07 §3 OM spike: same panda+fetch agent pair as AS (Stage-1's
foundation pair); the condition manipulation is on the observation
side, not on the agent count:

- Homo: vision-only observation (RGB-D from a fixed workspace camera).
- Hetero: vision + force-torque + proprioceptive state fused per uid.

Neither obs-modality variant is shipped yet — wiring real ManiSkill
v3 observation builders (RGB-D, force-torque, proprio) into the
Stage-1 pick-place env is plan/07 §T5b.2 follow-up work (Phase 1).
For Phase-0 the adapter uses
:class:`chamber.envs.mpe_cooperative_push.MPECooperativePushEnv` as
a CPU-friendly stand-in. Because OM does not change the agent count,
**both conditions use the same** ``agent_uids = ("panda_wristcam",
"fetch")`` and consequently produce identical Phase-0 SpikeRuns —
the OM distinction lives in the prereg's ``condition_id`` strings
(``stage1_pickplace_vision_only`` vs
``stage1_pickplace_vision_plus_force_torque_plus_proprio``), which
the Phase-1 obs-modality factory will read to pick its actual obs
keys. Episode success is rule-based
(``mean_reward > _SUCCESS_THRESHOLD``); the ego is a frozen-zero
policy (no training).

The ego trainer integration (``concerto.training.ego_aht.train``
per plan/07 §T5b.2) is scaffolded behind the ``_zero_ego_action``
callable injection point (the ``ego_action_fn`` kwarg on
:func:`_run_axis_with_factories`) — Phase-1 work wires a trained
``EgoTrainer`` here without touching the SpikeRun aggregation
shape. Same for the real Stage-1 obs-modality factory: replacing
:func:`_default_env_factory` is a one-spot edit.

The Tier-1 fake-env test injects :class:`tests.fakes.FakeMultiAgentEnv`
via the ``env_factory`` kwarg; the Tier-2 SAPIEN-gated test runs
the real env path on a Vulkan host.

Twinning with :mod:`chamber.benchmarks.stage1_as`. The OM adapter
mirrors the AS adapter's shape verbatim — same prereg loader,
episode loop, ``_run_one_episode``, partner construction, and
seed-derivation pattern. The two adapters duplicate ~90% of their
code; refactoring into a shared ``_stage1_common.py`` is a Phase-1
follow-up once plan/07 §T5b.3 / §T5b.4 add Stage-2/3 adapters and
the duplication becomes 6-way. For Phase-0 two copies is the right
trade per the project's "avoid premature abstraction" guidance.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

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
    from numpy.typing import NDArray

    EnvFactory = Callable[[str, tuple[str, str], int], gym.Env[Any, Any]]
    EgoActionFn = Callable[[str, Mapping[str, Any], NDArray[np.float32]], NDArray[np.float32]]

#: ADR-007 §3.4 axis label this adapter implements.
_AXIS: str = "OM"

#: Maximum env steps per evaluation episode (Phase-0 stand-in budget).
#:
#: MPE Cooperative-Push uses 50-step episodes by default; mirrors the
#: AS adapter's choice. Phase-1 raises this when the real Stage-1
#: pick-place env's natural horizon is longer.
_MAX_STEPS_PER_EPISODE: int = 50

#: Success threshold on the mean per-step reward over the episode.
#: Same value as :mod:`chamber.benchmarks.stage1_as` because both
#: adapters share the same Phase-0 MPE stand-in env; the AS
#: docstring carries the threshold rationale.
_SUCCESS_THRESHOLD: float = -0.30

#: Default per-condition agent-uid tuples. Per plan/07 §3 OM uses the
#: same panda + fetch agent pair across BOTH conditions — the OM
#: distinction is on the obs side, not the agent side. The two
#: condition_id strings map to identical tuples; the Phase-1
#: obs-modality factory will read ``condition_id`` directly to pick
#: the right obs builder.
_CONDITION_UIDS: dict[str, tuple[str, str]] = {
    "stage1_pickplace_vision_only": ("panda_wristcam", "fetch"),
    "stage1_pickplace_vision_plus_force_torque_plus_proprio": (
        "panda_wristcam",
        "fetch",
    ),
}

#: Canonical prereg path. Resolved relative to the repo root the
#: caller's CWD lives under; the prereg file is part of B5's
#: ``spikes/preregistration/`` set.
_PREREG_RELATIVE_PATH: Path = Path("spikes") / "preregistration" / "OM.yaml"


def run_axis(args: argparse.Namespace) -> SpikeRun:
    """Run the Stage-1 OM spike end-to-end (T5b.2; ADR-007 §Implementation staging).

    Entry point :mod:`chamber.cli._spike_run` dispatches to when a
    user invokes ``chamber-spike run --axis OM`` without ``--dry-run``.
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
        ValueError: If ``args.axis`` is not ``"OM"`` (defensive;
            B7's dispatch routes by axis name) or if a prereg
            condition_id is not in the Phase-0 condition map.
        FileNotFoundError: If the canonical prereg is missing
            (typically because the maintainer has not yet checked
            out the B5 commit; the friendly message names the path).
    """
    if args.axis != _AXIS:
        msg = f"stage1_om.run_axis: expected axis={_AXIS!r}, got {args.axis!r}"
        raise ValueError(msg)
    return _run_axis_with_factories(
        prereg=_load_canonical_prereg(),
        env_factory=_default_env_factory,
        ego_action_fn=_zero_ego_action,
    )


def _load_canonical_prereg() -> PreregistrationSpec:
    """Resolve and load ``spikes/preregistration/OM.yaml`` (plan/06 §6 #1)."""
    here = Path.cwd()
    prereg_path = here / _PREREG_RELATIVE_PATH
    if not prereg_path.exists():
        msg = (
            f"stage1_om.run_axis: pre-registration not found at {prereg_path}. "
            "The Stage-1 OM adapter expects spikes/preregistration/OM.yaml "
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
                    f"stage1_om.run_axis: prereg condition_id {condition_id!r} "
                    f"is not in the Phase-0 condition map {sorted(_CONDITION_UIDS)}. "
                    "This typically means the OM pre-registration was edited "
                    "after the Phase-0 adapter shipped — re-issue the prereg "
                    "with a new git_tag per ADR-007 §Discipline, or update "
                    "_CONDITION_UIDS in this module."
                )
                raise ValueError(msg)
            ego_uid, partner_uid = agent_uids
            for episode_idx in range(prereg.episodes_per_seed):
                initial_state_seed = _derive_episode_seed(seed=seed, episode_idx=episode_idx)
                # Phase-1: hoist env construction to the per-condition
                # loop (real ManiSkill envs take seconds each, MPE / Fake
                # take microseconds). The per-episode reset already
                # threads ``initial_state_seed`` into the inner RNG.
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
        spike_id=f"stage1_om_{prereg.git_tag}",
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
    # Phase-1: the real Stage-1 pick-place env exposes ``terminated=True``
    # on success, at which point the ``or terminated`` clause becomes the
    # dominant signal and the rule-based mean-reward fallback can be
    # dropped (plan/07 §T5b.2 follow-up).
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

    The real Stage-1 obs-modality variants (vision-only vs
    vision+FT+proprio on the panda+fetch pick-place task) live in
    Phase-1 work — see plan/07 §T5b.2. For Phase-0 the adapter uses
    :class:`chamber.envs.mpe_cooperative_push.MPECooperativePushEnv`
    so the SpikeRun shape, the partner-stack contract, and the
    chamber-side dispatch are all exercisable on CPU.

    Because OM does not change the agent count, both conditions map
    to the same ``agent_uids`` tuple; the ``condition_id`` argument
    is unused in Phase-0. The Phase-1 obs-modality factory will read
    ``condition_id`` to pick the actual obs builder.
    """
    del condition_id  # Phase-1 factory reads this; Phase-0 MPE stand-in ignores.
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
    ``initial_state_seed`` streams. The substream name is
    ``"chamber.benchmarks.stage1_om.episode.{episode_idx}"`` — distinct
    from the AS adapter's substream name so the two axes do not
    alias seeds when run back-to-back in the same process.
    """
    rng = derive_substream(
        f"chamber.benchmarks.stage1_om.episode.{episode_idx}", root_seed=seed
    ).default_rng()
    return int(rng.integers(0, 2**31 - 1))


__all__ = ["run_axis"]
