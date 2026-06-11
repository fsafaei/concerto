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
``("panda_wristcam", "fetch")``). Episode success is
``terminated and not truncated`` — i.e. the env's own
:meth:`Stage1PickPlaceEnv.evaluate` predicate fired
(``is_obj_placed & is_robot_static``). Stage-1a's MPE stand-in
does not emit ``terminated=True`` on success, so Stage-1a success
rates are structurally zero under this rule — correct per ADR-007
§Stage 1a (Stage 1a is rig-validation, not gap measurement). The
ego is a frozen-zero policy (no training).

Stage-1a uses :func:`_zero_ego_action_factory` as the production
default; the trained-ego path is Stage-1b (Phase-1, ADR-007 §Stage 1b).
The ego-action injection seam is the
:class:`chamber.benchmarks.stage1_common.EgoActionFactory` Protocol —
called once per ``(seed, condition)`` pair so the Phase-1 trained
policy is built once and reused across the 20 evaluation episodes
within that cell. Replacing the production default is a one-line
swap in :func:`_run_axis_with_factories`'s call site once the
trained-policy factory lands (see
:mod:`chamber.benchmarks.stage1_common` for the verbatim Phase-1
wiring contract).

The Tier-1 fake-env test injects :class:`tests.fakes.FakeMultiAgentEnv`
via the ``env_factory`` kwarg and a fake :class:`EgoActionFactory`
to pin the seam's per-cell lifecycle contract (factory called
exactly ``n_seeds x n_conditions = 5 x 2 = 10`` times per
``run_axis`` invocation); the Tier-2 SAPIEN-gated test runs the
real env path on a Vulkan host.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import gymnasium as gym

from chamber.benchmarks.stage1_common import (
    EgoActionCallable,
    EgoActionFactory,
    _zero_ego_action_factory,
)
from chamber.envs.mpe_cooperative_push import MPECooperativePushEnv
from chamber.evaluation.prereg import PreregistrationSpec, load_prereg, verify_git_tag
from chamber.evaluation.results import EpisodeResult, SpikeRun, SubStage
from chamber.partners.api import PartnerSpec
from chamber.partners.heuristic import ScriptedHeuristicPartner
from concerto.training.seeding import derive_substream

if TYPE_CHECKING:
    import argparse
    from collections.abc import Callable, Mapping

    EnvFactory = Callable[[str, tuple[str, str], int], gym.Env[Any, Any]]

#: ADR-007 §3.4 axis label this adapter implements.
_AXIS: str = "AS"

#: Stage-1a (MPE Cooperative-Push) evaluation horizon, in env ticks.
#: Matches ``chamber.envs.mpe_cooperative_push.DEFAULT_EPISODE_LENGTH``;
#: the adapter walks the full MPE horizon so the partner has time to
#: drive a meaningful trajectory.
_MAX_STEPS_STAGE_1A: int = 50


def _max_steps_for(sub_stage: SubStage) -> int:
    """Evaluation horizon per sub-stage (ADR-007 §Stage 1b).

    Sourced from the env's own truncation horizon so training and
    evaluation share one value. Stage-1a walks the 50-step MPE
    stand-in. Stage-1b walks the real Stage-1 pick-place env's natural
    horizon (:data:`chamber.envs.stage1_pickplace.DEFAULT_EPISODE_LENGTH`
    = 100). The Phase-0 holdover hardcoded 50 for both, so Stage-1b
    evaluated a policy *trained* over 100 steps on only its first 50 —
    a 2x train/eval horizon mismatch surfaced by the P1.05.9 third §4a
    firing (every eval episode stopped at exactly n_steps=50 with the
    env's own truncation never reached).
    """
    if sub_stage == "1b":
        from chamber.envs.stage1_pickplace import DEFAULT_EPISODE_LENGTH

        return DEFAULT_EPISODE_LENGTH
    return _MAX_STEPS_STAGE_1A


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

#: Canonical Stage-1b config path (P1.05; ADR-007 §Stage 1b).
#: Resolved relative to the repo root via :data:`Path.cwd`; the file
#: is shipped at
#: ``configs/training/ego_aht_happo/stage1_pickplace.yaml`` and read
#: only when ``args.sub_stage == "1b"``.
_STAGE1B_CONFIG_PATH: Path = (
    Path("configs") / "training" / "ego_aht_happo" / "stage1_pickplace.yaml"
)


def run_axis(args: argparse.Namespace) -> SpikeRun:
    """Run the Stage-1 AS spike end-to-end (T5b.2; ADR-007 §Implementation staging).

    Entry point :mod:`chamber.cli._spike_run` dispatches to when a
    user invokes ``chamber-spike run --axis AS`` without ``--dry-run``.
    Branches on ``args.sub_stage`` (P1.05; ADR-007 §Implementation
    staging Rev 8) — Stage-1a routes to the Phase-0 MPE stand-in +
    :func:`_zero_ego_action_factory`; Stage-1b routes to the real
    :class:`chamber.envs.stage1_pickplace.Stage1PickPlaceEnv` +
    :class:`chamber.benchmarks.stage1_common.TrainedPolicyFactory`
    (Phase-1 science evaluation).

    Cites ADR-007 §Implementation staging (Stage 1 axis grouping),
    ADR-007 §Stage 1a / §Stage 1b (sub-stage split), ADR-007
    §Validation criteria (≥20pp gap rule), and plan/07 §3
    (per-condition spike specification).

    Args:
        args: The argparse namespace from
            :func:`chamber.cli._spike_run.add_parser`. ``args.axis``
            and ``args.sub_stage`` (default ``"1a"``) drive dispatch;
            the prereg drives the rest (seeds, episodes_per_seed,
            condition_pair, git_tag).

    Returns:
        A :class:`SpikeRun` carrying the per-episode results,
        ready to be serialised by the CLI dispatcher.

    Raises:
        ValueError: If ``args.axis`` is not ``"AS"`` (defensive;
            B7's dispatch routes by axis name); if ``args.sub_stage``
            is not ``"1a"`` or ``"1b"``; or if a prereg condition_id
            is not in the condition map for the resolved sub-stage.
        FileNotFoundError: If the canonical prereg is missing
            (typically because the maintainer has not yet checked
            out the B5 commit; the friendly message names the path).
    """
    if args.axis != _AXIS:
        msg = f"stage1_as.run_axis: expected axis={_AXIS!r}, got {args.axis!r}"
        raise ValueError(msg)
    spec, prereg_path = _load_canonical_prereg()
    # ADR-007 §Discipline: the SpikeRun MUST carry the verified blob
    # SHA of the locked pre-registration YAML, so the audit chain
    # closes. ``verify_git_tag`` raises ``PreregistrationError`` on
    # any tag-mismatch / file-outside-repo / missing-tag condition,
    # which is exactly the loud-fail the discipline rule wants — do
    # not catch it here. The ``TestStage1ASPreregDiscipline`` class
    # in ``tests/integration/test_stage1_as_real.py`` is the
    # regression pin.
    prereg_sha = verify_git_tag(spec, prereg_path, repo_path=Path.cwd())

    sub_stage = getattr(args, "sub_stage", "1a")
    if sub_stage == "1a":
        return _run_axis_with_factories(
            prereg=spec,
            prereg_sha=prereg_sha,
            env_factory=_default_env_factory,
            ego_action_factory=_zero_ego_action_factory,
            sub_stage="1a",
        )
    if sub_stage == "1b":
        # P1.05: science-evaluation dispatch. Lazy import the
        # TrainedPolicyFactory + config loader so the Stage-1a path
        # stays Tier-1-import-safe (Hydra / OmegaConf are heavy +
        # only relevant when the trained ego is wired). The cfg is
        # the canonical Stage-1b config; TrainedPolicyFactory rewrites
        # per-cell from env.ego_uid / env.partner_uid / env.condition_id.
        from chamber.benchmarks.stage1_common import (
            TrainedPolicyFactory,
        )
        from concerto.training.config import load_config

        cfg = load_config(config_path=_STAGE1B_CONFIG_PATH)
        return _run_axis_with_factories(
            prereg=spec,
            prereg_sha=prereg_sha,
            env_factory=_stage1b_env_factory,
            ego_action_factory=TrainedPolicyFactory(cfg=cfg),
            sub_stage="1b",
        )
    # argparse choices=... makes this unreachable; defensive raise.
    msg = (
        f"stage1_as.run_axis: unknown sub_stage {sub_stage!r}. "
        "Valid: '1a' (Phase-0 MPE stand-in) or '1b' (Phase-1 real "
        "ManiSkill v3 pick-place + trained-ego factory)."
    )
    raise ValueError(msg)


def _load_canonical_prereg() -> tuple[PreregistrationSpec, Path]:
    """Resolve and load ``spikes/preregistration/AS.yaml`` (plan/06 §6 #1).

    Returns the validated spec alongside the absolute ``prereg_path``
    so the caller can verify the YAML's blob SHA against its tagged
    blob via :func:`chamber.evaluation.prereg.verify_git_tag`
    (ADR-007 §Discipline).
    """
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
    return load_prereg(prereg_path), prereg_path


def _run_axis_with_factories(
    *,
    prereg: PreregistrationSpec,
    prereg_sha: str,
    env_factory: EnvFactory,
    ego_action_factory: EgoActionFactory,
    sub_stage: SubStage = "1a",
) -> SpikeRun:
    """Drive the spike with injectable env + ego factories (T5b.2; plan/07 §3).

    Pulled out of :func:`run_axis` so the Tier-1 fake-env test can
    inject :class:`tests.fakes.FakeMultiAgentEnv` and a synthetic
    :class:`~chamber.benchmarks.stage1_common.EgoActionFactory` without
    monkey-patching module globals. The factory is called *once per*
    ``(seed, condition)`` pair (5 x 2 = 10 calls per Stage-1 spike);
    the returned per-step callable is then re-used across the 20
    evaluation episodes within that cell. Phase-1 (Stage 1b) supplies
    a factory that wires :func:`concerto.training.ego_aht.train` — see
    :mod:`chamber.benchmarks.stage1_common` for the verbatim contract.

    Env construction is also hoisted to *once per ``(seed, condition)``*
    (rather than once per episode) so the factory and the env see the
    same instance; ``env.reset(seed=initial_state_seed)`` re-seeds the
    per-episode RNG inside the env without re-allocating it.

    Args:
        prereg: The loaded :class:`PreregistrationSpec`.
        prereg_sha: 40-char hex blob SHA of the pre-registration YAML
            as stored at :attr:`PreregistrationSpec.git_tag`, returned
            by :func:`chamber.evaluation.prereg.verify_git_tag`. Made
            mandatory rather than defaulting to ``""`` so the empty-
            sha foot-gun that produced the 2026-05-17 audit-trail
            defect cannot reach production (ADR-007 §Discipline).
        env_factory: Callable returning the per-condition env;
            signature ``(condition_id, agent_uids, root_seed) -> gym.Env``.
        ego_action_factory: Stage-1 ego-action factory satisfying the
            :class:`~chamber.benchmarks.stage1_common.EgoActionFactory`
            Protocol. Stage-1a production default is
            :func:`~chamber.benchmarks.stage1_common._zero_ego_action_factory`.
        sub_stage: Sub-stage label stamped on the returned
            :attr:`SpikeRun.sub_stage` (ADR-016 §Decision). Stage-1a
            callers pass ``"1a"`` (the Phase-0 default), Stage-1b
            callers pass ``"1b"`` (P1.05 dispatch). The summarizer
            routes off this field directly.

    Returns:
        The aggregated :class:`SpikeRun`.
    """
    condition_pair = prereg.condition_pair
    conditions = (condition_pair.homogeneous_id, condition_pair.heterogeneous_id)

    episode_results: list[EpisodeResult] = []
    # Resolve the eval horizon once per spike from the sub-stage's env
    # (ADR-007 §Stage 1b; P1.05.9 firing #3 train/eval horizon fix).
    max_steps = _max_steps_for(sub_stage)
    for seed in prereg.seeds:
        for condition_id in conditions:
            agent_uids = _CONDITION_UIDS.get(condition_id)
            if agent_uids is None:
                msg = (
                    f"stage1_as.run_axis: prereg condition_id {condition_id!r} "
                    f"is not in the Phase-0 condition map {sorted(_CONDITION_UIDS)}. "
                    "This typically means the AS pre-registration was edited "
                    "after the Phase-0 adapter shipped — re-issue the prereg "
                    "with a new git_tag per ADR-007 §Discipline, or update "
                    "_CONDITION_UIDS in this module."
                )
                raise ValueError(msg)
            ego_uid, partner_uid = agent_uids
            # Build env + ego-action callable once per (seed, condition).
            # The factory contract names this cadence explicitly so the
            # Phase-1 trained-policy path can amortise its 100k-frame
            # training run across the 20 evaluation episodes within
            # the cell (ADR-007 §Stage 1b).
            env = env_factory(condition_id, agent_uids, seed)
            ego_action = ego_action_factory(env, seed)
            # Resolve per-condition partner action_dim for the scripted
            # partner. Stage-1a (MPE) keeps the Phase-0 default (2-D);
            # Stage-1b reads from the env's action_space (8-D panda_partner
            # under AS-homo; 13-D fetch under AS-hetero / OM-*) so the
            # scripted partner's act() returns a correctly-sized vector
            # that the ManiSkill v3 step boundary accepts.
            partner_action_dim = _resolve_partner_action_dim(
                env, partner_uid=partner_uid, sub_stage=sub_stage
            )
            for episode_idx in range(prereg.episodes_per_seed):
                initial_state_seed = _derive_episode_seed(seed=seed, episode_idx=episode_idx)
                partner = _make_scripted_partner(
                    partner_uid=partner_uid, action_dim=partner_action_dim
                )
                episode_results.append(
                    _run_one_episode(
                        env=env,
                        partner=partner,
                        ego_uid=ego_uid,
                        partner_uid=partner_uid,
                        ego_action=ego_action,
                        seed=seed,
                        episode_idx=episode_idx,
                        initial_state_seed=initial_state_seed,
                        condition_id=condition_id,
                        max_steps=max_steps,
                    )
                )

    return SpikeRun(
        spike_id=f"stage1_as_{prereg.git_tag}",
        prereg_sha=prereg_sha,
        git_tag=prereg.git_tag,
        axis=_AXIS,
        # P1.05 (ADR-007 §Stage 1a/§Stage 1b + ADR-016 §Decision): the
        # ``sub_stage`` field is now parametric. Stage-1a callers pass
        # ``sub_stage="1a"`` (the Phase-0 MPE-stand-in default). The
        # Stage-1b dispatch (chamber-spike run --axis AS --sub-stage 1b)
        # passes ``sub_stage="1b"``. The summarizer routes off this
        # field directly (no metadata-dict fallback per ADR-016).
        sub_stage=sub_stage,
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
    ego_action: EgoActionCallable,
    seed: int,
    episode_idx: int,
    initial_state_seed: int,
    condition_id: str,
    max_steps: int,
) -> EpisodeResult:
    """Roll one evaluation episode out and return the success record (plan/07 §3).

    ``max_steps`` is the sub-stage eval horizon from :func:`_max_steps_for`
    (ADR-007 §Stage 1b). If the rollout reaches ``max_steps`` without the
    env emitting ``terminated``/``truncated``, the episode is recorded as
    ``truncated=True`` so the eval-cap truncation is not silently lost
    (P1.05.9 firing #3).
    """
    obs_tuple = env.reset(seed=initial_state_seed)
    obs: Mapping[str, Any] = obs_tuple[0]
    partner.reset(seed=initial_state_seed)
    total_reward = 0.0
    n_steps = 0
    terminated = False
    truncated = False
    while n_steps < max_steps:
        partner_action = partner.act(obs, deterministic=True)
        ego_action_vec = ego_action(obs)
        action_dict = {ego_uid: ego_action_vec, partner_uid: partner_action}
        step_out = env.step(action_dict)
        obs = step_out[0]
        reward = float(step_out[1])
        terminated = bool(step_out[2])
        truncated = bool(step_out[3])
        total_reward += reward
        n_steps += 1
        if terminated or truncated:
            break
    if not terminated and not truncated:
        # The loop exhausted the eval horizon without the env emitting a
        # boundary: the episode was truncated by the eval cap. Record it
        # honestly so truncation-based audits stay correct — the prior
        # code left both flags False here, masking the horizon cap
        # (P1.05.9 firing #3; ADR-007 §Stage 1b).
        truncated = True
    mean_reward = total_reward / max(1, n_steps)
    # ADR-007 §Stage 1b Rev 12 / P1.05.8 (closes Surface 1): the real
    # Stage-1 pick-place env exposes ``terminated=True`` iff
    # ``evaluate()`` returns ``is_obj_placed & is_robot_static``; that
    # is the only honest success signal here. The pre-P1.05.8 rule
    # (``mean_reward > -0.30 or terminated``) was a Phase-0 MPE
    # holdover — ManiSkill's small-positive per-step rewards trivially
    # cleared the ``> -0.30`` threshold, producing 100 % rubber-stamp
    # passes that masked the Surface-6 obs-contract defect. Stage-1a's
    # MPE stand-in does NOT emit ``terminated=True`` on success, so
    # Stage-1a success rates become structurally zero under this rule;
    # that is correct per ADR-007 §Stage 1a (Stage 1a is rig-validation,
    # not the ≥20 pp gate). ``mean_reward`` and ``n_steps`` stay in
    # metadata as diagnostic fields; ``terminated`` / ``truncated`` are
    # also surfaced so post-run audits can distinguish the
    # ``evaluate()``-fired-on-final-step case from a truncation-only
    # episode.
    success = bool(terminated) and not bool(truncated)
    return EpisodeResult(
        seed=seed,
        episode_idx=episode_idx,
        initial_state_seed=initial_state_seed,
        success=success,
        metadata={
            "condition": condition_id,
            "mean_reward": f"{mean_reward:.4f}",
            "n_steps": str(n_steps),
            "terminated": str(bool(terminated)).lower(),
            "truncated": str(bool(truncated)).lower(),
        },
    )


def _default_env_factory(
    condition_id: str, agent_uids: tuple[str, str], root_seed: int
) -> gym.Env[Any, Any]:
    """Build the per-condition env (Stage-1a MPE stand-in; ADR-007 §Stage 1a).

    The real Stage-1 pick-place env tuples (panda-only vs panda+fetch
    on shared pick-place) live in the Stage-1b path —
    see :func:`_stage1b_env_factory` for the Phase-1 dispatch. For
    Stage-1a the adapter uses
    :class:`chamber.envs.mpe_cooperative_push.MPECooperativePushEnv`
    so the SpikeRun shape, the partner-stack contract, and the
    chamber-side dispatch are all exercisable on CPU. The
    ``agent_uids`` tuple encodes the AS distinction at the obs / action
    keying level (the only project surface the partner stack and
    chamber-eval bootstrap actually see).
    """
    del condition_id  # tuple encodes the distinction; module docstring explains.
    return MPECooperativePushEnv(agent_uids=agent_uids, root_seed=root_seed)


def _stage1b_env_factory(
    condition_id: str, agent_uids: tuple[str, str], root_seed: int
) -> gym.Env[Any, Any]:
    """Build the per-condition Stage-1b env (P1.05; ADR-007 §Stage 1b).

    Routes through
    :func:`chamber.envs.stage1_pickplace.make_stage1_pickplace_env` so
    the env's own ``_CONDITION_TABLE`` is the single source of truth
    for the AS-homo (panda + panda_partner) vs AS-hetero (panda +
    fetch) agent tuple. The ``agent_uids`` argument is unused — the
    env-side table is canonical; resolving from two tables would
    silently allow drift. Asserted equal for defensive parity.

    SAPIEN / ManiSkill imports are deferred to the inner factory body,
    so this module's top-level imports stay Tier-1-safe per ADR-001.

    Args:
        condition_id: One of the four Stage-1 ``condition_id`` strings.
        agent_uids: ``(ego_uid, partner_uid)`` tuple resolved by the
            chamber-side ``_CONDITION_UIDS`` lookup. Asserted to match
            the env-side ``_CONDITION_TABLE`` entry.
        root_seed: Project root seed routed through
            :func:`concerto.training.seeding.derive_substream` for P6
            reproducibility.

    Returns:
        :class:`Stage1PickPlaceEnv` instance, ready to call
        ``reset(seed=K)`` on.
    """
    from chamber.envs.stage1_pickplace import (
        _CONDITION_TABLE,
        make_stage1_pickplace_env,
    )

    expected_uids = _CONDITION_TABLE[condition_id].agent_uids
    if expected_uids != agent_uids:
        msg = (
            f"_stage1b_env_factory: condition_id {condition_id!r} env-side "
            f"agent_uids {expected_uids} != adapter-side {agent_uids}. "
            "The two _CONDITION_UIDS tables have drifted; the env-side "
            "table in chamber.envs.stage1_pickplace is canonical "
            "(ADR-007 §Discipline)."
        )
        raise ValueError(msg)
    return make_stage1_pickplace_env(condition_id=condition_id, root_seed=root_seed)


def _resolve_partner_action_dim(
    env: gym.Env[Any, Any],
    *,
    partner_uid: str,
    sub_stage: SubStage,
) -> int:
    """Resolve the scripted partner's action dim (P1.05; ADR-007 §Stage 1b).

    Stage-1a (MPE): all partners are 2-D — keeps the Phase-0 default.
    Stage-1b: read ``env.action_space.spaces[partner_uid].shape[0]``
    so AS-homo gets 8 (panda_partner under ``pd_joint_delta_pos``) and
    AS-hetero / OM-* get the fetch action shape. Falls back to the
    Phase-0 default of 2 if the env's action_space is not a
    :class:`gymnasium.spaces.Dict` (Tier-1 fake envs).
    """
    if sub_stage == "1a":
        return 2
    action_space = env.action_space
    if not isinstance(action_space, gym.spaces.Dict):
        return 2
    sub = action_space.spaces.get(partner_uid)
    if sub is None or not isinstance(sub, gym.spaces.Box):
        return 2
    if sub.shape is None:
        return 2
    return int(sub.shape[0])


def _make_scripted_partner(*, partner_uid: str, action_dim: int = 2) -> ScriptedHeuristicPartner:
    """Construct the scripted-heuristic partner with the right uid (plan/04 §3.4).

    Stage-1a keeps the Phase-0 default ``action_dim=2`` (MPE stand-in).
    Stage-1b passes the resolved per-condition action dim (8 for
    panda_partner, 13 for fetch) so the partner's act() returns a
    correctly-sized vector that the ManiSkill v3 step boundary accepts.
    """
    spec = PartnerSpec(
        class_name="scripted_heuristic",
        seed=0,
        checkpoint_step=None,
        weights_uri=None,
        extra={
            "uid": partner_uid,
            "target_xy": "0.0,0.0",
            "action_dim": str(action_dim),
        },
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
