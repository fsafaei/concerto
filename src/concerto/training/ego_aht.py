# SPDX-License-Identifier: Apache-2.0
"""Ego-AHT training loop entry point (T4b.11; ADR-002 §Decisions; plan/05 §3.5).

The single user-facing API for an ego-AHT training run is :func:`train`.
It wires up logging, seeding, per-step rollout collection, checkpoint
emission, and the trainer-factory seam, and returns a :class:`RewardCurve`
for T4b.13's empirical-guarantee assertion.

Dependency-direction discipline (project plan/10 §2 dependency-direction rule):
``concerto.*`` MUST NOT import from ``chamber.*``. The training loop
therefore takes the env and partner via dependency injection — the
chamber-side runner (``chamber.benchmarks.training_runner``) builds the
concrete env (:class:`chamber.envs.mpe_cooperative_push.MPECooperativePushEnv`)
and partner (via :func:`chamber.partners.registry.load_partner`) and
hands them to :func:`train`. That keeps every ``concerto.*`` arrow
pointing only at lower layers.

Trainer-factory injection: this loop lives in ``concerto.*`` and so cannot
import from ``chamber.*`` (project plan/10 §2). The real ego-PPO trainer
(:class:`chamber.benchmarks.ego_ppo_trainer.EgoPPOTrainer`) wraps HARL's
HAPPO and lives on the chamber side; the chamber-side
:func:`chamber.benchmarks.training_runner.run_training` selects it as the
default. :func:`train` therefore continues to accept ``trainer_factory``
and falls back to :class:`RandomEgoTrainer` when called directly without
an injected factory — which keeps the loop unit-testable inside
``concerto.*`` without violating the dependency direction.

ADR-002 risk-mitigation #1: this loop is the substrate the empirical-
guarantee assertion runs against. Do not silently drop steps, mutate
the rollout collection mid-loop, or change the partner-frozen contract
without a new ADR.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol, cast, runtime_checkable

import numpy as np

from concerto.training.checkpoints import CheckpointMetadata, save_checkpoint
from concerto.training.logging import bind_run_logger, compute_run_metadata
from concerto.training.seeding import derive_substream

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Mapping

    import structlog
    from numpy.typing import NDArray

    from concerto.safety.api import Bounds, EgoOnlySafetyFilter, SafetyState
    from concerto.safety.cbf_qp import AgentSnapshot
    from concerto.safety.emergency import EmergencyController
    from concerto.training.config import EgoAHTConfig

#: Fallback ego-action dim used by the default :class:`RandomEgoTrainer`
#: when :func:`train` is called without an explicit ``trainer_factory``.
#:
#: Matches :class:`chamber.envs.mpe_cooperative_push.MPECooperativePushEnv`'s
#: action shape. The real :class:`EgoPPOTrainer
#: <chamber.benchmarks.ego_ppo_trainer.EgoPPOTrainer>` reads the action
#: dim from the env directly, so this constant is only the Phase-0
#: ``RandomEgoTrainer`` fallback used when :func:`train` is called
#: without an injected ``trainer_factory``.
_RANDOM_TRAINER_FALLBACK_ACTION_DIM: int = 2


@runtime_checkable
class EnvLike(Protocol):
    """Structural Gymnasium-multi-agent env contract the loop needs (T4b.11; ADR-002 §Decisions).

    Defined locally so :func:`train` does not import a concrete class
    from ``chamber.*`` (project plan/10 §2 dependency-direction rule). The
    chamber-side :class:`chamber.envs.mpe_cooperative_push.MPECooperativePushEnv`
    satisfies this Protocol structurally.
    """

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        """Reset to the start of an episode (T4b.11; ADR-002 §Decisions)."""
        ...  # pragma: no cover

    def step(
        self,
        action: dict[str, NDArray[np.floating]],
    ) -> tuple[Mapping[str, Any], float, bool, bool, Mapping[str, Any]]:
        """Advance one tick (T4b.11; ADR-002 §Decisions).

        Takes a concrete ``dict`` rather than a ``Mapping`` because the
        chamber-side :class:`MPECooperativePushEnv` and the future
        ManiSkill v3 wrappers all require dict-keyed access (mutation
        of the action dict is intentional in some wrappers' rate-limit
        / collision-injection paths).
        """
        ...  # pragma: no cover


@runtime_checkable
class PartnerLike(Protocol):
    """Structural ``FrozenPartner`` contract the loop needs (T4b.11; ADR-009 §Decision).

    Defined locally to keep ``concerto.*`` free of ``chamber.*`` imports.
    The M4a :class:`chamber.partners.api.FrozenPartner` satisfies this
    Protocol structurally — both the heuristic + the future frozen-RL
    adapters drop in unchanged.

    The ``spec`` attribute is declared as :class:`Any` so concrete
    implementations can attach any spec type (M4a's
    :class:`chamber.partners.api.PartnerSpec`); the loop only uses it
    for diagnostic logging, never structurally. ``Any`` is required
    here rather than ``object`` because Protocol attributes are
    invariant and concrete subtypes (PartnerSpec) are not assignable to
    a mutable ``object`` annotation.
    """

    spec: Any

    def reset(self, *, seed: int | None = None) -> None:
        """Reset internal episode state (ADR-009 §Decision)."""
        ...  # pragma: no cover

    def act(
        self,
        obs: Mapping[str, Any],
        *,
        deterministic: bool = True,
    ) -> NDArray[np.floating]:
        """Pick an action for the partner's own uid (ADR-009 §Decision)."""
        ...  # pragma: no cover


class EgoTrainer(Protocol):
    """Minimal contract the training loop expects of an ego trainer (T4b.11; ADR-002 §Decisions).

    Two concrete implementations satisfy this Protocol:

    - :class:`chamber.benchmarks.ego_ppo_trainer.EgoPPOTrainer` — the
      real HARL-HAPPO-backed ego-PPO trainer (M4b-8a / plan/05 §3.5),
      and the default selected by
      :func:`chamber.benchmarks.training_runner.run_training`.
    - :class:`RandomEgoTrainer` — the Phase-0 reference fallback, used
      by :func:`train` when no ``trainer_factory`` is injected.

    The Protocol is intentionally tiny (:meth:`act` + :meth:`observe` +
    :meth:`update` + :meth:`state_dict`) so the seam stays small enough
    that an external user can plug in their own algorithm without
    changing :func:`train`.
    """

    def act(
        self,
        obs: Mapping[str, Any],
        *,
        deterministic: bool = False,
    ) -> NDArray[np.floating]:
        """Sample (or pick deterministically) the ego action (ADR-002 §Decisions)."""
        ...  # pragma: no cover

    def observe(
        self,
        obs: Mapping[str, Any],
        reward: float,
        done: bool,
        *,
        truncated: bool = False,
    ) -> None:
        """Record a (s, a, r, s', done, truncated) tuple (ADR-002 §Decisions; Pardo 2017).

        ``done`` keeps the historical "this step ended the episode"
        meaning (terminated OR truncated). ``truncated`` is a
        keyword-only flag that distinguishes time-limit truncation from
        true termination — needed by PPO-style trainers to bootstrap the
        GAE target correctly (see :func:`chamber.benchmarks.ego_ppo_trainer.compute_gae`
        and project issue #62 for the root-cause writeup). Default
        ``truncated=False`` keeps legacy callers' behavior unchanged.
        """
        ...  # pragma: no cover

    def update(self) -> None:
        """Run one optimisation epoch on the buffered rollout (ADR-002 §Decisions)."""
        ...  # pragma: no cover

    def state_dict(self) -> dict[str, Any]:
        """Return the trainer's torch state-dict for checkpointing (ADR-002 §Decisions)."""
        ...  # pragma: no cover


class TrainerFactory(Protocol):
    """Constructor that builds an :class:`EgoTrainer` (T4b.11; ADR-002 §Decisions).

    The seam through which :func:`train` plugs in the algorithm.
    :func:`chamber.benchmarks.training_runner.run_training` selects
    :meth:`EgoPPOTrainer.from_config <chamber.benchmarks.ego_ppo_trainer.EgoPPOTrainer.from_config>`
    by default (M4b-8a / plan/05 §3.5).

    The factory receives the partner so it can refuse to construct
    against a non-frozen partner before any tensor allocation
    (ADR-009 §Consequences; plan/05 §6 #3). The default factory
    :meth:`~chamber.benchmarks.ego_ppo_trainer.EgoPPOTrainer.from_config`
    calls :func:`~chamber.benchmarks.ego_ppo_trainer._assert_partner_is_frozen`
    on the partner as its first construction step.
    """

    def __call__(
        self,
        cfg: EgoAHTConfig,
        *,
        env: EnvLike,
        partner: PartnerLike,
        ego_uid: str,
        logger: structlog.BoundLogger | None = None,
    ) -> EgoTrainer:
        """Build a trainer (ADR-002 §Decisions; ADR-017; plan/05 §3.5; ADR-009 §Consequences).

        ``logger`` is the optional structlog BoundLogger from
        :func:`concerto.training.logging.bind_run_logger` (P1.05.11 /
        ADR-017 §Decisions). When supplied, concrete trainers like
        :class:`chamber.benchmarks.ego_ppo_trainer.EgoPPOTrainer`
        thread it through their constructor so PPO update scalars
        flow through the same JSONL + W&B sinks as the driver's
        own events. Default ``None`` preserves pre-P1.05.11 silent
        behaviour.
        """
        ...  # pragma: no cover


@dataclass(frozen=True)
class RewardCurve:
    """Result of one ego-AHT training run (T4b.11; T4b.13 consumer; ADR-002 §Decisions).

    Attributes:
        run_id: 16-hex run identifier (from
            :class:`concerto.training.logging.RunContext`).
        per_step_ego_rewards: One entry per env step. T4b.13's empirical-
            guarantee assertion runs over the moving-window-of-10 mean
            of this series.
        per_episode_ego_rewards: Sum-of-step reward per episode. Coarser
            than ``per_step``; the canonical "epoch reward" the
            empirical-guarantee plot embeds in
            ``docs/explanation/why-aht.md``.
        checkpoint_paths: Absolute paths of every ``.pt`` artefact saved
            during the run. T4b.14 reads the canonical 50%-step file
            from this list.
    """

    run_id: str
    per_step_ego_rewards: list[float] = field(default_factory=list)
    per_episode_ego_rewards: list[float] = field(default_factory=list)
    checkpoint_paths: list[Path] = field(default_factory=list)


class TrainingResult(NamedTuple):
    """Return value of :func:`train` and :func:`chamber.benchmarks.training_runner.run_training`.

    Pairs the diagnostic reward stream (:class:`RewardCurve`) with the
    trained ego policy (:class:`EgoTrainer`) so Phase-1 callers like
    :class:`chamber.benchmarks.stage1_common.TrainedPolicyFactory`
    (P1.04 / ADR-007 §Stage 1b) can wrap ``result.trainer.act`` in a
    per-step closure without paying a checkpoint round-trip cost. The
    trainer's internal state (HARL HAPPO actor + critic + optimisers)
    survives :func:`train`'s scope through this return value rather
    than being reloaded from disk.

    NamedTuple (not a bare ``tuple[RewardCurve, EgoTrainer]``) so call
    sites use the field names — ``result.curve`` /
    ``result.trainer`` — instead of positional indexing. Tuple-unpack
    still works (``curve, trainer = run_training(cfg)``) for callers
    that prefer it.

    Attributes:
        curve: The diagnostic per-step + per-episode reward stream
            T4b.13's empirical-guarantee assertion runs over. Same
            shape as before P1.04 — this NamedTuple is a wrapper, not
            a replacement.
        trainer: The trained :class:`EgoTrainer` instance, with the
            HARL HAPPO actor weights at their post-training state. Use
            ``trainer.act(obs, deterministic=True)`` for evaluation
            rollouts. When :func:`train` is called without a
            ``trainer_factory`` (Phase-0 reference path), this is the
            :class:`RandomEgoTrainer` fallback (parameter-free).
    """

    curve: RewardCurve
    trainer: EgoTrainer


class RandomEgoTrainer:
    """Reference :class:`EgoTrainer` that samples uniformly from the action space.

    Exists so :func:`train` can be exercised end-to-end before the HARL
    fork lands (T4b.4-T4b.7). Not a real learner — :meth:`update` is a
    no-op, :meth:`state_dict` returns an empty dict — but the loop's
    plumbing (env stepping, partner act, checkpoint emission, JSONL
    logging) is fully exercised. T4b.13 will swap to ``EgoAHTHAPPO``
    via the trainer-factory seam (ADR-002 §Decisions; plan/05 §3.5).
    """

    def __init__(self, *, ego_uid: str, action_dim: int, root_seed: int) -> None:
        """Build a uniform-random reference trainer (T4b.11; ADR-002 §Decisions).

        Args:
            ego_uid: Stored for downstream identity propagation (the
                trainer doesn't read it at action time but the loop
                uses it as the dict-action key).
            action_dim: Length of the ego's action vector.
            root_seed: Project seed; the trainer's RNG is derived via
                :func:`concerto.training.seeding.derive_substream` for
                P6 reproducibility.
        """
        self._ego_uid = ego_uid
        self._action_dim = action_dim
        self._rng: np.random.Generator = derive_substream(
            "training.ego_random", root_seed=root_seed
        ).default_rng()

    def act(
        self,
        obs: Mapping[str, Any],
        *,
        deterministic: bool = False,
    ) -> NDArray[np.floating]:
        """Sample uniformly from [-1, 1]^action_dim (ADR-002 §Decisions; plan/05 §3.5)."""
        del obs
        if deterministic:
            return np.zeros(self._action_dim, dtype=np.float32)
        return self._rng.uniform(-1.0, 1.0, size=self._action_dim).astype(np.float32)

    def observe(
        self,
        obs: Mapping[str, Any],
        reward: float,
        done: bool,
        *,
        truncated: bool = False,
    ) -> None:
        """No-op (ADR-002 §Decisions): reference trainer has no rollout buffer."""
        del obs, reward, done, truncated

    def update(self) -> None:
        """No-op (ADR-002 §Decisions): reference trainer has no learnable params."""

    def state_dict(self) -> dict[str, Any]:
        """Return an empty dict (ADR-002 §Decisions: trainer is parameter-free)."""
        return {}


def train(  # noqa: PLR0912, PLR0915 - P1.04.5 added safety-stack integration to the canonical training loop; the body's branch + statement counts exceed the default thresholds because the safety path is interleaved per step. Refactoring into a SafetyFilterDriver collaborator is the Phase-2 cleanup path (see Plan-subagent D1 alternative).
    cfg: EgoAHTConfig,
    *,
    env: EnvLike,
    partner: PartnerLike,
    trainer_factory: TrainerFactory | None = None,
    repo_root: Path | None = None,
    # ----- P1.04.5: safety-stack wiring (ADR-007 §Stage 1b). -----
    # All five default to ``None``; absence means pre-P1.04.5 behaviour
    # (the loop runs unfiltered against the partner). Presence requires
    # all five together — validated below. ``cfg.safety.enabled`` is the
    # operator-intent toggle; cfg-says-enabled-but-kwargs-not-passed and
    # cfg-says-disabled-but-kwargs-passed both loud-fail (intent
    # mismatch).
    safety_filter: EgoOnlySafetyFilter | None = None,
    safety_state: SafetyState | None = None,
    safety_bounds: Bounds | None = None,
    safety_snapshot_builder: (Callable[[EnvLike], dict[str, AgentSnapshot]] | None) = None,
    safety_dt: float | None = None,
    # ----- P1.04.6: braking-fallback parity (ADR-007 §Stage 1b Rev 8). -----
    # ``safety_emergency_controllers`` is the per-uid emergency-controller
    # dispatch map :func:`concerto.safety.braking.maybe_brake` consults.
    # When ``None`` the braking layer falls back to a per-uid
    # :class:`CartesianAccelEmergencyController` — correct for
    # double-integrator agents but a silent dimension-mismatch foot-gun
    # for 7-DOF arms; the chamber-side caller
    # (:class:`chamber.benchmarks.stage1_common.TrainedPolicyFactory`)
    # populates it via
    # :func:`chamber.benchmarks.training_runner.build_emergency_controllers`.
    # Read only when ``cfg.safety.enabled=True`` AND the other safety
    # kwargs are passed (the validation block below ties presence to
    # ``_safety_active``).
    safety_emergency_controllers: Mapping[str, EmergencyController] | None = None,
) -> TrainingResult:
    """Run one ego-AHT training loop (T4b.11; ADR-002 §Decisions; plan/05 §3.5).

    Wires up logging, seeding, per-step rollout, checkpoint emission, and
    the trainer-factory seam. Returns a :class:`TrainingResult` (P1.04 /
    ADR-007 §Stage 1b) carrying both the diagnostic
    :class:`RewardCurve` and the trained :class:`EgoTrainer` instance,
    so Phase-1 callers (notably
    :class:`chamber.benchmarks.stage1_common.TrainedPolicyFactory`) can
    wrap ``result.trainer.act`` in a per-cell closure without paying
    a checkpoint round-trip.

    The pre-P1.04 return type was the bare :class:`RewardCurve`; the
    NamedTuple wrapper is backward-compatible at the tuple-unpack level
    (``curve, trainer = train(...)``) but reads more cleanly at named
    call sites (``result.curve`` / ``result.trainer``).

    Args:
        cfg: Validated :class:`~concerto.training.config.EgoAHTConfig`.
        env: Concrete env instance satisfying the :class:`EnvLike`
            structural Protocol. Built upstream by
            :func:`chamber.benchmarks.training_runner.build_env`.
        partner: Concrete frozen partner satisfying :class:`PartnerLike`.
            Built upstream by
            :func:`chamber.benchmarks.training_runner.build_partner`.
        trainer_factory: Constructor that builds the
            :class:`EgoTrainer`. When ``None`` (default), uses
            :class:`RandomEgoTrainer` — the Phase-0 in-``concerto.*``
            fallback. The chamber-side wrapper
            :func:`chamber.benchmarks.training_runner.run_training`
            injects the M4b-8a
            :class:`~chamber.benchmarks.ego_ppo_trainer.EgoPPOTrainer`
            (the real HARL-HAPPO trainer; plan/05 §3.5) here.
        repo_root: Working-tree root for the run-metadata bundle
            (defaults to :func:`pathlib.Path.cwd`).
        safety_filter: Optional CBF-QP outer filter (P1.04.5;
            ADR-007 §Stage 1b). All five safety kwargs must be passed
            together when ``cfg.safety.enabled=True``; passing any
            with the cfg disabled (or partial sets) raises
            :class:`ValueError` (intent-mismatch contract).
        safety_state: Optional :class:`SafetyState` — the conformal
            slack vector (mutated in place per step by
            :func:`update_lambda_from_predictor`).
        safety_bounds: Optional per-task :class:`Bounds` — the
            cell's CBF/Cartesian envelope (audit-gate predicate A's
            RHS is sourced from ``cartesian_accel_capacity``).
        safety_snapshot_builder: Optional ``(env) -> dict[str,
            AgentSnapshot]`` callable that the loop invokes per step
            to build the per-uid Cartesian snapshot map the filter
            consumes via ``partner_predicted_states``.
        safety_dt: Optional control-step dt in seconds for the
            constant-velocity predictor lookahead + numerical-
            difference velocity. Typically ``env.control_timestep``.
        safety_emergency_controllers: Optional per-uid
            :class:`concerto.safety.emergency.EmergencyController`
            dispatch map (P1.04.6; ADR-007 §Stage 1b Rev 8). Consulted
            by :func:`concerto.safety.braking.maybe_brake` at each
            per-step braking call to translate the aggregate Cartesian
            repulsion into the per-uid control-space override.
            ``None`` falls back to a per-uid
            :class:`concerto.safety.emergency.CartesianAccelEmergencyController`
            — correct for double-integrator agents but a silent
            dimension-mismatch foot-gun for 7-DOF arms. Wired by
            :class:`chamber.benchmarks.stage1_common.TrainedPolicyFactory`
            via
            :func:`chamber.benchmarks.training_runner.build_emergency_controllers`.
            Honoured only when ``cfg.safety.enabled=True`` AND the rest
            of the safety stack is wired.

    Returns:
        :class:`TrainingResult` NamedTuple carrying ``curve`` (the
        :class:`RewardCurve` with per-step + per-episode ego rewards
        and saved checkpoint paths) and ``trainer`` (the trained
        :class:`EgoTrainer` instance whose ``act`` method evaluation
        rollouts wrap).
    """
    repo_root = repo_root or Path.cwd()
    ctx = compute_run_metadata(
        seed=cfg.seed,
        run_kind=cfg.algo,
        repo_root=repo_root,
        extra={
            "task": cfg.env.task,
            "partner_class": cfg.partner.class_name,
        },
    )
    jsonl_path = cfg.log_dir / f"{ctx.run_id}.jsonl"
    logger = bind_run_logger(ctx, jsonl_path=jsonl_path)
    logger.info(
        "training_start",
        total_frames=cfg.total_frames,
        checkpoint_every=cfg.checkpoint_every,
    )

    ego_uid, partner_uid = cfg.env.agent_uids
    if trainer_factory is None:
        trainer: EgoTrainer = RandomEgoTrainer(
            ego_uid=ego_uid,
            action_dim=_RANDOM_TRAINER_FALLBACK_ACTION_DIM,
            root_seed=cfg.seed,
        )
    else:
        # P1.05.11 / ADR-017: thread the bound logger into the trainer
        # factory so PPO update scalars flow through the same JSONL +
        # W&B sinks as the driver's own events. Factories that do not
        # accept ``logger`` (legacy / external) are still supported —
        # the kwarg is optional at the Protocol level.
        trainer = trainer_factory(
            cfg,
            env=env,
            partner=partner,
            ego_uid=ego_uid,
            logger=logger,
        )

    # P1.04.5: validate the safety-stack kwargs against cfg.safety.enabled
    # (ADR-007 §Stage 1b lifecycle contract). Intent-mismatch loud-fails:
    # cfg-says-enabled-but-some-kwarg-None and cfg-says-disabled-but-some-
    # kwarg-not-None both raise. Both-aligned paths (all five None when
    # disabled; all five non-None when enabled) proceed.
    _safety_kwargs_passed = (
        safety_filter is not None
        and safety_state is not None
        and safety_bounds is not None
        and safety_snapshot_builder is not None
        and safety_dt is not None
    )
    _safety_kwargs_partial = (
        safety_filter is not None
        or safety_state is not None
        or safety_bounds is not None
        or safety_snapshot_builder is not None
        or safety_dt is not None
    ) and not _safety_kwargs_passed
    if _safety_kwargs_partial:
        msg = (
            "train(): safety-stack kwargs must be passed as a complete set "
            "(safety_filter + safety_state + safety_bounds + "
            "safety_snapshot_builder + safety_dt) or all None. Partial "
            "wiring is an operator-intent mismatch."
        )
        raise ValueError(msg)
    if cfg.safety.enabled and not _safety_kwargs_passed:
        msg = (
            "train(): cfg.safety.enabled is True but the safety-stack kwargs "
            "were not passed. The chamber-side caller "
            "(chamber.benchmarks.stage1_common.TrainedPolicyFactory) wires "
            "them; tests that invoke train() directly with cfg.safety.enabled "
            "must construct the SafetyState + filter + bounds + snapshot "
            "builder + dt themselves (see "
            "chamber.benchmarks.training_runner.build_safety_*)."
        )
        raise ValueError(msg)
    if not cfg.safety.enabled and _safety_kwargs_passed:
        msg = (
            "train(): cfg.safety.enabled is False but safety-stack kwargs "
            "were passed. Either set cfg.safety.enabled=True (and run with "
            "the filter) or drop the kwargs (and run unfiltered). Mixed "
            "intent is loud-fail by design (ADR-007 §Stage 1b discipline)."
        )
        raise ValueError(msg)
    _safety_active: bool = cfg.safety.enabled and _safety_kwargs_passed

    obs, _ = env.reset(seed=cfg.seed)
    partner.reset(seed=cfg.seed)
    curve = RewardCurve(run_id=ctx.run_id)
    episode_reward_acc = 0.0
    episode_in_progress = True
    # Episode index is incremented on each terminated/truncated boundary,
    # NOT inferred from step // episode_length — the latter would silently
    # misnumber for envs that early-terminate (P6 anti-pattern flagged in
    # local pre-PR review).
    episode_index = 0

    # P1.04.5: safety-stack runtime state.
    aggregator = None
    snaps_prev: dict[str, AgentSnapshot] | None = None
    # The validation block above guarantees all five safety kwargs are
    # non-None iff _safety_active is True. The casts below narrow for
    # pyright across the loop body (S101 avoided by not using ``assert``).
    _filter = cast("EgoOnlySafetyFilter", safety_filter)
    _state = cast("SafetyState", safety_state)
    _bounds = cast("Bounds", safety_bounds)
    _builder = cast("Callable[[EnvLike], dict[str, AgentSnapshot]]", safety_snapshot_builder)
    _dt = cast("float", safety_dt)
    # Lazy imports to keep concerto.training.ego_aht Tier-1-import-safe;
    # the safety_telemetry module only resolves when train() is called,
    # not at concerto.training import time. ADR-004 §Decision: the
    # safety stack is itself a concerto.* module, so this keeps train()
    # importable for consumers that don't pull in safety_telemetry.
    # Imports are unconditional (not inside `if _safety_active:`) so
    # the symbols are bound for pyright's flow analysis across the
    # loop body; their *use* is still guarded by `_safety_active`.
    from concerto.safety.braking import maybe_brake  # noqa: PLC0415
    from concerto.safety.conformal import (  # noqa: PLC0415 - lazy by design (see comment above)
        constant_velocity_predict,
        update_lambda_from_predictor,
    )
    from concerto.safety.errors import ConcertoSafetyInfeasible  # noqa: PLC0415
    from concerto.training.safety_telemetry import SafetyAggregator  # noqa: PLC0415

    if _safety_active:
        _filter.reset(seed=cfg.seed)
        # P1.05.7 / #180: emit lambda_clamp_bound so the audit-gate hook
        # can distinguish the clamp-saturated regime (var=0 by design)
        # from "adapted but stuck" (var=0 by degenerate update).
        aggregator = SafetyAggregator(
            n_pairs=len(_state.lambda_),
            cartesian_accel_capacity=_bounds.cartesian_accel_capacity,
            saturation_threshold=cfg.safety.saturation_threshold,
            lambda_clamp_bound=cfg.safety.clamp_floor_ratio * _bounds.cartesian_accel_capacity,
        )

    for step in range(cfg.total_frames):
        ego_action_nominal = trainer.act(obs)
        partner_action = partner.act(obs)
        if _safety_active and aggregator is not None:
            snaps_now = _builder(env)
            # Conformal update against the previous step's snapshots
            # (skipped on first step + at episode boundaries, where
            # snaps_prev is None — the predictor cannot meaningfully
            # forecast across a reset).
            if snaps_prev is not None:
                # P1.05.7 / issue #180: symmetric clamp on λ. The bound
                # is ``clamp_floor_ratio x cartesian_accel_capacity``;
                # the audit-gate predicate A trips at
                # ``saturation_threshold x cap`` (the SafetyConfig
                # @model_validator guarantees the strict inequality
                # so the buffer is non-zero). Without the clamp the
                # negative-eps regime drifts unboundedly negative —
                # P1.05 100k-frame AS-hetero probe measured drift
                # exactly -η x |ε| = -5e-04 / step, projecting
                # λ_ss ≈ -49.76 at the production budget.
                update_lambda_from_predictor(
                    _state,
                    snaps_now=snaps_now,
                    snaps_prev=snaps_prev,
                    alpha_pair=2.0 * _bounds.cartesian_accel_capacity,
                    gamma=cfg.safety.cbf_gamma,
                    dt=_dt,
                    in_warmup=(_state.warmup_steps_remaining > 0),
                    lambda_bound=cfg.safety.clamp_floor_ratio * _bounds.cartesian_accel_capacity,
                )
            # Forecast partner state for the next control step
            # (constant-velocity stub per ADR-004 §Decision).
            partner_predicted = {
                uid: constant_velocity_predict(snaps_now[uid], _dt)
                for uid in snaps_now
                if uid != ego_uid
            }
            # P1.04.6 (ADR-007 §Stage 1b Rev 8): per-step braking
            # fallback in front of the CBF-QP — matches the deployment-
            # time composition order in
            # ``tests/integration/test_safety_in_loop.py``. Independent
            # of QP feasibility (Wang-Ames-Egerstedt 2017 eq. 17). We
            # build a proposed-action dict over BOTH uids so
            # :func:`maybe_brake` evaluates pairwise TTC over the cell,
            # but only adopt the override for ``ego_uid`` — the partner
            # is frozen per ADR-009 §Decision and its action stays
            # untouched. ``safe_ego`` is what propagates into the
            # trainer's learning signal.
            proposed_for_braking = {
                ego_uid: np.asarray(ego_action_nominal, dtype=np.float64),
                partner_uid: np.asarray(partner_action, dtype=np.float64),
            }
            braking_override, braking_fired = maybe_brake(
                proposed_for_braking,
                snaps_now,
                bounds=_bounds,
                tau_brake=cfg.safety.tau_brake,
                emergency_controllers=safety_emergency_controllers,
            )
            if braking_fired and braking_override is not None:
                # Per-step backstop fired: take the ego override and
                # skip the QP for this step. ``fallback_fired`` (the
                # CBF-QP's internal 1-D projection fallback flag) stays
                # ``False`` — those two flags are independent telemetry
                # channels on the audit-gate (one is "QP ran and used
                # its internal fallback"; the other is "QP did not run,
                # braking took over"). ``qp_infeasible`` is also False:
                # the QP did not raise, it simply was not called.
                ego_action = np.asarray(braking_override[ego_uid], dtype=ego_action_nominal.dtype)
                fallback_fired = False
                qp_infeasible = False
            else:
                try:
                    safe_ego, info = _filter.filter(
                        np.asarray(ego_action_nominal, dtype=np.float64),
                        obs={
                            "agent_states": snaps_now,
                            "meta": {"partner_id": cfg.partner.class_name},
                        },
                        state=_state,
                        bounds=_bounds,
                        ego_uid=ego_uid,
                        partner_predicted_states=partner_predicted,
                        dt=_dt,
                    )
                    ego_action = np.asarray(safe_ego, dtype=ego_action_nominal.dtype)
                    fallback_fired = bool(info["fallback_fired"])
                    qp_infeasible = False
                except ConcertoSafetyInfeasible:
                    # Pre-P1.04.6 training-time policy was to pass the
                    # nominal action through; P1.04.6's braking layer
                    # now runs upstream of this branch, so reaching
                    # here means the QP raised AFTER braking didn't
                    # fire (i.e. TTC >= tau_brake but the QP still
                    # found no feasible projection — a tighter QP
                    # constraint than the braking TTC predicts).
                    # Surface as the unfiltered-passthrough still, so
                    # the trainer can learn from the consequence; the
                    # ``qp_infeasible`` flag is what audits this.
                    ego_action = ego_action_nominal
                    fallback_fired = False
                    qp_infeasible = True
            snaps_prev = snaps_now
            aggregator.observe(
                _state.lambda_,
                fallback_fired=fallback_fired,
                qp_infeasible=qp_infeasible,
                braking_fired=braking_fired,
            )
        else:
            ego_action = ego_action_nominal
        action = {ego_uid: ego_action, partner_uid: partner_action}
        # The trainer.observe(s') contract: this loop passes the *next*
        # observation (the post-step obs), reward, and done flag. M4b-7's
        # HARL trainer is responsible for buffering pre-step obs internally
        # by tracking the last act() input. Phase-0 simplification per
        # plan/05 §3.5; revisit if a future on-policy trainer needs the
        # explicit (s, a, r, s', done) tuple.
        obs, reward, terminated, truncated, _ = env.step(action)
        curve.per_step_ego_rewards.append(reward)
        episode_reward_acc += reward
        # Pardo 2017 / issue #62: thread terminated AND truncated
        # separately. ``done`` keeps the historical episode-boundary
        # meaning (terminated OR truncated) so the loop's reset logic is
        # unchanged; ``truncated`` is the kwarg-only flag PPO-style
        # trainers use to bootstrap GAE correctly at time-limit
        # boundaries.
        trainer.observe(obs, reward, terminated or truncated, truncated=truncated)

        if (step + 1) % cfg.happo.rollout_length == 0:
            trainer.update()
            if aggregator is not None:
                window = aggregator.flush_window_stats()
                logger.info("safety_telemetry", step=step + 1, **window)
            logger.info(
                "rollout_update",
                step=step + 1,
                last_reward=reward,
            )

        if (step + 1) % cfg.checkpoint_every == 0:
            ckpt_path = _save_run_checkpoint(
                cfg=cfg,
                ctx_run_id=ctx.run_id,
                seed=cfg.seed,
                step=step + 1,
                git_sha=ctx.git_sha,
                pyproject_hash=ctx.pyproject_hash,
                state_dict=trainer.state_dict(),
            )
            curve.checkpoint_paths.append(ckpt_path)
            logger.info("checkpoint_saved", step=step + 1, path=str(ckpt_path))

        if terminated or truncated:
            curve.per_episode_ego_rewards.append(episode_reward_acc)
            episode_reward_acc = 0.0
            episode_in_progress = False
            episode_index += 1
            # P1.04.5: reset snaps_prev so the conformal update on the
            # first step of the new episode does not diff across the
            # reset boundary (the cross-episode position delta would
            # produce a spurious large prediction-gap loss). SafetyState
            # itself is NOT reset — D3 of the design pass mandates
            # per-cell accumulation per Huriot-Sibai §IV.A.
            if _safety_active:
                snaps_prev = None
            # P6 reproducibility: derive a per-episode seed via a stable
            # substream rather than additive int mixing (cfg.seed +
            # episode_index would collide across runs). The substream
            # name pins the episode index uniquely under cfg.seed.
            episode_seed_int = int(
                derive_substream(f"training.episode.{episode_index}", root_seed=cfg.seed)
                .default_rng()
                .integers(0, 2**31 - 1)
            )
            obs, _ = env.reset(seed=episode_seed_int)
            partner.reset(seed=episode_seed_int)
            episode_in_progress = True

    # Tail-end episode (final partial chunk) — capture so the curve has
    # full coverage for the empirical-guarantee assertion. Use the
    # episode_in_progress sentinel rather than `episode_reward_acc != 0.0`
    # so a legitimately-zero accumulated reward isn't dropped.
    if episode_in_progress:
        curve.per_episode_ego_rewards.append(episode_reward_acc)

    # P1.04.5: emit the per-cell safety_telemetry_final summary the
    # audit-gate hook reads (predicates A + B; ADR-007 §Stage 1b).
    if aggregator is not None:
        final_summary = aggregator.finalise(
            safety_enabled=cfg.safety.enabled,
            predictor_kind=cfg.safety.predictor_kind,
        )
        event_name = str(final_summary.pop("event", "safety_telemetry_final"))
        logger.info(event_name, **final_summary)
    logger.info(
        "training_end",
        n_episodes=len(curve.per_episode_ego_rewards),
        n_checkpoints=len(curve.checkpoint_paths),
    )
    return TrainingResult(curve=curve, trainer=trainer)


def _save_run_checkpoint(
    *,
    cfg: EgoAHTConfig,
    ctx_run_id: str,
    seed: int,
    step: int,
    git_sha: str,
    pyproject_hash: str,
    state_dict: dict[str, Any],
) -> Path:
    """Write one checkpoint .pt + sidecar (T4b.11 + T4b.12)."""
    uri = f"local://artifacts/{ctx_run_id}_step{step}.pt"
    metadata = CheckpointMetadata(
        run_id=ctx_run_id,
        seed=seed,
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
    "EgoTrainer",
    "EnvLike",
    "PartnerLike",
    "RandomEgoTrainer",
    "RewardCurve",
    "TrainerFactory",
    "TrainingResult",
    "train",
]
