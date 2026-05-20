# SPDX-License-Identifier: Apache-2.0
"""Hydra-driven config for ego-AHT training runs (T4b.11; ADR-002 §Decisions; plan/05 §2).

The training stack is configured via Hydra YAML files under
``configs/training/<algo>/<task>.yaml``. This module ships the
type-safe Pydantic v2 models that those YAMLs validate against:
``EgoAHTConfig`` is the root, with sub-models for the env / partner /
W&B / hyperparameter blocks.

Why two layers (Hydra + Pydantic)?

- Hydra owns the composition + CLI override surface (defaults, group
  selection, ``+key=value`` overrides).
- Pydantic owns the validation + Python-side type-safety (Path coercion,
  range checks, frozen-by-default to keep call sites from mutating
  config mid-run).

:func:`load_config` is the canonical entry point: it locates the project's
``configs/`` directory by walking up from the caller's CWD, composes the
named config via Hydra, and validates it through :class:`EgoAHTConfig`.

ADR-002 §Decisions ("Hydra config root"): the YAML schema is part of the
public reproducibility contract. Adding a required field is a breaking
change to the config; deprecate-then-remove via a Pydantic
:class:`Field` default and a ``DeprecationWarning`` rather than removing
in place.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _FrozenModel(BaseModel):
    """Base for every config model — frozen + extra='forbid' (loud-fail YAML typos)."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class WandbConfig(_FrozenModel):
    """W&B sink toggle + project name (ADR-002 §Decisions; plan/05 §2).

    Attributes:
        enabled: When ``False`` (default), :func:`bind_run_logger` is
            built without a W&B sink and only the JSONL fallback fires.
            CI runs and unit tests should leave this off.
        project: W&B project name. Used only when ``enabled`` is ``True``.
    """

    enabled: bool = False
    project: str = "concerto-m4b"


class EnvConfig(_FrozenModel):
    """Env-side configuration (T4b.13; ADR-002 §Decisions; plan/05 §3.5).

    Attributes:
        task: Registered task name. Phase-0 supports ``"mpe_cooperative_push"``
            (T4b.13's empirical-guarantee env) and ``"stage0_smoke"``
            (T4b.14's zoo-seed env, deferred to user-side GPU run).
            P1.04 adds ``"stage1_pickplace"`` (ADR-007 §Stage 1b real-env
            science-evaluation target).
        episode_length: Truncation horizon in env ticks. The empirical-
            guarantee experiment defaults to 50 (matches PettingZoo
            simple_spread).
        agent_uids: 2-element list of uids the env exposes; the first is
            the ego, the second is the frozen partner.
        condition_id: Stage-1b pre-registered condition_id string
            (P1.04 / ADR-007 §Stage 1b). Required when
            ``task == "stage1_pickplace"`` because OM-homo and
            OM-hetero share the ``("panda_wristcam", "fetch")``
            agent_uids tuple — the condition_id is the explicit
            disambiguator. The yaml carries a sensible default;
            :class:`chamber.benchmarks.stage1_common.TrainedPolicyFactory`
            overrides per ``(seed, condition)`` cell via
            ``model_copy``. ``None`` for non-Stage-1b tasks (MPE,
            Stage-0) — pre-P1.04 cfgs work unchanged.
    """

    task: str
    episode_length: int = Field(default=50, gt=0)
    agent_uids: tuple[str, str] = ("ego", "partner")
    condition_id: str | None = None

    @field_validator("agent_uids", mode="before")
    @classmethod
    def _coerce_agent_uids(cls, v: object) -> object:
        # Hydra/OmegaConf parses YAML lists as ListConfig (subclass of list),
        # not tuple — coerce to keep the tuple-typed contract intact.
        if isinstance(v, list):
            return tuple(v)
        return v

    @field_validator("agent_uids")
    @classmethod
    def _check_distinct_uids(cls, v: tuple[str, str]) -> tuple[str, str]:
        if v[0] == v[1]:
            raise ValueError(f"agent_uids must be distinct; got {v!r}")
        return v


class PartnerConfig(_FrozenModel):
    """Frozen partner spec (ADR-009 §Decision; plan/05 §3.5).

    Mirrors the M4a :class:`chamber.partners.api.PartnerSpec` fields so
    the training loop can build a ``PartnerSpec`` directly from the
    config without re-validating.

    Attributes:
        class_name: Registry key passed to
            :func:`chamber.partners.registry.load_partner`. Phase-0
            empirical-guarantee runs use ``"scripted_heuristic"``.
        seed: Per-partner training seed; for scripted partners use ``0``.
        checkpoint_step: Training step at which the checkpoint was
            taken; ``None`` for scripted partners.
        weights_uri: ``local://...`` URI for frozen-RL partners; ``None``
            for scripted partners.
        extra: Free-form string-string metadata routed to
            ``PartnerSpec.extra``.
    """

    class_name: str = "scripted_heuristic"
    seed: int = 0
    checkpoint_step: int | None = None
    weights_uri: str | None = None
    extra: dict[str, str] = Field(default_factory=dict)


class RuntimeConfig(_FrozenModel):
    """Device + perf knobs (ADR-002 §Decisions; plan/08 §4 GPU determinism caveat).

    Phase-0 development runs on Mac CPU (Apple M-series). Production
    zoo-seed runs target Linux + CUDA via
    ``scripts/repro/zoo_seed.sh`` (M4b-9b). The two regimes have
    different determinism contracts:

    - **CPU**: byte-identical across runs at the same seed (plan/05 §6
      criterion 7; pinned by ``tests/integration/test_cpu_determinism.py``).
      ``deterministic_torch=True`` calls
      :func:`torch.use_deterministic_algorithms` once at trainer
      construction so any non-deterministic CUDA-style op falls back to
      a deterministic implementation if one exists, or warns loudly
      otherwise (``warn_only=True``).
    - **CUDA / MPS**: plan/08 §4 grants the GPU determinism caveat —
      cuDNN / cuBLAS kernels are not bit-deterministic across hardware
      generations even with deterministic flags. Set
      ``deterministic_torch=False`` on GPU configs to disable the
      strict-mode flag and avoid the warning spam.

    Attributes:
        device: ``"auto"`` resolves via
            :func:`chamber.utils.device.torch_device` (CUDA > MPS >
            CPU). Explicit ``"cpu"`` / ``"cuda"`` / ``"mps"`` pins the
            choice and raises if unavailable (the trainer reports the
            ADR cite in the error message so a confused user
            searching for the message lands on ADR-002 §Decisions).
        deterministic_torch: When ``True``, the trainer calls
            :func:`torch.use_deterministic_algorithms(True, warn_only=True)`
            once at construction. Documented as a process-global side
            effect.
    """

    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    deterministic_torch: bool = True


class SafetyConfig(_FrozenModel):
    """Safety-stack runtime configuration for the training rollout (P1.04.5; ADR-007 §Stage 1b).

    Phase-1 P1.04.5 wires the CBF-QP outer filter into
    :func:`concerto.training.ego_aht.train` per ADR-007 §Stage 1b
    implementation-details paragraph 2. The filter runs once per env
    step between the trainer's ``act`` and ``env.step``; conformal
    slack ``state.lambda_`` accumulates per cell and the
    :class:`concerto.training.safety_telemetry.SafetyAggregator` emits
    a ``safety_telemetry_final`` JSONL event at end-of-cell carrying
    the audit-gate predicate's inputs.

    The block is read by
    :class:`chamber.benchmarks.stage1_common.TrainedPolicyFactory`'s
    per-cell construction path; non-Stage-1b yaml configs default
    ``enabled=False`` so the filter doesn't engage outside its target
    use case (the Phase-0 MPE empirical-guarantee config + the
    Stage-0 smoke adapter both keep the un-filtered training rollout).

    Kill-switch contract (D10 from the P1.04.5 Plan-subagent design):
    when ``enabled=False`` the training loop skips the filter entirely
    and emits ``safety_enabled=false`` in the JSONL summary. The
    audit-gate hook reads that flag and emits a non-failing
    "safety disabled by operator override; gate skipped" message — the
    operator's intent is preserved on the audit trail.

    Attributes:
        enabled: Master switch. ``True`` (default) wires the filter +
            telemetry into the training loop. ``False`` skips both;
            the audit gate reads ``safety_enabled`` and treats the
            cell as non-gated.
        saturation_threshold: Fraction of ``cartesian_accel_capacity``
            above which the cell is considered saturated. Default
            ``0.9`` leaves a 10% margin — operator-tunable for the
            audit-gate predicate A. ADR-007 §Stage 1b implementation
            details paragraph 2 names the predicate verbatim
            (``λ_steady_state < cartesian_accel_capacity``); the
            ``0.9`` factor is the engineering safety margin.
        cbf_gamma: Class-K function gain forwarded to
            :func:`concerto.safety.conformal.update_lambda_from_predictor`.
            Default ``5.0`` matches the existing
            :data:`concerto.safety.cbf_qp.DEFAULT_CBF_GAMMA`.
        lambda_safe: Per-pair slack reset value at cell start
            (Phase-0 placeholder ``0.0`` per ADR-004 §Open questions;
            the derived form is a Stage-2 prerequisite).
        n_warmup_steps: Length of the warmup window after the cell
            starts (matches
            :data:`concerto.safety.api.DEFAULT_WARMUP_STEPS`).
        predictor_kind: Partner-trajectory predictor (Phase-0/1
            default ``"constant_velocity"`` per
            :func:`concerto.safety.conformal.constant_velocity_predict`;
            AoI-conditioned variants are Phase-2 per ADR-004 §Open
            questions).
        tau_brake: Per-step braking-fallback TTC threshold in seconds
            (P1.04.6; ADR-007 §Stage 1b training-vs-deployment parity).
            When the smallest pairwise time-to-collision drops below
            this, :func:`concerto.safety.braking.maybe_brake` overrides
            the ego action with the per-uid emergency-controller's
            push-apart response — the per-step backstop that does not
            depend on QP feasibility (Wang-Ames-Egerstedt 2017 eq. 17;
            ADR-004 risk-mitigation #1). Default matches
            :data:`concerto.safety.braking.DEFAULT_TAU_BRAKE` (100 ms).
        clamp_floor_ratio: Per-step symmetric clamp on ``state.lambda_``
            (P1.05.7 / issue #180; ADR-004 §Decision Revision 6). Each
            per-pair λ is clamped to
            ``[-clamp_floor_ratio x cartesian_accel_capacity,
            +clamp_floor_ratio x cartesian_accel_capacity]`` after each
            :func:`concerto.safety.conformal.update_lambda_from_predictor`
            step. Must be strictly less than
            :attr:`saturation_threshold` so the audit-gate predicate A
            (which trips at ``|λ_ss| >= saturation_threshold x cap``)
            reserves a clean margin above the clamp boundary for "the
            clamp itself failed" pathological cases. The buffer
            ``(saturation_threshold - clamp_floor_ratio) x cap`` is the
            engineering safety margin between in-loop clamping
            (operational) and audit-gate tripping (diagnostic). With
            production defaults (0.9 vs 0.7, cap = 10) the buffer is
            2.0 m/s². Trades Huriot-Sibai 2025 §VI Theorem 3's exact
            long-run average-loss bound for operational robustness —
            the bound now holds modulo the clamped boundary, which is
            consistent with the engineering meaning of "conformal
            slack bounded by the conservative manipulation envelope".
            Pre-launch evidence: P1.05 100k-frame AS-hetero probe
            measured drift of exactly ``-η x |ε| = -5e-04`` per step
            (matches analytic to 4 decimal places), projecting
            unclamped λ_ss ≈ -49.76 at the production budget — well
            past the audit-gate boundary of ±9.0.
    """

    enabled: bool = False
    saturation_threshold: float = Field(default=0.9, gt=0.0, le=1.0)
    cbf_gamma: float = Field(default=5.0, gt=0.0)
    lambda_safe: float = 0.0
    n_warmup_steps: int = Field(default=50, ge=0)
    predictor_kind: Literal["constant_velocity"] = "constant_velocity"
    tau_brake: float = Field(default=0.100, gt=0.0)
    clamp_floor_ratio: float = Field(default=0.7, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_clamp_below_saturation(self) -> SafetyConfig:
        """clamp_floor_ratio must be strictly less than saturation_threshold (P1.05.7 / #180).

        Without the strict inequality the clamp would engage at the same
        boundary the audit-gate trips, producing |λ_ss| == threshold ==
        audit-gate-trip on every clamped step. The default 0.7 < 0.9
        leaves a 0.2 x cap = 2.0 m/s² buffer at production cap. Tests
        in ``tests/unit/test_safety_config_clamp.py`` pin both this
        invariant + the boundary-buffer arithmetic.
        """
        if self.clamp_floor_ratio >= self.saturation_threshold:
            msg = (
                f"SafetyConfig.clamp_floor_ratio={self.clamp_floor_ratio} must be "
                f"strictly less than saturation_threshold={self.saturation_threshold}. "
                "The clamp engages at clamp_floor_ratio x cartesian_accel_capacity; "
                "the audit-gate predicate A trips at saturation_threshold x cap "
                "(P1.04.5 / #178). The buffer between them is the engineering "
                "safety margin — see ADR-004 §Decision Revision 6 (#180)."
            )
            raise ValueError(msg)
        return self


class HAPPOHyperparams(_FrozenModel):
    """On-policy ego-AHT HAPPO hyperparameters (ADR-002 §Decisions; plan/05 §3.2).

    Defaults match the published HARL Bi-DexHands-style values where
    they translate; tuned conservatively for the small MPE task to keep
    the empirical-guarantee experiment within its 30-minute CPU budget
    (plan/05 §8).

    Attributes:
        lr: Adam learning rate for the actor + critic.
        gamma: Discount factor.
        gae_lambda: GAE-λ for advantage estimation.
        clip_eps: PPO clipping epsilon.
        n_epochs: Number of optimisation epochs per rollout.
        rollout_length: Frames collected per epoch before the update step.
        batch_size: SGD minibatch size; rollout_length must be a multiple.
        hidden_dim: MLP hidden width for actor + critic.
    """

    lr: float = Field(default=3.0e-4, gt=0.0)
    gamma: float = Field(default=0.99, ge=0.0, le=1.0)
    gae_lambda: float = Field(default=0.95, ge=0.0, le=1.0)
    clip_eps: float = Field(default=0.2, gt=0.0)
    n_epochs: int = Field(default=4, gt=0)
    rollout_length: int = Field(default=1024, gt=0)
    batch_size: int = Field(default=256, gt=0)
    hidden_dim: int = Field(default=64, gt=0)


class EgoAHTConfig(_FrozenModel):
    """Root config for an ego-AHT training run (T4b.11; ADR-002 §Decisions).

    Validates the composed Hydra config and exposes a frozen
    Python-side handle for :func:`concerto.training.ego_aht.train`.

    Attributes:
        algo: Trainer-class registry key. Phase-0: ``"ego_aht_happo"``;
            ``"ego_aht_hatd3"`` is the Phase-1 stub.
        seed: Project root seed routed to
            :func:`concerto.training.seeding.derive_substream`. Two
            runs with the same ``seed`` produce byte-identical CPU
            reward curves (plan/05 §6 criterion 7).
        total_frames: Training budget in env frames. T4b.13 = 100_000.
        checkpoint_every: Save a ``.pt`` artefact every K frames.
        artifacts_root: Directory the ``local://artifacts/...`` URIs
            resolve against (plan/04 §3.8).
        log_dir: Directory where the JSONL logs are written.
        wandb: :class:`WandbConfig`.
        env: :class:`EnvConfig`.
        partner: :class:`PartnerConfig`.
        happo: :class:`HAPPOHyperparams`.
        runtime: :class:`RuntimeConfig` — device + determinism knobs
            (ADR-002 §Decisions; plan/08 §4 GPU determinism caveat).
            Defaults to ``device="auto"`` + ``deterministic_torch=True``
            so an unset YAML on a Mac CPU dev box just works; the
            production ``mpe_cooperative_push.yaml`` pins
            ``device: cpu`` and ``stage0_smoke.yaml`` pins
            ``device: cuda``.
    """

    algo: str = "ego_aht_happo"
    seed: int = Field(default=0, ge=0)
    total_frames: int = Field(default=100_000, gt=0)
    checkpoint_every: int = Field(default=10_000, gt=0)
    artifacts_root: Path = Path("./artifacts")
    log_dir: Path = Path("./logs")
    wandb: WandbConfig = Field(default_factory=WandbConfig)
    env: EnvConfig
    partner: PartnerConfig = Field(default_factory=PartnerConfig)
    happo: HAPPOHyperparams = Field(default_factory=HAPPOHyperparams)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)


def load_config(
    *,
    config_path: Path,
    overrides: list[str] | None = None,
) -> EgoAHTConfig:
    """Compose a Hydra config + validate via :class:`EgoAHTConfig` (T4b.11; ADR-002 §Decisions).

    The function does NOT use Hydra's ``@hydra.main`` decorator (which
    takes over the process's CWD + argv); it uses the lower-level
    :class:`hydra.compose` API so the call is testable and re-entrant.

    Args:
        config_path: Absolute path to the YAML config file (e.g.
            ``configs/training/ego_aht_happo/mpe_cooperative_push.yaml``).
            The directory becomes Hydra's ``config_dir`` and the
            stem becomes the ``config_name`` it loads. Splitting via the
            file path keeps the YAML's root-level keys at the root of
            the composed config (rather than Hydra's path-folding).
        overrides: Optional list of Hydra CLI-style overrides
            (e.g. ``["seed=7", "happo.lr=1e-4"]``).

    Returns:
        A frozen :class:`EgoAHTConfig` ready for
        :func:`concerto.training.ego_aht.train`.

    Raises:
        pydantic.ValidationError: If the composed config violates the
            schema (missing required field, out-of-range hyperparam,
            duplicate agent uids, etc.).
    """
    config_dir = config_path.parent.resolve()
    config_name = config_path.stem
    # Re-entrancy: another caller (or a leftover pytest fixture) may have
    # left ``GlobalHydra`` initialised. Clear it so initialize_config_dir's
    # ``with`` block can re-bind to the new directory cleanly.
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = compose(config_name=config_name, overrides=overrides or [])
    raw = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(raw, dict):
        raise TypeError(
            f"Composed Hydra config must be a mapping at root; got {type(raw).__name__}"
        )
    return EgoAHTConfig.model_validate(raw)


__all__ = [
    "EgoAHTConfig",
    "EnvConfig",
    "HAPPOHyperparams",
    "PartnerConfig",
    "RuntimeConfig",
    "SafetyConfig",
    "WandbConfig",
    "load_config",
]
