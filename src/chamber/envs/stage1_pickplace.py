# SPDX-License-Identifier: Apache-2.0
# pyright: reportPrivateImportUsage=false
#
# Same suppression rationale as :mod:`chamber.benchmarks.stage0_smoke`:
# ``torch.zeros`` and ``torch.rand`` are exported but not advertised in
# the stub's ``__all__``. Suppressed file-locally so the multi-agent
# reward + obs logic stays free of per-line ``type: ignore`` noise.
"""Stage-1b pick-place env (ADR-007 §Stage 1b; P1.03 / PR #<this PR>).

Real ManiSkill v3 wrap of the canonical pick-place task with the
panda + fetch / panda + panda_partner robot tuples. Supports the four
Stage-1 ``condition_id`` strings the AS / OM pre-registrations name
verbatim (``spikes/preregistration/AS.yaml``,
``spikes/preregistration/OM.yaml``; git tags
``prereg-stage1-{AS,OM}-2026-05-15``):

- AS-homo: ``stage1_pickplace_panda_only_mappo_shared_param`` ->
  agents ``("panda_wristcam", "panda_partner")``.
- AS-hetero: ``stage1_pickplace_panda_plus_fetch_ego_aht_happo_per_agent`` ->
  agents ``("panda_wristcam", "fetch")``.
- OM-homo: ``stage1_pickplace_vision_only`` ->
  agents ``("panda_wristcam", "fetch")``; vision-only channel keep-set.
- OM-hetero: ``stage1_pickplace_vision_plus_force_torque_plus_proprio`` ->
  same agents; vision + synthesised FT + proprio channel keep-set.

Per ADR-007 §Discipline the pre-registration YAMLs and their git tags
carry forward unchanged from Phase 0 — Stage 1b uses the same
condition IDs, seed list (5 seeds, 20 episodes per seed = 100 episodes
per condition), and episode budget; only the env implementation behind
the condition IDs changes. Editing a prereg YAML post-launch is a
project anti-pattern (ADR-007 §Discipline).

Stage-1a's MPE stand-in
(:class:`chamber.envs.mpe_cooperative_push.MPECooperativePushEnv`)
sequenced rig validation; this env is the Stage-1b science evaluation
target. Compute budget per ADR-007 §Stage 1b: 5 GPU-h per axis on A100;
:data:`DEFAULT_EPISODE_LENGTH = 100` chosen to fit the budget given
~0.8-1.2 s/step at multi-agent SAPIEN scale.

Wrapper-only discipline (ADR-001 §Decision (wrapper-only)):
no patches to ``mani_skill/``; no ``_*`` private imports. The env
subclasses :class:`mani_skill.envs.sapien_env.BaseEnv` directly
(matching the Stage-0 smoke env's ``_Stage0SmokeEnv`` pattern, but with
the canonical pick-place reward and a two-robot tuple). The
:func:`chamber.envs._sapien_compat.load_agent_with_bare_uids` shim
handles the multi-agent uid-suffix strip-rebind that both Stage-0 and
Stage-1 need.

Action space and the AS axis (decision Q3 / ADR-004 §Decision):
All agents use ``control_mode="pd_joint_delta_pos"`` so the 7-DOF
panda arm action stays joint-space, not Cartesian — preserving the AS
axis's intended "7-DOF arm vs 2-DOF differential-drive base"
heterogeneity (ADR-007 §Stage 1 AS spike framing). The CBF-QP outer
filter's Cartesian-acceleration interpretation of the action is then
an approximation absorbed by the conformal-slack overlay (Huriot & Sibai
2025 §VI); see the ADR-007 §Stage 1b note for the quantitative
dimensional analysis ("off by 1/dt² (~2500x at dt=0.02s, small-Δp
scale)") and ADR-004 §Open questions / slice P1.05.5 for the
contingent long-term fix (extend :class:`AgentSnapshot` with ``qpos``).

Force-torque synthesis (decision Q2 / ADR-007 §Revision history rev. 5):
The OM-hetero "force_torque" channel is **synthesised** from the
panda gripper's fingertip contact wrenches via
:meth:`Articulation.get_net_contact_forces(["panda_leftfinger", "panda_rightfinger"])`
plus zero-mean Gaussian noise at
:data:`SYNTHESISED_FT_NOISE_SIGMA_N` Newtons (default 0.5 N — at the
lower bound of the real Franka wrist-FT noise floor of 0.5-1.5 N).
**This is not a real Panda wrist FT sensor reading.** Wrist-mounted
virtual FT is a Phase-1.5 ablation (ADR-007 §Open questions; Stage-1b+0.5
follow-up). The sigma value lives in this module + the ADR-007 §Revision
history entry only — **never in the prereg YAML** (ADR-007 §Discipline:
tag rotation forbidden).

Cooperation gradient (decision Q in design conversation):
Cube and goal spawn locations follow the upstream ``PickCubeEnv``
defaults for now (cube within ±5 cm of the table origin; goal within
the same envelope, z ∈ [0, 0.3]). The cooperative gradient between
AS-homo (two pandas) and AS-hetero (panda + fetch) is itself a
Stage-1b empirical question; pose tuning is sequenced as a P1.04
follow-up once the first pilot's success-rate landscape is known
(don't gold-plate at design time).

Tier-1 / Tier-2 split (decision 9):
The module's top-level imports are intentionally Tier-1-safe
(``numpy``, ``gymnasium``, ``concerto.safety.api`` types,
``chamber.envs.errors``). ManiSkill / SAPIEN /
``pytorch_kinematics`` imports happen inside
:func:`make_stage1_pickplace_env` so
``python -c "import chamber.envs.stage1_pickplace"`` succeeds on a
Vulkan-less host. The pure-function Tier-1 surface
(:func:`resolve_condition`, :func:`build_control_models_for_condition`)
is what the Tier-1 tests exercise; Tier-2 SAPIEN-gated tests cover
real env construction.

References:
- ADR-001 §Decision (wrapper-only); ADR-001 §Validation criteria
  (Stage-0 pattern this env mirrors).
- ADR-004 §Decision (CBF-QP + JacobianControlModel contract);
  ADR-004 §Open questions (AgentSnapshot.qpos extension, slice P1.05.5).
- ADR-005 §Decision (SAPIEN 3 + Warp-MPM substrate, inherited from
  ManiSkill v3).
- ADR-007 §Stage 1b (this env's spec); ADR-007 §Discipline (tag
  rotation forbidden); ADR-007 §Open questions (FT wrist-mount
  ablation, P1.05.5 contingent slice).
- ``spikes/preregistration/{AS,OM}.yaml`` (the condition_id strings
  this env resolves verbatim).
- :class:`concerto.safety.api.JacobianControlModel`,
  :class:`concerto.safety.api.DoubleIntegratorControlModel` (the
  per-uid control models :meth:`Stage1PickPlaceEnv.build_control_models`
  returns).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple

import gymnasium as gym
import numpy as np

from chamber.envs.errors import ChamberEnvCompatibilityError
from concerto.safety.api import (
    AgentControlModel,
    Bounds,
    DoubleIntegratorControlModel,
    JacobianControlModel,
)
from concerto.safety.cbf_qp import AgentSnapshot
from concerto.training.seeding import derive_substream

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Mapping

    from numpy.typing import NDArray

#: Default episode horizon, in env ticks (ADR-007 §Stage 1b compute budget).
#:
#: 100 steps balances the cooperative pick-place's natural duration
#: (reach ~20, grasp ~10, carry ~30, drop ~20, fetch positions ~20)
#: against the per-axis 5 GPU-h budget (200 episodes/axis x ~1 s/step
#: x 100 steps ≈ 5.6 h with rollout overhead). The upstream
#: ``PickCubeEnv`` ships with ``max_episode_steps=50``; the longer
#: horizon accommodates the two-robot handoff. Override via the
#: ``episode_length`` kwarg if pilot data warrants tuning.
DEFAULT_EPISODE_LENGTH: int = 100

#: sigma of the zero-mean Gaussian noise added to the synthesised
#: force-torque channel, in Newtons (decision Q2; ADR-007 §Revision history
#: rev. 5).
#:
#: 0.5 N is at the lower bound of the real Franka wrist-FT sensor's
#: noise floor (Franka spec: 0.5-1.5 N over the operating envelope).
#: This choice favours the **OM-hetero gate**: if the ≥20 pp gap clears
#: at sigma=0.5 N, it would also clear at the realistic 1-1.5 N (since
#: signal-to-noise only gets worse with more noise); if it fails at
#: sigma=0.5 N, a lower noise level would not rescue it. The constant is
#: load-bearing for the OM axis's scientific credibility; **must not**
#: be edited without an ADR-007 §Revision history entry naming the new
#: value and the empirical justification. The sigma value lives in code +
#: ADR-007 §Revision history only — **never in the prereg YAML**
#: (ADR-007 §Discipline: tag rotation forbidden).
SYNTHESISED_FT_NOISE_SIGMA_N: float = 0.5

#: Panda end-effector Cartesian acceleration cap, in m/s² (ADR-004 §Decision).
#:
#: Conservative half-of-rated-max from the Franka Emika Panda data
#: sheet (the real Panda's rated end-effector capacity is ~20 m/s²
#: under unloaded conditions; halving it gives the operating-envelope
#: cap the CBF-QP outer filter needs as ``Bounds.cartesian_accel_capacity``).
#: Consumed by
#: :class:`concerto.safety.api.JacobianControlModel.max_cartesian_accel_value`;
#: required to be finite (the default ``nan`` raises at the QP boundary
#: per ``concerto/safety/api.py:600-606``).
PANDA_CARTESIAN_ACCEL_CAPACITY_MS2: float = 10.0

#: Fetch base + arm Cartesian acceleration cap, in m/s² (ADR-004 §Decision).
#:
#: The fetch's wheel-driven base is closer to rated 2 m/s² under load;
#: a conservative 1.5 m/s² for the unified base+arm safety body is
#: load-bearing for the AS-hetero pair-row construction (fetch is the
#: partner in the AS-hetero / OM conditions). The fetch uses
#: :class:`DoubleIntegratorControlModel` so this number lands in
#: ``Bounds.cartesian_accel_capacity`` via the per-uid bounds dict,
#: not on the control model itself.
FETCH_CARTESIAN_ACCEL_CAPACITY_MS2: float = 1.5

#: Panda end-effector + wrist sphere radius for the CBF pairwise distance
#: barrier (P1.04.5 / ADR-007 §Stage 1b safety stack wiring).
#:
#: Engineering estimate — the gripper-extended envelope of the
#: ``panda_hand`` link (8 cm finger reach + 5 cm wrist mount + 5 cm cube
#: half-envelope ≈ 18 cm; rounding down to 10 cm so the configured
#: radius is the inscribing sphere around the *grasping configuration*
#: rather than the fully-extended-finger envelope). Used for both
#: ``panda_wristcam`` and ``panda_partner`` (the panda partner shares
#: the URDF + kinematic envelope; only the URDF visual mesh differs
#: per P1.03's ``PandaPartner`` agent definition).
#:
#: The :func:`_validate_safety_radii` env-construction-time assertion
#: pins this value against
#: :data:`_MINIMUM_SAFETY_RADIUS_BY_UID` — a future operator who
#: shrinks the constant below the URDF envelope minimum gets a
#: loud-fail at env construction with the URDF source + per-uid name
#: in the error message (not weeks later as a "λ saturates randomly"
#: bug).
PANDA_SAFETY_RADIUS_M: float = 0.10

#: Fetch mobile-base sphere radius for the CBF pairwise distance
#: barrier (P1.04.5 / ADR-007 §Stage 1b safety stack wiring).
#:
#: Engineering estimate — Fetch's ``base_link`` is a ~60 cm x 60 cm
#: cylindrical mobile base; the inscribing sphere is ~30 cm radius
#: with 5 cm margin for wheel-protrusion = 35 cm. The configured value
#: covers the base + wheels but NOT the arm-extended envelope (the
#: scripted-heuristic partner used in Phase-1 only animates the base;
#: arm-folded configuration is assumed). A future refactor that
#: animates the fetch's arm during partner-control must revisit this
#: constant — :func:`_validate_safety_radii` does not currently catch
#: the arm-extended case.
FETCH_SAFETY_RADIUS_M: float = 0.35

#: Per-uid minimum safety-radius envelope used by
#: :func:`_validate_safety_radii` (P1.04.5 / ADR-007 §Stage 1b).
#:
#: Asserted at env construction: each configured radius MUST be
#: ``>= _MINIMUM_SAFETY_RADIUS_BY_UID[uid]``. The minimums are derived
#: from the URDF link-frame extents (panda: gripper + wrist envelope
#: ~ 8 cm; fetch: base-link cylinder ~ 30 cm). The configured
#: ``PANDA_SAFETY_RADIUS_M`` / ``FETCH_SAFETY_RADIUS_M`` constants
#: above clear these minimums by design; the assertion is the
#: regression-pin against future operator edits that would shrink
#: them below safe values.
_MINIMUM_SAFETY_RADIUS_BY_UID: dict[str, float] = {
    "panda_wristcam": 0.08,
    "panda_partner": 0.08,
    "fetch": 0.30,
}


def _validate_safety_radii(
    safety_radii: dict[str, float],
    *,
    minimums: dict[str, float] = _MINIMUM_SAFETY_RADIUS_BY_UID,
) -> None:
    """Assert configured safety radii clear the per-uid URDF envelope minimums (P1.04.5).

    Called once at :class:`Stage1PickPlaceEnv` construction (Q3 from
    the P1.04.5 design pass). Loud-fails with the per-uid name +
    configured value + minimum envelope so a future edit that shrinks
    a radius below the URDF-derived safe envelope surfaces at env
    construction rather than weeks later as a "λ saturates randomly"
    bug.

    Args:
        safety_radii: ``{uid: radius_metres}`` map the env uses for the
            CBF pairwise distance barrier (sourced from
            :data:`PANDA_SAFETY_RADIUS_M` / :data:`FETCH_SAFETY_RADIUS_M`).
        minimums: Per-uid minimum-envelope map. Defaults to
            :data:`_MINIMUM_SAFETY_RADIUS_BY_UID`; tests override.

    Raises:
        ValueError: If any uid's configured radius is below its
            minimum. Names the uid + configured value + minimum value
            + the source-of-truth constant in the error message.
    """
    for uid, configured in safety_radii.items():
        minimum = minimums.get(uid)
        if minimum is None:
            # An unknown uid (Tier-2 fakes, Phase-2 envs) is a soft
            # warning — pass-through rather than refuse so non-Stage-1b
            # callers stay unaffected. Future Stage-2/3 envs should
            # contribute their own entries to the minimums table.
            continue
        if configured < minimum:
            msg = (
                f"_validate_safety_radii: uid={uid!r} configured "
                f"radius={configured:.4f} m is below the URDF-envelope "
                f"minimum={minimum:.4f} m. Update the corresponding "
                "module-level constant in chamber.envs.stage1_pickplace "
                "(PANDA_SAFETY_RADIUS_M / FETCH_SAFETY_RADIUS_M); the "
                "minimums table is _MINIMUM_SAFETY_RADIUS_BY_UID. "
                "ADR-007 §Stage 1b safety-stack wiring (P1.04.5)."
            )
            raise ValueError(msg)


class ConditionConfig(NamedTuple):
    """Resolved per-condition configuration (ADR-007 §Stage 1b).

    Returned by :func:`resolve_condition`. Pure-Python NamedTuple so
    Tier-1 tests can construct fakes without any SAPIEN dependency.

    Attributes:
        condition_id: The verbatim pre-registration ``condition_id``
            string (one of the four Stage-1 strings).
        agent_uids: Ordered ``(ego_uid, partner_uid)`` tuple. The ego
            is the agent the trainer updates / the safety filter
            authorises; the partner is the frozen / heuristic partner.
        obs_mode: ManiSkill v3 env-level ``obs_mode`` superset string
            passed to :class:`BaseEnv.__init__`. The OM channel-filter
            wrapper (:class:`chamber.envs.stage1_obs_filter.Stage1OMChannelFilter`)
            slices this superset per-uid to the condition's actual
            channel keep-set.
        is_om_condition: ``True`` for the two OM ``condition_id`` strings
            (``vision_only`` / ``vision_plus_force_torque_plus_proprio``).
            Drives whether the channel-filter wrapper is engaged
            downstream.
        ft_synthesis_enabled: ``True`` for the OM-hetero condition. The
            env always *injects* the synthesised force-torque channel
            into ``obs["agent"]["panda_wristcam"]["force_torque"]``; the
            channel filter then masks or keeps it per condition. The
            flag here is informational only (consumed by the smoke
            scripts and the docstring report).
    """

    condition_id: str
    agent_uids: tuple[str, str]
    obs_mode: str
    is_om_condition: bool
    ft_synthesis_enabled: bool


#: Stage-1 condition_id -> resolved configuration table (ADR-007 §Stage 1b).
#:
#: Single source of truth for the four pre-registered condition_id
#: strings the chamber-side spike adapters (``chamber.benchmarks.stage1_as``,
#: ``chamber.benchmarks.stage1_om``) dispatch through. Verbatim names
#: from ``spikes/preregistration/{AS,OM}.yaml`` (PR #109; git tags
#: ``prereg-stage1-{AS,OM}-2026-05-15``). Editing this table to rename
#: a condition_id is forbidden — re-issue the prereg with a new tag per
#: ADR-007 §Discipline.
_CONDITION_TABLE: dict[str, ConditionConfig] = {
    "stage1_pickplace_panda_only_mappo_shared_param": ConditionConfig(
        condition_id="stage1_pickplace_panda_only_mappo_shared_param",
        agent_uids=("panda_wristcam", "panda_partner"),
        obs_mode="state_dict",
        is_om_condition=False,
        ft_synthesis_enabled=False,
    ),
    "stage1_pickplace_panda_plus_fetch_ego_aht_happo_per_agent": ConditionConfig(
        condition_id="stage1_pickplace_panda_plus_fetch_ego_aht_happo_per_agent",
        agent_uids=("panda_wristcam", "fetch"),
        obs_mode="state_dict",
        is_om_condition=False,
        ft_synthesis_enabled=False,
    ),
    "stage1_pickplace_vision_only": ConditionConfig(
        condition_id="stage1_pickplace_vision_only",
        agent_uids=("panda_wristcam", "fetch"),
        obs_mode="rgb+depth+state_dict",
        is_om_condition=True,
        ft_synthesis_enabled=False,
    ),
    "stage1_pickplace_vision_plus_force_torque_plus_proprio": ConditionConfig(
        condition_id="stage1_pickplace_vision_plus_force_torque_plus_proprio",
        agent_uids=("panda_wristcam", "fetch"),
        obs_mode="rgb+depth+state_dict",
        is_om_condition=True,
        ft_synthesis_enabled=True,
    ),
}

#: Per-uid initial-base poses (xyz in metres) for the two-robot tuple.
#:
#: ``PickCubeEnv._load_agent`` places the (single) panda at
#: ``(-0.615, 0, 0)``; the partner / fetch is mirrored to
#: ``(+0.615, 0, 0)`` so neither robot's mounted base collides with the
#: cube spawn region (around the table origin). Pose tuning is a
#: P1.04 follow-up concern; these defaults exercise the rig.
_AGENT_BASE_POSE_XYZ: dict[str, tuple[float, float, float]] = {
    "panda_wristcam": (-0.615, 0.0, 0.0),
    "panda_partner": (+0.615, 0.0, 0.0),
    "fetch": (+0.615, 0.0, 0.0),
}

#: Substream name routed through :func:`concerto.training.seeding.derive_substream`
#: for the env's deterministic RNG (P6 reproducibility; P6 / ADR-002 determinism rule).
_SUBSTREAM_NAME: str = "env.stage1_pickplace"


def resolve_condition(condition_id: str) -> ConditionConfig:
    """Resolve a pre-registered ``condition_id`` to its config (ADR-007 §Stage 1b).

    Pure Tier-1 function — no SAPIEN dependency. Tier-1 tests pin the
    four-condition coverage and the ``ValueError`` raise on bogus IDs.

    Args:
        condition_id: One of the four verbatim Stage-1 pre-registration
            strings (see ``spikes/preregistration/{AS,OM}.yaml``).

    Returns:
        Resolved :class:`ConditionConfig` carrying the
        ``agent_uids`` tuple, env-level ``obs_mode`` superset, and the
        boolean flags downstream wrappers / scripts read.

    Raises:
        ValueError: If ``condition_id`` is not one of the four
            Stage-1 pre-registration strings. The message names the
            four valid options and cites ADR-007 §Discipline (the
            project anti-pattern for editing prereg YAMLs).
    """
    try:
        return _CONDITION_TABLE[condition_id]
    except KeyError as exc:
        msg = (
            f"Stage1PickPlaceEnv: condition_id {condition_id!r} is not one of the "
            f"four Stage-1 pre-registration strings {sorted(_CONDITION_TABLE)!r}. "
            "If the prereg YAML has been edited, re-issue with a new git_tag "
            "per ADR-007 §Discipline; do not rename condition_ids in code."
        )
        raise ValueError(msg) from exc


def build_control_models_for_condition(
    condition_id: str,
    *,
    jacobian_fn: Callable[[AgentSnapshot], NDArray[np.float64]] | None = None,
    fetch_action_dim: int | None = None,
    panda_action_dim: int = 8,
) -> dict[str, AgentControlModel]:
    """Build per-uid :class:`AgentControlModel` map for ``condition_id`` (ADR-004 §Decision).

    Pure Tier-1 function — no SAPIEN dependency. The Jacobian callable
    is plumbed in by the env at construction time (see
    :meth:`Stage1PickPlaceEnv.build_control_models`); Tier-1 tests pass
    ``jacobian_fn=None`` to exercise the model-construction path, which
    keeps the placeholder per :class:`JacobianControlModel`'s
    loud-fail-on-use contract (``concerto/safety/api.py:552``).

    Args:
        condition_id: Stage-1 ``condition_id`` (raises if unknown).
        jacobian_fn: Optional Jacobian callable to embed in the panda
            uid's :class:`JacobianControlModel`. ``None`` means the
            placeholder model is returned (Tier-1 path); pass the env's
            closure-via-env-reference callable
            (``lambda snap: env._jacobian_provider(snap, env._latest_qpos["panda_wristcam"])``)
            to get a working model for Tier-2 / production.
        fetch_action_dim: Action dimension of the fetch uid's Box.
            ``None`` (default) is the Tier-1 sentinel — :class:`DoubleIntegratorControlModel`
            with ``action_dim=2`` (just the differential-drive base axis;
            the arm + body + gripper are folded into the same model
            with the canonical "treat the whole action as a 1-D
            Cartesian acceleration" approximation P1.05.5 will tighten).
            Tier-2 paths pass the actual ``env.action_space[uid].shape[0]``.
        panda_action_dim: Action dimension of the panda uid's Box.
            Defaults to ``8`` (7 arm joint deltas + 1 mimic-gripper
            joint delta under ``pd_joint_delta_pos``). Both
            ``panda_wristcam`` and ``panda_partner`` share this shape.

    Returns:
        ``{uid: AgentControlModel}`` map keyed by the condition's
        :attr:`ConditionConfig.agent_uids`. The panda uid (always the
        ego, always present) gets a :class:`JacobianControlModel`; the
        partner uid gets a :class:`JacobianControlModel` for
        ``panda_partner`` (AS-homo) or
        :class:`DoubleIntegratorControlModel` for ``fetch`` (AS-hetero /
        OM-*).

    Raises:
        ValueError: If ``condition_id`` is not in the four-condition
            table (delegated to :func:`resolve_condition`).
    """
    config = resolve_condition(condition_id)
    ego_uid, partner_uid = config.agent_uids
    models: dict[str, AgentControlModel] = {}
    # Ego is always panda_wristcam — 7-DOF arm + mimic-gripper.
    # JacobianControlModel position_dim=3 (Cartesian) matches the
    # AgentSnapshot Cartesian-only contract (cbf_qp.py:103-110).
    models[ego_uid] = JacobianControlModel(
        uid=ego_uid,
        action_dim=panda_action_dim,
        position_dim=3,
        jacobian_fn=jacobian_fn,
        max_cartesian_accel_value=PANDA_CARTESIAN_ACCEL_CAPACITY_MS2,
    )
    if partner_uid == "panda_partner":
        # AS-homo: second panda, identical kinematics, same Jacobian
        # callable wired by the env via the closure pattern.
        models[partner_uid] = JacobianControlModel(
            uid=partner_uid,
            action_dim=panda_action_dim,
            position_dim=3,
            jacobian_fn=jacobian_fn,
            max_cartesian_accel_value=PANDA_CARTESIAN_ACCEL_CAPACITY_MS2,
        )
    else:
        # AS-hetero / OM-*: fetch base+arm uses DoubleIntegratorControlModel.
        # action_dim default 2 covers the 2-DOF differential-drive base
        # slice that ADR-007 §Stage 1 AS spike framing names; real
        # Tier-2 path passes the full action_space shape.
        effective_action_dim = 2 if fetch_action_dim is None else fetch_action_dim
        models[partner_uid] = DoubleIntegratorControlModel(
            uid=partner_uid,
            action_dim=effective_action_dim,
        )
    return models


def make_stage1_pickplace_env(
    *,
    condition_id: str,
    episode_length: int = DEFAULT_EPISODE_LENGTH,
    root_seed: int = 0,
    num_envs: int = 1,
    render_mode: str | None = None,
    render_backend: str | None = None,
    force_torque_noise_sigma: float = SYNTHESISED_FT_NOISE_SIGMA_N,
) -> gym.Env[Any, Any]:
    """Build a :class:`Stage1PickPlaceEnv` instance (ADR-007 §Stage 1b).

    Factory entry point. SAPIEN / ManiSkill imports are deferred to the
    body so ``python -c "import chamber.envs.stage1_pickplace"`` works
    on a Vulkan-less host (Tier-1 contract). The
    :class:`Stage1PickPlaceEnv` class itself is defined inside the
    factory body — same pattern as
    :func:`chamber.benchmarks.stage0_smoke.make_stage0_env`.

    Args:
        condition_id: One of the four Stage-1 ``condition_id`` strings;
            raises :class:`ValueError` on any other input
            (cf. :func:`resolve_condition`).
        episode_length: Truncation horizon, in env ticks. Default
            :data:`DEFAULT_EPISODE_LENGTH` (100). See module docstring
            for the budget rationale.
        root_seed: Project root seed routed to the env's
            :func:`concerto.training.seeding.derive_substream` substream
            for P6 reproducibility (P6 / ADR-002 determinism rule).
        num_envs: ManiSkill vectorisation count; default 1 (Stage-1b
            paths are single-env per cell). Higher values are supported
            but untested.
        render_mode: ManiSkill ``render_mode`` (``None``,
            ``"human"``, ``"rgb_array"``, etc.). ``None`` (default)
            disables rendering for the spike-runner path.
        render_backend: ManiSkill ``render_backend`` (``"gpu"``,
            ``"none"``, …). ``None`` (default) falls through to
            ManiSkill's own default. The Stage-0 smoke env uses
            ``CHAMBER_RENDER_BACKEND`` env-var probe + the
            visual-material strip patch; Stage-1b follows the same
            pattern but exposes the backend choice as a constructor
            kwarg too.
        force_torque_noise_sigma: sigma of the zero-mean Gaussian noise
            added to the synthesised force-torque channel, in Newtons.
            Default :data:`SYNTHESISED_FT_NOISE_SIGMA_N`. Override only
            with an ADR-007 §Revision history entry (the value is
            load-bearing for the OM gate; see module docstring).

    Returns:
        :class:`Stage1PickPlaceEnv` instance, ready to call
        ``reset(seed=K)`` on. The action space is a
        :class:`gym.spaces.Dict` keyed by the resolved
        :attr:`ConditionConfig.agent_uids`.

    Raises:
        ValueError: If ``condition_id`` is unknown (see
            :func:`resolve_condition`).
        ChamberEnvCompatibilityError: If ManiSkill / SAPIEN / Vulkan
            initialisation fails (CPU-only host without a fallback
            ``render_backend="none"``; see ADR-001 §Risks).
    """
    # Validate condition_id eagerly (the inner __init__ will resolve again,
    # but raising here gives the user a useful traceback in the Tier-1
    # bogus-id path without paying the SAPIEN-import cost).
    resolve_condition(condition_id)
    try:
        import mani_skill.envs  # noqa: F401 - registers ManiSkill env IDs
        import sapien
        import torch
        from mani_skill.envs.sapien_env import BaseEnv
        from mani_skill.utils.building import actors
        from mani_skill.utils.scene_builder.table import TableSceneBuilder
        from mani_skill.utils.structs.pose import Pose

        # Trigger PandaPartner registration before BaseEnv reaches the
        # multi-agent build loop. The chamber.agents package's
        # try/except ensures this import is the place where a Tier-1
        # context would have already failed if it was going to.
        import chamber.agents  # noqa: F401 - side-effect import for register_agent
        from chamber.agents.panda_jacobian import (
            CARTESIAN_POSITION_DIM,
            PANDA_ARM_DOF,
            PandaJacobianProvider,
        )
        from chamber.envs._sapien_compat import (
            load_agent_with_bare_uids,
            patch_sapien_urdf_no_visual_material,
        )
    except ImportError as exc:
        raise ChamberEnvCompatibilityError(
            "Stage1PickPlaceEnv requires mani_skill / sapien / pytorch_kinematics "
            "in the active venv. Install per pyproject.toml; see ADR-001 §Risks."
        ) from exc

    if render_backend == "none":
        # Headless host workaround — same idempotent SAPIEN URDF patch
        # the Stage-0 smoke env relies on (ADR-001 §Risks).
        patch_sapien_urdf_no_visual_material()

    class Stage1PickPlaceEnv(BaseEnv):  # type: ignore[misc, valid-type]
        """Two-robot pick-place env (ADR-007 §Stage 1b; ADR-001 §Decision).

        Subclasses :class:`mani_skill.envs.sapien_env.BaseEnv` directly
        (matching :class:`chamber.benchmarks.stage0_smoke._Stage0SmokeEnv`).
        See the module docstring for the wrapper-only-discipline /
        synthesised-FT / closure-via-env-Jacobian / cooperation-gradient
        rationale.

        Implements the four condition_id strings the AS / OM pre-
        registrations name; the obs-mode superset is set per condition
        at constructor time, the per-uid channel filtering happens in
        :class:`chamber.envs.stage1_obs_filter.Stage1OMChannelFilter`
        (kept separate so the AS path doesn't pay the filter cost).
        """

        SUPPORTED_ROBOTS: ClassVar[list[tuple[str, str]]] = [  # type: ignore[assignment]
            ("panda_wristcam", "panda_partner"),
            ("panda_wristcam", "fetch"),
        ]

        def __init__(
            self,
            *,
            condition_id: str,
            episode_length: int,
            root_seed: int,
            num_envs: int,
            render_mode: str | None,
            render_backend: str | None,
            force_torque_noise_sigma: float,
        ) -> None:
            """Build the env (ADR-007 §Stage 1b).

            Stores the condition-resolved metadata, sets up the
            determinism harness, then calls
            ``super().__init__(robot_uids=config.agent_uids, ...)``.
            The Jacobian provider is constructed *after* the super-init
            because it reads the panda's URDF directly (independent of
            SAPIEN scene state); the closure plumbed into
            :meth:`build_control_models` then reads ``self._latest_qpos``
            which the env updates each :meth:`_before_simulation_step`.
            """
            self._condition_config: ConditionConfig = resolve_condition(condition_id)
            self._episode_length: int = int(episode_length)
            self._root_seed: int = int(root_seed)
            self._force_torque_noise_sigma: float = float(force_torque_noise_sigma)
            self._rng: np.random.Generator = derive_substream(
                _SUBSTREAM_NAME, root_seed=self._root_seed
            ).default_rng()
            # Per-uid latest qpos cache for the Jacobian closure; populated
            # before each ``step``. See module docstring on the closure-
            # via-env-reference workaround (decision Q3 mod 3; ADR-004
            # §Open questions / slice P1.05.5).
            self._latest_qpos: dict[str, NDArray[np.float64]] = {}
            self._step_count: int = 0
            # PickCubeEnv-style task parameters; consistent with the
            # upstream defaults so the success / reward shaping path
            # behaves predictably under the multi-agent rig.
            self._cube_half_size: float = 0.02
            self._goal_thresh: float = 0.025
            self._cube_spawn_half_size: float = 0.05
            self._cube_spawn_center: tuple[float, float] = (0.0, 0.0)
            self._max_goal_height: float = 0.3
            # Attributes that ``_initialize_episode`` / ``_before_simulation_step``
            # touch MUST be assigned before ``super().__init__()``: ManiSkill's
            # ``BaseEnv.__init__`` calls ``self.reset(...)`` during super-init
            # (mani_skill/envs/sapien_env.py:327), which invokes
            # ``_initialize_episode`` *and* the post-reset qpos primer below
            # before the subclass body resumes. Subclass attributes written
            # after super-init do not yet exist when those hooks fire.
            self._panda_arm_dof: int = PANDA_ARM_DOF
            # Per-uid cache for the numerical-difference velocity in
            # build_agent_snapshots (D8 of the P1.04.5 design pass).
            # Cleared by _initialize_episode at every reset so the
            # cross-episode position delta doesn't produce a spurious
            # first-step velocity. The first-step velocity defaults to
            # zero on each fresh episode (the agents start at rest by
            # construction).
            self._prev_positions: dict[str, NDArray[np.float64]] = {}
            try:
                super().__init__(
                    robot_uids=self._condition_config.agent_uids,  # type: ignore[arg-type]
                    num_envs=num_envs,
                    obs_mode=self._condition_config.obs_mode,
                    control_mode="pd_joint_delta_pos",
                    render_mode=render_mode,
                    render_backend=render_backend if render_backend is not None else "gpu",
                )
            except RuntimeError as exc:
                raise ChamberEnvCompatibilityError(
                    "SAPIEN/Vulkan initialisation failed during "
                    f"Stage1PickPlaceEnv(condition_id={condition_id!r}) build: "
                    f"{exc}\nSet CHAMBER_RENDER_BACKEND=none (or "
                    'render_backend="none") on CUDA-only hosts; see ADR-001 §Risks.'
                ) from exc
            # Jacobian provider is wristcam-URDF based; the partner
            # (panda_partner) shares the kinematic chain so a single
            # provider serves both panda uids. ``panda_v2.urdf`` and
            # ``panda_v3.urdf`` differ only in the wristcam mount
            # (extrinsic frame); the 7 arm joints + TCP link are
            # identical, so reusing the v3-URDF chain for the partner
            # introduces no error in the linear Jacobian.
            self._jacobian_provider: PandaJacobianProvider = PandaJacobianProvider()
            # Per-uid safety radii (Q3 of the P1.04.5 design pass).
            # Asserted at construction against URDF-envelope minimums.
            self._safety_radii: dict[str, float] = {
                uid: PANDA_SAFETY_RADIUS_M
                if uid in ("panda_wristcam", "panda_partner")
                else FETCH_SAFETY_RADIUS_M
                for uid in self._condition_config.agent_uids
            }
            _validate_safety_radii(self._safety_radii)

        # ----- ManiSkill v3 BaseEnv hooks -----

        @property
        def _default_human_render_camera_configs(self) -> Any:  # noqa: ANN401 - ManiSkill CameraConfig has no project type
            """Third-person render camera for rollout visualisation (ADR-007 §Stage 1b).

            The Stage-1b env subclasses :class:`BaseEnv` directly and
            otherwise defines no human-render camera, so ``env.render()``
            returns ``None`` and a saved checkpoint cannot be visualised.
            This adds a single fixed third-person camera looking at the
            table workspace so a checkpoint can be rolled out and rendered
            to RGB frames (reach / grasp / lift inspection). Render-only:
            it feeds neither any observation channel nor the gate contract,
            so it does not affect determinism or the comparison protocol.
            """
            from mani_skill.sensors.camera import CameraConfig
            from mani_skill.utils import sapien_utils

            pose = sapien_utils.look_at(eye=[0.7, 0.7, 0.6], target=[0.0, 0.0, 0.1])
            return CameraConfig(
                "render_camera",
                pose=pose,
                width=640,
                height=480,
                fov=1.0,
                near=0.01,
                far=100.0,
            )

        def _load_agent(  # type: ignore[override]
            self,
            options: dict[str, Any],
            initial_agent_poses: object = None,
            build_separate: bool = False,
        ) -> None:
            """Multi-robot ``_load_agent`` override with per-uid base poses.

            Delegates to
            :func:`chamber.envs._sapien_compat.load_agent_with_bare_uids`
            for the suffix-strip rebind. Passes an explicit per-agent
            pose list so the two robots don't collide at the table
            origin.
            """
            if initial_agent_poses is None:
                poses: list[object] = []
                for uid in self.robot_uids:  # type: ignore[union-attr]
                    xyz = _AGENT_BASE_POSE_XYZ.get(uid)
                    if xyz is None:
                        # Unknown uid (shouldn't happen given the
                        # condition table) — leave SAPIEN to use URDF
                        # default root pose.
                        poses.append(None)
                    else:
                        poses.append(sapien.Pose(p=list(xyz)))
                initial_agent_poses = poses
            load_agent_with_bare_uids(
                self,
                options,
                initial_agent_poses=initial_agent_poses,
                build_separate=build_separate,
            )

        def _load_scene(self, options: dict[str, Any]) -> None:
            """Build the table + cube + goal site (PickCubeEnv recipe; ADR-007 §Stage 1b)."""
            del options
            self.table_scene = TableSceneBuilder(self, robot_init_qpos_noise=0.02)
            self.table_scene.build()
            self.cube = actors.build_cube(
                self.scene,
                half_size=self._cube_half_size,
                color=[1, 0, 0, 1],
                name="cube",
                initial_pose=sapien.Pose(p=[0, 0, self._cube_half_size]),
            )
            self.goal_site = actors.build_sphere(
                self.scene,
                radius=self._goal_thresh,
                color=[0, 1, 0, 1],
                name="goal_site",
                body_type="kinematic",
                add_collision=False,
                initial_pose=sapien.Pose(),
            )
            self._hidden_objects.append(self.goal_site)  # type: ignore[attr-defined]

        def _initialize_episode(self, env_idx: torch.Tensor, options: dict[str, Any]) -> None:
            """Reset cube + goal poses per episode (P6 determinism via :attr:`_rng`)."""
            del options
            # Clear the per-agent position cache so build_agent_snapshots'
            # first-step velocity (numerical-difference) defaults to
            # zero on each fresh episode rather than diffing against
            # the previous episode's final position (R2 of the P1.04.5
            # risk register).
            self._prev_positions.clear()
            with torch.device(self.device):  # type: ignore[attr-defined]
                b = len(env_idx)
                self.table_scene.initialize(env_idx)
                # Cube xy uniform in the spawn box; z at half-size so it
                # rests on the table surface. Routed through the env's
                # P6-harnessed RNG instead of torch.rand so two
                # reset(seed=K) calls reproduce byte-identical state
                # (P6 / ADR-002 determinism rule).
                xy_cube_np = self._rng.uniform(
                    -self._cube_spawn_half_size, self._cube_spawn_half_size, size=(b, 2)
                )
                xy_cube = torch.from_numpy(np.asarray(xy_cube_np, dtype=np.float32))
                xyz = torch.zeros((b, 3))
                xyz[:, 0] = xy_cube[:, 0] + self._cube_spawn_center[0]
                xyz[:, 1] = xy_cube[:, 1] + self._cube_spawn_center[1]
                xyz[:, 2] = self._cube_half_size
                # No rotation randomisation in P1.03 — upstream PickCubeEnv
                # uses ``randomization.random_quaternions(b, lock_x=True,
                # lock_y=True)``; for the two-robot env the orientation
                # matters less than the position. Re-enable if pilot data
                # shows it changes the AS gap.
                self.cube.set_pose(Pose.create_from_pq(xyz))

                # Goal xy in the same envelope; z uniform in [0, max].
                xy_goal_np = self._rng.uniform(
                    -self._cube_spawn_half_size, self._cube_spawn_half_size, size=(b, 2)
                )
                z_goal_np = self._rng.uniform(0.0, self._max_goal_height, size=(b,))
                xy_goal = torch.from_numpy(np.asarray(xy_goal_np, dtype=np.float32))
                z_goal = torch.from_numpy(np.asarray(z_goal_np, dtype=np.float32))
                goal_xyz = torch.zeros((b, 3))
                goal_xyz[:, 0] = xy_goal[:, 0] + self._cube_spawn_center[0]
                goal_xyz[:, 1] = xy_goal[:, 1] + self._cube_spawn_center[1]
                goal_xyz[:, 2] = z_goal + xyz[:, 2]
                self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))
            # Prime the qpos cache so the Jacobian closure has a valid
            # value at the first safety-filter call. The training loop
            # in ``concerto.training.ego_aht.train`` queries the filter
            # *before* the first ``env.step()`` (act-then-step order),
            # so the per-physics-step ``_before_simulation_step`` hook
            # has not yet fired after a reset. Without this primer the
            # closure raises the "qpos not yet cached" guard.
            self._refresh_latest_qpos()

        # ----- Observation / reward / success (panda-routed; ADR-007 §Stage 1b) -----

        @property
        def _panda_agent(self) -> Any:  # noqa: ANN401 - ManiSkill Panda agent has no project type
            """Return the panda_wristcam :class:`Panda` agent instance (ego).

            Routes through ``self.agent.agents_dict`` instead of the
            single-agent ``self.agent`` attribute upstream PickCubeEnv
            relies on. Used by :meth:`evaluate` and
            :meth:`compute_normalized_dense_reward` so the success
            predicates and reward shaping respect the multi-agent rig.
            """
            return self.agent.agents_dict["panda_wristcam"]  # type: ignore[attr-defined, no-any-return]

        def _get_obs_extra(self, info: dict[str, Any]) -> dict[str, Any]:
            """Inject task obs + synthesised force-torque (ADR-007 §Stage 1b).

            Extras include:

            - ``tcp_pose`` — panda end-effector pose (mirrors upstream
              PickCubeEnv).
            - ``goal_pos`` — the goal site's xyz position.
            - ``cube_pose`` / ``cube_to_tcp_pos`` / ``cube_to_goal_pos``
              — present only when ``"state"`` is in the env's
              ``obs_mode`` (matches upstream PickCubeEnv's branch).
            - **``force_torque`` (this env's contribution)** — a 6-D
              Cartesian force vector per panda fingertip
              (``[fx_l, fy_l, fz_l, fx_r, fy_r, fz_r]``), synthesised
              from
              :meth:`PhysxArticulation.get_net_contact_forces([finger_links])`
              plus zero-mean sigma=:data:`SYNTHESISED_FT_NOISE_SIGMA_N`
              Gaussian noise routed through the env's P6-harnessed RNG.
              **Not** a real Panda wrist FT sensor reading; see module
              docstring and ADR-007 §Open questions for the wrist-mount
              ablation deferral.
            """
            del info
            panda = self._panda_agent
            obs: dict[str, Any] = {
                "tcp_pose": panda.tcp_pose.raw_pose,
                "goal_pos": self.goal_site.pose.p,
            }
            if "state" in self.obs_mode:  # type: ignore[operator]
                obs["cube_pose"] = self.cube.pose.raw_pose
                obs["cube_to_tcp_pos"] = self.cube.pose.p - panda.tcp_pose.p
                obs["cube_to_goal_pos"] = self.goal_site.pose.p - self.cube.pose.p
            # Synthesised FT: always injected so the OM channel-filter
            # wrapper has something to keep/zero per condition. The
            # AS conditions' state_dict obs_mode ignores it downstream.
            obs["force_torque"] = self._compute_synthesised_force_torque()
            return obs

        def _compute_synthesised_force_torque(self) -> NDArray[np.float32]:
            """Synthesise the 6-D fingertip-contact FT vector (ADR-007 §Stage 1b)."""
            panda = self._panda_agent
            fingers = ["panda_leftfinger", "panda_rightfinger"]
            try:
                # ``get_net_contact_forces`` returns shape ``(N_envs, n_links, 3)``
                # (Cartesian force per link) at SAPIEN's contact-manager
                # boundary (mani_skill/utils/structs/articulation.py:490).
                forces = panda.robot.get_net_contact_forces(fingers)
            except (RuntimeError, KeyError):
                # Finger links not yet registered (pre-reset state) —
                # return zeros so the wrapper-chain shape stays stable.
                return np.zeros((1, 6), dtype=np.float32)
            f_np = np.asarray(forces.detach().cpu(), dtype=np.float32)  # type: ignore[union-attr]
            # Flatten the per-link Cartesian force into a 6-D vector;
            # add zero-mean Gaussian noise from the P6-harnessed RNG so
            # the OM-hetero gate's signal-to-noise lands at the
            # documented Franka-spec lower bound.
            flat = f_np.reshape(f_np.shape[0], -1)  # (N_envs, 6)
            if self._force_torque_noise_sigma > 0.0:
                noise = self._rng.normal(
                    loc=0.0,
                    scale=self._force_torque_noise_sigma,
                    size=flat.shape,
                ).astype(np.float32)
                flat = flat + noise
            return flat

        def evaluate(self) -> dict[str, Any]:
            """ADR-007 §Stage 1b: multi-agent success predicate (panda-routed)."""
            panda = self._panda_agent
            is_obj_placed = (
                torch.linalg.norm(self.goal_site.pose.p - self.cube.pose.p, axis=1)
                <= self._goal_thresh
            )
            is_grasped = panda.is_grasping(self.cube)
            is_robot_static = panda.is_static(0.2)
            return {
                "success": is_obj_placed & is_robot_static,
                "is_obj_placed": is_obj_placed,
                "is_robot_static": is_robot_static,
                "is_grasped": is_grasped,
            }

        def compute_normalized_dense_reward(
            self,
            obs: Any,  # noqa: ANN401 - ManiSkill obs is an unconstrained dict-of-tensors
            action: Any,  # noqa: ANN401 - ManiSkill action is an unconstrained dict-of-tensors
            info: dict[str, Any],
        ) -> Any:  # noqa: ANN401 - upstream returns torch tensor, no project type
            """ADR-007 §Stage 1b: multi-agent normalised dense reward (panda-routed).

            Matches the upstream ``PickCubeEnv.compute_dense_reward`` shape
            (reaching + grasp + place * grasp + static * placed + 5*success)
            divided by 5. The reward signal is panda-centric (the panda
            does the picking); the fetch / partner is graded implicitly
            through whether it gets out of the way / cooperates on
            placement — explicit fetch-side reward shaping is a P1.04
            follow-up.
            """
            del obs, action
            panda = self._panda_agent
            tcp_to_obj_dist = torch.linalg.norm(self.cube.pose.p - panda.tcp_pose.p, axis=1)
            reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
            reward = reaching_reward
            is_grasped = info["is_grasped"]
            reward = reward + is_grasped
            obj_to_goal_dist = torch.linalg.norm(self.goal_site.pose.p - self.cube.pose.p, axis=1)
            place_reward = 1 - torch.tanh(5 * obj_to_goal_dist)
            reward = reward + place_reward * is_grasped
            qvel = panda.robot.get_qvel()
            qvel = qvel[..., :-2]  # drop the two gripper-finger DOF
            static_reward = 1 - torch.tanh(5 * torch.linalg.norm(qvel, axis=1))
            reward = reward + static_reward * info["is_obj_placed"]
            reward[info["success"]] = 5
            return reward / 5

        # ----- Per-step qpos refresh (Jacobian closure feed) -----

        def _refresh_latest_qpos(self) -> None:
            """Snapshot every panda uid's arm qpos into :attr:`_latest_qpos`.

            P1.04.5; ADR-007 §Stage 1b. Called from both
            :meth:`_before_simulation_step` (each physics step) and
            :meth:`_initialize_episode` (post-reset primer). The latter
            primes the cache between ``env.reset()`` and the first
            ``env.step()`` so the safety filter — which the training
            loop runs *before* the first step — can read a valid qpos
            without raising the "qpos not yet cached" guard.
            """
            for uid in self.robot_uids:  # type: ignore[union-attr]
                if uid in ("panda_wristcam", "panda_partner"):
                    agent = self.agent.agents_dict[uid]  # type: ignore[attr-defined]
                    full_qpos = agent.robot.get_qpos()
                    # Slice off the 2 gripper-finger joints; the
                    # Jacobian chain to panda_hand_tcp covers the 7 arm
                    # joints only.
                    qpos_arm = np.asarray(
                        full_qpos.detach().cpu(),
                        dtype=np.float64,  # type: ignore[union-attr]
                    ).reshape(-1)[: self._panda_arm_dof]
                    self._latest_qpos[uid] = qpos_arm

        def _before_simulation_step(self) -> None:
            """Cache panda arm qpos before each physics step (Jacobian closure feed).

            The :class:`JacobianControlModel` callable wired by
            :meth:`build_control_models` reads
            ``self._latest_qpos["panda_wristcam"]`` to compute the
            Jacobian at the current configuration. ManiSkill's
            :class:`BaseEnv` calls this hook once per physics step
            before SAPIEN advances; updating the cache here gives the
            closure access to the most recent qpos at the next safety-
            filter invocation. See module docstring on the closure-
            via-env-reference workaround (P1.05.5 contingent
            follow-up).
            """
            super()._before_simulation_step()
            self._refresh_latest_qpos()

        # ----- Public API for the safety-stack consumer -----

        @property
        def condition_id(self) -> str:
            """ADR-007 §Stage 1b: the :attr:`ConditionConfig.condition_id` this env was built for."""  # noqa: E501
            return self._condition_config.condition_id

        @property
        def condition_config(self) -> ConditionConfig:
            """The resolved :class:`ConditionConfig` for read-only consumers (ADR-007 §Stage 1b)."""
            return self._condition_config

        @property
        def ego_uid(self) -> str:
            """ADR-007 §Stage 1b: the ego uid for the active condition.

            First element of :attr:`ConditionConfig.agent_uids` by the
            Phase-0 ``_CONDITION_UIDS`` convention. Consumed by the
            Phase-1 :class:`chamber.benchmarks.stage1_common.TrainedPolicyFactory`
            (P1.04) so the factory can read the ego uid from the env
            rather than from a side-table — single source of truth at
            the env layer.
            """
            return self._condition_config.agent_uids[0]

        @property
        def partner_uid(self) -> str:
            """ADR-007 §Stage 1b: the partner uid for the active condition.

            Second element of :attr:`ConditionConfig.agent_uids` (the
            ego is the first; the partner is the second by the Phase-0
            ``_CONDITION_UIDS`` convention from
            :mod:`chamber.benchmarks.stage1_as` /
            :mod:`chamber.benchmarks.stage1_om`). Consumed by the
            Phase-1 :class:`chamber.benchmarks.stage1_common.TrainedPolicyFactory`
            (P1.04) to route the per-condition partner-stack dispatch
            (``panda_partner`` for AS-homo; ``fetch`` for AS-hetero /
            OM-*). Single source of truth at the env layer — future
            Stage-2/3 envs (P1.06+) follow the same property contract
            without needing a factory-side dispatch table.
            """
            return self._condition_config.agent_uids[1]

        def build_control_models(self) -> Mapping[str, AgentControlModel]:
            """Return per-uid :class:`AgentControlModel` map (ADR-004 §Decision).

            Plumbed in by
            :func:`chamber.benchmarks.training_runner.build_control_models`
            via the ``getattr(env, "build_control_models", None)``
            delegation route (P1.03 / Task B).

            The Jacobian callable is the closure-via-env-reference
            workaround documented in the module docstring: it captures
            ``self`` and reads ``self._latest_qpos["panda_wristcam"]``
            at call time. The closure stays valid across the 20
            evaluation episodes within each ``(seed, condition)`` cell
            because
            :func:`chamber.benchmarks.stage1_as._run_axis_with_factories`
            builds the env once per cell and reuses it
            (``stage1_as.py:253-275``).

            For the AS-homo condition both panda uids share the same
            Jacobian provider (their kinematic chains are identical;
            see :class:`chamber.agents.panda_jacobian.PandaJacobianProvider`'s
            choice of URDF). The closure resolves to the correct uid's
            latest qpos at call time.
            """
            # Source the panda action dim from the action space (same
            # path the fetch dim is sourced from below). Falls back to
            # 8 — the ``pd_joint_delta_pos`` default (7 arm + 1 mimic-
            # gripper). The closure captures this value once at
            # build-time, which is correct: action-space shape is fixed
            # over an env's lifetime.
            panda_action_dim = 8
            action_space = self.action_space
            if isinstance(action_space, gym.spaces.Dict):
                sub_panda = action_space.spaces.get("panda_wristcam")
                if isinstance(sub_panda, gym.spaces.Box) and sub_panda.shape is not None:
                    panda_action_dim = int(sub_panda.shape[0])

            def _panda_jacobian(snap: AgentSnapshot) -> NDArray[np.float64]:
                # Pull the live qpos from the env's cache; the closure
                # captures self by reference so the latest pre-step
                # value is always used.
                qpos = self._latest_qpos.get("panda_wristcam")
                if qpos is None:
                    msg = (
                        "Stage1PickPlaceEnv: panda_wristcam qpos not yet cached. "
                        "The JacobianControlModel closure was invoked before "
                        "the first env.step() — reset the env first."
                    )
                    raise RuntimeError(msg)
                # PandaJacobianProvider returns the linear (3, 7) Jacobian
                # for the 7-joint chain to ``panda_hand_tcp``. The
                # ``pd_joint_delta_pos`` control mode's panda action is
                # 8-D (7 arm deltas + 1 mimic-gripper delta — see the
                # ``panda_action_dim=8`` default in
                # :func:`build_control_models_for_condition`). Pad with
                # a zero column so the JacobianControlModel matmul
                # ``jac @ action`` composes: the mimic-gripper joint is
                # a sibling of ``panda_hand``, not in the TCP kinematic
                # chain, so its contribution to TCP Cartesian position
                # is zero by construction.
                jac_3x7 = self._jacobian_provider(snap, qpos)
                jac_padded = np.zeros((CARTESIAN_POSITION_DIM, panda_action_dim), dtype=np.float64)
                jac_padded[:, : self._panda_arm_dof] = jac_3x7
                return jac_padded

            fetch_uid_dim: int | None = None
            partner_uid = self._condition_config.agent_uids[1]
            if partner_uid == "fetch" and isinstance(action_space, gym.spaces.Dict):
                sub = action_space.spaces.get(partner_uid)
                if isinstance(sub, gym.spaces.Box) and sub.shape is not None:
                    fetch_uid_dim = int(sub.shape[0])
            return build_control_models_for_condition(
                self._condition_config.condition_id,
                jacobian_fn=_panda_jacobian,
                fetch_action_dim=fetch_uid_dim,
                panda_action_dim=panda_action_dim,
            )

        # ----- P1.04.5: safety-stack snapshot + bounds wiring -----

        def build_agent_snapshots(self) -> dict[str, AgentSnapshot]:
            """Build per-uid Cartesian :class:`AgentSnapshot` map (P1.04.5; ADR-007 §Stage 1b).

            Reads each agent's Cartesian position (TCP pose for the
            pandas; base-link pose for the fetch) and computes velocity
            via numerical-difference against the cached
            :attr:`_prev_positions` from the previous step. First-step
            velocity defaults to zero on every fresh episode (the
            agents start at rest by construction; the cache is cleared
            in :meth:`_initialize_episode`).

            Used by the P1.04.5 safety-stack integration: the
            :func:`concerto.training.ego_aht.train` loop calls this
            once per step (when the safety filter is wired) to build
            the snapshot map the CBF-QP outer filter consumes via the
            ``partner_predicted_states`` kwarg + the conformal update's
            ``snaps_now`` / ``snaps_prev`` pair.

            Returns:
                ``{uid: AgentSnapshot(position, velocity, radius)}`` for
                every uid in :attr:`ConditionConfig.agent_uids`. Position
                is the 3-D Cartesian centre (panda TCP for the pandas;
                fetch base origin for the fetch); velocity is the
                numerical-difference against the previous call (zero on
                first call after :meth:`_initialize_episode`); radius
                is from :data:`PANDA_SAFETY_RADIUS_M` /
                :data:`FETCH_SAFETY_RADIUS_M` (validated at env
                construction by :func:`_validate_safety_radii`).
            """
            snapshots: dict[str, AgentSnapshot] = {}
            for uid in self._condition_config.agent_uids:
                agent = self.agent.agents_dict[uid]  # type: ignore[attr-defined]
                if uid in ("panda_wristcam", "panda_partner"):
                    # Panda: TCP pose. Squeeze the (num_envs=1, 3)
                    # batch dim to a flat 3-vector for the AgentSnapshot.
                    pos_raw = agent.tcp_pose.p
                else:
                    # Fetch: base-link pose (the robot's root pose).
                    pos_raw = agent.robot.get_pose().p
                position = np.asarray(
                    pos_raw.detach().cpu(),
                    dtype=np.float64,  # type: ignore[union-attr]
                ).reshape(-1)
                # ManiSkill emits (num_envs, 3) for each agent pose; the
                # ravel above flattens (1, 3) -> (3,) for num_envs=1, but
                # multi-env runs would emit (num_envs, 3) which ravels to
                # (3*num_envs,). The Stage-1b cell construction pins
                # num_envs=1 (factory contract); the slice is a defensive
                # guard against future multi-env builds.
                _cartesian_dim = 3  # ADR-004 §Decision: AgentSnapshot is 3-D Cartesian.
                if position.shape[0] > _cartesian_dim:
                    position = position[:_cartesian_dim]
                prev = self._prev_positions.get(uid)
                if prev is None:
                    velocity = np.zeros(3, dtype=np.float64)
                else:
                    # Numerical-difference velocity. The dt is the
                    # control_timestep; for the AgentSnapshot's contract
                    # this is the per-step Cartesian velocity. The
                    # safety filter's conformal update consumes this
                    # value alongside dt to compute the predicted
                    # pose; the consistency is the caller's
                    # responsibility (see
                    # chamber.benchmarks.training_runner.build_safety_dt).
                    velocity = (position - prev) / float(self.control_timestep)
                self._prev_positions[uid] = position
                snapshots[uid] = AgentSnapshot(
                    position=position,
                    velocity=velocity,
                    radius=self._safety_radii[uid],
                )
            return snapshots

        def build_bounds(self) -> Bounds:
            """Build the per-task :class:`Bounds` envelope (P1.04.5; ADR-006 §Decision).

            Sourced from the module-level constants pinned in P1.03
            (cartesian_accel_capacity) plus sensible Phase-1 defaults
            for the remaining fields. Consumed by the safety filter +
            the audit-gate predicate A's RHS (``cartesian_accel_capacity``).
            """
            return Bounds(
                action_linf_component=0.1,
                cartesian_accel_capacity=PANDA_CARTESIAN_ACCEL_CAPACITY_MS2,
                action_rate=10.0,
                comm_latency_ms=0.0,
                force_limit=50.0,
            )

    inner = Stage1PickPlaceEnv(
        condition_id=condition_id,
        episode_length=episode_length,
        root_seed=root_seed,
        num_envs=num_envs,
        render_mode=render_mode,
        render_backend=render_backend,
        force_torque_noise_sigma=force_torque_noise_sigma,
    )
    # AS conditions: ManiSkill v3 obs_mode="state_dict" emits per-agent
    # Dict(qpos, qvel) but EgoPPOTrainer.from_config reads
    # obs["agent"][ego_uid]["state"]. The AS synthesizer adds that key;
    # pass-through for OM conditions. The OM channel filter then masks
    # per-condition for OM-vision-only; pass-through for AS and OM-hetero.
    # Wired in fixed order so OM callers of the factory don't have to
    # wrap manually (issue #165 §Proposed scope 4).
    from chamber.envs.stage1_obs_filter import (
        Stage1ASStateSynthesizer,
        Stage1OMChannelFilter,
    )

    return Stage1OMChannelFilter(Stage1ASStateSynthesizer(inner))


__all__ = [
    "DEFAULT_EPISODE_LENGTH",
    "FETCH_CARTESIAN_ACCEL_CAPACITY_MS2",
    "PANDA_CARTESIAN_ACCEL_CAPACITY_MS2",
    "SYNTHESISED_FT_NOISE_SIGMA_N",
    "ConditionConfig",
    "build_control_models_for_condition",
    "make_stage1_pickplace_env",
    "resolve_condition",
]
