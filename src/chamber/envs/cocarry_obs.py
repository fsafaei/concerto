# SPDX-License-Identifier: Apache-2.0
# pyright: reportPrivateImportUsage=false
#
# Same suppression rationale as :mod:`chamber.envs.stage1_obs_filter`:
# ``torch.Tensor`` detach/cpu is exported but not advertised in torch's
# stub ``__all__``. Suppressed file-locally so the synthesised-state
# helper stays free of per-line ``type: ignore`` noise.
r"""Co-carry training observation wrapper (ADR-026 §Decision 1; ADR-007 §Stage 1b).

Rung 2 (R-2026-06-B §15) trains a learned ego against the frozen matched
:class:`chamber.partners.cocarry_impedance.CoCarryImpedancePartner`. The
trainer (:class:`chamber.benchmarks.ego_ppo_trainer.EgoPPOTrainer`) reads
the ego's flat policy input from ``obs["agent"][ego_uid]["state"]`` and
sizes its actor/critic from
``env.observation_space["agent"][ego_uid]["state"].shape[-1]``. The
co-carry env (:mod:`chamber.envs.cocarry`, ``obs_mode="state_dict"``)
emits per-agent ``Dict(qpos, qvel, ...)`` with **no** top-level
``"state"`` key, so this wrapper synthesises it — exactly the bridge
:class:`chamber.envs.stage1_obs_filter.Stage1ASStateSynthesizer` provides
for the Stage-1 AS conditions.

Ego ``state`` concatenation order (load-bearing for checkpoint-diff
reproducibility and the Tier-1 positional regression test)::

    [ego_qpos, ego_qvel, partner_qpos, partner_qvel, bar_pose, goal_pos]

This mirrors the Stage-1 AS widening (``[ego/partner qpos+qvel, cube_pose,
goal_pos, tcp_pose]``) adapted to the co-carry task: the shared **bar
pose** (7-D: xyz + wxyz quaternion) replaces the cube pose, and there is
no separate TCP-pose field (the bar is rigidly welded to both grippers,
so the bar pose already carries the cooperative state). The bar tilt and
wrist-stress proxy the env also exposes under ``obs["extra"]`` are
**recoverable** from ``bar_pose`` (the quaternion gives tilt) and are left
out of the policy input for parity with the Stage-1 widening rationale
(no linearly-dependent fields). The non-ego (partner) ``state`` is
``concat(qpos, qvel)`` of the partner alone — the partner does not see the
task (ADR-009 §Decision; the frozen impedance controller reads
``obs["extra"]["goal_pos"]`` + ``obs["agent"][uid]["qpos"]`` from the
**raw** leaves this wrapper leaves untouched, not from ``state``).

Pass-through contract: this wrapper **only adds** the ``state`` key to
each agent sub-dict; every raw leaf (``qpos`` / ``qvel`` / the whole
``obs["extra"]``) is preserved byte-for-byte so the matched
:class:`CoCarryImpedancePartner` on the partner seat keeps reading the
same obs it did at Rungs 0-1.

Batch contract: the Rung-2 training cell is **single-env** (``num_envs=1``)
because :class:`CoCarryImpedancePartner` reads env 0 only; the helpers
ravel the batch-1 ManiSkill obs to the historical flat 1-D ``state``
vector. The ``(num_envs, dim)`` batched path is supported for symmetry
with the Stage-1 wrapper but is not exercised by the co-carry cell.

References:
- ADR-026 §Decision 1 (the coupling-valid co-carry task).
- ADR-007 §Stage 1b (the trainer-obs ``state`` contract; the AS
  synthesiser this mirrors).
- ADR-009 §Decision (frozen black-box partner; partner reads raw leaves).
- R-2026-06-B §15 Rung 2 (the learned-incumbent design this serves).
- :class:`chamber.envs.stage1_obs_filter.Stage1ASStateSynthesizer` (the
  parallel Stage-1 AS implementation).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import gymnasium as gym
import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray

#: Task-extra fields the synthesiser injects into the ego's flat
#: ``state``, with their expected trailing dims. Order is load-bearing
#: (matches the concat order in the module docstring + the Tier-1
#: positional test). ``bar_pose`` is 7-D (xyz + wxyz quaternion, the
#: ManiSkill ``Pose.raw_pose`` layout); ``goal_pos`` is the 3-D goal
#: centroid.
_COCARRY_TASK_EXTRA_FIELDS: tuple[tuple[str, int], ...] = (
    ("bar_pose", 7),
    ("goal_pos", 3),
)

#: The ego-state partner block layout the FROZEN Rung-2 incumbent was trained
#: with: a Panda partner's 7 arm + 2 gripper joints (9 per qpos/qvel, 18
#: total). The incumbent's 46-D actor input is fixed at this layout, so a
#: non-Panda partner (Rung-4 embodiment heterogeneity; the 12-DOF xArm6) must
#: be ADAPTED into it — else the frozen actor/critic cannot load. For the
#: Panda partner the adapter is a no-op so the reference stays byte-identical
#: (see :func:`_adapt_partner_vec`; ADR-026 §Decision 4; R-2026-06-B §15 Rung 4).
_PARTNER_EGO_ARM_DOF: int = 7
_PARTNER_EGO_GRIP_DOF: int = 2
#: Per-field (qpos or qvel) width of the adapted partner block (= 9).
_PARTNER_EGO_BLOCK_PER_FIELD: int = _PARTNER_EGO_ARM_DOF + _PARTNER_EGO_GRIP_DOF


def _to_2d_float32(arr: Any, *, source: str) -> NDArray[np.float32]:  # noqa: ANN401 - torch.Tensor or np.ndarray
    """Coerce ``arr`` to a ``(batch, dim)`` float32 ndarray (ADR-026 §Decision 1).

    Handles torch tensors (real SAPIEN env emits them under
    ``obs_mode="state_dict"``) and numpy arrays (Tier-1 fakes). 1-D
    inputs become ``(1, dim)``; 2-D inputs pass through with the leading
    batch dim intact; rank > 2 is rejected loudly. Mirrors
    :func:`chamber.envs.stage1_obs_filter._to_2d_float32`.
    """
    try:
        import torch  # noqa: PLC0415 - lazy: Tier-1 import safety (module docstring)

        if isinstance(arr, torch.Tensor):
            arr = arr.detach().cpu().numpy()
    except ImportError:  # pragma: no cover - torch is a hard project dep
        pass
    arr_np = np.asarray(arr)
    if arr_np.ndim > 2:  # noqa: PLR2004 - rank-2 is the (batch, dim) contract
        msg = (
            f"CoCarryEgoStateSynthesizer: {source} has shape {arr_np.shape}; "
            "expected a 1-D per-env vector or a (num_envs, dim) batch. "
            "Rank > 2 would silently mis-pack the state vector "
            "(ADR-026 §Decision 1; ADR-007 §Stage 1b)."
        )
        raise TypeError(msg)
    if arr_np.ndim == 1:
        arr_np = arr_np.reshape(1, -1)
    return arr_np.astype(np.float32)


def _batched_concat(parts: list[NDArray[np.float32]], *, sources: list[str]) -> NDArray[np.float32]:
    """Concat ``(batch, dim_i)`` parts along the trailing axis (ADR-026 §Decision 1).

    All parts must share the same leading batch dim (a mismatch is a
    wiring bug — loud-fail naming the offenders). Returns flat 1-D when
    the batch dim is 1 (the single-env co-carry cell, byte-identical to
    a ``ravel``), else ``(batch, sum_dims)``. Mirrors
    :func:`chamber.envs.stage1_obs_filter._batched_concat`.
    """
    batch_dims = {int(p.shape[0]) for p in parts}
    if len(batch_dims) != 1:
        shaped = ", ".join(f"{s}={p.shape}" for s, p in zip(sources, parts, strict=True))
        msg = (
            "CoCarryEgoStateSynthesizer: inconsistent leading batch dims across "
            f"state-concat inputs ({shaped}). All per-uid and task-extra fields "
            "must share the env's num_envs batch (ADR-026 §Decision 1)."
        )
        raise TypeError(msg)
    out = np.concatenate(parts, axis=-1)
    if out.shape[0] == 1:
        return out.ravel()
    return out


def _fix_width(arr: NDArray[np.float32], target: int) -> NDArray[np.float32]:
    """Zero-pad (if narrower) or truncate (if wider) a ``(batch, dim)`` array to ``target`` cols."""
    cur = int(arr.shape[-1])
    if cur == target:
        return arr
    if cur > target:
        return arr[:, :target]
    pad = np.zeros((int(arr.shape[0]), target - cur), dtype=np.float32)
    return np.concatenate([arr, pad], axis=-1)


def _adapt_partner_vec(
    vec: Any,  # noqa: ANN401 - torch.Tensor / ndarray
    *,
    arm_dof: int,
) -> NDArray[np.float32]:
    """Adapt a partner qpos/qvel into the fixed Panda ego-state layout (ADR-026 §D4; Rung 4).

    Splits ``vec`` into the partner's ``arm_dof`` arm joints + the remaining
    gripper joints, then maps each to the frozen incumbent's expected layout
    (:data:`_PARTNER_EGO_ARM_DOF` arm + :data:`_PARTNER_EGO_GRIP_DOF` gripper,
    zero-padded / truncated). For the Panda partner (``arm_dof=7``, 7 arm + 2
    finger) this is the identity, so the matched reference's ego/partner state
    is byte-identical; for the 6-DOF xArm6 (``arm_dof=6``, 6 arm + 6 Robotiq)
    it embeds the 6 arm joints (+1 zero) and the first 2 gripper joints into
    the 9-D slot. Returns ``(batch, 9)`` float32. The bar pose — the actual
    cooperative coupling channel — flows through unchanged; this adapter only
    bridges the partner *proprioception* interface the frozen ego fixes.
    """
    arr = _to_2d_float32(vec, source="partner")
    arm = _fix_width(arr[:, :arm_dof], _PARTNER_EGO_ARM_DOF)
    grip = _fix_width(arr[:, arm_dof:], _PARTNER_EGO_GRIP_DOF)
    return np.concatenate([arm, grip], axis=-1)


def _partner_state_concat(qpos: Any, qvel: Any, *, arm_dof: int) -> NDArray[np.float32]:  # noqa: ANN401
    """Concat adapted ``qpos`` + ``qvel`` into the non-ego ``state`` (ADR-026 §D4; ADR-009)."""
    return _batched_concat(
        [_adapt_partner_vec(qpos, arm_dof=arm_dof), _adapt_partner_vec(qvel, arm_dof=arm_dof)],
        sources=["qpos", "qvel"],
    )


def _ego_state_concat(
    ego_qpos: Any,  # noqa: ANN401
    ego_qvel: Any,  # noqa: ANN401
    partner_qpos: Any,  # noqa: ANN401
    partner_qvel: Any,  # noqa: ANN401
    bar_pose: Any,  # noqa: ANN401
    goal_pos: Any,  # noqa: ANN401
    *,
    partner_arm_dof: int,
) -> NDArray[np.float32]:
    """Concat the ego's flat ``state`` (ADR-026 §Decision 1; R-2026-06-B §15 Rung 2/4).

    Order: ``[ego_qpos, ego_qvel, partner_qpos, partner_qvel, bar_pose,
    goal_pos]`` — load-bearing (see the module docstring + the Tier-1
    positional regression test). The partner qpos/qvel are ADAPTED into the
    fixed Panda 18-D layout (:func:`_adapt_partner_vec`) so the frozen
    incumbent's 46-D actor input is dimension-stable across partner
    embodiments; identity for the Panda partner (Rung-4 EH). 1-D for the
    single-env co-carry cell.
    """
    sources = [
        "ego.qpos",
        "ego.qvel",
        "partner.qpos",
        "partner.qvel",
        "extra.bar_pose",
        "extra.goal_pos",
    ]
    return _batched_concat(
        [
            _to_2d_float32(ego_qpos, source="ego.qpos"),
            _to_2d_float32(ego_qvel, source="ego.qvel"),
            _adapt_partner_vec(partner_qpos, arm_dof=partner_arm_dof),
            _adapt_partner_vec(partner_qvel, arm_dof=partner_arm_dof),
            _to_2d_float32(bar_pose, source="extra.bar_pose"),
            _to_2d_float32(goal_pos, source="extra.goal_pos"),
        ],
        sources=sources,
    )


def _resolve_qpos_qvel_dims(agent: gym.spaces.Dict, *, uid: str, role: str) -> tuple[int, int]:
    """Resolve and validate (qpos_dim, qvel_dim) for ``uid`` (ADR-026 §Decision 1).

    Loud-fails naming the missing/wrong-typed key, the uid, and the role
    so a future operator who breaks the contract surfaces it at env
    construction. Mirrors
    :func:`chamber.envs.stage1_obs_filter._resolve_qpos_qvel_dims`.
    """
    sub = agent.spaces.get(uid)
    if not isinstance(sub, gym.spaces.Dict):
        msg = (
            f"CoCarryEgoStateSynthesizer: missing {role} agent sub-Dict for uid={uid!r}. "
            "Required to compose the per-uid 'state' Box (ADR-026 §Decision 1)."
        )
        raise TypeError(msg)
    qpos = sub.spaces.get("qpos")
    qvel = sub.spaces.get("qvel")
    if not isinstance(qpos, gym.spaces.Box) or not isinstance(qvel, gym.spaces.Box):
        msg = (
            f"CoCarryEgoStateSynthesizer: {role} uid={uid!r} sub-Dict missing "
            "Box-typed 'qpos' / 'qvel' entries (ADR-026 §Decision 1)."
        )
        raise TypeError(msg)
    return int(qpos.shape[-1]), int(qvel.shape[-1])


def _validate_and_sum_task_extras(inner: gym.spaces.Dict) -> int:
    """Validate ``obs["extra"]`` carries bar_pose + goal_pos; return their total trailing dim.

    ADR-026 §Decision 1. Mirrors
    :func:`chamber.envs.stage1_obs_filter._validate_and_sum_task_extras`.
    """
    extra = inner.spaces.get("extra")
    if not isinstance(extra, gym.spaces.Dict):
        msg = (
            "CoCarryEgoStateSynthesizer: inner env's observation_space is missing a "
            "Dict 'extra' sub-space. The ego state contract requires bar_pose / "
            "goal_pos under obs['extra'] (ADR-026 §Decision 1)."
        )
        raise TypeError(msg)
    total = 0
    for field_name, expected_last_dim in _COCARRY_TASK_EXTRA_FIELDS:
        sub = extra.spaces.get(field_name)
        if not isinstance(sub, gym.spaces.Box):
            msg = (
                f"CoCarryEgoStateSynthesizer: missing or wrong-typed task field "
                f"'extra.{field_name}' (expected gym.spaces.Box with trailing dim "
                f"{expected_last_dim}; ADR-026 §Decision 1)."
            )
            raise TypeError(msg)
        if not sub.shape or int(sub.shape[-1]) != expected_last_dim:
            msg = (
                f"CoCarryEgoStateSynthesizer: task field 'extra.{field_name}' has shape "
                f"{sub.shape} but expected trailing dim {expected_last_dim} "
                "(ADR-026 §Decision 1)."
            )
            raise TypeError(msg)
        total += expected_last_dim
    return total


class CoCarryEgoStateSynthesizer(gym.ObservationWrapper):  # type: ignore[type-arg]
    """Synthesise ``obs["agent"][uid]["state"]`` for the co-carry env (ADR-026 §Decision 1).

    Bridges the co-carry env's ``obs_mode="state_dict"`` layout (per-agent
    ``Dict(qpos, qvel, ...)``, no ``"state"`` key) to
    :meth:`chamber.benchmarks.ego_ppo_trainer.EgoPPOTrainer.from_config`'s
    ``obs["agent"][ego_uid]["state"]`` contract. The ego ``state`` is the
    widened ``[ego_qpos, ego_qvel, partner_qpos, partner_qvel, bar_pose,
    goal_pos]`` concat; the partner ``state`` is ``concat(qpos, qvel)``.
    All raw leaves are preserved so the frozen matched partner reads the
    same obs (ADR-009 §Decision).

    The inner env must expose ``ego_uid`` / ``partner_uid`` attributes
    (the :class:`chamber.envs.cocarry.CoCarryEnv` properties) and a
    :class:`gym.spaces.Dict` observation space.

    Raises:
        TypeError: If ``env.observation_space`` is not a
            :class:`gym.spaces.Dict`, or the required ``obs["extra"]``
            task fields / per-uid qpos/qvel are missing or mistyped. The
            message names the offending key and cites ADR-026 §Decision 1.
    """

    def __init__(
        self, env: gym.Env[Any, Any], *, partner_arm_dof: int = _PARTNER_EGO_ARM_DOF
    ) -> None:
        """Detect uids + build the widened observation space (ADR-026 §Decision 1; §D4).

        Args:
            env: The inner co-carry env.
            partner_arm_dof: The partner's arm DOF, used to adapt its
                proprioception into the frozen incumbent's fixed Panda
                ego-state layout (7 for the Panda partner — a no-op keeping
                the reference byte-identical; 6 for the xArm6, Rung-4 EH).
        """
        super().__init__(env)
        if not isinstance(env.observation_space, gym.spaces.Dict):
            msg = (
                "CoCarryEgoStateSynthesizer requires a gym.spaces.Dict observation "
                f"space; got {type(env.observation_space).__name__} (ADR-026 §Decision 1)."
            )
            raise TypeError(msg)
        self._ego_uid: str = str(env.get_wrapper_attr("ego_uid"))
        self._partner_uid: str = str(env.get_wrapper_attr("partner_uid"))
        self._partner_arm_dof: int = int(partner_arm_dof)
        self.observation_space = self._build_observation_space(env.observation_space)

    def _build_observation_space(self, inner: gym.spaces.Dict) -> gym.spaces.Dict:
        """Inject the per-agent ``state`` Box (ADR-026 §Decision 1).

        Ego ``state`` Box shape ``(ego_qpos + ego_qvel + partner_qpos +
        partner_qvel + 7 + 3,)``; non-ego entries keep the per-uid
        ``concat(qpos, qvel)`` shape. Mirrors the inner space's batching
        (flat 1-D for the single-env cell).
        """
        agent = inner.spaces.get("agent")
        if not isinstance(agent, gym.spaces.Dict):
            msg = (
                "CoCarryEgoStateSynthesizer: inner env's observation_space is missing "
                "a Dict 'agent' sub-space (ADR-026 §Decision 1)."
            )
            raise TypeError(msg)
        ego_qpos_dim, ego_qvel_dim = _resolve_qpos_qvel_dims(agent, uid=self._ego_uid, role="ego")
        # Validate the partner sub-Dict exists, but size its contribution from
        # the FIXED adapted layout (18 = 2 x 9), not the native partner dims, so
        # the ego-state width is embodiment-stable (Rung-4 EH; ADR-026 §D4).
        _resolve_qpos_qvel_dims(agent, uid=self._partner_uid, role="partner")
        partner_block_dim = 2 * _PARTNER_EGO_BLOCK_PER_FIELD
        extra_total_dim = _validate_and_sum_task_extras(inner)
        ego_state_dim = ego_qpos_dim + ego_qvel_dim + partner_block_dim + extra_total_dim

        new_agent_spaces: dict[str, gym.spaces.Space[Any]] = {}
        for uid, sub in agent.spaces.items():
            if isinstance(sub, gym.spaces.Dict):
                qpos = sub.spaces.get("qpos")
                qvel = sub.spaces.get("qvel")
                if isinstance(qpos, gym.spaces.Box) and isinstance(qvel, gym.spaces.Box):
                    # Ego: the 46-D widened state. Partner: the adapted 18-D
                    # block (NOT native dims) so the frozen critic also loads.
                    state_dim = ego_state_dim if uid == self._ego_uid else partner_block_dim
                    qpos_shape = qpos.shape if qpos.shape is not None else (state_dim,)
                    state_shape = (
                        (int(qpos_shape[0]), state_dim)
                        if len(qpos_shape) > 1 and int(qpos_shape[0]) > 1
                        else (state_dim,)
                    )
                    augmented = dict(sub.spaces)
                    augmented["state"] = gym.spaces.Box(
                        low=-np.inf, high=np.inf, shape=state_shape, dtype=np.float32
                    )
                    new_agent_spaces[uid] = gym.spaces.Dict(augmented)
                    continue
            new_agent_spaces[uid] = sub
        new_spaces = dict(inner.spaces)
        new_spaces["agent"] = gym.spaces.Dict(new_agent_spaces)
        return gym.spaces.Dict(new_spaces)

    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:  # type: ignore[override]
        """Inject the ego + partner ``state`` keys, preserving raw leaves (ADR-026 §Decision 1)."""
        agent = observation.get("agent")
        if not isinstance(agent, dict):
            return observation
        extra = observation.get("extra")
        if not isinstance(extra, dict):
            msg = (
                "CoCarryEgoStateSynthesizer: runtime obs is missing the 'extra' sub-dict "
                "required by the ego state contract (bar_pose, goal_pos; "
                "ADR-026 §Decision 1)."
            )
            raise TypeError(msg)
        partner_sub = agent.get(self._partner_uid)
        if (
            not isinstance(partner_sub, dict)
            or "qpos" not in partner_sub
            or "qvel" not in partner_sub
        ):
            msg = (
                f"CoCarryEgoStateSynthesizer: runtime obs missing partner uid "
                f"{self._partner_uid!r} with qpos+qvel keys (ADR-026 §Decision 1; ADR-009)."
            )
            raise TypeError(msg)
        task_arrays: dict[str, Any] = {}
        for field_name, _ in _COCARRY_TASK_EXTRA_FIELDS:
            value = extra.get(field_name)
            if value is None:
                msg = (
                    f"CoCarryEgoStateSynthesizer: runtime obs missing task field "
                    f"'extra.{field_name}' (ADR-026 §Decision 1)."
                )
                raise TypeError(msg)
            task_arrays[field_name] = value

        new_agent: dict[str, Any] = {}
        for uid, sub in agent.items():
            if isinstance(sub, dict) and "qpos" in sub and "qvel" in sub:
                augmented = dict(sub)
                if uid == self._ego_uid:
                    augmented["state"] = _ego_state_concat(
                        sub["qpos"],
                        sub["qvel"],
                        partner_sub["qpos"],
                        partner_sub["qvel"],
                        task_arrays["bar_pose"],
                        task_arrays["goal_pos"],
                        partner_arm_dof=self._partner_arm_dof,
                    )
                else:
                    augmented["state"] = _partner_state_concat(
                        sub["qpos"], sub["qvel"], arm_dof=self._partner_arm_dof
                    )
                new_agent[uid] = augmented
            else:
                new_agent[uid] = sub
        out = dict(observation)
        out["agent"] = new_agent
        return out

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401 - mirrors gym.Wrapper forwarding
        """Forward attribute access to the inner env (Tier-2 caller contract).

        Gymnasium 1.3 removed :class:`gym.Wrapper`'s implicit forwarding;
        co-carry callers read inner-env attributes directly (``ego_uid``,
        ``partner_uid``, ``single_arm``, ``get_telemetry``, ``goal_centroid``).
        Mirrors :meth:`chamber.envs.stage1_obs_filter.Stage1ASStateSynthesizer.__getattr__`.
        """
        if name == "env":
            raise AttributeError(name)
        return getattr(self.env, name)


def make_cocarry_training_env(
    *,
    condition_id: str = "cocarry_matched_panda_pair",
    episode_length: int | None = None,
    root_seed: int = 0,
    num_envs: int = 1,
    render_mode: str | None = None,
    render_backend: str | None = None,
    goal_centroid: tuple[float, float, float] | None = None,
    drive_stiffness: float | None = None,
    drive_damping: float | None = None,
    drive_force_limit: float | None = None,
    stress_measure: str = "wrist",
    stress_max: float | None = None,
    stress_penalty_threshold: float | None = None,
    stress_penalty_scale: float | None = None,
    xarm6_base_x: float | None = None,
    xarm6_ready_qpos: list[float] | None = None,
) -> gym.Env[Any, Any]:
    """Build a co-carry env wrapped for ego-AHT training (ADR-026 §Decision 1; R-2026-06-B §15).

    Composes :func:`chamber.envs.cocarry.make_cocarry_env` with
    :class:`CoCarryEgoStateSynthesizer` so the ego's flat ``state`` channel
    is present for :class:`chamber.benchmarks.ego_ppo_trainer.EgoPPOTrainer`.
    SAPIEN / ManiSkill imports stay deferred inside ``make_cocarry_env``'s
    body (Tier-1 import-safety on Vulkan-less hosts).

    Args:
        condition_id: A co-carry ``condition_id`` (default the matched
            pair; see :func:`chamber.envs.cocarry.resolve_cocarry_condition`).
        episode_length: Truncation horizon; ``None`` uses the env default
            (:data:`chamber.envs.cocarry.COCARRY_DEFAULT_EPISODE_LENGTH`).
        root_seed: Project root seed (P6 / ADR-002).
        num_envs: ManiSkill vectorisation count. The Rung-2 training cell
            uses ``1`` (the matched partner reads env 0).
        render_mode: ManiSkill ``render_mode``, forwarded to the inner factory.
        render_backend: ManiSkill ``render_backend``, forwarded to the inner factory.
        goal_centroid: Optional goal-centroid override.
        drive_stiffness: Dual-hold drive stiffness (N/m); ``None`` ⇒ rigid
            default. Lower ⇒ the Rung-4b compliant coupling (ADR-026 §D4 4b).
        drive_damping: Dual-hold drive damping (N*s/m); ``None`` ⇒ derived.
        drive_force_limit: Max drive force (N); ``None`` ⇒ unbounded (a finite
            value near f_max selects the Variant-B coupling; ADR-026 §D4 4b).
        stress_measure: ``"wrist"`` (default) or ``"coupling"`` (the Rung-4c
            embodiment-invariant bar coupling force; ADR-026 §D4 4c).
        stress_max: Over-stress success ceiling (N); ``None`` ⇒ the wrist
            default (130 N, byte-identical). Drive to the grounded coupling
            f_max (365.6 N) with ``stress_measure="coupling"`` so the trained
            incumbent's reward + success predicate share one measure
            (ADR-026 §D4 4c; the re-freeze ceiling-consistency fix).
        stress_penalty_threshold: Excess-stress penalty soft threshold (N);
            ``None`` ⇒ wrist-grounded 110 N (byte-identical). Re-ground to the
            matched-coupling band for the coupling measure (ADR-026 §Decision 4).
        stress_penalty_scale: Excess-stress penalty tanh scale (N); ``None`` ⇒
            wrist-grounded 20 N (byte-identical).
        xarm6_base_x: Optional xArm6 base-x override (Rung-4d fair-pose).
        xarm6_ready_qpos: Optional xArm6 12-D ready-qpos override (Rung-4d).

    Returns:
        The synthesizer-wrapped co-carry env, ready to ``reset(seed=K)``.
    """
    from chamber.envs.cocarry import (  # noqa: PLC0415 - lazy: Tier-1/SAPIEN import safety
        COCARRY_DEFAULT_EPISODE_LENGTH,
        make_cocarry_env,
    )
    from chamber.envs.errors import (  # noqa: PLC0415 - lazy: Tier-1 import safety
        ChamberEnvCompatibilityError,
    )

    inner = make_cocarry_env(
        condition_id=condition_id,
        episode_length=episode_length
        if episode_length is not None
        else COCARRY_DEFAULT_EPISODE_LENGTH,
        root_seed=root_seed,
        num_envs=num_envs,
        render_mode=render_mode,
        render_backend=render_backend,
        goal_centroid=goal_centroid,
        drive_stiffness=drive_stiffness,
        drive_damping=drive_damping,
        drive_force_limit=drive_force_limit,
        stress_measure=stress_measure,
        stress_max=stress_max,
        stress_penalty_threshold=stress_penalty_threshold,
        stress_penalty_scale=stress_penalty_scale,
        xarm6_base_x=xarm6_base_x,
        xarm6_ready_qpos=xarm6_ready_qpos,
    )
    # Rung 2 trains on the env's dense reward (Rungs 0-1 discarded it), so the
    # reward mode is now load-bearing: fail loudly if a host resolves to a
    # non-``normalized_dense`` mode rather than silently mistraining
    # (R-2026-06-B §15 reward-mode robustness). The co-carry env defines only
    # ``compute_normalized_dense_reward``, so ManiSkill resolves this mode by
    # default; the assert is the guard against a future env/ManiSkill change.
    resolved_reward_mode = getattr(inner, "reward_mode", None)
    if resolved_reward_mode != "normalized_dense":
        inner.close()
        raise ChamberEnvCompatibilityError(
            "make_cocarry_training_env requires reward_mode='normalized_dense' "
            f"(the dense co-carry reward the ego trains on); resolved "
            f"{resolved_reward_mode!r}. See ADR-026 §Decision 1 + R-2026-06-B §15."
        )
    # Partner arm DOF for the ego-state proprioception adapter (Rung-4 EH):
    # the xArm6 is 6-DOF, every other (Panda) partner is 7-DOF. The adapter is
    # a no-op for the Panda partner (the matched reference stays byte-identical).
    partner_uid = str(inner.get_wrapper_attr("partner_uid"))
    partner_arm_dof = 6 if partner_uid == "xarm6_robotiq" else _PARTNER_EGO_ARM_DOF
    return CoCarryEgoStateSynthesizer(inner, partner_arm_dof=partner_arm_dof)


__all__ = ["CoCarryEgoStateSynthesizer", "make_cocarry_training_env"]
