# SPDX-License-Identifier: Apache-2.0
# pyright: reportPrivateImportUsage=false
#
# Same rationale as :mod:`chamber.benchmarks.stage0_smoke`:
# ``torch.as_tensor`` is exported but not advertised in torch's stub
# ``__all__``. Suppressed file-locally so the synthesised-state helper
# stays free of per-line ``type: ignore`` noise.
"""Stage-1 observation wrappers (ADR-007 §Stage 1b; ADR-009 §Decision).

Two wrappers, applied in fixed order by
:func:`chamber.envs.stage1_pickplace.make_stage1_pickplace_env`:

1. :class:`Stage1ASStateSynthesizer` — synthesises
   ``obs["agent"][uid]["state"]`` for AS conditions, bridging ManiSkill
   v3 ``obs_mode="state_dict"`` (which emits per-agent
   ``Dict(qpos, qvel, ...)``) to
   :class:`chamber.benchmarks.ego_ppo_trainer.EgoPPOTrainer`'s
   ``obs["agent"][ego_uid]["state"]`` contract. Pass-through for OM
   conditions and for envs without a recognised ``condition_id``.

   **Ego state widening (P1.05.8; ADR-007 §Stage 1b Rev 12;
   ADR-009 §Decision direct-observation-of-partner-pose).** The ego
   uid's synthesised ``state`` Box concatenates, in this order:

       [ego_qpos, ego_qvel,
        partner_qpos, partner_qvel,
        cube_pose, goal_pos, tcp_pose]

   The order is load-bearing for the audit-trail reproducibility of
   trained-checkpoint diffs and for the positional Tier-1 regression
   test in ``tests/unit/test_stage1_pickplace_tier1.py``. The
   non-ego (partner) entry's ``state`` Box stays at the existing
   ``concat(qpos, qvel)`` of the partner alone — no task fields, no
   cross-agent observation. Only the ego sees the task.

   This widening is the Surface 6 remediation from the failed
   2026-05-20 Stage-1b AS launch (PR #182). Pre-P1.05.8 the
   synthesiser composed the ego's ``state`` Box from the ego's own
   ``concat(qpos, qvel)`` only — cube / goal / TCP / partner state
   were structurally invisible to
   :func:`chamber.benchmarks.ego_ppo_trainer._flat_ego_obs` even
   though the env populated them under ``obs["extra"]`` and the
   partner's sibling subtree. The consultation brief at
   ``spikes/results/stage1-failure-investigation/2026-05-20/CONSULTATION_BRIEF.md``
   §4 Q1 decided to widen at this seam (the trainer's
   ``_flat_ego_obs`` reads ``state``; the wrapper's job is to define
   what ``state`` means for the env). ADR-009 §Decision was
   amended in the same slice to record that direct partner-pose
   observation does not violate the black-box-policy contract — that
   contract is about *policy* access (weights / reward / training
   data, enforced by
   :data:`chamber.partners.interface._FORBIDDEN_ATTRS`), not pose
   visibility. A real robot's cameras / LIDAR / proximity sensors
   see other robots' poses; observing the partner's joint state in
   simulation is the same affordance.

   The relative task fields (``cube_to_tcp_pos``,
   ``cube_to_goal_pos``) are deliberately excluded — they are linear
   combinations of the absolute pose fields already concatenated and
   would inflate the actor's input dimensionality without adding
   rank. The synthesised ``force_torque`` channel is also excluded;
   it is the OM-hetero discriminant and the OM remediation (slice
   P1.05.6 / issue #177) owns it.

   Per-condition ``state`` dim is env-emit-dependent (panda_partner
   under AS-homo vs fetch under AS-hetero) and pinned per-condition
   by the Tier-1 + Tier-2 tests in the P1.05.8 PR. The trainer's
   :meth:`chamber.benchmarks.ego_ppo_trainer.EgoPPOTrainer.from_config`
   reads ``env.observation_space["agent"][ego_uid]["state"].shape[0]``
   at construction time per ``(seed, condition)`` cell, so the
   per-condition divergence does not trip any global constant.

2. :class:`Stage1OMChannelFilter` — per-condition channel keep-set for
   the OM axis. The Stage-1 AS conditions don't filter at all (they
   use ``obs_mode="state_dict"`` which already excludes the camera
   data the OM axis differentiates on); the wrapper passes those
   conditions through untouched.

For the two OM conditions:

- ``stage1_pickplace_vision_only`` — keep ``obs["sensor_data"]`` (the
  RGB-D camera channels) plus the task-extra fields the policy needs
  to be functional at the Stage-1b episode budget (``tcp_pose``,
  ``goal_pos``). **Zero-mask** the agent-proprio sub-dicts under
  ``obs["agent"][uid]`` and the synthesised ``obs["extra"]["force_torque"]``
  channel.
- ``stage1_pickplace_vision_plus_force_torque_plus_proprio`` —
  pass-through (no filtering); the policy sees everything.

ManiSkill v3 obs layout reminder (``mani_skill/envs/sapien_env.py:546-630``):
``obs["agent"][uid]`` carries proprio (state, joint_pos, joint_vel);
``obs["extra"]`` carries task-level fields from
:meth:`_get_obs_extra`; ``obs["sensor_data"]`` carries camera channels
when the env's ``obs_mode`` is image-bearing.

Shape preservation: zero-masked channels become zero arrays of the same
shape and dtype (mirrors
:class:`chamber.envs.texture_filter.TextureFilterObsWrapper`'s pattern).
This keeps downstream policy networks oblivious to which modalities a
given uid actually receives at runtime — only the *values* change
across conditions, not the *shapes*.

Single-env contract. Stage-1b builds the env with ``num_envs=1`` per
``(seed, condition)`` cell (see
:func:`chamber.envs.stage1_pickplace.make_stage1_pickplace_env`); the
synthesiser's concat helpers assert this contract at runtime via
:func:`_to_1d_float32`. Multi-env builds are not supported at this
layer — they would silently mis-pack the leading batch dim into the
flat state vector.

Quaternion sign-flip caveat. ``cube_pose`` and ``tcp_pose`` carry
quaternion entries (``qw, qx, qy, qz``). SAPIEN does not pin a
quaternion sign convention at the obs layer — two physically identical
poses may emit ``q`` or ``-q`` depending on integrator floating-point
drift. This is a downstream science concern (manifests as policy
non-determinism in extreme cases), not a Surface-6 concern; flagged
here so a future investigator has the lead.

References:
- ADR-007 §Stage 1b (the AS / OM axis spec); ``spikes/preregistration/AS.yaml``
  and ``spikes/preregistration/OM.yaml`` (the condition_id strings
  these wrappers resolve verbatim).
- ADR-007 §Stage 1b Rev 12 (the Surface 6 remediation entry).
- ADR-009 §Decision (direct-observation-of-partner-pose paragraph).
- ADR-002 §Revision history 2026-05-21 (the Tier-2 acceptance-gap
  finding + extension of the trainer-obs-reader contract).
- :class:`chamber.envs.texture_filter.TextureFilterObsWrapper` (the
  parallel pattern for the Stage-0 smoke env's per-uid keep-set).
- :class:`chamber.envs.stage1_pickplace.Stage1PickPlaceEnv` (the
  inner env that supplies the synthesised FT channel and the
  ``condition_id`` attribute).
- ``spikes/results/stage1-failure-investigation/2026-05-20/CONSULTATION_BRIEF.md``
  (the §4a runbook decision rationale).
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

#: Channels under ``obs["extra"]`` the OM-vision-only condition keeps.
#:
#: The policy needs at least *some* task-relevant signal beyond the
#: RGB-D camera to be solvable inside the Stage-1b episode budget
#: (100 steps); ``tcp_pose`` + ``goal_pos`` is the minimum "where is
#: the gripper" + "where is the target" pair. Cube pose / cube-relative
#: deltas / FT are zero-masked. If the AS-hetero pilot shows the gap
#: vs vision_only is artefact-of-this-keep-set, tighten or loosen here
#: in a follow-up patch — the keep-set is design-tunable inside the
#: condition_id's spec since the prereg only names the *modality
#: families* (vision vs vision+ft+proprio), not the per-field keep-set
#: (ADR-007 §Discipline).
_OM_VISION_ONLY_EXTRA_KEEP: frozenset[str] = frozenset({"tcp_pose", "goal_pos"})

#: Condition_ids the wrapper actually mutates obs for. Conditions
#: outside this set pass through untouched.
_FILTERED_CONDITIONS: frozenset[str] = frozenset({"stage1_pickplace_vision_only"})

#: Task-extra fields the AS synthesiser injects into the ego's
#: widened ``state`` Box (ADR-007 §Stage 1b Rev 12). Order is
#: load-bearing (matches the concat order documented in the module
#: docstring); ``(field_name, expected_last_dim)``.
_AS_TASK_EXTRA_FIELDS: tuple[tuple[str, int], ...] = (
    ("cube_pose", 7),
    ("goal_pos", 3),
    ("tcp_pose", 7),
)


def _to_1d_float32(arr: Any, *, source: str) -> np.ndarray:  # type: ignore[type-arg]  # noqa: ANN401 - inputs are torch.Tensor or np.ndarray depending on env source
    """Coerce ``arr`` to a 1-D float32 ndarray (ADR-007 §Stage 1b Rev 12).

    Handles both torch tensors (real SAPIEN env, ``obs_mode="state_dict"``
    emits torch tensors when the device is CUDA/CPU torch) and numpy
    arrays (Tier-1 fakes). Pins the Stage-1b single-env contract: any
    leading batch dim must be 1 or absent. Multi-env builds
    (``num_envs > 1``) are not supported at the synthesiser layer
    because ``ravel()`` across a leading batch dim would silently
    mis-pack the flat state vector.

    Args:
        arr: Input array-like (torch.Tensor or np.ndarray).
        source: Human-readable name of the input (used in the
            TypeError message; e.g. ``"obs['agent'][ego_uid]['qpos']"``).

    Returns:
        Flat 1-D ``np.ndarray`` of dtype ``float32``.

    Raises:
        TypeError: If ``arr`` has a leading batch dim with size > 1.
    """
    try:
        import torch

        if isinstance(arr, torch.Tensor):
            arr = arr.detach().cpu().numpy()
    except ImportError:
        # Torch is a hard dep of the project (chamber.benchmarks.*),
        # but the local import keeps this module Tier-1 importable on
        # the rare host that ships without it.
        pass
    arr_np = np.asarray(arr)
    if arr_np.ndim > 1 and arr_np.shape[0] != 1:
        msg = (
            f"Stage1ASStateSynthesizer: {source} has shape {arr_np.shape}; "
            "expected a 1-D array or a single-env batch (shape[0]==1). "
            "Multi-env builds (num_envs>1) are not supported at the synthesiser "
            "layer — they would silently mis-pack the leading batch dim into "
            "the flat state vector. See ADR-007 §Stage 1b Rev 12."
        )
        raise TypeError(msg)
    return arr_np.ravel().astype(np.float32)


def _flat_state_concat(qpos: Any, qvel: Any) -> np.ndarray:  # type: ignore[type-arg]  # noqa: ANN401 - inputs are torch.Tensor or np.ndarray depending on env source
    """Concat ``qpos`` + ``qvel`` into a 1-D float32 ndarray (per-uid state).

    Used for non-ego (partner) ``state`` entries. ManiSkill v3
    ``obs_mode="state"`` flattens with the order ``concat(qpos, qvel)``;
    mirrored here so anyone running the underlying env with the flat
    ``obs_mode`` would see the same shape. The ego's widened ``state``
    uses :func:`_flat_widened_state_concat` instead.
    """
    return np.concatenate(
        [
            _to_1d_float32(qpos, source="qpos"),
            _to_1d_float32(qvel, source="qvel"),
        ]
    )


def _flat_widened_state_concat(
    ego_qpos: Any,  # noqa: ANN401 - torch.Tensor or np.ndarray
    ego_qvel: Any,  # noqa: ANN401
    partner_qpos: Any,  # noqa: ANN401
    partner_qvel: Any,  # noqa: ANN401
    cube_pose: Any,  # noqa: ANN401
    goal_pos: Any,  # noqa: ANN401
    tcp_pose: Any,  # noqa: ANN401
) -> np.ndarray:  # type: ignore[type-arg]
    """Concat the ego's widened ``state`` per ADR-007 §Stage 1b Rev 12.

    Order: ``[ego_qpos, ego_qvel, partner_qpos, partner_qvel,
    cube_pose, goal_pos, tcp_pose]``. Load-bearing — see the
    module docstring + the positional regression test in
    ``tests/unit/test_stage1_pickplace_tier1.py``.
    """
    return np.concatenate(
        [
            _to_1d_float32(ego_qpos, source="ego.qpos"),
            _to_1d_float32(ego_qvel, source="ego.qvel"),
            _to_1d_float32(partner_qpos, source="partner.qpos"),
            _to_1d_float32(partner_qvel, source="partner.qvel"),
            _to_1d_float32(cube_pose, source="extra.cube_pose"),
            _to_1d_float32(goal_pos, source="extra.goal_pos"),
            _to_1d_float32(tcp_pose, source="extra.tcp_pose"),
        ]
    )


class Stage1ASStateSynthesizer(gym.ObservationWrapper):  # type: ignore[type-arg]
    """Synthesise ``obs["agent"][uid]["state"]`` for AS conditions (ADR-007 §Stage 1b).

    ManiSkill v3.0.1 ``obs_mode="state_dict"`` (used by the Stage-1 AS
    conditions) emits per-agent ``Dict(qpos: Box, qvel: Box, ...)`` with
    no top-level ``"state"`` key. But
    :meth:`chamber.benchmarks.ego_ppo_trainer.EgoPPOTrainer.from_config`
    reads ``env.observation_space["agent"][ego_uid]["state"]`` as a 1-D
    Box and derives ``obs_dim = ego_state_space.shape[0]``. This
    wrapper synthesises that ``state`` key.

    **Ego state widening (P1.05.8; ADR-007 §Stage 1b Rev 12;
    ADR-009 §Decision).** Per the module docstring, the ego uid's
    synthesised ``state`` concatenates ``[ego_qpos, ego_qvel,
    partner_qpos, partner_qvel, cube_pose, goal_pos, tcp_pose]``.
    The non-ego (partner) entry's ``state`` stays at
    ``concat(qpos, qvel)`` of the partner alone (the partner does
    not see the task or the ego). Per-condition state dim is
    env-emit-dependent and pinned by the Tier-1 + Tier-2 tests; the
    trainer reads the shape off the env at construction time.

    Pass-through for OM conditions (``is_om_condition=True`` — those go
    through :class:`Stage1OMChannelFilter` instead) and for envs whose
    ``condition_id`` is missing or unrecognised (Tier-1 safety: keeps
    the wrapper a no-op for envs that don't satisfy the Stage-1b
    ``condition_id`` contract).

    Args:
        env: Inner env exposing a ``condition_id`` attribute. For AS
            conditions, must also expose ``obs["agent"][ego_uid]``
            and ``obs["agent"][partner_uid]`` Dicts with ``qpos`` /
            ``qvel`` Boxes, and ``obs["extra"]`` Dict with
            ``cube_pose`` (last-dim 7), ``goal_pos`` (last-dim 3),
            ``tcp_pose`` (last-dim 7).

    Raises:
        TypeError: If ``env.observation_space`` is not a
            :class:`gym.spaces.Dict`, or — for active AS conditions —
            if the required ``obs["extra"]`` task fields are missing
            or mistyped. The error message names the missing /
            wrong-typed key and cites ADR-007 §Stage 1b Rev 12, so a
            future operator who breaks the contract surfaces it at
            env construction rather than weeks later as a silent
            zero-fill regression of Surface 6.
    """

    def __init__(self, env: gym.Env[Any, Any]) -> None:
        """Detect AS-condition envs at construction (ADR-007 §Stage 1b Rev 12)."""
        super().__init__(env)
        if not isinstance(env.observation_space, gym.spaces.Dict):
            msg = (
                "Stage1ASStateSynthesizer requires a gym.spaces.Dict observation "
                f"space; got {type(env.observation_space).__name__}."
            )
            raise TypeError(msg)
        condition_id = getattr(env, "condition_id", None)
        if condition_id is None and hasattr(env, "get_wrapper_attr"):
            try:
                condition_id = env.get_wrapper_attr("condition_id")
            except AttributeError:
                condition_id = None
        # Local import to avoid the chamber.envs.stage1_pickplace ↔
        # chamber.envs.stage1_obs_filter circular import that arises now
        # that make_stage1_pickplace_env applies this wrapper.
        from chamber.envs.stage1_pickplace import resolve_condition

        config = None
        if isinstance(condition_id, str):
            try:
                config = resolve_condition(condition_id)
            except ValueError:
                config = None
        self._active: bool = config is not None and not config.is_om_condition
        # Cache ego / partner uids when active so the per-step
        # ``observation()`` callable does not have to re-resolve.
        self._ego_uid: str | None = None
        self._partner_uid: str | None = None
        if self._active and config is not None:
            self._ego_uid, self._partner_uid = config.agent_uids
            self.observation_space = self._build_observation_space(env.observation_space)

    def _build_observation_space(self, inner: gym.spaces.Dict) -> gym.spaces.Dict:
        """Inject the widened 1-D ``state`` Box per agent (ADR-007 §Stage 1b Rev 12).

        Ego uid's ``state`` Box has shape ``(ego_qpos + ego_qvel +
        partner_qpos + partner_qvel + 7 + 3 + 7,)``; non-ego entries
        keep the per-uid ``concat(qpos, qvel)`` shape.
        """
        # pragma: no cover - guarded by _active
        if self._ego_uid is None or self._partner_uid is None:
            return inner
        agent = inner.spaces.get("agent")
        if not isinstance(agent, gym.spaces.Dict):
            msg = (
                "Stage1ASStateSynthesizer: inner env's observation_space is missing "
                "a Dict 'agent' sub-space. Required by the AS synthesiser to "
                "compose per-uid 'state' Boxes. ADR-007 §Stage 1b Rev 12."
            )
            raise TypeError(msg)
        # Resolve ego / partner sub-spaces and validate qpos / qvel Box-ness.
        ego_qpos_dim, ego_qvel_dim = _resolve_qpos_qvel_dims(agent, uid=self._ego_uid, role="ego")
        partner_qpos_dim, partner_qvel_dim = _resolve_qpos_qvel_dims(
            agent, uid=self._partner_uid, role="partner"
        )
        # Validate ``obs["extra"]`` carries the three task fields with the
        # expected trailing dims.
        extra_total_dim = _validate_and_sum_task_extras(inner)
        ego_state_dim = (
            ego_qpos_dim + ego_qvel_dim + partner_qpos_dim + partner_qvel_dim + extra_total_dim
        )

        new_agent_spaces: dict[str, gym.spaces.Space[Any]] = {}
        for uid, sub in agent.spaces.items():
            if isinstance(sub, gym.spaces.Dict):
                qpos = sub.spaces.get("qpos")
                qvel = sub.spaces.get("qvel")
                if isinstance(qpos, gym.spaces.Box) and isinstance(qvel, gym.spaces.Box):
                    if uid == self._ego_uid:
                        state_dim = ego_state_dim
                    else:
                        state_dim = int(np.prod(qpos.shape)) + int(np.prod(qvel.shape))
                    state_box = gym.spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(state_dim,),
                        dtype=np.float32,
                    )
                    augmented = dict(sub.spaces)
                    augmented["state"] = state_box
                    new_agent_spaces[uid] = gym.spaces.Dict(augmented)
                    continue
            new_agent_spaces[uid] = sub
        new_spaces = dict(inner.spaces)
        new_spaces["agent"] = gym.spaces.Dict(new_agent_spaces)
        return gym.spaces.Dict(new_spaces)

    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:  # type: ignore[override]
        """Inject the widened ego ``state`` (ADR-007 §Stage 1b Rev 12)."""
        if not self._active:
            return observation
        agent = observation.get("agent")
        if not isinstance(agent, dict):
            return observation
        if self._ego_uid is None or self._partner_uid is None:  # pragma: no cover
            return observation
        extra = observation.get("extra")
        if not isinstance(extra, dict):
            msg = (
                "Stage1ASStateSynthesizer: runtime obs is missing the 'extra' "
                "sub-dict required by the widened ego state contract. "
                "Expected keys: cube_pose, goal_pos, tcp_pose. "
                "See ADR-007 §Stage 1b Rev 12."
            )
            raise TypeError(msg)
        partner_sub = agent.get(self._partner_uid)
        if (
            not isinstance(partner_sub, dict)
            or "qpos" not in partner_sub
            or "qvel" not in partner_sub
        ):
            msg = (
                f"Stage1ASStateSynthesizer: runtime obs missing partner uid "
                f"{self._partner_uid!r} with qpos+qvel keys. The widened ego "
                "state contract requires direct partner-pose observation "
                "(ADR-009 §Decision direct-observation-of-partner-pose). "
                "See ADR-007 §Stage 1b Rev 12."
            )
            raise TypeError(msg)
        task_arrays: dict[str, Any] = {}
        for field_name, _ in _AS_TASK_EXTRA_FIELDS:
            value = extra.get(field_name)
            if value is None:
                msg = (
                    f"Stage1ASStateSynthesizer: runtime obs missing task field "
                    f"'extra.{field_name}'. The widened ego state contract "
                    "requires cube / goal / TCP pose. See ADR-007 §Stage 1b Rev 12."
                )
                raise TypeError(msg)
            task_arrays[field_name] = value

        new_agent: dict[str, Any] = {}
        for uid, sub in agent.items():
            if isinstance(sub, dict) and "qpos" in sub and "qvel" in sub:
                augmented = dict(sub)
                if uid == self._ego_uid:
                    augmented["state"] = _flat_widened_state_concat(
                        sub["qpos"],
                        sub["qvel"],
                        partner_sub["qpos"],
                        partner_sub["qvel"],
                        task_arrays["cube_pose"],
                        task_arrays["goal_pos"],
                        task_arrays["tcp_pose"],
                    )
                else:
                    augmented["state"] = _flat_state_concat(sub["qpos"], sub["qvel"])
                new_agent[uid] = augmented
            else:
                new_agent[uid] = sub
        out = dict(observation)
        out["agent"] = new_agent
        return out

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401 - mirrors gym.Wrapper's inherited __getattr__ signature
        """Forward attribute access to the inner env (Tier-2 caller contract).

        Gymnasium 1.3 removed :class:`gym.Wrapper`'s implicit
        ``__getattr__``; existing P1.03 callers of
        :func:`make_stage1_pickplace_env` read SAPIEN-env attributes
        directly (``env.agent``, ``env._jacobian_provider``,
        ``env.condition_config``...). Forwarding here keeps that
        contract without forcing every caller to switch to
        :meth:`gym.Wrapper.get_wrapper_attr`.

        ``__getattr__`` only fires when normal attribute lookup misses;
        ``self.env`` is set by :meth:`gym.Wrapper.__init__` before any
        downstream access, so the delegation is safe. The ``env``-name
        guard is a defensive backstop against recursive lookup on the
        env attr itself (only reachable in pathological half-init
        paths).
        """
        if name == "env":
            raise AttributeError(name)
        return getattr(self.env, name)


def _resolve_qpos_qvel_dims(agent: gym.spaces.Dict, *, uid: str, role: str) -> tuple[int, int]:
    """Resolve and validate (qpos_dim, qvel_dim) for ``uid`` (ADR-007 §Stage 1b Rev 12).

    Loud-fails with a TypeError naming the missing / wrong-typed key,
    the uid, the role (ego/partner), and the ADR pointer so a future
    operator who breaks the contract surfaces it at env construction.
    """
    sub = agent.spaces.get(uid)
    if not isinstance(sub, gym.spaces.Dict):
        msg = (
            f"Stage1ASStateSynthesizer: missing {role} agent sub-Dict for uid={uid!r} "
            "(required for ego-state widening). See ADR-007 §Stage 1b Rev 12."
        )
        raise TypeError(msg)
    qpos = sub.spaces.get("qpos")
    qvel = sub.spaces.get("qvel")
    if not isinstance(qpos, gym.spaces.Box) or not isinstance(qvel, gym.spaces.Box):
        msg = (
            f"Stage1ASStateSynthesizer: {role} uid={uid!r} sub-Dict missing "
            "Box-typed 'qpos' / 'qvel' entries. See ADR-007 §Stage 1b Rev 12."
        )
        raise TypeError(msg)
    return int(np.prod(qpos.shape)), int(np.prod(qvel.shape))


def _validate_and_sum_task_extras(inner: gym.spaces.Dict) -> int:
    """Validate ``obs["extra"]`` carries cube_pose / goal_pos / tcp_pose and return their total dim.

    ADR-007 §Stage 1b Rev 12. Returns the sum of the *trailing* dims
    of the three Box fields (i.e. 7 + 3 + 7 = 17 when shapes are
    ``(1, 7) / (1, 3) / (1, 7)`` and also when shapes are ``(7,) /
    (3,) / (7,)``).
    """
    extra = inner.spaces.get("extra")
    if not isinstance(extra, gym.spaces.Dict):
        msg = (
            "Stage1ASStateSynthesizer: inner env's observation_space is missing "
            "a Dict 'extra' sub-space. The widened ego state contract requires "
            "cube_pose / goal_pos / tcp_pose under obs['extra']. "
            "See ADR-007 §Stage 1b Rev 12."
        )
        raise TypeError(msg)
    total = 0
    for field_name, expected_last_dim in _AS_TASK_EXTRA_FIELDS:
        sub = extra.spaces.get(field_name)
        if not isinstance(sub, gym.spaces.Box):
            msg = (
                f"Stage1ASStateSynthesizer: missing or wrong-typed task field "
                f"'extra.{field_name}' (expected gym.spaces.Box with trailing "
                f"dim {expected_last_dim}). See ADR-007 §Stage 1b Rev 12."
            )
            raise TypeError(msg)
        if not sub.shape or int(sub.shape[-1]) != expected_last_dim:
            msg = (
                f"Stage1ASStateSynthesizer: task field 'extra.{field_name}' "
                f"has shape {sub.shape} but expected trailing dim "
                f"{expected_last_dim}. See ADR-007 §Stage 1b Rev 12."
            )
            raise TypeError(msg)
        total += expected_last_dim
    return total


class Stage1OMChannelFilter(gym.ObservationWrapper):  # type: ignore[type-arg]
    """OM-axis per-condition channel filter (ADR-007 §Stage 1b).

    Reads the inner env's ``condition_id`` at construction time and
    captures the resulting filter policy. The wrapper does not re-read
    ``condition_id`` on every step — Stage-1b envs are built once per
    ``(seed, condition)`` cell (see ``stage1_as.py:253-275``) and the
    condition is fixed for the env's lifetime.

    Args:
        env: Inner env exposing a ``condition_id`` attribute. The
            :class:`chamber.envs.stage1_pickplace.Stage1PickPlaceEnv`
            class satisfies this; the wrapper falls back to a
            pass-through identity if ``condition_id`` is missing or
            not one of the two OM strings.

    Raises:
        TypeError: If the inner env's observation space is not a
            :class:`gym.spaces.Dict` (the wrapper relies on dict-shaped
            obs).
    """

    def __init__(self, env: gym.Env[Any, Any]) -> None:
        """Store the per-condition filter policy (ADR-007 §Stage 1b)."""
        super().__init__(env)
        if not isinstance(env.observation_space, gym.spaces.Dict):
            msg = (
                "Stage1OMChannelFilter requires a gym.spaces.Dict observation "
                f"space; got {type(env.observation_space).__name__}."
            )
            raise TypeError(msg)
        # Read condition_id once at construction; the env-per-cell
        # cadence makes this safe and saves a hot-path attr lookup.
        condition_id = getattr(env, "condition_id", None)
        if condition_id is None and hasattr(env, "get_wrapper_attr"):
            try:
                condition_id = env.get_wrapper_attr("condition_id")
            except AttributeError:
                condition_id = None
        self._condition_id: str | None = condition_id
        self._filter_active: bool = condition_id in _FILTERED_CONDITIONS

    @property
    def condition_id(self) -> str | None:
        """ADR-007 §Stage 1b OM axis: the condition_id resolved at construction."""
        return self._condition_id

    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:  # type: ignore[override]
        """Apply the per-condition channel mask (ADR-007 §Stage 1b).

        OM-vision-only zero-masks the per-uid agent-proprio sub-dicts
        and the ``obs["extra"]["force_torque"]`` channel; keeps
        ``obs["extra"]["tcp_pose"]`` + ``obs["extra"]["goal_pos"]``
        (the minimal task-relevant fields) plus ``obs["sensor_data"]``
        (RGB-D camera) untouched. Other conditions pass through.
        """
        if not self._filter_active:
            return observation
        out = dict(observation)
        # Zero-mask per-uid agent proprio (the "no proprioception" half
        # of the vision-only condition_id semantics).
        agent = observation.get("agent")
        if isinstance(agent, dict):
            zeroed_agent: dict[str, Any] = {}
            for uid, sub in agent.items():
                if isinstance(sub, dict):
                    zeroed_agent[uid] = {
                        ch: np.zeros_like(val) if hasattr(val, "shape") else val
                        for ch, val in sub.items()
                    }
                else:
                    zeroed_agent[uid] = sub
            out["agent"] = zeroed_agent
        # Zero-mask task extras NOT in the vision-only keep-set (the
        # "no force-torque" half of the vision-only semantics).
        extra = observation.get("extra")
        if isinstance(extra, dict):
            filtered_extra: dict[str, Any] = {}
            for ch, val in extra.items():
                if ch in _OM_VISION_ONLY_EXTRA_KEEP:
                    filtered_extra[ch] = val
                elif hasattr(val, "shape"):
                    filtered_extra[ch] = np.zeros_like(val)
                else:
                    filtered_extra[ch] = val
            out["extra"] = filtered_extra
        return out

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401 - mirrors Stage1ASStateSynthesizer's forwarding contract
        """Forward attribute access to the inner env (Tier-2 caller contract).

        Mirrors :meth:`Stage1ASStateSynthesizer.__getattr__`: Gymnasium 1.3
        removed :class:`gym.Wrapper`'s implicit forwarding, so callers of
        :func:`chamber.envs.stage1_pickplace.make_stage1_pickplace_env`
        — which now applies this wrapper on top of the AS synthesizer —
        would otherwise lose access to SAPIEN-env attributes such as
        ``env.agent``, ``env.obs_mode``, ``env.condition_config``.
        """
        if name == "env":
            raise AttributeError(name)
        return getattr(self.env, name)


__all__ = ["Stage1ASStateSynthesizer", "Stage1OMChannelFilter"]
