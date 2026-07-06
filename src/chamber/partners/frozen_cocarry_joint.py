# SPDX-License-Identifier: Apache-2.0
# pyright: reportPrivateImportUsage=false
#
# Same rationale as :mod:`chamber.partners.frozen_harl`: torch's stubs
# do not export ``no_grad`` / ``from_numpy`` via ``__all__`` even though
# they are documented public API.
"""Frozen B-JOINT partner-side policy for co-carry (ADR-011 §Decision as amended; ADR-018).

B-JOINT (ADR-011 §Decision as amended 2026-07-05) is the jointly-trained
MAPPO pair, evaluated **as the pair it trained as** — the non-AHT upper
anchor. :mod:`chamber.benchmarks.joint_mappo_trainer` saves one pair
checkpoint ``{"actor_ego": …, "actor_partner": …, "critic": …, …}``;
this wrapper puts the *partner-side* actor on the partner seat at
evaluation time (and, via ``spec.extra["actor_key"]``, either actor on
either seat — the CB-06 zoo-enrichment path admits partner-side
checkpoints to the co-carry partner set ``version 2`` with provenance
"trained jointly with <ego hash>").

Observation interface: the joint trainer feeds the partner actor the
**symmetric full state** :func:`joint_partner_full_state` — the mirror
of the ego's 46-D synthesised state
(:mod:`chamber.envs.cocarry_obs`)::

    [own_qpos, own_qvel, other_qpos, other_qvel, bar_pose, goal_pos]

assembled from the RAW obs leaves (``obs["agent"][uid]["qpos"/"qvel"]``
+ ``obs["extra"]["bar_pose"/"goal_pos"]``), which every co-carry env
preserves byte-for-byte (the synthesizer's pass-through contract). One
shared helper is used by the trainer at training time and by this
wrapper at evaluation time so the two interfaces cannot drift.

Frozen-partner contract (ADR-018/I3): subclasses
:class:`~chamber.partners.interface.PartnerBase` (the no-joint-training
``_FORBIDDEN_ATTRS`` shield), loads via
:func:`concerto.training.checkpoints.load_checkpoint` (SHA-256 sidecar
verified), freezes every parameter, and runs inference under
:func:`torch.no_grad` with ``deterministic=True`` pinned (the
:mod:`chamber.partners.frozen_harl` determinism rationale).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import gymnasium as gym
import numpy as np
import torch

from chamber.partners.frozen_harl import (
    _HARL_INFERENCE_ARGS,
    _infer_shape,
    _resolve_artifacts_root,
)
from chamber.partners.interface import PartnerBase
from chamber.partners.registry import register_partner
from concerto.training.checkpoints import load_checkpoint

if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy.typing import NDArray

    from chamber.partners.api import PartnerSpec

from harl.models.policy_models.stochastic_policy import StochasticPolicy

#: Registry name of the frozen B-JOINT co-carry policy (ADR-011 as amended).
FROZEN_COCARRY_JOINT_CLASS: str = "frozen_cocarry_joint"

#: Default pair-checkpoint sub-key this wrapper loads (the partner-side
#: actor; the ego-side is ``"actor_ego"`` — see
#: :meth:`chamber.benchmarks.joint_mappo_trainer.JointMAPPOTrainer.state_dict`).
DEFAULT_ACTOR_KEY: str = "actor_partner"


def _read_leaf(obs: Mapping[str, object], *path: str) -> NDArray[np.float32]:
    """Read a nested obs leaf as a flat float32 vector, loud-failing on absence."""
    node: object = obs
    for key in path:
        if not isinstance(node, dict) or key not in node:
            joined = "][".join(repr(p) for p in path)
            msg = (
                f"joint_partner_full_state: obs[{joined}] missing — the co-carry "
                "raw-leaf pass-through contract is broken (ADR-026 §Decision 1; "
                "ADR-011 §Decision as amended)."
            )
            raise ValueError(msg)
        node = node[key]
    if hasattr(node, "detach"):
        node = node.detach().cpu().numpy()  # type: ignore[attr-defined]
    return np.asarray(node, dtype=np.float32).reshape(-1)


def joint_partner_full_state(
    obs: Mapping[str, object],
    *,
    own_uid: str,
    other_uid: str,
) -> NDArray[np.float32]:
    """The symmetric full-state view for a joint-trained seat (ADR-011 §Decision as amended).

    Mirror of the ego's synthesised state
    (:mod:`chamber.envs.cocarry_obs` concat order) with own/other seats
    swapped: ``[own_qpos, own_qvel, other_qpos, other_qvel, bar_pose,
    goal_pos]`` — 46-D for the matched Panda pair. Assembled from the
    raw obs leaves so the same function serves training
    (:mod:`chamber.benchmarks.joint_mappo_trainer`) and frozen
    evaluation (:class:`FrozenCoCarryJointPartner`) — one definition,
    no train/eval drift.

    Args:
        obs: Gymnasium observation mapping with the co-carry raw
            leaves preserved.
        own_uid: The seat this vector is *for* (its qpos/qvel lead).
        other_uid: The opposite seat.

    Returns:
        The flat float32 state vector.

    Raises:
        ValueError: When a required raw leaf is missing (the message
            names the leaf).
    """
    return np.concatenate(
        [
            _read_leaf(obs, "agent", own_uid, "qpos"),
            _read_leaf(obs, "agent", own_uid, "qvel"),
            _read_leaf(obs, "agent", other_uid, "qpos"),
            _read_leaf(obs, "agent", other_uid, "qvel"),
            _read_leaf(obs, "extra", "bar_pose"),
            _read_leaf(obs, "extra", "goal_pos"),
        ]
    )


@register_partner(FROZEN_COCARRY_JOINT_CLASS)
class FrozenCoCarryJointPartner(PartnerBase):
    """Frozen joint-MAPPO co-carry seat (ADR-011 §Decision as amended; ADR-018/I3).

    Spec contract:

    - ``spec.class_name == "frozen_cocarry_joint"``.
    - ``spec.weights_uri`` — the ``local://artifacts/<pair>.pt`` pair
      checkpoint URI (SHA-256 sidecar verified at load).
    - ``spec.extra["uid"]`` — the env-side seat this policy acts on.
    - ``spec.extra["other_uid"]`` — the opposite seat (needed to
      assemble the symmetric full state from raw leaves).
    - ``spec.extra["actor_key"]`` — optional pair-checkpoint sub-key;
      defaults to :data:`DEFAULT_ACTOR_KEY` (the partner-side actor).

    Statefulness: stateless across episodes; :meth:`reset` only ensures
    the checkpoint is loaded (mirrors
    :class:`~chamber.partners.frozen_harl.FrozenHARLPartner`).
    """

    def __init__(self, spec: PartnerSpec) -> None:
        """Bind the spec; defer checkpoint I/O (ADR-009 §Decision; ADR-018/I3).

        Raises:
            ValueError: If ``spec.weights_uri`` is ``None`` or
                ``spec.extra`` lacks ``uid`` / ``other_uid``.
        """
        super().__init__(spec)
        if spec.weights_uri is None:
            msg = (
                "FrozenCoCarryJointPartner requires spec.weights_uri (the pair "
                "checkpoint; ADR-011 §Decision as amended); got None."
            )
            raise ValueError(msg)
        uid = spec.extra.get("uid", "")
        other_uid = spec.extra.get("other_uid", "")
        if not uid or not other_uid:
            msg = (
                "FrozenCoCarryJointPartner requires spec.extra['uid'] and "
                "spec.extra['other_uid'] (the two co-carry seats; ADR-011 "
                "§Decision as amended)."
            )
            raise ValueError(msg)
        self._uid: str = uid
        self._other_uid: str = other_uid
        self._actor_key: str = spec.extra.get("actor_key", DEFAULT_ACTOR_KEY)
        self._weights_uri: str = spec.weights_uri
        self._actor: StochasticPolicy | None = None
        self._obs_dim: int = 0
        self._hidden_dim: int = 0

    def reset(self, *, seed: int | None = None) -> None:
        """Ensure the checkpoint is loaded; no per-episode state (ADR-009 §Decision)."""
        del seed
        self._ensure_loaded()

    def act(
        self,
        obs: Mapping[str, object],
        *,
        deterministic: bool = True,
    ) -> NDArray[np.floating]:
        """Deterministic actor forward pass on the symmetric full state (ADR-011; ADR-018).

        Args:
            obs: Gymnasium observation mapping; raw co-carry leaves are
                read via :func:`joint_partner_full_state`.
            deterministic: Accepted for Protocol conformance but pinned
                to ``True`` (the frozen-partner global-RNG decoupling
                rationale, :mod:`chamber.partners.frozen_harl`).

        Returns:
            ``np.float32`` action vector for this seat.

        Raises:
            ValueError: When the assembled state width disagrees with
                the loaded actor's input dimension.
        """
        del deterministic
        actor = self._ensure_loaded()
        state = joint_partner_full_state(obs, own_uid=self._uid, other_uid=self._other_uid)
        if state.shape != (self._obs_dim,):
            msg = (
                f"FrozenCoCarryJointPartner: assembled full state has shape "
                f"{tuple(state.shape)}, expected ({self._obs_dim},) matching the "
                f"loaded {self._actor_key!r} actor's input dim."
            )
            raise ValueError(msg)
        obs_t = state.reshape(1, -1)
        rnn_states = np.zeros((1, 1, self._hidden_dim), dtype=np.float32)
        masks = np.ones((1, 1), dtype=np.float32)
        with torch.no_grad():
            action_t, _, _ = actor(obs_t, rnn_states, masks, None, True)
        return action_t.detach().cpu().numpy().squeeze(0).astype(np.float32)

    def _ensure_loaded(self) -> StochasticPolicy:
        """Load + freeze the named actor sub-dict on first use; idempotent (ADR-018/I3).

        Raises:
            CheckpointError: Propagated from
                :func:`concerto.training.checkpoints.load_checkpoint`.
            ValueError: When the pair checkpoint lacks the named actor
                sub-key or its layout is not the HARL StochasticPolicy.
        """
        if self._actor is not None:
            return self._actor
        loaded, _ = load_checkpoint(uri=self._weights_uri, artifacts_root=_resolve_artifacts_root())
        inner = loaded.get(self._actor_key)
        if not isinstance(inner, dict):
            msg = (
                f"FrozenCoCarryJointPartner: pair checkpoint has no dict under "
                f"{self._actor_key!r}. Got top-level keys: {sorted(loaded.keys())}. "
                "Expected the joint_mappo_trainer pair layout (ADR-011 as amended)."
            )
            raise ValueError(msg)
        obs_dim, hidden_dim, _action_dim = _infer_shape(inner)
        args = dict(_HARL_INFERENCE_ARGS)
        args["hidden_sizes"] = [hidden_dim, hidden_dim]
        obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        act_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(_action_dim,), dtype=np.float32)
        actor = StochasticPolicy(args, obs_space, act_space, torch.device("cpu"))
        try:
            actor.load_state_dict(dict(inner))
        except RuntimeError as exc:
            msg = (
                f"FrozenCoCarryJointPartner: load_state_dict failed for "
                f"{self._weights_uri!r}[{self._actor_key!r}]: {exc}"
            )
            raise ValueError(msg) from exc
        for param in actor.parameters():
            param.requires_grad = False
        actor.eval()
        self._actor = actor
        self._obs_dim = obs_dim
        self._hidden_dim = hidden_dim
        return actor


__all__ = [
    "DEFAULT_ACTOR_KEY",
    "FROZEN_COCARRY_JOINT_CLASS",
    "FrozenCoCarryJointPartner",
    "joint_partner_full_state",
]
