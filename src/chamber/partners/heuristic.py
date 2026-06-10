# SPDX-License-Identifier: Apache-2.0
"""Scripted heuristic partner — Stage-3 draft-zoo entry #1 (ADR-009 §Consequences).

The simplest possible partner: a deterministic policy that greedily reduces
the planar distance between the partner's current xy and a configured target
xy. Phase-0 ships exactly this so the ADR-007 Stage-3 PF spike has a known-
low-skill partner against which to measure the trained-with vs. frozen-novel
gap (plan/04 §1; ADR-007 rev 3 Stage 3 entry).

The policy reads:

* ``obs["comm"]["pose"][uid]["xyz"]`` — partner's own pose, when the
  ADR-003 fixed-format channel is wired up;
* ``obs["agent"][uid]["state"][:2]`` — fallback proprio xy when no comm
  packet is present (e.g. the unit-test :class:`FakeMultiAgentEnv`).

The action is a unit-clipped xy velocity command toward the target. The
remaining components of the action vector (configured via
``spec.extra["action_dim"]``) are filled with zeros — adequate for the
Stage-0 smoke env where the heuristic only drives the planar base.

Determinism: the policy is fully deterministic given ``spec`` + ``obs``;
:meth:`ScriptedHeuristicPartner.reset` accepts a seed only to honour the
:class:`~chamber.partners.api.FrozenPartner` Protocol (P6 reproducibility).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, cast

import numpy as np

from chamber.partners.interface import PartnerBase
from chamber.partners.registry import register_partner

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from chamber.partners.api import PartnerSpec

#: Maximum absolute value of any action component (matches FakeMultiAgentEnv's Box(-1, 1)).
#:
# TODO(M4b): read from env.action_space[uid].high once the heuristic is wired
# into a real ManiSkill env. The hard-coded value is correct for Stage-0 smoke
# + the unit-test fake env; mismatched action-space ranges in Phase 1 would
# silently distort Stage-3 PF gap measurements.
_ACTION_CLIP: float = 1.0

#: Number of comma-separated floats expected in ``spec.extra["target_xy"]``.
_XY_DIMS: int = 2

#: Minimum action-vector length needed to carry the planar xy step.
_MIN_ACTION_DIM: int = 2


def _parse_xy(raw: str) -> tuple[float, float]:
    """Parse an ``"x,y"`` string into a (float, float) tuple (plan/04 §3.4).

    Raises:
        ValueError: If ``raw`` is not exactly two comma-separated floats.
    """
    parts = raw.split(",")
    if len(parts) != _XY_DIMS:
        raise ValueError(f"target_xy must be 'x,y' with two comma-separated floats; got {raw!r}")
    try:
        return float(parts[0]), float(parts[1])
    except ValueError as exc:
        raise ValueError(f"target_xy components must be floats; got {raw!r}") from exc


@register_partner("scripted_heuristic")
class ScriptedHeuristicPartner(PartnerBase):
    """Greedy planar-reach heuristic partner (ADR-009 §Consequences; plan/04 §3.4).

    Reads the partner's own xy from ``obs["comm"]["pose"][uid]["xyz"]`` (ADR-003
    §Decision) when available, else from ``obs["agent"][uid]["state"][:2]``.
    Outputs a unit-clipped xy velocity toward the configured target xy.

    The partner's own uid is read from ``spec.extra["uid"]`` and the target
    from ``spec.extra["target_xy"]`` (default ``"0.0,0.0"``). Action dim is
    set via ``spec.extra["action_dim"]`` (default ``"2"``); components beyond
    the planar xy are zeroed.
    """

    def __init__(self, spec: PartnerSpec) -> None:
        """Bind the spec and parse the heuristic's tunables (ADR-009 §Decision).

        Args:
            spec: Partner identity. ``spec.extra`` controls the policy:
                ``uid`` selects the env-side agent, ``target_xy`` the goal,
                ``action_dim`` the output vector length.

        Raises:
            ValueError: If ``spec.extra["target_xy"]`` is malformed or
                ``action_dim`` is not a positive integer ≥ 2.
        """
        super().__init__(spec)
        self._uid: str = spec.extra.get("uid", "")
        self._target_xy: tuple[float, float] = _parse_xy(spec.extra.get("target_xy", "0.0,0.0"))
        action_dim_raw = spec.extra.get("action_dim", "2")
        try:
            action_dim = int(action_dim_raw)
        except ValueError as exc:
            raise ValueError(
                f"action_dim must be a positive int ≥ 2; got {action_dim_raw!r}"
            ) from exc
        if action_dim < _MIN_ACTION_DIM:
            raise ValueError(f"action_dim must be a positive int ≥ 2; got {action_dim}")
        self._action_dim: int = action_dim

    def reset(self, *, seed: int | None = None) -> None:
        """No-op reset (ADR-009 §Decision; plan/04 §2 stateless across episodes).

        Phase-0 heuristic carries no episode state. The seed is ignored
        deliberately; the policy is fully deterministic from ``obs``.

        Args:
            seed: Accepted for Protocol conformance only.
        """
        del seed

    def act(
        self,
        obs: Mapping[str, object],
        *,
        deterministic: bool = True,
    ) -> NDArray[np.floating]:
        """Greedy planar-reach action toward ``spec.extra["target_xy"]`` (ADR-009 §Consequences).

        Args:
            obs: Gymnasium observation mapping keyed by ``"agent"`` and
                optionally ``"comm"``. ADR-003 §Decision: when ``"comm"`` is
                present, pose is read from ``obs["comm"]["pose"][uid]``.
            deterministic: Ignored; the heuristic is deterministic regardless.

        Returns:
            Float32 action vector of length ``spec.extra["action_dim"]``;
            components 0/1 hold the unit-clipped xy velocity toward target,
            remaining components are zero.
        """
        del deterministic
        batched_xy = self._read_agent_xy_batched(obs)
        if batched_xy is not None:
            # P1.05.10 (ADR-007 §Stage 1b regime-alignment): vectorised
            # cells emit per-uid state as (num_envs, dim); the heuristic
            # returns (num_envs, action_dim) so the batched env step
            # receives one planar-reach command per env. The policy per
            # env is identical to the single-env path.
            tx, ty = self._target_xy
            n = batched_xy.shape[0]
            action_b = np.zeros((n, self._action_dim), dtype=np.float32)
            action_b[:, 0] = np.clip(tx - batched_xy[:, 0], -_ACTION_CLIP, _ACTION_CLIP)
            action_b[:, 1] = np.clip(ty - batched_xy[:, 1], -_ACTION_CLIP, _ACTION_CLIP)
            return action_b
        x, y = self._read_agent_xy(obs)
        tx, ty = self._target_xy
        dx = float(np.clip(tx - x, -_ACTION_CLIP, _ACTION_CLIP))
        dy = float(np.clip(ty - y, -_ACTION_CLIP, _ACTION_CLIP))
        action = np.zeros(self._action_dim, dtype=np.float32)
        action[0] = dx
        action[1] = dy
        return action

    def _read_agent_xy_batched(self, obs: Mapping[str, object]) -> NDArray[np.floating] | None:
        """Return the partner's planar xy as ``(num_envs, 2)``, or ``None`` (ADR-007 §Stage 1b).

        P1.05.10: detects the vectorised-cell obs layout — a 2-D
        ``obs["agent"][uid]["state"]`` of shape ``(num_envs, dim)``
        with ``num_envs > 1`` — and returns its first two columns.
        Returns ``None`` for the single-env layout (1-D state or
        ``shape[0] == 1`` is still routed through the historical
        scalar path so pre-P1.05.10 behaviour is byte-identical).
        Torch tensors are accepted (ManiSkill GPU obs) and moved to
        numpy.
        """
        agent = obs.get("agent")
        if not isinstance(agent, Mapping) or self._uid not in agent:
            return None
        entry = agent[self._uid]
        if not isinstance(entry, Mapping):
            return None
        state = entry.get("state")
        if state is None:
            return None
        if hasattr(state, "detach"):
            state = state.detach().cpu().numpy()  # type: ignore[union-attr]
        if not isinstance(state, np.ndarray):
            return None
        if state.ndim == 2 and state.shape[0] > 1 and state.shape[1] >= _XY_DIMS:  # noqa: PLR2004 - rank-2 is the (batch, dim) layout
            return state[:, :_XY_DIMS].astype(np.float64)
        return None

    def _read_agent_xy(self, obs: Mapping[str, object]) -> tuple[float, float]:
        """Extract the partner's planar xy from ``obs`` (ADR-003 §Decision).

        Reads ``obs["comm"]["pose"][uid]["xyz"]`` when present (the M2
        fixed-format channel surface), falling back to
        ``obs["agent"][uid]["state"][:2]`` for envs without comm wired up
        (e.g. the unit-test :class:`tests.fakes.FakeMultiAgentEnv`).

        Args:
            obs: Gymnasium observation mapping.

        Returns:
            ``(x, y)`` tuple of Python floats.
        """
        comm = obs.get("comm")
        if isinstance(comm, Mapping):
            pose_table = comm.get("pose")
            if isinstance(pose_table, Mapping) and self._uid in pose_table:
                pose = pose_table[self._uid]
                if isinstance(pose, Mapping):
                    xyz = pose.get("xyz")
                    if isinstance(xyz, (tuple, list)) and len(xyz) >= _XY_DIMS:
                        return float(xyz[0]), float(xyz[1])
        agent = obs.get("agent")
        if isinstance(agent, Mapping) and self._uid in agent:
            entry = agent[self._uid]
            if isinstance(entry, Mapping):
                state = entry.get("state")
                if isinstance(state, np.ndarray) and state.shape[0] >= _XY_DIMS:
                    arr = cast("NDArray[np.floating]", state)
                    return float(arr[0]), float(arr[1])
        return 0.0, 0.0


__all__ = ["ScriptedHeuristicPartner"]
