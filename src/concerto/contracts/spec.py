# SPDX-License-Identifier: Apache-2.0
"""Task contracts: the observation/action agreement shared by sim and real.

A :class:`TaskContract` states, for one task version, what a flat observation
and a flat action *mean*: field order, dimensions, units, frames, control rate,
reset joint configuration, and how the simulator applies a delta-pose action.
The sim env and a real-robot env expose the same contract, so a parity test can
assert they agree. Contracts are frozen data; any change bumps ``version``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

N_ARM_JOINTS: int = 7
_N_XYZ: int = 3


@dataclass(frozen=True)
class ObsField:
    """One contiguous slice of the flat observation vector."""

    name: str
    dim: int
    unit: str
    frame: str

    def __post_init__(self) -> None:
        """Reject non-positive dimensions."""
        if self.dim <= 0:
            raise ValueError(f"obs field {self.name!r}: dim must be positive, got {self.dim}")


@dataclass(frozen=True)
class ActionField:
    """One contiguous slice of the flat action vector.

    ``[low, high]`` bounds the value the policy emits; it maps affinely onto
    ``[physical_low, physical_high]`` in ``unit`` (ManiSkill's
    ``clip_and_scale_action``). A decreasing physical range encodes a sign flip.
    """

    name: str
    dim: int
    low: float
    high: float
    physical_low: float
    physical_high: float
    unit: str
    frame: str

    def __post_init__(self) -> None:
        """Reject non-positive dimensions and empty action ranges."""
        if self.dim <= 0 or not self.low < self.high:
            raise ValueError(f"action field {self.name!r}: bad dim/bounds")

    def to_physical(self, value: float) -> float:
        """Clip ``value`` to ``[low, high]`` and map it to physical units."""
        t = (min(max(value, self.low), self.high) - self.low) / (self.high - self.low)
        return self.physical_low + t * (self.physical_high - self.physical_low)


@dataclass(frozen=True)
class TaskContract:
    """The full observation/action agreement for one task version.

    ``env_id``/``robot_uid``/``obs_mode``/``control_mode`` are the ManiSkill
    ``gym.make`` arguments of the reference sim env. ``base_position_in_world``
    is where the sim places the robot base (metres, identity orientation);
    fields with ``frame="world"`` are offset from the base frame by it.
    """

    task_id: str
    version: int
    env_id: str
    robot_uid: str
    obs_mode: str
    control_mode: str
    control_hz: int
    max_episode_steps: int
    obs_spec: tuple[ObsField, ...]
    action_spec: tuple[ActionField, ...]
    quaternion_order: Literal["wxyz"]
    reset_joint_config: tuple[float, ...]
    base_position_in_world: tuple[float, float, float]
    delta_pose_frame: str
    notes: str = ""

    def __post_init__(self) -> None:
        """Validate structure; the ``ValueError`` names every failing field."""
        problems: list[str] = []
        if self.quaternion_order != "wxyz":
            problems.append(f"quaternion_order must be 'wxyz', got {self.quaternion_order!r}")
        if len(self.reset_joint_config) != N_ARM_JOINTS:
            problems.append(f"reset_joint_config needs {N_ARM_JOINTS} values")
        if len(self.base_position_in_world) != _N_XYZ:
            problems.append("base_position_in_world needs 3 values")
        if min(self.control_hz, self.max_episode_steps) <= 0 or self.version < 0:
            problems.append("control_hz/max_episode_steps must be positive, version >= 0")
        for label, spec in (("obs_spec", self.obs_spec), ("action_spec", self.action_spec)):
            names = [f.name for f in spec]
            if not names:
                problems.append(f"{label} is empty")
            elif len(set(names)) != len(names):
                problems.append(f"{label} has duplicate field names")
        if problems:
            raise ValueError(f"invalid contract {self.task_id!r}: " + "; ".join(problems))

    @property
    def obs_dim(self) -> int:
        """Total flat observation size."""
        return sum(f.dim for f in self.obs_spec)

    @property
    def action_dim(self) -> int:
        """Total flat action size."""
        return sum(f.dim for f in self.action_spec)

    def obs_slices(self) -> dict[str, slice]:
        """Name -> slice into the flat observation, in spec order."""
        return _slices(self.obs_spec)

    def action_slices(self) -> dict[str, slice]:
        """Name -> slice into the flat action, in spec order."""
        return _slices(self.action_spec)

    def action_field(self, name: str) -> ActionField:
        """Look up one action field by name."""
        for f in self.action_spec:
            if f.name == name:
                return f
        raise KeyError(f"contract {self.task_id!r} has no action field {name!r}")


def _slices(fields: tuple[ObsField, ...] | tuple[ActionField, ...]) -> dict[str, slice]:
    out: dict[str, slice] = {}
    start = 0
    for f in fields:
        out[f.name] = slice(start, start + f.dim)
        start += f.dim
    return out


_REGISTRY: dict[str, TaskContract] = {}


def register(contract: TaskContract) -> TaskContract:
    """Add a contract; registering the same ``task_id`` twice is an error."""
    if contract.task_id in _REGISTRY:
        raise ValueError(f"task contract {contract.task_id!r} is already registered")
    _REGISTRY[contract.task_id] = contract
    return contract


def get(task_id: str) -> TaskContract:
    """Return the contract for ``task_id``; the ``KeyError`` lists the known ids."""
    if task_id not in _REGISTRY:
        raise KeyError(f"unknown task contract {task_id!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[task_id]


def list_contracts() -> list[str]:
    """Sorted ids of every registered contract."""
    return sorted(_REGISTRY)
