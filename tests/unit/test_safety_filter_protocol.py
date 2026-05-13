# SPDX-License-Identifier: Apache-2.0
"""Tests for the typed safety-filter Protocols (ADR-004 §Public API; spike_004A).

The pre-refactor :data:`concerto.safety.api.SafetyFilter` Protocol was a
single dict-in / dict-out contract that the deployment-default
:attr:`SafetyMode.EGO_ONLY` implementation did not satisfy (reviewer
P0-2). The Protocol now splits into :class:`EgoOnlySafetyFilter` and
:class:`JointSafetyFilter`; this module verifies that:

- The typed classmethod constructors on
  :class:`concerto.safety.cbf_qp.ExpCBFQP` return instances that
  structurally satisfy the matching Protocol under
  ``@runtime_checkable``.
- The deprecated :data:`SafetyFilter` union alias still resolves at
  import time (so external code importing it does not break) and
  emits a single :class:`DeprecationWarning` on first access.
"""

from __future__ import annotations

import warnings

import numpy as np

from concerto.safety.api import (
    AgentControlModel,
    DoubleIntegratorControlModel,
    EgoOnlySafetyFilter,
    JointSafetyFilter,
)
from concerto.safety.cbf_qp import ExpCBFQP


def _models() -> dict[str, AgentControlModel]:
    out: dict[str, AgentControlModel] = {}
    out["ego"] = DoubleIntegratorControlModel(uid="ego", action_dim=2)
    out["partner"] = DoubleIntegratorControlModel(uid="partner", action_dim=2)
    return out


def test_ego_only_constructor_returns_ego_only_protocol_instance() -> None:
    """spike_004A §Three-mode taxonomy: ``ExpCBFQP.ego_only`` ↦ ``EgoOnlySafetyFilter``."""
    cbf = ExpCBFQP.ego_only(control_models=_models())
    assert isinstance(cbf, EgoOnlySafetyFilter)


def test_centralized_constructor_returns_joint_protocol_instance() -> None:
    """spike_004A §Three-mode taxonomy: ``ExpCBFQP.centralized`` ↦ ``JointSafetyFilter``."""
    cbf = ExpCBFQP.centralized(control_models=_models())
    assert isinstance(cbf, JointSafetyFilter)


def test_shared_control_constructor_returns_joint_protocol_instance() -> None:
    """spike_004A §Three-mode taxonomy: ``ExpCBFQP.shared_control`` ↦ ``JointSafetyFilter``."""
    cbf = ExpCBFQP.shared_control(control_models=_models())
    assert isinstance(cbf, JointSafetyFilter)


def test_deprecated_safety_filter_alias_emits_deprecation_warning() -> None:
    """ADR-004 §Public API: the legacy alias resolves but warns on first use."""
    from concerto.safety import api as safety_api

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        alias = safety_api.SafetyFilter  # noqa: F841 — the lookup itself triggers the warning.
    deprecation = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(deprecation) == 1
    assert "SafetyFilter" in str(deprecation[0].message)
    assert "0.3.0" in str(deprecation[0].message)


def test_deprecated_safety_filter_alias_round_trips_via_package_reexport() -> None:
    """``from concerto.safety import SafetyFilter`` resolves via the package __getattr__."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        from concerto.safety import SafetyFilter

        del SafetyFilter
    deprecation = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert deprecation, "deprecated SafetyFilter alias did not surface a DeprecationWarning"


def test_deprecated_safety_filter_alias_resolves_to_union_of_typed_protocols() -> None:
    """The legacy alias resolves to ``Union[EgoOnlySafetyFilter, JointSafetyFilter]``."""
    from concerto.safety import api as safety_api

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        alias = safety_api.SafetyFilter
    # The alias's ``__args__`` carries the union members; both new
    # Protocols must be present (order-independent).
    args = set(getattr(alias, "__args__", ()))
    assert EgoOnlySafetyFilter in args
    assert JointSafetyFilter in args


def test_runtime_checkable_rejects_class_missing_filter() -> None:
    """Sanity check: ``runtime_checkable`` rejects objects that don't implement ``filter``."""

    class _Bogus:
        def reset(self, *, seed: int | None = None) -> None:
            del seed

    assert not isinstance(_Bogus(), EgoOnlySafetyFilter)
    assert not isinstance(_Bogus(), JointSafetyFilter)


def test_ego_only_filter_returns_array_shape_matching_ego_action_dim() -> None:
    """Smoke check that the typed constructor's runtime behaviour matches the EGO_ONLY contract."""
    from concerto.safety.api import Bounds, SafetyState
    from concerto.safety.cbf_qp import AgentSnapshot

    cbf = ExpCBFQP.ego_only(control_models=_models())
    snaps = {
        "ego": AgentSnapshot(
            position=np.array([0.0, 0.0], dtype=np.float64),
            velocity=np.array([0.0, 0.0], dtype=np.float64),
            radius=0.2,
        ),
        "partner": AgentSnapshot(
            position=np.array([5.0, 0.0], dtype=np.float64),
            velocity=np.array([0.0, 0.0], dtype=np.float64),
            radius=0.2,
        ),
    }
    safe, info = cbf.filter(
        np.zeros(2, dtype=np.float64),
        {"agent_states": snaps, "meta": {"partner_id": None}},
        SafetyState(lambda_=np.zeros(1, dtype=np.float64)),
        Bounds(action_norm=2.0, action_rate=0.5, comm_latency_ms=1.0, force_limit=20.0),
        ego_uid="ego",
        partner_predicted_states={"partner": snaps["partner"]},
    )
    assert safe.shape == (2,)
    assert info["constraint_violation"].shape == (1,)
