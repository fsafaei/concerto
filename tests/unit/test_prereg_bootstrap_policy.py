# SPDX-License-Identifier: Apache-2.0
"""Reviewer P1-5: enforce iid-not-allowed-for-leaderboard bootstrap policy.

Pre-registration YAMLs declare a ``run_purpose`` (``"leaderboard"`` /
``"power"`` / ``"debug"``). Only ``"leaderboard"`` runs are admitted to
the public ranking, and only those runs are barred from using the
pooled IID bootstrap on seed-clustered episode data — which would
understate CI width (ADR-007 §Pre-registration discipline,
docs/reference/evaluation.md §3.3).
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from chamber.evaluation.prereg import PreregistrationSpec
from chamber.evaluation.results import ConditionPair


def _kwargs(
    *,
    bootstrap_method: str,
    run_purpose: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "axis": "CM",
        "condition_pair": ConditionPair(
            homogeneous_id="homo_panda_panda",
            heterogeneous_id="hetero_panda_fetch",
        ),
        "seeds": [11, 22, 33],
        "episodes_per_seed": 5,
        "estimator": "iqm_success_rate",
        "bootstrap_method": bootstrap_method,
        "failure_policy": "strict",
        "git_tag": "prereg/test",
        "notes": "policy-matrix unit test",
    }
    if run_purpose is not None:
        payload["run_purpose"] = run_purpose
    return payload


def test_default_run_purpose_is_leaderboard() -> None:
    """Default :attr:`run_purpose` is ``"leaderboard"`` (reviewer P1-5)."""
    spec = PreregistrationSpec(**_kwargs(bootstrap_method="cluster"))
    assert spec.run_purpose == "leaderboard"


def test_leaderboard_iid_is_rejected() -> None:
    """``leaderboard`` + ``iid`` raises a ``ValidationError`` (reviewer P1-5)."""
    with pytest.raises(ValidationError, match="iid bootstrap is not permitted"):
        PreregistrationSpec(
            **_kwargs(bootstrap_method="iid", run_purpose="leaderboard"),
        )


def test_leaderboard_iid_is_rejected_via_default() -> None:
    """The same rule fires when ``run_purpose`` is omitted (default = leaderboard)."""
    with pytest.raises(ValidationError, match="iid bootstrap is not permitted"):
        PreregistrationSpec(**_kwargs(bootstrap_method="iid"))


def test_leaderboard_cluster_is_ok() -> None:
    """``leaderboard`` + ``cluster`` is the canonical leaderboard configuration."""
    spec = PreregistrationSpec(
        **_kwargs(bootstrap_method="cluster", run_purpose="leaderboard"),
    )
    assert spec.run_purpose == "leaderboard"
    assert spec.bootstrap_method == "cluster"


def test_leaderboard_hierarchical_is_ok() -> None:
    """``leaderboard`` + ``hierarchical`` is the documented back-compat alias."""
    spec = PreregistrationSpec(
        **_kwargs(bootstrap_method="hierarchical", run_purpose="leaderboard"),
    )
    assert spec.run_purpose == "leaderboard"
    assert spec.bootstrap_method == "hierarchical"


def test_power_iid_is_allowed() -> None:
    """``power`` runs may use ``iid`` (reviewer P1-5; power baselines)."""
    spec = PreregistrationSpec(
        **_kwargs(bootstrap_method="iid", run_purpose="power"),
    )
    assert spec.run_purpose == "power"
    assert spec.bootstrap_method == "iid"


def test_debug_iid_is_allowed() -> None:
    """``debug`` runs may use ``iid`` (reviewer P1-5; not admitted to leaderboard)."""
    spec = PreregistrationSpec(
        **_kwargs(bootstrap_method="iid", run_purpose="debug"),
    )
    assert spec.run_purpose == "debug"
    assert spec.bootstrap_method == "iid"


def test_unknown_run_purpose_is_rejected() -> None:
    """An unknown ``run_purpose`` label is rejected by the ``Literal`` typing."""
    with pytest.raises(ValidationError):
        PreregistrationSpec(
            **_kwargs(bootstrap_method="cluster", run_purpose="production"),
        )
