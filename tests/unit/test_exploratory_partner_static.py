# SPDX-License-Identifier: Apache-2.0
"""Tier-1 pins for the EXPLORATORY partner-static knob (2026-06-11 homo-static slice §2).

The structural-gating contract: default-off is byte-identical (no
wrapper constructed); the gate-facing factory REFUSES the flag at
construction; the override emits zero actions of the inner partner's
exact shape while preserving the FrozenPartner Protocol surface; the
flag stamps the run's JSONL extra. ADR-002 (default-off identity);
ADR-009 (the partner contract the wrapper preserves).
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from chamber.benchmarks.stage1_common import TrainedPolicyFactory
from chamber.partners.api import PartnerSpec
from chamber.partners.exploratory.static_override import ExploratoryStaticPartnerOverride
from chamber.partners.heuristic import ScriptedHeuristicPartner
from chamber.partners.registry import list_registered
from concerto.training.config import (
    EgoAHTConfig,
    EnvConfig,
    ExploratoryConfig,
)


def _partner(action_dim: int = 13) -> ScriptedHeuristicPartner:
    return ScriptedHeuristicPartner(
        spec=PartnerSpec(
            class_name="scripted_heuristic",
            seed=0,
            checkpoint_step=None,
            weights_uri=None,
            extra={"uid": "fetch", "target_xy": "0.5,0.5", "action_dim": str(action_dim)},
        )
    )


class TestExploratoryConfig:
    def test_default_off(self) -> None:
        """ADR-002: the default config never activates the override."""
        assert ExploratoryConfig().partner_static_override is False
        cfg = EgoAHTConfig(env=EnvConfig(task="mpe_cooperative_push"))
        assert cfg.exploratory.partner_static_override is False


class TestFactoryRefusal:
    def test_gate_facing_factory_refuses_the_flag(self) -> None:
        """The safety-loud-fail pattern: the production dispatch cannot run it."""
        cfg = EgoAHTConfig(
            env=EnvConfig(
                task="stage1_pickplace",
                condition_id="stage1_pickplace_panda_only_mappo_shared_param",
            ),
            exploratory=ExploratoryConfig(partner_static_override=True),
        )
        with pytest.raises(ValueError, match=r"EXPLORATORY|partner_static_override"):
            TrainedPolicyFactory(cfg=cfg)

    def test_factory_accepts_default_off(self) -> None:
        """Default-off constructs normally (no behaviour change)."""
        cfg = EgoAHTConfig(
            env=EnvConfig(
                task="stage1_pickplace",
                condition_id="stage1_pickplace_panda_only_mappo_shared_param",
            ),
        )
        TrainedPolicyFactory(cfg=cfg)  # must not raise


class TestOverrideWrapper:
    def test_zero_actions_with_inner_shape(self) -> None:
        """act() returns zeros of the inner partner's exact action shape."""
        inner = _partner(action_dim=13)
        obs = {"agent": {"fetch": {"state": np.asarray([0.0, 0.0, 0.0], dtype=np.float32)}}}
        raw = inner.act(obs)
        assert np.abs(raw).max() > 0  # the heuristic would move (target 0.5,0.5)
        wrapped = ExploratoryStaticPartnerOverride(inner)
        frozen = wrapped.act(obs)
        assert frozen.shape == raw.shape
        np.testing.assert_array_equal(frozen, 0.0)

    def test_protocol_surface_preserved(self) -> None:
        """spec + reset delegate (PartnerLike structural contract; ADR-009)."""
        inner = _partner()
        wrapped = ExploratoryStaticPartnerOverride(inner)
        assert wrapped.spec is inner.spec
        wrapped.reset(seed=7)  # must not raise


class TestExploratoryQuarantine:
    """The exploratory namespace quarantine (ADR-009 §Decision; ADR-027 §Reporting rules)."""

    def test_deprecated_import_path_warns_and_reexports(self) -> None:
        """The legacy path re-exports the same class under DeprecationWarning."""
        sys.modules.pop("chamber.partners.static_override", None)
        with pytest.warns(DeprecationWarning, match=r"exploratory"):
            from chamber.partners.static_override import (
                ExploratoryStaticPartnerOverride as LegacyExport,
            )
        assert LegacyExport is ExploratoryStaticPartnerOverride

    def test_structurally_ineligible_for_leaderboard_runs(self) -> None:
        """No exploratory partner is reachable through the registry surface.

        Leaderboard-facing runs build partners exclusively via
        ``chamber.partners.registry.load_partner`` (registered classes)
        and the gate-facing factory refuses the exploratory knob
        (:class:`TestFactoryRefusal`); the quarantined namespace has no
        registry entry to load.
        """
        assert not any(
            "static_override" in name or "exploratory" in name for name in list_registered()
        )
