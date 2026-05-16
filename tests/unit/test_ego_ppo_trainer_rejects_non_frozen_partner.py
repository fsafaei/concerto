# SPDX-License-Identifier: Apache-2.0
"""``EgoPPOTrainer.from_config`` must reject a non-frozen partner
(ADR-009 §Consequences; ADR-002 §Risks #1; plan/05 §6 #3).

The black-box AHT contract (ADR-009 §Decision) requires every partner
to be frozen during ego training. The :mod:`chamber.partners.interface`
shield enforces this at attribute-lookup time for every
:class:`~chamber.partners.interface.PartnerBase` subclass — a careless
caller that reaches for ``partner.named_parameters()`` from the ego
trainer hits :class:`AttributeError` with the ADR-009 message.

But that shield does not catch a *custom* partner adapter (e.g. a
research-fork experiment or a regression-test fixture) that does
expose torch parameters with ``requires_grad=True``. Plan/05 §6 #3
pins the trainer-side guard: ``EgoPPOTrainer.from_config`` must refuse
to construct against such a partner, citing ADR-009 §Consequences and
the offending parameter path so the error is debuggable without
spelunking the partner adapter.

Tests pin both paths:

- A bare :class:`torch.nn.Module` partner with one
  ``requires_grad=True`` parameter → :class:`ValueError` carrying
  ``ADR-009 §Consequences`` + the parameter path.
- A bare :class:`torch.nn.Module` partner whose every parameter has
  ``requires_grad=False`` constructs without raise.
- A :class:`~chamber.partners.heuristic.ScriptedHeuristicPartner`
  (real :class:`PartnerBase` subclass; the shield blocks
  ``named_parameters`` access) constructs without raise.
"""

from __future__ import annotations

import numpy as np
import pytest
from torch import nn

from chamber.benchmarks.ego_ppo_trainer import EgoPPOTrainer
from chamber.envs.mpe_cooperative_push import MPECooperativePushEnv
from chamber.partners.api import PartnerSpec
from chamber.partners.heuristic import ScriptedHeuristicPartner
from concerto.training.config import (
    EgoAHTConfig,
    EnvConfig,
    HAPPOHyperparams,
    PartnerConfig,
    RuntimeConfig,
)


def _tiny_cfg() -> EgoAHTConfig:
    """Minimal config sized for fast-CI unit construction."""
    return EgoAHTConfig(
        seed=0,
        total_frames=16,
        checkpoint_every=16,
        env=EnvConfig(task="mpe_cooperative_push", episode_length=16),
        partner=PartnerConfig(
            class_name="scripted_heuristic",
            extra={"uid": "partner", "target_xy": "0.0,0.0", "action_dim": "2"},
        ),
        happo=HAPPOHyperparams(rollout_length=16, batch_size=16, n_epochs=1, hidden_dim=32),
        runtime=RuntimeConfig(device="cpu"),
    )


class _NonFrozenPartner(nn.Module):
    """A bare ``nn.Module`` partner with one trainable parameter (regression fixture).

    Deliberately does NOT extend :class:`chamber.partners.interface.PartnerBase`,
    so the :data:`_FORBIDDEN_ATTRS` shield does not block
    ``named_parameters`` — the trainer-side
    ``_assert_partner_is_frozen`` check is the only line of defence.
    """

    def __init__(self) -> None:
        super().__init__()
        # One named parameter whose path is predictable in the error message.
        self.head = nn.Linear(in_features=4, out_features=2, bias=False)

    def reset(self, *, seed: int | None = None) -> None:
        del seed

    def act(
        self,
        obs: object,
        *,
        deterministic: bool = True,
    ) -> np.ndarray:  # type: ignore[type-arg]
        del obs, deterministic
        return np.zeros(2, dtype=np.float32)


class _FrozenBareModulePartner(nn.Module):
    """Bare ``nn.Module`` partner with every parameter explicitly frozen."""

    def __init__(self) -> None:
        super().__init__()
        self.head = nn.Linear(in_features=4, out_features=2, bias=False)
        for param in self.parameters():
            param.requires_grad = False

    def reset(self, *, seed: int | None = None) -> None:
        del seed

    def act(
        self,
        obs: object,
        *,
        deterministic: bool = True,
    ) -> np.ndarray:  # type: ignore[type-arg]
        del obs, deterministic
        return np.zeros(2, dtype=np.float32)


def _build_scripted_partner() -> ScriptedHeuristicPartner:
    """Real PartnerBase subclass; the shield blocks ``named_parameters``."""
    return ScriptedHeuristicPartner(
        spec=PartnerSpec(
            class_name="scripted_heuristic",
            seed=0,
            checkpoint_step=None,
            weights_uri=None,
            extra={"uid": "partner", "target_xy": "0.0,0.0", "action_dim": "2"},
        )
    )


class TestFromConfigRejectsNonFrozenPartner:
    """``EgoPPOTrainer.from_config`` partner-freeze gate (ADR-009 §Consequences)."""

    def test_raises_value_error_with_adr_009_and_parameter_path(self) -> None:
        """A partner with one trainable parameter → ``ValueError`` citing the path."""
        cfg = _tiny_cfg()
        env = MPECooperativePushEnv(root_seed=0)
        partner = _NonFrozenPartner()
        with pytest.raises(ValueError, match=r"ADR-009 §Consequences") as excinfo:
            EgoPPOTrainer.from_config(cfg, env=env, partner=partner, ego_uid="ego")
        msg = str(excinfo.value)
        # The offending parameter path is part of the message so a debugger
        # can jump straight to the unfrozen weight.
        assert "head.weight" in msg, (
            f"expected parameter path 'head.weight' in error message; got: {msg!r}"
        )

    def test_accepts_bare_module_partner_with_all_params_frozen(self) -> None:
        """Every parameter explicitly frozen → constructs without raise."""
        cfg = _tiny_cfg()
        env = MPECooperativePushEnv(root_seed=0)
        partner = _FrozenBareModulePartner()
        trainer = EgoPPOTrainer.from_config(cfg, env=env, partner=partner, ego_uid="ego")
        assert isinstance(trainer, EgoPPOTrainer)

    def test_accepts_partner_with_partnerbase_shield(self) -> None:
        """``ScriptedHeuristicPartner`` (PartnerBase subclass) → constructs.

        :class:`PartnerBase.__getattr__` raises :class:`AttributeError` on
        ``named_parameters`` access; the trainer-side check treats that as
        ``contract enforced by the shield`` and proceeds.
        """
        cfg = _tiny_cfg()
        env = MPECooperativePushEnv(root_seed=0)
        partner = _build_scripted_partner()
        trainer = EgoPPOTrainer.from_config(cfg, env=env, partner=partner, ego_uid="ego")
        assert isinstance(trainer, EgoPPOTrainer)


class TestAssertPartnerIsFrozenHelper:
    """Direct contract pins for the private helper (ADR-009 §Consequences)."""

    def test_helper_returns_on_partnerbase_shield(self) -> None:
        """Module-level helper accepts a PartnerBase-subclass partner."""
        from chamber.benchmarks.ego_ppo_trainer import _assert_partner_is_frozen

        _assert_partner_is_frozen(_build_scripted_partner())

    def test_helper_returns_on_frozen_module(self) -> None:
        """Module-level helper accepts a bare nn.Module with all params frozen."""
        from chamber.benchmarks.ego_ppo_trainer import _assert_partner_is_frozen

        _assert_partner_is_frozen(_FrozenBareModulePartner())

    def test_helper_raises_on_first_unfrozen_parameter(self) -> None:
        """The helper raises immediately on the first unfrozen parameter found."""
        from chamber.benchmarks.ego_ppo_trainer import _assert_partner_is_frozen

        with pytest.raises(ValueError, match=r"ADR-009 §Consequences"):
            _assert_partner_is_frozen(_NonFrozenPartner())
