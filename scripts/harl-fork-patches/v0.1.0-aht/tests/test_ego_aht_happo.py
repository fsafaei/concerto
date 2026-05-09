# SPDX-License-Identifier: Apache-2.0
"""Smoke tests for ``harl.algorithms.actors.ego_aht_happo`` (CONCERTO T4b.6).

Lives inside the ``concerto-org/harl-fork`` repo at
``tests/test_ego_aht_happo.py``. Generated from CONCERTO's
``scripts/harl-fork-patches/v0.1.0-aht/`` recipe.

Verifies the structural contract:

- :class:`EgoAHTHAPPO` subclasses :class:`HAPPO`.
- :meth:`_validate_partner_is_frozen` rejects a partner with any
  trainable parameter (ADR-009 §Consequences).
- :meth:`_validate_partner_is_frozen` accepts a properly-frozen
  partner (no exception at construction).

The deeper integration tests (full collect_rollout / update path,
end-to-end empirical-guarantee assertion) live on the CONCERTO side as
``tests/integration/test_empirical_guarantee.py`` (M4b-8 / T4b.13).
"""

from __future__ import annotations

import pytest

# UPSTREAM-VERIFY: same import-path caveat as ego_aht_happo.py.
from harl.algorithms.actors.ego_aht_happo import EgoAHTHAPPO
from harl.algorithms.actors.happo import HAPPO
from torch import nn


class _FakeFrozenPartner(nn.Module):
    """Minimal partner stub with a frozen torch parameter."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(10, 2)
        for param in self.linear.parameters():
            param.requires_grad = False


class _FakeNonFrozenPartner(nn.Module):
    """Minimal partner stub with at least one trainable parameter."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(10, 2)  # default requires_grad=True


# UPSTREAM-VERIFY: Replace ``...`` with the actual upstream HAPPO
# constructor args once you have the signature. Common HARL releases
# accept (args, obs_space, act_space, device); pass minimal stubs
# here so the construction path runs.
_HAPPO_INIT_ARGS: tuple = ()
_HAPPO_INIT_KWARGS: dict = {}


def test_ego_aht_happo_subclasses_happo() -> None:
    """Plan/05 §3.2 + ADR-002 §Decisions: drop-in subclass of HAPPO."""
    assert issubclass(EgoAHTHAPPO, HAPPO)


@pytest.mark.skip(reason="Fill in _HAPPO_INIT_ARGS once upstream signature is verified")
def test_ego_aht_happo_refuses_non_frozen_partner() -> None:
    """ADR-009 §Consequences: AHT contract requires every partner param frozen."""
    with pytest.raises(ValueError, match="ADR-009"):
        EgoAHTHAPPO(
            *_HAPPO_INIT_ARGS,
            partner_adapter=_FakeNonFrozenPartner(),
            **_HAPPO_INIT_KWARGS,
        )


@pytest.mark.skip(reason="Fill in _HAPPO_INIT_ARGS once upstream signature is verified")
def test_ego_aht_happo_accepts_frozen_partner() -> None:
    """ADR-009 §Consequences: a properly-frozen partner is accepted."""
    EgoAHTHAPPO(
        *_HAPPO_INIT_ARGS,
        partner_adapter=_FakeFrozenPartner(),
        **_HAPPO_INIT_KWARGS,
    )  # No exception.


def test_frozen_partner_helper_actually_freezes() -> None:
    """Self-test on the test fixture itself: every param is frozen."""
    partner = _FakeFrozenPartner()
    for param in partner.parameters():
        assert param.requires_grad is False


def test_non_frozen_partner_helper_actually_trains() -> None:
    """Self-test on the test fixture itself: at least one param is trainable."""
    partner = _FakeNonFrozenPartner()
    assert any(p.requires_grad for p in partner.parameters())
