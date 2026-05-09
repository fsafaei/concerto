# SPDX-License-Identifier: Apache-2.0
"""Hypothesis property tests for ``chamber.partners.interface`` (T4.3 / plan/04 §5).

Covers ADR-009 §Consequences: any name in ``_FORBIDDEN_ATTRS`` raises with
the ADR-009 message; any other random name raises with Python's default
attribute-error message.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from chamber.partners.api import PartnerSpec
from chamber.partners.interface import _FORBIDDEN_ATTRS, PartnerBase


class _Concrete(PartnerBase):
    """Minimal concrete subclass used as the property fixture."""

    def reset(self, *, seed: int | None = None) -> None:
        del seed

    def act(self, obs, *, deterministic: bool = True):  # type: ignore[no-untyped-def]
        del obs, deterministic
        return np.zeros(2, dtype=np.float32)


def _spec() -> PartnerSpec:
    return PartnerSpec(class_name="x", seed=0, checkpoint_step=None, weights_uri=None)


@given(name=st.sampled_from(sorted(_FORBIDDEN_ATTRS)))
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=50)
def test_every_forbidden_name_raises_with_adr_009(name: str) -> None:
    """ADR-009 §Consequences: every forbidden name produces the ADR-cited error."""
    partner = _Concrete(_spec())
    with pytest.raises(AttributeError, match="ADR-009") as excinfo:
        getattr(partner, name)
    assert name in str(excinfo.value)


def _name_is_truly_missing(name: str) -> bool:
    """Filter to names that are not already resolvable on PartnerBase.

    Excludes ``_FORBIDDEN_ATTRS`` (those should hit the shield) and excludes
    anything Python's standard MRO would resolve (e.g. ``__dict__``,
    ``__class__``, ``spec`` etc.). What remains is the set of "looks like an
    attribute but isn't on the object" cases the shield exists to handle.
    """
    if name in _FORBIDDEN_ATTRS:
        return False
    return not hasattr(_Concrete(_spec()), name)


@given(
    name=st.text(
        alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd"), whitelist_characters="_"),
        min_size=1,
        max_size=24,
    ).filter(lambda s: s.isidentifier() and _name_is_truly_missing(s)),
)
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=100)
def test_arbitrary_other_names_raise_default_message(name: str) -> None:
    """ADR-009 §Consequences: non-forbidden names raise the bare default form.

    The bare form is critical so ``hasattr(partner, name)`` returns False for
    benign attribute checks. If the shield used the ADR-009 message for every
    miss, ``hasattr`` would raise rather than returning False, and
    ``isinstance`` checks against unrelated Protocols would behave oddly.
    """
    partner = _Concrete(_spec())
    with pytest.raises(AttributeError) as excinfo:
        getattr(partner, name)
    assert "ADR-009" not in str(excinfo.value)
    assert name in str(excinfo.value)
