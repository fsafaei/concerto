# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``chamber.partners.interface`` (T4.3).

Covers ADR-009 §Consequences (the ``_FORBIDDEN_ATTRS`` shield).
"""

from __future__ import annotations

import numpy as np
import pytest

from chamber.partners.api import PartnerSpec
from chamber.partners.interface import _FORBIDDEN_ATTRS, PartnerBase


def _make_spec() -> PartnerSpec:
    return PartnerSpec(class_name="x", seed=0, checkpoint_step=None, weights_uri=None)


class _Concrete(PartnerBase):
    """Minimal concrete subclass for shield-behaviour tests."""

    def reset(self, *, seed: int | None = None) -> None:
        del seed

    def act(self, obs, *, deterministic: bool = True):  # type: ignore[no-untyped-def]
        del obs, deterministic
        return np.zeros(2, dtype=np.float32)


class TestForbiddenAttrsList:
    def test_canonical_names_present(self) -> None:
        """ADR-009 §Consequences: shield covers the named-extension-point list."""
        for name in (
            "train",
            "learn",
            "update",
            "update_params",
            "fit",
            "set_weights",
            "load_state_dict",
            "named_parameters",
        ):
            assert name in _FORBIDDEN_ATTRS

    def test_is_frozenset(self) -> None:
        """The shield list is a frozenset so it can't be mutated at runtime."""
        assert isinstance(_FORBIDDEN_ATTRS, frozenset)


class TestShieldBehaviour:
    @pytest.mark.parametrize("forbidden", sorted(_FORBIDDEN_ATTRS))
    def test_forbidden_attribute_raises_with_adr_message(self, forbidden: str) -> None:
        """ADR-009 §Consequences: the AttributeError must cite ADR-009."""
        partner = _Concrete(_make_spec())
        with pytest.raises(AttributeError, match=r"ADR-009") as excinfo:
            getattr(partner, forbidden)
        assert forbidden in str(excinfo.value)
        assert "_Concrete" in str(excinfo.value)

    def test_unknown_attribute_falls_back_to_default_attribute_error(self) -> None:
        """Pythonic default: unknown names raise the bare AttributeError(name) form.

        Important: ``hasattr(partner, "thing")`` must return False for benign
        unknown attributes, so the message must NOT cite ADR-009.
        """
        partner = _Concrete(_make_spec())
        with pytest.raises(AttributeError) as excinfo:
            partner.no_such_attr  # noqa: B018  # access has the side-effect we care about
        assert "ADR-009" not in str(excinfo.value)
        assert "no_such_attr" in str(excinfo.value)

    def test_hasattr_returns_false_for_forbidden(self) -> None:
        """Belt-and-braces: forbidden attrs return False through hasattr too."""
        partner = _Concrete(_make_spec())
        assert hasattr(partner, "train") is False

    def test_hasattr_returns_false_for_unknown(self) -> None:
        """Pythonic default for unrelated attribute lookup."""
        partner = _Concrete(_make_spec())
        assert hasattr(partner, "xyzzy") is False

    def test_legitimate_methods_still_resolve(self) -> None:
        """The shield is __getattr__-only — defined methods resolve normally."""
        partner = _Concrete(_make_spec())
        partner.reset(seed=0)
        out = partner.act({})
        assert isinstance(out, np.ndarray)
        assert partner.spec is not None


class TestSpecBinding:
    def test_init_stores_spec(self) -> None:
        """ADR-009 §Decision: PartnerBase(spec) stores the spec on the instance."""
        spec = _make_spec()
        partner = _Concrete(spec)
        assert partner.spec is spec
