# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``chamber.partners.registry`` (T4.2).

Covers ADR-009 §Decision (P7 plug-in registry; load_partner factory).
"""

from __future__ import annotations

import numpy as np
import pytest

from chamber.partners import registry as registry_module
from chamber.partners.api import PartnerSpec
from chamber.partners.interface import PartnerBase
from chamber.partners.registry import (
    list_registered,
    load_partner,
    register_partner,
)


@pytest.fixture
def isolated_registry(monkeypatch: pytest.MonkeyPatch) -> dict[str, type]:
    """Swap in an empty registry so test cases don't pollute the global one."""
    fresh: dict[str, type] = {}
    monkeypatch.setattr(registry_module, "_REGISTRY", fresh)
    return fresh


class TestRegisterPartner:
    def test_decorator_records_class(self, isolated_registry: dict[str, type]) -> None:
        """ADR-009 §Decision: decorator side-effect adds to the registry mapping."""

        @register_partner("dummy_a")
        class _Dummy(PartnerBase):
            def reset(self, *, seed: int | None = None) -> None: ...
            def act(self, obs, *, deterministic=True):  # type: ignore[no-untyped-def]
                del obs, deterministic
                return np.zeros(2, dtype=np.float32)

        assert isolated_registry["dummy_a"] is _Dummy

    def test_decorator_returns_class_unchanged(self, isolated_registry: dict[str, type]) -> None:
        """The decorator does not wrap the class — it just records it."""

        @register_partner("dummy_b")
        class _Dummy(PartnerBase):
            def reset(self, *, seed: int | None = None) -> None: ...
            def act(self, obs, *, deterministic=True):  # type: ignore[no-untyped-def]
                del obs, deterministic
                return np.zeros(2, dtype=np.float32)

        assert _Dummy.__name__ == "_Dummy"

    def test_double_register_raises_value_error(self, isolated_registry: dict[str, type]) -> None:
        """ADR-009 §Decision: double-registration is a loud failure."""

        @register_partner("collide")
        class _A(PartnerBase):
            def reset(self, *, seed: int | None = None) -> None: ...
            def act(self, obs, *, deterministic=True):  # type: ignore[no-untyped-def]
                del obs, deterministic
                return np.zeros(2, dtype=np.float32)

        with pytest.raises(ValueError, match="already registered"):

            @register_partner("collide")
            class _B(PartnerBase):
                def reset(self, *, seed: int | None = None) -> None: ...
                def act(self, obs, *, deterministic=True):  # type: ignore[no-untyped-def]
                    del obs, deterministic
                    return np.zeros(2, dtype=np.float32)


class TestLoadPartner:
    def test_load_builds_instance_from_registered_class(
        self, isolated_registry: dict[str, type]
    ) -> None:
        """ADR-009 §Decision: load_partner(spec) returns Class(spec)."""

        @register_partner("loadme")
        class _Built(PartnerBase):
            def reset(self, *, seed: int | None = None) -> None: ...
            def act(self, obs, *, deterministic=True):  # type: ignore[no-untyped-def]
                del obs, deterministic
                return np.zeros(2, dtype=np.float32)

        spec = PartnerSpec(class_name="loadme", seed=1, checkpoint_step=None, weights_uri=None)
        partner = load_partner(spec)
        assert isinstance(partner, _Built)
        assert partner.spec is spec

    def test_load_unknown_class_raises_key_error_listing_known(
        self, isolated_registry: dict[str, type]
    ) -> None:
        """ADR-009 §Decision: unknown class name lists known keys for typo recovery."""

        @register_partner("known_a")
        class _A(PartnerBase):
            def reset(self, *, seed: int | None = None) -> None: ...
            def act(self, obs, *, deterministic=True):  # type: ignore[no-untyped-def]
                del obs, deterministic
                return np.zeros(2, dtype=np.float32)

        spec = PartnerSpec(class_name="missing", seed=0, checkpoint_step=None, weights_uri=None)
        with pytest.raises(KeyError) as excinfo:
            load_partner(spec)
        # KeyError repr wraps the message in quotes; check the inner string.
        assert "missing" in str(excinfo.value)
        assert "known_a" in str(excinfo.value)

    def test_load_when_registry_empty_lists_none(self, isolated_registry: dict[str, type]) -> None:
        """KeyError message stays well-formed when nothing is registered."""
        spec = PartnerSpec(class_name="missing", seed=0, checkpoint_step=None, weights_uri=None)
        with pytest.raises(KeyError, match="<none>"):
            load_partner(spec)


class TestListRegistered:
    def test_list_returns_sorted_keys(self, isolated_registry: dict[str, type]) -> None:
        """ADR-009 §Decision: sorted list for stable CLI output."""

        @register_partner("zeta")
        class _Z(PartnerBase):
            def reset(self, *, seed: int | None = None) -> None: ...
            def act(self, obs, *, deterministic=True):  # type: ignore[no-untyped-def]
                del obs, deterministic
                return np.zeros(2, dtype=np.float32)

        @register_partner("alpha")
        class _A(PartnerBase):
            def reset(self, *, seed: int | None = None) -> None: ...
            def act(self, obs, *, deterministic=True):  # type: ignore[no-untyped-def]
                del obs, deterministic
                return np.zeros(2, dtype=np.float32)

        assert list_registered() == ["alpha", "zeta"]

    def test_list_returns_fresh_list(self, isolated_registry: dict[str, type]) -> None:
        """Mutating the returned list does not affect the registry."""

        @register_partner("solo")
        class _S(PartnerBase):
            def reset(self, *, seed: int | None = None) -> None: ...
            def act(self, obs, *, deterministic=True):  # type: ignore[no-untyped-def]
                del obs, deterministic
                return np.zeros(2, dtype=np.float32)

        out = list_registered()
        out.append("PHANTOM")
        assert "PHANTOM" not in list_registered()


class TestRegistryStartsEmpty:
    def test_registry_is_empty_until_partners_register(self) -> None:
        """Plan/04 §3.2: the registry is built up by ``@register_partner`` decorators.

        This PR ships only the registry mechanism; concrete Phase-0 partners
        (scripted heuristic, frozen MAPPO, frozen HARL) and Phase-1 stubs
        (OpenVLA, CrossFormer) land in subsequent M4a PRs and will surface
        their class names here at import time.
        """
        assert list_registered() == []
