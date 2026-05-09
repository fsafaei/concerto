# SPDX-License-Identifier: Apache-2.0
"""Partner-class registry (ADR-009 §Decision; plan/04 §3.2 — P7 plug-in registry).

Partner classes register themselves by stable string id via the
:func:`register_partner` decorator. The :func:`load_partner` factory builds an
instance from a :class:`~chamber.partners.api.PartnerSpec` by looking up the
registered class. This is the single extension point for Phase-1 partners
(OpenVLA, CrossFormer, additional HARL variants) — adding a partner is a
decorator + a module import; nothing else in the call-graph changes.

The registry is a process-global mapping; double-registration of the same
``class_name`` raises :class:`ValueError` to surface accidental name
collisions loudly. Listing the registry (e.g. for the eval CLI) is supported
via :func:`list_registered`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from chamber.partners.api import FrozenPartner, PartnerSpec

if TYPE_CHECKING:
    from collections.abc import Callable

T = TypeVar("T", bound=FrozenPartner)

_REGISTRY: dict[str, type[FrozenPartner]] = {}


def register_partner(class_name: str) -> Callable[[type[T]], type[T]]:
    """Register a partner class by name (ADR-009 §Decision; plan/04 §3.2).

    Used as a class decorator:

    .. code-block:: python

        @register_partner("scripted_heuristic")
        class ScriptedHeuristicPartner(PartnerBase): ...

    Args:
        class_name: Registry key. Must be unique process-wide; re-registering
            the same key raises :class:`ValueError` so name collisions are
            caught at import time.

    Returns:
        A decorator that records the class against ``class_name`` and returns
        the class unchanged.

    Raises:
        ValueError: If ``class_name`` is already registered.
    """

    def _decorator(cls: type[T]) -> type[T]:
        if class_name in _REGISTRY:
            raise ValueError(f"Partner class {class_name!r} is already registered")
        _REGISTRY[class_name] = cls
        return cls

    return _decorator


def load_partner(spec: PartnerSpec) -> FrozenPartner:
    """Build a partner instance from its spec via the registry (ADR-009 §Decision).

    Args:
        spec: The :class:`~chamber.partners.api.PartnerSpec` whose
            :attr:`~chamber.partners.api.PartnerSpec.class_name` selects the
            registered class.

    Returns:
        A fresh :class:`~chamber.partners.api.FrozenPartner` instance built
        by ``Class(spec)``.

    Raises:
        KeyError: If ``spec.class_name`` is not registered. The error message
            lists the currently-registered keys to make typos easy to fix.
    """
    if spec.class_name not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY.keys())) or "<none>"
        raise KeyError(f"No partner class registered as {spec.class_name!r}; known: {known}")
    cls = _REGISTRY[spec.class_name]
    # ``FrozenPartner`` is a Protocol without a constructor signature, so
    # pyright cannot prove ``cls(spec)`` is well-typed. The registry's
    # contract (plan/04 §3.2) is that every registered class accepts
    # ``__init__(self, spec: PartnerSpec)`` — :class:`PartnerBase` enforces
    # this. Do NOT widen the Protocol with ``__init__`` to silence this.
    return cls(spec)  # type: ignore[call-arg]


def list_registered() -> list[str]:
    """Return the sorted list of registered partner class names (ADR-009 §Decision).

    Used by the eval CLI to enumerate the available zoo strata and by tests
    to assert the Phase-0 draft zoo classes are wired up.

    Returns:
        A new list, freshly sorted on every call.
    """
    return sorted(_REGISTRY.keys())


__all__ = ["list_registered", "load_partner", "register_partner"]
