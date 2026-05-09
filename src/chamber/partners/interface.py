# SPDX-License-Identifier: Apache-2.0
"""Partner base class providing the AHT-frozen-partner shield (ADR-009 §Consequences).

The :class:`PartnerBase` class is the runtime enforcement point for the
black-box AHT no-joint-training constraint. Any attribute lookup that would
let the ego training loop nudge the partner's weights (``train``, ``learn``,
``update_params``, ``fit``, ``set_weights``, ``load_state_dict``,
``named_parameters``) raises :class:`AttributeError` referencing ADR-009.

The shield is intentionally over-broad: it also blocks legitimate read-only
inspection of parameters (e.g. for MEP entropy computation in Phase 1). When
Phase 1 needs that, the path is a separate read-only inspector module and an
ADR amendment — NOT a relaxation of this list (plan/04 §8 Notes).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NoReturn

if TYPE_CHECKING:
    from chamber.partners.api import PartnerSpec

#: Names that subclasses MUST NOT expose — runtime enforcement of black-box AHT.
#:
#: ADR-009 §Consequences: partners are frozen during ego training; this set is
#: the names a careless caller might reach for to bypass the freeze.
_FORBIDDEN_ATTRS: frozenset[str] = frozenset(
    {
        "train",
        "learn",
        "update",
        "update_params",
        "fit",
        "set_weights",
        "load_state_dict",
        "named_parameters",
    }
)


class PartnerBase:
    """Common base class for every concrete partner (ADR-009 §Consequences).

    Subclasses MUST implement ``reset(*, seed)`` and
    ``act(obs, *, deterministic)`` per the
    :class:`~chamber.partners.api.FrozenPartner` Protocol. The base class
    blocks any attribute lookup in :data:`_FORBIDDEN_ATTRS` so the AHT
    no-joint-training constraint is enforced even when a careless caller
    imports the underlying torch module directly.

    Attributes:
        spec: The :class:`~chamber.partners.api.PartnerSpec` the partner was
            constructed from. Read-only by convention; the registry assumes
            ``Class(spec)`` is the only constructor signature.
    """

    spec: PartnerSpec

    def __init__(self, spec: PartnerSpec) -> None:
        """Bind the partner spec (ADR-009 §Decision).

        Args:
            spec: Identity-bearing handle from
                :class:`~chamber.partners.api.PartnerSpec`. The constructor
                does no checkpoint I/O so subclasses can defer heavy loads
                until :meth:`reset`.
        """
        self.spec = spec

    def __getattr__(self, name: str) -> NoReturn:
        """Block forbidden attribute lookups (ADR-009 §Consequences).

        ``__getattr__`` is only invoked for names not found via the normal
        attribute lookup path, so this method does not interfere with
        legitimate methods defined on subclasses or with ``self.spec``.

        Args:
            name: Attribute name being looked up.

        Raises:
            AttributeError: Always — either with the ADR-009 message for
                forbidden names, or with the Pythonic-default message for
                anything else (so ``hasattr`` works as expected).
        """
        if name in _FORBIDDEN_ATTRS:
            raise AttributeError(
                f"FrozenPartner of class {type(self).__name__!r} forbids attribute "
                f"{name!r} (ADR-009 §Consequences: black-box AHT — no joint training)."
            )
        raise AttributeError(name)


__all__ = ["PartnerBase"]
