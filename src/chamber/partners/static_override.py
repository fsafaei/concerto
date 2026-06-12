# SPDX-License-Identifier: Apache-2.0
"""EXPLORATORY zero-action partner override (ADR-009 §Decision; 2026-06-11 homo-static slice).

Wraps a built :class:`~chamber.partners.api.FrozenPartner` so ``act()``
returns a zero vector of the inner partner's exact action shape — the
partner stands motionless at its reset pose. The inner partner's class,
spec, registry entry, and freeze contract are untouched (ADR-009: the
wrapper holds no parameters and delegates ``reset``/``spec``).

EXPLORATORY only. Activated solely via
``EgoAHTConfig.exploratory.partner_static_override`` (default off =
this module is never imported), and structurally barred from
gate-facing runs:
:class:`chamber.benchmarks.stage1_common.TrainedPolicyFactory` refuses
the flag at construction, so the production ``chamber-spike`` dispatch
cannot train against a static-override partner without an ADR. The
sole pre-stated use is the 2026-06-11 homo-static slice
(``spikes/results/stage1-failure-investigation/2026-06-11-homo-static-exploratory/PRESTATEMENT.md``
§2), which probes whether the #230 partner-motion asymmetry explains
the AS gate's sign reversal.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping

    from numpy.typing import NDArray

__all__ = ["ExploratoryStaticPartnerOverride"]


class ExploratoryStaticPartnerOverride:
    """Zero-action wrapper over a frozen partner (EXPLORATORY; ADR-009 §Decision).

    ``act()`` calls the inner partner and zeros its output — the
    cheapest shape-faithful freeze (the inner act is deterministic and
    parameter-free for every scripted partner; for any future stateful
    partner the inner call also preserves its internal bookkeeping).

    Args:
        inner: The built partner to freeze. Must satisfy the
            :class:`~concerto.training.ego_aht.PartnerLike` Protocol;
            the wrapper satisfies it too (structural).
    """

    def __init__(self, inner: Any) -> None:  # noqa: ANN401 - PartnerLike is a structural Protocol
        """Bind the inner partner (EXPLORATORY; 2026-06-11 homo-static slice §2)."""
        self._inner = inner
        self.spec = inner.spec

    def reset(self, *, seed: int | None = None) -> None:
        """Delegate reset to the inner partner (ADR-009 §Decision)."""
        self._inner.reset(seed=seed)

    def act(
        self,
        obs: Mapping[str, Any],
        *,
        deterministic: bool = True,
    ) -> NDArray[np.floating]:
        """Zero vector of the inner partner's action shape (EXPLORATORY; ADR-009 §Decision)."""
        return np.zeros_like(np.asarray(self._inner.act(obs, deterministic=deterministic)))
