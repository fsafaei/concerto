# SPDX-License-Identifier: Apache-2.0
"""Conformal slack lambda update — TDD stub for T3.6 (ADR-004 §Decision).

Placeholder module so the failing reproduction test
(``tests/reproduction/test_repro_huriot_sibai_table1.py``) imports
cleanly while exercising the contract from plan/03 §3.3. The real
implementation — Theorem 3's update rule
``lambda_{k+1} = lambda_k + eta * (eps - l_k)`` plus the constant-
velocity predictor stub and partner-swap reset (ADR-004 risk-mitigation
#2) — lands in the follow-up commit and makes the test pass.

Plan/03 §8 prescribes this TDD sequence: commit the failing test first
so the sign of the update rule (easy to flip) and the sign of epsilon
during warmup (also easy to flip) are pinned by the reproduction's
+/-5% tolerance before any code lands.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from concerto.safety.api import FloatArray, SafetyState


def update_lambda(
    state: SafetyState,
    loss_k: FloatArray,
    *,
    in_warmup: bool = False,
) -> None:
    """TDD stub for the conformal slack update (ADR-004 §Decision; T3.6).

    Raises :class:`NotImplementedError` until the follow-up commit lands
    the real ``lambda_{k+1} = lambda_k + eta * (eps - l_k)`` rule.
    """
    del state, loss_k, in_warmup
    msg = "concerto.safety.conformal.update_lambda lands in T3.6 follow-up commit"
    raise NotImplementedError(msg)


__all__ = ["update_lambda"]
