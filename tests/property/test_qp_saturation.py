# SPDX-License-Identifier: Apache-2.0
"""QP-saturation guard property tests (T2.7).

Covers ADR-006 §Risks R5 (the URLLC sweep saturates the inner CBF QP solver
at the most aggressive setting) and ADR-004 §"OSCBF target" (1 ms solve-time
budget).

:func:`concerto.safety.solve_qp_stub` is no longer M2's no-op: T3.10 replaced
it with a real Clarabel solve on a small representative QP. Its wall-clock
cost is therefore genuine and, under ``pytest --cov`` instrumentation on a
shared runner, can land near the 1 ms budget. The two contracts pinned here
are kept independent of that noise:

  * the ADR-006 R5 *regime* test (drop_rate >= 10 % or latency mean >= 100 ms)
    singles out the ``saturation`` profile. It is asserted with the timing
    branch disabled (``qp_time_budget_ms=math.inf``) so a slow-but-correct
    solve cannot masquerade as a regime hit. The timing branch keeps its own
    deterministic coverage in
    :func:`test_saturation_guard_fires_on_synthetic_slow_qp`.
  * the ADR-004 budget is asserted on the *fastest* of several solves.
    Scheduler preemption and coverage tracing only ever add time, so the
    minimum is the honest estimate of the solver's cost.
"""

from __future__ import annotations

import math
import time
import warnings
from typing import TYPE_CHECKING

import pytest

from chamber.comm import URLLC_3GPP_R17, ChamberCommQPSaturationWarning, saturation_guard
from concerto.safety import solve_qp_stub

if TYPE_CHECKING:
    from chamber.comm.degradation import DegradationProfile

_PROFILES = list(URLLC_3GPP_R17.items())

#: ADR-004 §"OSCBF target" solve-time budget, mirrored from
#: ``chamber.comm.degradation._QP_TIME_BUDGET_MS``.
_OSCBF_BUDGET_MS: float = 1.0

#: Number of solves timed in the budget check; the fastest one is asserted.
_TIMING_SAMPLES: int = 5


def _measure_solve_ms() -> float:
    """Return the wall-clock cost of one :func:`solve_qp_stub` call, in ms."""
    start = time.perf_counter()
    solve_qp_stub()
    return (time.perf_counter() - start) * 1000.0


@pytest.mark.parametrize(("name", "profile"), _PROFILES, ids=[n for n, _ in _PROFILES])
def test_only_saturation_profile_emits_warning(name: str, profile: DegradationProfile) -> None:
    """ADR-006 R5: every URLLC profile except ``saturation`` is silent.

    The saturation profile fires :class:`ChamberCommQPSaturationWarning`;
    every other profile is silent (any emission is escalated to an error
    under ``warnings.simplefilter("error", ...)``).

    ``qp_time_budget_ms=math.inf`` disables the guard's timing branch so this
    asserts the regime contract alone — see the module docstring.
    """
    if name == "saturation":
        with pytest.warns(ChamberCommQPSaturationWarning, match="saturation regime"):
            saturation_guard(profile, qp_time_budget_ms=math.inf)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ChamberCommQPSaturationWarning)
            saturation_guard(profile, qp_time_budget_ms=math.inf)


@pytest.mark.parametrize(("name", "profile"), _PROFILES, ids=[n for n, _ in _PROFILES])
def test_qp_solve_time_under_oscbf_budget(name: str, profile: DegradationProfile) -> None:
    """ADR-004 §"OSCBF target": the QP solve completes within 1 ms.

    Asserted on the fastest of :data:`_TIMING_SAMPLES` solves so coverage
    instrumentation and scheduler noise — which only ever add time — cannot
    turn a fast solver into a red build. M3 must keep this true.
    """
    del name, profile
    best_ms = min(_measure_solve_ms() for _ in range(_TIMING_SAMPLES))
    assert best_ms < _OSCBF_BUDGET_MS, (
        f"fastest of {_TIMING_SAMPLES} QP solves took {best_ms:.4f} ms "
        f"(budget: {_OSCBF_BUDGET_MS} ms)"
    )


def test_saturation_guard_warning_message_cites_regime() -> None:
    """ADR-006 R5: the warning identifies the saturation regime explicitly."""
    profile = URLLC_3GPP_R17["saturation"]
    with pytest.warns(ChamberCommQPSaturationWarning) as captured:
        saturation_guard(profile, qp_time_budget_ms=math.inf)
    assert any("saturation regime" in str(w.message) for w in captured)


def test_saturation_guard_fires_on_synthetic_slow_qp() -> None:
    """ADR-004 §"OSCBF target": a slow custom solver also trips the guard.

    Provides a slow ``qp_solve_fn`` that exceeds the 1 ms budget and asserts
    the warning fires even for the otherwise-feasible ``ideal`` profile. M3
    will replace the default solver with the real one; this case ensures
    the timing branch is covered today.
    """

    def _slow(*args: object, **kwargs: object) -> tuple[float, float]:
        del args, kwargs
        time.sleep(0.005)  # 5 ms — exceeds the 1 ms budget
        return (0.0, 5e-3)

    with pytest.warns(ChamberCommQPSaturationWarning, match="exceeds"):
        saturation_guard(URLLC_3GPP_R17["ideal"], _slow)
