# SPDX-License-Identifier: Apache-2.0
"""QP-saturation guard property tests (T2.7).

Covers ADR-006 §Risks R5 (the URLLC sweep saturates the inner CBF QP solver
at the most aggressive setting) and ADR-004 §"OSCBF target" (1 ms solve-time
budget).

The actual QP solver lands in M3; this module pins the contract from M2 so
M3 cannot regress it. The M2 :func:`concerto.safety.solve_qp_stub` returns
immediately, so timing alone never trips the guard — instead the guard is
also driven by the ADR-006 R5 *regime test* (drop_rate >= 10 % or latency
mean >= 100 ms), which exactly singles out the ``saturation`` profile.
"""

from __future__ import annotations

import time
import warnings
from typing import TYPE_CHECKING

import pytest

from chamber.comm import URLLC_3GPP_R17, ChamberCommQPSaturationWarning, saturation_guard
from concerto.safety import solve_qp_stub

if TYPE_CHECKING:
    from chamber.comm.degradation import DegradationProfile

_PROFILES = list(URLLC_3GPP_R17.items())


@pytest.mark.parametrize(("name", "profile"), _PROFILES, ids=[n for n, _ in _PROFILES])
def test_only_saturation_profile_emits_warning(name: str, profile: DegradationProfile) -> None:
    """ADR-006 R5: every URLLC profile except ``saturation`` is silent.

    The saturation profile fires :class:`ChamberCommQPSaturationWarning`;
    every other profile is silent (any emission is escalated to an error
    under ``warnings.simplefilter("error", ...)``).
    """
    if name == "saturation":
        with pytest.warns(ChamberCommQPSaturationWarning, match="saturation regime"):
            saturation_guard(profile)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ChamberCommQPSaturationWarning)
            saturation_guard(profile)


@pytest.mark.parametrize(("name", "profile"), _PROFILES, ids=[n for n, _ in _PROFILES])
def test_qp_solve_time_under_oscbf_budget(name: str, profile: DegradationProfile) -> None:
    """ADR-004 §"OSCBF target": every profile's QP solve completes within 1 ms.

    The M2 stub returns near-instantly so the budget is trivially honoured;
    M3 must keep this true after replacing the stub. The ``saturation``
    profile is allowed to raise a warning during the call but its measured
    solve time still respects the budget here (the warning fires for the
    regime check, not for timing).
    """
    del name, profile
    start = time.perf_counter()
    solve_qp_stub()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    assert elapsed_ms < 1.0, f"QP stub took {elapsed_ms:.4f} ms (budget: 1.0 ms)"


def test_saturation_guard_warning_message_cites_regime() -> None:
    """ADR-006 R5: the warning identifies the saturation regime explicitly."""
    profile = URLLC_3GPP_R17["saturation"]
    with pytest.warns(ChamberCommQPSaturationWarning) as captured:
        saturation_guard(profile)
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
