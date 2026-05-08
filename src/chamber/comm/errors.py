# SPDX-License-Identifier: Apache-2.0
"""Comm-stack error and warning types.

ADR-003 §Decision (every concrete channel raises :class:`ChamberCommError`
on contract violations, never silently degrades). ADR-006 §Risks #5 / Risk
register R5 (the QP-saturation guard fires :class:`ChamberCommQPSaturationWarning`
when the URLLC sweep saturates the inner CBF QP solver).
"""

from __future__ import annotations


class ChamberCommError(RuntimeError):
    """Raised when a comm-channel contract is violated (ADR-003 §Decision).

    Examples include malformed packets (missing ``schema_version``,
    unrecognised packet keys), unsupported profile lookups, and channel
    state that cannot be reconciled (e.g., decoding a packet whose
    schema version differs from the channel's).
    """


class ChamberCommQPSaturationWarning(UserWarning):
    """Emitted when a degradation profile saturates the inner CBF QP (ADR-006 R5).

    The QP-saturation guard test in
    ``tests/property/test_qp_saturation.py`` sweeps the
    :data:`chamber.comm.profiles.URLLC_3GPP_R17` table and asserts that
    the inner CBF QP solve time stays below 1 ms (OSCBF target from
    ADR-004). If the most aggressive profile saturates, the wrapper
    raises this warning rather than silently allowing it (ADR-006
    Risk register R5).
    """
