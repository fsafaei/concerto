# SPDX-License-Identifier: Apache-2.0
"""Safety-stack error types (ADR-004 §Decision; ADR-006 §Risks).

ADR-004 commits to the exp CBF-QP backbone with conformal slack overlay and
OSCBF inner filter. When the QP is infeasible even after slack relaxation
(Morton & Pavone 2025 §IV slack + penalty pattern; ADR-006 risk-mitigation
referencing Cavorsi 2022 nested-CBF fallback), the filter raises
:class:`ConcertoSafetyInfeasible` so the caller can route to the hard
braking fallback (ADR-004 risk-mitigation #1, Wang-Ames-Egerstedt 2017
eq. 17). The conformal QP is **not** the recovery path because Theorem 3
(Huriot & Sibai 2025) is an *average-loss* guarantee, not per-step.

The error class name is preserved by plan/10 §2 ("ConcertoSafetyInfeasible
stays — safety is method"), even though sibling chamber-side errors were
renamed under the CONCERTO/CHAMBER split.
"""

from __future__ import annotations


# N818: plan/10 §2 normatively pins the name ``ConcertoSafetyInfeasible``
# (the rename map explicitly notes "stays — safety is method"). The
# Error-suffix convention is overridden by the plan; the noqa is the
# documented escape hatch.
class ConcertoSafetyInfeasible(RuntimeError):  # noqa: N818
    """Raised when the safety QP is infeasible (ADR-004 §Decision; ADR-006 §Risks).

    The conformal CBF-QP can become infeasible when partner-induced
    constraints conflict with within-robot OSCBF constraints under
    aggressive actuator limits — exactly the open problem named in
    Garg et al. 2024 §7.3 and tracked as ADR-006 R5. Callers MUST route
    to the braking fallback (ADR-004 risk-mitigation #1) on this error;
    the conformal QP is not a valid recovery path because Theorem 3
    (Huriot & Sibai 2025) only bounds the *average* loss.
    """
