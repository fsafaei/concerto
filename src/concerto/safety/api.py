# SPDX-License-Identifier: Apache-2.0
"""Public safety-stack API: Protocol + Bounds + SafetyState (ADR-004 + ADR-006 + ADR-014).

ADR-004 §Decision pins the three-layer architecture (exp CBF-QP backbone +
conformal slack overlay + OSCBF inner filter) plus a hard braking fallback
(ADR-004 risk-mitigation #1, Wang-Ames-Egerstedt 2017 eq. 17). This module
declares the contracts every layer's caller depends on:

- :class:`Bounds` — per-task numeric envelope (ADR-006 §Decision Option C).
- :class:`SafetyState` — mutable conformal-CBF state (Huriot & Sibai 2025
  §IV; ADR-004 risk-mitigation #2 governs the partner-swap warmup).
- :class:`FilterInfo` — telemetry payload returned alongside the safe
  action; consumed by the three-table renderer (ADR-014 §Decision).
- :class:`SafetyFilter` — the Protocol the integrated stack implements
  (``cbf_qp.py`` + ``conformal.py`` + ``oscbf.py`` + ``braking.py``).

The module imports nothing from ``chamber.*`` — the dependency-direction
rule (plan/10 §2) keeps the method side stand-alone and testable against a
stub env. The partner-swap contract reads ``obs["meta"]["partner_id"]``
(M4 will produce the hash; M3 ships the consumer side, gated to
single-partner mode when the field is ``None``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypedDict, runtime_checkable

import numpy as np
import numpy.typing as npt

#: Per-agent action vector alias. Public so Protocol signatures stay readable.
FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Bounds:
    r"""Per-task numeric bounds for the safety stack (ADR-006 §Decision Option C).

    The ``Bounds`` envelope is the explicit-numeric half of ADR-006's hybrid
    decision: bounds gate QP feasibility and reportability while the
    conformal slack (:class:`SafetyState`) governs online adaptation. The
    enumerated values track ADR-006 §Consequences — the comm-latency bound
    is anchored to the URLLC-3GPP-R17 sweep table from M2; ``action_norm``
    and ``action_rate`` are task-specific (Phase-1 fills them; Phase-0
    ships sane defaults). The ``force_limit`` field is a Phase-0 stub for
    ADR-007 Open Question #4 (per-vendor decomposition, ADR-004 Open
    Question #5) — a strategy-interfaced per-vendor handler slots in once
    Stage-3 SA resolves the open question.

    Attributes:
        action_norm: Maximum :math:`\lVert u \rVert_2` per agent, per task
            (Wang-Ames-Egerstedt 2017 §IV per-pair budget input).
        action_rate: Maximum :math:`\lVert u_k - u_{k-1} \rVert_2` per
            agent, per task. Bounds the actuator-rate slack the conformal
            CBF must remain feasible under (ADR-006 §Decision).
        comm_latency_ms: Default mean comm latency, in milliseconds. Set
            from the URLLC profile selected for the run (ADR-006
            §Consequences; ``chamber.comm.URLLC_3GPP_R17``).
        force_limit: Global force threshold for the per-pair contact-force
            constraint (ADR-004 Open Question #5; per-vendor handler slots
            in once Stage-3 SA resolves the decomposition).
    """

    action_norm: float
    action_rate: float
    comm_latency_ms: float
    force_limit: float


#: Default conformal target loss in the conservative manipulation regime
#: (ADR-004 §Decision; Huriot & Sibai 2025 §VI).
DEFAULT_EPSILON: float = -0.05

#: Default conformal learning rate eta for ``lambda_{k+1} = lambda_k + eta * (eps - l_k)``
#: (ADR-004 §Decision; Huriot & Sibai 2025 Theorem 3).
DEFAULT_ETA: float = 0.01

#: Default partner-swap warmup window length in control steps
#: (ADR-004 risk-mitigation #2; ADR-006 risk #3).
DEFAULT_WARMUP_STEPS: int = 50


@dataclass
class SafetyState:
    """Mutable conformal CBF state (Huriot & Sibai 2025 §IV; ADR-004 §Decision).

    The conformal slack vector ``lambda_`` carries one entry per agent
    pair ``(i, j)``; it is updated each control step by
    ``concerto.safety.conformal.update_lambda`` according to Theorem 3's
    rule ``lambda_{k+1} = lambda_k + eta * (eps - l_k)``. ADR-004
    risk-mitigation #2 motivates the partner-swap warmup window: on
    partner identity change (detected via ``obs["meta"]["partner_id"]``)
    the state is reset to ``lambda_safe`` (the value guaranteeing QP
    feasibility under the worst-case bounded prediction error per ADR-006
    Assumption A2) and the next ``warmup_steps_remaining`` steps run with
    a tighter eps.

    Attributes:
        lambda_: Per-pair slack values, shape ``(N_pairs,)`` and dtype
            ``float64``. Initialised to ``lambda_safe``; mutated in place.
        epsilon: Target average loss. ADR-004 §Decision pins the default
            to ``-0.05`` (conservative manipulation regime).
        eta: Conformal learning rate (default ``0.01``).
        warmup_steps_remaining: Decremented each step while in warmup;
            zero outside the warmup window (ADR-004 risk-mitigation #2).
    """

    lambda_: FloatArray
    epsilon: float = DEFAULT_EPSILON
    eta: float = DEFAULT_ETA
    warmup_steps_remaining: int = 0


# Functional TypedDict form: plan/03 §3.1 fixes the wire-key as ``"lambda"``
# (a Python keyword), and the three-table renderer (ADR-014) reads
# ``info["lambda"]`` directly. The functional form lets us keep the spec key.
FilterInfo = TypedDict(
    "FilterInfo",
    {
        "lambda": FloatArray,
        "loss_k": FloatArray,
        "fallback_fired": bool,
        "qp_solve_ms": float,
    },
)
"""Telemetry payload returned by :meth:`SafetyFilter.filter` (ADR-014 §Decision).

The fields populate the three-table renderer's row data: ``lambda`` and
``loss_k`` feed Table 3 (conservativeness gap λ mean/var vs. oracle
gt/noLearn), ``fallback_fired`` feeds Table 2's "fallback fired" column,
``qp_solve_ms`` feeds the OSCBF 1 kHz target check (ADR-004 §"OSCBF target").
"""


@runtime_checkable
class SafetyFilter(Protocol):
    """Contract for the integrated safety-filter pipeline (ADR-004 §Decision).

    Implementations compose the three layers from ADR-004 — exp CBF-QP
    backbone (Wang-Ames-Egerstedt 2017), conformal slack overlay
    (Huriot & Sibai 2025), OSCBF inner filter (Morton & Pavone 2025) —
    plus the hard braking fallback (ADR-004 risk-mitigation #1) into a
    single propose-check-replan boundary that wraps any nominal
    controller without accessing its internals (the property the
    black-box-partner setting requires per ADR-006 §Decision).
    """

    def reset(self, *, seed: int | None = None) -> None:
        """Reset filter state at episode start (ADR-004 §Decision).

        Args:
            seed: Optional root seed for the filter's deterministic RNG
                substream (P6). Implementations route this through
                ``concerto.training.seeding.derive_substream`` so two
                CPU runs with the same seed produce byte-identical
                outputs.
        """
        ...

    def filter(
        self,
        proposed_action: dict[str, FloatArray],
        obs: dict[str, object],
        state: SafetyState,
        bounds: Bounds,
    ) -> tuple[dict[str, FloatArray], FilterInfo]:
        """Project the nominal action onto the safe set (ADR-004 §Decision).

        Args:
            proposed_action: Per-agent nominal control inputs, keyed by
                uid. The QP minimises ``||u - u_hat||^2`` with this as
                the reference (Wang-Ames-Egerstedt 2017 eq. 9).
            obs: Observation dict; the filter reads ``obs["comm"]`` (M2
                contract — see ``chamber.comm.api.CommPacket``) and
                ``obs["meta"]["partner_id"]`` (M4 contract; ``None`` ⇒
                single-partner mode, no hard fail; identity change ⇒
                reset ``lambda`` to ``lambda_safe`` and enter warmup
                per ADR-004 risk-mitigation #2).
            state: Mutable :class:`SafetyState`; updated in place by the
                conformal layer each control step.
            bounds: Per-task :class:`Bounds` envelope (ADR-006 §Decision).

        Returns:
            A pair ``(safe_action, info)`` where ``safe_action`` is the
            QP-projected per-agent action and ``info`` is a
            :class:`FilterInfo` telemetry payload feeding the ADR-014
            three-table renderer.

        Raises:
            ConcertoSafetyInfeasible: When the QP is infeasible even after
                slack relaxation (ADR-004 §Decision; ADR-006 §Risks). The
                caller MUST route to the braking fallback in this case.
        """
        ...


__all__ = [
    "DEFAULT_EPSILON",
    "DEFAULT_ETA",
    "DEFAULT_WARMUP_STEPS",
    "Bounds",
    "FilterInfo",
    "FloatArray",
    "SafetyFilter",
    "SafetyState",
]
