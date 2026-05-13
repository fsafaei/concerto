# SPDX-License-Identifier: Apache-2.0
"""Analytic sign-convention tests for EGO_ONLY CBF rows (ADR-004 §Risks #4; reviewer P0-3).

Three sign-sensitive transformations carry the safety-correctness
invariant in :func:`concerto.safety.cbf_qp._build_ego_only_row`:

1. The ego-acceleration coefficient ``-n_hat^T a_ego`` on the LHS of
   the pairwise constraint.
2. The partner-disturbance subtraction ``n_hat^T a_partner_pred`` on
   the RHS (the term that lets the ego "see" the partner's predicted
   motion without making the partner a decision variable).
3. The closing-velocity inner product ``n_hat dot Delta v`` inside the
   pairwise barrier value ``h_ij``.

A flipped sign in any of these would silently invert the pairwise
constraint and make the entire safety filter unsafe — the bug-class
the second technical review flagged as P0-3. The tests in this
module pin each sign analytically against the closed-form row
coefficients rather than against the QP-projected output, sidestepping
the solver tolerance noise that the existing integration tests live
with.

The :func:`test_mutant_inverse_sign_breaks_2_3_and_2_4_assertions`
mutant-detector verifies that the analytic assertions in
:func:`test_2_3_partner_accel_toward_ego_tightens_rhs` and
:func:`test_2_4_partner_accel_away_from_ego_loosens_rhs` actually
catch the bug they are designed to catch — a hand-rolled mutant of
the helper with the disturbance sign flipped exhibits the opposite
inequality, and the real-function assertions would fail under it.

A separate reduction test pins the spike_004A §Reduction promise:
under :class:`~concerto.safety.api.DoubleIntegratorControlModel` and
zero partner-predicted acceleration, the EGO_ONLY row coefficients
on the ego variable equal the corresponding slot of the CENTRALIZED
row that :func:`concerto.safety.cbf_qp._project_cartesian_row_to_action`
emits.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from concerto.safety.api import DoubleIntegratorControlModel
from concerto.safety.cbf_qp import (
    _NORM_FLOOR,
    _PENETRATION_RHS,
    AgentSnapshot,
    _build_cartesian_pair_row,
    _build_ego_only_row,
    _project_cartesian_row_to_action,
)

_FloatArr = npt.NDArray[np.float64]

# Helper-test fixtures: a single (alpha_pair, gamma, lambda_ij,
# radius) shared across scenarios so the closed-form arithmetic
# below is reproducible without per-test boilerplate.
_ALPHA_PAIR: float = 10.0
_GAMMA: float = 2.0
_LAMBDA_IJ: float = 0.0
_RADIUS: float = 0.1
_SAFETY_DISTANCE: float = 2.0 * _RADIUS  # D_s = r_i + r_j in _build_cartesian_pair_row.


def _snap(position: tuple[float, float], velocity: tuple[float, float]) -> AgentSnapshot:
    return AgentSnapshot(
        position=np.asarray(position, dtype=np.float64),
        velocity=np.asarray(velocity, dtype=np.float64),
        radius=_RADIUS,
    )


def _zero_accel() -> _FloatArr:
    return np.zeros(2, dtype=np.float64)


def _row_rhs(
    ego: AgentSnapshot,
    partner: AgentSnapshot,
    partner_pred_accel: _FloatArr,
    *,
    lambda_ij: float = _LAMBDA_IJ,
) -> tuple[_FloatArr, float, float]:
    return _build_ego_only_row(
        ego_snap=ego,
        partner_snap=partner,
        partner_predicted_accel=partner_pred_accel,
        alpha_pair=_ALPHA_PAIR,
        gamma=_GAMMA,
        lambda_ij=lambda_ij,
    )


# --- Scenario 2.1 -- ego behind, partner ahead, closing distance ------------


def test_2_1_closing_row_binds_acceleration_toward_partner() -> None:
    """Ego behind, partner ahead, closing — row binds acceleration toward partner.

    With ego at the origin and partner at +x = 1, the helper's
    convention ``n_hat = (ego - partner) / |Dp|`` gives ``n_hat =
    (-1, 0)``. The Cartesian row over the ego acceleration variable
    is ``-n_hat = (+1, 0)``. Since the constraint reads
    ``row @ a_ego <= rhs``, a positive coefficient on ``a_ego[0]``
    upper-bounds the +x (toward-partner) acceleration. The sign of
    the coefficient is convention-relative; the load-bearing fact is
    that the constraint binds motion *toward* the partner ahead.
    """
    ego = _snap(position=(0.0, 0.0), velocity=(1.0, 0.0))
    partner = _snap(position=(1.0, 0.0), velocity=(0.0, 0.0))

    cart_row, rhs, h_ij = _row_rhs(ego, partner, _zero_accel())

    # n_hat points from partner to ego: (-1, 0). cart_row = -n_hat.
    np.testing.assert_allclose(cart_row, [1.0, 0.0])
    # The row coefficient projected onto the partner direction (+x)
    # is strictly positive: the constraint forbids unbounded forward
    # acceleration of the ego toward the partner. A flipped sign on
    # the row would reverse this and *allow* unbounded forward accel.
    partner_direction = np.array([1.0, 0.0], dtype=np.float64)
    assert float(cart_row @ partner_direction) > 0.0
    assert np.isfinite(rhs)
    assert np.isfinite(h_ij)


# --- Scenario 2.2 -- separating partners produce a more permissive rhs ------


def test_2_2_separating_rhs_more_permissive_than_closing() -> None:
    """Separating partners produce a larger (more permissive) rhs than closing.

    The Cartesian row geometry is identical between 2.1 and 2.2 —
    only the velocities differ. ``h_ij`` carries the closing-velocity
    sign via ``n_hat dot Delta v``: closing implies a negative inner
    product, smaller ``h_ij``, smaller rhs; separating implies a
    positive inner product, larger ``h_ij``, larger rhs. A flipped
    closing-velocity sign inside ``h_ij`` would invert this ordering.
    """
    ego_close = _snap(position=(0.0, 0.0), velocity=(1.0, 0.0))
    partner_close = _snap(position=(1.0, 0.0), velocity=(0.0, 0.0))
    _, rhs_close, h_close = _row_rhs(ego_close, partner_close, _zero_accel())

    ego_sep = _snap(position=(0.0, 0.0), velocity=(-1.0, 0.0))
    partner_sep = _snap(position=(1.0, 0.0), velocity=(1.0, 0.0))
    _, rhs_sep, h_sep = _row_rhs(ego_sep, partner_sep, _zero_accel())

    assert rhs_sep > rhs_close
    assert h_sep > h_close


# --- Scenario 2.3 -- partner predicted to accelerate toward ego -------------


def test_2_3_partner_accel_toward_ego_tightens_rhs() -> None:
    """A partner predicted to accelerate toward the ego has less ego headroom.

    Partner ahead of ego at +x; ``partner_predicted_accel = (-1, 0)``
    means the partner is forecast to accelerate in -x (toward the
    ego). The disturbance term ``n_hat dot a_partner_pred`` is +1
    (since ``n_hat = (-1, 0)``); the helper *subtracts* it from rhs,
    so the constraint tightens. A flipped sign on the subtraction
    would *loosen* the constraint instead — a silent safety failure
    of exactly the class reviewer P0-3 flagged.
    """
    ego = _snap(position=(0.0, 0.0), velocity=(1.0, 0.0))
    partner = _snap(position=(1.0, 0.0), velocity=(0.0, 0.0))

    _, rhs_baseline, _ = _row_rhs(ego, partner, _zero_accel())
    _, rhs_toward, _ = _row_rhs(ego, partner, np.array([-1.0, 0.0], dtype=np.float64))

    assert rhs_toward < rhs_baseline


# --- Scenario 2.4 -- partner predicted to accelerate away from ego ----------


def test_2_4_partner_accel_away_from_ego_loosens_rhs() -> None:
    """A partner predicted to accelerate away from the ego gives more headroom.

    ``partner_predicted_accel = (+1, 0)`` means the partner is
    forecast to accelerate in +x (away from the ego). The disturbance
    term is -1; the helper *subtracts* it from rhs, increasing rhs.
    A flipped sign on the subtraction would *tighten* instead of
    loosening — symmetric counterpart to the 2.3 sign-flip class.
    """
    ego = _snap(position=(0.0, 0.0), velocity=(1.0, 0.0))
    partner = _snap(position=(1.0, 0.0), velocity=(0.0, 0.0))

    _, rhs_baseline, _ = _row_rhs(ego, partner, _zero_accel())
    _, rhs_away, _ = _row_rhs(ego, partner, np.array([1.0, 0.0], dtype=np.float64))

    assert rhs_away > rhs_baseline


# --- Scenario 2.5 -- near-coincident centres (|Dp| < _NORM_FLOOR) -----------


def test_2_5_near_coincident_centres_emit_penetration_sentinel() -> None:
    """When ``|Dp| < _NORM_FLOOR`` the helper emits a deterministic row.

    The coincident-centre branch in ``_build_cartesian_pair_row``
    returns a stable unit normal so the QP still produces a
    deterministic row; the braking fallback (ADR-004 risk-mitigation
    #1) is the intended recovery for the physical pathology. The
    pairwise barrier saturates to ``-safety_distance`` and rhs reads
    the ``_PENETRATION_RHS`` sentinel.
    """
    ego = _snap(position=(0.0, 0.0), velocity=(0.0, 0.0))
    partner = _snap(position=(1e-10, 0.0), velocity=(0.0, 0.0))
    assert float(np.linalg.norm(ego.position - partner.position)) < _NORM_FLOOR

    cart_row, rhs, h_ij = _row_rhs(ego, partner, _zero_accel())

    assert np.all(np.isfinite(cart_row))
    assert np.isfinite(rhs)
    assert rhs == pytest.approx(_PENETRATION_RHS + _LAMBDA_IJ)
    assert h_ij <= 0.0
    assert h_ij == pytest.approx(-_SAFETY_DISTANCE)


# --- Scenario 2.6 -- boundary case ||Dp|| == D_s ----------------------------


def test_2_6_h_ij_monotone_across_safety_distance_boundary() -> None:
    """``h_ij`` is non-decreasing as ``||Dp||`` crosses ``D_s`` from inside to outside.

    At zero velocity, ``h_ij`` reduces to ``psi`` (the
    Wang-Ames-Egerstedt 2017 §III square-root term) when
    ``||Dp|| > D_s`` and to the closing inner product (zero here) in
    the penetration regime. Sweeping ``||Dp||`` through
    ``D_s - 1e-6``, ``D_s``, ``D_s + 1e-6`` must yield a finite,
    non-decreasing sequence; a flipped sign anywhere along the
    ``h_ij`` assembly would manifest as a non-monotone crossing.
    """
    eps = 1e-6
    h_vals: list[float] = []
    for d in (_SAFETY_DISTANCE - eps, _SAFETY_DISTANCE, _SAFETY_DISTANCE + eps):
        ego = _snap(position=(0.0, 0.0), velocity=(0.0, 0.0))
        partner = _snap(position=(d, 0.0), velocity=(0.0, 0.0))
        _, _, h_ij = _row_rhs(ego, partner, _zero_accel())
        assert np.isfinite(h_ij)
        h_vals.append(h_ij)

    assert h_vals[0] <= h_vals[1] <= h_vals[2], (
        f"h_ij must be non-decreasing across D_s; got {h_vals!r}"
    )


# --- Mutant detector --------------------------------------------------------


def _mutant_build_ego_only_row(
    ego_snap: AgentSnapshot,
    partner_snap: AgentSnapshot,
    partner_predicted_accel: _FloatArr,
    *,
    alpha_pair: float,
    gamma: float,
    lambda_ij: float,
) -> tuple[_FloatArr, float, float]:
    """Hand-rolled mutant of ``_build_ego_only_row`` with the disturbance sign flipped.

    Use **only** to verify the analytic-test assertions catch sign
    flips. Adds the partner-disturbance term to rhs instead of
    subtracting it — the exact bug-class reviewer P0-3 flagged. Kept
    deliberately separate from the production helper so the mutant
    can never be invoked from anywhere other than this test module.
    """
    pair_row = _build_cartesian_pair_row(
        snap_i=ego_snap,
        snap_j=partner_snap,
        alpha_pair=alpha_pair,
        gamma=gamma,
        lambda_ij=lambda_ij,
    )
    partner_disturbance = float(
        pair_row.n_hat @ partner_predicted_accel.astype(np.float64, copy=False)
    )
    row: _FloatArr = (-pair_row.n_hat).astype(np.float64, copy=True)
    # MUTANT: the production helper computes ``pair_row.rhs -
    # partner_disturbance``; this mutant flips that sign.
    rhs = pair_row.rhs + partner_disturbance
    return row, rhs, pair_row.h_ij


def test_mutant_inverse_sign_breaks_2_3_and_2_4_assertions() -> None:
    """The mutant with a flipped disturbance sign violates the 2.3 / 2.4 assertions.

    Confirms that the analytic assertions in
    :func:`test_2_3_partner_accel_toward_ego_tightens_rhs` and
    :func:`test_2_4_partner_accel_away_from_ego_loosens_rhs`
    actually catch the reviewer P0-3 sign-flip bug class. Under the
    real helper: ``rhs(toward) < rhs(baseline)`` and
    ``rhs(away) > rhs(baseline)``. Under the mutant the inequalities
    reverse — these mutant-assertions are the *negation* of the
    real-function assertions, demonstrating they would fail under
    the flipped-sign bug.
    """
    ego = _snap(position=(0.0, 0.0), velocity=(1.0, 0.0))
    partner = _snap(position=(1.0, 0.0), velocity=(0.0, 0.0))

    def _mutant_rhs(accel: _FloatArr) -> float:
        _, rhs, _ = _mutant_build_ego_only_row(
            ego_snap=ego,
            partner_snap=partner,
            partner_predicted_accel=accel,
            alpha_pair=_ALPHA_PAIR,
            gamma=_GAMMA,
            lambda_ij=_LAMBDA_IJ,
        )
        return rhs

    rhs_baseline = _mutant_rhs(_zero_accel())
    rhs_toward = _mutant_rhs(np.array([-1.0, 0.0], dtype=np.float64))
    rhs_away = _mutant_rhs(np.array([1.0, 0.0], dtype=np.float64))

    # The real helper would give rhs_toward < rhs_baseline; the
    # mutant flips the sign so rhs_toward > rhs_baseline. Likewise
    # rhs_away < rhs_baseline under the mutant.
    assert rhs_toward > rhs_baseline
    assert rhs_away < rhs_baseline


# --- Centralized reduction (spike_004A §Reduction) --------------------------


def test_ego_only_row_matches_centralized_ego_slot_for_double_integrator() -> None:
    """EGO_ONLY ego-variable row coincides with the CENTRALIZED ego slot.

    Pins the spike_004A §Reduction promise: with
    :class:`~concerto.safety.api.DoubleIntegratorControlModel`
    (Jacobian = identity) and zero partner-predicted acceleration,
    the EGO_ONLY Cartesian row equals the ego slot of the
    CENTRALIZED row built via the same Cartesian-to-action projection
    pipeline. Together with the disturbance-zero rhs equality, this
    pins the property that any post-refactor row-builder edit cannot
    drift the two modes apart in the homogeneous double-integrator
    setting.
    """
    ego = _snap(position=(0.0, 0.0), velocity=(1.0, 0.0))
    partner = _snap(position=(1.0, 0.0), velocity=(0.0, 0.0))

    ego_cart_row, ego_rhs, ego_h = _row_rhs(ego, partner, _zero_accel())

    pair_row = _build_cartesian_pair_row(
        snap_i=ego,
        snap_j=partner,
        alpha_pair=_ALPHA_PAIR,
        gamma=_GAMMA,
        lambda_ij=_LAMBDA_IJ,
    )
    ego_model = DoubleIntegratorControlModel(uid="ego", action_dim=2)
    partner_model = DoubleIntegratorControlModel(uid="partner", action_dim=2)
    centralized_row = _project_cartesian_row_to_action(
        pair_row,
        snap_i=ego,
        snap_j=partner,
        model_i=ego_model,
        model_j=partner_model,
        n_total=4,
        slot_i=0,
        slot_j=2,
    )

    np.testing.assert_allclose(ego_cart_row, centralized_row[:2])
    # Disturbance is zero in this configuration, so the EGO_ONLY rhs
    # equals the pair_row.rhs that the CENTRALIZED row would use
    # verbatim (see ``_filter_joint`` in cbf_qp.py).
    assert ego_rhs == pytest.approx(pair_row.rhs)
    assert ego_h == pytest.approx(pair_row.h_ij)
