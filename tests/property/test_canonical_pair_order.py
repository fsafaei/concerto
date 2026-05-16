# SPDX-License-Identifier: Apache-2.0
"""Property test: lambda/pair alignment survives dict reconstruction (review P1).

External-review finding (2026-05-16, reviewer's "P0-3-listed-as-P1"):
`state.lambda_: numpy.ndarray` is indexed by pair order, where pair
order is determined independently at three call sites by
`list(<dict>.keys())`. Python 3.7+ guarantees insertion order, but the
invariant is implicit: any caller that reconstructs the snapshot dict
in a different insertion order between `update_lambda_from_predictor`
and the next `ExpCBFQP.filter` call silently misaligns lambda values
with pair constraints. Silent pair-lambda misalignment is the worst
kind of safety bug — the wrong constraint gets the wrong slack.

This PR implements **Option (c)** from the Plan subagent's design
pass: canonical lexicographic UID sort at every entry point that
iterates pairs (the helper is `concerto.safety.api.canonical_pair_order`).
The runtime invariant is "pair iteration is sorted-by-uid"; the
structural fix that promotes `lambda_` to `dict[tuple[str, str],
float]` is deferred to Phase-1 (it requires bumping ADR-014's
`FilterInfo["lambda"]` wire contract and the renderer at the same
time; out of this sprint's diff budget). See the tracking issue
opened alongside this PR.

These tests pin the alignment under three reconstruction stressors:

- Reordered snapshot dict between consecutive `ExpCBFQP.filter` calls
  must produce identical outputs (the post-fix invariant).
- Reordered `proposed_action` dict (CENTRALIZED) must produce identical
  `safe_action[uid]` values.
- `compute_prediction_gap_for_pairs` returns identical per-pair losses
  whether the snapshot dict is in `(a, b, c)` or `(c, a, b)` order.

References:
- ADR-004 §Decisions (2026-05-16 amendment: canonical-pair-keying
  invariant).
- ``src/concerto/safety/api.py::canonical_pair_order`` — the helper.
- Plan subagent design pass — recommended Option (c) at ~120 LOC.
"""

from __future__ import annotations

import numpy as np
import pytest

from concerto.safety.api import (
    Bounds,
    DoubleIntegratorControlModel,
    SafetyState,
)
from concerto.safety.cbf_qp import AgentSnapshot, ExpCBFQP
from concerto.safety.conformal import (
    compute_prediction_gap_for_pairs,
    constant_velocity_predict,
    update_lambda,
)


def _snap(x: float, y: float, vx: float, vy: float, *, radius: float = 0.2) -> AgentSnapshot:
    return AgentSnapshot(
        position=np.array([x, y], dtype=np.float64),
        velocity=np.array([vx, vy], dtype=np.float64),
        radius=radius,
    )


def _bounds() -> Bounds:
    return Bounds(action_norm=5.0, action_rate=0.5, comm_latency_ms=1.0, force_limit=20.0)


def _models(uids: tuple[str, ...]) -> dict[str, DoubleIntegratorControlModel]:
    return {uid: DoubleIntegratorControlModel(uid=uid, action_dim=2) for uid in uids}


def test_lambda_pair_alignment_survives_dict_reconstruction() -> None:
    """Reordering the snapshot dict between calls must NOT shift lambda assignments.

    Construct a 3-agent scene twice: once with snaps in insertion order
    ``(a, b, c)`` and once in ``(c, a, b)``. The conformal pair-index
    that each consumer uses MUST be canonical (lexicographic by uid),
    so both forms produce identical ``FilterInfo["lambda"]``,
    ``FilterInfo["constraint_violation"]``, and per-uid safe actions.

    Pre-fix (insertion-order pair iteration) this test would have shown
    different ``constraint_violation`` vectors and different
    ``safe_action`` entries depending on dict order.
    """
    cbf = ExpCBFQP.centralized(control_models=_models(("a", "b", "c")), cbf_gamma=2.0)
    bounds = _bounds()
    state_a = SafetyState(lambda_=np.array([0.1, 0.2, 0.3], dtype=np.float64))
    state_b = SafetyState(lambda_=np.array([0.1, 0.2, 0.3], dtype=np.float64))

    # Geometry: pair (a, b) is in penetration (distance < safety) so its
    # CBF constraint is strongly active; pair (a, c) is mildly closing;
    # pair (b, c) is roughly stationary. The per-pair `constraint_violation`
    # vector therefore has materially different entries by pair index —
    # any insertion-order misalignment shows up as the wrong value at
    # the wrong index.
    snaps_natural = {
        "a": _snap(-0.18, 0.0, 0.2, 0.0),
        "b": _snap(0.18, 0.0, -0.2, 0.0),
        "c": _snap(0.0, 2.0, 0.0, -0.2),
    }
    proposed_natural = {
        "a": np.array([0.0, 0.0], dtype=np.float64),
        "b": np.array([0.0, 0.0], dtype=np.float64),
        "c": np.array([0.0, 0.0], dtype=np.float64),
    }
    # Reconstruct snapshots and proposed actions in a different insertion
    # order. Same payload, different iteration order — pure stressor.
    snaps_reordered = {key: snaps_natural[key] for key in ("c", "a", "b")}
    proposed_reordered = {key: proposed_natural[key] for key in ("c", "a", "b")}

    raw_natural, info_natural = cbf.filter(
        proposed_action=proposed_natural,
        obs={"agent_states": snaps_natural, "meta": {"partner_id": None}},
        state=state_a,
        bounds=bounds,
    )
    raw_reordered, info_reordered = cbf.filter(
        proposed_action=proposed_reordered,
        obs={"agent_states": snaps_reordered, "meta": {"partner_id": None}},
        state=state_b,
        bounds=bounds,
    )

    assert isinstance(raw_natural, dict)
    assert isinstance(raw_reordered, dict)
    # The per-uid safe action MUST be identical under reordering — pair
    # constraints are addressed by uid, not by dict insertion order.
    for uid in ("a", "b", "c"):
        np.testing.assert_allclose(raw_natural[uid], raw_reordered[uid], rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(
        info_natural["lambda"], info_reordered["lambda"], rtol=1e-9, atol=1e-9
    )
    np.testing.assert_allclose(
        info_natural["constraint_violation"],
        info_reordered["constraint_violation"],
        rtol=1e-9,
        atol=1e-9,
    )


def test_lambda_pair_alignment_three_agents() -> None:
    """Three agents yield three pairs; lambda values stay pinned to specific pairs.

    With canonical pair order (sorted by uid), the three pairs are
    ``(a, b)``, ``(a, c)``, ``(b, c)`` regardless of how the caller
    builds the input dict. Asserts the pair index for any (uid_i,
    uid_j) is stable across dict-reconstruction stress.
    """
    cbf = ExpCBFQP.centralized(control_models=_models(("a", "b", "c")), cbf_gamma=2.0)
    bounds = _bounds()

    # Penetrating geometry: pair (a, b) has h<0, the other two are clear.
    # Misalignment would put the large constraint_violation on the wrong
    # pair_idx after reordering.
    snaps = {
        "a": _snap(-0.18, 0.0, 0.2, 0.0),
        "b": _snap(0.18, 0.0, -0.2, 0.0),
        "c": _snap(0.0, 2.0, 0.0, -0.2),
    }
    proposed = {
        "a": np.zeros(2, dtype=np.float64),
        "b": np.zeros(2, dtype=np.float64),
        "c": np.zeros(2, dtype=np.float64),
    }
    state = SafetyState(lambda_=np.array([0.05, 0.10, 0.15], dtype=np.float64))

    _, info = cbf.filter(
        proposed_action=proposed,
        obs={"agent_states": snaps, "meta": {"partner_id": None}},
        state=state,
        bounds=bounds,
    )
    assert info["lambda"].shape == (3,)
    assert info["constraint_violation"].shape == (3,)
    # Repeat under reordered input. The two outputs MUST agree elementwise.
    state_reordered = SafetyState(lambda_=np.array([0.05, 0.10, 0.15], dtype=np.float64))
    snaps_reordered = {key: snaps[key] for key in ("b", "c", "a")}
    proposed_reordered = {key: proposed[key] for key in ("b", "c", "a")}
    _, info_reordered = cbf.filter(
        proposed_action=proposed_reordered,
        obs={"agent_states": snaps_reordered, "meta": {"partner_id": None}},
        state=state_reordered,
        bounds=bounds,
    )
    np.testing.assert_allclose(
        info["constraint_violation"],
        info_reordered["constraint_violation"],
        rtol=1e-9,
        atol=1e-9,
    )


def test_compute_prediction_gap_for_pairs_invariant_under_reordering() -> None:
    """`compute_prediction_gap_for_pairs` returns identical loss under reordered snaps."""
    snaps_natural = {
        "a": _snap(-2.0, 0.0, 1.0, 0.0),
        "b": _snap(2.0, 0.0, -1.0, 0.0),
        "c": _snap(0.0, 3.0, 0.0, -1.0),
    }
    snaps_predicted = {
        uid: constant_velocity_predict(snap, 0.05) for uid, snap in snaps_natural.items()
    }

    loss_natural = compute_prediction_gap_for_pairs(
        snaps_natural, snaps_predicted, alpha_pair=10.0, gamma=2.0
    )
    # Reorder both dicts — same payload, different insertion order.
    snaps_reordered = {key: snaps_natural[key] for key in ("c", "a", "b")}
    predicted_reordered = {key: snaps_predicted[key] for key in ("c", "a", "b")}
    loss_reordered = compute_prediction_gap_for_pairs(
        snaps_reordered, predicted_reordered, alpha_pair=10.0, gamma=2.0
    )
    np.testing.assert_allclose(loss_natural, loss_reordered, rtol=1e-12, atol=1e-12)


def test_update_lambda_loss_alignment_after_dict_reconstruction() -> None:
    """A reconstructed snap dict + canonical predict ⇒ same lambda update.

    End-to-end: build snaps in natural order, predict, compute loss,
    update lambda. Repeat with snaps in different order. The two final
    ``state.lambda_`` vectors MUST agree — pair-lambda alignment is
    independent of the caller's dict insertion order.
    """
    eta = 0.05
    epsilon = -0.01
    snaps_natural = {
        "a": _snap(-2.0, 0.0, 1.0, 0.0),
        "b": _snap(2.0, 0.0, -1.0, 0.0),
        "c": _snap(0.0, 3.0, 0.0, -1.0),
    }
    snaps_predicted = {
        uid: constant_velocity_predict(snap, 0.05) for uid, snap in snaps_natural.items()
    }

    # Run 1: natural order.
    state_natural = SafetyState(
        lambda_=np.array([0.1, 0.2, 0.3], dtype=np.float64), epsilon=epsilon, eta=eta
    )
    loss_natural = compute_prediction_gap_for_pairs(
        snaps_natural, snaps_predicted, alpha_pair=10.0, gamma=2.0
    )
    update_lambda(state_natural, loss_natural)

    # Run 2: reordered.
    state_reordered = SafetyState(
        lambda_=np.array([0.1, 0.2, 0.3], dtype=np.float64), epsilon=epsilon, eta=eta
    )
    snaps_reorder = {key: snaps_natural[key] for key in ("c", "a", "b")}
    predicted_reorder = {key: snaps_predicted[key] for key in ("c", "a", "b")}
    loss_reordered = compute_prediction_gap_for_pairs(
        snaps_reorder, predicted_reorder, alpha_pair=10.0, gamma=2.0
    )
    update_lambda(state_reordered, loss_reordered)

    np.testing.assert_allclose(
        state_natural.lambda_, state_reordered.lambda_, rtol=1e-12, atol=1e-12
    )


def test_ego_only_partner_set_invariant_under_dict_reordering() -> None:
    """EGO_ONLY mode: partner pair order must be canonical, not insertion-order."""
    cbf = ExpCBFQP.ego_only(control_models=_models(("ego", "p1", "p2")), cbf_gamma=2.0)
    bounds = _bounds()

    snaps_natural = {
        "ego": _snap(0.0, 0.0, 1.0, 0.0),
        "p1": _snap(2.0, 0.0, -1.0, 0.0),
        "p2": _snap(0.0, 2.0, 0.0, -1.0),
    }
    partner_pred_natural = {
        "p1": constant_velocity_predict(snaps_natural["p1"], 0.05),
        "p2": constant_velocity_predict(snaps_natural["p2"], 0.05),
    }
    state = SafetyState(lambda_=np.array([0.05, 0.10], dtype=np.float64))

    raw_natural, info_natural = cbf.filter(
        proposed_action=np.zeros(2, dtype=np.float64),
        obs={"agent_states": snaps_natural, "meta": {"partner_id": None}},
        state=state,
        bounds=bounds,
        ego_uid="ego",
        partner_predicted_states=partner_pred_natural,
        dt=0.05,
    )

    # Reorder both partner dicts (snapshots + predictions). Same payload.
    snaps_reordered = {key: snaps_natural[key] for key in ("p2", "ego", "p1")}
    partner_pred_reordered = {key: partner_pred_natural[key] for key in ("p2", "p1")}
    state_reordered = SafetyState(lambda_=np.array([0.05, 0.10], dtype=np.float64))

    raw_reordered, info_reordered = cbf.filter(
        proposed_action=np.zeros(2, dtype=np.float64),
        obs={"agent_states": snaps_reordered, "meta": {"partner_id": None}},
        state=state_reordered,
        bounds=bounds,
        ego_uid="ego",
        partner_predicted_states=partner_pred_reordered,
        dt=0.05,
    )

    assert isinstance(raw_natural, np.ndarray)
    assert isinstance(raw_reordered, np.ndarray)
    np.testing.assert_allclose(raw_natural, raw_reordered, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(
        info_natural["constraint_violation"],
        info_reordered["constraint_violation"],
        rtol=1e-9,
        atol=1e-9,
    )


def test_compute_prediction_gap_for_pairs_accepts_distinct_key_orders() -> None:
    """After the canonical-pair-keying fix, snaps_now and snaps_pred no longer
    need identical insertion order — both are sorted internally. The
    only invariant is that the key *sets* match."""
    snaps_now = {
        "a": _snap(-2.0, 0.0, 1.0, 0.0),
        "b": _snap(2.0, 0.0, -1.0, 0.0),
    }
    # snaps_predicted ships in reverse insertion order — pre-fix this
    # would raise "must share identical key order"; post-fix it works.
    snaps_pred = {
        "b": constant_velocity_predict(snaps_now["b"], 0.05),
        "a": constant_velocity_predict(snaps_now["a"], 0.05),
    }
    loss = compute_prediction_gap_for_pairs(snaps_now, snaps_pred, alpha_pair=10.0, gamma=2.0)
    assert loss.shape == (1,)


def test_compute_prediction_gap_for_pairs_rejects_mismatched_key_sets() -> None:
    """Key-set mismatch (a uid missing) still raises — only ordering is now tolerated."""
    snaps_now = {"a": _snap(0.0, 0.0, 0.0, 0.0), "b": _snap(1.0, 0.0, 0.0, 0.0)}
    snaps_pred = {"a": constant_velocity_predict(snaps_now["a"], 0.05)}  # missing "b"
    with pytest.raises(ValueError, match="key"):
        compute_prediction_gap_for_pairs(snaps_now, snaps_pred, alpha_pair=10.0, gamma=2.0)
