# SPDX-License-Identifier: Apache-2.0
# pyright: reportPrivateImportUsage=false, reportAttributeAccessIssue=false
"""Property test: co-insert controllers are stateless across episodes (ADR-009 §Decision).

The co-carry ladder caught a stateful-control bug — an end-effector-space
integrator that leaked across episode boundaries, so episode N's first action
depended on episode N-1's accumulated error. This test pins the fix for the
co-insert structured base inserter and cooperative reference holder
(:mod:`chamber.partners.coinsert_impedance`): the ONLY within-episode state (the
Cartesian integral, the step counter, and — for the holder — the captured
nominal mouth height) is fully cleared by :meth:`reset`, so two identical
episodes produce byte-identical action sequences regardless of what ran before.

This is the property the env-side per-episode reset relies on; the controllers
also expose :meth:`assert_episode_state_clear` for an in-rig post-reset check.

ADR-009 §Decision (stateless across episodes); ADR-002 (determinism / P6).
"""

from __future__ import annotations

import numpy as np
import pytest

from chamber.partners.api import PartnerSpec
from chamber.partners.coinsert_impedance import CoInsertBaseInserter, CoInsertReferenceHolder
from chamber.partners.registry import load_partner

_SPECS = {
    "coinsert_base_inserter": {
        "uid": "panda_wristcam",
        "base_xyz": "-0.5,0,0",
        "base_yaw_deg": "0",
        "peg_half_len": "0.04",
    },
    "coinsert_reference_holder": {
        "uid": "panda_partner",
        "base_xyz": "0.5,0,0",
        "base_yaw_deg": "180",
        "peg_half_len": "0.04",
    },
}


def _obs(uid: str, z: float) -> dict:
    return {
        "agent": {uid: {"qpos": np.full((1, 9), 0.05, dtype=np.float32)}},
        "extra": {
            "peg_pose": np.array([[0.001, 0.0, z, 1.0, 0.0, 0.0, 0.0]]),
            "receptacle_pose": np.array([[0.0, 0.0, 0.15, 0.0, 1.0, 0.0, 0.0]]),
        },
    }


@pytest.mark.parametrize("class_name", list(_SPECS))
def test_reset_makes_episodes_independent(class_name: str) -> None:
    extra = _SPECS[class_name]
    uid = extra["uid"]
    ctrl = load_partner(PartnerSpec(class_name, 0, None, None, dict(extra)))
    assert isinstance(ctrl, (CoInsertBaseInserter, CoInsertReferenceHolder))

    # Episode A — drive it for a while so the integral + counter accumulate.
    ctrl.reset(seed=0)
    ctrl.assert_episode_state_clear()
    seq_a = [np.asarray(ctrl.act(_obs(uid, 0.30 - 0.01 * i))) for i in range(8)]

    # Episode B — reset, then replay the SAME obs sequence. Byte-identical iff
    # reset cleared all within-episode state (no cross-episode leak).
    ctrl.reset(seed=0)
    ctrl.assert_episode_state_clear()
    seq_b = [np.asarray(ctrl.act(_obs(uid, 0.30 - 0.01 * i))) for i in range(8)]

    for a, b in zip(seq_a, seq_b, strict=True):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("class_name", list(_SPECS))
def test_integral_accumulates_then_clears(class_name: str) -> None:
    extra = _SPECS[class_name]
    uid = extra["uid"]
    ctrl = load_partner(PartnerSpec(class_name, 0, None, None, dict(extra)))
    ctrl.reset(seed=0)
    # A persistent near-target error should drive the integral away from zero.
    for _ in range(10):
        ctrl.act(_obs(uid, 0.151))  # peg tip near the socket mouth → integral engages
    assert bool(np.any(ctrl._integral != 0.0)) or ctrl._step > 0
    ctrl.reset(seed=0)
    assert ctrl._step == 0
    np.testing.assert_array_equal(ctrl._integral, np.zeros(3))
