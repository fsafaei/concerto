# SPDX-License-Identifier: Apache-2.0
# pyright: reportPrivateImportUsage=false, reportAttributeAccessIssue=false
"""Tier-2 SAPIEN-gated tests for the co-insert rig (ADR-026 §Decision 1-4).

Real ManiSkill v3 env construction + a short matched rollout that needs a
Vulkan/GPU host. Mirrors :mod:`tests.integration.test_cocarry_real`'s skipif
pattern; the whole module is skipped on CPU-only runners.

The co-insert bet honest-closed at a HARD_STOP (the competent matched pair
cannot seat the peg — a geometric tilt-wedge; see
``spikes/results/coinsert/COINSERT_CLOSURE_2026-06-24.md``), so these tests do
NOT assert a seat. They assert what the closure established and what the
instrument must do: the matched / single-inserter / fidelity-probe rigs build
under SAPIEN; the socket is held as a fixed articulation link that BRACES (no
solver blow-up, no assembly drag); the success predicate + the friction-inclusive
wrench + the contact-force instruments compute finite values; the structured base
inserter + cooperative reference holder run their full phase machine; and the
two-robot-necessity positive control holds (single inserter ⇒ success ≈ 0).

ADR-026 §Decision 1-4; ADR-009 §Decision; ADR-005 §Decision; ADR-004 §Decision.
"""

from __future__ import annotations

import numpy as np
import pytest

from chamber.envs.coinsert import make_coinsert_env
from chamber.partners.api import PartnerSpec
from chamber.partners.registry import load_partner
from chamber.utils.device import sapien_gpu_available

pytestmark = [
    pytest.mark.slow,
    pytest.mark.gpu,
    pytest.mark.skipif(
        not sapien_gpu_available(),
        reason="Requires Vulkan/GPU (SAPIEN); skipped on CPU-only machines",
    ),
]

_MATCHED = "coinsert_matched_reference"
_SINGLE = "coinsert_single_inserter_positive_control"
_EGO = {
    "uid": "panda_wristcam",
    "base_xyz": "-0.5,0,0",
    "base_yaw_deg": "0",
    "peg_half_len": "0.04",
}
_HOLDER = {
    "uid": "panda_partner",
    "base_xyz": "0.5,0,0",
    "base_yaw_deg": "180",
    "peg_half_len": "0.04",
}


def _matched_controllers() -> tuple[object, object]:
    base = load_partner(PartnerSpec("coinsert_base_inserter", 0, None, None, dict(_EGO)))
    hold = load_partner(PartnerSpec("coinsert_reference_holder", 0, None, None, dict(_HOLDER)))
    return base, hold


def _rollout(env: object, base: object, hold: object, seed: int, n_steps: int) -> dict:
    """Drive the matched pair for n_steps; return the final info + a finite flag."""
    obs, _ = env.reset(seed=seed)
    base.reset(seed=seed)
    hold.reset(seed=seed)
    finite = True
    info: dict = {}
    for _ in range(n_steps):
        action = {
            "panda_wristcam": np.asarray(base.act(obs), dtype=np.float32),
            "panda_partner": np.asarray(hold.act(obs), dtype=np.float32),
        }
        obs, reward, terminated, _truncated, info = env.step(action)
        if not np.isfinite(np.asarray(reward).reshape(-1)[0]):
            finite = False
        if bool(np.asarray(terminated).reshape(-1)[0]):
            break
    return {"info": info, "finite": finite}


class TestConstruction:
    """The co-insert conditions construct a SAPIEN scene with the two-Panda tuple."""

    @pytest.mark.parametrize("condition_id", [_MATCHED, _SINGLE])
    def test_two_pandas_load(self, condition_id: str) -> None:
        env = make_coinsert_env(condition_id=condition_id, episode_length=250)
        try:
            assert set(env.agent.agents_dict.keys()) == {"panda_wristcam", "panda_partner"}
            assert env.single_inserter is (condition_id == _SINGLE)
        finally:
            env.close()

    def test_instruments_finite_at_reset(self) -> None:
        env = make_coinsert_env(condition_id=_MATCHED, episode_length=250)
        try:
            env.reset(seed=0)
            assert np.isfinite(float(env.workpiece_interaction_wrench()[0]))
            assert np.isfinite(env.peg_socket_contact_force())
            ev = env.evaluate()
            for key in (
                "seated",
                "both_static",
                "settled",
                "seated_depth_m",
                "peak_couple_wrench_n",
            ):
                assert key in ev
        finally:
            env.close()


class TestMatchedBracesAndRuns:
    """The fixed-link socket BRACES and the controllers run their phase machine (no blow-up)."""

    def test_matched_short_rollout_is_stable_and_braces(self) -> None:
        env = make_coinsert_env(condition_id=_MATCHED, episode_length=250, peg_clearance_m=1.0e-3)
        base, hold = _matched_controllers()
        try:
            sock_z0 = float(env.receptacle.pose.p[0, 2])
            out = _rollout(env, base, hold, seed=0, n_steps=60)
            info = out["info"]
            assert out["finite"], "reward went non-finite — solver blow-up"
            depth_mm = float(info["seated_depth_m"][0]) * 1000.0
            align = float(info["axis_align_deg"][0])
            couple = float(info["peak_couple_wrench_n"][0])
            # The competent pair inserts part-way (the ~30 mm tilt-wedge) and holds
            # alignment; it does NOT seat (HARD_STOP). Bounds catch a blow-up only.
            assert np.isfinite(depth_mm)
            assert depth_mm < 38.0  # not seated — the HARD_STOP close
            assert np.isfinite(align)
            assert align < 30.0
            assert np.isfinite(couple)
            assert couple < 600.0  # fixed-link braces (no create_drive over-constraint)
            sock_z1 = float(env.receptacle.pose.p[0, 2])
            # The socket is a fixed articulation link held by the holder — it does
            # not free-fall / drag to the table during a short braced rollout.
            assert abs(sock_z1 - sock_z0) < 0.30
        finally:
            env.close()


class TestSingleInserterPositiveControl:
    """Two-robot necessity: a lone inserter on a free, unheld socket ⇒ success ≈ 0."""

    def test_single_inserter_does_not_seat(self) -> None:
        env = make_coinsert_env(condition_id=_SINGLE, episode_length=250, peg_clearance_m=1.0e-3)
        base, hold = _matched_controllers()
        try:
            out = _rollout(env, base, hold, seed=0, n_steps=60)
            seated = bool(np.asarray(out["info"]["seated"]).reshape(-1)[0])
            assert not seated, (
                "a lone inserter seated an unheld socket — two-robot necessity broken"
            )
        finally:
            env.close()


class TestFidelityProbeRig:
    """The S1 fidelity-probe rig (kinematic peg + anchored socket) builds and reads contact."""

    def test_probe_builds_and_reads_contact(self) -> None:
        env = make_coinsert_env(
            condition_id=_MATCHED, episode_length=250, peg_clearance_m=1.0e-3, fidelity_probe=True
        )
        try:
            e = env.unwrapped
            env.reset(seed=0)
            socket = np.array([0.0, 0.0, 0.30])
            e.set_peg_pose(
                (socket + np.array([0.0, 0.0, -0.02])).astype(np.float32), [1.0, 0.0, 0.0, 0.0]
            )
            for _ in range(6):
                e.scene.step()
            assert np.isfinite(e.peg_socket_contact_force())
            assert e.socket_inner_half_width > 0.0
        finally:
            env.close()
