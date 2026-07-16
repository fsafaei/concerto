# SPDX-License-Identifier: Apache-2.0
# pyright: reportPrivateImportUsage=false, reportAttributeAccessIssue=false
"""Tier-2 SAPIEN-gated contract tests for the co-hold-secure rig (ADR-029).

Real ManiSkill v3 env construction on a Vulkan/GPU host; the whole module is
skipped on CPU-only runners (the :mod:`tests.integration.test_coinsert_real`
skipif pattern). Pins the PR-A env contract (ADR-029 §Validation):

- every control flag (``matched`` / ``limp`` / ``none`` / ``fixture``)
  constructs and resets;
- ``chamber.tasks.make("co_hold_secure")`` returns the env — the
  ``NotImplementedError`` placeholder path is gone;
- reset/step determinism across fresh envs (P6 / ADR-002);
- telemetry emits finite values (the holder workpiece-wrench stress channel,
  seat depth, part-pose excursion) and the fixed-link part shows no phantom
  preload (the banked co-insert finding 1);
- the black-box holder interface discipline holds (the holder partner enters
  through :class:`chamber.partners.api.FrozenPartner`; the ego controller
  reads only the observation dict — no holder-internal state).

The measured precheck (P1-P4) is NOT a test — it is the pre-stated,
non-registered engineering precheck archived under
``spikes/results/coholdsecure/`` (ADR-029 §Decision; rule before result).
"""

from __future__ import annotations

import numpy as np
import pytest

from chamber.envs.co_hold_secure import make_co_hold_secure_env
from chamber.utils.device import sapien_gpu_available

pytestmark = [
    pytest.mark.slow,
    pytest.mark.gpu,
    pytest.mark.skipif(
        not sapien_gpu_available(),
        reason="Requires Vulkan/GPU (SAPIEN); skipped on CPU-only machines",
    ),
]

_UIDS = ("panda_wristcam", "panda_partner")


def _zero_actions() -> dict[str, np.ndarray]:
    return {uid: np.zeros(8, dtype=np.float32) for uid in _UIDS}


@pytest.mark.parametrize("control", ["matched", "limp", "none", "fixture"])
def test_each_control_flag_constructs_and_steps(control: str) -> None:
    env = make_co_hold_secure_env(control=control, render_backend="none")
    try:
        obs, _ = env.reset(seed=0)
        assert "peg_pose" in obs["extra"]
        assert "receptacle_pose" in obs["extra"]
        obs, _reward, _term, _trunc, info = env.step(_zero_actions())
        for key in (
            "seated_depth_m",
            "axis_align_deg",
            "pose_excursion_m",
            "pose_tilt_deg",
            "peak_secure_force_n",
            "peak_couple_wrench_n",
        ):
            assert np.all(np.isfinite(np.asarray(info[key])))
        assert env.control_id == control
    finally:
        env.close()


def test_registry_make_returns_the_env() -> None:
    # The spec-only NotImplementedError path is gone (ADR-029 §Validation).
    import chamber.tasks

    env = chamber.tasks.make("co_hold_secure", render_backend="none")
    try:
        assert env.control_id == "matched"
        obs, _ = env.reset(seed=0)
        assert "receptacle_pose" in obs["extra"]
    finally:
        env.close()


def test_reset_step_determinism_across_fresh_envs() -> None:
    """Two fresh envs, same seed, same actions ⇒ byte-identical states (P6)."""

    def rollout() -> tuple[np.ndarray, np.ndarray]:
        env = make_co_hold_secure_env(control="matched", render_backend="none", root_seed=0)
        try:
            obs, _ = env.reset(seed=0)
            acts = {uid: np.full(8, 0.1, dtype=np.float32) for uid in _UIDS}
            for _ in range(30):
                obs, _r, _te, _tr, _info = env.step(acts)
            return (
                np.asarray(obs["extra"]["peg_pose"]).copy(),
                np.asarray(obs["extra"]["receptacle_pose"]).copy(),
            )
        finally:
            env.close()

    a_plug, a_part = rollout()
    b_plug, b_part = rollout()
    np.testing.assert_array_equal(a_plug, b_plug)
    np.testing.assert_array_equal(a_part, b_part)


def test_fixed_link_part_shows_no_phantom_preload() -> None:
    """The banked co-insert finding 1: fixed link ⇒ ~zero zero-action wrench."""
    env = make_co_hold_secure_env(control="matched", render_backend="none")
    try:
        env.reset(seed=0)
        for _ in range(30):
            env.step(_zero_actions())
        wrench = float(env.holder_workpiece_wrench()[0])
        # The create_drive artifact was 573-1306 N; the fixed link is ~0.
        assert wrench < 5.0
    finally:
        env.close()


def test_stress_channel_reports_absent_without_a_holder() -> None:
    env = make_co_hold_secure_env(control="fixture", render_backend="none")
    try:
        env.reset(seed=0)
        env.step(_zero_actions())
        assert float(env.holder_workpiece_wrench()[0]) == pytest.approx(0.0)
    finally:
        env.close()


# The black-box holder-shield contract test needs no SAPIEN scene and lives
# in tests/unit/test_co_hold_secure_tier1.py (CPU CI coverage; ADR-009/I3).
