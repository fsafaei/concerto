# SPDX-License-Identifier: Apache-2.0
"""Tier-1 tests for the presenter-ablated handover variant (ADR-027 §Admission protocol A2).

The A2 intervention on ``handover_place@v1``: with no presentation
event the part never enters the ego's workspace, so the placement must
resolve honestly to failure on the lateral conjunct — demonstrated by
the kinematics, never asserted. The default (``presenter_ablated=False``)
must stay byte-identical (the established default-off override
precedent).
"""

from __future__ import annotations

import numpy as np

from chamber.agents.handover_ego_scripted import ScriptedHandoverEgo
from chamber.envs.handover_place import (
    HANDOVER_ABLATED_STAGING_OFFSET_M,
    make_handover_place_env,
)


def _roll(env: object, presenter_action: np.ndarray) -> dict[str, object]:
    """One 2-phase episode with the scripted ego; returns the terminal info."""
    ego = ScriptedHandoverEgo(
        translation_range_m=env.translation_range_m,  # type: ignore[attr-defined]
        wrist_correction_deg=env.wrist_correction_deg,  # type: ignore[attr-defined]
    )
    obs, _ = env.reset(seed=7)  # type: ignore[attr-defined]
    obs, _, _, _, _ = env.step(presenter_action)  # type: ignore[attr-defined]
    _, _, _, _, info = env.step(ego.act(obs))  # type: ignore[attr-defined]
    return info


def test_default_is_byte_identical() -> None:
    """presenter_ablated=False leaves the resolved episode unchanged."""
    presentation = np.asarray([2.0e-4, -1.0e-4, 3.0, 0.0])
    baseline = _roll(make_handover_place_env(root_seed=0), presentation)
    with_param = _roll(make_handover_place_env(root_seed=0, presenter_ablated=False), presentation)
    assert baseline == with_param


def test_ablated_part_never_enters_workspace() -> None:
    """No presentation → the part stays out of reach → lateral failure, success ≈ 0."""
    env = make_handover_place_env(root_seed=0, presenter_ablated=True)
    # The presenter's action is consumed but must carry no part — even a
    # perfectly matched presentation is ignored under the ablation.
    info = _roll(env, np.asarray([0.0, 0.0, 0.0, 0.0]))
    assert not info["success"]
    residual = float(info["residual_lateral_m"])  # type: ignore[arg-type]
    assert residual >= HANDOVER_ABLATED_STAGING_OFFSET_M - env.translation_range_m - 1e-9
    assert info["binding_conjunct"] in ("lateral", "lateral|force")


def test_ablated_ignores_presenter_values() -> None:
    """The ablated presentation is invariant to whatever the presenter seat emits."""
    env = make_handover_place_env(root_seed=0, presenter_ablated=True)
    info_zero = _roll(env, np.asarray([0.0, 0.0, 0.0, 0.0]))
    info_biased = _roll(env, np.asarray([5.0e-4, 5.0e-4, 45.0, 2.0]))
    assert info_zero == info_biased
