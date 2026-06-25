# SPDX-License-Identifier: Apache-2.0
# pyright: reportPrivateImportUsage=false, reportAttributeAccessIssue=false
"""Tier-1 (no-SAPIEN-scene) tests for the co-insert S2 round-geometry helpers (ADR-026 §Decision 1).

The S2 honest-close swapped the peg/socket cross-section square → canonical round
(cylinder peg + an N-gon convex-box bore; ADR-001 §Risks no-mesh) and attaches the
held socket as a fixed child link of the holder articulation via a generated URDF.
These pure module-level helpers in :mod:`chamber.envs.coinsert` build that geometry
and need no SAPIEN scene (they do string / NumPy work + read the panda URDF), so the
Tier-1 tier exercises them directly:

- :func:`_round_bore_boxes` — floor + ``_ROUND_BORE_FACETS`` tangential wall facets
  of a regular N-gon bore of inscribed radius ``peg_radius + clearance/2``.
- :func:`_quat_wxyz_to_rpy` — URDF roll-pitch-yaw from a wxyz quaternion (the socket
  fixed-joint origin).
- :func:`_augmented_socket_holder_urdf` — panda_v2 + the socket as a fixed child link.

ADR-026 §Decision 1; ADR-001 §Risks (wrapper-only, no-mesh convex decomposition).
"""

from __future__ import annotations

import math
import os

import numpy as np

from chamber.envs.coinsert import (
    _ROUND_BORE_FACETS,
    _SOCKET_LINK_NAME,
    COINSERT_PEG_DIAMETER_M,
    _augmented_socket_holder_urdf,
    _quat_wxyz_to_rpy,
    _round_bore_boxes,
    coinsert_socket_inner_half_width,
)


def test_round_bore_boxes_count_is_floor_plus_facets() -> None:
    boxes = _round_bore_boxes(1.0e-3)
    assert len(boxes) == _ROUND_BORE_FACETS + 1  # floor + N wall facets
    # Each entry is (half_size[3], centre[3], yaw_about_z).
    for half, centre, yaw in boxes:
        assert len(half) == 3
        assert len(centre) == 3
        assert isinstance(yaw, float)


def test_round_bore_facets_ring_at_inscribed_radius() -> None:
    clearance = 1.0e-3
    r_in = COINSERT_PEG_DIAMETER_M / 2.0 + clearance / 2.0
    boxes = _round_bore_boxes(clearance)
    facets = boxes[1:]  # drop the floor
    # The facet centres lie on a ring outside the inscribed bore radius, spread
    # around the full circle (distinct yaws), and the floor sits on the axis.
    floor_centre = boxes[0][1]
    assert floor_centre[0] == 0.0
    assert floor_centre[1] == 0.0
    radii = [math.hypot(c[0], c[1]) for _, c, _ in facets]
    assert all(r > r_in for r in radii)
    yaws = {round(y, 4) for _, _, y in facets}
    assert len(yaws) == _ROUND_BORE_FACETS  # all distinct


def test_round_bore_clearance_scales_inscribed_radius() -> None:
    # A tighter clearance → a smaller inscribed bore → facet ring closer to the axis.
    loose = [math.hypot(c[0], c[1]) for _, c, _ in _round_bore_boxes(1.0e-3)[1:]]
    tight = [math.hypot(c[0], c[1]) for _, c, _ in _round_bore_boxes(2.0e-4)[1:]]
    assert min(tight) < min(loose)


def test_quat_wxyz_to_rpy_flip_is_pi_roll() -> None:
    roll, pitch, yaw = _quat_wxyz_to_rpy((0.0, 1.0, 0.0, 0.0))  # 180° about x
    assert abs(abs(roll) - math.pi) < 1e-6
    assert abs(pitch) < 1e-6
    assert abs(yaw) < 1e-6


def test_quat_wxyz_to_rpy_identity_is_zero() -> None:
    rpy = _quat_wxyz_to_rpy((1.0, 0.0, 0.0, 0.0))
    assert all(abs(v) < 1e-9 for v in rpy)


def test_augmented_socket_holder_urdf_injects_fixed_socket_link() -> None:
    path = _augmented_socket_holder_urdf(1.0e-3)
    try:
        assert os.path.exists(path)
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        # The socket is a rigid child LINK of panda_hand via a fixed joint.
        assert f'<link name="{_SOCKET_LINK_NAME}">' in text
        assert 'type="fixed"' in text
        assert '<parent link="panda_hand"/>' in text
        # One box collision per round-bore box (floor + facets).
        assert text.count("<collision>") >= _ROUND_BORE_FACETS + 1
        # Mesh paths absolutised so the temp URDF resolves the panda meshes.
        assert "franka_description/" in text
    finally:
        os.unlink(path)


def test_socket_inner_half_width_matches_clearance() -> None:
    # The Tier-1 geometry contract the round bore is built from.
    r = COINSERT_PEG_DIAMETER_M / 2.0
    assert coinsert_socket_inner_half_width(1.0e-3) == r + 0.5e-3
    assert np.isclose(coinsert_socket_inner_half_width(2.0e-4), r + 1.0e-4)
