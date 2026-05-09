# SPDX-License-Identifier: Apache-2.0
"""Unit + property tests for ``concerto.safety.geometry`` (T3.3).

Covers ADR-004 §Decision (Morton & Pavone 2025 §IV.B sphere-decomposition
geometry primitives consumed by ``oscbf.py``).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from concerto.safety.geometry import (
    Sphere,
    SphereCloud,
    decompose_capsule,
    min_pair_distance,
    pair_distance,
    pair_distance_with_gradient,
    transform_cloud,
)


def _sphere(x: float, y: float, z: float, r: float) -> Sphere:
    return Sphere(center=np.array([x, y, z], dtype=np.float64), radius=r)


def test_pair_distance_is_symmetric_in_centres() -> None:
    a = _sphere(0.0, 0.0, 0.0, 0.5)
    b = _sphere(2.0, 0.0, 0.0, 0.3)
    assert pair_distance(a, b) == pytest.approx(2.0 - 0.8)
    assert pair_distance(b, a) == pytest.approx(pair_distance(a, b))


def test_pair_distance_is_negative_on_overlap() -> None:
    a = _sphere(0.0, 0.0, 0.0, 1.0)
    b = _sphere(0.5, 0.0, 0.0, 1.0)
    assert pair_distance(a, b) == pytest.approx(0.5 - 2.0)


def test_pair_distance_with_gradient_returns_unit_normal() -> None:
    a = _sphere(2.0, 0.0, 0.0, 0.1)
    b = _sphere(0.0, 0.0, 0.0, 0.1)
    d, grad = pair_distance_with_gradient(a, b)
    assert d == pytest.approx(2.0 - 0.2)
    assert np.linalg.norm(grad) == pytest.approx(1.0)
    np.testing.assert_allclose(grad, [1.0, 0.0, 0.0])


def test_pair_distance_with_gradient_handles_coincident_centres() -> None:
    a = _sphere(0.0, 0.0, 0.0, 1.0)
    b = _sphere(0.0, 0.0, 0.0, 1.0)
    d, grad = pair_distance_with_gradient(a, b)
    assert d == pytest.approx(-2.0)
    assert np.linalg.norm(grad) == pytest.approx(1.0)
    assert not np.any(np.isnan(grad))


def test_decompose_capsule_n2_places_spheres_at_endpoints() -> None:
    start = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    end = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    cloud = decompose_capsule(start, end, radius=0.05, n_spheres=2)
    assert len(cloud) == 2
    np.testing.assert_allclose(cloud.centers[0], start)
    np.testing.assert_allclose(cloud.centers[1], end)
    np.testing.assert_allclose(cloud.radii, [0.05, 0.05])


def test_decompose_capsule_uniform_spacing() -> None:
    start = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    end = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    cloud = decompose_capsule(start, end, radius=0.05, n_spheres=5)
    assert len(cloud) == 5
    expected = np.linspace(0.0, 1.0, 5)[:, None] * np.array([1.0, 0.0, 0.0])
    np.testing.assert_allclose(cloud.centers, expected)


def test_decompose_capsule_rejects_zero_n_spheres() -> None:
    with pytest.raises(ValueError, match=">= 1"):
        decompose_capsule(np.zeros(3), np.ones(3), radius=0.1, n_spheres=0)


def test_decompose_capsule_rejects_nonpositive_radius() -> None:
    with pytest.raises(ValueError, match="positive"):
        decompose_capsule(np.zeros(3), np.ones(3), radius=0.0, n_spheres=2)


def test_min_pair_distance_finds_argmin_pair() -> None:
    cloud_a = SphereCloud(
        centers=np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float64),
        radii=np.array([0.1, 0.1], dtype=np.float64),
    )
    cloud_b = SphereCloud(
        centers=np.array([[5.0, 0.0, 0.0], [11.0, 0.0, 0.0]], dtype=np.float64),
        radii=np.array([0.1, 0.1], dtype=np.float64),
    )
    d, i, j = min_pair_distance(cloud_a, cloud_b)
    # Closest pair: A[1] at x=10 vs B[1] at x=11 → centre-distance 1.0,
    # signed distance = 1.0 - 0.2 = 0.8.
    assert (i, j) == (1, 1)
    assert d == pytest.approx(0.8)


def test_min_pair_distance_rejects_empty_cloud() -> None:
    empty = SphereCloud(
        centers=np.zeros((0, 3), dtype=np.float64),
        radii=np.zeros(0, dtype=np.float64),
    )
    nonempty = SphereCloud(
        centers=np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
        radii=np.array([0.1], dtype=np.float64),
    )
    with pytest.raises(ValueError, match="non-empty"):
        min_pair_distance(empty, nonempty)
    with pytest.raises(ValueError, match="non-empty"):
        min_pair_distance(nonempty, empty)


def test_transform_cloud_rotates_and_translates() -> None:
    cloud = SphereCloud(
        centers=np.array([[1.0, 0.0, 0.0]], dtype=np.float64),
        radii=np.array([0.1], dtype=np.float64),
    )
    R = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )  # 90 deg about z
    t = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    rotated = transform_cloud(cloud, R, t)
    np.testing.assert_allclose(rotated.centers[0], [0.0, 1.0, 1.0])
    np.testing.assert_allclose(rotated.radii, cloud.radii)


def test_transform_cloud_preserves_pairwise_distances() -> None:
    """Rigid transforms preserve metric — sphere-pair distances must be invariant."""
    rng = np.random.default_rng(0)
    cloud = SphereCloud(
        centers=rng.standard_normal((4, 3)).astype(np.float64),
        radii=(np.abs(rng.standard_normal(4)) + 0.05).astype(np.float64),
    )
    # Rotation about z by 30 deg + translation.
    theta = np.pi / 6
    R = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    t = np.array([0.5, -0.3, 0.7], dtype=np.float64)
    transformed = transform_cloud(cloud, R, t)

    d_before, _, _ = min_pair_distance(cloud, cloud)
    d_after, _, _ = min_pair_distance(transformed, transformed)
    assert d_before == pytest.approx(d_after, abs=1e-12)


def _naive_min_pair_distance(a: SphereCloud, b: SphereCloud) -> tuple[float, int, int]:
    best = float("inf")
    bi, bj = -1, -1
    for ii in range(len(a)):
        for jj in range(len(b)):
            sa = Sphere(center=a.centers[ii], radius=float(a.radii[ii]))
            sb = Sphere(center=b.centers[jj], radius=float(b.radii[jj]))
            d = pair_distance(sa, sb)
            if d < best:
                best = d
                bi, bj = ii, jj
    return best, bi, bj


@settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(
    n_a=st.integers(min_value=1, max_value=5),
    n_b=st.integers(min_value=1, max_value=5),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_min_pair_distance_matches_naive_loop(n_a: int, n_b: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    centers_a: npt.NDArray[np.float64] = rng.standard_normal((n_a, 3)) * 2.0
    centers_b: npt.NDArray[np.float64] = rng.standard_normal((n_b, 3)) * 2.0
    radii_a: npt.NDArray[np.float64] = np.abs(rng.standard_normal(n_a)) + 0.01
    radii_b: npt.NDArray[np.float64] = np.abs(rng.standard_normal(n_b)) + 0.01
    cloud_a = SphereCloud(centers=centers_a, radii=radii_a)
    cloud_b = SphereCloud(centers=centers_b, radii=radii_b)

    d_vec, i, j = min_pair_distance(cloud_a, cloud_b)
    d_ref, bi, bj = _naive_min_pair_distance(cloud_a, cloud_b)
    assert d_vec == pytest.approx(d_ref)
    assert (i, j) == (bi, bj)


def test_sphere_cloud_len_matches_centers_shape() -> None:
    cloud = SphereCloud(
        centers=np.zeros((3, 3), dtype=np.float64),
        radii=np.zeros(3, dtype=np.float64),
    )
    assert len(cloud) == 3
