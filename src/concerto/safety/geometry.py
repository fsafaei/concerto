# SPDX-License-Identifier: Apache-2.0
"""Sphere decomposition for OSCBF whole-body collision (Morton & Pavone 2025 §IV.B).

ADR-004 §Decision pins the OSCBF inner filter as Morton & Pavone 2025; that
formulation approximates each robot link as a union of spheres so collision
constraints reduce to pairwise sphere-distance checks. This module provides
the geometry primitives — :class:`Sphere`, :class:`SphereCloud`, capsule
decomposition, pairwise distance and its gradient, and the min-distance
reduction — that the CBF-QP constraint generator (``oscbf.py``) needs to
build ``A_ij u <= b_ij`` rows.

The kinematic Jacobian (∂c/∂q for each sphere centre c with respect to
the joint configuration q) is supplied by the env-side robot model and
chained downstream in ``oscbf.py``. This module computes ∂d/∂c only.

All operations are deterministic numpy ops; no randomness is consumed,
so the determinism harness (P6) is not threaded through here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from concerto.safety.api import FloatArray

#: Numerical floor below which two sphere centres are treated as coincident
#: (ADR-004 §Decision; Morton & Pavone 2025 §IV.B). Below this the gradient
#: direction is ill-defined; ``pair_distance_with_gradient`` returns a
#: stable +x-axis fallback so downstream solvers never see NaN.
_COINCIDENT_TOL: float = 1e-12


@dataclass(frozen=True)
class Sphere:
    """A single sphere primitive (Morton & Pavone 2025 §IV.B; ADR-004 §Decision).

    Attributes:
        center: Cartesian centre, shape ``(3,)``, dtype ``float64``.
        radius: Strictly positive radius (validation is the caller's
            responsibility — :func:`decompose_capsule` enforces it).
    """

    center: FloatArray
    radius: float


@dataclass(frozen=True)
class SphereCloud:
    """An ordered set of spheres approximating a robot link (ADR-004 §Decision).

    Stored as parallel ``centers`` and ``radii`` arrays so distance and
    Jacobian computation can be vectorised over the whole cloud
    (Morton & Pavone 2025 §IV.B; ADR-004 §"OSCBF target" 1 kHz budget).

    Attributes:
        centers: Stacked centres, shape ``(N, 3)``, dtype ``float64``.
        radii: Per-sphere radii, shape ``(N,)``, dtype ``float64``.
    """

    centers: FloatArray
    radii: FloatArray

    def __len__(self) -> int:
        """Return the number of spheres in the cloud (ADR-004 §Decision)."""
        return int(self.centers.shape[0])


def pair_distance(a: Sphere, b: Sphere) -> float:
    """Signed sphere-pair distance, negative on overlap (ADR-004 §Decision).

    Returns ``||a.center - b.center||_2 - (a.radius + b.radius)``. The
    CBF barrier ``h(x) = pair_distance >= 0`` defines the safe set as
    its zero-superlevel set (Wang-Ames-Egerstedt 2017 §III construction;
    Morton & Pavone 2025 §IV.B for the spherical-link approximation).
    """
    delta = a.center - b.center
    return float(np.linalg.norm(delta)) - (a.radius + b.radius)


def pair_distance_with_gradient(a: Sphere, b: Sphere) -> tuple[float, FloatArray]:
    """Signed pair distance plus ``d/d(c_a - c_b)`` unit gradient (ADR-004 §Decision).

    Returns ``(d, grad)`` where ``grad`` is the unit vector pointing from
    ``b`` toward ``a`` — i.e. the direction in which moving ``c_a``
    *increases* the safety margin. When the two centres coincide
    (degenerate geometry) the gradient direction is mathematically
    ill-defined; this function returns the +x-axis as a stable fallback
    so the CBF-QP downstream never sees NaN (Morton & Pavone 2025 §IV.B
    sphere overlap handling).
    """
    delta = a.center - b.center
    norm = float(np.linalg.norm(delta))
    if norm < _COINCIDENT_TOL:
        grad: FloatArray = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        return -(a.radius + b.radius), grad
    grad_dir: FloatArray = (delta / norm).astype(np.float64, copy=False)
    return norm - (a.radius + b.radius), grad_dir


def min_pair_distance(cloud_a: SphereCloud, cloud_b: SphereCloud) -> tuple[float, int, int]:
    """Vectorised minimum sphere-pair distance over the Cartesian product (ADR-004 §Decision).

    Returns ``(distance, i, j)`` where ``i`` indexes ``cloud_a`` and
    ``j`` indexes ``cloud_b``. The CBF-QP constraint generator
    (``oscbf.py``) builds one row per minimum-distance pair, then calls
    :func:`pair_distance_with_gradient` on the achieving pair to get
    the constraint normal (Morton & Pavone 2025 §IV.B; ADR-004
    §"OSCBF target").

    Args:
        cloud_a: Sphere cloud for body A, ``N_a`` spheres.
        cloud_b: Sphere cloud for body B, ``N_b`` spheres.

    Returns:
        ``(d, i, j)`` — the minimum signed pair distance and the indices
        of the achieving spheres.

    Raises:
        ValueError: If either cloud is empty (no pairs to compare).
    """
    if len(cloud_a) == 0 or len(cloud_b) == 0:
        msg = (
            f"min_pair_distance requires non-empty clouds; got "
            f"|A|={len(cloud_a)}, |B|={len(cloud_b)}"
        )
        raise ValueError(msg)
    diffs = cloud_a.centers[:, None, :] - cloud_b.centers[None, :, :]
    dists = np.linalg.norm(diffs, axis=-1)  # (N_a, N_b)
    pad = cloud_a.radii[:, None] + cloud_b.radii[None, :]
    signed = dists - pad
    flat = int(np.argmin(signed))
    n_b = signed.shape[1]
    i, j = flat // n_b, flat % n_b
    return float(signed[i, j]), int(i), int(j)


def decompose_capsule(
    start: FloatArray,
    end: FloatArray,
    radius: float,
    n_spheres: int,
) -> SphereCloud:
    """Sphere-decompose a capsule along its axis (Morton & Pavone 2025 §IV.B; ADR-004 §Decision).

    Places ``n_spheres`` of equal radius at uniform parameter
    ``t ∈ [0, 1]`` along the segment ``start → end``. The union of
    spheres conservatively covers the original capsule (centre-line +
    radius) when ``n_spheres`` is large enough that consecutive sphere
    centres are within ``radius`` of each other — that's a caller-side
    check, not enforced here, because Morton & Pavone parameterise
    coverage error per task in §V.

    Args:
        start: Capsule start point, shape ``(3,)``, dtype ``float64``.
        end: Capsule end point, shape ``(3,)``, dtype ``float64``.
        radius: Strictly positive sphere radius.
        n_spheres: Number of spheres to place along the axis (``>= 1``).

    Returns:
        A :class:`SphereCloud` of ``n_spheres`` spheres.

    Raises:
        ValueError: If ``n_spheres < 1`` or ``radius <= 0``.
    """
    if n_spheres < 1:
        msg = f"n_spheres must be >= 1, got {n_spheres}"
        raise ValueError(msg)
    if radius <= 0:
        msg = f"radius must be positive, got {radius}"
        raise ValueError(msg)
    ts = np.linspace(0.0, 1.0, n_spheres, dtype=np.float64)
    centers: FloatArray = (1.0 - ts)[:, None] * start[None, :] + ts[:, None] * end[None, :]
    radii: FloatArray = np.full(n_spheres, radius, dtype=np.float64)
    return SphereCloud(centers=centers, radii=radii)


def transform_cloud(
    cloud: SphereCloud, rotation: FloatArray, translation: FloatArray
) -> SphereCloud:
    """Apply a rigid SE(3) transform to a SphereCloud (ADR-004 §Decision).

    Computes ``c' = R c + t`` for each centre; radii are unchanged
    because rigid transforms preserve metric. Used to place a
    link-frame sphere cloud (defined in URDF link coordinates) into the
    world frame after evaluating the kinematic chain (Morton & Pavone
    2025 §IV.B; ADR-004 §"OSCBF target").

    Args:
        cloud: The input sphere cloud (link-frame).
        rotation: Rotation matrix, shape ``(3, 3)``, dtype ``float64``.
        translation: Translation vector, shape ``(3,)``, dtype ``float64``.

    Returns:
        A new :class:`SphereCloud` with transformed centres.
    """
    new_centers: FloatArray = (cloud.centers @ rotation.T + translation).astype(
        np.float64, copy=False
    )
    return SphereCloud(centers=new_centers, radii=cloud.radii)


__all__ = [
    "Sphere",
    "SphereCloud",
    "decompose_capsule",
    "min_pair_distance",
    "pair_distance",
    "pair_distance_with_gradient",
    "transform_cloud",
]
