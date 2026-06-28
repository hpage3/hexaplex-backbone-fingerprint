"""Small geometry helpers used by peptide-plane fingerprint analysis."""

from __future__ import annotations

import numpy as np


def normalize(vector: np.ndarray) -> np.ndarray:
    """Return a unit vector in the same direction as *vector*."""
    arr = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(arr)
    if norm == 0:
        raise ValueError("Cannot normalize a zero-length vector.")
    return arr / norm


def fit_plane(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit a plane to 3D points with SVD.

    Returns ``(center, normal, rms)`` where ``rms`` is the root-mean-square
    distance of points from the best-fit plane.
    """
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("Plane fitting requires an array with shape (n, 3).")
    if arr.shape[0] < 3:
        raise ValueError("At least three points are required to fit a plane.")

    center = arr.mean(axis=0)
    centered = arr - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = normalize(vh[-1])
    signed_distances = centered @ normal
    rms = float(np.sqrt(np.mean(signed_distances**2)))
    return center, normal, rms


def angle_between_vectors(a: np.ndarray, b: np.ndarray) -> float:
    """Return the smaller angle between vectors in degrees."""
    a_unit = normalize(a)
    b_unit = normalize(b)
    cosine = float(np.clip(np.dot(a_unit, b_unit), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def unsigned_normal_angle(a: np.ndarray, b: np.ndarray) -> float:
    """Return the angle between unoriented normals in degrees, from 0 to 90."""
    a_unit = normalize(a)
    b_unit = normalize(b)
    cosine = abs(float(np.clip(np.dot(a_unit, b_unit), -1.0, 1.0)))
    return float(np.degrees(np.arccos(cosine)))


def distance(a: np.ndarray, b: np.ndarray) -> float:
    """Return Euclidean distance between two 3D coordinates."""
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def signed_distance_to_plane(point: np.ndarray, plane_center: np.ndarray, plane_normal: np.ndarray) -> float:
    """Return signed distance from a point to a plane."""
    return float((np.asarray(point, dtype=float) - np.asarray(plane_center, dtype=float)) @ normalize(plane_normal))


def dihedral_degrees(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Return signed dihedral angle for four points in degrees, in -180 to 180."""
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    p3 = np.asarray(p3, dtype=float)

    b0 = -(p1 - p0)
    b1 = normalize(p2 - p1)
    b2 = p3 - p2

    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1

    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return float(np.degrees(np.arctan2(y, x)))
