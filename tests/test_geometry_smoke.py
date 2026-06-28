import numpy as np

from hexaplex_backbone_fingerprint.geometry import fit_plane, normalize


def test_normalize_returns_unit_vector():
    unit = normalize(np.array([3.0, 4.0, 0.0]))
    assert np.allclose(unit, np.array([0.6, 0.8, 0.0]))
    assert np.isclose(np.linalg.norm(unit), 1.0)


def test_fit_plane_exact_planar_points_has_near_zero_rms():
    points = np.array(
        [
            [0.0, 0.0, 2.0],
            [1.0, 0.0, 2.0],
            [0.0, 1.0, 2.0],
            [1.0, 1.0, 2.0],
        ]
    )
    center, normal, rms = fit_plane(points)

    assert np.allclose(center, np.array([0.5, 0.5, 2.0]))
    assert np.isclose(abs(normal[2]), 1.0)
    assert rms < 1e-12
