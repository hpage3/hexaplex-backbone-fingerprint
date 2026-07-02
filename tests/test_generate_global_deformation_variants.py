from __future__ import annotations

import numpy as np

from scripts.generate_global_deformation_variants import (
    VariantSpec,
    apply_deformation,
    format_pdb_coord_line,
    variant_grid,
)


def test_radial_scaling_changes_xy_not_z() -> None:
    coord = np.array([2.0, 3.0, 4.0])
    center = np.array([1.0, 1.0, 1.0])
    out = apply_deformation(coord, VariantSpec("r", "radial_scale_xy", radial_scale_xy=2.0), center, 0.0, 10.0)
    assert np.allclose(out, [3.0, 5.0, 4.0])


def test_axial_scaling_changes_z_not_xy() -> None:
    coord = np.array([2.0, 3.0, 4.0])
    center = np.array([1.0, 1.0, 1.0])
    out = apply_deformation(coord, VariantSpec("a", "axial_scale_z", axial_scale_z=2.0), center, 0.0, 10.0)
    assert np.allclose(out, [2.0, 3.0, 7.0])


def test_zero_twist_leaves_coordinate_unchanged() -> None:
    coord = np.array([2.0, 3.0, 4.0])
    center = np.array([1.0, 1.0, 1.0])
    out = apply_deformation(coord, VariantSpec("t", "twist_about_z", twist_total_deg=0.0), center, 0.0, 10.0)
    assert np.allclose(out, coord)


def test_nonzero_twist_preserves_radial_distance_from_center() -> None:
    coord = np.array([3.0, 1.0, 10.0])
    center = np.array([1.0, 1.0, 0.0])
    out = apply_deformation(coord, VariantSpec("t", "twist_about_z", twist_total_deg=10.0), center, 0.0, 20.0)
    assert np.isclose(np.linalg.norm(out[:2] - center[:2]), np.linalg.norm(coord[:2] - center[:2]))
    assert out[2] == coord[2]


def test_anisotropic_scaling_applies_distinct_xy_factors() -> None:
    coord = np.array([3.0, 5.0, 7.0])
    center = np.array([1.0, 1.0, 1.0])
    out = apply_deformation(
        coord,
        VariantSpec("x", "anisotropic_xy", x_scale=2.0, y_scale=0.5),
        center,
        0.0,
        10.0,
    )
    assert np.allclose(out, [5.0, 3.0, 7.0])


def test_variant_grid_contains_expected_one_at_a_time_variants() -> None:
    ids = [spec.variant_id for spec in variant_grid()]
    assert ids == [
        "radial_m1",
        "radial_0",
        "radial_p1",
        "axial_m1",
        "axial_0",
        "axial_p1",
        "twist_m05",
        "twist_0",
        "twist_p05",
        "anis_xy_p",
        "anis_xy_0",
        "anis_xy_m",
    ]


def test_pdb_coordinate_formatting_preserves_coordinate_columns() -> None:
    line = "ATOM      1  CA  GLU A   1       1.000   2.000   3.000  1.00  0.00           C"
    out = format_pdb_coord_line(line, np.array([12.345, -6.789, 0.12]))
    assert float(out[30:38]) == 12.345
    assert float(out[38:46]) == -6.789
    assert float(out[46:54]) == 0.12
    assert out[:30] == line[:30]
