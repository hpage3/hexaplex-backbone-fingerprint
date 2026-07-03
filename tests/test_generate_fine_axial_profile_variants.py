from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.generate_axial_only_extension_variants import apply_axial_only_transform
from scripts.generate_fine_axial_profile_variants import fine_axial_grid, output_path, variant_id


def test_grid_contains_exactly_7_fine_axial_values() -> None:
    values = [spec.axial_scale_z for spec in fine_axial_grid()]
    assert len(values) == 7
    assert values == [0.97, 0.9725, 0.975, 0.9775, 0.98, 0.9825, 0.985]


def test_grid_includes_endpoints() -> None:
    values = [spec.axial_scale_z for spec in fine_axial_grid()]
    assert 0.97 in values
    assert 0.985 in values


def test_variant_ids_are_stable_and_readable() -> None:
    assert variant_id(0.9725) == "fine_axial_0p9725"
    assert variant_id(0.985) == "fine_axial_0p9850"


def test_coordinate_transform_leaves_xy_and_scales_z_around_center() -> None:
    coord = np.array([3.0, 5.0, 7.0])
    center = np.array([1.0, 1.0, 1.0])
    out = apply_axial_only_transform(coord, axial_scale_z=0.5, center=center)
    assert np.allclose(out, [3.0, 5.0, 4.0])


def test_output_paths_are_under_fine_axial_directory() -> None:
    spec = fine_axial_grid()[0]
    path = output_path(Path("outputs/coordinates/fine_axial_profile_variants"), spec)
    assert str(path).startswith("outputs\\coordinates\\fine_axial_profile_variants") or str(path).startswith(
        "outputs/coordinates/fine_axial_profile_variants"
    )
    assert path.name.endswith(".pdb")
