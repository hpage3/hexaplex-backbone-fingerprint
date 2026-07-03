from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.generate_axial_only_extension_variants import (
    apply_axial_only_transform,
    axial_grid,
    output_path,
    variant_id,
)


def test_grid_contains_exactly_7_axial_values() -> None:
    grid = axial_grid()
    assert len(grid) == 7
    assert [spec.axial_scale_z for spec in grid] == [0.97, 0.975, 0.98, 0.985, 0.99, 0.995, 1.0]


def test_grid_includes_baseline_control() -> None:
    assert 1.0 in [spec.axial_scale_z for spec in axial_grid()]


def test_variant_ids_are_stable_and_readable() -> None:
    assert variant_id(0.97) == "axial_only_0p9700"
    assert variant_id(1.0) == "axial_only_1p0000"


def test_coordinate_transform_leaves_xy_and_scales_z_around_center() -> None:
    coord = np.array([3.0, 5.0, 7.0])
    center = np.array([1.0, 1.0, 1.0])
    out = apply_axial_only_transform(coord, axial_scale_z=0.5, center=center)
    assert np.allclose(out, [3.0, 5.0, 4.0])


def test_output_paths_are_under_axial_only_directory() -> None:
    spec = axial_grid()[0]
    path = output_path(Path("outputs/coordinates/axial_only_extension_variants"), spec)
    assert str(path).startswith("outputs\\coordinates\\axial_only_extension_variants") or str(path).startswith(
        "outputs/coordinates/axial_only_extension_variants"
    )
    assert path.name.endswith(".pdb")
