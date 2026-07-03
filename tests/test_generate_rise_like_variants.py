from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.generate_rise_like_variants import (
    apply_axial_only_transform,
    estimated_percent_compression,
    output_path,
    rise_like_grid,
    variant_id,
)


def test_grid_contains_requested_rise_like_values() -> None:
    values = [spec.axial_rise_scale for spec in rise_like_grid()]
    assert len(values) == 9
    assert values == [0.96, 0.965, 0.97, 0.975, 0.98, 0.985, 0.99, 0.995, 1.0]


def test_variant_ids_are_stable_and_readable() -> None:
    assert variant_id(0.96) == "rise_like_0p9600"
    assert variant_id(0.97) == "rise_like_0p9700"
    assert variant_id(1.0) == "rise_like_1p0000"


def test_percent_compression_relative_to_baseline() -> None:
    assert estimated_percent_compression(0.96) == 4.0000000000000036
    assert estimated_percent_compression(1.0) == 0.0


def test_coordinate_transform_leaves_xy_and_scales_z_around_center() -> None:
    coord = np.array([3.0, 5.0, 7.0])
    center = np.array([1.0, 1.0, 1.0])
    out = apply_axial_only_transform(coord, axial_scale_z=0.5, center=center)
    assert np.allclose(out, [3.0, 5.0, 4.0])


def test_output_paths_are_under_rise_like_directory() -> None:
    spec = rise_like_grid()[0]
    path = output_path(Path("outputs/coordinates/rise_like_variants"), spec)
    assert str(path).startswith("outputs\\coordinates\\rise_like_variants") or str(path).startswith(
        "outputs/coordinates/rise_like_variants"
    )
    assert path.name == "rise_like_0p9600.pdb"
