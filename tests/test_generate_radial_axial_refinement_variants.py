from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.generate_radial_axial_refinement_variants import (
    apply_radial_axial_transform,
    output_path,
    refinement_grid,
    variant_id,
)


def test_grid_contains_25_radial_axial_combinations() -> None:
    grid = refinement_grid()
    assert len(grid) == 25
    pairs = {(spec.radial_scale_xy, spec.axial_scale_z) for spec in grid}
    assert len(pairs) == 25


def test_grid_includes_baseline_control() -> None:
    assert (1.0, 1.0) in {(spec.radial_scale_xy, spec.axial_scale_z) for spec in refinement_grid()}


def test_variant_ids_are_stable_and_readable() -> None:
    assert variant_id(1.0025, 0.99) == "radial_1p0025__axial_0p9900"


def test_coordinate_transform_applies_radial_and_axial_scaling() -> None:
    coord = np.array([3.0, 5.0, 7.0])
    center = np.array([1.0, 1.0, 1.0])
    out = apply_radial_axial_transform(coord, radial_scale_xy=2.0, axial_scale_z=0.5, center=center)
    assert np.allclose(out, [5.0, 9.0, 4.0])


def test_output_paths_are_under_refinement_directory() -> None:
    spec = refinement_grid()[0]
    path = output_path(Path("outputs/coordinates/radial_axial_refinement_variants"), spec)
    assert str(path).startswith("outputs\\coordinates\\radial_axial_refinement_variants") or str(path).startswith(
        "outputs/coordinates/radial_axial_refinement_variants"
    )
    assert path.name.endswith(".pdb")
