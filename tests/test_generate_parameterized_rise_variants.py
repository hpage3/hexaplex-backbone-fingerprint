from __future__ import annotations

import numpy as np

from scripts.audit_parent_axial_layers import LayerModel
from scripts.generate_global_deformation_variants import PdbAtomLine
from scripts.generate_parameterized_rise_variants import (
    estimated_percent_rise_compression,
    parameterized_rise_z,
    rise_grid,
    transform_atom,
    variant_id,
)


def atom(z: float) -> PdbAtomLine:
    return PdbAtomLine("ATOM", 0, "ATOM", "CA", "GLU", "A", "1", 1.0, 2.0, z)


def test_rise_scale_grid_contains_exactly_9_values() -> None:
    assert [spec.rise_scale for spec in rise_grid()] == [0.96, 0.965, 0.97, 0.975, 0.98, 0.985, 0.99, 0.995, 1.0]


def test_variant_ids_are_stable_and_readable() -> None:
    assert variant_id(0.96) == "parameterized_rise_0p9600"
    assert variant_id(1.0) == "parameterized_rise_1p0000"


def test_layer_center_transform_preserves_local_z_offset() -> None:
    assert np.isclose(parameterized_rise_z(z=11.2, layer_center=11.0, global_center_z=10.0, rise_scale=0.5), 10.7)


def test_transform_leaves_xy_unchanged() -> None:
    model = LayerModel([0.0, 10.0], "test", 0.2)
    out = transform_atom(atom(10.2), model, global_center_z=5.0, rise_scale=0.5)
    assert np.allclose(out, [1.0, 2.0, 7.7])


def test_estimated_percent_compression_calculation() -> None:
    assert estimated_percent_rise_compression(0.97) == 3.0000000000000027
