from __future__ import annotations

import numpy as np

from scripts.audit_parent_axial_layers import (
    assign_layer_index,
    infer_layers_from_ca_z,
    mean_layer_rise,
)


def test_z_gap_layer_inference_on_synthetic_ca_values() -> None:
    model = infer_layers_from_ca_z([0.0, 0.02, 1.0, 1.03, 2.1], gap_threshold_A=0.2)
    assert len(model.layer_centers) == 3
    assert np.allclose(model.layer_centers, [0.01, 1.015, 2.1])
    assert model.layer_model == "ca_z_gap_layers"


def test_mean_layer_rise_calculation() -> None:
    assert mean_layer_rise([0.0, 1.0, 2.5]) == 1.25


def test_stable_layer_ids_from_nearest_center() -> None:
    centers = [0.0, 1.0, 2.0]
    assert assign_layer_index(0.1, centers) == 0
    assert assign_layer_index(1.4, centers) == 1
    assert assign_layer_index(1.6, centers) == 2
