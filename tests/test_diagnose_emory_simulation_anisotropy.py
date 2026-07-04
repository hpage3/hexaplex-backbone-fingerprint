from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.diagnose_emory_simulation_anisotropy import (
    axis_vectors,
    build_report,
    classify_failure_mode,
    classify_simulated_orientation,
    control_rows,
    fingerprint_summary,
    ladder_levels,
    radius_sensitive,
    rotation_matrix_from_vectors,
    synthetic_arc_map,
    synthetic_uniform_ring_map,
)


def test_synthetic_arc_map_classifies_as_fiber_oriented() -> None:
    summary = fingerprint_summary("arc", "synthetic", synthetic_arc_map(size=128), grid_size=128)

    assert summary["orientation_classification"] == "fiber_oriented_arc_like"
    assert float(summary["anisotropy_ratio"]) > 2.0


def test_synthetic_uniform_ring_classifies_as_isotropic() -> None:
    summary = fingerprint_summary("ring", "synthetic", synthetic_uniform_ring_map(size=128), grid_size=128)

    assert summary["orientation_classification"] == "isotropic_ring_like"


def test_control_rows_detect_arc_and_ring_controls() -> None:
    controls, controls_ok = control_rows(grid_size=128, orientation_samples=8)

    assert controls_ok
    by_id = dict(zip(controls["model_id"], controls["orientation_classification"]))
    assert by_id["synthetic_arc_control"] == "fiber_oriented_arc_like"
    assert by_id["synthetic_uniform_ring_control"] == "isotropic_ring_like"


def test_axis_list_generation_includes_cartesian_and_principal_axes() -> None:
    coords = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]])
    axes = axis_vectors(coords)

    for name in [
        "x_axis",
        "y_axis",
        "z_axis",
        "principal_component_1",
        "principal_component_2",
        "principal_component_3",
        "helical_axis_if_existing_utility_available",
    ]:
        assert name in axes


def test_rotation_matrix_maps_source_to_target() -> None:
    rot = rotation_matrix_from_vectors(np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]))
    mapped = rot @ np.array([1.0, 0.0, 0.0])

    assert np.allclose(mapped, np.array([0.0, 0.0, 1.0]), atol=1e-7)


def test_ladder_levels_include_requested_averaging_modes() -> None:
    names = [name for name, _, _ in ladder_levels(orientation_samples=8)]

    assert "single_orientation" in names
    assert "rotate_about_fiber_axis_8" in names
    assert "tilt_plus_rotate_small" in names
    assert "full_or_near_powder_average_if_available" in names


def test_failure_mode_averaging_washes_out_arcs() -> None:
    axis = pd.DataFrame({"anisotropy_ratio": [3.4]})
    ladder = pd.DataFrame(
        {
            "averaging_level": ["single_orientation", "rotate_about_fiber_axis_16"],
            "anisotropy_ratio": [3.4, 1.2],
        }
    )

    assert classify_failure_mode(axis, ladder, pd.DataFrame(), pd.DataFrame(), controls_ok=True) == "averaging_washes_out_arcs"


def test_failure_mode_weak_even_single_orientation() -> None:
    axis = pd.DataFrame({"anisotropy_ratio": [1.1, 1.2, 1.3]})

    assert classify_failure_mode(axis, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), controls_ok=True) == "weak_even_single_orientation"


def test_failure_mode_axis_choice_sensitive() -> None:
    axis = pd.DataFrame({"anisotropy_ratio": [1.1, 3.2, 1.4]})

    assert classify_failure_mode(axis, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), controls_ok=True) == "axis_choice_sensitive"


def test_failure_mode_grid_resolution_sensitive() -> None:
    axis = pd.DataFrame({"anisotropy_ratio": [1.7, 1.8]})
    grid = pd.DataFrame({"anisotropy_ratio": [1.2, 2.4]})

    assert classify_failure_mode(axis, pd.DataFrame(), grid, pd.DataFrame(), controls_ok=True) == "grid_resolution_sensitive"


def test_failure_mode_radius_choice_sensitive() -> None:
    axis = pd.DataFrame({"anisotropy_ratio": [1.7, 1.8]})
    radius = pd.DataFrame({"anisotropy_ratio": [1.2, 2.4]})

    assert classify_failure_mode(axis, pd.DataFrame(), pd.DataFrame(), radius, controls_ok=True) == "radius_choice_sensitive"
    assert radius_sensitive(radius)


def test_failure_mode_simulator_control_failure_and_insufficient_data() -> None:
    assert classify_failure_mode(pd.DataFrame({"anisotropy_ratio": [3.4]}), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), controls_ok=False) == "simulator_control_failure"
    assert classify_failure_mode(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), controls_ok=True) == "insufficient_data"


def test_classification_thresholds() -> None:
    assert classify_simulated_orientation(1.0, 2) == "isotropic_ring_like"
    assert classify_simulated_orientation(1.3, 2) == "weak_orientation"
    assert classify_simulated_orientation(1.6, 2) == "mixed_orientation"
    assert classify_simulated_orientation(3.2, 2) == "fiber_oriented_arc_like"


def test_report_wording_contains_required_cautions_and_modes() -> None:
    inventory = pd.DataFrame(
        [
            {
                "model_id": "m",
                "status": "found",
                "coordinate_atom_count": 10,
                "axis_status": "defaulted",
                "path": "m.pdb",
            }
        ]
    )
    controls = pd.DataFrame(
        [
            {
                "model_id": "synthetic_arc_control",
                "simulation_mode": "synthetic",
                "anisotropy_ratio": 3.5,
                "orientation_classification": "fiber_oriented_arc_like",
            }
        ]
    )
    axis = pd.DataFrame(
        [
            {
                "model_id": "m",
                "axis_mode": "z_axis",
                "anisotropy_ratio": 1.2,
                "orientation_classification": "weak_orientation",
                "preferred_angle_deg": 10.0,
            }
        ]
    )
    ladder = pd.DataFrame(
        [
            {
                "model_id": "m",
                "averaging_level": "single_orientation",
                "anisotropy_ratio": 1.2,
            }
        ]
    )
    modes = pd.DataFrame(
        [
            {
                "model_id": "m",
                "anisotropy_failure_mode": "weak_even_single_orientation",
                "max_single_orientation_anisotropy": 1.2,
                "min_single_orientation_anisotropy": 1.1,
            }
        ]
    )

    text = build_report(inventory, controls, axis, ladder, pd.DataFrame(), pd.DataFrame(), modes, controls_ok=True)

    for phrase in [
        "diagnostic of the first-pass simulator",
        "not structural proof",
        "provenance-limited",
        "Candidate elimination is not justified",
        "averaging_washes_out_arcs",
        "weak_even_single_orientation",
        "axis_choice_sensitive",
    ]:
        assert phrase in text
