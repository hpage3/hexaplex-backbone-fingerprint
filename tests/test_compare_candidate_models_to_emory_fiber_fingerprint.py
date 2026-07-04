from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.compare_candidate_models_to_emory_fiber_fingerprint import (
    CandidateRecord,
    angular_lobe_score,
    anisotropy_score,
    build_report,
    classify_simulated_orientation,
    expected_candidate_records,
    fingerprint_intensity_map,
    inventory_dataframe,
    projected_fft_intensity,
    score_fingerprint_row,
    top_score_rows,
)


def synthetic_arc_map(size: int = 128, radius: float = 34.0, angle_deg: float = 40.0) -> np.ndarray:
    center = ((size - 1) / 2.0, (size - 1) / 2.0)
    yy, xx = np.indices((size, size))
    rr = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2)
    theta = (np.degrees(np.arctan2(yy - center[1], xx - center[0])) + 360.0) % 360.0
    ring = np.exp(-((rr - radius) ** 2) / 10.0)
    diff_a = np.abs(((theta - angle_deg + 180.0) % 360.0) - 180.0)
    diff_b = np.abs(((theta - (angle_deg + 180.0) + 180.0) % 360.0) - 180.0)
    arcs = np.exp(-(diff_a**2) / 90.0) + np.exp(-(diff_b**2) / 90.0)
    return ring * arcs


def synthetic_ring_map(size: int = 128, radius: float = 34.0) -> np.ndarray:
    center = ((size - 1) / 2.0, (size - 1) / 2.0)
    yy, xx = np.indices((size, size))
    rr = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2)
    return np.exp(-((rr - radius) ** 2) / 10.0)


def test_candidate_inventory_records_missing_and_found_paths() -> None:
    records = [
        CandidateRecord(
            model_id="found_model",
            path=Path("model.pdb"),
            inferred_family="family",
            inferred_scale="",
            inferred_twist="",
            inferred_rise="",
            inclusion_reason="known model",
            provenance_caveat="diagnostic",
            status="found",
        ),
        CandidateRecord(
            model_id="missing_model",
            path=None,
            inferred_family="family",
            inferred_scale="",
            inferred_twist="",
            inferred_rise="",
            inclusion_reason="expected model",
            provenance_caveat="missing",
            status="missing_candidate_coordinates",
        ),
    ]
    inventory = inventory_dataframe(records)

    assert list(inventory["status"]) == ["found", "missing_candidate_coordinates"]
    assert inventory.loc[0, "path"] == "model.pdb"
    assert inventory.loc[1, "path"] == ""


def test_expected_candidate_inventory_is_conservative() -> None:
    model_ids = {record.model_id for record in expected_candidate_records()}

    assert "omega_clean_scale_0p9825" in model_ids
    assert "omega_clean_scale_0p9700" in model_ids
    assert "guarded_full_chain_prototype" in model_ids
    assert "pnab_parallel_candidate" in model_ids


def test_synthetic_2d_map_radial_and_arc_fingerprint() -> None:
    fp, arcs = fingerprint_intensity_map("synthetic_arc", "single_oriented", synthetic_arc_map())

    assert not fp.empty
    assert not arcs.empty
    assert fp["anisotropy_ratio"].max() > 1.5
    assert set(fp["orientation_classification"]).issubset(
        {"mixed_orientation", "fiber_oriented_arc_like", "weak_orientation", "isotropic_ring_like"}
    )


def test_rotationally_aligned_angular_similarity_matches_rotated_arc_pair() -> None:
    base = angular_lobe_score(42.5, 222.5)
    rotated = angular_lobe_score(142.5, 322.5)

    assert base > 0.95
    assert rotated > 0.95


def test_arc_like_maps_score_higher_than_isotropic_rings() -> None:
    arc_fp, _ = fingerprint_intensity_map("arc", "single_oriented", synthetic_arc_map())
    ring_fp, _ = fingerprint_intensity_map("ring", "single_oriented", synthetic_ring_map())
    arc_row = arc_fp.sort_values("anisotropy_ratio", ascending=False).iloc[0]
    ring_row = ring_fp.sort_values("anisotropy_ratio", ascending=False).iloc[0]

    arc_score = score_fingerprint_row(arc_row, grid_size=128, provenance_caveat="")["emory_orientation_similarity_score"]
    ring_score = score_fingerprint_row(ring_row, grid_size=128, provenance_caveat="")["emory_orientation_similarity_score"]

    assert arc_score > ring_score


def test_anisotropy_scoring_prefers_target_range() -> None:
    assert anisotropy_score(3.5) > anisotropy_score(1.1)
    assert anisotropy_score(3.5) > anisotropy_score(12.0)


def test_classification_logic_includes_isotropic_and_fiber_classes() -> None:
    assert classify_simulated_orientation(float("nan"), 0) == "insufficient_quality"
    assert classify_simulated_orientation(1.02, 2) == "isotropic_ring_like"
    assert classify_simulated_orientation(1.25, 2) == "weak_orientation"
    assert classify_simulated_orientation(1.6, 2) == "mixed_orientation"
    assert classify_simulated_orientation(2.5, 2) == "fiber_oriented_arc_like"


def test_top_score_rows_returns_best_per_model_mode() -> None:
    fingerprints = pd.DataFrame(
        [
            {
                "model_id": "m",
                "simulation_mode": "single_oriented",
                "radius_px": 20,
                "preferred_angle_deg": 10,
                "opposite_angle_deg": 190,
                "anisotropy_ratio": 1.1,
                "orientation_classification": "weak_orientation",
                "horizontal_sector_mean": 1,
                "vertical_sector_mean": 1,
            },
            {
                "model_id": "m",
                "simulation_mode": "single_oriented",
                "radius_px": 35,
                "preferred_angle_deg": 40,
                "opposite_angle_deg": 220,
                "anisotropy_ratio": 3.5,
                "orientation_classification": "fiber_oriented_arc_like",
                "horizontal_sector_mean": 4,
                "vertical_sector_mean": 1,
            },
        ]
    )
    inventory = pd.DataFrame([{"model_id": "m", "provenance_caveat": ""}])

    scores = top_score_rows(fingerprints, inventory, grid_size=128)

    assert len(scores) == 1
    assert float(scores.iloc[0]["anisotropy_ratio"]) == 3.5


def test_projected_fft_intensity_returns_grid() -> None:
    coords = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 1]], dtype=float)
    intensity = projected_fft_intensity(coords, grid_size=32)

    assert intensity.shape == (32, 32)
    assert np.isfinite(intensity).all()


def test_report_wording_contains_required_cautions() -> None:
    inventory = pd.DataFrame(
        [
            {
                "model_id": "m",
                "status": "found",
                "inferred_family": "omega_clean",
                "inferred_scale": "0.9825",
                "path": "m.pdb",
                "provenance_caveat": "diagnostic",
                "inclusion_reason": "test",
            }
        ]
    )
    fingerprints = pd.DataFrame([{"orientation_classification": "fiber_oriented_arc_like"}])
    scores = pd.DataFrame(
        [
            {
                "model_id": "m",
                "simulation_mode": "single_oriented",
                "emory_orientation_similarity_score": 0.8,
                "orientation_classification": "fiber_oriented_arc_like",
                "anisotropy_ratio": 3.5,
                "preferred_angle_deg": 42.5,
                "opposite_angle_deg": 222.5,
            }
        ]
    )

    text = build_report(inventory, fingerprints, scores, mode="both")

    for phrase in [
        "first-pass exploratory",
        "provenance-limited",
        "not structural proof",
        "arbitrary detector rotation alignment",
        "No candidate should be eliminated solely from this analysis",
        "simulated 2D/fiber diffraction",
        "additional constraints beyond A/B/C/D",
    ]:
        assert phrase in text
