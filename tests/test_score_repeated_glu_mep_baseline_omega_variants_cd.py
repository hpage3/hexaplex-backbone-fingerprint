from __future__ import annotations

import pandas as pd

from scripts.score_repeated_glu_mep_baseline_omega_variants_cd import (
    add_baseline_relative_intensity,
    best_variant,
    build_report_text,
    classify_two_point_movement,
    filter_safe_baseline_parent_variants,
    skipped_unsafe_count,
)
from scripts.score_constrained_phi_psi_candidates_cd import combined_abs_error


def test_filter_safe_baseline_parent_variants_and_skipped_count() -> None:
    df = pd.DataFrame(
        {
            "variant_id": ["safe", "unsafe", "wrong_mode"],
            "omega_mode": ["baseline_parent", "baseline_parent", "fixed_180"],
            "safe_for_diffraction_scoring": ["True", "False", "True"],
        }
    )
    safe = filter_safe_baseline_parent_variants(df)
    assert safe["variant_id"].tolist() == ["safe"]
    assert skipped_unsafe_count(df) == 2


def test_combined_abs_error_shared_helper() -> None:
    assert combined_abs_error(0.2, -0.1) == 0.30000000000000004


def test_best_variant_selection() -> None:
    scores = pd.DataFrame({"variant_id": ["a", "b"], "combined_abs_error_A": [0.2, 0.1]})
    assert best_variant(scores)["variant_id"] == "b"


def test_two_point_peak_movement_classification() -> None:
    scores = pd.DataFrame(
        {
            "variant_id": ["m1", "p0"],
            "fixed_torsion_delta_deg": [-1.0, 0.0],
            "C_peak_A": [5.75, 5.75],
            "D_peak_A": [7.2, 7.3],
        }
    )
    assert classify_two_point_movement(scores, "C_peak_A") == "no change"
    assert classify_two_point_movement(scores, "D_peak_A").startswith("changed by")


def test_baseline_relative_intensity_calculation() -> None:
    scores = pd.DataFrame(
        {
            "variant_id": ["m1", "p0"],
            "fixed_torsion_delta_deg": [-1.0, 0.0],
            "C_peak_intensity_or_score": [90.0, 100.0],
            "D_peak_intensity_or_score": [110.0, 100.0],
        }
    )
    out = add_baseline_relative_intensity(scores)
    assert out["C_relative_to_baseline"].tolist() == [0.9, 1.0]
    assert out["D_relative_to_baseline"].tolist() == [1.1, 1.0]


def test_report_text_includes_baseline_parent_caution_and_unsafe_exclusion() -> None:
    scores = pd.DataFrame(
        {
            "variant_id": ["m1", "p0"],
            "fixed_torsion_delta_deg": [-1.0, 0.0],
            "C_peak_A": [5.7, 5.7],
            "D_peak_A": [7.3, 7.3],
            "C_error_A": [0.1, 0.1],
            "D_error_A": [0.0, 0.0],
            "combined_abs_error_A": [0.1, 0.1],
            "C_relative_to_baseline": [0.99, 1.0],
            "D_relative_to_baseline": [1.0, 1.0],
        }
    )
    text = build_report_text(scores, safe_count=2, skipped_count=3)
    assert "baseline_parent" in text
    assert "Unsafe variants were excluded" in text
    assert "not a broad or unconstrained omega scan" in text
