from __future__ import annotations

import pandas as pd

from scripts.score_coupled_cyp_glu_glu_mep_variant_cd import (
    add_baseline_relative_scores,
    baseline_control_row,
    best_combined_error_row,
    build_report_text,
    classify_peak_shift,
    filter_geometry_safe_rows,
    skipped_unsafe_count,
)


def test_filter_geometry_safe_rows_and_skipped_count() -> None:
    df = pd.DataFrame(
        {
            "variant_id": ["safe", "unsafe", "also_safe"],
            "geometry_safe": ["True", "False", True],
        }
    )
    safe = filter_geometry_safe_rows(df)
    assert safe["variant_id"].tolist() == ["safe", "also_safe"]
    assert skipped_unsafe_count(df) == 1


def test_baseline_control_pair_is_zero_zero() -> None:
    scores = pd.DataFrame(
        {
            "variant_id": ["m1_m1", "p0_p0"],
            "cyp_glu_delta_deg": [-1.0, 0.0],
            "glu_mep_delta_deg": [-1.0, 0.0],
        }
    )
    assert baseline_control_row(scores)["variant_id"] == "p0_p0"


def test_relative_score_calculation_uses_baseline_control() -> None:
    scores = pd.DataFrame(
        {
            "variant_id": ["m1_m1", "p0_p0"],
            "cyp_glu_delta_deg": [-1.0, 0.0],
            "glu_mep_delta_deg": [-1.0, 0.0],
            "C_score": [90.0, 100.0],
            "D_score": [110.0, 100.0],
        }
    )
    out = add_baseline_relative_scores(scores)
    assert out["relative_C_score_vs_baseline"].tolist() == [0.9, 1.0]
    assert out["relative_D_score_vs_baseline"].tolist() == [1.1, 1.0]


def test_flat_shifted_peak_classifier() -> None:
    assert classify_peak_shift(pd.Series([5.7454, 5.7454, 5.7454])) == "flat"
    assert classify_peak_shift(pd.Series([5.70, 5.75, 5.80])).startswith("shifted")


def test_best_combined_error_selection() -> None:
    scores = pd.DataFrame(
        {
            "variant_id": ["worse", "best"],
            "combined_abs_error_A": [0.25, 0.10],
        }
    )
    assert best_combined_error_row(scores)["variant_id"] == "best"


def test_report_text_includes_coupled_pilot_and_geometry_safe_gate() -> None:
    scores = pd.DataFrame(
        {
            "variant_id": ["m1_m1", "p0_p0"],
            "cyp_glu_delta_deg": [-1.0, 0.0],
            "glu_mep_delta_deg": [-1.0, 0.0],
            "C_peak_A": [5.7, 5.7],
            "D_peak_A": [7.3, 7.3],
            "C_error_A": [0.1, 0.1],
            "D_error_A": [0.0, 0.0],
            "combined_abs_error_A": [0.1, 0.1],
            "C_score": [90.0, 100.0],
            "D_score": [105.0, 100.0],
            "relative_C_score_vs_baseline": [0.9, 1.0],
            "relative_D_score_vs_baseline": [1.05, 1.0],
        }
    )
    text = build_report_text(scores, scored_count=2, skipped_count=0)
    assert "coupled perturbation pilot" in text
    assert "geometry-safe before scoring" in text
    assert "Baseline/control variant ID" in text
