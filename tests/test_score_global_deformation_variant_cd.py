from __future__ import annotations

import pandas as pd

from scripts.score_global_deformation_variant_cd import (
    add_relative_scores_by_mode,
    best_variant,
    build_report_text,
    classify_directional_trend,
    filter_geometry_interpretable,
    mode_baseline_row,
    worst_variant,
)


def test_filter_geometry_interpretable_rows() -> None:
    df = pd.DataFrame(
        {
            "variant_id": ["safe", "unsafe", "also_safe"],
            "geometry_interpretable": ["True", "False", True],
        }
    )
    assert filter_geometry_interpretable(df)["variant_id"].tolist() == ["safe", "also_safe"]


def test_identifies_correct_mode_baselines() -> None:
    scores = pd.DataFrame(
        {
            "variant_id": ["radial_0", "axial_0", "twist_0", "anis_xy_0"],
            "deformation_mode": ["radial_scale_xy", "axial_scale_z", "twist_about_z", "anisotropic_xy"],
        }
    )
    assert mode_baseline_row(scores, "radial_scale_xy")["variant_id"] == "radial_0"
    assert mode_baseline_row(scores, "axial_scale_z")["variant_id"] == "axial_0"
    assert mode_baseline_row(scores, "twist_about_z")["variant_id"] == "twist_0"
    assert mode_baseline_row(scores, "anisotropic_xy")["variant_id"] == "anis_xy_0"


def test_relative_scores_use_mode_specific_baseline() -> None:
    scores = pd.DataFrame(
        {
            "variant_id": ["radial_m1", "radial_0", "axial_m1", "axial_0"],
            "deformation_mode": ["radial_scale_xy", "radial_scale_xy", "axial_scale_z", "axial_scale_z"],
            "radial_scale_xy": [0.99, 1.0, 1.0, 1.0],
            "axial_scale_z": [1.0, 1.0, 0.995, 1.0],
            "C_score": [90.0, 100.0, 40.0, 50.0],
            "D_score": [120.0, 100.0, 60.0, 50.0],
        }
    )
    out = add_relative_scores_by_mode(scores)
    radial = out[out["variant_id"] == "radial_m1"].iloc[0]
    axial = out[out["variant_id"] == "axial_m1"].iloc[0]
    assert radial["relative_C_score_vs_mode_baseline"] == 0.9
    assert radial["relative_D_score_vs_mode_baseline"] == 1.2
    assert axial["relative_C_score_vs_mode_baseline"] == 0.8
    assert axial["relative_D_score_vs_mode_baseline"] == 1.2


def test_best_and_worst_combined_error_selection() -> None:
    scores = pd.DataFrame(
        {
            "variant_id": ["mid", "best", "worst"],
            "combined_abs_error_A": [0.2, 0.1, 0.3],
        }
    )
    assert best_variant(scores)["variant_id"] == "best"
    assert worst_variant(scores)["variant_id"] == "worst"


def test_directional_trend_classification() -> None:
    assert classify_directional_trend(pd.Series([1.0, 1.0, 1.0])) == "flat"
    assert classify_directional_trend(pd.Series([1.0, 2.0, 3.0])) == "monotonic increasing"
    assert classify_directional_trend(pd.Series([3.0, 2.0, 1.0])) == "monotonic decreasing"
    assert classify_directional_trend(pd.Series([1.0, 3.0, 2.0])) == "nonmonotonic"


def test_report_text_contains_required_cautions() -> None:
    scores = pd.DataFrame(
        {
            "variant_id": ["radial_m1", "radial_0", "radial_p1"],
            "deformation_mode": ["radial_scale_xy"] * 3,
            "radial_scale_xy": [0.99, 1.0, 1.01],
            "axial_scale_z": [1.0, 1.0, 1.0],
            "twist_total_deg": [0.0, 0.0, 0.0],
            "x_scale": [1.0, 1.0, 1.0],
            "y_scale": [1.0, 1.0, 1.0],
            "C_peak_A": [5.7, 5.7, 5.7],
            "D_peak_A": [7.3, 7.3, 7.3],
            "C_error_A": [0.1, 0.1, 0.1],
            "D_error_A": [0.0, 0.0, 0.0],
            "combined_abs_error_A": [0.1, 0.1, 0.1],
            "relative_C_score_vs_mode_baseline": [0.9, 1.0, 1.1],
            "relative_D_score_vs_mode_baseline": [1.1, 1.0, 0.9],
        }
    )
    for mode, baseline in [
        ("axial_scale_z", "axial_0"),
        ("twist_about_z", "twist_0"),
        ("anisotropic_xy", "anis_xy_0"),
    ]:
        scores.loc[len(scores)] = {
            "variant_id": baseline,
            "deformation_mode": mode,
            "radial_scale_xy": 1.0,
            "axial_scale_z": 1.0,
            "twist_total_deg": 0.0,
            "x_scale": 1.0,
            "y_scale": 1.0,
            "C_peak_A": 5.7,
            "D_peak_A": 7.3,
            "C_error_A": 0.1,
            "D_error_A": 0.0,
            "combined_abs_error_A": 0.1,
            "relative_C_score_vs_mode_baseline": 1.0,
            "relative_D_score_vs_mode_baseline": 1.0,
        }
    text = build_report_text(scores, scored_count=len(scores), skipped_count=0)
    assert "controlled diagnostic perturbations" in text
    assert "not minimized physical structures" in text
    assert "local torsion basin" in text
    assert "geometry-interpretable variants" in text
