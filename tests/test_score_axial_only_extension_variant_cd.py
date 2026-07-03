from __future__ import annotations

import pandas as pd

from scripts.score_axial_only_extension_variant_cd import (
    add_relative_scores,
    baseline_control_row,
    best_variant,
    build_report_text,
    filter_geometry_interpretable,
)


def test_filters_only_geometry_interpretable_rows() -> None:
    df = pd.DataFrame(
        {
            "variant_id": ["safe", "unsafe", "also_safe"],
            "geometry_interpretable": ["True", "False", True],
        }
    )
    assert filter_geometry_interpretable(df)["variant_id"].tolist() == ["safe", "also_safe"]


def test_identifies_baseline_control_axial_1p0000() -> None:
    scores = pd.DataFrame(
        {
            "variant_id": ["compressed", "baseline"],
            "axial_scale_z": [0.99, 1.0],
        }
    )
    assert baseline_control_row(scores)["variant_id"] == "baseline"


def test_relative_scores_use_baseline() -> None:
    scores = pd.DataFrame(
        {
            "variant_id": ["compressed", "baseline"],
            "axial_scale_z": [0.97, 1.0],
            "C_score": [80.0, 100.0],
            "D_score": [120.0, 100.0],
        }
    )
    out = add_relative_scores(scores)
    low = out[out["variant_id"] == "compressed"].iloc[0]
    assert low["relative_C_score_vs_baseline"] == 0.8
    assert low["relative_D_score_vs_baseline"] == 1.2


def test_best_combined_error_selection() -> None:
    scores = pd.DataFrame({"variant_id": ["worse", "best"], "combined_abs_error_A": [0.2, 0.1]})
    assert best_variant(scores)["variant_id"] == "best"


def test_report_text_contains_required_cautions() -> None:
    scores = pd.DataFrame(
        {
            "variant_id": ["compressed", "baseline"],
            "radial_scale_xy": [1.0, 1.0],
            "axial_scale_z": [0.97, 1.0],
            "geometry_interpretable": [True, True],
            "C_peak_A": [5.6, 5.8],
            "D_peak_A": [7.3, 7.3],
            "C_error_A": [0.0, 0.2],
            "D_error_A": [0.0, 0.0],
            "combined_abs_error_A": [0.0, 0.2],
            "C_score": [100.0, 100.0],
            "D_score": [100.0, 100.0],
            "relative_C_score_vs_baseline": [1.0, 1.0],
            "relative_D_score_vs_baseline": [1.0, 1.0],
        }
    )
    text = build_report_text(scores, scored_count=2, skipped_count=0)
    assert "axial-only extension" in text
    assert "controlled diagnostic perturbations" in text
    assert "not minimized structures" in text
