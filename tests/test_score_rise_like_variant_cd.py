from __future__ import annotations

import pandas as pd

from scripts.score_rise_like_variant_cd import (
    add_relative_scores,
    baseline_row,
    best_variant,
    build_report_text,
    filter_geometry_interpretable,
    sort_by_rise,
)


def test_filters_only_geometry_interpretable_rows() -> None:
    df = pd.DataFrame({"variant_id": ["safe", "unsafe"], "geometry_interpretable": ["True", "False"]})
    assert filter_geometry_interpretable(df)["variant_id"].tolist() == ["safe"]


def test_sort_by_rise_and_baseline_identification() -> None:
    scores = pd.DataFrame({"variant_id": ["base", "low"], "axial_rise_scale": [1.0, 0.96]})
    assert sort_by_rise(scores)["variant_id"].tolist() == ["low", "base"]
    assert baseline_row(scores)["variant_id"] == "base"


def test_relative_scores_use_baseline_control() -> None:
    scores = pd.DataFrame(
        {
            "variant_id": ["low", "base"],
            "axial_rise_scale": [0.96, 1.0],
            "C_score": [80.0, 100.0],
            "D_score": [120.0, 100.0],
        }
    )
    out = add_relative_scores(scores)
    low = out[out["variant_id"] == "low"].iloc[0]
    assert low["relative_C_score_vs_baseline"] == 0.8
    assert low["relative_D_score_vs_baseline"] == 1.2


def test_best_combined_error_selection() -> None:
    scores = pd.DataFrame({"variant_id": ["worse", "best"], "combined_abs_error_A": [0.2, 0.1]})
    assert best_variant(scores)["variant_id"] == "best"


def test_report_text_contains_required_cautions() -> None:
    scores = pd.DataFrame(
        {
            "variant_id": ["rise_like_1p0000"],
            "axial_rise_scale": [1.0],
            "estimated_percent_compression": [0.0],
            "C_peak_A": [5.74],
            "D_peak_A": [7.28],
            "C_error_A": [0.14],
            "D_error_A": [-0.02],
            "combined_abs_error_A": [0.16],
            "C_score": [100.0],
            "D_score": [100.0],
            "relative_C_score_vs_baseline": [1.0],
            "relative_D_score_vs_baseline": [1.0],
        }
    )
    text = build_report_text(scores, scored_count=1, skipped_count=0)
    assert "rise-like proxy" in text
    assert "controlled diagnostic perturbations" in text
    assert "not be treated as an optimized molecular model" in text
