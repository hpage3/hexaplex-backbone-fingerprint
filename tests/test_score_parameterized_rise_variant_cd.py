from __future__ import annotations

import pandas as pd

from scripts.score_parameterized_rise_variant_cd import (
    add_relative_scores,
    baseline_row,
    best_variant,
    build_report_text,
    filter_geometry_interpretable,
)


def test_filters_only_geometry_interpretable_rows() -> None:
    df = pd.DataFrame({"variant_id": ["safe", "unsafe"], "geometry_interpretable": ["True", "False"]})
    assert filter_geometry_interpretable(df)["variant_id"].tolist() == ["safe"]


def test_identifies_parameterized_rise_baseline() -> None:
    scores = pd.DataFrame({"variant_id": ["base", "low"], "rise_scale": [1.0, 0.97]})
    assert baseline_row(scores)["variant_id"] == "base"


def test_relative_scores_use_baseline() -> None:
    scores = pd.DataFrame(
        {
            "variant_id": ["low", "base"],
            "rise_scale": [0.97, 1.0],
            "C_score": [110.0, 100.0],
            "D_score": [90.0, 100.0],
        }
    )
    out = add_relative_scores(scores)
    low = out[out["variant_id"] == "low"].iloc[0]
    assert low["relative_C_score_vs_baseline"] == 1.1
    assert low["relative_D_score_vs_baseline"] == 0.9


def test_best_combined_error_row_selection() -> None:
    scores = pd.DataFrame({"variant_id": ["worse", "best"], "combined_abs_error_A": [0.2, 0.1]})
    assert best_variant(scores)["variant_id"] == "best"


def test_report_text_contains_required_phrases() -> None:
    scores = pd.DataFrame(
        {
            "variant_id": ["parameterized_rise_1p0000"],
            "rise_scale": [1.0],
            "estimated_percent_rise_compression": [0.0],
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
    assert "layer/repeat-aware rise compression" in text
    assert "not minimized physical structures" in text
    assert "Comparison To Earlier rise_like Diagnostic" in text
