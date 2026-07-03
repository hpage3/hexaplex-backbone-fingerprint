from __future__ import annotations

import math

import pandas as pd

from scripts.summarize_global_rise_diagnostic_phase import (
    best_score_row,
    build_report_text,
    construct_summary_rows,
    overall_best_row,
    safe_float,
    summarize_geometry_counts,
    summary_row_from_inputs,
    trend_strings_for_phase,
)


def test_safe_float_parsing_from_csv_rows() -> None:
    assert safe_float("5.6422") == 5.6422
    assert math.isnan(safe_float(""))
    assert math.isnan(safe_float("not-a-number"))
    assert safe_float(None, default=None) is None


def test_best_row_selection_by_combined_error() -> None:
    scores = pd.DataFrame(
        {
            "variant_id": ["worse", "best"],
            "combined_abs_error_A": [0.2, 0.1],
        }
    )
    assert best_score_row(scores)["variant_id"] == "best"


def test_summarize_geometry_counts() -> None:
    geometry = pd.DataFrame({"geometry_interpretable": ["True", "False", True]})
    generated, interpretable = summarize_geometry_counts(geometry)
    assert generated == 3
    assert interpretable == 2


def test_classify_trend_strings_from_branch_labels() -> None:
    c_trend, d_trend, interpretation = trend_strings_for_phase("global_deformation")
    assert "axial-sensitive" in c_trend
    assert "radial/inter-strand-distance sensitive" in d_trend
    assert "separated" in interpretation


def test_construct_csv_summary_row_from_scores_and_geometry() -> None:
    scores = pd.DataFrame(
        {
            "variant_id": ["a", "b"],
            "C_peak_A": [5.7, 5.64],
            "D_peak_A": [7.28, 7.28],
            "C_error_A": [0.1, 0.04],
            "D_error_A": [-0.02, -0.02],
            "combined_abs_error_A": [0.12, 0.06],
        }
    )
    geometry = pd.DataFrame({"geometry_interpretable": [True, True]})
    row = summary_row_from_inputs("axial_only_extension", "branch", scores, geometry)
    assert row["variants_generated"] == 2
    assert row["geometry_interpretable"] == 2
    assert row["variants_scored"] == 2
    assert row["best_variant"] == "b"
    assert row["best_combined_abs_error_A"] == 0.06


def test_construct_summary_rows_contains_expected_phases() -> None:
    rows = construct_summary_rows()
    phases = [row["phase"] for row in rows]
    assert "constrained_backbone_context" in phases
    assert "global_deformation" in phases
    assert "rise_like_diagnostic" in phases
    assert phases[-1] == "overall_best"


def test_overall_best_prefers_current_rise_like_branch_on_tie() -> None:
    rows = [
        {
            "phase": "axial_only_extension",
            "branch": "axial",
            "best_variant": "axial_only_0p9700",
            "best_combined_abs_error_A": 0.0667,
        },
        {
            "phase": "rise_like_diagnostic",
            "branch": "rise",
            "best_variant": "rise_like_0p9700",
            "best_combined_abs_error_A": 0.0667,
        },
    ]
    assert overall_best_row(rows)["best_variant"] == "rise_like_0p9700"


def test_report_text_contains_required_phrases() -> None:
    summary = pd.DataFrame(
        [
            {
                "phase": "overall_best",
                "branch": "overall best diagnostic",
                "variants_generated": 9,
                "geometry_interpretable": 9,
                "variants_scored": 9,
                "best_variant": "rise_like_0p9700",
                "best_C_peak_A": 5.6422,
                "best_D_peak_A": 7.2756,
                "best_C_error_A": 0.0422,
                "best_D_error_A": -0.0244,
                "best_combined_abs_error_A": 0.0667,
                "primary_C_trend": "",
                "primary_D_trend": "",
                "interpretation": "",
            }
        ]
    )
    text = build_report_text(summary, missing=[])
    assert "Global/Rise Diagnostic Phase Summary" in text
    assert "C is mainly axial/rise-like sensitive" in text
    assert "D is mainly radial/inter-strand-distance sensitive" in text
    assert "not minimized physical structures" in text
    assert "do not claim the final structure requires literal uniform 3% z-scaling" in text
