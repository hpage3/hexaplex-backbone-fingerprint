from __future__ import annotations

import pandas as pd

from scripts.score_fine_axial_profile_variant_cd import (
    add_relative_scores,
    best_variant,
    build_report_text,
    filter_geometry_interpretable,
    reference_row,
)


def test_filters_only_geometry_interpretable_rows() -> None:
    df = pd.DataFrame({"variant_id": ["safe", "unsafe"], "geometry_interpretable": ["True", "False"]})
    assert filter_geometry_interpretable(df)["variant_id"].tolist() == ["safe"]


def test_identifies_local_reference_axial_0p9700() -> None:
    scores = pd.DataFrame({"variant_id": ["ref", "other"], "axial_scale_z": [0.97, 0.975]})
    assert reference_row(scores)["variant_id"] == "ref"


def test_relative_scores_use_reference() -> None:
    scores = pd.DataFrame(
        {
            "variant_id": ["ref", "other"],
            "axial_scale_z": [0.97, 0.975],
            "C_score": [100.0, 90.0],
            "D_score": [100.0, 110.0],
        }
    )
    out = add_relative_scores(scores)
    other = out[out["variant_id"] == "other"].iloc[0]
    assert other["relative_C_score_vs_0p9700"] == 0.9
    assert other["relative_D_score_vs_0p9700"] == 1.1


def test_best_combined_error_selection() -> None:
    scores = pd.DataFrame({"variant_id": ["worse", "best"], "combined_abs_error_A": [0.2, 0.1]})
    assert best_variant(scores)["variant_id"] == "best"


def test_report_text_contains_required_phrases() -> None:
    scores = pd.DataFrame(
        {
            "variant_id": ["fine_axial_0p9700"],
            "axial_scale_z": [0.97],
            "C_peak_A": [5.64],
            "D_peak_A": [7.28],
            "C_error_A": [0.04],
            "D_error_A": [-0.02],
            "combined_abs_error_A": [0.06],
            "C_score": [100.0],
            "D_score": [100.0],
            "relative_C_score_vs_0p9700": [1.0],
            "relative_D_score_vs_0p9700": [1.0],
        }
    )
    text = build_report_text(scores, scored_count=1, skipped_count=0)
    assert "fine axial profile diagnostic" in text
    assert "controlled diagnostic perturbations" in text
    assert "discretized C peak response" in text
