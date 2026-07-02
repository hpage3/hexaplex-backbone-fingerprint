from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.diagnose_coupled_cyp_glu_glu_mep_cd_profiles import (
    build_report_text,
    classify_tiny_shifts,
    integrated_intensity,
    intensity_weighted_centroid,
    parabolic_peak_estimate,
    profile_difference_metrics,
)
from scripts.score_coupled_cyp_glu_glu_mep_variant_cd import baseline_control_row


def test_intensity_weighted_centroid_simple_profile() -> None:
    d_values = np.array([5.4, 5.6, 5.8])
    intensities = np.array([1.0, 2.0, 1.0])
    assert np.isclose(intensity_weighted_centroid(d_values, intensities), 5.6)


def test_parabolic_peak_estimate_known_vertex() -> None:
    d_values = np.array([5.5, 5.6, 5.7])
    intensities = -100.0 * (d_values - 5.62) ** 2 + 10.0
    assert abs(parabolic_peak_estimate(d_values, intensities) - 5.62) < 1e-10


def test_parabolic_peak_estimate_returns_none_for_edge_or_degenerate() -> None:
    assert parabolic_peak_estimate(np.array([5.5, 5.6, 5.7]), np.array([3.0, 2.0, 1.0])) is None
    assert parabolic_peak_estimate(np.array([5.5, 5.6, 5.7]), np.array([2.0, 2.0, 2.0])) is None


def test_profile_difference_metrics_l2_max_and_correlation() -> None:
    d_values = np.array([5.4, 5.6, 5.8])
    baseline = np.array([1.0, 2.0, 3.0])
    variant = np.array([2.0, 3.0, 4.0])
    metrics = profile_difference_metrics(d_values, variant, d_values, baseline)
    assert np.isclose(metrics["l2"], np.sqrt(3.0))
    assert metrics["max_abs"] == 1.0
    assert np.isclose(metrics["corr"], 1.0)


def test_integrated_intensity_relative_fixture() -> None:
    d_values = np.array([5.4, 5.6, 5.8])
    baseline = integrated_intensity(d_values, np.array([1.0, 1.0, 1.0]))
    variant = integrated_intensity(d_values, np.array([2.0, 2.0, 2.0]))
    assert baseline == 0.39999999999999947
    assert np.isclose(variant / baseline, 2.0)


def test_baseline_control_identification_zero_zero() -> None:
    scores = pd.DataFrame(
        {
            "variant_id": ["cyp_glu_m1__glu_mep_m1", "cyp_glu_p0__glu_mep_p0"],
            "cyp_glu_delta_deg": [-1.0, 0.0],
            "glu_mep_delta_deg": [-1.0, 0.0],
        }
    )
    assert baseline_control_row(scores)["variant_id"] == "cyp_glu_p0__glu_mep_p0"


def test_tiny_shift_classifier_uses_documented_tolerance() -> None:
    assert classify_tiny_shifts(pd.Series([0.0, 5e-5]), tolerance_A=1e-4).startswith("flat/tiny")
    assert classify_tiny_shifts(pd.Series([0.0, 2e-4]), tolerance_A=1e-4).startswith("nonzero")


def test_report_text_contains_required_cautions() -> None:
    diagnostics = pd.DataFrame(
        {
            "variant_id": ["cyp_glu_p0__glu_mep_p0", "cyp_glu_p0__glu_mep_p0"],
            "cyp_glu_delta_deg": [0.0, 0.0],
            "glu_mep_delta_deg": [0.0, 0.0],
            "band": ["C", "D"],
            "picked_peak_shift_vs_baseline_A": [0.0, 0.0],
            "centroid_shift_vs_baseline_A": [0.0, 0.0],
            "parabolic_shift_vs_baseline_A": [0.0, 0.0],
            "local_profile_l2_diff_vs_baseline": [0.0, 0.0],
            "local_profile_max_abs_diff_vs_baseline": [0.0, 0.0],
            "local_profile_corr_vs_baseline": [1.0, 1.0],
            "integrated_intensity_relative_vs_baseline": [1.0, 1.0],
        }
    )
    scores = pd.DataFrame({"variant_id": ["cyp_glu_p0__glu_mep_p0"]})
    baseline = pd.Series({"variant_id": "cyp_glu_p0__glu_mep_p0"})
    text = build_report_text(diagnostics, scores, baseline)
    assert "binning/peak-picking" in text
    assert "coupled safe basin" in text
    assert "not a new structural proof" in text
