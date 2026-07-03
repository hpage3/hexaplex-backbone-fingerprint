from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.diagnose_coupled_cyp_glu_glu_mep_cd_profiles import (
    intensity_weighted_centroid,
    parabolic_peak_estimate,
    profile_difference_metrics,
)
from scripts.diagnose_parameterized_rise_cd_profiles import build_report_text
from scripts.score_parameterized_rise_variant_cd import baseline_row


def test_centroid_helper() -> None:
    assert np.isclose(intensity_weighted_centroid(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 1.0])), 2.0)


def test_parabolic_helper() -> None:
    d_values = np.array([5.5, 5.6, 5.7])
    intensities = -100.0 * (d_values - 5.62) ** 2 + 10.0
    assert abs(parabolic_peak_estimate(d_values, intensities) - 5.62) < 1e-10


def test_profile_difference_helper() -> None:
    d_values = np.array([1.0, 2.0, 3.0])
    metrics = profile_difference_metrics(d_values, np.array([2.0, 3.0, 4.0]), d_values, np.array([1.0, 2.0, 3.0]))
    assert np.isclose(metrics["l2"], np.sqrt(3.0))


def test_baseline_identification() -> None:
    scores = pd.DataFrame({"variant_id": ["parameterized_rise_0p9700", "parameterized_rise_1p0000"], "rise_scale": [0.97, 1.0]})
    assert baseline_row(scores)["variant_id"] == "parameterized_rise_1p0000"


def test_report_text_contains_required_phrases() -> None:
    diagnostics = pd.DataFrame(
        {
            "variant_id": ["parameterized_rise_1p0000", "parameterized_rise_1p0000"],
            "rise_scale": [1.0, 1.0],
            "band": ["C", "D"],
            "picked_peak_A": [5.74, 7.28],
            "picked_peak_shift_vs_baseline_A": [0.0, 0.0],
            "local_centroid_A": [5.6, 7.3],
            "centroid_shift_vs_baseline_A": [0.0, 0.0],
            "local_parabolic_peak_A": [5.6, 7.3],
            "parabolic_shift_vs_baseline_A": [0.0, 0.0],
            "integrated_intensity_relative_vs_baseline": [1.0, 1.0],
        }
    )
    scores = pd.DataFrame({"variant_id": ["parameterized_rise_1p0000"]})
    baseline = pd.Series({"variant_id": "parameterized_rise_1p0000"})
    text = build_report_text(diagnostics, scores, baseline)
    assert "parameterized rise" in text
    assert "C moves smoothly" in text
    assert "D remains stable" in text
