from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.diagnose_coupled_cyp_glu_glu_mep_cd_profiles import (
    integrated_intensity,
    intensity_weighted_centroid,
    parabolic_peak_estimate,
    profile_difference_metrics,
)
from scripts.diagnose_fine_axial_profile_cd_profiles import build_report_text
from scripts.score_fine_axial_profile_variant_cd import reference_row


def test_intensity_weighted_centroid_simple_profile() -> None:
    assert np.isclose(intensity_weighted_centroid(np.array([5.4, 5.6, 5.8]), np.array([1.0, 2.0, 1.0])), 5.6)


def test_parabolic_peak_estimate_known_and_degenerate() -> None:
    d_values = np.array([5.5, 5.6, 5.7])
    intensities = -100.0 * (d_values - 5.62) ** 2 + 10.0
    assert abs(parabolic_peak_estimate(d_values, intensities) - 5.62) < 1e-10
    assert parabolic_peak_estimate(d_values, np.array([1.0, 1.0, 1.0])) is None
    assert parabolic_peak_estimate(d_values, np.array([3.0, 2.0, 1.0])) is None


def test_profile_difference_metrics_l2_max_corr() -> None:
    d_values = np.array([5.4, 5.6, 5.8])
    metrics = profile_difference_metrics(d_values, np.array([2.0, 3.0, 4.0]), d_values, np.array([1.0, 2.0, 3.0]))
    assert np.isclose(metrics["l2"], np.sqrt(3.0))
    assert metrics["max_abs"] == 1.0
    assert np.isclose(metrics["corr"], 1.0)


def test_relative_integrated_intensity() -> None:
    d_values = np.array([5.4, 5.6, 5.8])
    baseline = integrated_intensity(d_values, np.array([1.0, 1.0, 1.0]))
    variant = integrated_intensity(d_values, np.array([2.0, 2.0, 2.0]))
    assert np.isclose(variant / baseline, 2.0)


def test_reference_variant_identification_0p9700() -> None:
    scores = pd.DataFrame({"variant_id": ["fine_axial_0p9700", "fine_axial_0p9750"], "axial_scale_z": [0.97, 0.975]})
    assert reference_row(scores)["variant_id"] == "fine_axial_0p9700"


def test_report_text_contains_required_cautions() -> None:
    diagnostics = pd.DataFrame(
        {
            "variant_id": ["fine_axial_0p9700", "fine_axial_0p9700"],
            "axial_scale_z": [0.97, 0.97],
            "band": ["C", "D"],
            "picked_peak_A": [5.64, 7.28],
            "picked_peak_shift_vs_0p9700_A": [0.0, 0.0],
            "local_centroid_A": [5.6, 7.3],
            "centroid_shift_vs_0p9700_A": [0.0, 0.0],
            "local_parabolic_peak_A": [5.6, 7.3],
            "parabolic_shift_vs_0p9700_A": [0.0, 0.0],
            "integrated_intensity_relative_vs_0p9700": [1.0, 1.0],
            "local_profile_l2_diff_vs_0p9700": [0.0, 0.0],
            "local_profile_max_abs_diff_vs_0p9700": [0.0, 0.0],
            "local_profile_corr_vs_0p9700": [1.0, 1.0],
        }
    )
    scores = pd.DataFrame({"variant_id": ["fine_axial_0p9700"]})
    reference = pd.Series({"variant_id": "fine_axial_0p9700"})
    text = build_report_text(diagnostics, scores, reference)
    assert "underneath discretized picked peaks" in text
    assert "not minimized structures" in text
