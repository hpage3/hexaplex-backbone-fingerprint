from __future__ import annotations

import pandas as pd
import pytest

from scripts.analyze_repeated_phi_psi_intensity_sensitivity import (
    build_report_text,
    classify_trend,
    normalize_to_baseline,
    require_intensity_columns,
)


def score_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "variant_id": ["m1", "p0", "p1"],
            "fixed_torsion_delta_deg": [-1.0, 0.0, 1.0],
            "C_peak_A": [5.745, 5.745, 5.745],
            "D_peak_A": [7.276, 7.276, 7.276],
            "C_peak_intensity_or_score": [90.0, 100.0, 110.0],
            "D_peak_intensity_or_score": [102.0, 100.0, 98.0],
        }
    )


def test_baseline_normalization_and_percent_change() -> None:
    result = normalize_to_baseline(score_rows())
    assert result["C_relative_to_baseline"].tolist() == [0.9, 1.0, 1.1]
    assert result["D_percent_change"].tolist() == [2.0000000000000018, 0.0, -2.0000000000000018]


def test_classify_trend_with_tolerance() -> None:
    assert classify_trend(pd.Series([0.999, 1.0, 1.001]), flat_tolerance_fraction=0.005) == "flat"
    assert classify_trend(pd.Series([0.98, 1.0, 1.02]), flat_tolerance_fraction=0.005) == "monotonic increasing"
    assert classify_trend(pd.Series([1.02, 1.0, 1.01]), flat_tolerance_fraction=0.005) == "asymmetric/non-monotonic"


def test_missing_intensity_column_handling() -> None:
    with pytest.raises(ValueError, match="Missing required intensity columns"):
        require_intensity_columns(pd.DataFrame({"C_peak_intensity_or_score": [1.0]}))


def test_report_text_cautions_intensity_not_peak_position() -> None:
    result = normalize_to_baseline(score_rows())
    text = build_report_text(result)
    assert "Intensity sensitivity is not the same as peak-position movement" in text
    assert "tiny fixed-omega pilot" in text
