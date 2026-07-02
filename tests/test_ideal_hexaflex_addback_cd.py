import numpy as np
import pandas as pd
import pytest

from scripts.analyze_ideal_hexaflex_addback_cd import (
    build_addback_summary,
    subtract_profiles,
    variant_metrics,
    window_max,
)


def profile(ds, intensities):
    q = 2.0 * np.pi / np.asarray(ds, dtype=float)
    return pd.DataFrame({"q_Ainv": q, "d_A": ds, "intensity": intensities}).sort_values("d_A").reset_index(drop=True)


def test_subtract_profiles_requires_shared_grid():
    a = profile([5.5, 7.2], [10.0, 20.0])
    b = profile([5.5, 7.2], [3.0, 5.0])
    diff = subtract_profiles(a, b)
    assert diff["intensity"].tolist() == [7.0, 15.0]
    with pytest.raises(ValueError):
        subtract_profiles(a, profile([5.6, 7.2], [1.0, 1.0]))


def test_window_max_returns_peak_inside_window():
    prof = profile([5.0, 5.55, 5.75, 7.2], [1.0, 3.0, 2.0, 9.0])
    intensity, d_value = window_max(prof, 5.4, 5.8)
    assert intensity == 3.0
    assert d_value == 5.55


def test_variant_metrics_calculates_shift_vs_backbone():
    baseline = profile([5.54, 7.19, 8.0], [10.0, 20.0, 1.0])
    variant = profile([5.74, 7.28, 8.0], [12.0, 25.0, 1.0])
    row = variant_metrics(
        "backbone_plus_carboxylate",
        variant,
        baseline,
        5.6,
        7.3,
        (5.4, 5.8),
        (7.0, 7.5),
        0.2,
    )
    assert row["C_shift_vs_backbone_only_A"] == pytest.approx(0.2)
    assert row["D_shift_vs_backbone_only_A"] == pytest.approx(0.09)
    assert row["D_moves_toward_7p3_vs_backbone_only"] is True


def test_build_addback_summary_with_fixture_profiles():
    ds = [5.54, 5.74, 7.19, 7.28, 7.54, 7.62]
    profiles = {
        "backbone_only": profile(ds, [10.0, 4.0, 20.0, 5.0, 1.0, 1.0]),
        "backbone_plus_carboxylate": profile(ds, [4.0, 12.0, 5.0, 25.0, 1.0, 1.0]),
        "full": profile(ds, [4.0, 5.0, 1.0, 5.0, 30.0, 1.0]),
        "no_h": profile(ds, [4.0, 5.0, 1.0, 5.0, 1.0, 29.0]),
    }
    summary = build_addback_summary(profiles, 5.6, 7.3, (5.4, 5.8), (7.0, 7.5), 0.2)
    assert {"variant", "difference"}.issubset(set(summary["comparison_type"]))
    variant = summary[summary["comparison"] == "backbone_plus_carboxylate"].iloc[0]
    assert variant["D_moves_toward_7p3_vs_backbone_only"] is True
