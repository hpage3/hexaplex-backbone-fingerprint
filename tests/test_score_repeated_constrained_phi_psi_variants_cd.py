from __future__ import annotations

import pandas as pd

from scripts.score_constrained_phi_psi_candidates_cd import combined_abs_error, monotonic_trend
from scripts.score_repeated_constrained_phi_psi_variants_cd import (
    best_variant,
    build_report_text,
    filter_safe_repeated_variants,
    skipped_unsafe_count,
    sort_by_fixed_delta,
)


def test_filter_safe_repeated_variants_and_skipped_count() -> None:
    df = pd.DataFrame(
        {
            "variant_id": ["m1", "p0", "p2"],
            "safe_for_diffraction_scoring": ["True", True, "False"],
        }
    )
    safe = filter_safe_repeated_variants(df)
    assert safe["variant_id"].tolist() == ["m1", "p0"]
    assert skipped_unsafe_count(df) == 1


def test_combined_abs_error_shared_helper() -> None:
    assert combined_abs_error(0.1, -0.25) == 0.35


def test_best_variant_selection() -> None:
    scores = pd.DataFrame(
        {
            "variant_id": ["worse", "best"],
            "combined_abs_error_A": [0.4, 0.2],
        }
    )
    assert best_variant(scores)["variant_id"] == "best"


def test_monotonic_flat_nonmonotonic_trends() -> None:
    assert monotonic_trend(pd.Series([1.0, 1.0, 1.0])) == "flat"
    assert monotonic_trend(pd.Series([1.0, 2.0, 3.0])) == "monotonic increasing"
    assert monotonic_trend(pd.Series([1.0, 3.0, 2.0])) == "nonmonotonic"


def test_sort_by_fixed_delta() -> None:
    df = pd.DataFrame(
        {
            "variant_id": ["p1", "m1", "p0"],
            "fixed_torsion_delta_deg": [1.0, -1.0, 0.0],
        }
    )
    assert sort_by_fixed_delta(df)["variant_id"].tolist() == ["m1", "p0", "p1"]


def test_report_text_includes_fixed_omega_and_unsafe_exclusion() -> None:
    scores = pd.DataFrame(
        {
            "variant_id": ["m1", "p0", "p1"],
            "fixed_torsion_delta_deg": [-1.0, 0.0, 1.0],
            "C_peak_A": [5.7, 5.6, 5.8],
            "D_peak_A": [7.2, 7.3, 7.4],
            "C_error_A": [0.1, 0.0, 0.2],
            "D_error_A": [-0.1, 0.0, 0.1],
            "combined_abs_error_A": [0.2, 0.0, 0.3],
        }
    )
    text = build_report_text(scores, safe_count=3, skipped_count=3)
    assert "tiny fixed-omega pilot" in text
    assert "Unsafe repeated variants were not scored" in text
    assert "Repeated variants skipped as unsafe: 3" in text
