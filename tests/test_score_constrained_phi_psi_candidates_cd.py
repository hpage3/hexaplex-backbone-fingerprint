from __future__ import annotations

import pandas as pd

from scripts.score_constrained_phi_psi_candidates_cd import (
    best_candidate,
    build_report_text,
    combined_abs_error,
    filter_safe_candidates,
    parse_bool,
    skipped_unsafe_count,
    sort_by_fixed_delta,
)


def test_parse_bool_accepts_common_csv_values() -> None:
    assert parse_bool(True)
    assert parse_bool("True")
    assert parse_bool("1")
    assert parse_bool("yes")
    assert not parse_bool(False)
    assert not parse_bool("False")
    assert not parse_bool("")


def test_filter_safe_candidates_and_skipped_count() -> None:
    df = pd.DataFrame(
        {
            "candidate_id": ["a", "b", "c"],
            "safe_for_diffraction_scoring": ["True", "False", "0"],
        }
    )
    safe = filter_safe_candidates(df)
    assert safe["candidate_id"].tolist() == ["a"]
    assert skipped_unsafe_count(df) == 2


def test_combined_abs_error() -> None:
    assert combined_abs_error(-0.1, 0.2) == 0.30000000000000004


def test_best_candidate_selects_lowest_combined_error() -> None:
    scores = pd.DataFrame(
        {
            "candidate_id": ["worse", "best"],
            "combined_abs_error_A": [0.3, 0.1],
        }
    )
    assert best_candidate(scores)["candidate_id"] == "best"


def test_sort_by_fixed_delta_orders_numeric_values() -> None:
    df = pd.DataFrame(
        {
            "candidate_id": ["p2", "m1", "p0"],
            "fixed_torsion_delta_deg": ["2", "-1", "0"],
        }
    )
    assert sort_by_fixed_delta(df)["candidate_id"].tolist() == ["m1", "p0", "p2"]


def test_report_text_includes_fixed_omega_pilot_caution() -> None:
    scores = pd.DataFrame(
        {
            "candidate_id": ["cand_001", "cand_003"],
            "repeat_type": ["CYP->GLU", "CYP->GLU"],
            "solve_mode": ["one_torsion", "one_torsion"],
            "fixed_torsion_delta_deg": [0.0, 1.0],
            "C_peak_A": [5.5, 5.55],
            "D_peak_A": [7.2, 7.25],
            "C_error_A": [-0.1, -0.05],
            "D_error_A": [-0.1, -0.05],
            "combined_abs_error_A": [0.2, 0.1],
        }
    )
    text = build_report_text(scores, safe_count=2, skipped_count=1)
    assert "tiny fixed-omega pilot" in text
    assert "Unsafe candidates skipped: 1" in text
