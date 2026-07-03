import math

import pandas as pd

from scripts.run_phi_psi_omega_closure_scan import (
    best_rows_by_segment,
    build_report,
    class_for_chain,
    closure_class,
    detect_every_other_pattern,
    omega_window_class,
    parent_centered_scan_values,
    trans_deviation_deg,
)


def test_omega_trans_deviation_wraparound() -> None:
    assert trans_deviation_deg(180.0) == 0.0
    assert trans_deviation_deg(-180.0) == 0.0
    assert trans_deviation_deg(172.0) == 8.0
    assert trans_deviation_deg(-170.0) == 10.0


def test_omega_window_classification_for_8_and_10_degree_windows() -> None:
    assert omega_window_class(8.0, value_is_deviation=True) == "within_8deg"
    assert omega_window_class(9.0, value_is_deviation=True) == "within_10deg"
    assert omega_window_class(11.0, value_is_deviation=True) == "outside_10deg"


def test_parent_centered_phi_psi_scan_generation_wraps_angles() -> None:
    assert parent_centered_scan_values(179.0, [-10.0, 0.0, 10.0]) == [169.0, 179.0, -171.0]


def test_closure_classification_good_borderline_poor() -> None:
    assert closure_class(0.01) == "good_closure"
    assert closure_class(0.2) == "borderline_closure"
    assert closure_class(0.3) == "poor_closure"


def test_best_row_selection_under_any_10_and_8_windows() -> None:
    scan = pd.DataFrame(
        [
            {
                "segment_id": "A:1",
                "chain": "A",
                "class_label": "triketo_cyanuric_like",
                "scan_status": "scored",
                "omega_window_class": "outside_10deg",
                "omega_trans_deviation_deg": 12.0,
                "closure_residual_A": 0.01,
                "closure_class": "good_closure",
                "scanned_phi_deg": 1.0,
                "scanned_psi_deg": 2.0,
                "scanned_omega_deg": -168.0,
            },
            {
                "segment_id": "A:1",
                "chain": "A",
                "class_label": "triketo_cyanuric_like",
                "scan_status": "scored",
                "omega_window_class": "within_10deg",
                "omega_trans_deviation_deg": 10.0,
                "closure_residual_A": 0.05,
                "closure_class": "good_closure",
                "scanned_phi_deg": 1.0,
                "scanned_psi_deg": 2.0,
                "scanned_omega_deg": -170.0,
            },
            {
                "segment_id": "A:1",
                "chain": "A",
                "class_label": "triketo_cyanuric_like",
                "scan_status": "scored",
                "omega_window_class": "within_8deg",
                "omega_trans_deviation_deg": 4.0,
                "closure_residual_A": 0.08,
                "closure_class": "good_closure",
                "scanned_phi_deg": 1.0,
                "scanned_psi_deg": 2.0,
                "scanned_omega_deg": -176.0,
            },
        ]
    )
    best = best_rows_by_segment(scan)
    by_window = {row["best_window"]: row for row in best.to_dict("records")}
    assert by_window["best_any_omega"]["scanned_omega_deg"] == -168.0
    assert by_window["best_within_10deg_omega"]["scanned_omega_deg"] == -170.0
    assert by_window["best_within_8deg_omega"]["scanned_omega_deg"] == -176.0


def test_class_assignment_for_chain_families() -> None:
    assert class_for_chain("A") == "triketo_cyanuric_like"
    assert class_for_chain("C") == "triketo_cyanuric_like"
    assert class_for_chain("E") == "triketo_cyanuric_like"
    assert class_for_chain("B") == "triamino_melamine_like"
    assert class_for_chain("D") == "triamino_melamine_like"
    assert class_for_chain("F") == "triamino_melamine_like"


def test_every_other_pattern_detection_on_synthetic_best_omega_values() -> None:
    deviations = [trans_deviation_deg(value) for value in [-180.0, -168.0, -180.0, -168.0, -180.0, -168.0]]
    pattern = detect_every_other_pattern(deviations)
    assert pattern["every_other_detected"] is True


def test_report_wording_contains_required_scope_terms() -> None:
    scan = pd.DataFrame()
    best = pd.DataFrame()
    summary = pd.DataFrame(
        [
            {
                "group": "all_segments",
                "segment_count": 2,
                "fully_phi_psi_omega_scannable_count": 1,
                "missing_phi_or_psi_context_count": 1,
                "good_within_8deg_count": 1,
                "good_within_10deg_count": 1,
                "borderline_or_better_within_8deg_count": 1,
                "borderline_or_better_within_10deg_count": 1,
                "median_best_residual_within_8deg_A": 0.01,
                "median_best_residual_within_10deg_A": 0.01,
                "parent_omega_every_other_detected": False,
                "best_feasible_omega_every_other_detected": False,
            },
            {
                "group": "triamino_melamine_like",
                "segment_count": 1,
                "fully_phi_psi_omega_scannable_count": 1,
                "missing_phi_or_psi_context_count": 0,
                "good_within_8deg_count": 1,
                "good_within_10deg_count": 1,
                "borderline_or_better_within_8deg_count": 1,
                "borderline_or_better_within_10deg_count": 1,
                "median_best_residual_within_8deg_A": 0.01,
                "median_best_residual_within_10deg_A": 0.01,
                "parent_omega_every_other_detected": False,
                "best_feasible_omega_every_other_detected": False,
            },
            {
                "group": "triketo_cyanuric_like",
                "segment_count": 1,
                "fully_phi_psi_omega_scannable_count": 1,
                "missing_phi_or_psi_context_count": 0,
                "good_within_8deg_count": 1,
                "good_within_10deg_count": 1,
                "borderline_or_better_within_8deg_count": 1,
                "borderline_or_better_within_10deg_count": 1,
                "median_best_residual_within_8deg_A": 0.01,
                "median_best_residual_within_10deg_A": 0.01,
                "parent_omega_every_other_detected": False,
                "best_feasible_omega_every_other_detected": False,
            },
        ]
    )
    text = build_report(scan, best, summary)
    assert "phi/psi/omega internal-coordinate closure prototype" in text
    assert "not a final structure" in text
    assert "not energy minimized" in text
    assert "omega" in text
    assert "pNAB" in text
    assert "every-other" in text
    assert "+/- 8" in text
    assert "+/- 10" in text
