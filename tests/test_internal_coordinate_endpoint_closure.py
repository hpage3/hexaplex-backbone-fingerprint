import math

import numpy as np
import pandas as pd

from scripts.run_internal_coordinate_endpoint_closure import (
    PeptideSegment,
    build_report,
    class_for_chain,
    closure_class,
    omega_window_class,
    point_from_internal,
    reconstruct_downstream_ca,
    scan_segment,
    trans_deviation_deg,
)


def synthetic_segment(parent_omega: float = -175.0) -> PeptideSegment:
    n_i = np.array([0.0, 0.0, 0.0])
    ca_i = np.array([1.45, 0.0, 0.0])
    c_i = np.array([2.05, 1.20, 0.15])
    parent_psi = -43.0
    c_n_length = 1.33
    n_ca_length = 1.46
    ca_c_n_angle = 116.0
    c_n_ca_angle = 121.0
    n_j = point_from_internal(n_i, ca_i, c_i, c_n_length, ca_c_n_angle, parent_psi)
    ca_j = point_from_internal(ca_i, c_i, n_j, n_ca_length, c_n_ca_angle, parent_omega)
    return PeptideSegment(
        chain="A",
        class_label="triketo_cyanuric_like",
        segment_index=1,
        residue_pair="CYP1->GLU2",
        res_i_index=1,
        res_j_index=2,
        n_i=n_i,
        ca_i=ca_i,
        c_i=c_i,
        n_j=n_j,
        ca_j=ca_j,
        parent_phi_deg=math.nan,
        parent_psi_deg=parent_psi,
        parent_omega_deg=parent_omega,
        parent_theta_deg=math.nan,
        c_n_length_A=c_n_length,
        n_ca_length_A=n_ca_length,
        ca_c_n_angle_deg=ca_c_n_angle,
        c_n_ca_angle_deg=c_n_ca_angle,
    )


def test_trans_deviation_wraparound_near_plus_minus_180() -> None:
    assert trans_deviation_deg(180.0) == 0.0
    assert trans_deviation_deg(-180.0) == 0.0
    assert trans_deviation_deg(172.0) == 8.0
    assert trans_deviation_deg(-171.0) == 9.0


def test_omega_window_classification() -> None:
    assert omega_window_class(-172.0) == "within_8deg"
    assert omega_window_class(-170.0) == "within_10deg"
    assert omega_window_class(-168.0) == "outside_10deg"
    assert omega_window_class(math.nan) == "insufficient_data"


def test_closure_classification() -> None:
    assert closure_class(0.05) == "good_closure"
    assert closure_class(0.20) == "borderline_closure"
    assert closure_class(0.30) == "poor_closure"
    assert closure_class(math.nan) == "insufficient_data"


def test_synthetic_internal_coordinate_fixture_reproduces_known_endpoint() -> None:
    segment = synthetic_segment(parent_omega=-175.0)
    modeled = reconstruct_downstream_ca(segment, -175.0)
    assert np.linalg.norm(modeled - segment.ca_j) < 1e-10
    rows = scan_segment(segment, omega_scan=[-180.0, -175.0, -170.0])
    best = min(rows, key=lambda row: row["closure_residual_A"])
    assert best["scanned_omega_deg"] == -175.0
    assert best["closure_class"] == "good_closure"


def test_class_assignment_for_two_threefold_families() -> None:
    assert class_for_chain("A") == "triketo_cyanuric_like"
    assert class_for_chain("C") == "triketo_cyanuric_like"
    assert class_for_chain("E") == "triketo_cyanuric_like"
    assert class_for_chain("B") == "triamino_melamine_like"
    assert class_for_chain("D") == "triamino_melamine_like"
    assert class_for_chain("F") == "triamino_melamine_like"
    assert class_for_chain("Z") == "unclassified"


def test_report_wording_contains_required_scope_terms() -> None:
    scan = pd.DataFrame(
        [
            {
                "segment_id": "A:1:CYP1->GLU2",
                "class_label": "triketo_cyanuric_like",
                "closure_class": "good_closure",
            }
        ]
    )
    summary = pd.DataFrame(
        [
            {
                "group": "all_segments",
                "analyzable_segment_count": 1,
                "parent_omega_median_deg": -175.0,
                "parent_omega_trans_deviation_median_deg": 5.0,
                "parent_within_8deg_count": 1,
                "parent_within_10deg_count": 1,
                "parent_outside_10deg_count": 0,
                "parent_every_other_detected": False,
                "segments_with_good_closure_within_8deg": 1,
                "segments_with_good_closure_within_10deg": 1,
                "best_scanned_omega_deg": -175.0,
                "best_closure_residual_A": 0.0,
            },
            {
                "group": "triamino_melamine_like",
                "analyzable_segment_count": 0,
                "parent_omega_median_deg": math.nan,
                "parent_omega_trans_deviation_median_deg": math.nan,
                "parent_within_8deg_count": 0,
                "parent_within_10deg_count": 0,
                "parent_outside_10deg_count": 0,
                "parent_every_other_detected": False,
                "segments_with_good_closure_within_8deg": 0,
                "segments_with_good_closure_within_10deg": 0,
                "best_scanned_omega_deg": math.nan,
                "best_closure_residual_A": math.nan,
            },
            {
                "group": "triketo_cyanuric_like",
                "analyzable_segment_count": 1,
                "parent_omega_median_deg": -175.0,
                "parent_omega_trans_deviation_median_deg": 5.0,
                "parent_within_8deg_count": 1,
                "parent_within_10deg_count": 1,
                "parent_outside_10deg_count": 0,
                "parent_every_other_detected": False,
                "segments_with_good_closure_within_8deg": 1,
                "segments_with_good_closure_within_10deg": 1,
                "best_scanned_omega_deg": -175.0,
                "best_closure_residual_A": 0.0,
            },
        ]
    )
    text = build_report(scan, summary)
    assert "minimal endpoint-closure prototype" in text
    assert "not a final structure" in text
    assert "omega" in text
    assert "pNAB" in text
    assert "every-other" in text
    assert "+/- 8" in text
    assert "+/- 10" in text
