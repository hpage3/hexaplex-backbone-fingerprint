import math

import pandas as pd

from scripts.spec_two_class_peptide_backbone_builder import (
    build_report,
    detect_every_other_pattern,
    omega_window_class,
    scan_rotation_summary_rows,
    trans_deviation_deg,
)


def test_trans_deviation_handles_plus_minus_180_wraparound() -> None:
    assert trans_deviation_deg(180.0) == 0.0
    assert trans_deviation_deg(-180.0) == 0.0
    assert trans_deviation_deg(172.0) == 8.0
    assert trans_deviation_deg(-171.0) == 9.0


def test_omega_window_classification() -> None:
    assert omega_window_class(7.9) == "within_8deg"
    assert omega_window_class(8.0) == "within_8deg"
    assert omega_window_class(9.5) == "within_10deg"
    assert omega_window_class(10.0) == "within_10deg"
    assert omega_window_class(10.1) == "outside_10deg"
    assert omega_window_class(math.nan) == "insufficient_data"


def test_detect_every_other_omega_pattern() -> None:
    pattern = detect_every_other_pattern([1.0, 14.0, 2.0, 15.0, 1.5, 13.5], threshold_deg=10.0)
    assert pattern["every_other_detected"] is True
    assert pattern["alternating_fraction"] == 1.0


def test_detect_every_other_rejects_flat_pattern() -> None:
    pattern = detect_every_other_pattern([2.0, 3.0, 2.5, 3.5, 2.2, 3.1], threshold_deg=10.0)
    assert pattern["every_other_detected"] is False


def test_scan_rotation_summary_says_existing_scans_only_monitored_omega() -> None:
    rows = scan_rotation_summary_rows()
    prior_rows = [row for row in rows if row["row_type"] == "scan_scope"]
    assert prior_rows
    assert all(row["rotated_omega"] is False for row in prior_rows)
    assert all("monitored omega" in row["notes"] for row in prior_rows)


def test_report_wording_contains_required_scope_and_builder_terms() -> None:
    summary = pd.DataFrame(
        [
            {
                "row_type": "omega_summary",
                "model_id": "parent_reference",
                "group": "all_six_chains",
                "omega_count": 4,
                "omega_median_deg": -176.0,
                "trans_deviation_median_deg": 4.0,
                "within_8deg_fraction": 1.0,
                "within_10deg_fraction": 1.0,
                "outside_10deg_fraction": 0.0,
                "every_other_detected": False,
            },
            {
                "row_type": "file_presence",
                "model_id": "pnab_or_builder_input",
                "group": "input_inventory",
                "present": False,
                "path": "",
                "notes": "no pNAB input found",
            },
            {
                "row_type": "scan_scope",
                "model_id": "two_class_axial_theta_scan",
                "group": "prior_scan",
                "rotated_omega": False,
                "notes": "monitored omega/theta; omega was not independently scanned",
            },
            {
                "row_type": "builder_spec",
                "model_id": "external_two_class_peptide_backbone_builder",
                "group": "recommended_next_step",
                "rotated_omega": True,
                "notes": "scan omega explicitly",
            },
        ]
    )
    text = build_report(summary)
    assert "external two-class peptide-backbone builder" in text
    assert "not a final structure" in text
    assert "omega" in text
    assert "pNAB" in text
    assert "every-other" in text
    assert "+/- 8" in text
    assert "+/- 10" in text
