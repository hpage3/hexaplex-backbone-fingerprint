from __future__ import annotations

import pandas as pd

from scripts.compare_glu_mep_omega_modes_closure import (
    OMEGA_MODES,
    baseline_omega_values,
    mode_comparison_summary,
    omega_values_for_mode,
    write_report,
)
from scripts.prototype_glu_mep_fixed_omega_closure_refinement import classify_geometry


class Window:
    chain_id = "B"
    repeat_start_index = 1
    baseline_torsions = {"omega0_deg": -167.5, "omega1_deg": 179.0}


def test_omega_mode_metadata_and_values() -> None:
    assert OMEGA_MODES == ("fixed_180", "baseline_parent")
    assert omega_values_for_mode(Window(), "fixed_180") == (180.0, 180.0)
    assert omega_values_for_mode(Window(), "baseline_parent") == (-167.5, 179.0)


def test_baseline_omega_extraction() -> None:
    assert baseline_omega_values(Window()) == {"omega0_deg": -167.5, "omega1_deg": 179.0}


def test_geometry_safe_classification_and_failure_reason() -> None:
    audit = {
        "candidate_file_exists": True,
        "atom_count_match": True,
        "labels_preserved": True,
        "max_ca_shift_A": 0.0,
        "max_backbone_bond_delta_A": 0.01,
        "max_backbone_angle_delta_deg": 1.0,
        "max_omega_trans_deviation_deg": 5.0,
    }
    assert classify_geometry(audit) == (True, "")
    audit["max_omega_trans_deviation_deg"] = 25.0
    safe, reason = classify_geometry(audit)
    assert not safe
    assert reason == "omega_trans_deviation_exceeds_threshold"


def test_mode_comparison_summary_counts_nonzero_safe() -> None:
    df = pd.DataFrame(
        {
            "omega_mode": ["fixed_180", "baseline_parent", "baseline_parent"],
            "attempt_id": ["a", "b", "c"],
            "closure_success": [False, True, True],
            "geometry_safe": [False, True, True],
            "fixed_torsion_delta_deg": [0.0, 0.0, 1.0],
            "endpoint_error_A": [0.2, 0.0, 0.01],
        }
    )
    summary = mode_comparison_summary(df)
    baseline = summary[summary["omega_mode"] == "baseline_parent"].iloc[0]
    assert baseline["geometry_safe"] == 2
    assert baseline["nonzero_geometry_safe"] == 1


def test_report_text_includes_mode_conclusion(tmp_path) -> None:
    df = pd.DataFrame(
        {
            "omega_mode": ["fixed_180", "baseline_parent"],
            "attempt_id": ["a", "b"],
            "solve_mode": ["one_torsion", "one_torsion"],
            "fixed_torsion_delta_deg": [1.0, 1.0],
            "endpoint_error_A": [0.2, 0.01],
            "closure_success": [False, True],
            "geometry_safe": [False, True],
            "max_backbone_bond_delta_A": [0.2, 0.01],
            "max_backbone_angle_delta_deg": [10.0, 1.0],
            "max_omega_trans_deviation_deg": [20.0, 5.0],
            "failure_reason": ["endpoint_closure_failed", ""],
        }
    )
    report = tmp_path / "report.md"
    write_report(df, [{"chain_id": "B"}], ["B"], Window(), report)
    text = report.read_text(encoding="utf-8")
    assert "fixed_180" in text
    assert "baseline_parent" in text
    assert "Did baseline_parent rescue any nonzero GLU->MEP perturbations? yes" in text
