from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.audit_backbone_torsion_repeat import Residue
from scripts.prototype_glu_mep_fixed_omega_closure_refinement import (
    OMEGA_POLICY,
    classify_geometry,
    deterministic_sort_attempts,
    identify_glu_mep_windows,
    write_report,
)


def residue(chain: str, resseq: int, resname: str) -> Residue:
    return Residue(
        chain=chain,
        resseq=resseq,
        resname=resname,
        atoms={"CA": np.array([float(resseq), 0.0, 0.0])},
        atom_names_in_order=("CA",),
    )


def test_identify_glu_mep_windows_by_coordinate_order() -> None:
    residues = {
        "B": [residue("B", 99, "MEP"), residue("B", 10, "GLU"), residue("B", 8, "MEP")],
        "D": [residue("D", 2, "GLU"), residue("D", 1, "MEP")],
    }
    windows = identify_glu_mep_windows(residues)
    assert [(w["chain_id"], w["repeat_start_index"], w["res_i"], w["res_j"]) for w in windows] == [
        ("B", 1, 10, 8),
        ("D", 0, 2, 1),
    ]


def test_classify_geometry_and_failure_reason() -> None:
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
    audit["max_backbone_bond_delta_A"] = 0.2
    safe, reason = classify_geometry(audit)
    assert not safe
    assert reason == "backbone_bond_delta_exceeds_threshold"


def test_deterministic_sort_by_delta_and_solve_mode() -> None:
    df = pd.DataFrame(
        {
            "attempt_id": ["c", "a", "b"],
            "fixed_torsion_delta_deg": [0.0, -1.0, -1.0],
            "solve_mode": ["local_refine", "two_torsion", "one_torsion"],
        }
    )
    sorted_df = deterministic_sort_attempts(df)
    assert sorted_df["attempt_id"].tolist() == ["b", "a", "c"]


def test_fixed_omega_policy_metadata() -> None:
    assert OMEGA_POLICY == "fixed_180"


def test_report_text_includes_fixed_omega_and_decision_branch(tmp_path) -> None:
    results = pd.DataFrame(
        {
            "attempt_id": ["a"],
            "solve_mode": ["one_torsion"],
            "fixed_torsion_delta_deg": [1.0],
            "chain_id": ["B"],
            "repeat_start_index": [1],
            "endpoint_error_A": [0.1],
            "closure_success": [False],
            "geometry_safe": [False],
            "max_backbone_bond_delta_A": [0.2],
            "max_backbone_angle_delta_deg": [1.0],
            "max_omega_trans_deviation_deg": [5.0],
            "failure_reason": ["endpoint_closure_failed"],
        }
    )
    report = tmp_path / "report.md"
    write_report(results, [{"chain_id": "B"}], ["B"], report)
    text = report.read_text(encoding="utf-8")
    assert "fixed_180" in text
    assert "omega sensitivity" in text.lower()
    assert "fixed-omega repeated GLU->MEP variant generation" in text or "defer omega sensitivity" in text
